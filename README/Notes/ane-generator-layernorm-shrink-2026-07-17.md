# ANE generator shrink — layer_norm lowering and the rate-boundary split

Collected: 2026-07-17. Task T6 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on
the Status log's DEVICE GATE RUN entry (the A14 rejected the T4 ANE generator
with `Too many fvmlibs (>255)` — a program-SIZE cap, not an op-type rejection)
and T4's export
([ane-generator-coreml-export-2026-07-17.md](ane-generator-coreml-export-2026-07-17.md)).

## Verdict

The manual-AdaIN unroll was the biggest single contributor to the program size
the A14 rejected. Lowering it to `layer_norm` cuts the generator **35%**
(1,039 → 679 non-const ops) with the waveform bit-identical to T4's graph
(max-abs 1.71e-6). That alone lands above the ~350-op target, so — per the plan's
secondary trigger — the graph is ALSO split at the 2,400-frame rate boundary into
two packages of **337 / 346** non-const ops. Two candidate ladders for the owner:
the single ln package first, the split halves if the phone still rejects it.

**No A14 admittance is claimed.** The census is a Mac-side proxy for the fvmlib
object count; we cannot read the true fvmlib threshold off this Mac. The phone is
the judge (plan Phase 1 gate). Everything below is measured on this Mac.

## What was built

Both live in `export_synth/convert_ane.py`, driven by
`export_synth/wrappers.py`'s `lower_generator_adain_to_layernorm` (the lowering)
and `GeneratorTrunkANE` / `GeneratorBodyANE` (the split). `kokoro/istftnet.py` is
untouched — the lowering is an export-time module swap, matched by class NAME
(the file-loader gives `AdaIN1d` a class identity distinct from a normal import).

| package | CLI | verify script |
| --- | --- | --- |
| `kokoro_decoder_har_ane_ln_3s` | `--mode decoder-har-ane-ln` | `scripts/verify_decoder_har_ane_ln.py` |
| `kokoro_decoder_har_ane_ln_trunk_3s` + `..._body_3s` | `--mode decoder-har-ane-split` | `scripts/verify_decoder_har_ane_split.py` |

`AdaIN1dLayerNorm` replaces `AdaIN1d`'s manual instance norm —
`(x - x.mean(dim=2)) / sqrt(x.var(dim=2, unbiased=False) + eps)` plus two
`.expand` tiles — with a single `F.layer_norm(x, (T,), eps)` over the time axis
and pure broadcasting of the style-projected `(1 + gamma)` / `beta`. Same trained
`fc`, same math up to fp ulps; coremltools maps the call to exactly one MIL
`layer_norm` op (verified — it does not decompose back to primitives).

## The MIL op census (the metric T6 is judged on)

Full histograms via `scripts/count_mil_ops.py`; the acceptance assertions live in
the verify scripts.

| op | baseline (T4 manual) | ln-lowered | trunk | body |
| --- | ---: | ---: | ---: | ---: |
| **NON-CONST** | **1,039** | **679** | **337** | **346** |
| total | 2,196 | 1,572 | — | — |
| const | 1,157 | 893 | — | — |
| `layer_norm` | 0 | 44 | 22 | 22 |
| `reduce_mean` | 88 | **0** | — | — |
| `tile` | 96 | **0** | — | — |
| `sub` | 44 | 0 | — | — |
| `square` | 44 | 0 | — | — |
| `sqrt` | 44 | 0 | — | — |
| `real_div` | 44 | 0 | — | — |

The lowering is **clean**: every manual-norm primitive (`reduce_mean`, `square`,
`sqrt`, `real_div`, `tile`) goes to **zero**, and `layer_norm` appears exactly
44 times. A nonzero `reduce_mean` in the lowered program would have meant
coremltools decomposed `layer_norm` back into primitives — the plan's stop
condition. It did not.

**Why 44 and not 48.** The generator holds 48 `AdaIN1d` applications, but
coremltools keeps only 44 normalization chains — in BOTH the manual and the
lowered graph (48 style projections trace, 44 normalization chains survive its
graph passes). This is input-independent: a random-input trace elides the same 4
(88 `reduce_mean` / 44 `sqrt` / 48 `linear` either way). So the lowering replaces
exactly the 44 chains that exist, 1:1 — the census compares like for like, and
the 4 elided chains are identical on both sides. The waveform settles it anyway.

**The split halves the program.** The largest single split program is **346
non-const ops** — below the ~350 target and roughly half the monolithic ln. Both
halves take the same `har` (14,401); the seam is the trunk tensor
`(1, 256, 2,400)`, the natural boundary between the 240→2,400 and 2,400→14,401
rate regions.

## Numerical parity (all on this Mac)

Real inputs throughout (T4's construction path: real misaki phonemes, real
BERT/duration/F0, real `af_heart`), one `har` tensor shared across every side.
The reference is the PyTorch fp32 pre-trim graph (`_forward_pretrim`), never the
T1 baseline package — the same two traps T4 was built around.

**Allclose vs the manual graph (PyTorch fp32):** the lowered wrapper's waveform
is max-abs **1.71e-6** (SNR **109.59 dB**) from the manual `GeneratorFromHarANE`;
the split chain (trunk→body) is identical at 1.71e-6. The gate is on the
waveform, not `spec`/`phase`: `spec = exp(x)` amplifies layer_norm's ulp-level
reduction-order difference (spec max-abs 2.19e-5, phase 1.42e-5), so a raw `atol`
on those intermediates is the wrong unit. The waveform — the wrapper's product,
and the quantity the 40 dB Core ML gate judges — is a strict pass at atol 1e-5.

**Core ML fp16 spec/phase → Python iSTFT → SNR vs pre-trim, by compute unit:**

| compute unit | ln (monolithic) | split (chained) |
| --- | ---: | ---: |
| `CPU_AND_GPU` | **46.14 dB** ✅ | **46.02 dB** ✅ |
| `CPU_ONLY` | 24.47 dB | 24.47 dB |
| `CPU_AND_NE` | **non-finite** ❌ | **non-finite** ❌ |

The gate reads `CPU_AND_GPU` for the same reason T4's did: this Mac's ANE ADMITS
these graphs and then miscomputes them into non-finite output (not a reroute —
`CPU_AND_GPU` clears the gate, so the graph is correct). `CPU_AND_NE` on this Mac
measures the Mac's Neural Engine, not the graph. The A14's ANE is a different
generation; whether it also miscomputes is the owner's open question (T4's "For
the owner" decision table still applies).

## Compute plan — per-op device, CPU_AND_NE, this Mac

| package | ANE-mapped |
| --- | ---: |
| `kokoro_decoder_har_ane_ln_3s` | 659/679 (97.1%) |
| `kokoro_decoder_har_ane_ln_trunk_3s` | 323/337 (95.8%) |
| `kokoro_decoder_har_ane_ln_body_3s` | 340/346 (98.3%) |

Same residency as T4's 98.1% — the lowering did not push work off the ANE. The
~20 CPU ops per package are boundary shaping (input `cast`s, the spec/phase
`slice`/`pad`), not a hot-loop fallback.

## Bench-app plumbing

- **`--generator-package <name>`** (`ios-bench/Sources/BenchApp.swift`,
  `BenchRunner.generatorPackageOverride`): overrides which bundled package the
  `aneGenerator` policy loads for the generator stage (default unchanged,
  `kokoro_decoder_har_ane_3s`). Threaded into `BundleModelCache` via a new
  `generatorPackage:` init parameter. Lets the owner walk candidate packages in
  one device session.
- **Staged into the bundle:** `kokoro_decoder_har_ane_ln_3s`,
  `..._ln_trunk_3s`, `..._ln_body_3s` copied into
  `ios-bench/Resources/coreml/` (an optional-copy hunk added to
  `prepare_resources.sh` alongside the T5 hunk), then `xcodegen generate`. The ln
  package is a drop-in for `--policy aneGenerator`; the split halves are NOT wired
  into the Swift executor (that is a follow-on), so they are reachable only as
  single-half `--mode computeplan` candidates.
- Acceptance met: `xcodebuild ... CODE_SIGNING_ALLOWED=NO build` **succeeded**,
  `swift test` **48/48** (T3's goldens untouched).

## Exact commands for the owner's device ladder

Set these as scheme arguments (Xcode → Scheme → Run → Arguments) or write them to
`Documents/launch_args.txt` for an untethered run, same conventions as the
existing `--policy` / `--mode` paths.

**Rung 1 — the single ln package end to end (does it load, predict, and stay
finite on the A14?):**

```
--arms coreml --keys 3s --policy aneGenerator --generator-package kokoro_decoder_har_ane_ln_3s --out aneGenerator_ln_3s.json
```

Watch the console for `ANEGEN: geometry ...` and `ANEGEN: non-finite output
check finite-fraction=...` (1.0000 = clean; less means the Mac-ANE corruption
reproduces on the A14 — T4's decision table). Compute-plan residency:

```
--mode computeplan --model kokoro_decoder_har_ane_ln_3s --out computeplan_ln_3s.json
```

**Rung 2 — the split halves, only if rung 1's package still hits the fvmlib cap.**
The halves are not wired into the pipeline, so probe each half's compile/residency
individually (the fvmlib rejection surfaces here as a load/compile failure):

```
--mode computeplan --model kokoro_decoder_har_ane_ln_trunk_3s --out computeplan_ln_trunk_3s.json
--mode computeplan --model kokoro_decoder_har_ane_ln_body_3s --out computeplan_ln_body_3s.json
```

If both halves compile where the monolithic ln package did not, wiring the split
through the Swift executor (trunk → body → host iSTFT) becomes the follow-on task.

## Stop conditions checked (none tripped)

- layer_norm mapped to a single MIL op — no decomposition (`reduce_mean` == 0).
- parity 46.14 / 46.02 dB, both ≥ 40 dB.
- census 1,039 → 679 (ln) → 346 (largest split half), all meaningfully below
  baseline.

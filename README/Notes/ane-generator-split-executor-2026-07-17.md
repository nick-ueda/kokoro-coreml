# ANE generator split — Swift executor wiring (trunk → body)

Collected: 2026-07-17. Task T7 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on
T6's rate-boundary split
([ane-generator-layernorm-shrink-2026-07-17.md](ane-generator-layernorm-shrink-2026-07-17.md))
and the Status log's SPLIT DEVICE GATE RUN entry (the A14 rejects the monolithic
ln package with `ANECCompile() FAILED (11)` — the fvmlib cap — but both split
halves compile clean).

## Device outcome (owner, iPhone 12 Pro / A14, 2026-07-17) — the split WORKS

`--arms coreml --keys 3s --policy aneGeneratorSplit`, 7 iterations, every pass:

```
ANEGEN: split trunk finite-fraction=1.0000 body finite-fraction=1.0000 trunkNonFinite=0 bodyNonFinite=0 total=931222
```

**The A14 ANE admits AND correctly executes the chained split.** The Mac's
admits-then-miscompute corruption does NOT reproduce, and — unlike the monolithic
ln package — there is no `ANECCompile() FAILED` fallback: the generator runs at
**~0.49→0.55 s**, ~40% faster than the monolithic's CPU-fallback 0.74–0.95 s.
That plus the clean compile (T6's ~90%-ANE-mapped halves) is strong evidence the
split is genuinely on the Neural Engine. Still to measure — the spike's actual
goal — is the **power/heat win**: the untethered battery soak
(`SPIKE_RUNBOOK.md` Test 3b) under `--policy aneGeneratorSplit`, plus optionally
`powermetrics --samplers ane` for a direct residency receipt. Recorded in the
plan's Status log (SPLIT EXECUTION GATE RUN). The "success criterion" section
below is the criterion this run met.

## Verdict

T6's two split packages now run end to end through the Swift generator stage:
`kokoro_decoder_har_ane_ln_trunk_3s` (x_pre/ref_s/har → the 2,400-frame seam) →
`kokoro_decoder_har_ane_ln_body_3s` (seam/ref_s/har → spec/phase) → the existing
host iSTFT. Mac-side, the chained split reproduces the monolithic ln generator's
waveform at **48.73 dB SNR** under `CPU_AND_GPU` (gate 40 dB) — the wiring is
correct. The owner now has a one-session device path (`--policy
aneGeneratorSplit`) that reports finiteness of BOTH halves, so the open question —
does the A14 ANE *execute* the admitted split correctly, or miscompute it like
this Mac? — is finally observable on the phone.

**No A14 execution is claimed.** Both split halves *compile* on the A14 (T6 gate);
whether they *predict* finite is exactly what the device deliverable below lets
the owner measure. `CPU_AND_NE` on this Mac is still non-finite (T4's Mac-ANE
miscompute, unchanged — the GPU proves the graph).

## What was built

The split lives ENTIRELY behind the generator-stage boundary. Decoder-pre, F0,
the host iSTFT, the word-timing surface, and the `SynthesisResult` /
`StageTimings` shape are untouched.

### Executor (`swift/Sources/KokoroPipeline/`)

- **`GeneratorSplitModels`** — a `{ trunk, body }` `MLModel` pair carried through
  the generator-stage boundary.
- **`KokoroModelProvider.generatorSplitModels(bucketSec:)`** — new optional
  protocol method, default `nil`. Non-nil ⇒ the executor runs the split; `nil` ⇒
  the single-package path, unchanged. The runtime `KokoroPipeline` inherits the
  default, so production is unaffected.
- **`predictGeneratorSplit(trunk:body:xPre:refS:har:)`** — the ONE place the seam
  contract lives: the trunk emits a tensor named `trunk` `(1, 256, 2,400)`, the
  body consumes it under the same name alongside the same `ref_s`/`har`. Both the
  executor and the parity test call it, so the wiring is asserted in exactly one
  spot.
- **`executeKokoroSynthesis` Stage 8/9** — branches on the split pair. Input
  shaping (x_pre/har padding) keys on the trunk (identical inputs to the
  monolithic package); the ANE spec/phase detection keys on the body. `warmModels`
  warms both halves.
- **`finiteCensus`** — shared non-finite census, so the single-package and split
  finiteness gates read off one helper.

### Bench (`ios-bench/Sources/BenchApp.swift`)

- **`StagePolicy.aneGeneratorSplit`** — same compute units as `aneGenerator`
  (duration/f0n `.cpuOnly`, decoderPre/generator `.cpuAndNeuralEngine`), but the
  generator stage runs the split. Addressable via `--policy aneGeneratorSplit`.
- **`BundleModelCache.generatorSplitModels`** — vends the trunk/body pair for the
  3 s bucket, throws for any other bucket (same loud guard as `aneGenerator`,
  never a silent reroute to the legacy package).
- **`--generator-split-packages <trunk>,<body>`** — override the pair (default
  `kokoro_decoder_har_ane_ln_trunk_3s`,`kokoro_decoder_har_ane_ln_body_3s`); a
  malformed value is a fatal launch error, not a fallback.

The trunk/body packages were already staged into `ios-bench/Resources/coreml/`
and the Xcode project by T6; `prepare_resources.sh` was NOT re-run (empty source
dir, exits nonzero), and both `.mlmodelc` are confirmed present in the built
`.app` bundle.

## Numerical parity (this Mac, CPU_AND_GPU)

New test `swift/Tests/KokoroPipelineTests/GeneratorSplitParityTests.swift` — the
Swift analogue of `scripts/verify_decoder_har_ane_split.py`. It loads the trunk,
body, and monolithic ln packages, feeds identical REAL inputs, and asserts the
split reproduces the monolithic path:

| quantity | split vs monolithic ln |
| --- | ---: |
| waveform SNR (gate 40 dB) | **48.73 dB** ✅ |
| waveform max-abs | 0.0031 |
| spec max-abs | 0.047 |
| phase max-abs | 0.019 |

Not bit-identical, and honestly so: the split materializes the seam as an fp32
output and re-feeds it to the body, where the monolithic graph keeps it internal
in fp16 and fuses across the boundary — so the fp16 GPU backend rounds slightly
differently. 48.73 dB is consistent with the Python verifier's framing (split and
monolithic each ~46 dB from the shared `_forward_pretrim` reference, i.e. very
close to each other). Far above the 40 dB fp16 Core ML gate; a mis-wired seam
(wrong input name/order) would collapse to garbage, not 48 dB.

**Inputs are real, not random**, and this is load-bearing: the generator ends in
`spec = exp(conv_post)`, and every AdaIN block layer-normalizes its activations,
so their magnitude tracks the learned weights and voice, NOT the input scale.
Random `x_pre` overflows fp16 in `exp()` (`spec` → `inf`) and the finiteness
check goes spurious. The fixture
(`swift/Tests/KokoroPipelineTests/Fixtures/generator_split/{x_pre,ref_s,har}.f32`)
is dumped from the same `_build_real_inputs` the Python gates use, via
`scripts/dump_generator_split_inputs.py`.

`CPU_AND_NE` is never gated — this Mac's ANE admits the graph and miscomputes it
into non-finite output (T4/T6). The A14 is a different generation; the device run
is the deciding test.

## Device instrumentation (the point of T7)

Under `--policy aneGeneratorSplit` the console emits, per synthesis pass:

```
ANEGEN: geometry x_pre_real=240/240 har_real=14401/14401
ANEGEN: split trunk finite-fraction=<t> body finite-fraction=<b> trunkNonFinite=<n> bodyNonFinite=<m> total=<N>
```

The split line reports finiteness of BOTH the trunk seam tensor and the final
spec/phase on one line, so a non-finite result localizes to a half. If any value
is non-finite the pass throws `PipelineError.nonFiniteGeneratorOutput` rather than
scheduling NaN audio — same discipline as the single-package `aneGenerator` path,
so a successful run only ever reports `1.0000` on both halves.

## Exact commands for the owner's device gate

Set as scheme arguments (Xcode → Scheme → Run → Arguments) or write to
`Documents/launch_args.txt` for an untethered run — same conventions as the T5/T6
`--policy` / `--mode` paths.

**End-to-end split synthesis (does the A14 ANE load, predict, and stay finite?):**

```
--arms coreml --keys 3s --policy aneGeneratorSplit --out aneGeneratorSplit_3s.json
```

Watch the console for the two `ANEGEN:` lines above. Optional per-half
compute-plan residency (unchanged from T6's rung 2):

```
--mode computeplan --model kokoro_decoder_har_ane_ln_trunk_3s --out computeplan_ln_trunk_3s.json
--mode computeplan --model kokoro_decoder_har_ane_ln_body_3s --out computeplan_ln_body_3s.json
```

To walk a different pair in the same session:
`--policy aneGeneratorSplit --generator-split-packages <trunk>,<body>`.

### Success criterion — read the finite-fraction

- **`ANEGEN: split trunk finite-fraction=1.0000 body finite-fraction=1.0000`
  across a full run** = the A14 ANE executes the chained split correctly. The
  generator runs on the Neural Engine end to end, and the ~5–10× synthesis-power
  win the whole plan is chasing is on the table. Proceed to the battery soak
  (`SPIKE_RUNBOOK.md` Test 3b) under `--policy aneGeneratorSplit`.
- **Either fraction < 1.0** = the A14 ANE miscomputes the admitted graph, exactly
  as this Mac does. The split is a dead end on this silicon: the graph is
  ANE-*admissible* but not ANE-*correct*, and the localization (trunk vs body)
  tells you which half to bisect first. This would confirm T4's "genuinely new
  failure mode" branch for the A14, and the next probe is bisecting the failing
  half stage by stage.

## Stop conditions checked (none tripped)

- Split parity **48.73 dB** ≥ 40 dB under `CPU_AND_GPU` (never gated on
  `CPU_AND_NE`).
- The two-stage generator fits behind the generator-stage boundary with no
  pipeline surgery — one optional protocol method + one predict helper; the
  pipeline, iSTFT, and result shape are untouched.
- `swift test` **49/49** (T3 goldens untouched, + the split-parity assertion);
  `xcodebuild ... CODE_SIGNING_ALLOWED=NO build` **SUCCEEDED**.
- No package re-export, no `kokoro/istftnet.py` change, not wired into the real
  app, `~/Git/FreeReader` untouched.

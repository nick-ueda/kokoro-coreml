# ANE generator bench plumbing — geometry is already correct, and the owner's device commands

Collected: 2026-07-17. Task T5 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on
T4's Status log entry
([ane-generator-coreml-export-2026-07-17.md](ane-generator-coreml-export-2026-07-17.md))
and T2's geometry flag
([ane-pretrim-equivalence-2026-07-17.md](ane-pretrim-equivalence-2026-07-17.md),
"Flagged for the owner — geometry").

## Verdict

`kokoro_decoder_har_ane_3s.mlpackage` is now wired into `ios-bench/` behind
`--policy aneGenerator`, with T3's host iSTFT completing the tail and a
finiteness gate instrumenting every pass. `xcodebuild ... build` succeeds and
`swift test` still passes 48/48 (T3's goldens untouched). No device or
simulator run was performed — that is the owner's step, and the exact
commands are below.

The headline finding is a correction to expectations, not a bug: **the ANE
package needs no geometry fix in the Swift pipeline.** T2's "half-fill"
defect is real, but it turns out to apply only to the *baseline* package's
oversized `har` axis — not to `x_pre`, and not to the ANE package at all.

## Input-geometry investigation (required before wiring)

The task asked me to determine empirically what the Swift pipeline actually
feeds the generator for the 3 s bucket, since T2 flagged a "120 of 240 /
14,401 of 28,801" half-fill in the *Python* runtime
(`build_decoder_har_post_inputs_np`) and warned the Swift path might share
it. Rather than instrument a live Swift run, I reproduced the Python
function directly (it is the single source of truth `decoder_har_post_bucket_impl`
also calls) with real inputs, at `sec=3` — the exact, un-doubled geometry the
Swift executor already computes (`fullF0Len = round(bucketSamples / 300)`,
`KokoroSynthesisExecutor.swift` Stage 5; `HarmonicConstants.upsampleScale`
== Python's `f0_samples_per_step`, both 300):

```
uv run python -c "... build_decoder_har_post_inputs_np(dec, vi, sec=3, asr_len=240, har_t=28801) ..."
→ x_pre_np final shape (1, 512, 240); nonzero extent (0, 240) — 240/240 real
→ har_np   final shape (1, 22, 28801); nonzero extent (0, 14401) — 14401/28801 real

... har_t=14401 (the ANE package's actual axis) instead of 28801 ...
→ x_pre_np final shape (1, 512, 240); nonzero extent (0, 240) — 240/240 real
→ har_np   final shape (1, 22, 14401); nonzero extent (0, 14401) — 14401/14401 real
```

Two things fall out of this:

1. **`x_pre` was never half-filled.** Decoder-pre's `asr` input really is
   120 real frames at native (`sec=3`) geometry — that IS T2's "120" figure
   — but decoder-pre's internal 2x time-upsample (one `upsample_type !=
   "none"` block in `dec.decode`) always produces a *fully computed* 240-frame
   `x_pre` output from those 120 real frames. Confirmed directly: running
   `DecoderPreWrapper` at native geometry (`full_f0_len=240, frame_count=120`)
   returns `x_pre.shape == (1, 512, 240)`, no truncation branch taken. T2's
   table conflated `frame_count` (asr's own INPUT axis, correctly 120) with
   `x_pre`'s OUTPUT axis (always fully real) — worth a correction if that note
   is revisited.
2. **`har`'s half-fill is a property of the baseline package's axis, not of
   the geometry computation.** `har`'s natural (native, un-doubled) STFT
   frame count for a 3 s bucket is exactly 14,401 — `(72,000 padded to
   72,020) / hop 5 + 1`. The T1 baseline package's `har` axis is 28,801
   (2x, matching the plan's Ground Truth "geometry_samples = bucket_samples
   * 2"), so native geometry genuinely only fills half of it. The **ANE
   package's `har` axis is 14,401 by construction** (T4's pre-trim export) —
   which is *exactly* the natural frame count. Feeding it native geometry
   fills it completely.

I also confirmed the Swift side computes the identical geometry (same
`fullF0Len` formula, same `upsampleScale`/`f0_samples_per_step` constant),
and `buildHar`'s STFT (`HarmonicSource.swift`) produces `harFrames` via the
same `(paddedLen - nfft) / hop + 1` arithmetic as `CustomSTFT` — so
`harFrames` for a 3 s bucket is 14,401 in Swift too, with no adaptation
required. Stage 8 of `KokoroSynthesisExecutor.swift` already padded
generator inputs to the model's *actual* input shapes via `inputShapes(from:
genModel)` (pre-existing code, not new to T5) — for the ANE package that
target is 14,401, which `harFrames` already equals, so `zeroPad3D` is a
no-op copy, not a padding operation.

**Net: no geometry-fix code was needed inside the new ANE branch.** What I
added instead is a DEBUG assertion + an always-on log line
(`KokoroSynthesisExecutor.swift`, right after generator-input construction)
that encodes this invariant so a future regression is loud rather than a
silent quality drop:

```
ANEGEN: geometry x_pre_real=240/240 har_real=14401/14401
```

The T1 baseline package's `har` half-fill remains untouched, per the task's
instruction not to fix the legacy path.

## What was built

- **`StagePolicy.aneGenerator`** (`ios-bench/Sources/BenchApp.swift`):
  duration/f0n `.cpuOnly`, decoderPre/generator `.cpuAndNeuralEngine`.
  Addressable via `--policy aneGenerator`. `BundleModelCache.generatorModel`
  vends `kokoro_decoder_har_ane_3s` under this policy and **throws** (fails
  loudly, no silent fallback) for any bucket other than 3s.
- **ANE-package detection** (`KokoroSynthesisExecutor.swift`, Stage 8/9): via
  `genModel.modelDescription.outputDescriptionsByName` containing `spec`+
  `phase` — not a policy flag threaded through the pipeline, per the plan.
  Stage 9 branches: the legacy package's `waveform` output is used directly
  (unchanged behavior); the ANE package's `spec`/`phase` outputs go through
  T3's `hostISTFTInverse(spec:phase:frameCount:)`
  (`swift/Sources/KokoroPipeline/HostISTFT.swift`) before the same final
  `[..., :bucketSamples]`-equivalent trim both paths already shared.
- **Finiteness gate** (T4's inheritance — on this Mac the ANE admits the
  decoder-har-ane graph and then miscomputes it into non-finite output; see
  T4's note). Every ANE-package pass checks `spec`+`phase` for non-finite
  values, prints a greppable line unconditionally (illustrative format below
  — no device/simulator run was performed this session, so this is not a
  captured value; `total` for the 3s bucket works out to
  `2 * 11 * 14,401 = 316,822`, the combined `spec`+`phase` element count):
  ```
  ANEGEN: non-finite output check finite-fraction=1.0000 nonFinite=0 total=316822
  ```
  and throws `PipelineError.nonFiniteGeneratorOutput(finiteFraction:)` on
  ANY non-finite value (not a threshold — one non-finite value fails the
  pass) instead of returning audio. `SynthesisResult.aneGeneratorFiniteFraction`
  carries the value through to callers/results JSON on success (`nil` for the
  legacy package). A failed pass's finite-fraction is recoverable from the
  thrown error's `errorDescription`, which `BenchApp.swift`'s existing
  `recordFailure` already serializes into the results JSON — no new JSON
  plumbing needed for the failure case.
- **`--mode computeplan --model <name>`** (`BenchApp.swift`): loads the named
  bundled `.mlmodelc` under `.cpuAndNeuralEngine`, walks
  `MLComputePlan.load(contentsOf:configuration:).modelStructure.program`
  (confirmed against the actual iOS 26 SDK `CoreML.swiftinterface` — the
  guide's Python sketch doesn't specify Swift-side type names precisely
  enough to guess safely), prints `COMPUTEPLAN: <device>: <n> ops` per
  distinct preferred device and appends `total_ops`/`device_counts`/
  `ane_ops`/`ane_percent` to the results JSON. `MLComputePlan` is iOS 17.4+;
  this app's deployment target is 18.0 (`project.yml`), so no availability
  guard is needed.
- **`prepare_resources.sh`**: stages `kokoro_decoder_har_ane_3s.mlpackage`
  if present under `../coreml/`, skips with a message (does not fail the
  script) if absent — same pattern as the exact-duration packages.

## Build verification (owner-run steps NOT performed)

```
cd swift && swift test                 # 48/48 passed, T3's goldens intact
cd ios-bench && bash prepare_resources.sh   # stages the ANE package (see caveat below)
xcodegen generate --spec project.yml
xcodebuild -project KokoroIPhoneBench.xcodeproj -scheme KokoroIPhoneBench \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build   # BUILD SUCCEEDED
```

Caveat found while staging: `../coreml/` on this checkout currently holds
only `kokoro_decoder_har_post_3s.mlpackage` and
`kokoro_decoder_har_ane_3s.mlpackage` — the decoder-pre/duration/f0ntrain/
other-bucket packages were already absent before this session (unrelated to
T5). `prepare_resources.sh` exits 1 listing them as `MISSING`, but its
`copy()` function only skips a package when its source is absent — it never
deletes an existing destination copy — so the build still succeeded off
`ios-bench/Resources/coreml`'s pre-existing (gitignored) contents plus the
newly-staged ANE package. No device or simulator execution was attempted, no
model files were deleted.

## Exact commands for the owner's device gate

All via `--policy aneGenerator` (StagePolicy defined above) or
`--mode computeplan` (new mode above). Run from a tethered Xcode
launch-argument configuration (`ios-bench/project.yml`'s scheme, or edit in
Xcode → Scheme → Run → Arguments) or via `Documents/launch_args.txt` for an
untethered run — same conventions the existing `--policy backgroundSafe` /
`--mode soak` paths already use.

**Phase 1 gate — single pinned-policy run, no ladder fallback (does
`kokoro_decoder_har_ane_3s` load and predict on the phone, and is the output
finite?):**

```
--arms coreml --keys 3s --policy aneGenerator --out aneGenerator_3s.json
```

Watch the console for `ANEGEN: geometry ...` (confirms full real-content
fill, matching this note's finding) and `ANEGEN: non-finite output check
finite-fraction=...` (the correctness half — 1.0000 means clean; anything
less means the Mac-ANE corruption reproduces on the A14, per T4's "For the
owner" decision table).

**Untethered battery soak (`SPIKE_RUNBOOK.md` Test 3b), same policy:**

```
--mode soak --keys 3s --policy aneGenerator --soak-seconds 900 --out soak_aneGenerator.json
```

Every pass logs its own `ANEGEN:` lines (from inside `executeKokoroSynthesis`,
shared by every mode) in addition to the existing `SOAK:` telemetry line, so
a jetsam or crash mid-soak still leaves per-pass finiteness history in the
Xcode console / device log.

**Compute-plan residency, decoder-pre and the ANE generator (does the graph
actually map to the ANE on THIS device, not just admit at predict time):**

```
--mode computeplan --model kokoro_decoder_pre_3s --out computeplan_decoder_pre_3s.json
--mode computeplan --model kokoro_decoder_har_ane_3s --out computeplan_generator_ane_3s.json
```

Each prints `COMPUTEPLAN: cpu: N ops`, `COMPUTEPLAN: gpu: N ops`,
`COMPUTEPLAN: neuralEngine: N ops` and appends the same counts plus
`ane_percent` to the results JSON — the A14 equivalent of T4's Mac-side
"1,019/1,039 ops on the ANE (98.1%)" figure. A gap between this number and
the Phase 1 gate's finiteness result is itself informative: high ANE
residency with non-finite output would sharpen T4's open question (backend
miscompute, not an admittance problem) rather than leave it a Mac-only
anomaly.

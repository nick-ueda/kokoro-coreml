# Long-form device bench + retrievable WAV capture (T10)

Collected: 2026-07-18. Task T10 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on T9's
windowed executor
([status log entries "T9 COMPLETE"](../Plans/ane-generator-a14-v1.md) and "T9
FOLLOW-UP") and T8's fp32-on-Mac ear-check pass
([ane-generator-windowing-experiment-2026-07-18.md](ane-generator-windowing-experiment-2026-07-18.md)).
Coding only — no device run performed (owner-run per the plan's execution protocol).

## What this closes

T8's "plan globally, vocode in 3 s windows" ear-check ran fp32 on the Mac. T9 wired
the windowed executor (`vocodeWindowed`,
`swift/Sources/KokoroPipeline/WindowedGeneratorExecutor.swift`) behind the T7 split
seam and Mac-verified it (`WINDOWEDPARITY` 46.31 dB), but no A14 execution was
claimed. T10 gives the owner the missing device artifact: a real fp16 A14 WAV of a
long-form (>3 s) utterance synthesized through the windowed path, plus the long-form
RTF and per-run ANE finiteness aggregate — the device gate that decides whether the
windowing design (validated fp32/Mac) also holds up fp16/A14.

## What was found — no new bench flag needed

Per the plan's inheritance note, the windowed path already activates automatically
under `--policy aneGeneratorSplit` for any bucket > 3 s — the provider's 3 s-only
split guard IS the switch (`fullF0Len > WindowedVocodeConstants.winASR` in
`KokoroSynthesisExecutor.swift` Stage 8/9). Checked before writing any code (the
plan's stated gotcha):

- `ios-bench/Resources/bench_inputs/15s.json` is already a REAL ~14 s utterance
  ("The ancient lighthouse stood alone on the rocky cliff…", 219 tokens,
  `canonical_duration_s: 13.9`) — not a synthetic warmup fixture. No `--text` flag
  or new fixture was needed; `--keys 15s` already selects it.
- `kokoro_decoder_pre_15s.mlpackage` and `kokoro_f0ntrain_t600.mlpackage` (the 15 s
  bucket's planner packages — `PipelineConstants.tFramesForBucket[15] == 600`, a
  DIFFERENT axis than the windowing `fullF0Len` below, despite both being
  bucket-keyed frame counts) are both staged in `ios-bench/Resources/coreml/`, and
  `kokoro_duration_t256.mlpackage` covers the 219-token input (padded to 256) — the
  plan's "confirm the planner packages exist at the target bucket" gotcha is clear.
  At 15 s, `fullF0Len = 15*24000/300 = 1200 > 240` (the executor's own windowing
  axis, computed in `KokoroSynthesisExecutor.swift`, NOT `tFramesForBucket`), so the
  windowed path is guaranteed to trigger — 1200 divides evenly by the 200-frame
  core stride into exactly 6 windows (T9's status-log entry already established
  this `fullF0Len` mapping; re-derived and confirmed here, not assumed).
- The bench had NO existing audio-persist path (only `soak` mode plays audio through
  `AVAudioEngine`, never to disk) and no `SynthesisResult` field went unread — so a
  new minimal WAV writer was warranted, not a duplicate.

## What was built

All changes are in `ios-bench/Sources/BenchApp.swift`; nothing in
`WindowedGeneratorExecutor.swift` or `KokoroSynthesisExecutor.swift` needed to
change — T9 already computes and logs everything T10 needed to persist, just
per-call rather than per-run.

- **`writeWavMono16NoPeakNormalize(path:samples:sampleRate:)`** — a fresh minimal
  writer (24 kHz mono 16-bit PCM), NOT a reuse of the Mac `kokoro-bench` CLI's
  `writeWavMono16` (`swift/Sources/KokoroBenchmark/main.swift`), because that one
  peak-normalizes per file — exactly what the plan's gotcha warns against (it would
  hide real output level from the owner's ear-check). Mirrors
  `scripts/probe_har_pretrim_adain_equivalence.py`'s `_write_wav` /
  `scripts/probe_windowed_vocode.py`'s `P._write_wav` instead: plain clip-to-[-1,1]
  and quantize, since the pipeline's `SynthesisResult.audio` is already scaled to
  that range.
- **`WarmSeries`** gained `aneFiniteFractions: [Double?]` (one entry per timed
  iteration, `nil` when that call used a legacy in-graph-iSTFT package) and
  `lastAudio: [Float]` (the most recent timed iteration's raw audio — every timed
  call re-synthesizes the identical input under a fixed seed, so any one iteration
  is representative).
- **`persistWavAndLogSummary(key:policyName:series:canonicalDuration:)`** (new,
  called from `runCoreMLArm` only — ladder mode, the default `--mode`) writes
  `Documents/audio-<key>-<policy>.wav` and logs both the WAV path and an aggregate
  `ANEGEN SUMMARY <key> policy=<name>: iterations=N min-finite-fraction=…
  mean-finite-fraction=… total-wall-s=… total-audio-s=… rtf=…` console line — the
  per-window/per-call `ANEGEN:` lines T7/T9 already emit are per-timed-iteration;
  this is the same data rolled up over the WHOLE run. `--mode matrix` shares
  `recordSuccess`'s new JSON fields (harmless bonus visibility) but does not get a
  WAV write — out of T10's stated scope (long-form ladder-mode bench).
- **`recordSuccess`** now also writes `total_wall_s`, `total_audio_s`, `overall_rtf`
  (summed/averaged over every TIMED iteration, vs the existing `median_s`/`rtf`'s
  single-call figures) and, when at least one call went through an ANE spec/phase
  package, `ane_finite_fractions` / `ane_min_finite_fraction` /
  `ane_mean_finite_fraction`, plus `wav_path` when a WAV was written — all additive
  keys in the existing `Documents/<out>.json` record shape.
- The existing `--keys 3s` / non-windowed paths are unaffected: `aneFiniteFractions`
  is `nil`-filled (so the new JSON keys are simply absent) for any non-ANE package,
  and `persistWavAndLogSummary` still writes a WAV for them too (harmless, and
  useful in its own right for a quick 3 s ear-check) — additive only, per the
  plan's gotcha.

## Exact commands for the owner's device gate

Set as scheme arguments (Xcode → Scheme → Run → Arguments) or write to
`Documents/launch_args.txt` for an untethered run — same conventions as the T5–T9
`--policy` paths.

**Long-form windowed synthesis, fp16 A14, with WAV capture:**

```
--arms coreml --keys 15s --policy aneGeneratorSplit --out aneGeneratorSplit_15s.json
```

This is 2 warmups + 5 timed iterations (`BenchRunner.warmups`/`.iterations`) of the
same ~14 s utterance, entirely through the windowed path — `fullF0Len = 1200` at the
15 s bucket, which divides evenly by the 200-ASR-frame core stride into exactly **6
windows** (`windowedVocodePlan`; no partial last core at this bucket). First load
includes the ANE AOT compile (can take minutes; silence after the "loading" line =
compiler, not a hang). Watch the console for, per timed iteration:

```
ANEGEN: windowed windows=6 finite-fraction=1.0000 nonFinite=0 total=<N>
```

then once at the end of the (key, policy) pair:

```
coreml 15s policy=aneGeneratorSplit WAV written: file:///…/Documents/audio-15s-aneGeneratorSplit.wav
ANEGEN SUMMARY 15s policy=aneGeneratorSplit: iterations=5 min-finite-fraction=1.0000 mean-finite-fraction=1.0000 total-wall-s=… total-audio-s=69.5 rtf=…
```

Optional extra coverage (other buckets that also exceed one 3 s window — 7 s and
30 s bucket packages are staged too):

```
--arms coreml --keys 7s,30s --policy aneGeneratorSplit --out aneGeneratorSplit_7s_30s.json
```

### Success criterion — read the finite-fraction and RTF

- **`min-finite-fraction=1.0000` across all 5 timed iterations** = the A14 ANE
  executes the windowed split correctly for the WHOLE long-form run, not just one
  pass — the fp16-on-device analogue of T9's Mac-side `WINDOWEDPARITY`. Below 1.0 on
  any iteration would instead have THROWN (`PipelineError.nonFiniteGeneratorOutput`)
  — the run reaching the summary line at all is itself a first-order signal, not
  just the printed number.
- **`rtf`** (in the `ANEGEN SUMMARY` line, or `overall_rtf` in
  `Documents/aneGeneratorSplit_15s.json`) is the long-form wall-time-to-audio-time
  ratio this task was asked to surface — read it off directly, no arithmetic on the
  owner's part needed.

### WAV retrieval (devicectl / Finder container)

The app already sets `UIFileSharingEnabled` (see `SPIKE_RUNBOOK.md`'s soak-CSV
retrieval note), so any of these work — pick whichever is fastest given how the
phone is connected:

1. **Cable + Finder (no devicectl needed):** connect the iPhone to the Mac, open
   Finder, select the iPhone in the sidebar, click the **Files** tab, find
   **KokoroIPhoneBench** in the app list, and drag `audio-15s-aneGeneratorSplit.wav`
   (and `aneGeneratorSplit_15s.json`, for the RTF/finite-fraction numbers) out to the
   Mac.
2. **Xcode Devices and Simulators container download:** Xcode → **Window → Devices
   and Simulators** → select the iPhone → **Installed Apps** → select
   **KokoroIPhoneBench** → gear icon → **Download Container…** → the WAV is at
   `AppData/Documents/audio-15s-aneGeneratorSplit.wav` inside the saved
   `.xcappdata` (right-click → **Show Package Contents**).
3. **No cable (AirDrop):** on the phone, **Files** app → **On My iPhone** →
   **KokoroIPhoneBench** → long-press the WAV → **Share** → AirDrop to the Mac.

Then listen (fp16 A14 quality + seam continuity — window boundaries land roughly
every 2.5 s, at the 200-ASR-frame core stride) against the fp32-on-Mac artifacts T8
already ear-checked, as a control:

- `Scratchpad/windowed_vocode_windowed.wav` — T8's fp32 windowed render (owner
  already found this indistinguishable from `original`).
- `Scratchpad/windowed_vocode_original.wav` — the shipped full-length model, T8's
  ear-check anchor.

A/B: does the fp16-on-A14 WAV sound as close to `original` as T8's fp32 render did?
Any new seam pumping or texture shift versus the fp32 windowed render would be a
fp16-specific artifact T8's Mac listen could not have caught.

## Stop conditions checked (none tripped)

- `swift test` **62/62** (unchanged from T9's FOLLOW-UP commit — no
  `WindowedGeneratorExecutor.swift`/`KokoroSynthesisExecutor.swift` change was
  needed for this task).
- `xcodebuild -project ios-bench/KokoroIPhoneBench.xcodeproj -scheme
  KokoroIPhoneBench -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO
  build` **SUCCEEDED**; only pre-existing Swift 6-mode actor-isolation warnings
  (unrelated call sites in `runG2PMode`/`runComputePlanMode`/`runMLXArm`, present
  before this task) — no new warnings from the added code.
- No package re-export, no `kokoro/istftnet.py` or `WindowedGeneratorExecutor.swift`
  change, `~/Git/FreeReader` untouched. Additive only: the existing `--keys 3s`
  micro-bench (T7 path) is unaffected beyond also gaining a WAV write and the new
  (harmless, empty-when-N/A) JSON keys.
- No `ANECCompile`/espresso error, no missing planner package, no deviation from
  Ground Truth — nothing to escalate.

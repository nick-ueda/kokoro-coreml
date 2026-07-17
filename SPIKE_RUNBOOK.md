# FreeReader spike runbook — Kokoro Core ML on the iPhone 12 Pro

Local fork of mattmireles/kokoro-coreml prepared 2026-07-15 for the spike in
`FreeReader/KOKORO_COREML_SPIKE.md`. Everything below runs from
`ios-bench/KokoroIPhoneBench.xcodeproj` (already generated; models already
downloaded and staged). Results append to the app's `Documents/<out>.json`
after every pass, so pull partial data even after a crash via Xcode →
Devices & Simulators → the app → download container.

## What was changed vs upstream (all in this clone, not in FreeReader)

- `ios-bench/Sources/BenchApp.swift`
  - new `StagePolicy.cpuAndNeuralEngine` — ALL four stages on CPU+ANE, GPU
    excluded. Deliberately not on the fallback ladder.
  - `--policy <name>` pins ladder mode to one policy, no fallback.
  - `--mode soak` — the background test. Synthesizes in a loop and PLAYS the
    audio (AVAudioSession .playback); pacing keeps ~20 s buffered ahead so
    synthesis keeps firing the whole run. Logs `SOAK:` lines with app state,
    thermal state, wall time, x-realtime. `--soak-seconds N` (default 900).
  - lifecycle logging: `SOAK: >>> screen LOCKED / didEnterBackground / ...`
- `ios-bench/project.yml` — your team (7S9CS3LT7F) + com.nickueda bundle id;
  explicit Info.plist with **UIBackgroundModes=audio** (without it iOS
  suspends the app on lock and the soak proves nothing). Also
  **UIFileSharingEnabled + LSSupportsOpeningDocumentsInPlace** so Documents
  (soak CSVs, results JSON, launch_args.txt) shows up in the Files app.
- Soak telemetry CSV + untethered launch support (see "Test 3b") —
  `SoakCSVLogger`, battery/footprint columns, `launch_args.txt` fallback.
- `ios-bench/prepare_resources.sh` — `kokoro_duration_exact_t*` optional
  (they're not in the HF set; only `--exact-duration 1` needs them).

## One-time setup

Open `ios-bench/KokoroIPhoneBench.xcodeproj`, plug in the iPhone 12 Pro,
select it as the destination. Signing is automatic under your team; first
install may need Settings → General → VPN & Device Management trust.

Launch arguments are set per run in Product → Scheme → Edit Scheme… → Run →
Arguments. The app runs on launch; watch the Xcode console.

## Test 1 — foreground RTF, default/ladder policies (spike step 1)

Arguments: `--arms coreml --keys 15s,30s`

First iteration per bucket triggers the on-device ANE AOT compile — the Mac
spent ~20 min on the 30 s bucket; expect longer on A14. Silence after the
"loading" log line is the compiler, not a hang. Warm medians and the policy
that actually ran land in `Documents/results.json` (`rtf` there is
wall/audio — LOWER is better; 0.43 ≈ 2.3x realtime).

## Test 2 — THE decisive number: ANE-only RTF (spike step 2)

Arguments: `--arms coreml --keys 15s,30s --policy cpuAndNeuralEngine --out ane.json`

No ladder fallback: if a stage can't run without the GPU, the run records
the failure and that is the answer. Viability bar from the spike brief:
**≥1.2x realtime** (i.e. `rtf` ≤ 0.83) — below that, background listening
can't keep up and the whole Core ML move loses its reason.

## Test 3 — locked-screen soak (spike step 3)

Arguments: `--mode soak --keys 15s --soak-seconds 900 --out soak.json`

(Policy defaults to cpuAndNeuralEngine in soak mode.)

1. Launch; wait for `SOAK: pass N …` lines to flow (first pass includes the
   AOT compile) and audio to come out of the speaker.
2. **Lock the screen. Leave it locked 2+ minutes.** Console keeps working
   over the debug cable. Pass: `SOAK:` lines keep appearing with
   `app=background`, audio keeps playing, no
   `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`.
3. Unlock, then repeat via home-swipe + open another app (true backgrounding
   without lock).
4. Optional control that the harness detects the crash it's designed to
   catch: rerun with `--policy cpuAndGPU` and lock — the MLX experiment
   predicts a fast abort.

## Test 3b — untethered cold soak (the real thermal/battery number)

Test 3's `thermal=serious` reading was confounded: phone pre-heated by prior
runs, **charging**, and attached to Xcode. The cold soak removes all three.
Soak mode logs every pass AND every lock/unlock/background transition to
`Documents/soak-<timestamp>.csv` (fsync'd per line; columns include
x-realtime, thermal, buffered s, battery % + charge state, phys_footprint
MB), so no console is needed.

Launch arguments don't exist on a home-screen launch, so the harness falls
back to `Documents/launch_args.txt` — seeded automatically by the first soak
run with its own settings, editable afterwards in the Files app (On My
iPhone → KokoroIPhoneBench). The file is only read when the process has NO
arguments: Xcode-tethered runs are never affected by it.

1. Run the soak once from Xcode (installs the app, seeds `launch_args.txt`,
   and pays the ANE AOT compile so the cold run isn't polluted by it). Press
   Stop. Edit `launch_args.txt` in the Files app if you want a different
   bucket/duration — e.g. `--soak-seconds 3600` for the full hour.
2. Unplug. Let the phone cool to ambient (30+ min, screen off).
3. Launch from the home screen. Wait for audio, glance at the on-screen
   status line, then lock the screen and leave it locked for the duration.
4. Pull `soak-<timestamp>.csv` from the Files app (or AirDrop it). Gates:
   x-realtime ≥ 1.2 sustained, thermal ≤ serious, battery %/h acceptable
   (first vs last `battery_pct` row), `footprint_mb` flat (jetsam headroom).
   `battery_state` must read `unplugged` throughout — that column IS the
   proof the charging confound is gone.

Live logs without the debugger, if wanted: Console.app over Wi-Fi/cable
still shows the `SOAK:` lines; `soak.json` in Documents has the same data
as before, now with battery/footprint fields.

## Test 4 — on-disk size (spike step 4)

The staged bundle here is ~981 MB for ALL buckets/sizes (`du -sh coreml/`).
The FreeReader-relevant subset is smaller — sum the buckets you'd actually
ship (e.g. one or two decoder buckets + matching f0n + duration sizes):
`du -sh coreml/kokoro_decoder_*_15s.mlpackage coreml/kokoro_f0ntrain_t600.mlpackage coreml/kokoro_duration_t*.mlpackage`
Compare against the current 327 MB safetensors download.

## Test 5 — quality by ear (spike step 5)

Soak mode plays through the speaker — listen during Test 3 (bucket-boundary
artifacts, fp16 vocoder harshness) against the same passage on the MLX path
in FreeReader.

## Recording the verdict

Append measured numbers to `FreeReader/KOKORO_COREML_SPIKE.md` (new
"Results" section): per-test RTFs, which policies the ladder settled on,
soak outcome, sizes, ear notes. The decision tree is
memory:kokoro-coreml-keep-or-kill — ANE-only pass → port Kokoro to Core ML;
fail → Kokoro likely comes out of the app.

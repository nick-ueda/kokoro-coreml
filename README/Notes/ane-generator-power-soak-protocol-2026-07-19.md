# Hard power number under windowed long-form (T11) — owner soak protocol

Written: 2026-07-19. Task T11 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on T10's
device gate
([status log entry "T10 DEVICE GATE RUN"](../Plans/ane-generator-a14-v1.md)) and the
"Why" section's baseline claim (backgroundSafe policy, CPU generator, ~2.5–3.5 W ≈
31–43 %/h on the aged pack). **No code was added.** This note is the owner-run
protocol; the actual soak and the resulting Watts number are still owed.

## What this closes — and why zero code was needed

T11's job was to make the sustained soak runnable and document the exact protocol,
not to build new infrastructure. Checked before writing anything (the plan's own
"PREFER no new code" gate):

- `SPIKE_RUNBOOK.md` **Test 3b** already drives a hands-free, minutes-to-hours-long,
  untethered, screen-locked soak: `--mode soak` loops `synthesizeOnce` (the identical
  `executeKokoroSynthesis` call the ladder/T10 device gate used) for `--soak-seconds`,
  plays the audio through `AVAudioSession .playback` so `UIBackgroundModes=audio`
  keeps the process alive when the screen locks, and logs every pass plus every
  lock/unlock/background transition to `Documents/soak-<timestamp>.csv`
  (`ios-bench/Sources/BenchApp.swift:1071` `runSoak()`, `SoakCSVLogger` at line 547).
  Columns: `time,elapsed_s,event,pass,wall_s,audio_s,x_realtime,app_state,thermal,
  buffered_s,battery_pct,battery_state,footprint_mb`.
- Soak mode already takes an arbitrary `--policy` (`BenchRunner.policyOverride`,
  defaults to `backgroundSafe`) and an arbitrary `--keys` bucket (`soakKey`, defaults
  to `15s`) — both plumbed straight into the same `BundleModelCache` +
  `executeKokoroSynthesis` path every other mode uses. Nothing in `runSoak` is
  policy- or bucket-specific.
- Because of that, `--mode soak --keys 15s --policy aneGeneratorSplit` **already**
  exercises T9's windowed executor on every pass, for the exact same reason T10's
  ladder-mode run did: the provider's 3 s-only split guard is what activates
  windowing for any bucket > 3 s (see T9/T10's status-log entries) — soak mode calls
  the identical `synthesizeOnce`/`executeKokoroSynthesis` function, so there is no
  separate "soak doesn't know about windowing" code path to add. The per-pass
  `ANEGEN: windowed windows=6 finite-fraction=…` console line T9 already emits fires
  on every soak pass too, for free.
- Soak mode already logs untethered (`launch_args.txt` fallback,
  `seedLaunchArgsFileIfMissing`), already has the battery/thermal/footprint columns
  Test 3b's protocol reads, and already excludes the charging confound via the
  `battery_state` column. There is nothing left for a "loop long-form synthesis
  driver" to add — Test 3b **is** that driver.

Net: T11 is a documentation task. `ios-bench/Sources/BenchApp.swift` and
`swift/Sources/KokoroPipeline/` are unchanged; `swift test` / `xcodebuild` were not
re-run since nothing was touched (both were green at HEAD per T9's follow-up commit).

## Owner protocol — exact commands

Two arms, same conditions, same input, only the policy differs. Set as scheme
arguments (Xcode → Scheme → Edit Scheme… → Run → Arguments) for the tethered priming
run of each arm; the untethered leg then reads them back from
`Documents/launch_args.txt`.

**Arm A — ANE (the number this task exists to produce):**

```
--mode soak --keys 15s --policy aneGeneratorSplit --soak-seconds 3600 --out soak_ane.json
```

**Arm B — CPU baseline (what ships today; the ~2.5–3.5 W this must beat):**

```
--mode soak --keys 15s --policy backgroundSafe --soak-seconds 3600 --out soak_cpu.json
```

`backgroundSafe` is already soak mode's default policy, so this reproduces the exact
configuration the plan's "Why" section's 31–43 %/h figure came from — rerun it fresh
rather than reusing the old number, so both arms share today's battery health, ambient
temperature, and SoC band.

3600 s (60 min) is a deliberately generous default: the CPU arm should show ≥2 clean
5 %-battery ticks well inside 20–30 min (31–43 %/h) and can be stopped early once the
CSV shows 3+ ticks; the ANE arm's drain rate is the unknown this soak exists to
measure; if it turns out to be far below CPU, don't be surprised if the 60-minute
window is what's needed for 2 clean ticks. Stopping early is always safe — the CSV is
fsync'd per line, so a partial run still has usable data. If 60 minutes isn't enough,
edit `--soak-seconds` in `Documents/launch_args.txt` (Files app) and relaunch from the
home screen; a fresh launch starts a new `soak-<timestamp>.csv`, so treat each launch
as an independent trial and use whichever single trial has ≥2 clean ticks (don't
splice CSVs across a phone reboot/relaunch — the battery-tick math needs one
continuous unplugged window).

### Step by step (mirrors `SPIKE_RUNBOOK.md` Test 3b, once per arm)

1. **Tethered priming run.** Launch from Xcode with the arm's arguments above. Wait
   for `SOAK:` lines to flow and audio to play. This installs the app, seeds
   `launch_args.txt`, and pays the one-time ANE AOT compile for Arm A (silence after
   the "loading" line is the compiler, not a hang — can take minutes on a
   thermally-cool phone, much longer under pressure per T10's "4.9 s cool → 411 s
   thermally-pressured" finding). Confirm Arm A's console shows
   `ANEGEN: windowed windows=6 finite-fraction=1.0000` before moving on — that's the
   same sanity check T10 already validated on-device; seeing it here just confirms
   this specific soak launch reached the windowed path. Press Stop once a handful of
   passes have flowed and wall times look stable (see "Warm first" below).
2. **⚠️ Between arms, clear `Documents/launch_args.txt` before priming the next
   arm.** `seedLaunchArgsFileIfMissing()` never overwrites an existing file — if Arm
   A's file is still present when you tether-launch Arm B, the untethered leg will
   silently keep replaying Arm A's `--policy aneGeneratorSplit`, not Arm B's
   `backgroundSafe`. Delete or hand-edit `launch_args.txt` in the Files app (On My
   iPhone → KokoroIPhoneBench) between arms — same file, same gotcha the plan's
   "seeded automatically... editable afterwards" language is warning about.
3. **Unplug. Let the phone cool to ambient** (30+ min, screen off) before each
   untethered leg — this also resets the thermal-state confound between the two arms
   so neither starts pre-heated by the other's run or by the tethered priming pass.
4. **Match starting SoC across arms.** Both legs must run inside the **35–70 %**
   battery band (low SoC + Low Power Mode inflate the number). Recharge (tethered,
   doesn't count) between arms if needed so both start around the same percentage —
   e.g. both starting ~60–65 % gives headroom to stay in-band for a full 60-minute
   run even at the CPU arm's faster drain.
5. **Launch from the home screen** (untethered — no debugger, no charging, no
   Xcode-attached confound). Wait for audio, glance at the on-screen status line,
   then **lock the screen and leave it locked for the whole run**. Screen-on adds
   display power that pollutes the draw — this is the real background-reading use
   case, so it's also the number that matters.
6. **Pull `soak-<timestamp>.csv`** from the Files app (or AirDrop it) after the run.
   Repeat steps 1–6 for the other arm.

### Warm first — exclude the cold-compile transient

The soak's `BundleModelCache` is constructed once and reused for every pass in the
loop, so only the *first* pass through a freshly-launched process pays model
load/compile cost — but per T10, that cost can be large and variable under thermal
pressure. Before starting the %/h clock for either arm:

- On the **tethered priming run**, watch `SOAK: pass N wall=…` lines until wall time
  stabilizes (no more multi-second-plus outliers) — that is the process's ANE
  program becoming resident, not per-pass work.
- On the **untethered leg**, don't start reading the CSV's battery-tick window from
  `pass 1`'s row. Let the run flow for ~1 minute after audio starts (a few passes),
  then take the *first* battery-tick reading from there — this keeps any residual
  cold-load bump for this specific process instance out of the steady-state number,
  mirroring the plan's "Cold ANE compile is a STARTUP cost, not a per-synthesis cost"
  instruction.

### Hygiene checklist (all from `SPIKE_RUNBOOK.md` Test 3b + this task's brief)

- [ ] `battery_state` column reads `unplugged` for **every** row in the window you use
      for the %/h calculation — any `charging` rows mean the run was contaminated
      (a past soak was polluted by a mid-run plug-in; this is exactly that confound).
- [ ] Battery % stayed inside **35–70 %** for the whole measured window.
- [ ] `app_state=background` for the measured window (screen was actually locked, not
      just dimmed) — `thermal` column present per row for the trajectory field below.
- [ ] `footprint_mb` roughly flat (no runaway growth → jetsam risk, though this is a
      secondary check, not the power number).
- [ ] If the CSV/console stops producing new `pass` rows before `--soak-seconds`
      elapses, check `Documents/<out>.json` for a `"soak run failed"` record — the
      likely cause would be T9's finiteness gate throwing
      (`PipelineError.nonFiniteGeneratorOutput`), which has not occurred on the A14
      in any prior gate run (T10: `finite-fraction=1.0000` across 5 iterations), but
      a multi-tens-of-minutes soak under sustained thermal load is a new stress
      regime this task hasn't tested. If it happens, that itself is the finding —
      escalate, don't retry silently.
- [ ] **Do not use `powermetrics` on the phone.** It is a macOS-only tool; it cannot
      attach to or measure an iPhone process. (The plan's earlier "powermetrics"
      shorthand in the CLAUDE-facing task text meant Mac-side measurement from a
      different context — not applicable here.) The untethered battery-%-tick method
      below is the primary number; Instruments (next section) is the only valid
      on-device component-level cross-check.

### Optional cross-check — tethered Instruments Energy / Neural Engine track

Not a replacement for the untethered number (tethered adds cable/debugger overhead
and Instruments itself costs some power), but useful to confirm the Neural Engine is
actually active during Arm A and silent during Arm B, per this repo's standard
Level 2/3 validation ladder (`CLAUDE.md` Part 5):

1. Xcode → **Product → Profile** (⌘I) → **Core ML** template (or the separate
   **Energy Log** template for a dedicated power number).
2. Add the **Neural Engine** instrument track if not already present.
3. Run the same `--mode soak --keys 15s --policy aneGeneratorSplit` args for a few
   minutes tethered. Look for sustained activity on the Neural Engine track and an
   Energy Log "high"/elevated impact level; repeat briefly for `backgroundSafe` and
   confirm the Neural Engine track goes quiet.
4. This is a qualitative "is the ANE actually doing the work" receipt, not the
   headline Watts number — record it as a sanity note, not the primary result.

## %/h → W conversion

Aged-pack invariant (already used to derive the plan's own ~2.5–3.5 W baseline
figure — reused here, not re-derived): **76 % battery health ⇒ ~8.2 Wh** usable
capacity on this iPhone 12 Pro. So:

```
W = (%/h) × 8.2 Wh / 100 %  =  (%/h) × 0.082
```

Sanity check against the plan's own numbers: 31 %/h × 0.082 = **2.54 W**, 43 %/h ×
0.082 = **3.53 W** — matches the "Why" section's ~2.5–3.5 W range, confirming this is
the same formula, not a new one.

Compute the drain rate from each arm's CSV as: take the first and last `pass` row
inside the hygiene-checked unplugged/mid-SoC/steady-state window, `Δ% = battery_pct
(first) − battery_pct (last)`, `Δh = (elapsed_s (last) − elapsed_s (first)) / 3600`,
`%/h = Δ% / Δh`.

## Results skeleton — fill in from the two CSVs

| Metric | Arm A — ANE (`aneGeneratorSplit`) | Arm B — CPU baseline (`backgroundSafe`) |
|---|---|---|
| CSV file | `soak-____.csv` | `soak-____.csv` |
| Measured window (unplugged, mid-SoC, steady-state) | ____ → ____ (elapsed ____ min) | ____ → ____ (elapsed ____ min) |
| Battery % (first → last in window) | ____% → ____% (Δ____%) | ____% → ____% (Δ____%) |
| Drain rate | ____ %/h | ____ %/h |
| **Power (× 0.082 W per %/h)** | **____ W** | **____ W** |
| Thermal trajectory (nominal→fair→serious?, any excursions) | ____ | ____ |
| Sustained-load RTF (median `x_realtime` in-window; audio/wall, higher=better) | ____ | ____ |
| Any non-finite / thrown errors mid-soak? | ____ | ____ |
| Instruments cross-check (Neural Engine track active? Energy Log level) | ____ | (should be quiet) |

**Verdict:** ANE power is **____×** lower than the CPU baseline (plan's stated
expectation: ~5–10×, measured on the generator stage alone — the whole-system number
here will likely land lower than that since duration/f0n/decoder-pre draw is
unchanged between arms; report what's actually measured, don't round to the
expectation).

## What T11 leaves for T12

Once this note's table is filled in, the plan's next task (T12) wires the runtime
`KokoroModelProvider` to vend the windowed split for production background synthesis
and adds launch-time ANE pre-warm (already flagged MANDATORY by T10's cold-compile
finding). T12 is unblocked by this task being *documented*, not by the soak actually
running — but the owner should still run it before green-lighting T12, since a
surprise (non-finite under sustained thermal load, or a smaller-than-expected power
win) would change T12's urgency/design, not just its paperwork.

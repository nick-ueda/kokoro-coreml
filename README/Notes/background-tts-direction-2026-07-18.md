# Background TTS on the A14 — architecture direction & FluidAudio evaluation

Collected: 2026-07-18. Strategic memory for the FreeReader background-synthesis
spike, sitting above the task-level notes
([ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md) T1–T7). Written after
T7 proved the split generator runs finite on the A14 ANE, when the conversation
turned from "does it work" to "how do we ship long-form reading."

## The binding constraint (never lose this)

FreeReader synthesizes TTS **in the iOS background**, where **Metal/GPU is
banned**. So the only power path is **CPU + Apple Neural Engine**, and every
design choice is downstream of that one fact. The T7 soak is the proof it holds:
the app hit `didEnterBackground (Metal now banned)` and kept producing finite
audio on the A14 at ~2× realtime, ~half the CPU baseline's battery drain.

## The core insight: plan prosody globally, vocode in 3 s windows

The ANE generator is capped at a **3 s** bucket (the 16,384-element axis limit).
But that cap is **only the generator (vocoder)**. The stages that actually *plan*
prosody — duration, F0Ntrain (the F0/energy contour), the text encoder,
decoder-pre — run over the whole sentence and are NOT axis-limited (a 15 s
sentence's F0 curve is ~1,200 frames, `x_pre` is 512×1,200 — all well under
16,384; decoder-pre already has 15 s/30 s packages).

So the requirement "15–30 s buckets for good prosody on long sentences" resolves
to: **the prosody-planning span must be sentence-length (15 s holds most
sentences, 30 s holds essentially any), and that is FREE on the background-safe
path.** Only the vocoding must be windowed. The vocoder is mostly a local
operation, so windowing the rendering of a globally-planned contour should be far
less lossy than windowing the planning itself. The one residual unknown — whether
per-window AdaIN normalization is audible — is exactly what the
[windowing experiment (T8)](ane-generator-windowing-experiment-2026-07-18.md)
tests. **That experiment is the pivot** for the entire background-reading
direction: pass → ship the 3 s ANE generator with global planning; fail → extend
the bucket or use the GPU handoff.

## The GPU-foreground / ANE-background handoff (contingent design)

If windowed 3 s vocoding proves insufficient, the fallback is a lifecycle handoff,
because each mode has a different strength:

- **Foreground (Metal allowed):** the existing, validated `.cpuAndGPU` pipeline
  with the **full bucket range (7/15/30 s)** — true single-pass long-sentence
  vocoding, best prosody, no windowing. Also the place to **aggressively
  pre-buffer** (synthesize minutes ahead while the user is present).
- **Background (Metal banned):** the `aneGeneratorSplit` 3 s path — power-efficient,
  background-safe, topping up the buffer.

Mechanics: Core ML fixes compute units at model-load time, so swap `MLModel`
instances on the lifecycle transitions already logged in the soak. Costs:
pre-warm the ANE models at launch (cold ANE compile was 4.9 s cool → 411 s on a
thermally-pressured phone — a first-background-handoff stall otherwise); keep both
pipelines resident (RAM budget on 4 GB); switch only at chunk boundaries.
**Limitation:** a long *background* session still leans on ANE-3 s, so the handoff
improves foreground quality but does not remove the 3 s-quality question — it only
confines it. Prefer the simpler "ANE-3 s everywhere" if T8 passes; add the handoff
only if it fails. (Simpler is better.)

## FluidAudio evaluation — does it already do this? (investigated 2026-07-18)

[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) has a
`KokoroAne` backend that runs Kokoro 82M on the ANE — the same target as this
spike. Apache-2.0, iOS 17+, actively maintained (~2.5k stars, v0.12.4). Verdict:
**strong independent validation of our architecture, but NOT a drop-in for
FreeReader's background/A14 use case.**

**Why it doesn't drop in:**

1. **Not background-safe as shipped.** Their 7-stage split puts the **iSTFT
   "tail" on the GPU** (current default `ane-tail-gpu`), because — in their words —
   *"ANE rejects the exp/sin/iSTFT."* The GPU is banned in the iOS background. The
   two GPU-free presets (`all-ane`, `cpu-only`) that would dodge it are
   **documented to crash** (SIGSEGV in libBNNS on that tail). So there is **no
   validated GPU-free path.** Notably, this is the exact stage *we* took out of
   Core ML and do in host Swift/vDSP (`HostISTFT.swift`) — so *our* iSTFT is
   background-safe and theirs is not. Their weakness is precisely our design choice.
2. **Zero A14 evidence.** Every KokoroAne benchmark is a Mac (M1/M2/M5). No hits
   for "A14"/"iPhone 12"/"4GB" anywhere in their repo. The only iPhone TTS demo is
   a *different* backend (Supertonic on iPhone 17 Pro). And they've already hit the
   admit-then-crash ANE failure class on the **M5** — a newer/beefier ANE than the
   A14 — so A14 fragility is a real risk they've never tested.
3. **Forward risk:** their open issue #738 flags that iOS 27 beta may restrict
   background *ANE* too, not just GPU. Unconfirmed, but a watch item for everyone.

**What it validates and teaches:**

- **Independent convergence on our architecture:** multi-graph split, generator
  convolutions on the ANE, iSTFT kept off the ANE. Second data point that the hard
  part (generator on the ANE) is real.
- **They're ahead on input length:** ~510 phonemes / ≤2,000 acoustic frames
  ≈ **25–30 s per pass** (vs our 3 s), via the multi-graph split + hard frame
  caps. This is our #1 gap. **The most valuable thing to extract from FluidAudio
  is their length trick** — how they keep the ANE generator under axis limits at
  2,000 frames — as the fallback if T8's windowing fails. Their models come from a
  different conversion lineage (`laishere/kokoro-coreml`, not our `mattmireles`
  upstream), so the models themselves don't drop into our pipeline; the *technique*
  is the takeaway.
- Peak RSS ~0.8 GB (M5) — plausibly within a 4 GB budget but unverified on iOS/A14.
  No streaming; no built-in chunker (you segment upstream).

## Open decisions, in priority order

1. **Run T8 (windowed vocoding).** Decides whether the background 3 s path can
   deliver long-sentence prosody. Cheapest de-risk; blocks everything else.
2. **Listen to actual A14 device audio.** `finite-fraction=1.0000` proves no
   NaN/Inf, not that it sounds good (fp16 ANE can be finite-but-degraded). Still
   owed, and stacked on T2's never-run ear check.
3. **A hard power number** (tethered `powermetrics --samplers ane`, or a longer
   soak) to replace the coarse "~half the baseline" from 5%-granularity battery.
4. **Only if T8 fails:** the GPU handoff, or extending the ANE bucket (FluidAudio
   technique).
5. **Production wiring:** the split lives behind `KokoroModelProvider`; only the
   bench provider vends it. The runtime `KokoroPipeline` inherits the `nil` default
   (deliberate — T7 did not wire the real app). That plus a warmup strategy is the
   eventual integration work.

# Windowed vocoding of a global prosody plan — experiment design (T8)

Collected: 2026-07-18. Task **T8** of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md). Refines and
supersedes the loosely-scoped "AdaIN windowed-stats experiment" in that plan's
Tasks list, using the geometry we now know is actually ANE-deployable.

**Status: RUN + EAR-CHECK PASSED 2026-07-18 — GREEN-LIT.** Probe:
`scripts/probe_windowed_vocode.py`. Headline: naive per-window vocoding is 2.1 dB
SNR vs a full-length reference — a REAL AdaIN-context effect, not a bug — BUT that
reference is an *undeployable* global vocode (12.5 s body > the 16,384 limit), and
the owner ear-check (the actual gate) found **`windowed` indistinguishable from the
shipped model**. So windowed 3 s vocoding of a global prosody plan is
**perceptually transparent → green-lit**. See Results + Verdict below.

## Results (2026-07-18)

Real ~14.5 s utterance (510 tokens, ALBERT's cap — the frontend itself maxes at
~15 s, so long text must be frontend-chunked at ≤510 tokens regardless; this
tests the finer 3 s generator windowing within one such plan). Geometry: 1,000
ASR frames → 12.5 s vocoded, 5 windows.

| comparison | SNR | note |
| --- | ---: | --- |
| **windowed vs full-length reference** | **2.11 dB** | HEADLINE — the windowing effect |
| pre-trim reference vs shipped | 32.70 dB | the reference is VALID (T2's ~28 dB, better at 15 s) |
| worst boundary seam (±0.1 s) | −4.0 dB | per-seam [2.6, 0.7, 5.8, −4.0] |

**It is a real effect, not a bug** — four checks:

- **Aligned:** per-window cross-correlation peaks at lag 0 (no slicing offset),
  but only corr ≈ 0.75 even at the peak → genuinely decorrelated fine structure.
- **Uniform:** every window's *interior* core SNR is ~1–3 dB (not just the seams),
  so it isn't a boundary/halo artifact — the whole window is affected.
- **Energy preserved:** per-window RMS ratios 0.93–1.32 (no blow-up); the
  difference is phase/texture, not level.
- **Monotonic in window size:** SNR climbs as the window grows toward the full
  utterance — 3 s → **2.1 dB**, 5 s → **2.8 dB**, 6.25 s → **5.2 dB** — the
  signature of the AdaIN *normalization window*, and only reaching global quality
  as the window → the whole utterance.

**Mechanism.** Every `AdaIN1d` normalizes its features over the *time axis*
(per-channel mean/var reduced over ALL frames). A 3 s window computes those stats
over its own 3 s; the full-length reference over the whole 12.5 s. The stats
differ, so each window's vocoder fine structure is perturbed uniformly (energy
preserved, texture decorrelated). Across 48 AdaIN layers this compounds to corr
0.75 / 2 dB. The layer_norm-lowered generator (T6) has the identical problem —
`F.layer_norm` over time is the same per-window normalization.

**Why the obvious fix doesn't cleanly deploy.** Feeding each window the GLOBAL
per-channel AdaIN stats would recover the reference (they're position-independent
scalars). But computing them requires an AdaIN-layer forward over the *whole*
utterance — i.e. running the full generator, the very thing the 3 s ANE cap
forbids. So "global stats" is a 2-pass + model-change path, not a free fix.

## What this means (honest read)

- **You cannot window the vocoder and reproduce a single global synthesis.** The
  "plan globally, vocode in 3 s windows" idea works for *prosody* (F0/durations
  ARE global here — they live in the shared plan, not in AdaIN) but NOT for the
  vocoder's fine texture, which AdaIN ties to near-global context.
- **BUT the global reference is a fiction for deployment** (a 15 s vocode can't
  run on the A14 ANE — the 16,384 limit is the whole reason we window). The
  deployed 3 s bucket *already* normalizes AdaIN over 3 s. So the operative
  question is NOT "does windowed match a global vocode" (it can't) but "does
  per-3 s-window vocoding, with global prosody, **sound good** — natural, and
  continuous across the 2.5 s seams?" The per-window RMS spread (up to 1.32×)
  flags possible level "pumping" at seams as the concrete thing to listen for.
- **SNR is the wrong final judge** (T2's lesson: run-to-run is 20 dB and sounds
  identical). 2 dB is below that floor, so it's *likely* audible — but the
  prosody is preserved and only texture differs, so it may still be acceptable.
  **The ear decides.** WAVs: `Scratchpad/windowed_vocode_{windowed,reference,original}.wav`.

## Verdict — PASSED (owner ear-check, 2026-07-18)

The owner A/B'd `windowed` vs `original` (the shipped full-length model — a
*stricter* anchor than the `windowed` vs `reference` the acceptance asked for) and
**could not tell them apart.** So the 2.11 dB SNR was measuring against an
undeployable global vocode; the ear confirms the per-window AdaIN texture shift is
inaudible, and the indistinguishable A/B also subsumes the seam check (a pumping
−4.0 dB seam at ~10 s would have been audible). **Windowed 3 s vocoding of a global
prosody plan is perceptually transparent — the "plan globally, vocode in windows"
architecture is confirmed with the current split generator** (no bigger bucket, no
FluidAudio extension, no GPU handoff).

**Caveat:** this listen was fp32 on the Mac. It closes the *windowing-algorithm*
question, NOT fp16-on-A14 rendering — the on-device listen (owed regardless) covers
both at once and is folded into the T10 device gate below.

## Next (green-lit path → Plan T9–T12)

1. **T9** — wire plan-globally / vocode-in-3 s-windows into the Swift executor,
   Mac-verified against this probe's `wav_W`.
2. **T10** — long-form device bench + retrievable WAV so the owner hears real
   fp16 A14 output (the on-device counterpart to this fp32 listen) and reads RTF.
3. **T11** — the hard power number (soak / `powermetrics --samplers ane`) — the
   spike's actual goal, still owed.
4. **T12** — production wiring behind the runtime `KokoroModelProvider`.

Only if the on-device fp16 audio surprises us: crossfade the seams (cheap; the
−4.0 dB seam at ~10 s is the candidate) → else extend the ANE bucket (FluidAudio-
style; the monotonic 3 s→2.1 / 5 s→2.8 / 6.25 s→5.2 dB sweep quantifies the payoff)
or the GPU-foreground / pre-buffer handoff → global-AdaIN stats is the last resort
(2-pass + model change).

---

## Original design spec (below) — as executed

## The question this answers (and why it decides the product)

FreeReader reads long-form text in the **background**, where iOS bans Metal, so
the only power-efficient path is the **A14 ANE** — and the ANE generator is
capped at a **3 s** bucket (the 16,384-element axis limit; see Ground Truth).
Long sentences (~15–30 s of speech) therefore can't be vocoded in one ANE pass.

The insight this experiment tests (see
[background-tts-direction-2026-07-18.md](background-tts-direction-2026-07-18.md)):
**prosody is *planned* by the stages that are NOT 3 s-limited.** Duration,
F0Ntrain, and decoder-pre run over the whole sentence and produce a globally
coherent F0/energy contour and `x_pre` — none of those hit the axis limit. Only
the **generator (vocoder)** is 3 s-capped. So the proposed background architecture
is:

> **Plan prosody over the whole sentence once** (CPU/ANE, already works) →
> **vocode that plan in 3 s ANE windows** (with halos, per-window AdaIN stats) →
> concat.

The vocoder is mostly a *local* operation, so windowing only the rendering should
be far less lossy than windowing the planning. **The one unknown is the AdaIN
statistics:** each window normalizes over its own 3 s of activations instead of
the whole sentence. T2 already measured a *single* AdaIN window-shift at ~28 dB
SNR and judged it indistinguishable
([ane-pretrim-equivalence-2026-07-17.md](ane-pretrim-equivalence-2026-07-17.md));
windowing stacks one such shift per window. This experiment measures whether the
stack is still inaudible.

- **Pass** (windowed ≈ full-length, and owner ear-check says indistinguishable):
  the background-safe ANE-3 s path CAN deliver sentence-level prosody. Green-light
  the plan-globally / vocode-in-windows design.
- **Fail** (audible boundary artifacts or prosody choppiness): windowing is
  insufficient; the fallback is extending the ANE bucket past 3 s
  (FluidAudio-style multi-graph split — see the direction note) or the
  GPU-foreground / pre-buffer handoff.

## Design (pure PyTorch fp32, NO Core ML export)

Matches the plan's "do not build any Core ML export for this" — this isolates the
**windowing algorithm**, not fp16/ANE effects (those are separate, already-known
questions). One script, `scripts/probe_windowed_vocode.py`, mirroring the
construction discipline of `scripts/probe_har_pretrim_adain_equivalence.py` and
`scripts/verify_decoder_har_ane.py`.

### 1. Global prosody plan (shared by both sides)

Build ONE long real utterance (~12–15 s of speech; a couple of real sentences,
long enough that the F0/ASR axes fill — reuse the real-input path:
`_real_vocoder_inputs` + `build_decoder_har_post_inputs_np` from the verify/probe
scripts, real misaki phonemes / BERT / duration / F0Ntrain / `af_heart`, fixed
seed). Produce at the utterance's **natural full length**:

- `x_pre` `(1, 512, F)` — F ≈ 1,200 ASR frames for 15 s (decoder-pre output).
- `ref_s` `(1, 256)`.
- `har` `(1, 22, H_trim)` at the **trimmed body length** H_trim ≈ 72,001 for 15 s
  (the pre-trim discipline: build/trim har to body length, as `_build_real_inputs`
  does at 3 s, then generalize to the full utterance).

**Draw `har` ONCE and share it across both sides** — the load-bearing T2/T4 trap;
regenerating it re-draws the hn-nsf RNG and collapses any comparison to ~20 dB.

### 2. Reference (R) — full-length single-pass vocode

Run the pre-trim generator over the FULL plan in one pass (fp32, no axis limit):

```
spec, phase = GeneratorFromHarANE(gen)(x_pre, ref_s, har)      # full length
waveform_R  = gen.stft.inverse(spec, phase)[:SAMPLES]           # AdaIN stats over the whole sentence
```

Equivalently `_forward_pretrim(gen, x_pre, ref_s, har)`. This is the gold: one
globally-normalized vocode of the global plan.

### 3. Windowed (W) — 3 s windows, halos, per-window AdaIN

**Deployable window geometry (this is the key refinement of the plan's loose
"3 s window + 0.25 s halo").** The 3 s ANE package is exactly 240 ASR frames /
14,401 body frames / 72,000 samples. A 3 s window PLUS halos would exceed both the
package and the 16,384 axis limit (3.5 s → body 16,801 > 16,384). So the halo must
live **inside** the 3 s budget:

| quantity | ASR frames | samples | body/har frames | seconds |
| --- | ---: | ---: | ---: | ---: |
| window (= the 3 s package) | 240 | 72,000 | 14,401 | 3.0 |
| halo, each side | 20 | 6,000 | 1,200 | 0.25 |
| **new content / stride** | **200** | **60,000** | **12,000** | **2.5** |

Conversions (from Ground Truth 240↔14,401↔72,000): **1 ASR frame = 300 samples =
60 body/har frames = 12.5 ms.**

For each window k (start = k·200 ASR frames, minus a 20-frame left halo except the
first window; plus 220 frames, i.e. +20-frame right halo except the last):

```
x_pre_win = x_pre[:, :, win_start : win_start+240]              # 240 ASR frames
har_win   = har[:,   :, win_start*60 : win_start*60 + 14401]    # aligned body slice
spec, phase = GeneratorFromHarANE(gen)(x_pre_win, ref_s, har_win)   # AdaIN over THIS window only
wav_win   = gen.stft.inverse(spec, phase)                       # 72,000 samples
core      = wav_win[left_halo_samples : 72000 - right_halo_samples]  # drop the halo(s)
```

Hard-concat the cores in order → `waveform_W`, same length as `waveform_R`.
(First window: no left halo, keep from 0. Last window: no right halo, keep to end.
Pad/trim the final window's plan to 240 frames if the utterance doesn't divide
evenly, and account for it in the concat.)

The halo gives each window the conv receptive-field + AdaIN context near its edges
that it would otherwise be missing; discarding the halo output means the concat is
built only from well-conditioned core regions.

### 4. Metrics + deliverables

- **SNR(R, W)** and **max-abs(R, W)** over the aligned waveforms. Expect the T2
  ballpark (~28 dB) or somewhat lower (more windows = more stacked shifts). SNR
  alone is NOT the verdict — it is a screen; the ear is the gate.
- **Boundary zoom:** SNR/max-abs in a ±0.1 s region around each concat seam, to
  catch discontinuities the global SNR would average away. This is the number that
  actually predicts audible clicks.
- **WAVs (24 kHz mono, one shared gain, never per-file peak-normalized — same as
  T2):** `Scratchpad/windowed_vocode_reference.wav` (R),
  `..._windowed.wav` (W), and `..._original.wav` (the shipped full-length model as
  an ear anchor / gold).
- **Owner ear-check (the deciding gate):** run `audio-judge` (Gemini) or an A/B
  listen: R vs W, and original vs W. Verdict: **indistinguishable / audible**.
  Follow T2's protocol — A/B against the model's own run-to-run noise floor first.

### 5. Acceptance / decision

- **Indistinguishable + boundary SNR clean** → windowed vocoding is viable; the
  3 s ANE background path delivers sentence-level prosody. Next: wire
  plan-globally / vocode-in-windows into the executor (a follow-on task) and
  re-validate end to end on the A14.
- **Audible** → windowing is insufficient. Try, in order: (a) a halo-size sweep
  (0.1 / 0.25 / 0.5 s) — cheap, may fix marginal boundaries; (b) output
  overlap-add / equal-power crossfade instead of hard-concat; (c) if still audible,
  the AdaIN per-window normalization itself is the problem → extend the ANE bucket
  past 3 s (FluidAudio-style split) or fall back to GPU-foreground synthesis.

## Traps to respect (from T2/T4/T5)

- **One `har` draw, shared.** Never regenerate between R and W.
- **fp32, pure PyTorch.** This is an algorithm test; fp16/ANE-miscompute is a
  separate, already-characterized axis (T4). Do not conflate.
- **2x internal geometry.** The 3 s bucket traces at 2x internal geometry
  (`GEOMETRY_SEC = 6` in the probe helper means "the 3 s bucket's doubled
  geometry"). Building a long utterance's real inputs must use the correct `sec` /
  `asr_len` / `har_t` for the *actual* utterance length — verify shapes against
  the 240↔14,401↔72,000 ratios above before trusting any number.
- **Halo lives inside the 3 s budget** (2.5 s content + 2×0.25 s), because 3 s +
  halo exceeds the ANE axis limit. This is what makes the experiment
  deployment-faithful rather than aspirational.

## Why this is worth running before any executor work

If it passes, the background long-form story is solved with the generator we
already have (no new export, no bigger ANE bucket) — the executor change is
"run stages 1–7 whole-sentence, loop stage 8/9 over windows." If it fails, we
learn that *before* building any of that, and pivot to extending the bucket or the
GPU handoff. Either way it's one PyTorch script and an ear-check — the cheapest
possible way to de-risk the whole background-reading direction.

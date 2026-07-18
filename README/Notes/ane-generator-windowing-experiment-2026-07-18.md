# Windowed vocoding of a global prosody plan — experiment design (T8)

Collected: 2026-07-18. Task **T8** of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md). Refines and
supersedes the loosely-scoped "AdaIN windowed-stats experiment" in that plan's
Tasks list, using the geometry we now know is actually ANE-deployable.

**Status: DESIGNED, NOT YET RUN.** This note is the spec; a follow-on session
builds `scripts/probe_windowed_vocode.py` and fills in the numbers + verdict.

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

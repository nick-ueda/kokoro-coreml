# har pre-trim equivalence — the AdaIN statistics shift, quantified

Collected: 2026-07-17. Task T2 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md).

## Verdict

**Indistinguishable** — and the pre-trimmed graph is arguably the *more*
correct of the two. T2's gate on T4 is met.

Caveat stated plainly: the deciding A/B **by ear is owner-run** (WAVs below).
This machine has no `llm-workflows` checkout and no `GEMINI_API_KEY`, so the
**audio-judge** listening gate could not run here. The verdict rests on the
measurements below, of which the run-to-run control is the load-bearing one,
and it is falsifiable by the owner in about a minute with the control clip.

## What was measured

Pure PyTorch fp32 on the Mac, no Core ML, via
`scripts/probe_har_pretrim_adain_equivalence.py`
(`uv run python scripts/probe_har_pretrim_adain_equivalence.py`).

Two variants of the vocoder tail, at the 3 s bucket's Ground Truth geometry
(240 ASR frames, 480 F0 frames, 28,801 natural har frames, 72,000 samples):

- **original** — hand-copy of `GeneratorFromHar.forward`
  (`export_synth/wrappers.py`), full 28,801-frame har into the noise branch,
  `x_source` trimmed AFTER `noise_res[i]`. Asserted bit-identical to the real
  module in-script, so the numbers describe the graph that actually ships.
- **pre-trim** — har at 14,401, `x_source` crop/pad moved BEFORE
  `noise_res[i]`; every tensor <= 14,401, i.e. ANE-admissible on the A14.
  This is the graph T4 would export.

Inputs are real end to end, which matters because a statistics shift is only
meaningful on realistic activations: real misaki phonemes, real BERT/duration/
`F0Ntrain` (a genuinely voiced F0 curve, range -2.4 to 313.8 Hz — **not** the
all-zero curve every export-time trace in this repo uses), real `af_heart`
embedding, real `x_pre`/`har` from `build_decoder_har_post_inputs_np`. The
bench sentence overfills 480 F0 frames, so the 3 s crop is speech throughout
with no zero padding (asserted). Both variants receive the same `har` tensor,
so the hn-nsf RNG cannot contribute to the difference.

## Numbers

| Comparison | SNR | max-abs |
| --- | ---: | ---: |
| **original vs pre-trim (the headline)** | **28.06 dB** | **0.032** |
| — attributable to the AdaIN window shift | 28.06 dB | 0.033 |
| — attributable to the shorter har input (edge effect) | 79.11 dB | 0.0001 |
| *control:* shipped graph vs **itself**, fresh noise draw | *20.46 dB* | *0.085* |
| *defect:* original, real vs zeroed discarded har tail | *12.42 dB* | *0.214* |

Waveform peak 0.352 (original) vs 0.357 (pre-trim); RMS 0.048579 vs 0.048596.
`scripts/audio_quality_probe.py` rates original `reference_pass` and pre-trim
`needs_listening`, both with **no reject reasons**, on near-identical metrics
(RMS 4233.1 vs 4234.6, spectral centroid 907.5 vs 908.9 Hz, ZCR 12.92% vs
12.95%, voiced-band energy ratio 0.9684 vs 0.9683).

## Why 28 dB is a pass, not a worry

An SNR in dB means nothing perceptual by itself, so the script measures the
scale that decides it. **`har` is stochastic**: `SourceModuleHnNSF`
(`kokoro/istftnet.py`) draws a random initial sine phase (`torch.rand`, ~line
269) and additive Gaussian noise (`randn_like`, ~line 339) on every call. The
shipped, unmodified graph therefore renders the same sentence differently every
single time — and those renders sit **20.46 dB** apart.

The pre-trim change moves the waveform **28.06 dB**, i.e. **7.6 dB less than
the model's own render-to-render randomness**. It is strictly inside a
stochastic envelope production has always shipped. For pre-trim to be audible,
two consecutive renders of today's unmodified model would have to be *more*
audibly different — which would already be a known problem.

The attribution also confirms the plan's predicted mechanism rather than a bug:
essentially 100% of the difference is the AdaIN normalization window (2,400 /
14,401 instead of 4,800 / 28,801), and the trimmed-har edge effect is
negligible at 79 dB. (That edge effect is real but tiny: `noise_convs[0]`
(k=12, s=6, pad=3) still emits the required 2,400 frames from a 14,401-frame
har, but its last window reaches har indices 14,401-14,402, so exactly one
frame in 2,400 sees zero padding where the original saw signal.)

## The pre-trim graph is better-defined than the original

The strongest result is the one the plan only hypothesized ("plausibly it is
*more* correct"). The noise branch's output is trimmed away past 14,401, so
har frames 14,401..28,800 **cannot legitimately affect the waveform** — they
reach it only by contaminating AdaIN's normalization window.

- **original**: zeroing that discarded tail changes the output by **12.42 dB
  SNR, max-abs 0.214 — ~60% of waveform peak.** The shipped graph's audible
  output depends heavily on har content it throws away.
- **pre-trim**: **bit-identical** across the same change (asserted in-script).
  It never reads the tail, so it is invariant by construction.

Pre-trim does not degrade the graph; it removes a leak.

## WAVs — for the owner's A/B

24 kHz mono 16-bit, all three written with one shared gain (never per-file
peak-normalized, which would hide level differences and invalidate the A/B):

- `Scratchpad/t2_har_original.wav` — shipped graph.
- `Scratchpad/t2_har_pretrim.wav` — the T4 candidate.
- `Scratchpad/t2_har_original_noise_redraw.wav` — **positive control**: the
  shipped graph, unmodified, on a fresh noise draw.

Suggested protocol: A/B original vs the **control** first. That is the
difference production already ships on every render, and it measures *larger*
(20.46 dB) than the pre-trim difference (28.06 dB). If the control is
indistinguishable, pre-trim must be too; if the control is audible, the
concern is the model's inherent stochasticity, not this change.

## Flagged for the owner — geometry, NOT fixed here (out of T2's remit)

> **Correction (2026-07-17, T5):** the x_pre row in the table below is wrong —
> it conflates `frame_count` (decoder-pre's asr INPUT axis, correctly 120 at
> native geometry) with x_pre's OUTPUT axis, which decoder-pre's internal 2x
> upsample always fills completely (240/240 real). Only `har` is half-filled,
> and only because the baseline package's 28,801 axis is 2x the natural frame
> count. See `ane-generator-bench-plumbing-2026-07-17.md` for the evidence.

Verified in code while building real inputs; the fix is a separate decision.
`build_decoder_har_post_inputs_np` derives `full_f0_len` from `bucket_samples`
(`sec * 24000 / 300`), **not** from the 2x internal geometry the decoder-har
export traces (`export_synth/convert.py`: `geometry_samples = bucket_samples * 2`).
So for the 3 s package the runtime fills only **half of each input axis** with
signal and zero-pads the rest:

| | package axis | real content at runtime (sec=3) |
| --- | ---: | ---: |
| `x_pre` | 240 frames | 120 |
| `har` | 28,801 frames | 14,401 |

This is exactly what that function's `warn_geometry` print reports
(`frame_count 120 != Core ML x_pre length 240`), and it is why T2 had to build
inputs at the export geometry to test the real thing (the script passes
`sec=6` to that helper to mean "the 3 s bucket's 2x internal geometry", with
all four resulting shapes asserted against Ground Truth). Combined with the
12.42 dB tail sensitivity above, it means today's runtime is already feeding
the original graph a har whose AdaIN window is half zeros. Worth its own
investigation; T2 did not chase it.

## Inheritance for T4 — read this before writing the parity check

**T4's stated acceptance gate will fail as written, for a non-bug reason.**
The plan says: "spec/phase → T3's golden Python iSTFT → SNR vs the T1 baseline
package's waveform >= 40 dB (fp16 tolerance)". But the T1 baseline package is
the **original** graph, and T4 exports the **pre-trim** graph — they differ by
~28 dB *by design*, per this note. A 40 dB gate against that reference is
unreachable, and no amount of fp16 debugging will close it.

T4 should instead gate against a PyTorch **pre-trim** reference — i.e.
`_forward_pretrim` in `scripts/probe_har_pretrim_adain_equivalence.py`, run on
the same inputs — where 40 dB is a fair fp16 tolerance. Two further traps:

- **Feed both sides the identical `har` tensor.** Regenerating it re-draws the
  hn-nsf RNG and the comparison collapses to ~20 dB regardless of correctness.
- **Do not reuse the plan's ~20 dB "that's a bug" heuristic for anything
  compared across noise draws** — this model's own run-to-run SNR is 20.46 dB,
  so that threshold sits at its noise floor. It remains a fine heuristic for
  same-input comparisons like T4's.

## Environment note

misaki's English G2P needs a spaCy model that the venv lacked; the probe
script cannot phonemize without it. Installed into the local venv only
(`uv.lock` untouched):

```bash
uv pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

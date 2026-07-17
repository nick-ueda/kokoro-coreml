#!/usr/bin/env python3
"""Quantify the har pre-trim AdaIN statistics shift in pure PyTorch (fp32).

README/Plans/ane-generator-a14-v1.md task T2. NO Core ML here — this is the
gate that decides whether T4 may build an ANE export on the pre-trimmed
graph, so it must be measured on the reference PyTorch numerics, not on
anything already distorted by fp16 or by a converter.

WHAT IS BEING TESTED
--------------------
``GeneratorFromHar`` (export_synth/wrappers.py) feeds the FULL har tensor to
both noise branches and trims the result afterwards::

    x_source = gen.noise_convs[i](har)     # i=0: 28,801 -> 4,800
    x_source = gen.noise_res[i](x_source, s)   # AdaIN stats over 4,800
    ...
    x_source = x_source[:, :, :tx]         # trim to 2,400 — AFTER the AdaIN

``AdaIN1d`` (kokoro/istftnet.py) normalizes over the CURRENT tensor's time
axis, so those mean/var are computed over a region that is twice as long as
the region actually kept. The har input (1, 22, 28,801) and the noise-branch
intermediates at har length are the ONLY tensors in the 3 s generator that
exceed the A14 ANE's 16,384-elements-per-axis limit; the body maxes out at
14,401. Pre-trimming har to 14,401 removes the violation, but it also moves
the AdaIN statistics onto the kept region only (2,400 / 14,401 instead of
4,800 / 28,801). That is NOT bit-equivalent to the shipped graph. This script
measures how far apart the two waveforms actually land.

Ground truth for the 3 s bucket (README/Plans/ane-generator-a14-v1.md, do not
re-derive): 240 ASR frames, 480 F0 frames, 28,801 natural har frames, 14,401
iSTFT frames, 72,000 output samples.

VARIANTS
--------
- ``_forward_original``: hand-copy of ``GeneratorFromHar.forward``, har at
  28,801. Cross-checked below against the real module for bit equality, so
  this stays honest even if the shipped wrapper changes.
- ``_forward_pretrim``: har at 14,401, and the x_source crop/pad moved BEFORE
  ``noise_res[i]``. At this har length both noise convs already emit exactly
  the kept length (2,400 / 14,401), so the crop is a no-op and the reorder is
  only there to make the variant correct for any har length.

Two things change at once between those, so ``main`` also runs
``_forward_pretrim`` on the FULL har ("reorder-only"): its noise convs see
every har sample the original saw, but AdaIN still normalizes over the kept
region. That splits the total difference into the part caused by the AdaIN
window (original vs reorder-only) and the part caused by the shorter har
input (reorder-only vs pre-trim). The latter is a pure edge effect: with
har trimmed to 14,401, ``noise_convs[0]`` (k=12, s=6, pad=3) still emits the
required 2,400 frames, but its LAST window reaches har indices 14,401-14,402,
which only exist in the untrimmed tensor — so exactly one of 2,400 frames
sees zero padding where the original saw signal.

INPUTS
------
Maximally real, which is the whole point — a statistics shift must be
measured on realistic activations, not on noise:

- real phonemes (misaki G2P) for a bench sentence long enough to fill 3 s;
- real BERT / duration predictor / F0Ntrain -> real asr, F0 and N curves
  (F0 is a genuine voiced curve, not the all-zero curve every export-time
  trace in this repo uses);
- real ``af_heart`` voice embedding — an unconstrained ``randn`` ref_s drives
  conv_post to +-90 and ``spec = exp(x)`` to ~1e22 (see the detour recorded
  in scripts/dump_istft_golden.py), which would make any SNR meaningless;
- real x_pre and har straight out of ``build_decoder_har_post_inputs_np``,
  the same geometry function the runtime decoder-har path uses.

Both variants receive the SAME har tensor (B gets a slice of A's), so the
hn-nsf noise RNG cannot contribute to the measured difference.

IS THE MEASURED SNR BIG OR SMALL?
---------------------------------
An SNR in dB means nothing perceptual on its own, so ``main`` also measures
the reference scale that matters: ``har`` is STOCHASTIC. ``SourceModuleHnNSF``
(kokoro/istftnet.py) draws a random initial sine phase (``torch.rand``, ~line
269) and additive Gaussian noise (``randn_like``, ~line 339) on every call, so
the SHIPPED graph already produces a different waveform every time it
synthesizes the same sentence, and nobody calls those renders "different
audio". Re-running the unmodified original on a fresh noise draw (identical
f0/asr/x_pre — asserted) gives the run-to-run SNR that production already
lives with. If the pre-trim difference is far SMALLER than that, it sits
inside the model's own stochastic envelope and cannot be a ship blocker.
The owner's A/B by ear is still the deciding vote (plan T2).

Run from the repo root: ``uv run python scripts/probe_har_pretrim_adain_equivalence.py``
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402

from kokoro.pipeline import KPipeline, voice_embedding_for_phoneme_string  # noqa: E402
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np  # noqa: E402
from export_synth.wrappers import CoreMLExportConstants, GeneratorFromHar  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel  # noqa: E402

# --- Geometry, fixed by the trained checkpoint + the 3 s bucket contract ---
# (README/Plans/ane-generator-a14-v1.md "Ground truth"). Asserted, never derived.
BUCKET_SAMPLES = CoreMLExportConstants.audio_samples_for_seconds(3)  # 72,000
ASR_FRAMES = 240
FULL_F0_LEN = 480
NATURAL_HAR_FRAMES = 28_801  # what the shipped 3 s package's har input axis is
TRIMMED_HAR_FRAMES = 14_401  # the ANE-admissible length T4 would export
SAMPLE_RATE = CoreMLExportConstants.SAMPLE_RATE  # 24,000

# ``build_decoder_har_post_inputs_np`` derives full_f0_len = sec * 24,000 / 300.
# The 3 s decoder-har package traces at 2x internal geometry (export_synth/
# convert.py: `geometry_samples = bucket_samples * 2`, because the iSTFT tail
# emits half as many samples as the F0/har geometry nominally covers), so the
# f0 length that produces the 3 s package's shapes is the one for sec=6:
# 144,000 internal samples -> 480 F0 -> 240 ASR -> 28,801 har. Passing 6 here
# means "the 3 s bucket's internal geometry", NOT "six seconds of audio" —
# the output is 72,000 samples = 3 s. Asserted against the constants above.
GEOMETRY_SEC = 6

# Long enough that the real utterance overfills 480 F0 frames, so the 3 s
# crop below is real speech end to end with no zero padding (asserted).
BENCH_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning light "
    "spreads slowly across the quiet valley below the old stone bridge."
)
VOICE = "af_heart"
SEED = 0

WAV_DIR = _ROOT / "Scratchpad"


def _real_vocoder_inputs(kmodel: torch.nn.Module) -> dict:
    """Return real ``asr`` / ``f0_curve`` / ``n`` / ``ref_s`` for ``BENCH_TEXT``.

    Mirrors ``HybridTTSPipeline.extract_vocoder_inputs``
    (kokoro/coreml_pipeline.py) but without constructing that pipeline, which
    would try to load Core ML packages this task must not touch. Requires the
    spaCy model misaki's English G2P downloads on first use
    (``uv pip install en_core_web_sm@<wheel url>`` if missing).
    """
    g2p_pipeline = KPipeline(lang_code="a", model=False)
    phonemes = None
    for _, ps, _ in g2p_pipeline(BENCH_TEXT, voice=None):
        phonemes = ps
        break
    if not phonemes:
        raise SystemExit("G2P produced no phonemes for BENCH_TEXT")
    print(f"phonemes: {phonemes}")

    voice_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{VOICE}.pt")
    voice_pack = torch.load(voice_path, weights_only=True)
    ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes).float()

    token_ids = [i for i in (kmodel.vocab.get(p) for p in phonemes) if i is not None]
    input_ids = torch.LongTensor([[0, *token_ids, 0]])

    with torch.no_grad():
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
        text_mask = (
            torch.arange(input_lengths.max())
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, CoreMLExportConstants.VOICE_STYLE_DIM :]

        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lstm_out, _ = kmodel.predictor.lstm(d)
        duration = kmodel.predictor.duration_proj(lstm_out)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_curve, n_curve = kmodel.predictor.F0Ntrain(en, s)
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

    print(
        f"real utterance: {int(asr.shape[-1])} ASR frames / {int(f0_curve.shape[-1])} F0 frames "
        f"(need {ASR_FRAMES} / {FULL_F0_LEN}), F0 range "
        f"({f0_curve.min().item():.1f}, {f0_curve.max().item():.1f}) Hz"
    )
    if int(f0_curve.shape[-1]) < FULL_F0_LEN or int(asr.shape[-1]) < ASR_FRAMES:
        raise SystemExit(
            "BENCH_TEXT is too short to fill the 3 s geometry; the tail would be "
            "zero-padded and the har statistics under test would be measured on zeros"
        )

    return {
        "asr": asr.numpy().astype(np.float32),
        "f0_curve": f0_curve.numpy().astype(np.float32),
        "n": n_curve.numpy().astype(np.float32),
        "ref_s": ref_s.numpy().astype(np.float32),
    }


def _forward_original(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har: torch.Tensor
) -> tuple[torch.Tensor, list[dict]]:
    """Hand-copy of ``GeneratorFromHar.forward`` (export_synth/wrappers.py), the
    shipped 3 s graph: full-length har into the noise branch, trim AFTER
    ``noise_res``. Bit-equality against the real module is asserted in ``main``.

    Also returns, per upsample stage, the x_source the AdaIN saw and the kept
    (trimmed) x_source that actually reaches the residual add — the two places
    the pre-trim variant can diverge.
    """
    s = ref_s[:, : CoreMLExportConstants.VOICE_BASELINE_DIM]
    x = x_pre
    stages: list[dict] = []
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        adain_input_len = int(x_source.shape[-1])
        x_source = gen.noise_res[i](x_source, s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        tx = x.size(2)
        ts = x_source.size(2)
        if ts < tx:
            x_source = F.pad(x_source, (0, tx - ts))
        elif ts > tx:
            x_source = x_source[:, :, :tx]
        stages.append({"adain_input_len": adain_input_len, "kept_x_source": x_source})
        x = x + x_source
        xs = None
        for j in range(gen.num_kernels):
            if xs is None:
                xs = gen.resblocks[i * gen.num_kernels + j](x, s)
            else:
                xs = xs + gen.resblocks[i * gen.num_kernels + j](x, s)
        x = xs / gen.num_kernels
    x = F.leaky_relu(x)
    x = gen.conv_post(x)
    spec = torch.exp(x[:, : gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, gen.post_n_fft // 2 + 1 :, :])
    return gen.stft.inverse(spec, phase), stages


def _forward_pretrim(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har: torch.Tensor
) -> tuple[torch.Tensor, list[dict]]:
    """The T4 candidate: identical to ``_forward_original`` except the x_source
    crop/pad happens BEFORE ``noise_res[i]``, so AdaIN normalizes over the kept
    region only. Call it with har pre-trimmed to ``TRIMMED_HAR_FRAMES``; at that
    length the crop is already a no-op and every tensor in the graph is
    <= 14,401 (ANE-admissible on the A14).
    """
    s = ref_s[:, : CoreMLExportConstants.VOICE_BASELINE_DIM]
    x = x_pre
    stages: list[dict] = []
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        tx = x.size(2)
        ts = x_source.size(2)
        if ts < tx:
            x_source = F.pad(x_source, (0, tx - ts))
        elif ts > tx:
            x_source = x_source[:, :, :tx]
        adain_input_len = int(x_source.shape[-1])
        x_source = gen.noise_res[i](x_source, s)
        stages.append({"adain_input_len": adain_input_len, "kept_x_source": x_source})
        x = x + x_source
        xs = None
        for j in range(gen.num_kernels):
            if xs is None:
                xs = gen.resblocks[i * gen.num_kernels + j](x, s)
            else:
                xs = xs + gen.resblocks[i * gen.num_kernels + j](x, s)
        x = xs / gen.num_kernels
    x = F.leaky_relu(x)
    x = gen.conv_post(x)
    spec = torch.exp(x[:, : gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, gen.post_n_fft // 2 + 1 :, :])
    return gen.stft.inverse(spec, phase), stages


def _snr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB treating ``reference`` as signal and
    ``reference - other`` as noise. inf when they are bit-identical."""
    noise_power = float(((reference - other) ** 2).sum())
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(float((reference**2).sum()) / noise_power))


def _write_wav(path: Path, waveform: torch.Tensor, scale: float) -> None:
    """Write mono 16-bit ``waveform`` at ``SAMPLE_RATE``.

    ``scale`` is passed in (not derived per file) so both variants get the
    IDENTICAL gain: per-file peak normalization would hide any level
    difference between them and invalidate the owner's A/B.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(waveform.numpy() * scale, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    print(f"wrote {path} ({pcm.size} samples, {pcm.size / SAMPLE_RATE:.2f} s)")


def main() -> None:
    torch.manual_seed(SEED)
    kmodel = _load_kmodel()
    kmodel.eval()
    dec = kmodel.decoder
    gen = dec.generator

    vi = _real_vocoder_inputs(kmodel)
    x_pre_np, ref_s_np, har_np, _t_f0, frame_count = build_decoder_har_post_inputs_np(
        dec, vi, sec=GEOMETRY_SEC, asr_len=ASR_FRAMES, har_t=NATURAL_HAR_FRAMES, warn_geometry=False
    )
    if frame_count != ASR_FRAMES:
        raise SystemExit(f"geometry drift: frame_count {frame_count} != {ASR_FRAMES}")

    x_pre = torch.from_numpy(x_pre_np)
    ref_s = torch.from_numpy(ref_s_np)
    har_full = torch.from_numpy(har_np)
    if tuple(har_full.shape[-2:]) != (22, NATURAL_HAR_FRAMES) or int(x_pre.shape[-1]) != ASR_FRAMES:
        raise SystemExit(f"geometry drift: x_pre {tuple(x_pre.shape)} har {tuple(har_full.shape)}")
    har_trimmed = har_full[:, :, :TRIMMED_HAR_FRAMES]
    print(f"x_pre {tuple(x_pre.shape)} har {tuple(har_full.shape)} -> trimmed {tuple(har_trimmed.shape)}")

    with torch.no_grad():
        wave_original, stages_original = _forward_original(gen, x_pre, ref_s, har_full)
        wave_pretrim, stages_pretrim = _forward_pretrim(gen, x_pre, ref_s, har_trimmed)
        # Reorder-only: full har into the convs, AdaIN over the kept region.
        # Isolates the statistics shift from the trimmed-har edge effect.
        wave_reorder, _ = _forward_pretrim(gen, x_pre, ref_s, har_full)

        # The hand-copy must BE the shipped graph, or every number below is
        # about some other model (same guard as scripts/dump_istft_golden.py).
        reference = GeneratorFromHar(gen).eval()(x_pre, ref_s, har_full)
        drift = float((wave_original - reference).abs().max())
        if drift > 0.0:
            raise SystemExit(f"_forward_original diverged from GeneratorFromHar.forward: {drift}")

    print("\n--- noise branch, per upsample stage ---")
    for i, (a, b) in enumerate(zip(stages_original, stages_pretrim)):
        kept_a = a["kept_x_source"]
        kept_b = b["kept_x_source"]
        if kept_a.shape != kept_b.shape:
            raise SystemExit(f"stage {i} kept x_source shape mismatch {kept_a.shape} vs {kept_b.shape}")
        print(
            f"stage {i}: AdaIN sees {a['adain_input_len']} frames (original) vs "
            f"{b['adain_input_len']} (pre-trim); kept x_source {tuple(kept_a.shape)} "
            f"max-abs diff {float((kept_a - kept_b).abs().max()):.6f} "
            f"SNR {_snr_db(kept_a, kept_b):.2f} dB"
        )

    wave_original = wave_original.squeeze()[:BUCKET_SAMPLES]
    wave_pretrim = wave_pretrim.squeeze()[:BUCKET_SAMPLES]
    wave_reorder = wave_reorder.squeeze()[:BUCKET_SAMPLES]
    snr = _snr_db(wave_original, wave_pretrim)
    max_abs = float((wave_original - wave_pretrim).abs().max())

    # Reference scale: the shipped graph vs ITSELF on a fresh hn-nsf noise draw.
    # Same f0/asr/x_pre (asserted), only the har RNG differs — exactly what two
    # production renders of one sentence differ by today.
    torch.manual_seed(SEED + 1)
    x_pre_redraw_np, _ref_s2, har_redraw_np, _t2, _fc2 = build_decoder_har_post_inputs_np(
        dec, vi, sec=GEOMETRY_SEC, asr_len=ASR_FRAMES, har_t=NATURAL_HAR_FRAMES, warn_geometry=False
    )
    if not np.array_equal(x_pre_redraw_np, x_pre_np):
        raise SystemExit("noise redraw changed x_pre; it must isolate the har RNG only")
    har_redraw = torch.from_numpy(har_redraw_np)
    if float((har_redraw - har_full).abs().max()) == 0.0:
        raise SystemExit("noise redraw produced identical har; the RNG is not being exercised")
    with torch.no_grad():
        wave_redraw, _ = _forward_original(gen, x_pre, ref_s, har_redraw)
    wave_redraw = wave_redraw.squeeze()[:BUCKET_SAMPLES]

    print("\n--- attribution ---")
    print(
        f"AdaIN window shift alone (original vs reorder-only): "
        f"SNR {_snr_db(wave_original, wave_reorder):.2f} dB, "
        f"max-abs {float((wave_original - wave_reorder).abs().max()):.6f}"
    )
    print(
        f"trimmed-har edge effect alone (reorder-only vs pre-trim): "
        f"SNR {_snr_db(wave_reorder, wave_pretrim):.2f} dB, "
        f"max-abs {float((wave_reorder - wave_pretrim).abs().max()):.6f}"
    )

    print("\n--- final waveform (72,000 samples, 3 s) ---")
    print(f"original peak {float(wave_original.abs().max()):.6f} pre-trim peak {float(wave_pretrim.abs().max()):.6f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"max-abs diff: {max_abs:.6f}")
    print(f"RMS original {float((wave_original**2).mean().sqrt()):.6f} pre-trim {float((wave_pretrim**2).mean().sqrt()):.6f}")

    redraw_snr = _snr_db(wave_original, wave_redraw)
    print("\n--- reference scale: shipped graph vs itself, fresh hn-nsf noise draw ---")
    print(f"run-to-run SNR: {redraw_snr:.2f} dB, max-abs {float((wave_original - wave_redraw).abs().max()):.6f}")
    print(
        f"=> the pre-trim change is {snr - redraw_snr:+.2f} dB relative to the model's "
        "own run-to-run variation "
        f"({'INSIDE' if snr > redraw_snr else 'OUTSIDE'} the existing stochastic envelope)"
    )

    # How much does each graph depend on har frames 14,401.. — the region the
    # noise branch's output is trimmed away to and which therefore CANNOT
    # legitimately affect the waveform, except through AdaIN's normalization
    # window? Zeroing that tail is not hypothetical: the runtime
    # (build_decoder_har_post_inputs_np at sec=3, what decoder_har_post_bucket_impl
    # feeds today) derives full_f0_len from bucket_samples instead of the 2x
    # trace geometry, so it fills only 14,401 of the package's 28,801 har frames
    # and zero-pads the rest — the mismatch its `warn_geometry` print reports.
    # The pre-trim variant never reads the tail, so it must be BIT-IDENTICAL
    # across the two conditions (asserted); any movement is the original's.
    har_zero_tail = har_full.clone()
    har_zero_tail[:, :, TRIMMED_HAR_FRAMES:] = 0.0
    with torch.no_grad():
        wave_original_zero_tail, _ = _forward_original(gen, x_pre, ref_s, har_zero_tail)
        wave_pretrim_zero_tail, _ = _forward_pretrim(
            gen, x_pre, ref_s, har_zero_tail[:, :, :TRIMMED_HAR_FRAMES]
        )
    wave_original_zero_tail = wave_original_zero_tail.squeeze()[:BUCKET_SAMPLES]
    wave_pretrim_zero_tail = wave_pretrim_zero_tail.squeeze()[:BUCKET_SAMPLES]

    pretrim_drift = float((wave_pretrim_zero_tail - wave_pretrim).abs().max())
    if pretrim_drift != 0.0:
        raise SystemExit(f"pre-trim variant read the discarded har tail: max-abs {pretrim_drift}")

    print("\n--- sensitivity to the DISCARDED har tail (frames 14,401..28,800) ---")
    print(
        f"original graph, real tail vs zeroed tail: "
        f"SNR {_snr_db(wave_original, wave_original_zero_tail):.2f} dB, "
        f"max-abs {float((wave_original - wave_original_zero_tail).abs().max()):.6f}"
    )
    print("pre-trim graph, real tail vs zeroed tail: bit-identical (invariant by construction)")

    peak = max(
        float(wave_original.abs().max()),
        float(wave_pretrim.abs().max()),
        float(wave_redraw.abs().max()),
    )
    scale = 0.95 / peak if peak > 0 else 1.0
    _write_wav(WAV_DIR / "t2_har_original.wav", wave_original, scale)
    _write_wav(WAV_DIR / "t2_har_pretrim.wav", wave_pretrim, scale)
    # Positive control for the owner's A/B: the shipped graph, unmodified, on a
    # fresh noise draw. Whatever "different" sounds like between this and
    # t2_har_original is a difference production already ships every render.
    _write_wav(WAV_DIR / "t2_har_original_noise_redraw.wav", wave_redraw, scale)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T8: windowed vocoding of a global prosody plan (pure PyTorch fp32).

README/Plans/ane-generator-a14-v1.md T8; design spec:
README/Notes/ane-generator-windowing-experiment-2026-07-18.md.

THE QUESTION
------------
FreeReader reads long-form text in the iOS background, where Metal is banned, so
the only power path is the A14 ANE — and the ANE generator is capped at a 3 s
bucket (the 16,384-element axis limit). Long sentences can't be vocoded in one
ANE pass. BUT prosody is *planned* by stages that are NOT 3 s-limited (duration /
F0Ntrain / decoder-pre run over the whole sentence). So the proposed background
design is: plan prosody globally, then vocode that plan in 3 s ANE windows.

This script tests whether that windowed vocoding degrades the audio. It builds
ONE long (~15 s) real utterance's global plan, then vocodes it two ways with the
SAME generator and the SAME shared ``har``:

  R (reference): full-length single-pass vocode — AdaIN stats over the whole
                 sentence. The gold this experiment compares against.
  W (windowed):  3 s windows, per-window AdaIN stats, halos discarded, concat —
                 exactly what the A14 ANE would do.

The only variable is windowing. If W ≈ R (and the owner ear-check agrees), the
background-safe 3 s ANE path can deliver sentence-level prosody.

WHY REAL INPUTS, fp32, ONE har DRAW
-----------------------------------
Same discipline as T2/T4 (scripts/probe_har_pretrim_adain_equivalence.py): the
AdaIN statistics shift is only meaningful on realistic activations, and the
hn-nsf har is stochastic — regenerating it collapses any comparison to ~20 dB, so
``har`` is drawn ONCE and sliced. fp32 pure PyTorch isolates the WINDOWING
algorithm from the separate, already-characterized fp16/ANE-miscompute axis (T4).
No Core ML export.

THE ~15 s CAP IS REAL (BERT, not just our bucket)
-------------------------------------------------
ALBERT's context is 512 tokens (≤510 phonemes), so the global prosody plan itself
maxes out at ~15 s per pass — FreeReader must chunk long text at the FRONTEND at
≤510-token / sentence-group boundaries regardless (FluidAudio hit the identical
cap). That is a natural ~15 s chunk (several sentences, split at sentence
boundaries); THIS experiment tests the finer 3 s generator windowing WITHIN one
such plan. We truncate the utterance to 510 tokens and derive the geometry from
the actual frame count.

DEPLOYABLE WINDOW GEOMETRY (the refinement of the plan's loose "3 s + halo")
---------------------------------------------------------------------------
A 3 s window PLUS halos would exceed the 16,384 axis limit (3.5 s → body 16,801),
so the halo lives INSIDE the 3 s budget. Per Ground Truth, 1 ASR frame = 300
samples = 60 body frames = 12.5 ms:

  window = 240 ASR frames  (= the 3 s package: 14,401 body frames, 72,000 samples)
  halo   = 20 ASR frames each side (0.25 s = 6,000 samples)
  core / stride = 200 ASR frames (2.5 s = 60,000 samples)

The generator maps x_pre length T_asr -> output 60·T_asr + 1 frames; iSTFT ->
300·T_asr samples. So window k's x_pre slice [lo, hi] produces output samples
corresponding to full samples [300·lo, 300·hi]; we keep only the 60,000-sample
core and discard the halo output (which is starved of conv/AdaIN context).

Run: ``uv run python scripts/probe_windowed_vocode.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

# Reuse the T2 probe's validated plumbing verbatim — same frontend symbols, same
# har discipline, same WAV writer — so this script only adds the windowing.
import probe_har_pretrim_adain_equivalence as P  # noqa: E402
from export_synth.wrappers import GeneratorFromHarANE  # noqa: E402
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np  # noqa: E402

# --- Length-independent geometry (Ground Truth ratios) ------------------------
# WIN_ASR / MIN_WINDOWS are env-overridable for the window-size sweep that
# characterizes the AdaIN-context effect; the deployable case is the default 240.
WIN_ASR = int(os.environ.get("WIN_ASR", "240"))   # 240 = the 3 s ANE package
HALO_ASR = 20           # 0.25 s each side
CORE_ASR = WIN_ASR - 2 * HALO_ASR    # 200 (= stride)
SAMPLES_PER_ASR = 300   # 1 ASR frame -> 300 output samples
BODY_PER_ASR = 60       # 1 ASR frame -> 60 body/har frames
F0_PER_ASR = 2          # 1 ASR frame -> 2 F0 frames (480 F0 / 240 ASR at 3 s)
SEAM_HALF_WINDOW = 2_400            # +/- 0.1 s around each concat seam
MAX_BERT_TOKENS = 512               # ALBERT context; truncate to 510 phonemes + BOS/EOS
MIN_WINDOWS = int(os.environ.get("MIN_WINDOWS", "4"))   # enough windows for the seam test to mean anything

# A ~15 s+ paragraph. Truncated to 510 tokens (below), so it only needs to be at
# least that long. Natural prosody on purpose — the windowing seams fall
# mid-phrase, which is exactly the stress case.
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning light spreads "
    "slowly across the quiet valley below the old stone bridge. A gentle wind "
    "moves through the tall grass, carrying the scent of rain and distant pine, "
    "and somewhere far away a church bell rings twice before the silence returns. "
    "The river bends around the meadow, catching the pale gold of the early sun, "
    "and the whole countryside seems to hold its breath, waiting for the day to "
    "begin in earnest as the last stars fade from the lightening sky above the "
    "sleeping town, where chimneys wake one by one and the roads lie still and empty."
)


def _long_vocoder_inputs(kmodel: torch.nn.Module, text: str) -> dict:
    """Real ``asr`` / ``f0_curve`` / ``n`` / ``ref_s`` for a LONG utterance.

    Adapted from ``P._real_vocoder_inputs``, which ``break``s after the first G2P
    chunk (KPipeline splits on sentence boundaries, capping it at ~1 sentence).
    Here every chunk's phonemes are joined and TRUNCATED to 510 tokens (ALBERT's
    512-token context), then run through the frontend as one sequence — a
    genuinely long global prosody plan with sentence boundaries planned
    INTERNALLY, the long-form reading case the windowing must survive.
    """
    g2p_pipeline = P.KPipeline(lang_code="a", model=False)
    chunks = [ps for _, ps, _ in g2p_pipeline(text, voice=None) if ps]
    phonemes = " ".join(chunks)
    if not phonemes:
        raise SystemExit("G2P produced no phonemes for LONG_TEXT")

    voice_path = P.hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{P.VOICE}.pt")
    voice_pack = torch.load(voice_path, weights_only=True)
    ref_s = P.voice_embedding_for_phoneme_string(voice_pack, phonemes).float()

    token_ids = [i for i in (kmodel.vocab.get(p) for p in phonemes) if i is not None]
    token_ids = token_ids[: MAX_BERT_TOKENS - 2]   # room for BOS/EOS
    input_ids = torch.LongTensor([[0, *token_ids, 0]])
    print(f"phonemes: {len(chunks)} G2P chunks joined -> {len(token_ids)} tokens (capped at {MAX_BERT_TOKENS - 2})")

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
        s = ref_s[:, P.CoreMLExportConstants.VOICE_STYLE_DIM :]

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
        f"long utterance: {int(asr.shape[-1])} ASR frames / {int(f0_curve.shape[-1])} F0 frames "
        f"(~{int(asr.shape[-1]) / 80:.1f} s), F0 range "
        f"({f0_curve.min().item():.1f}, {f0_curve.max().item():.1f}) Hz"
    )
    return {
        "asr": asr.numpy().astype(np.float32),
        "f0_curve": f0_curve.numpy().astype(np.float32),
        "n": n_curve.numpy().astype(np.float32),
        "ref_s": ref_s.numpy().astype(np.float32),
    }


def _window_vocode(
    wrapper: GeneratorFromHarANE,
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trim: torch.Tensor,
    asr_len: int,
) -> torch.Tensor:
    """Vocode the global plan in 3 s windows; return the concatenated waveform.

    Each window runs the SAME ``GeneratorFromHarANE`` the reference uses, on a
    240-frame x_pre slice + its aligned body-har slice, then keeps only the
    200-frame (60,000-sample) core, discarding the 0.25 s halo output at each end
    (the first window has no left halo; the last has no right halo).
    """
    n_windows = asr_len // CORE_ASR
    pieces: list[torch.Tensor] = []
    for k in range(n_windows):
        core_lo = k * CORE_ASR
        core_hi = core_lo + CORE_ASR
        lo = max(0, core_lo - HALO_ASR)
        hi = min(asr_len, core_hi + HALO_ASR)
        x_win = x_pre[:, :, lo:hi]
        # Body-har slice aligned to the x_pre window: length 60*(hi-lo)+1 == the
        # window's output frame count.
        har_win = har_trim[:, :, BODY_PER_ASR * lo : BODY_PER_ASR * hi + 1]
        with torch.no_grad():
            spec, phase = wrapper(x_win, ref_s, har_win)
            wav = gen.stft.inverse(spec, phase).squeeze()
        loc_lo = SAMPLES_PER_ASR * (core_lo - lo)   # drop left halo (0 for k=0)
        loc_hi = SAMPLES_PER_ASR * (core_hi - lo)   # keep 60,000-sample core
        pieces.append(wav[loc_lo:loc_hi])
    return torch.cat(pieces)[: SAMPLES_PER_ASR * asr_len]


def _boundary_seam_report(
    ref: torch.Tensor, win: torch.Tensor, asr_len: int
) -> tuple[float, list[float]]:
    """SNR in a +/- 0.1 s region around each concat seam; return (worst, per-seam).

    The global SNR averages boundary artifacts away — this is the number that
    actually predicts audible clicks at the window joins.
    """
    n_samples = SAMPLES_PER_ASR * asr_len
    per_seam: list[float] = []
    for k in range(1, asr_len // CORE_ASR):
        seam = k * CORE_ASR * SAMPLES_PER_ASR
        a = max(0, seam - SEAM_HALF_WINDOW)
        b = min(n_samples, seam + SEAM_HALF_WINDOW)
        per_seam.append(P._snr_db(ref[a:b], win[a:b]))
    return (min(per_seam) if per_seam else float("inf")), per_seam


def main() -> None:
    torch.manual_seed(P.SEED)
    kmodel = P._load_kmodel()
    kmodel.eval()
    gen = kmodel.decoder.generator

    # Global prosody plan for ONE long utterance (real frontend, all chunks joined).
    vi = _long_vocoder_inputs(kmodel, LONG_TEXT)
    real_asr = int(vi["asr"].shape[-1])

    # Geometry derived from the actual frame count (see the module docstring):
    # ASR_LEN a clean multiple of the 200-frame stride, never exceeding real
    # content (a zero-padded tail would test windowing over silence).
    asr_len = (real_asr // CORE_ASR) * CORE_ASR
    n_windows = asr_len // CORE_ASR
    if n_windows < MIN_WINDOWS:
        raise SystemExit(
            f"utterance too short: {real_asr} ASR frames -> {n_windows} windows "
            f"(need >= {MIN_WINDOWS}). Lengthen LONG_TEXT."
        )
    sec = asr_len // 40                              # full_f0_len = sec*80 = 2*asr_len; frame_count = sec*40
    full_f0_len = asr_len * F0_PER_ASR
    har_natural = full_f0_len * BODY_PER_ASR + 1     # 120*asr_len + 1
    body_frames = asr_len * BODY_PER_ASR + 1         # 60*asr_len + 1
    n_samples = SAMPLES_PER_ASR * asr_len

    x_pre_np, ref_s_np, har_np, _t_f0, frame_count = build_decoder_har_post_inputs_np(
        kmodel.decoder, vi, sec=sec, asr_len=asr_len, har_t=har_natural, warn_geometry=True
    )
    if frame_count != asr_len:
        raise SystemExit(f"geometry drift: frame_count {frame_count} != {asr_len}")
    if har_np.shape[-1] != har_natural or x_pre_np.shape[-1] != asr_len:
        raise SystemExit(
            f"geometry drift: x_pre {x_pre_np.shape} har {har_np.shape} "
            f"(expected (...,{asr_len}) / (...,{har_natural}))"
        )

    x_pre = torch.from_numpy(x_pre_np)
    ref_s = torch.from_numpy(ref_s_np)
    har_natural_t = torch.from_numpy(har_np)                  # for _forward_original (shipped)
    har_trim = har_natural_t[:, :, :body_frames].contiguous()  # pre-trim body
    print(
        f"global plan: x_pre {tuple(x_pre.shape)} har natural {tuple(har_natural_t.shape)} "
        f"-> body {tuple(har_trim.shape)} -> {n_samples} samples ({n_samples / 24000:.1f} s), "
        f"{n_windows} windows (sec={sec})"
    )

    wrapper = GeneratorFromHarANE(gen).eval()
    with torch.no_grad():
        # R: full-length single-pass pre-trim vocode (global AdaIN stats).
        spec_r, phase_r = wrapper(x_pre, ref_s, har_trim)
        wav_R = gen.stft.inverse(spec_r, phase_r).squeeze()[:n_samples]
        # O: shipped full-length graph (natural har) — ear anchor / gold.
        wav_O, _ = P._forward_original(gen, x_pre, ref_s, har_natural_t)
        wav_O = wav_O.squeeze()[:n_samples]
    # W: windowed pre-trim vocode (per-window AdaIN stats).
    wav_W = _window_vocode(wrapper, gen, x_pre, ref_s, har_trim, asr_len)

    # Per-window core diagnostics: SNR + best cross-correlation lag vs the
    # reference. Uniform low core SNR at lag~0 (energy preserved) = the real
    # per-window AdaIN normalization effect, not a slicing/seam bug.
    for k in range(n_windows):
        a = k * CORE_ASR * SAMPLES_PER_ASR
        b = a + CORE_ASR * SAMPLES_PER_ASR
        w = wav_W[a + 6000:b - 6000]
        best_lag, best_corr = 0, -1e9
        for lag in range(-100, 101):
            r = wav_R[a + 6000 + lag:b - 6000 + lag]
            c = float((w * r).sum() / (w.norm() * r.norm() + 1e-9))
            if c > best_corr:
                best_corr, best_lag = c, lag
        print(f"  window {k}: core_snr={P._snr_db(wav_R[a:b], wav_W[a:b]):5.1f} dB  best_lag={best_lag:+d}  corr={best_corr:.3f}")

    # --- Metrics -------------------------------------------------------------
    def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
        m = min(len(a), len(b))
        return float((a[:m] - b[:m]).abs().max())

    snr_WR = P._snr_db(wav_R, wav_W)      # HEADLINE: windowing effect in isolation
    snr_RO = P._snr_db(wav_O, wav_R)      # pre-trim vs shipped at ~15 s (expect ~T2's 28 dB)
    snr_WO = P._snr_db(wav_O, wav_W)      # total (windowing + pre-trim) vs shipped
    worst_seam, per_seam = _boundary_seam_report(wav_R, wav_W, asr_len)

    print("\n=== T8 windowed-vocode parity (fp32 PyTorch, real long utterance) ===")
    print(f"  windowed vs full-length reference : {snr_WR:6.2f} dB SNR   (max-abs {maxabs(wav_R, wav_W):.4f})   [HEADLINE]")
    print(f"  pre-trim reference vs shipped     : {snr_RO:6.2f} dB SNR   (max-abs {maxabs(wav_O, wav_R):.4f})")
    print(f"  windowed vs shipped (total)       : {snr_WO:6.2f} dB SNR   (max-abs {maxabs(wav_O, wav_W):.4f})")
    print(f"  boundary seams (+/-0.1 s), worst  : {worst_seam:6.2f} dB   per-seam {[round(s, 1) for s in per_seam]}")

    # --- WAVs (one SHARED gain, never per-file peak-normalized — T2 discipline) ---
    peak = max(float(wav_R.abs().max()), float(wav_W.abs().max()), float(wav_O.abs().max()))
    scale = 0.95 / peak if peak > 0 else 1.0
    P.WAV_DIR.mkdir(parents=True, exist_ok=True)
    for name, wav in {
        "windowed_vocode_reference.wav": wav_R,
        "windowed_vocode_windowed.wav": wav_W,
        "windowed_vocode_original.wav": wav_O,
    }.items():
        P._write_wav(P.WAV_DIR / name, wav, scale)
    print(f"\n  WAVs (shared gain {scale:.4f}) -> {P.WAV_DIR}/windowed_vocode_{{reference,windowed,original}}.wav")
    print("  Owner ear-check decides: A/B windowed vs reference (and vs original).")


if __name__ == "__main__":
    main()

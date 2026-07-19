#!/usr/bin/env python3
"""T9: dump windowed-vocode golden fixtures for the Swift executor.

README/Plans/ane-generator-a14-v1.md T9. `scripts/probe_windowed_vocode.py`
(T8) validated the windowing ALGORITHM by ear; this script feeds the Swift
executor's `WindowedGeneratorExecutor.swift` (`windowedVocodePlan` +
`vocodeWindowed`) a golden fixture to prove its NEW contribution — the
window-boundary geometry and the crop/concat/trim that stitches windows back
together — reproduces the validated design exactly, in isolation from the
generator model's own numerics (already gated at ~40-46 dB by
`GeneratorSplitParityTests`/T4/T6/T7; re-litigating that here would just
re-measure fp16-CoreML-vs-fp32-PyTorch noise, not test anything new).

WHY THIS SCRIPT IS NOT JUST `probe_windowed_vocode.main()`
-----------------------------------------------------------
The validated probe runs pure PyTorch, which accepts any window width, so its
first/last windows are naturally SHORTER than the 3 s package's fixed
240-frame span (no halo on the missing side). The exported `.mlpackage`
trunk/body pair cannot accept a short window, and zero-padding would pollute
the AdaIN time-statistics T8's ear-check validated (the entire point of
windowing is real context, never synthetic zeros) — see
`WindowedGeneratorExecutor.swift`'s header for the full rationale. This
script's `_window_vocode_edge_overlap` is the PyTorch mirror of that exact
deployable policy (shift a clipped edge window INWARD to a full 240-frame
span, overlapping the neighbor's territory for CONTEXT only; the KEPT core
region is byte-for-byte the same as the probe's wherever both are defined).
Reuses the probe's frontend plumbing and window constants verbatim — it does
NOT re-derive the window/halo/stride/har-slice math, only the edge policy.

WHAT'S DUMPED (flat little-endian float32, no header, into
swift/Tests/KokoroPipelineTests/Fixtures/windowed_vocode/):

- ``x_pre.f32`` (1, 512, asr_len), ``ref_s.f32`` (1, 256), ``har.f32``
  (1, 22, asr_len*60+1) — the FULL-CHUNK inputs, for an end-to-end
  integration test that runs the real trunk/body `.mlpackage`s (analogous to
  `GeneratorSplitParityTests`, SNR-gated — CoreML fp16 vs this script's fp32
  PyTorch has the same ~40-46 dB floor T4/T6/T7 already established).
- ``window{k}_wav.f32`` — each window's FULL (uncropped) 72,000-sample
  waveform, k = 0..<window_count. Lets the Swift test do its OWN crop (via
  ``windowedVocodePlan``) into these Python-computed waveforms and compare
  the result to ``wav_w.f32`` at tight tolerance — the part of T9 that is
  pure arithmetic (crop indices, concat order, trim), with T3's
  already-proven ``hostISTFTInverse`` and the generator's own numerics held
  fixed, giving a real (not just nominal) shot at matching T3's 1e-4 golden
  bar.
- ``wav_w.f32`` — the final windowed, edge-overlap-policy waveform (length
  ``300*asr_len``).
- ``meta.json`` — ``asr_len``, channel counts, ``window_count``, and each
  window's ``{lo, hi, core_lo, core_hi}`` (ASR frames) so the Swift test can
  assert ``windowedVocodePlan(asrLen:)`` reproduces this exact plan
  independently, not just trust this script's own concat.

Run from the repo root: ``uv run python scripts/dump_windowed_vocode_golden.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

# Reuse T8's validated probe verbatim for the frontend plumbing and the
# window/halo/stride/har-slice constants — this script only adds the
# fixed-shape edge policy the deployed `.mlpackage` requires.
import probe_windowed_vocode as W  # noqa: E402
import probe_har_pretrim_adain_equivalence as P  # noqa: E402
from export_synth.wrappers import GeneratorFromHarANE  # noqa: E402
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np  # noqa: E402

OUT_DIR = _ROOT / "swift" / "Tests" / "KokoroPipelineTests" / "Fixtures" / "windowed_vocode"

# Long enough to floor-truncate (see `probe_windowed_vocode.main`'s own
# `asr_len = (real_asr // CORE_ASR) * CORE_ASR` discipline) to >= 3 windows —
# exercising BOTH edge-overlap cases (first, last) plus at least one true
# interior window (no adjustment needed) — while staying well short of the
# validated probe's own ~15 s / 510-token / 5-window run, so this fixture
# stays a golden-test-sized artifact, not a re-run of T8's experiment.
SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning light "
    "spreads slowly across the quiet valley below the old stone bridge. A "
    "gentle wind moves through the tall grass, carrying the scent of rain "
    "and distant pine, and somewhere far away a church bell rings twice "
    "before the silence returns over the sleeping town."
)
MIN_WINDOWS = 3


def _window_vocode_edge_overlap(
    wrapper: GeneratorFromHarANE,
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trim: torch.Tensor,
    asr_len: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict]]:
    """T9's DEPLOYABLE edge policy — see the module docstring.

    Returns ``(wav_W, per_window_full_waveforms, window_plan)``.
    """
    n_windows = (asr_len + W.CORE_ASR - 1) // W.CORE_ASR
    pieces: list[torch.Tensor] = []
    windows: list[dict] = []
    per_window_wav: list[torch.Tensor] = []
    for k in range(n_windows):
        core_lo = k * W.CORE_ASR
        core_hi = min(asr_len, core_lo + W.CORE_ASR)
        lo = max(0, core_lo - W.HALO_ASR)
        hi = min(asr_len, core_hi + W.HALO_ASR)
        if hi - lo < W.WIN_ASR:
            if lo == 0:
                hi = min(asr_len, W.WIN_ASR)
            elif hi == asr_len:
                lo = max(0, asr_len - W.WIN_ASR)
        x_win = x_pre[:, :, lo:hi]
        har_win = har_trim[:, :, W.BODY_PER_ASR * lo : W.BODY_PER_ASR * hi + 1]
        with torch.no_grad():
            spec, phase = wrapper(x_win, ref_s, har_win)
            wav = gen.stft.inverse(spec, phase).squeeze()
        per_window_wav.append(wav)
        windows.append({"lo": lo, "hi": hi, "core_lo": core_lo, "core_hi": core_hi})
        loc_lo = W.SAMPLES_PER_ASR * (core_lo - lo)
        loc_hi = W.SAMPLES_PER_ASR * (core_hi - lo)
        pieces.append(wav[loc_lo:loc_hi])
    wav_w = torch.cat(pieces)[: W.SAMPLES_PER_ASR * asr_len]
    return wav_w, per_window_wav, windows


def _write_f32(path: Path, tensor: torch.Tensor) -> None:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    path.write_bytes(arr.tobytes())
    print(f"  wrote {path.relative_to(_ROOT)}  shape={tuple(arr.shape)}  bytes={arr.nbytes}")


def main() -> None:
    torch.manual_seed(P.SEED)
    kmodel = P._load_kmodel()
    kmodel.eval()
    gen = kmodel.decoder.generator

    vi = W._long_vocoder_inputs(kmodel, SHORT_TEXT)
    real_asr = int(vi["asr"].shape[-1])
    asr_len = (real_asr // W.CORE_ASR) * W.CORE_ASR
    n_windows = asr_len // W.CORE_ASR
    if n_windows < MIN_WINDOWS:
        raise SystemExit(
            f"SHORT_TEXT too short: {real_asr} ASR frames -> {n_windows} windows "
            f"(need >= {MIN_WINDOWS}). Lengthen SHORT_TEXT."
        )

    sec = asr_len // 40
    full_f0_len = asr_len * W.F0_PER_ASR
    har_natural = full_f0_len * W.BODY_PER_ASR + 1
    body_frames = asr_len * W.BODY_PER_ASR + 1

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
    har_trim = torch.from_numpy(har_np)[:, :, :body_frames].contiguous()

    wrapper = GeneratorFromHarANE(gen).eval()
    wav_w, per_window_wav, windows = _window_vocode_edge_overlap(
        wrapper, gen, x_pre, ref_s, har_trim, asr_len
    )

    print(
        f"golden fixture: x_pre {tuple(x_pre.shape)} har {tuple(har_trim.shape)} "
        f"asr_len={asr_len} windows={len(windows)} wav_W {tuple(wav_w.shape)} "
        f"({wav_w.shape[-1] / 24000:.2f} s)"
    )
    for k, w in enumerate(windows):
        print(f"  window {k}: {w}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_f32(OUT_DIR / "x_pre.f32", x_pre)
    _write_f32(OUT_DIR / "ref_s.f32", ref_s)
    _write_f32(OUT_DIR / "har.f32", har_trim)
    _write_f32(OUT_DIR / "wav_w.f32", wav_w)
    for k, wav in enumerate(per_window_wav):
        _write_f32(OUT_DIR / f"window{k}_wav.f32", wav)

    meta = {
        "asr_len": asr_len,
        "x_pre_channels": int(x_pre.shape[1]),
        "har_channels": int(har_trim.shape[1]),
        "har_body_frames": body_frames,
        "window_count": len(windows),
        "windows": windows,
    }
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {meta_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump golden spec/phase/waveform fixtures for HostISTFT.swift's golden test.

``HostISTFT.swift`` (``swift/Sources/KokoroPipeline/HostISTFT.swift``) is a
from-scratch Swift/vDSP port of ``kokoro/custom_stft.py``
``CustomSTFT.inverse``. This script is the Python half of proving that port
correct: it writes flat float32 binaries that
``swift/Tests/KokoroPipelineTests/HostISTFTTests.swift`` loads and compares
against, via ``Bundle.module``, asserting max-abs error < 1e-4.

Fixtures (written to
``swift/Tests/KokoroPipelineTests/Fixtures/istft_golden/``, raw float32,
little-endian, row-major — no header; shapes are fixed constants mirrored in
the paired Swift test):

- ``synthetic_{spec,phase,waveform}.f32`` — fixed-seed random magnitude/phase
  (frame_count=40), matching ``tests/test_custom_stft.py``'s
  inconsistent-spectrogram case: the inverse must hold for ANY spec/phase
  pair the network could produce, not just round-trips of real audio.
- ``real_3s_{spec,phase,waveform}.f32`` — spec/phase captured from the
  actual ``GeneratorFromHar`` body (T1's checkpoint, 3s-bucket production
  geometry: 240 ASR frames, 480 F0 frames, 14,401 iSTFT frames) immediately
  before its call to ``gen.stft.inverse``. The reference waveform is
  ``gen.stft.inverse(spec, phase)`` itself — pure PyTorch, no Core ML — at
  T1's baseline geometry (``coreml/kokoro_decoder_har_post_3s.mlpackage``,
  no re-export needed; see ``README/Plans/ane-generator-a14-v1.md`` T1/T3).

  Getting a numerically sane real_3s case took one detour worth recording:
  feeding the generator an unconstrained ``ref_s = randn(1, 256)`` (as
  ``export_synth/convert.py``'s own trace/validation inputs do — fine for
  shape-only tracing, never for value-sensitive checks) drives conv_post's
  pre-activation to +-90, so ``spec = exp(x)`` overflows to ~1e22 and a
  meaningless floating-point-cancellation "error" swamps the 1e-4 golden
  tolerance at that scale. Real voice embeddings (``voices/af_heart.pt``,
  one small Hugging Face Hub download, cached locally) are far smaller and
  more structured than raw Gaussian noise — swapping to one real embedding
  drops conv_post's range to a normal +-20ish and the waveform to a normal
  audio range. ``asr`` itself turned out not to matter for this stability
  (bounded random reproduces the same sane range as a real BERT-derived
  ``asr`` would), so this script skips running the duration/BERT stack
  entirely and uses bounded random ``asr`` — simpler, and the golden test's
  job is numeric parity of the iSTFT port, not semantic speech content.

Run from the repo root: ``uv run python scripts/dump_istft_golden.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402

from kokoro.custom_stft import CustomSTFT  # noqa: E402
from kokoro.conv_length import conv1d_output_length_from_module  # noqa: E402
from kokoro.pipeline import voice_embedding_for_phoneme_string  # noqa: E402
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np  # noqa: E402
from export_synth.wrappers import CoreMLExportConstants, GeneratorFromHar  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel  # noqa: E402

FIXTURE_DIR = _ROOT / "swift" / "Tests" / "KokoroPipelineTests" / "Fixtures" / "istft_golden"

# Kokoro generator iSTFT geometry (kokoro/istftnet.py Generator.__init__,
# gen_istft_n_fft=20, gen_istft_hop_size=5). Fixed by the trained checkpoint,
# not a tunable — see README/Plans/ane-generator-a14-v1.md Ground Truth.
N_FFT = 20
HOP = 5
FREQ_BINS = N_FFT // 2 + 1  # 11


def _save(name: str, array: torch.Tensor) -> None:
    """Write ``array`` as a flat little-endian float32 binary under FIXTURE_DIR."""
    path = FIXTURE_DIR / f"{name}.f32"
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.ascontiguousarray(array.detach().cpu().numpy().astype(np.float32))
    flat.tofile(path)
    print(f"wrote {path} shape={tuple(array.shape)} ({flat.nbytes} bytes)")


def dump_synthetic(seed: int = 0, frame_count: int = 40) -> None:
    """Random (non-network) magnitude/phase — mirrors
    ``tests/test_custom_stft.py::test_inverse_matches_torch_istft_on_inconsistent_spectrogram``,
    the closest existing coverage of ``CustomSTFT.inverse`` on arbitrary input."""
    torch.manual_seed(seed)
    magnitude = torch.rand(1, FREQ_BINS, frame_count) + 0.1
    phase = (torch.rand(1, FREQ_BINS, frame_count) * 2 - 1) * np.pi

    stft = CustomSTFT(filter_length=N_FFT, hop_length=HOP, win_length=N_FFT)
    waveform = stft.inverse(magnitude, phase)

    _save("synthetic_spec", magnitude.squeeze(0))
    _save("synthetic_phase", phase.squeeze(0))
    _save("synthetic_waveform", waveform.squeeze(0))
    print(f"synthetic_frame_count={frame_count}")


def _generator_spec_phase(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``GeneratorFromHar.forward`` (export_synth/wrappers.py) up to
    spec/phase, stopping one line before its ``gen.stft.inverse`` call — the
    same split point T4 needs for the ANE package output boundary."""
    s = ref_s[:, : CoreMLExportConstants.VOICE_BASELINE_DIM]
    x = x_pre
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
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
    return spec, phase


def _real_voice_ref_s(bench_text: str = "kokoro drop in sdk voice reference embedding row") -> torch.Tensor:
    """Return a real (1, 256) voice embedding row from the ``af_heart`` voice
    pack — NOT a random vector. ``voices/{name}.pt`` files are trained
    speaker embeddings on HF Hub (``kokoro/pipeline.py`` ``load_voice``);
    ``bench_text``'s length only selects which row (see
    ``voice_embedding_for_phoneme_string``), so it doesn't need to be real
    phonemes for this script's purpose. One small (~1MB) download into the
    local HF cache — never touches ``checkpoints/`` or ``coreml/``."""
    voice_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/af_heart.pt")
    voice_pack = torch.load(voice_path, weights_only=True)
    return voice_embedding_for_phoneme_string(voice_pack, bench_text).float()


def dump_real_3s(seed: int = 0) -> None:
    """Real spec/phase/waveform at the shipped 3s decoder-har-post geometry,
    computed directly in PyTorch from the checkpoint (T1) — no Core ML
    involved. Ground truth (README/Plans/ane-generator-a14-v1.md): 240 ASR
    frames, 480 F0 frames, 14,401 iSTFT frames, 72,000-sample waveform.

    x_pre/har come from ``kokoro.synthesis_backends.build_decoder_har_post_inputs_np``
    — the same single-source-of-truth geometry function
    ``decoder_har_post_bucket_impl`` uses at runtime — fed a real voice
    embedding (see ``_real_voice_ref_s``; this is what keeps conv_post's
    output in a sane range, see module docstring), zero F0/N curves (every
    export-time trace in this codebase uses zero F0/N; an all-unvoiced
    curve, not a stability workaround), and bounded random ``asr`` (real
    ``asr`` from a full BERT/duration-predictor run gives the same sane
    output range, so this script skips that stack — not load-bearing here).
    """
    torch.manual_seed(seed)
    kmodel = _load_kmodel()
    kmodel.eval()
    dec = kmodel.decoder
    gen = dec.generator

    bucket_samples = CoreMLExportConstants.audio_samples_for_seconds(3)  # 72,000
    # decoder-har traces at 2x internal geometry (see export_synth/convert.py
    # comment on `geometry_samples`) because the iSTFT tail emits half as
    # many samples as the F0/HAR geometry nominally covers.
    geometry_samples = bucket_samples * 2
    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
    full_f0_len = int(round(geometry_samples / float(f0_samples_per_step)))  # 480
    frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)  # 240

    ref_s_out = _real_voice_ref_s()
    expected_in = dec.encode.conv1.in_channels - 2
    asr = torch.clamp(torch.randn(1, expected_in, frame_count) * 0.3, -1.0, 1.0)

    with torch.no_grad():
        # Natural (untrimmed) har frame count for this F0 geometry — probed
        # once so build_decoder_har_post_inputs_np's har_t crop/pad below is
        # a no-op (avoids padding out to an unrelated, much larger length).
        f0_probe = torch.zeros((1, full_f0_len), dtype=torch.float32)
        f0_u = gen.f0_upsamp(f0_probe[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_u)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, _ = gen.stft.transform(har_source)
        natural_har_t = int(har_spec.shape[-1])  # 28,801 per Ground Truth

    vi = {
        "asr": asr.numpy().astype(np.float32),
        "f0_curve": np.zeros((1, full_f0_len), dtype=np.float32),
        "n": np.zeros((1, full_f0_len), dtype=np.float32),
        "ref_s": ref_s_out.numpy().astype(np.float32),
    }
    x_pre_np, ref_s_np, har_np, _T_f0, _fc = build_decoder_har_post_inputs_np(
        dec, vi, sec=3, asr_len=frame_count, har_t=natural_har_t, warn_geometry=False
    )
    x_pre = torch.from_numpy(x_pre_np)
    ref_s_t = torch.from_numpy(ref_s_np)
    har_rep = torch.from_numpy(har_np)

    with torch.no_grad():
        spec, phase = _generator_spec_phase(gen, x_pre, ref_s_t, har_rep)
        waveform = gen.stft.inverse(spec, phase)

        # Cross-check: the hand-copied loop above must reproduce the
        # unmodified GeneratorFromHar module exactly, or this fixture would
        # be golden data for a different graph than what actually ships.
        reference_waveform = GeneratorFromHar(gen).eval()(x_pre, ref_s_t, har_rep)
        max_diff = (waveform - reference_waveform).abs().max().item()
        if max_diff > 0:
            raise SystemExit(
                "_generator_spec_phase diverged from GeneratorFromHar.forward: "
                f"max abs diff {max_diff}"
            )

    print(
        f"real_3s frame_count={frame_count} full_f0_len={full_f0_len} "
        f"har_frames={har_rep.shape[-1]} spec_frames={spec.shape[-1]} "
        f"waveform_samples={waveform.shape[-1]} "
        f"waveform_range=({waveform.min().item():.4f}, {waveform.max().item():.4f})"
    )
    _save("real_3s_spec", spec.squeeze(0))
    _save("real_3s_phase", phase.squeeze(0))
    _save("real_3s_waveform", waveform.squeeze(0))


def main() -> None:
    dump_synthetic()
    dump_real_3s()


if __name__ == "__main__":
    main()

"""Core ML export for the ANE-admissible 3 s generator (``decoder-har-ane`` mode).

Task T4 of README/Plans/ane-generator-a14-v1.md. Produces
``coreml/kokoro_decoder_har_ane_3s.mlpackage``: the same vocoder tail as the
shipped ``decoder-har`` export, reshaped so that EVERY tensor axis stays under
the A14 Neural Engine's 16,384-element limit.

WHY THIS IS A SEPARATE MODULE FROM convert.py
---------------------------------------------
``convert.py``'s ``export_synthesizers`` multiplexes three modes through six
separate if/elif chains, and carries a DurationModel/BERT warm-up plus an
IdentityAdaIN pass that this export does not use at all (the generator's
AdaIN1d style conditioning is deliberately LIVE here — see the note in
``convert.py``'s decoder-har trace branch). A fourth branch through all of that
would cost more than it saves. This module does one thing, at one bucket, and
reads top to bottom.

WHAT DIFFERS FROM ``decoder-har``, AND WHAT MUST NOT
---------------------------------------------------
Differs (all three are the point — see ``GeneratorFromHarANE`` in wrappers.py):

- ``har`` input is pre-trimmed to the generator body length (14,401 at 3 s)
  instead of its natural 28,801; the caller trims on the CPU.
- The noise branch's AdaIN normalizes over the kept region (T2-cleared: the
  shift is smaller than the model's own run-to-run noise —
  README/Notes/ane-pretrim-equivalence-2026-07-17.md).
- Outputs are ``spec``/``phase`` (1, 11, 14,401), not a 72,000-sample waveform.
  The host runs the iSTFT: ``HostISTFT.swift`` (T3) on device.

Held identical to ``decoder-har`` on purpose, because the compute-unit behavior
this export exists to test is only comparable if they match: fp16 compute
precision, ``mlprogram`` backend, ``ct.target.macOS13``, ``ComputeUnit.ALL`` at
convert time, fp32 tensor I/O, the ``rewrite_generator_ups_conv_transpose``
zero-insert rewrite, and the checkpoint/geometry derivation.

Run via the CLI: ``uv run python -m export_synth.main --mode decoder-har-ane``.
Parity and compute-plan verification live in ``scripts/verify_decoder_har_ane.py``
(the acceptance gate; this module does not self-validate numerics).
"""
from __future__ import annotations

import os
import time

import coremltools as ct
import numpy as np
import torch

from kokoro.conv_length import conv1d_output_length_from_module
from kokoro.coreml_export_verify import (
    assert_no_cpu_fallback_in_logs,
    capture_ane_logs,
)

from .convert import prepare_pytorch_models
from .wrappers import (
    CoreMLExportConstants,
    GeneratorFromHarANE,
    rewrite_generator_ups_conv_transpose,
)

# The A14 Neural Engine rejects any tensor axis above this many elements.
# Ground truth (README/Plans/ane-generator-a14-v1.md); this export exists
# solely to keep every axis under it, so it is asserted, never assumed.
ANE_MAX_AXIS_ELEMENTS = 16_384

# Ground truth for the 3 s bucket, asserted against the values derived from the
# checkpoint below. Hardcoded on purpose: if a checkpoint or geometry change
# ever moves these, this export must fail loudly rather than silently ship a
# graph that no longer fits the ANE.
BUCKET_SECONDS = 3
EXPECTED_ASR_FRAMES = 240
EXPECTED_FULL_F0_LEN = 480
EXPECTED_NATURAL_HAR_FRAMES = 28_801  # what the shipped decoder-har package takes
EXPECTED_BODY_FRAMES = 14_401  # trimmed har length == spec/phase length


def _derive_geometry(kmodel: torch.nn.Module) -> dict:
    """Derive the 3 s bucket's tensor geometry from the checkpoint, then assert
    it against Ground Truth.

    Mirrors ``export_synth/convert.py``'s ``decoder-har`` derivation exactly,
    including the 2x internal geometry (the iSTFT tail emits half as many
    samples as the F0/har geometry nominally covers, so a "3 s" package must be
    traced at 144,000 internal samples to emit 72,000).

    ``body_frames`` — the trimmed har length — is probed from the real upsample
    stack rather than computed from a formula: ``noise_convs[1]`` is k=1/s=1, so
    the har axis must equal the post-``reflection_pad`` body axis exactly.
    Probing is the only derivation that cannot drift from the modules involved.

    - Parameter kmodel: loaded ``KModel`` (see ``prepare_pytorch_models``).
    - Returns: dict of ``bucket_samples``, ``full_f0_len``, ``frame_count``,
      ``natural_har_t``, ``body_frames``, ``har_c``, ``dec_out_ch``.
    """
    gen = kmodel.decoder.generator
    bucket_samples = CoreMLExportConstants.audio_samples_for_seconds(BUCKET_SECONDS)
    geometry_samples = bucket_samples * 2

    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
    if f0_samples_per_step <= 0:
        raise ValueError(f"invalid f0_upsamp scale: {gen.f0_upsamp.scale_factor}")
    full_f0_len = int(round(geometry_samples / float(f0_samples_per_step)))
    frame_count = conv1d_output_length_from_module(full_f0_len, kmodel.decoder.F0_conv)
    dec_out_ch = int(kmodel.decoder.decode[-1].conv1.out_channels)

    with torch.no_grad():
        # Natural har: what hn-nsf actually emits at this F0 geometry (28,801).
        f0_rep = torch.zeros((1, full_f0_len), dtype=torch.float32)
        f0_u = gen.f0_upsamp(f0_rep[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_u)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har_rep = torch.cat([har_spec, har_phase], dim=1)
        natural_har_t = int(har_rep.shape[2])
        har_c = int(har_rep.shape[1])

        # Body axis: run the upsample stack the wrapper runs. This is the length
        # the pre-trimmed har must match.
        probe = torch.zeros((1, dec_out_ch, frame_count), dtype=torch.float32)
        for i in range(gen.num_upsamples):
            probe = gen.ups[i](probe)
        body_frames = int(gen.reflection_pad(probe).shape[-1])

    geometry = {
        "bucket_samples": bucket_samples,
        "full_f0_len": full_f0_len,
        "frame_count": frame_count,
        "natural_har_t": natural_har_t,
        "body_frames": body_frames,
        "har_c": har_c,
        "dec_out_ch": dec_out_ch,
    }
    for label, actual, expected in (
        ("ASR frames", frame_count, EXPECTED_ASR_FRAMES),
        ("F0 length", full_f0_len, EXPECTED_FULL_F0_LEN),
        ("natural har frames", natural_har_t, EXPECTED_NATURAL_HAR_FRAMES),
        ("body frames", body_frames, EXPECTED_BODY_FRAMES),
    ):
        if actual != expected:
            raise ValueError(
                f"geometry drift: {label} is {actual}, Ground Truth says {expected} "
                "(README/Plans/ane-generator-a14-v1.md). Refusing to export."
            )
    if body_frames >= ANE_MAX_AXIS_ELEMENTS:
        raise ValueError(
            f"body axis {body_frames} >= the ANE's {ANE_MAX_AXIS_ELEMENTS}-element "
            "limit; this export would be inadmissible by construction"
        )
    return geometry


def export_decoder_har_ane(
    output_dir: str = "coreml",
    *,
    precision: str | None = None,
    rewrite_ups_conv_transpose: bool = True,
) -> str:
    """Export the ANE-admissible 3 s generator to ``kokoro_decoder_har_ane_3s.mlpackage``.

    3 s only, by design: 3 s is the largest standard bucket whose generator body
    fits under the ANE's 16,384-element axis limit (the ceiling is ~3.4 s — see
    Ground Truth). There is no ``seconds`` parameter because there is no other
    admissible bucket to pass.

    Called by:
        - ``export_synth.main`` when ``--mode decoder-har-ane``.

    Verified by:
        - ``scripts/verify_decoder_har_ane.py`` — asserts the traced wrapper
          against T2's PyTorch pre-trim reference (>= 40 dB SNR through the
          Python iSTFT) and dumps the MLComputePlan op-device split. This
          function deliberately does NOT gate on numerics: a shape/finiteness
          smoke test on synthetic har (what ``convert.py``'s decoder-har mode
          does) is not evidence of anything, because the generator's output is
          ``exp()``-scaled and degenerate inputs make fp16 error meaningless.

    - Parameter output_dir: directory for the .mlpackage. Created if absent.
    - Parameter precision: ``'float16'``/``'fp16'`` (default, and what the A14
      requires — its ANE is fp16-only) or ``'float32'``/``'fp32'`` for debugging
      a suspected precision problem on the Mac.
    - Parameter rewrite_ups_conv_transpose: keep the zero-insert conv1d rewrite
      of the generator's ConvTranspose1d upsamples. Defaults ON here (the
      decoder-har CLI defaults it off) because this export exists for the ANE
      and the rewrite is part of the graph shape being tested.
    - Returns: path to the saved .mlpackage.
    """
    precision_norm = (precision or "").strip().lower()
    if precision_norm in ("float32", "fp32"):
        chosen_precision = ct.precision.FLOAT32
    elif precision_norm in ("", "float16", "fp16"):
        chosen_precision = ct.precision.FLOAT16
    else:
        raise ValueError(f"unsupported precision {precision!r}; use 'fp16' or 'fp32'")

    print("--- Loading Model ---")
    kmodel = prepare_pytorch_models("checkpoints/config.json", "checkpoints/kokoro-v1_0.pth")
    kmodel.eval()
    os.makedirs(output_dir, exist_ok=True)

    geometry = _derive_geometry(kmodel)
    frame_count = geometry["frame_count"]
    body_frames = geometry["body_frames"]
    print(
        f"decoder-har-ane geometry: {geometry['bucket_samples']} output samples "
        f"({geometry['bucket_samples'] * 2} internal) -> F0/N {geometry['full_f0_len']} "
        f"-> ASR {frame_count} -> har {geometry['natural_har_t']} trimmed to "
        f"{body_frames} -> spec/phase {body_frames} "
        f"(max axis {body_frames} < ANE limit {ANE_MAX_AXIS_ELEMENTS})"
    )

    x_pre_shape = (1, geometry["dec_out_ch"], frame_count)
    ref_s_shape = (1, CoreMLExportConstants.VOICE_EMBEDDING_DIM)
    har_shape = (1, geometry["har_c"], body_frames)

    gen_from_har = GeneratorFromHarANE(kmodel.decoder.generator).eval()
    if rewrite_ups_conv_transpose:
        rewritten = rewrite_generator_ups_conv_transpose(gen_from_har.generator)
        print(
            f"decoder-har-ane graph rewrite: replaced {rewritten} main "
            "ConvTranspose1d upsample layers with zero-insert conv1d"
        )

    print(f"[{time.ctime()}] Tracing model with torch.jit.trace...")
    # Zeros are adequate here and only here: tracing records graph structure, and
    # every op in this wrapper is shape-static. Numeric checks use real inputs
    # (scripts/verify_decoder_har_ane.py) precisely because zeros would not.
    trace_inputs = (
        torch.zeros(x_pre_shape, dtype=torch.float32),
        torch.zeros(ref_s_shape, dtype=torch.float32),
        torch.zeros(har_shape, dtype=torch.float32),
    )
    torch.manual_seed(0)
    with torch.no_grad():
        traced_model = torch.jit.trace(gen_from_har, trace_inputs, strict=False, check_trace=False)
        spec_out, phase_out = traced_model(*trace_inputs)

    expected_bins = int(kmodel.decoder.generator.post_n_fft // 2 + 1)
    for label, tensor in (("spec", spec_out), ("phase", phase_out)):
        actual = tuple(int(d) for d in tensor.shape)
        expected = (1, expected_bins, body_frames)
        if actual != expected:
            raise ValueError(f"traced {label} shape {actual}, expected {expected}")
    print(
        f"decoder-har-ane traced spec/phase: {tuple(int(d) for d in spec_out.shape)} "
        f"(host iSTFT emits {(body_frames - 1) * 5} samples, advertised bucket "
        f"{geometry['bucket_samples']})"
    )

    print(f"[{time.ctime()}] Converting to Core ML...")
    with capture_ane_logs() as convert_buf:
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="x_pre", shape=x_pre_shape, dtype=np.float32),
                ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
                ct.TensorType(name="har", shape=har_shape, dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="spec"), ct.TensorType(name="phase")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=chosen_precision,
            compute_units=ct.ComputeUnit.ALL,
        )
    assert_no_cpu_fallback_in_logs(convert_buf.getvalue(), phase="decoder-har-ane ct.convert")
    print(f"[{time.ctime()}] Core ML conversion complete.")

    output_path = os.path.join(output_dir, f"kokoro_decoder_har_ane_{BUCKET_SECONDS}s.mlpackage")
    mlmodel.save(output_path)
    print(f"✅ Saved ANE generator to: {output_path}")
    print(
        "   Next: uv run python scripts/verify_decoder_har_ane.py "
        "(parity + compute plan). ANE admittance itself is proven on the phone, "
        "never on this Mac — macOS silently reroutes graphs the ANE rejects."
    )
    return output_path

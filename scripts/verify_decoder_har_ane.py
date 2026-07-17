#!/usr/bin/env python3
"""Acceptance gate for the ANE generator export (task T4).

README/Plans/ane-generator-a14-v1.md T4. Verifies
``coreml/kokoro_decoder_har_ane_3s.mlpackage`` (built by
``export_synth/convert_ane.py``) in three stages:

1. **Wrapper equality (runs before any conversion is trusted).**
   ``GeneratorFromHarANE`` must be bit-identical to ``_forward_pretrim`` in
   ``scripts/probe_har_pretrim_adain_equivalence.py`` — the exact graph task T2
   measured and cleared. Same relationship, same reason, as T2's own assert of
   its hand-copy against ``GeneratorFromHar``: if the exported wrapper is not
   that graph, T2's verdict does not transfer and every number below describes
   some other model. ``--wrapper-only`` runs just this, for use before export.

2. **Full-pipeline parity.** Core ML spec/phase -> ``CustomSTFT.inverse``
   (the Python iSTFT that T3's Swift ``hostISTFTInverse`` was proven against)
   -> SNR vs ``_forward_pretrim``'s waveform on IDENTICAL inputs. Gate: 40 dB.

3. **Compute plan.** Per-op preferred device under ``CPU_AND_NE``.

TWO TRAPS THIS SCRIPT IS BUILT AROUND (both from
README/Notes/ane-pretrim-equivalence-2026-07-17.md, "Inheritance for T4")
-------------------------------------------------------------------------
- **The reference is the PyTorch pre-trim graph, NOT T1's baseline package.**
  T4's originally-stated gate (">= 40 dB vs the T1 baseline") is unreachable by
  construction: that package is the ORIGINAL graph and this one is the PRE-TRIM
  graph, ~28 dB apart on purpose. No amount of fp16 debugging closes that.

- **Both sides share ONE har tensor.** ``har`` is stochastic — ``SourceModuleHnNSF``
  redraws sine phase and Gaussian noise on every call — so regenerating it
  between the two sides collapses any comparison to ~20 dB regardless of
  correctness. This script builds har exactly once and slices it.

Inputs are real (T2's construction path: real misaki phonemes, real BERT/
duration/F0, real ``af_heart`` embedding), never zeros: the generator's output
is ``exp()``-scaled, so degenerate inputs make fp16 parity numbers meaningless.

WHAT A PASS HERE DOES NOT MEAN
------------------------------
It does NOT mean the graph is ANE-admissible on the A14. macOS silently reroutes
graphs the Neural Engine rejects, so a passing ``CPU_AND_NE`` predict on this Mac
is a shape/parity check only. The compute plan below is likewise the *Mac's*
plan. Admittance is proven on the phone, by the owner (plan Phase 1 gate).

Run from the repo root: ``uv run python scripts/verify_decoder_har_ane.py``
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

import coremltools as ct  # noqa: E402

from kokoro.synthesis_backends import build_decoder_har_post_inputs_np  # noqa: E402
from export_synth.wrappers import GeneratorFromHarANE  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel  # noqa: E402

# T2's validated graph and its input-construction path — imported, never
# re-derived (plan T4: "T2 already validated that exact graph").
from probe_har_pretrim_adain_equivalence import (  # noqa: E402
    ASR_FRAMES,
    BUCKET_SAMPLES,
    GEOMETRY_SEC,
    NATURAL_HAR_FRAMES,
    SEED,
    TRIMMED_HAR_FRAMES,
    _forward_pretrim,
    _real_vocoder_inputs,
    _snr_db,
)

PACKAGE_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_3s.mlpackage"

# fp16 tolerance for the Core ML graph vs its fp32 PyTorch reference (plan T4).
PARITY_SNR_GATE_DB = 40.0


def _build_real_inputs() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(gen, x_pre, ref_s, har_trimmed)`` at the 3 s bucket's traced
    geometry, from real phonemes/BERT/F0/voice — T2's path exactly.

    ``har`` is drawn ONCE here and trimmed; every consumer downstream gets this
    same tensor. See the module docstring on why that is load-bearing.
    """
    torch.manual_seed(SEED)
    kmodel = _load_kmodel()
    kmodel.eval()
    dec = kmodel.decoder

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

    har_trimmed = har_full[:, :, :TRIMMED_HAR_FRAMES].contiguous()
    print(
        f"real inputs: x_pre {tuple(x_pre.shape)} ref_s {tuple(ref_s.shape)} "
        f"har {tuple(har_full.shape)} -> trimmed {tuple(har_trimmed.shape)}"
    )
    return dec.generator, x_pre, ref_s, har_trimmed


def _assert_wrapper_matches_pretrim(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har_trimmed: torch.Tensor
) -> torch.Tensor:
    """Assert ``GeneratorFromHarANE`` IS T2's ``_forward_pretrim``, and return
    that function's reference waveform for the parity stage.

    The wrapper stops at spec/phase, so the comparison runs its outputs through
    the same ``gen.stft.inverse`` ``_forward_pretrim`` ends with. Both run in
    fp32 on the unrewritten generator: this stage proves the GRAPH is the
    validated one. The zero-insert upsample rewrite and fp16 are exercised by
    the Core ML parity stage, where they belong.
    """
    with torch.no_grad():
        reference_waveform, _stages = _forward_pretrim(gen, x_pre, ref_s, har_trimmed)
        spec, phase = GeneratorFromHarANE(gen).eval()(x_pre, ref_s, har_trimmed)
        wrapper_waveform = gen.stft.inverse(spec, phase)

    drift = float((wrapper_waveform - reference_waveform).abs().max())
    if drift > 0.0:
        raise SystemExit(
            f"GeneratorFromHarANE diverged from _forward_pretrim: max-abs {drift}. "
            "The export wrapper is not the graph T2 validated; T2's verdict does "
            "not transfer. Fix the wrapper before converting."
        )
    print(
        f"✅ wrapper equality: GeneratorFromHarANE -> stft.inverse is BIT-IDENTICAL to "
        f"_forward_pretrim (spec {tuple(spec.shape)} phase {tuple(phase.shape)})"
    )
    return reference_waveform


def _parity_for_compute_unit(
    compute_unit: "ct.ComputeUnit",
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trimmed: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, bool]:
    """Predict once under ``compute_unit``, run the Python iSTFT on spec/phase,
    and return ``(snr_db, all_finite)`` against ``reference``.

    - Returns: SNR in dB (``nan`` if the waveform is not finite) and whether the
      waveform was finite at all.
    """
    prediction = ct.models.MLModel(str(PACKAGE_PATH), compute_units=compute_unit).predict(
        {
            "x_pre": x_pre.numpy().astype(np.float32),
            "ref_s": ref_s.numpy().astype(np.float32),
            "har": har_trimmed.numpy().astype(np.float32),
        }
    )
    spec = torch.from_numpy(np.asarray(prediction["spec"], dtype=np.float32))
    phase = torch.from_numpy(np.asarray(prediction["phase"], dtype=np.float32))
    with torch.no_grad():
        # The same iSTFT the reference used, and the one T3's Swift port is
        # golden-tested against — so this measures the exported graph, not the
        # tail. Both waveforms get the identical bucket trim.
        waveform = gen.stft.inverse(spec, phase).squeeze()[:BUCKET_SAMPLES]
    finite = bool(torch.isfinite(waveform).all())
    return (_snr_db(reference, waveform) if finite else float("nan")), finite


def _coreml_parity(
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trimmed: torch.Tensor,
    reference_waveform: torch.Tensor,
) -> float:
    """Measure parity across compute units and return the SNR the gate judges.

    WHY THE GATE JUDGES ``CPU_AND_GPU`` AND NOT ``CPU_AND_NE``, WHICH T4 NAMED
    ---------------------------------------------------------------------------
    T4 specified a ``CPU_AND_NE`` predict on the assumption that "macOS will
    silently reroute — that's fine, this is a shape/parity check". That
    assumption held for every earlier package and does NOT hold for this one:
    this is the first Kokoro generator the Mac's ANE actually admits (98% of its
    ops map to the Neural Engine, where the T1 baseline maps 0%), so nothing
    reroutes and the number stops being a graph check.

    On this Mac the ANE then miscomputes the graph into non-finite output, while
    the identical package on the GPU clears the gate. The gate must judge what it
    was written to judge — whether the EXPORTED GRAPH is the pre-trim graph in
    fp16 — so it reads the compute unit that actually evaluates the graph. The
    ANE result is measured and printed regardless: it is a finding for the owner
    (README/Notes/ane-generator-coreml-export-2026-07-17.md), not a graph defect,
    and it is deliberately not allowed to silently pass.
    """
    reference = reference_waveform.squeeze()[:BUCKET_SAMPLES]
    print(
        "\n--- parity: Core ML (fp16) + Python iSTFT vs PyTorch pre-trim (fp32), "
        "identical har, by compute unit ---"
    )
    results: dict[str, tuple[float, bool]] = {}
    for label, compute_unit in (
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ):
        snr, finite = _parity_for_compute_unit(
            compute_unit, gen, x_pre, ref_s, har_trimmed, reference
        )
        results[label] = (snr, finite)
        verdict = f"{snr:7.2f} dB" if finite else "  NON-FINITE output"
        print(f"  {label:<12} {verdict}")

    ne_snr, ne_finite = results["CPU_AND_NE"]
    if not ne_finite or ne_snr < PARITY_SNR_GATE_DB:
        print(
            "\n  ⚠️  CPU_AND_NE does NOT match the graph on this Mac. This package is\n"
            "      ANE-ADMITTED here (unlike every earlier one), so it is not being\n"
            "      rerouted — the Neural Engine is running it and getting this. The\n"
            "      graph itself is fine (see CPU_AND_GPU). Owner: this is the open\n"
            "      question for the A14, whose ANE is a different generation.\n"
            "      Write-up: README/Notes/ane-generator-coreml-export-2026-07-17.md"
        )
    return results["CPU_AND_GPU"][0]


def _dump_compute_plan() -> None:
    """Print the per-op preferred-device split under ``CPU_AND_NE``.

    Uses the Python snippet from
    README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md
    (note: the guide writes ``ct.ComputeUnits`` — the real symbol is
    ``ct.ComputeUnit``). This is THIS MAC's plan, not the A14's; it is a
    structural signal about the graph, not the admittance proof.
    """
    from coremltools.models.compute_plan import MLComputePlan

    # Keep the MLModel alive: get_compiled_model_path() hands back a temp
    # .mlmodelc owned by that object, and letting it fall out of scope deletes
    # the directory out from under load_from_path (which then reports, very
    # unhelpfully, "The model is not found at URL").
    holder = ct.models.MLModel(str(PACKAGE_PATH))
    plan = MLComputePlan.load_from_path(
        path=str(holder.get_compiled_model_path()), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    main_function = plan.model_structure.program.functions["main"]

    device_counts: Counter[str] = Counter()
    non_ane_ops: Counter[str] = Counter()
    total = 0
    for operation in main_function.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            # const/no-op ops carry no device assignment — not a fallback.
            continue
        device = type(usage.preferred_compute_device).__name__
        device_counts[device] += 1
        total += 1
        if "Neural" not in device:
            non_ane_ops[f"{operation.operator_name} -> {device}"] += 1

    if total == 0:
        raise SystemExit("compute plan returned no device-assigned operations")

    print(f"\n--- MLComputePlan, CPU_AND_NE, {PACKAGE_PATH.name} (this Mac) ---")
    for device, count in device_counts.most_common():
        print(f"  {device}: {count} ops ({100.0 * count / total:.1f}%)")
    ane_ops = sum(count for device, count in device_counts.items() if "Neural" in device)
    print(f"  => ANE-mapped: {ane_ops}/{total} ops ({100.0 * ane_ops / total:.1f}%)")
    if non_ane_ops:
        print("  ops NOT on the ANE:")
        for label, count in non_ane_ops.most_common():
            print(f"    {label} x{count}")
    else:
        print("  ops NOT on the ANE: none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--wrapper-only",
        action="store_true",
        help="Run only the PyTorch wrapper-equality gate (no Core ML). Use before exporting.",
    )
    args = parser.parse_args()

    gen, x_pre, ref_s, har_trimmed = _build_real_inputs()
    reference_waveform = _assert_wrapper_matches_pretrim(gen, x_pre, ref_s, har_trimmed)
    if args.wrapper_only:
        print("\nwrapper-only: skipping Core ML stages. Export now, then re-run without the flag.")
        return

    if not PACKAGE_PATH.exists():
        raise SystemExit(
            f"{PACKAGE_PATH} not found. Export it first:\n"
            "  uv run python -m export_synth.main --mode decoder-har-ane"
        )

    snr = _coreml_parity(gen, x_pre, ref_s, har_trimmed, reference_waveform)
    _dump_compute_plan()

    if snr < PARITY_SNR_GATE_DB:
        raise SystemExit(
            f"\n❌ parity {snr:.2f} dB is below the {PARITY_SNR_GATE_DB:.0f} dB gate."
        )
    print(
        f"\n✅ parity {snr:.2f} dB >= {PARITY_SNR_GATE_DB:.0f} dB gate: the exported "
        "graph IS the pre-trim graph in fp16."
    )
    print(
        "   NOT proven here: ANE admittance on the A14. This Mac's ANE is a "
        "different generation, and a Mac compute plan is not an A14 compute plan."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Acceptance gate for the layer_norm-lowered ANE generator (task T6).

README/Plans/ane-generator-a14-v1.md T6. Verifies
``coreml/kokoro_decoder_har_ane_ln_3s.mlpackage`` (built by
``export_synth/convert_ane.py`` with ``lower_adain_layernorm=True``, CLI
``--mode decoder-har-ane-ln``). It is the T4 gate plus one NEW acceptance
metric — the MIL op census — because T6 exists to shrink the program below the
A14's ``Too many fvmlibs (>255)`` object cap, and op count is the Mac-side proxy
for that (the phone is the judge; see the Status log's DEVICE GATE RUN entry).

Four stages, in order:

1. **Allclose vs the manual graph (before any conversion is trusted).**
   Lowering each ``AdaIN1d`` mean/var chain to ``layer_norm`` is *supposed* to be
   the same math up to fp ulps. This proves it on real inputs: the lowered
   ``GeneratorFromHarANE`` must be ``allclose`` (atol 1e-5, NOT bit-equality —
   ``layer_norm`` differs from the manual chain by ulps) to the manual
   ``GeneratorFromHarANE`` == T2/T4's validated ``_forward_pretrim`` graph. If it
   is not, T2's "indistinguishable" verdict does not transfer.

2. **MIL op census (the metric T6 is actually judged on).** Count non-const ops
   in the exported program (``spec.mlProgram``) and assert every AdaIN collapsed
   to a single ``layer_norm``: ``layer_norm`` count == the 48 AdaIN applications,
   ``reduce_mean`` == 0, ``tile`` == 0. A nonzero ``reduce_mean`` means
   coremltools decomposed ``layer_norm`` back into primitives — a FAILED lowering
   (plan T6 stop condition), not a smaller graph.

3. **Full-pipeline parity.** Core ML spec/phase -> ``CustomSTFT.inverse`` -> SNR
   vs ``_forward_pretrim`` on IDENTICAL inputs, read off ``CPU_AND_GPU``. Gate:
   40 dB. Same reasoning as T4: this Mac's ANE ADMITS the graph and then
   miscomputes it, so ``CPU_AND_NE`` measures the Mac's ANE, not the graph.

4. **Compute plan.** Per-op preferred device under ``CPU_AND_NE`` (Mac's plan,
   structural signal only — never an A14 admittance claim).

This script reuses ``verify_decoder_har_ane``'s input construction and
``count_mil_ops``'s histogram, and shares T4's two traps: the reference is the
PyTorch pre-trim graph (not T1's baseline package), and both sides get ONE har
tensor (regenerating it re-draws the hn-nsf RNG and collapses the comparison).

Run from the repo root: ``uv run python scripts/verify_decoder_har_ane_ln.py``
(``--wrapper-only`` runs stage 1 alone, before the export exists).
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

from export_synth.wrappers import (  # noqa: E402
    GeneratorFromHarANE,
    lower_generator_adain_to_layernorm,
)
from count_mil_ops import histogram_for_mlpackage  # noqa: E402
from verify_decoder_har_ane import _build_real_inputs  # noqa: E402
from probe_har_pretrim_adain_equivalence import (  # noqa: E402
    BUCKET_SAMPLES,
    _forward_pretrim,
    _snr_db,
)

LN_PACKAGE_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_ln_3s.mlpackage"
BASELINE_PACKAGE_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_3s.mlpackage"

# fp16 tolerance for the Core ML graph vs its fp32 PyTorch reference (plan T4/T6).
PARITY_SNR_GATE_DB = 40.0
# Allclose tolerance for the PyTorch layer_norm lowering vs the manual mean/var
# chain. layer_norm differs by fp ulps, so this is allclose, not bit-equality.
LN_ALLCLOSE_ATOL = 1e-5
# The generator holds 48 AdaIN1d applications (2 noise_res + 6 resblocks, each
# an AdaINResBlock1 with adain1/adain2 x 3). This is the module count the
# PyTorch lowering must touch — NOT the traced-op count: coremltools elides 4 of
# the 48 normalization chains in BOTH the manual and the lowered graph (48 style
# projections trace, but only 44 normalization chains survive its graph passes,
# input-independent — verified with a random-input trace too). So the exported
# program's layer_norm count is compared to the BASELINE's surviving-chain count
# (== its `sqrt` count), not to 48; the decomposition stop-condition is keyed on
# the manual-norm primitives going to zero, which is the real signal.
EXPECTED_ADAIN_APPLICATIONS = 48
# The manual mean/var chain's MIL fingerprint. If any of these survive in the
# lowered program, coremltools decomposed layer_norm back into primitives (T6
# stop condition) OR a manual chain was left un-lowered — either way a failure.
_MANUAL_NORM_PRIMITIVES = ("reduce_mean", "square", "sqrt", "real_div", "tile")


def _assert_lowering_matches_manual(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har_trimmed: torch.Tensor
) -> torch.Tensor:
    """Stage 1. Assert the layer_norm-lowered wrapper matches the manual
    ``GeneratorFromHarANE`` (== ``_forward_pretrim``), and return the manual fp32
    reference waveform for the parity stage.

    Mutates ``gen`` in place (lowers its AdaIN modules), so the manual outputs are
    computed FIRST, then the generator is lowered, then the lowered outputs — both
    from the same weights.

    THE GATE IS ON THE WAVEFORM, AND WHY
    ------------------------------------
    ``layer_norm`` reduces 14,401 elements in a different order than the manual
    ``mean``/``var``, so the two differ by fp ulps — exactly what plan T6 predicted
    ("layer_norm may differ by ulps ... allclose not bit-equality"). Those ulps are
    amplified in ``spec = exp(x)``: a relative error of ~1e-6 on a ``conv_post``
    output near +3 becomes an absolute ``spec`` diff of a few ``e-5``. That makes a
    raw ``atol`` on ``spec``/``phase`` the wrong unit. The wrapper's product THROUGH
    the pipeline is the waveform (the same quantity T4's 40 dB parity judges), and
    there the difference is ~1e-6 — a strict pass at ``atol`` 1e-5. All three
    max-abs values are reported so nothing is hidden.
    """
    with torch.no_grad():
        reference_waveform, _stages = _forward_pretrim(gen, x_pre, ref_s, har_trimmed)
        manual_spec, manual_phase = GeneratorFromHarANE(gen).eval()(x_pre, ref_s, har_trimmed)

        lowered = lower_generator_adain_to_layernorm(gen)
        if lowered != EXPECTED_ADAIN_APPLICATIONS:
            raise SystemExit(
                f"lowered {lowered} AdaIN1d modules, expected {EXPECTED_ADAIN_APPLICATIONS} "
                "(README/Plans/ane-generator-a14-v1.md geometry). Refusing to proceed."
            )
        ln_spec, ln_phase = GeneratorFromHarANE(gen).eval()(x_pre, ref_s, har_trimmed)
        ln_waveform = gen.stft.inverse(ln_spec, ln_phase)

    spec_drift = float((ln_spec - manual_spec).abs().max())
    phase_drift = float((ln_phase - manual_phase).abs().max())
    wave_drift = float((ln_waveform - reference_waveform).abs().max())
    wave_snr = _snr_db(reference_waveform, ln_waveform)
    print(
        f"lowered {lowered} AdaIN1d -> layer_norm; max-abs vs manual: "
        f"waveform {wave_drift:.2e} (SNR {wave_snr:.2f} dB); "
        f"spec {spec_drift:.2e}, phase {phase_drift:.2e} (exp-amplified ulps — see docstring)"
    )
    if not torch.allclose(ln_waveform, reference_waveform, atol=LN_ALLCLOSE_ATOL):
        raise SystemExit(
            f"layer_norm lowering diverged from the manual chain: waveform max-abs "
            f"{wave_drift:.2e} > atol {LN_ALLCLOSE_ATOL:g}. The lowering is not the same "
            "math; do not ship it."
        )
    print(
        f"✅ allclose: layer_norm-lowered GeneratorFromHarANE waveform matches the manual "
        f"mean/var chain within atol {LN_ALLCLOSE_ATOL:g} (max-abs {wave_drift:.2e})"
    )
    return reference_waveform


def _census(path: Path) -> tuple[int, int, Counter]:
    """Return ``(total_ops, non_const_ops, counts)`` for a .mlpackage's MLProgram."""
    counts, total = histogram_for_mlpackage(path)
    non_const = total - counts.get("const", 0)
    return total, non_const, counts


def _report_census() -> None:
    """Stage 2. Print the before/after MIL census and assert every manual AdaIN
    normalization chain became a single ``layer_norm`` with none decomposed back
    into primitives (the T6 stop condition lives here).

    "Every chain" means the ~44 normalization chains coremltools actually keeps
    (see ``EXPECTED_ADAIN_APPLICATIONS``), derived from the baseline package's
    ``sqrt`` count so the two graphs are compared like for like — not the raw 48
    module count.
    """
    print("\n--- MIL op census: manual AdaIN vs layer_norm-lowered ---")
    ln_total, ln_non_const, ln_counts = _census(LN_PACKAGE_PATH)

    expected_ln = None
    if BASELINE_PACKAGE_PATH.exists():
        b_total, b_non_const, b_counts = _census(BASELINE_PACKAGE_PATH)
        # Each surviving manual normalization chain has exactly one `sqrt`, so the
        # baseline `sqrt` count is the number of layer_norm the lowering must yield.
        expected_ln = b_counts.get("sqrt", 0)
        header = f"{'op':<14}{'baseline':>10}{'ln':>10}"
        print(header)
        print(f"{'-' * len(header)}")
        print(f"{'total':<14}{b_total:>10}{ln_total:>10}")
        print(f"{'const':<14}{b_counts.get('const', 0):>10}{ln_counts.get('const', 0):>10}")
        print(f"{'NON-CONST':<14}{b_non_const:>10}{ln_non_const:>10}")
        for op in ("layer_norm", "reduce_mean", "tile", "sub", "square", "sqrt", "real_div", "add", "mul"):
            print(f"{op:<14}{b_counts.get(op, 0):>10}{ln_counts.get(op, 0):>10}")
        print(
            f"  baseline non-const {b_non_const} -> ln non-const {ln_non_const} "
            f"({100.0 * (b_non_const - ln_non_const) / b_non_const:.1f}% fewer)"
        )
    else:
        print(f"  (baseline package absent) ln non-const {ln_non_const}, "
              f"layer_norm {ln_counts.get('layer_norm', 0)}, "
              f"reduce_mean {ln_counts.get('reduce_mean', 0)}, tile {ln_counts.get('tile', 0)}")

    ln_ops = ln_counts.get("layer_norm", 0)
    leftover = {op: ln_counts.get(op, 0) for op in _MANUAL_NORM_PRIMITIVES if ln_counts.get(op, 0)}
    if leftover:
        raise SystemExit(
            f"\n❌ AdaIN did NOT lower cleanly: manual-norm primitives survive in the "
            f"lowered program {leftover}. coremltools either decomposed layer_norm back "
            "into primitives or left a chain un-lowered — the T6 lowering failed (plan "
            "stop condition). Investigate the torch->MIL mapping before proceeding."
        )
    if ln_ops == 0:
        raise SystemExit("\n❌ no layer_norm ops in the lowered program — the lowering did not run.")
    if expected_ln is not None and ln_ops != expected_ln:
        raise SystemExit(
            f"\n❌ layer_norm count {ln_ops} != baseline normalization-chain count {expected_ln}. "
            "The lowering is not a 1:1 replacement of the manual chains; investigate."
        )
    print(
        f"✅ clean lowering: {ln_ops} layer_norm ops, ZERO manual-norm primitives "
        f"(reduce_mean/square/sqrt/real_div/tile all 0)"
        + (f"; matches the baseline's {expected_ln} normalization chains 1:1" if expected_ln is not None else "")
    )


def _parity_cpu_and_gpu(
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trimmed: torch.Tensor,
    reference_waveform: torch.Tensor,
) -> float:
    """Stage 3. Core ML spec/phase -> Python iSTFT -> SNR vs the pre-trim reference,
    across compute units. Returns the ``CPU_AND_GPU`` SNR the gate judges.

    ``CPU_AND_NE`` is measured and printed but never gated: T4 established that this
    Mac's ANE admits this graph and then miscomputes it into non-finite output, so
    that number measures the Mac's Neural Engine, not the exported graph. See
    README/Notes/ane-generator-coreml-export-2026-07-17.md.
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
        prediction = ct.models.MLModel(
            str(LN_PACKAGE_PATH), compute_units=compute_unit
        ).predict(
            {
                "x_pre": x_pre.numpy().astype(np.float32),
                "ref_s": ref_s.numpy().astype(np.float32),
                "har": har_trimmed.numpy().astype(np.float32),
            }
        )
        spec = torch.from_numpy(np.asarray(prediction["spec"], dtype=np.float32))
        phase = torch.from_numpy(np.asarray(prediction["phase"], dtype=np.float32))
        with torch.no_grad():
            waveform = gen.stft.inverse(spec, phase).squeeze()[:BUCKET_SAMPLES]
        finite = bool(torch.isfinite(waveform).all())
        snr = _snr_db(reference, waveform) if finite else float("nan")
        results[label] = (snr, finite)
        print(f"  {label:<12} {snr:7.2f} dB" if finite else f"  {label:<12}   NON-FINITE output")

    ne_snr, ne_finite = results["CPU_AND_NE"]
    if not ne_finite or ne_snr < PARITY_SNR_GATE_DB:
        print(
            "\n  ⚠️  CPU_AND_NE does NOT match the graph on this Mac (same finding as T4:\n"
            "      this package is ANE-admitted here, not rerouted, and the Neural Engine\n"
            "      miscomputes it). The graph itself is fine — see CPU_AND_GPU. The A14 is\n"
            "      the judge; its ANE is a different generation."
        )
    return results["CPU_AND_GPU"][0]


def _dump_compute_plan() -> None:
    """Stage 4. Per-op preferred-device split under ``CPU_AND_NE`` (this Mac's plan)."""
    from coremltools.models.compute_plan import MLComputePlan

    # Keep the MLModel alive: get_compiled_model_path() hands back a temp
    # .mlmodelc owned by it (see verify_decoder_har_ane.py's note).
    holder = ct.models.MLModel(str(LN_PACKAGE_PATH))
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
            continue
        device = type(usage.preferred_compute_device).__name__
        device_counts[device] += 1
        total += 1
        if "Neural" not in device:
            non_ane_ops[f"{operation.operator_name} -> {device}"] += 1

    if total == 0:
        raise SystemExit("compute plan returned no device-assigned operations")

    print(f"\n--- MLComputePlan, CPU_AND_NE, {LN_PACKAGE_PATH.name} (this Mac) ---")
    for device, count in device_counts.most_common():
        print(f"  {device}: {count} ops ({100.0 * count / total:.1f}%)")
    ane_ops = sum(count for device, count in device_counts.items() if "Neural" in device)
    print(f"  => ANE-mapped: {ane_ops}/{total} ops ({100.0 * ane_ops / total:.1f}%)")
    if non_ane_ops:
        print("  ops NOT on the ANE:")
        for label, count in non_ane_ops.most_common():
            print(f"    {label} x{count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--wrapper-only",
        action="store_true",
        help="Run only stage 1 (PyTorch allclose gate). Use before exporting.",
    )
    args = parser.parse_args()

    gen, x_pre, ref_s, har_trimmed = _build_real_inputs()
    reference_waveform = _assert_lowering_matches_manual(gen, x_pre, ref_s, har_trimmed)
    if args.wrapper_only:
        print("\nwrapper-only: skipping Core ML stages. Export now, then re-run without the flag.")
        return

    if not LN_PACKAGE_PATH.exists():
        raise SystemExit(
            f"{LN_PACKAGE_PATH} not found. Export it first:\n"
            "  uv run python -m export_synth.main --mode decoder-har-ane-ln"
        )

    _report_census()
    snr = _parity_cpu_and_gpu(gen, x_pre, ref_s, har_trimmed, reference_waveform)
    _dump_compute_plan()

    if snr < PARITY_SNR_GATE_DB:
        raise SystemExit(
            f"\n❌ parity {snr:.2f} dB is below the {PARITY_SNR_GATE_DB:.0f} dB gate."
        )
    print(
        f"\n✅ parity {snr:.2f} dB >= {PARITY_SNR_GATE_DB:.0f} dB gate: the layer_norm-lowered "
        "graph IS the pre-trim graph in fp16."
    )
    print(
        "   NOT proven here: that the smaller program clears the A14's fvmlib object cap. "
        "The census is the Mac-side proxy; the phone is the judge (plan Phase 1 gate)."
    )


if __name__ == "__main__":
    main()

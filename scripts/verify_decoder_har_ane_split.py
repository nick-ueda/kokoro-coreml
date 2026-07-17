#!/usr/bin/env python3
"""Acceptance gate for the rate-boundary split ANE generator (task T6, secondary).

README/Plans/ane-generator-a14-v1.md T6. Verifies the two split packages built by
``export_synth/convert_ane.py`` (``--mode decoder-har-ane-split``):

- ``coreml/kokoro_decoder_har_ane_ln_trunk_3s.mlpackage`` (x_pre/ref_s/har -> trunk)
- ``coreml/kokoro_decoder_har_ane_ln_body_3s.mlpackage`` (trunk/ref_s/har -> spec/phase)

WHY THE SPLIT EXISTS
--------------------
The A14 device gate rejected the monolithic ANE generator with ``Too many
fvmlibs (>255)`` — a program-SIZE cap (Status log, DEVICE GATE RUN). The
layer_norm lowering (``verify_decoder_har_ane_ln.py``) cuts the program ~35%; if
that still exceeds the cap, splitting the graph at the 240->2,400->14,401 rate
boundary yields two smaller programs, neither as large as the whole. The phone is
the judge (per-half compute-plan / load probe); this script is the Mac-side proof
that the split is NUMERICALLY the same graph and a census of the two halves.

Three stages:

1. **Chained allclose (PyTorch).** Trunk wrapper -> body wrapper, ln-lowered, must
   reproduce ``_forward_pretrim``'s waveform within atol 1e-5 (same waveform-gate
   reasoning as the ln script: exp amplifies layer_norm ulps in spec/phase).
2. **Chained Core ML parity.** trunk package -> body package under ``CPU_AND_GPU``
   -> Python iSTFT -> SNR vs ``_forward_pretrim``. Gate: 40 dB. ``CPU_AND_NE`` is
   the Mac's ANE (miscomputes, per T4) — measured, never gated.
3. **Per-stage census.** Non-const op count per half, vs the monolithic ln
   package, so the "does each half fit" question has a Mac-side proxy.

Shares T4's two traps: reference is the pre-trim graph (not T1 baseline), and
both sides get ONE har tensor.

Run from the repo root: ``uv run python scripts/verify_decoder_har_ane_split.py``
(``--wrapper-only`` runs stage 1 alone, before the export exists).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

import coremltools as ct  # noqa: E402

from export_synth.wrappers import (  # noqa: E402
    GeneratorBodyANE,
    GeneratorTrunkANE,
    lower_generator_adain_to_layernorm,
)
from count_mil_ops import histogram_for_mlpackage  # noqa: E402
from verify_decoder_har_ane import _build_real_inputs  # noqa: E402
from probe_har_pretrim_adain_equivalence import (  # noqa: E402
    BUCKET_SAMPLES,
    _forward_pretrim,
    _snr_db,
)

TRUNK_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_ln_trunk_3s.mlpackage"
BODY_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_ln_body_3s.mlpackage"
LN_PACKAGE_PATH = _ROOT / "coreml" / "kokoro_decoder_har_ane_ln_3s.mlpackage"

PARITY_SNR_GATE_DB = 40.0
LN_ALLCLOSE_ATOL = 1e-5
EXPECTED_ADAIN_APPLICATIONS = 48


def _assert_chain_matches_pretrim(
    gen: torch.nn.Module, x_pre: torch.Tensor, ref_s: torch.Tensor, har_trimmed: torch.Tensor
) -> torch.Tensor:
    """Stage 1. Chain the ln-lowered trunk+body wrappers and assert their waveform
    matches ``_forward_pretrim`` within atol 1e-5. Returns the pre-trim reference.
    """
    with torch.no_grad():
        reference_waveform, _stages = _forward_pretrim(gen, x_pre, ref_s, har_trimmed)
        lowered = lower_generator_adain_to_layernorm(gen)
        if lowered != EXPECTED_ADAIN_APPLICATIONS:
            raise SystemExit(f"lowered {lowered} AdaIN1d, expected {EXPECTED_ADAIN_APPLICATIONS}")
        trunk = GeneratorTrunkANE(gen).eval()(x_pre, ref_s, har_trimmed)
        spec, phase = GeneratorBodyANE(gen).eval()(trunk, ref_s, har_trimmed)
        chained_waveform = gen.stft.inverse(spec, phase)

    wave_drift = float((chained_waveform - reference_waveform).abs().max())
    wave_snr = _snr_db(reference_waveform, chained_waveform)
    print(
        f"chained trunk->body (ln): trunk {tuple(trunk.shape)} -> spec {tuple(spec.shape)}; "
        f"waveform max-abs vs pre-trim {wave_drift:.2e} (SNR {wave_snr:.2f} dB)"
    )
    if not torch.allclose(chained_waveform, reference_waveform, atol=LN_ALLCLOSE_ATOL):
        raise SystemExit(
            f"split chain diverged from the manual graph: waveform max-abs {wave_drift:.2e} "
            f"> atol {LN_ALLCLOSE_ATOL:g}. The two halves do not reassemble the graph."
        )
    print(f"✅ allclose: split chain reproduces the pre-trim waveform within atol {LN_ALLCLOSE_ATOL:g}")
    return reference_waveform


def _report_census() -> None:
    """Stage 3. Per-half non-const op count vs the monolithic ln package."""
    print("\n--- MIL op census: monolithic ln vs split halves ---")
    rows = []
    if LN_PACKAGE_PATH.exists():
        c, total = histogram_for_mlpackage(LN_PACKAGE_PATH)
        rows.append(("ln (monolithic)", total - c.get("const", 0), c.get("layer_norm", 0)))
    for name, path in (("trunk", TRUNK_PATH), ("body", BODY_PATH)):
        c, total = histogram_for_mlpackage(path)
        rows.append((name, total - c.get("const", 0), c.get("layer_norm", 0)))

    print(f"{'package':<18}{'non-const':>12}{'layer_norm':>12}")
    print("-" * 42)
    for label, non_const, ln_ops in rows:
        print(f"{label:<18}{non_const:>12}{ln_ops:>12}")
    trunk_nc = next(nc for lbl, nc, _ in rows if lbl == "trunk")
    body_nc = next(nc for lbl, nc, _ in rows if lbl == "body")
    print(f"  largest single split program: {max(trunk_nc, body_nc)} non-const ops")


def _predict(path: Path, feed: dict, compute_unit) -> dict:
    return ct.models.MLModel(str(path), compute_units=compute_unit).predict(feed)


def _chained_parity(
    gen: torch.nn.Module,
    x_pre: torch.Tensor,
    ref_s: torch.Tensor,
    har_trimmed: torch.Tensor,
    reference_waveform: torch.Tensor,
) -> float:
    """Stage 2. trunk package -> body package -> Python iSTFT -> SNR, by compute unit.
    Returns the ``CPU_AND_GPU`` SNR the gate judges.
    """
    reference = reference_waveform.squeeze()[:BUCKET_SAMPLES]
    x_pre_np = x_pre.numpy().astype(np.float32)
    ref_s_np = ref_s.numpy().astype(np.float32)
    har_np = har_trimmed.numpy().astype(np.float32)
    print("\n--- chained parity: trunk pkg -> body pkg + Python iSTFT vs pre-trim, by compute unit ---")
    results: dict[str, tuple[float, bool]] = {}
    for label, compute_unit in (
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ):
        trunk_pred = _predict(TRUNK_PATH, {"x_pre": x_pre_np, "ref_s": ref_s_np, "har": har_np}, compute_unit)
        trunk_np = np.asarray(trunk_pred["trunk"], dtype=np.float32)
        body_pred = _predict(BODY_PATH, {"trunk": trunk_np, "ref_s": ref_s_np, "har": har_np}, compute_unit)
        spec = torch.from_numpy(np.asarray(body_pred["spec"], dtype=np.float32))
        phase = torch.from_numpy(np.asarray(body_pred["phase"], dtype=np.float32))
        with torch.no_grad():
            waveform = gen.stft.inverse(spec, phase).squeeze()[:BUCKET_SAMPLES]
        finite = bool(torch.isfinite(waveform).all())
        snr = _snr_db(reference, waveform) if finite else float("nan")
        results[label] = (snr, finite)
        print(f"  {label:<12} {snr:7.2f} dB" if finite else f"  {label:<12}   NON-FINITE output")

    ne_snr, ne_finite = results["CPU_AND_NE"]
    if not ne_finite or ne_snr < PARITY_SNR_GATE_DB:
        print(
            "\n  ⚠️  CPU_AND_NE does NOT match the graph on this Mac (T4 finding: this Mac's\n"
            "      ANE admits and then miscomputes these graphs). The graph is fine — see\n"
            "      CPU_AND_GPU. The A14 is the judge."
        )
    return results["CPU_AND_GPU"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wrapper-only", action="store_true", help="Run only stage 1 (PyTorch chain allclose).")
    args = parser.parse_args()

    gen, x_pre, ref_s, har_trimmed = _build_real_inputs()
    reference_waveform = _assert_chain_matches_pretrim(gen, x_pre, ref_s, har_trimmed)
    if args.wrapper_only:
        print("\nwrapper-only: skipping Core ML stages. Export now, then re-run without the flag.")
        return

    for path in (TRUNK_PATH, BODY_PATH):
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Export first:\n"
                "  uv run python -m export_synth.main --mode decoder-har-ane-split"
            )

    _report_census()
    snr = _chained_parity(gen, x_pre, ref_s, har_trimmed, reference_waveform)

    if snr < PARITY_SNR_GATE_DB:
        raise SystemExit(f"\n❌ chained parity {snr:.2f} dB is below the {PARITY_SNR_GATE_DB:.0f} dB gate.")
    print(
        f"\n✅ chained parity {snr:.2f} dB >= {PARITY_SNR_GATE_DB:.0f} dB gate: the two halves "
        "reassemble the pre-trim graph in fp16."
    )
    print(
        "   NOT proven here: that either half clears the A14's fvmlib cap. The per-stage census "
        "is the Mac-side proxy; the phone is the judge (plan Phase 1 gate)."
    )


if __name__ == "__main__":
    main()

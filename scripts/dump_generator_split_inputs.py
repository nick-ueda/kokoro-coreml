#!/usr/bin/env python3
"""Dump REAL generator inputs (x_pre / ref_s / har) for the Swift split-parity test.

Task T7 of README/Plans/ane-generator-a14-v1.md. The Swift test
``GeneratorSplitParityTests`` proves the T7 executor's split wiring (trunk ->
body) reproduces the monolithic ``kokoro_decoder_har_ane_ln_3s`` on identical
inputs, under ``CPU_AND_GPU``.

WHY REAL INPUTS, NOT RANDOM
---------------------------
The generator ends in ``spec = exp(conv_post)``, and every AdaIN block
layer-normalizes its activations — so the activation magnitudes are set by the
learned weights and the voice, NOT by the input scale. Random ``x_pre`` drives
``conv_post`` outside fp16 range and ``spec`` overflows to ``inf``, which would
make the finiteness check spurious. Real speech-distributed inputs keep
``conv_post`` in range (T4 measured it -66..+3, ``spec`` max 22.2), so the
CPU_AND_GPU output is finite and the parity number is meaningful. This is
exactly why the Python verifiers (``verify_decoder_har_ane*.py``) use real
inputs; this script reuses their ``_build_real_inputs`` so the fixture is the
same tensor those gates validate.

Writes flat little-endian float32 (no header), one file per tensor, into the
Swift test's resource dir:

    swift/Tests/KokoroPipelineTests/Fixtures/generator_split/{x_pre,ref_s,har}.f32

Shapes (3 s bucket Ground Truth): x_pre (1, 512, 240), ref_s (1, 256),
har (1, 22, 14401). Regenerate with: ``uv run python scripts/dump_generator_split_inputs.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from verify_decoder_har_ane import _build_real_inputs  # noqa: E402

# Ground Truth geometry (README/Plans/ane-generator-a14-v1.md T4/T6). Asserted so
# a geometry regression fails here, not silently in the Swift test.
EXPECTED_SHAPES = {
    "x_pre": (1, 512, 240),
    "ref_s": (1, 256),
    "har": (1, 22, 14401),
}

OUT_DIR = _ROOT / "swift" / "Tests" / "KokoroPipelineTests" / "Fixtures" / "generator_split"


def _write_f32(path: Path, tensor) -> None:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    path.write_bytes(arr.tobytes())
    print(f"  wrote {path.relative_to(_ROOT)}  shape={tuple(arr.shape)}  bytes={arr.nbytes}")


def main() -> None:
    _gen, x_pre, ref_s, har_trimmed = _build_real_inputs()
    tensors = {"x_pre": x_pre, "ref_s": ref_s, "har": har_trimmed}
    for name, tensor in tensors.items():
        got = tuple(int(d) for d in tensor.shape)
        if got != EXPECTED_SHAPES[name]:
            raise SystemExit(f"{name} shape {got} != expected {EXPECTED_SHAPES[name]} (geometry drift)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"dumping real generator inputs -> {OUT_DIR.relative_to(_ROOT)}")
    for name, tensor in tensors.items():
        _write_f32(OUT_DIR / f"{name}.f32", tensor)
    print("done.")


if __name__ == "__main__":
    main()

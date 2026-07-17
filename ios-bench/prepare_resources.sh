#!/bin/bash
# Populate ios-bench/Resources from repo-local artifacts.
#
# Resources/coreml and Resources/mlx are gitignored (hundreds of MB of model
# artifacts); this script makes the checkout buildable. Run it before
# `xcodegen generate` — xcodegen globs Resources/* at generate time, so new
# packages need a re-generate to enter the Xcode resources phase.
#
# Sources:
#   - Core ML packages: ../coreml/ (produced by export_duration.py,
#     export_f0ntrain.py, export_decoder_pre.py, export_synth/ — see
#     README/Wiki/coreml-export.md)
#   - Bench inputs: produced by scripts/prepare_swift_bench_inputs.py
#     (committed — small JSON)
#   - MLX weights (optional, only for the mlx arm): kokoro-v1_0.safetensors
#     and voices.npz from the mlalma/KokoroTestApp distribution; place them
#     in Resources/mlx manually if benchmarking the MLX arm.
set -euo pipefail
cd "$(dirname "$0")"

SRC=../coreml
DST=Resources/coreml
mkdir -p "$DST"

# Runtime buckets shipped in the bench app (10s omitted: no 10s bakeoff input).
BUCKETS=(3 7 15 30)
# Padded duration sizes covering the 44/105/219/476-token bench inputs.
PADDED=(64 128 256 512)
# Exact-native-LSTM duration sizes, one per bench input's true token count
# (--exact-duration path; 780-op graphs vs 17k-134k-op unrolled).
EXACT=(44 105 219 476)
# F0Ntrain frame sizes per bucket (PipelineConstants.tFramesForBucket).
F0N=(120 280 600 1200)

missing=0
copy() {
  local name="$1"
  if [ ! -d "$SRC/$name" ]; then
    echo "MISSING: $SRC/$name" >&2
    missing=1
    return
  fi
  rsync -a --delete "$SRC/$name" "$DST/"
}

for b in "${BUCKETS[@]}"; do
  copy "kokoro_decoder_pre_${b}s.mlpackage"
  copy "kokoro_decoder_har_post_${b}s.mlpackage"
done
for t in "${PADDED[@]}"; do copy "kokoro_duration_t${t}.mlpackage"; done
# Exact-native-LSTM duration packages are produced by export_duration.py and
# are NOT in the HF download set — optional, only the --exact-duration 1
# bench path needs them.
for t in "${EXACT[@]}"; do
  if [ -d "$SRC/kokoro_duration_exact_t${t}.mlpackage" ]; then
    copy "kokoro_duration_exact_t${t}.mlpackage"
  else
    echo "skipping optional kokoro_duration_exact_t${t}.mlpackage (not exported; --exact-duration 1 unavailable)"
  fi
done
for t in "${F0N[@]}"; do copy "kokoro_f0ntrain_t${t}.mlpackage"; done

# T5 (README/Plans/ane-generator-a14-v1.md): ANE-admissible 3s generator
# package (decoder-har-ane export, T4). Optional like the exact-duration
# packages above — skip gracefully if not yet exported; --policy aneGenerator
# just won't be available until it's staged.
if [ -d "$SRC/kokoro_decoder_har_ane_3s.mlpackage" ]; then
  copy "kokoro_decoder_har_ane_3s.mlpackage"
else
  echo "skipping optional kokoro_decoder_har_ane_3s.mlpackage (not exported; --policy aneGenerator unavailable)"
fi

# T6 (README/Plans/ane-generator-a14-v1.md): candidate ANE generator packages
# that shrink the program below the A14's fvmlib object cap. The ln package is
# a drop-in for --policy aneGenerator via --generator-package; the trunk/body
# split halves are single-half compute-plan candidates (--mode computeplan
# --model <name>). All optional — skip gracefully like the T5 package above.
for name in \
  kokoro_decoder_har_ane_ln_3s \
  kokoro_decoder_har_ane_ln_trunk_3s \
  kokoro_decoder_har_ane_ln_body_3s; do
  if [ -d "$SRC/${name}.mlpackage" ]; then
    copy "${name}.mlpackage"
  else
    echo "skipping optional ${name}.mlpackage (not exported; T6 candidate)"
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "Some packages are missing from $SRC — export them first." >&2
  exit 1
fi
echo "Resources/coreml populated. Re-run 'xcodegen generate' before building."

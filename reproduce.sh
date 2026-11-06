#!/usr/bin/env bash
# Build the CPU grader/trainer and run a short smoke grade.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v cmake >/dev/null; then
  echo "cmake required" >&2
  exit 1
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"

ctest --test-dir build --output-on-failure

if [[ ! -x build/rlv2_grade ]]; then
  echo "rlv2_grade not built (need a BLAS: apt install libopenblas-dev on Linux, or Accelerate on macOS)" >&2
  exit 1
fi

export RLV2_DEPTH_PRESET=hd_bm320_120
export RLV2_PRESET_DESPECKLE=1
export RLV2_HFOV=120
export RLV2_MEM_GUARD=1
export RLV2_NARROW=1
export RLV2_BLANK_GUARD=1
export RLV2_THREADS="${RLV2_THREADS:-2}"

mkdir -p build/smoke
echo "Smoke grade: s120hd on public-visible-Simple-8, 2 seeds..."
build/rlv2_grade missions_smoke assets/raptor.bin checkpoints/s120hd.bin student 1 2 \
  build/smoke/s120hd_smoke.csv | tee build/smoke/s120hd_smoke.txt

echo
echo "Done. Full evaluation protocol and tables: docs/"

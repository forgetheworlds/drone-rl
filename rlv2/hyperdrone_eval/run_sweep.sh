#!/bin/bash
# Throughput sweep for HyperDrone on this Mac. Writes JSON lines to results/hd_sweep.jsonl.
# Usage: MISSIONS=/path/to/drone-rl/missions ./run_sweep.sh
set -u
cd "$(dirname "$0")"
M=${MISSIONS:?set MISSIONS to drone-rl/missions}
OUT=results/hd_sweep.jsonl
mkdir -p results
VOXC="voxel:$M/public/public-visible-Complex-3.task"
VOXF="voxel:$M/source/furniture-280000.task"
run() { python bench_hyperdrone.py --seconds 5 "$@" 2>/dev/null | grep '^{' | tee -a $OUT; }
# 1. paper shape: 64x64, 4096 envs, render-only and with physics
for s in simple20 procthor "$VOXC" "$VOXF"; do
  for o in depth rgb; do
    for m in render physics; do run --scene "$s" --output $o --width 64 --height 64 --envs 4096 --mode $m; done
  done
done
# paper fidelity levels on ProcTHOR RGB
for f in medium high; do run --scene procthor --output rgb --width 64 --height 64 --envs 4096 --mode physics --fidelity $f; done
# 2. our shape: 20x16 = 320 rays, our 50 ms tick (5 L2F substeps per frame), env sweep
for s in "$VOXC" "$VOXF" procthor simple20; do
  for n in 1024 4096 16384; do run --scene "$s" --output depth --width 20 --height 16 --envs $n --mode physics50; done
done
run --scene "$VOXC" --output depth --width 20 --height 16 --envs 4096 --mode render
run --scene "$VOXC" --output depth --width 20 --height 16 --envs 4096 --mode physics50 --readback
# 3. stereo-camera shape: RGB 320x240 (OV2640 QVGA), 1024 envs (one frame per camera)
for s in procthor "$VOXC"; do
  run --scene "$s" --output rgb --width 320 --height 240 --envs 1024 --mode physics50 --fov 66
  run --scene "$s" --output depth --width 320 --height 240 --envs 1024 --mode physics50 --fov 66
done

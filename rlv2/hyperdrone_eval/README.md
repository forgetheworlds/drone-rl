# HyperDrone evaluation scripts

These scripts use HyperDrone/HyperLux as pip packages only (MIT, github.com/rl-tools/hyperdrone).
No HyperDrone code or scenes are vendored here. Results and write-up: docs/HYPERDRONE.md.

```
uv venv -p 3.12 && source .venv/bin/activate
uv pip install hyperdrone opencv-python-headless trimesh
MISSIONS=/path/to/drone-rl/missions ./run_sweep.sh                    # throughput -> results/hd_sweep.jsonl
python depth_parity.py $MISSIONS rlv2_reference_output.bin            # depth parity vs our ray caster
python probe_clearance.py $MISSIONS                                   # probe rays vs exact clearance
python stereo_sgbm.py --scene procthor --fidelity high                # RGB stereo + SGBM vs sensors.h
```

- `voxmap.py`: a numpy port of our ray caster. Also has the voxel-to-.glb/HyperLux export and the
  camera basis that reproduces our 320 rays.
- `bench_hyperdrone.py`: one throughput configuration per run.
- `results/`: the Mac M3 numbers, including `ours_*_bench.csv` from `rlv2 bench` and `rlv2_train bench`.

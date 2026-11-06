# HyperDrone eval

Scripts in `rlv2/hyperdrone_eval/` use HyperDrone / HyperLux as **pip packages only** ([rl-tools/hyperdrone](https://github.com/rl-tools/hyperdrone), MIT). No HyperDrone source or scenes are vendored.

Closed-loop: HyperLux RGB stereo from ProcTHOR → optional OV2640-style effects → OpenCV StereoBM → same student + RAPTOR as `rlv2_grade`.

| Run (`s120hd`, best camera profile) | Completion | Contacts | Flights |
|---|---:|---:|---:|
| ProcTHOR-Train-1, real stereo 320 px | **0.708** | 24 | 120 |
| Held-out ProcTHOR-26, real stereo | **0.700** | 14 | 60 |
| Same scenes, calibrated `hd_bm320_120` preset | 0.783 / 0.867 | 20 / 7 | 120 / 60 |

The preset is optimistic vs real stereo. Harsh camera profiles (large sync skew, dim light, heavy JPEG) collapse completion — see `rlv2/hyperdrone_eval/results/`.

```sh
uv venv -p 3.12 && source .venv/bin/activate
uv pip install hyperdrone opencv-python-headless trimesh
# sweeps: rlv2/hyperdrone_eval/README.md
```

ProcTHOR voxel exports used for those runs are not shipped in this repo.

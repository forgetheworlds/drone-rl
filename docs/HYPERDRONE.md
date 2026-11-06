# HyperDrone evaluation

Scripts under `rlv2/hyperdrone_eval/` use **HyperDrone / HyperLux as pip packages only**
(MIT, [rl-tools/hyperdrone](https://github.com/rl-tools/hyperdrone)). No HyperDrone source
or scenes are vendored in this repository.

## What was measured

Closed-loop flights: HyperLux renders left/right RGB from ProcTHOR rooms, optional OV2640-style
camera effects (sync skew, misalignment, JPEG, dim light), OpenCV StereoBM depth, then the
same student policy + RAPTOR stack used in `rlv2_grade`.

| Run (s120hd, best camera profile) | Completion | Contacts | Flights |
|---|---:|---:|---:|
| ProcTHOR-Train-1, real stereo 320 px | **.708** | 24 | 120 |
| Held-out ProcTHOR-26, real stereo | **.700** | 14 | 60 |
| Same scenes, calibrated `hd_bm320_120` preset (no RGB) | .783 / .867 | 20 / 7 | 120 / 60 |

The calibrated preset is optimistic relative to real stereo (~.78 vs .71 on ProcTHOR-1). Round-7
blank-guard A/B on ProcTHOR used the preset for speed; treat those as upper bounds.

## Camera-effect profiles

- **Best:** sync ≤2 ms, rectification ≤0.2 px, matched exposure, JPEG quality 85–95.
- **Harsh:** 0–20 ms sync offset, 0.3–0.5 px misalignment, dim light, JPEG quality 55–80
  (completion collapses; see `rlv2/hyperdrone_eval/results/`).

## Reproduce (optional)

```sh
uv venv -p 3.12 && source .venv/bin/activate
uv pip install hyperdrone opencv-python-headless trimesh
# See rlv2/hyperdrone_eval/README.md for sweep / closed-loop scripts.
```

Requires a missions tree with ProcTHOR voxel exports; those assets are **not** shipped here.

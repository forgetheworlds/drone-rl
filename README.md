# drone-rl

Stereo-depth indoor drone navigation with a learned policy, RAPTOR flight control, and
honest simulated evaluation under camera noise.

Waypoints come from an upstream planner (in deployment: a VLM looking at the camera). The
onboard stack’s job is to reach each waypoint quickly without contacts, using only depth,
ego state, and the waypoint — no prior map.

```text
Stereo depth + ego state + waypoint
        ↓
Local map + student policy (MLP) + blank-view / memory / narrow-gap guards
        ↓
Body velocity + yaw rate @ 20 Hz
        ↓
RAPTOR neural controller @ 100 Hz → motors
```

This repository replaces an earlier map-and-route prototype. That path-based planner worked
in clean depth and failed under realistic stereo holes and textureless walls. The v2 stack
here trains a student against a privileged teacher (DAgger) in a voxel simulator, then
grades it with calibrated stereo-BM noise and (separately) HyperDrone RGB stereo.

## Results (simulated)

All numbers below are **simulated**. Nothing here is measured on hardware yet.

**Round 6** — student `s120hd` (120° FOV), memory guard + narrow-gap fallback, depth preset
`hd_bm320_120` with despeckle. Completion / contacts:

| Set | Completion | Contacts | Flights |
|---|---:|---:|---:|
| HyperDrone ProcTHOR-1 (real stereo + OV2640 effects) | **.708** | 24 | 120 |
| Held-out ProcTHOR-26 (real stereo) | **.700** | 14 | 60 |
| Original 63 missions (hd preset) | **.672** | 154 | 504 |

**Round 7** — same student with a **blank-view guard** (when most of the forward sector
returns no depth, slow to 0.3 m/s and yaw to find texture). Original missions, 1008 flights:

| Config | Completion | Contacts |
|---|---:|---:|
| Baseline (mem + narrow) | .680 | 302 |
| **+ blank guard** | **.715** | **255** |

Main remaining failure modes: textureless surfaces that stereo never measures, obstacles
beside / behind the drone, and low blockers (counters, people) that need an early climb or
side-step. Fly-under-table missions are excluded from headline scores; they are a missing
skill, not a perception bug.

Details: [docs/RESULTS.md](docs/RESULTS.md), [docs/FAILURES.md](docs/FAILURES.md),
[docs/BLANK_GUARD.md](docs/BLANK_GUARD.md), [docs/HYPERDRONE.md](docs/HYPERDRONE.md),
[docs/DEPTH.md](docs/DEPTH.md).

## Checkpoints

| File | Role |
|---|---|
| [`checkpoints/s120hd.bin`](checkpoints/s120hd.bin) | Best overall 120° student |
| [`checkpoints/s90hd.bin`](checkpoints/s90hd.bin) | 90° student (stronger on small held-out set) |

See [checkpoints/README.md](checkpoints/README.md).

## Quick start

Linux needs a BLAS (`sudo apt install cmake g++ libopenblas-dev`). macOS uses Accelerate.

```sh
./reproduce.sh
```

This builds `rlv2_grade` / `rlv2_train_cpu`, runs the RAPTOR runtime test, and grades
`s120hd` for two seeds on one public mission.

Grade a full portfolio (slow):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
RLV2_DEPTH_PRESET=hd_bm320_120 RLV2_PRESET_DESPECKLE=1 RLV2_HFOV=120 \
RLV2_MEM_GUARD=1 RLV2_NARROW=1 RLV2_BLANK_GUARD=1 RLV2_THREADS=4 \
./build/rlv2_grade missions assets/raptor.bin checkpoints/s120hd.bin student 1 8 out.csv
```

Train a teacher then a student on CPU (gen3d mix; see `rlv2/train_cpu.cpp`):

```sh
./build/rlv2_train_cpu assets/raptor.bin 64 40
./build/rlv2_train_cpu assets/raptor.bin 64 60 256 teacher_cpu.bin student_out
```

## Layout

| Path | What |
|---|---|
| `rlv2/` | CPU training (`train_cpu`), grading (`grade`), depth presets, guards, gen3d missions |
| `checkpoints/` | Released student weights |
| `include/navigation/` | RAPTOR adapter + voxel world used by the grader |
| `missions/` | Evaluation missions (source / public / fresh) |
| `docs/` | Depth research, HyperDrone eval, failure breakdown, blank-guard A/B |
| `physics.hpp` / `raptor.hpp` | L2F dynamics + RAPTOR policy (also under `include/`) |

macOS Metal high-throughput training lives in `rlv2/rlv2.mm`, `train.mm`, and the `.metal`
shaders. It is **not required** to grade the released checkpoints on Linux. See
[docs/MAC_METAL.md](docs/MAC_METAL.md).

## Hardware target (in progress)

Intended deployment: ESP32-S3 flight controller, dual OV2640 (or OV5647 via a MIPI host),
onboard stereo depth + student policy, VLM waypoints off-board. Firmware and board selection
are not finished in this tree; the policy and stereo noise model are the sim-to-real prep.

## Licence

MIT. RAPTOR weights and L2F dynamics retain their upstream MIT notices in
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

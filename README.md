# drone-rl

**Learned collision-free flight to a camera-visible waypoint**, for a cheap indoor drone that only has stereo cameras, IMU, and an onboard policy.

A VLM (or any upstream planner) picks the next waypoint that is already in the camera image. The drone’s job is just: get there fast, don’t hit anything. No map of the room. No depth camera — only stereo that looks like real cheap OV cameras.

<p align="center">
  <img src="docs/assets/furniture-moving.gif" width="860" alt="Drone flying past furniture and a moving obstacle while a voxel map fills in"><br>
  <em>Same indoor sim language: voxel rooms, depth inset, waypoints. Furniture + moving obstacle mission.</em>
</p>

<p align="center">
  <img src="docs/assets/staggered-vertical.gif" width="420" alt="Staggered walls mission">
  <img src="docs/assets/warframe-complex.gif" width="420" alt="Warframe Moving AI voxel map"><br>
  <em>Left: staggered walls. Right: a Warframe level from the Moving AI voxel benchmarks.</em>
</p>

<p align="center">
  <img src="docs/assets/gap-overhang.gif" width="520" alt="Gap and overhang mission"><br>
  <em>Gap and overhang — the depth inset is the 16×20 view the navigator actually gets.</em>
</p>

Sim-to-real prep for an under-$50 build (ESP32-S3 flight control, dual cheap cameras, stereo + policy onboard; VLM off-board). Firmware is not in this tree yet. **All numbers below are simulated.**

<p align="center">
  <img src="docs/assets/stack.png" width="900" alt="Onboard stack: stereo → policy → blank-view guard → RAPTOR">
</p>

---

## Problem

I first built a local path planner over a voxel map. On clean depth it looked great (including zero-contact runs). Under realistic stereo — blank walls, JPEG, narrow FOV — the depth image is full of holes, the planner treats holes as free space, and it flies into walls it never measured.

So the policy now aims **straight at the waypoint** as if unseen space is open, and only reacts when nearby depth confirms an obstacle. Far-away unknown cells don’t matter. Not hitting what’s close does.

The old planner is under [`archive/path_planner/`](archive/path_planner/) and is not built by default. The flight gifs above are from that sim visual language (map fills as the camera sees).

## Approach

| Piece | Choice |
|---|---|
| **Learning** | Privileged teacher (PPO on true geometry) → onboard student via **DAgger**, then optional student PPO |
| **Sensing** | **Stereo on purpose** — matches the real hardware. StereoBM-style noise in training; HyperDrone RGB stereo for headline checks |
| **Reward** | Get closer to the waypoint each tick, small time cost, big penalty for contact. No explore bonus |
| **Randomization** | Procedural rooms (gaps, pillars, slabs, 3D goals through doors/windows) + FOV/depth noise. Guards on top: blank-view, short memory, narrow-gap. “Remember holes as obstacles” made things worse — off |
| **Dropped** | Fly-under-table missions; claiming the old clean-depth planner scores |

Released weights: [`s120hd.bin`](checkpoints/s120hd.bin) (best overall), [`s90hd.bin`](checkpoints/s90hd.bin).

## What worked

When most of the view ahead has **no depth** (blank wall / hole), the blank-view guard slows the drone and turns until texture returns. Current simulated headline numbers for `s120hd`:

<p align="center">
  <img src="docs/assets/results-now.png" width="900" alt="Current simulated completion rates and blank-guard contact A/B">
</p>

Under real HyperDrone stereo, `s120hd` also completes most of ProcTHOR-1 and a held-out ProcTHOR-26 split ([docs/RESULTS.md](docs/RESULTS.md)). Remaining hits are mostly still unmeasured surfaces and side/low blockers ([docs/FAILURES.md](docs/FAILURES.md)).

## Quick start

```sh
# Linux: sudo apt install cmake g++ libopenblas-dev
./reproduce.sh
```

Builds grade/train binaries, runs a RAPTOR + depth smoke test, grades `s120hd` on one public mission. Details: `rlv2/grade.cpp`, `rlv2/train_cpu.cpp`, [`docs/`](docs/).

## Licence

MIT. RAPTOR / L2F notices: [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

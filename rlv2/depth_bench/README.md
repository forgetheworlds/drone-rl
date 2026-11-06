# depth_bench — planner-facing depth benchmark on HyperDrone renders

Scores any depth pipeline for the $50 ESP32 micro-drone (two OV2640s, 6 cm baseline) against ground
truth, with the metrics a planner cares about: is a ray's free/occupied/unknown answer right, and does
the fused voxel map hide obstacles near the drone. Goal: improve the inputs until they are accurate, then
fit a *residual* noise model for RL training (`docs/DEPTH.md`).

Results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md) (test split) and
[`results/LEADERBOARD_dev.md`](results/LEADERBOARD_dev.md) (dev split, used for tuning).

## Dataset v1 (fixed, versioned, not in git)

| | |
|---|---|
| images | 320x240, HFOV 66° (f = 246.4 px), left + right at 6 cm and 12 cm baselines |
| sequences | 8 frames at 10 Hz; per scene 8 sequences alternating *cruise* (0.8–1.5 m/s, yaw up to ±0.7 rad/s) and *approach* (fly at a wall from 1.0–1.7 m down to 0.35 m) |
| GT | planar depth, surface class (floor/wall/ceiling/other), thin-object mask, 4x4 pose |
| dev split | ProcTHOR-Train-1, -2; voxel courses furniture-280000, overhang-282000 |
| test split (held out) | ProcTHOR-Train-150..153; fresh-dense-00, narrow_gap-281000, gap_overhang-284000 |
| nav split (held out) | ProcTHOR-Train-154..159; fresh-dense-00, narrow_gap-281000, gap_overhang-284000. A *navigation view*: starts only where >2.2 m is clear straight ahead, flies 0.8–1.5 m/s with gentle yaw (±0.3 rad/s), never aims at a wall; thin poles/rods placed 0.6–2 m beyond the 1.05 m travelled. 72 sequences, checksums in `data/SHA1SUMS_nav` |
| hard cases | textureless walls (voxel courses use one flat colour per wall orientation), thin poles 2–4 cm and rods 1–2 cm in every sequence, approaches below 1 m, low light (40 ms exposure), motion blur, rolling shutter with 25 Hz vibration |

The rolling shutter (30 ms top→bottom readout) and motion blur are rendered geometrically from 15
sub-frames, not faked in 2D. 11 `.npz` files, ~380 MB total, checksums in `data/SHA1SUMS_v1`.

Generate (needs `hyperdrone`/`hyperlux` with a GPU backend and the `missions/` voxel courses;
ProcTHOR houses download automatically; ~10 s per scene on an M3):

```bash
MISSIONS=/path/to/missions python gen_dataset.py --out data/v1 --split dev
MISSIONS=/path/to/missions python gen_dataset.py --out data/v1 --split test
cd data/v1 && sha1sum -c ../SHA1SUMS_v1   # GPU float differences may change bytes, not results
export DEPTH_BENCH_DATA=$PWD
```

### Camera realism profiles (`dataset.py`, applied on load, seeded and deterministic)

| profile | what |
|---|---|
| `clean` | global shutter, synced, perfect calibration (upper bound) |
| `ov2640` (**default**) | rolling shutter + 10 ms blur, Poisson shot noise (900 e⁻ full well, 4 e⁻ read noise), per-sequence right-camera auto-exposure gain ±15 %, residual rectification error (0.4 px vertical + ≤0.1° roll), JPEG q50 |
| `ov2640_dark` | as ov2640, but 40 ms exposure, 5x less light and 5x gain |
| `ov2640_unsync` | as ov2640, right camera starts 12 ms late (free-running pair, no frame sync) |
| `ov2640_dots` | as ov2640, plus an IR random-dot projector (~5 k dots, co-located with the left camera, brightness ∝ 1/z²) seen by cameras without IR-cut filters. Reference only: it ignores dot occlusion and is optimistic past ~3 m |

Methods get **noisy VO poses** (random walk of 1 cm and 0.3° per frame) and the nominal calibration;
scoring uses GT.

## Plugin interface (`methods.py`)

```python
from methods import Method, register

@register
class MyDepth(Method):
    name = "my_depth"
    baseline = 0.06            # or 0.12
    note = "one-line description for the leaderboard"

    def reset(self, calib):    # new sequence; calib = dict(f, cx, cy, baseline, W, H, dt)
        ...

    def process(self, frame, calib):
        # frame = dict(left, right (HxWx3 uint8 RGB, rectified), pose (4x4 cam->world VO), k)
        return dict(depth=z)   # HxW float32 planar metres, NaN = unknown; any resolution (resized nearest)
```

Run it: `python run_bench.py --plugin my_file.py --methods my_depth --split dev --profile ov2640`.

## Metrics (`metrics.py`)

- **Pixel**: tolerance `max(0.10 m, 0.15 z)`. Hole rate; false-near (estimate nearer than GT, i.e.
  a phantom obstacle); false-far; validity and accuracy per range bin (0.2–1, 1–2, 2–4, 4–8 m); hole and
  ok rate per surface (textureless / textured / depth edge / thin / floor / wall / ceiling).
- **Ray** (what the policy sees): 16x12 rays, each a 20x20 px block; ray depth = 5th percentile of the
  block. **False-free** = ray claims more free space than GT (the dangerous error; reported overall
  and for GT < 1.5 m), **false-occupied**, **unknown** (<10 % of the block valid), free coverage,
  thin-object ray detection.
- **Voxel map**: per sequence, fuse with the method's VO poses into a 0.125 m log-odds grid; compare
  with GT depth fused with GT poses within 3 m of the trajectory. Occupied and free IoU, occupied
  recall, false-free voxels within 1 m of the path (count per sequence), unknown and false-occupied
  near the path.
- **Noise model**: `k` in σ = k·z² (robust MAD of relative error / z on valid pixels).
- **Runtime**: single-thread box ms, scaled to Pi Zero 2 W (10x) / Pi Zero v1 (40x) / ESP32-P4 core
  (30x). Rough estimates, not measurements.
- **planner_cost** = `100·(10·FF<1.5m + 3·false-occ + unknown)`: the legacy sort key.
- **Navigation metrics** (`nav_frame_metrics`, the headline since the nav split): same ray blocks, but
  stop metrics count only obstacle rays (block's nearest surface is not floor/ceiling). Methods with
  a range cap report pixels past it as *far* (matched, free up to the cap), not unknown.
  `stop{v}_ff` / `stop{v}_unknown`: false-free / unknown rays whose obstacle is inside the stopping
  distance `d_stop(v) = 0.3 + 0.4v + v²/(2·1.5)` (1.03 m at 1 m/s, 2.43 m at 2 m/s).
  Unknown by cause: `unk_textureless` (≥50 % of the block textureless in the clean image),
  `unk_beyond_range` (GT past the 3 m cap), `unk_occluded_other`; and unknown rate per GT band
  (`unk_band_0-1` … `unk_band_5-99`). `nav_fo_3m`: false-occupied within 3 m.
  **stop_cost** = `100·(10·FF<1.5m + 10·stop1_ff + stop1_unknown + 3·nav_fo_3m)`: the sort key.

## Commands

```bash
python run_bench.py --methods sgbm320,bm160_lrc --split test --profile ov2640,clean,ov2640_dark
python leaderboard.py              # -> results/LEADERBOARD.md
python leaderboard.py --split dev  # -> results/LEADERBOARD_dev.md
python leaderboard.py --split nav  # -> results/LEADERBOARD_nav.md (headline: stop_cost)
python tune.py ov2640              # dev-split grid over the filtered SGBM family
```

Tune on `dev` only; `test` is for the leaderboard.

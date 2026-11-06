# Depth sensing research (summary)

Goal: cheap stereo depth that is honest about holes, then train the policy against that noise.

## Findings that shaped training

1. **Textureless walls dominate failures.** After JPEG + camera effects, ~75% of pixels can
   have no disparity (vs ~48% without effects). Most contacts happen where stereo returned
   depth on only a few percent of the rays aimed at the obstacle within 0.8 m.
2. **StereoBM at 160–320 px with a left–right check** was the practical onboard choice for
   ESP-class hardware; SGBM is heavier for little gain at this resolution.
3. **Presets** `hd_bm160_90`, `hd_bm320_90`, `hd_bm320_120` approximate HyperDrone StereoBM
   hole rates and outliers so CPU grading does not need a GPU renderer every run.
4. **Despeckle** removes isolated disparity speckles that otherwise look like free corridors.

Full method tables and texture studies: `rlv2/depth_bench/` (leaderboard markdown retained;
large raw dumps omitted from this export).

## Transfer habit

Behaviours that only use quantities the real drone has:

- Fraction of forward rays with valid depth (blank-view guard).
- Short-term occupancy from rays that *did* return depth (memory guard).
- Narrow-gap route fallback when the local map shows a door/window-scale opening.

Behaviours that remember “depth holes” as obstacles were tried and **hurt** completion
(see [BLANK_GUARD.md](BLANK_GUARD.md)); they are off by default.

# Depth

Cheap stereo that is honest about holes, then train against that noise.

**Findings that shaped training**

1. After JPEG + OV-class camera effects, ~75% of pixels can have no disparity (vs ~48% without). Most contacts land where near-field rays on the obstacle were almost empty.
2. StereoBM at 160–320 px with a left–right check fits ESP-class budgets; SGBM adds little at that resolution.
3. Presets `hd_bm160_90`, `hd_bm320_90`, `hd_bm320_120` match HyperDrone StereoBM hole rates so CPU grading need not re-render RGB every run.
4. Despeckle removes isolated disparity speckles that look like free corridors.

Method tables: `rlv2/depth_bench/`.

**Transfer rule:** only use quantities the real drone has (valid-depth fraction, short occupancy from measured rays, narrow-gap geometry). Remembering holes as obstacles was tried and hurt completion — off by default ([BLANK_GUARD.md](BLANK_GUARD.md)).

# Blank-view guard

When most of the forward sector returns no depth, cap speed at 0.3 m/s and yaw until texture appears.

Baseline for A/B: `s120hd` + memory + narrow, preset `hd_bm320_120` + despeckle, 120° HFOV. Under-table missions excluded from aggregates.

| Config | Set | Flights | Completion | Contacts | Timeouts | s / wp |
|---|---|---:|---:|---:|---:|---:|
| base | orig | 1008 | 0.680 | 302 | 21 | 9.1 |
| **blank** | orig | 1008 | **0.715** | **255** | 32 | 10.4 |
| hole-memory | orig | 1008 | 0.660 | 288 | 55 | 11.9 |
| persist escape | orig | 1008 | 0.671 | 306 | 26 | 9.0 |
| base | m3d | 1152 | 0.660 | 359 | 33 | 10.6 |
| **blank** | m3d | 1152 | **0.694** | **295** | 57 | 11.9 |

Blank helps everywhere. Hole-memory hurts. Persist cuts some timeouts but not contacts. Raw: `rlv2/results/r7/summary.md`.

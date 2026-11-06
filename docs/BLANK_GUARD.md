# Round 7: blank guard A/B

Baseline: `s120hd` + `RLV2_MEM_GUARD=1` + `RLV2_NARROW=1`, preset `hd_bm320_120` + despeckle,
HFOV 120°, estimated attitude, `RLV2_DIAG=1`.

Under-table gen3d kinds are **excluded** from headline aggregates.

## Flags tested

| Flag | Behaviour |
|---|---|
| `RLV2_BLANK_GUARD` | Mostly-blank forward sector → cap speed 0.3 m/s + yaw dither to find texture |
| `RLV2_HOLE_MEM` | Remember blank rays as near-field suspects (20-frame decay) |
| `RLV2_PERSIST` | No progress for 3 s → 1.5 s climb/slide/drop escape |

## Results (simulated)

| Config | Set | Flights | Completion | Contacts | Timeouts | s / wp |
|---|---|---:|---:|---:|---:|---:|
| base | orig | 1008 | 0.680 | 302 | 21 | 9.1 |
| **blank** | orig | 1008 | **0.715** | **255** | 32 | 10.4 |
| hole | orig | 1008 | 0.660 | 288 | 55 | 11.9 |
| persist | orig | 1008 | 0.671 | 306 | 26 | 9.0 |
| base | m3d | 1152 | 0.660 | 359 | 33 | 10.6 |
| **blank** | m3d | 1152 | **0.694** | **295** | 57 | 11.9 |
| base | pt1 (preset) | 120 | 0.783 | 20 | 6 | 9.5 |
| **blank** | pt1 | 120 | **0.833** | **15** | 5 | 10.4 |
| base | h26 (preset) | 60 | 0.867 | 7 | 1 | 8.9 |
| **blank** | h26 | 60 | **0.900** | **6** | 0 | 10.6 |

**Takeaway:** blank guard helps everywhere (fewer contacts, higher completion) at ~1.3 s
extra per waypoint. Hole-memory hurts. Persist cuts some timeouts but not contacts.

Raw tables: `rlv2/results/r7/summary.md`.

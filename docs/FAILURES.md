# Failures

Simulated failure taxonomy for `s120hd` + memory + narrow (round 6), from `RLV2_DIAG=1` grader side-files.

| Rank | Cause |
|---:|---|
| 1 | In-view depth holes / textureless surfaces |
| 2 | Side / behind obstacles |
| 3 | Low blockers (late climb or slide) |
| 4 | Floor / ceiling / thin diagonal edges |
| 5 | Timeouts |

Real HyperDrone stereo outcomes (done / contact / timeout):

| Student | ProcTHOR-1 | Held-out h26 |
|---|---|---|
| s120hd | 85 / 24 / 11 | 42 / 14 / 4 |
| s90hd | 81 / 27 / 12 | 49 / 10 / 1 |
| s4 | 79 / 32 / 9 | 41 / 15 / 4 |

Scripts and per-cause CSVs: `rlv2/results/r6/failures/`, `rlv2/hyperdrone_eval/failure_breakdown.py`.

**Caveat:** per-cause rows for ProcTHOR under real stereo were not fully re-logged; cause totals for those rooms use the calibrated `hd` preset on the same missions (optimistic vs real stereo). Outcome counts above are from closed-loop CSVs.

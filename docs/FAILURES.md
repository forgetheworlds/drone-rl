# Failure breakdown (round 6)

Every failed flight of `s120hd` + memory + narrow (and comparison students), classified from
`RLV2_DIAG=1` grader side-files. **Simulated only.**

## Top causes (s120hd, all graded sets combined)

1. **In-view depth holes / textureless surfaces** — obstacle ahead, but stereo returned almost
   no valid depth on it (often plain walls). Largest bucket.
2. **Side / behind obstacles** — outside the narrow FOV or never painted into the local map.
3. **Low blockers** — counters / people; climb or side-step starts too late.
4. **Floor / ceiling / thin diagonal edges** — clearance mistakes at openings.
5. **Timeouts** — stopped short of the waypoint, or hovered at an obstacle the shield would
   not cross.

Only a small minority of contacts were “seen clearly and still hit” (in-view with ≥50% of
near rays measured). Perception holes, not control alone, dominate.

## Real-stereo outcomes (round 6)

| Student | ProcTHOR-1 done/contact/timeout | h26 done/contact/timeout |
|---|---|---|
| s120hd | 85 / 24 / 11 | 42 / 14 / 4 |
| s90hd | 81 / 27 / 12 | 49 / 10 / 1 |
| s4 | 79 / 32 / 9 | 41 / 15 / 4 |

Per-cause tables and scripts: `rlv2/results/r6/failures/`,
`rlv2/hyperdrone_eval/failure_breakdown.py`.

## Caveat

Per-cause rows for ProcTHOR under **real** stereo + logging were not fully re-run in round 6
(tooling block). Cause totals for those rooms use the calibrated hd preset on the same
missions (optimistic vs real stereo). Outcome counts (done / contact / timeout) for real
stereo come from the closed-loop CSVs.

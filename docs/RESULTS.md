# Results

Everything here is **simulated**. No hardware flights are claimed.

## Headline

| Eval | Setup | Completion | Contacts | Flights |
|---|---|---:|---:|---:|
| ProcTHOR-1 | `s120hd`, real HyperDrone stereo | **0.708** | 24 | 120 |
| Held-out ProcTHOR-26 | same | **0.700** | 14 | 60 |
| Original missions | `hd` preset + blank guard | **0.715** | 255 | 1008 |
| Same, no blank guard | memory + narrow only | 0.680 | 302 | 1008 |

Blank-view guard is the clearest controlled win: higher completion, fewer contacts, ~1.3 s slower per waypoint. Hole-memory and a “persist escape” were worse or mixed — left off. Full A/B: [BLANK_GUARD.md](BLANK_GUARD.md).

HyperDrone closed-loop detail: [HYPERDRONE.md](HYPERDRONE.md). Depth presets and hole rates: [DEPTH.md](DEPTH.md).

## Failure modes

Ranked causes for `s120hd` failures:

1. Textureless / in-view depth holes (stereo never measured the obstacle)
2. Side or behind obstacles outside FOV / local map
3. Low blockers (counters, people) climbed or slid too late
4. Floor / ceiling / thin edges at openings
5. Timeouts

Almost none are “saw it clearly and still hit.” Breakdown: [FAILURES.md](FAILURES.md).

## Not claimed

- Zero-contact flight in real rooms
- The old path planner’s clean-depth “0 contacts” score (it fails under stereo holes; code in `archive/path_planner/`)
- Fly-under-table competence

# Results index

All figures are **simulated** unless a section explicitly says otherwise. No hardware flights
are claimed in this export.

## Headline (job-app)

- Round 6 `s120hd` + memory + narrow, real HyperDrone stereo: ProcTHOR-1 **.708 / 24 contacts**
  (120 flights); held-out h26 **.700 / 14** (60); original missions hd preset **.672 / 154** (504).
- Round 7 blank guard on original missions: **.715 / 255** vs baseline **.680 / 302** (1008 flights).

## Where to look

| Doc / path | Content |
|---|---|
| [HYPERDRONE.md](HYPERDRONE.md) | Real stereo closed-loop summary |
| [FAILURES.md](FAILURES.md) | Cause taxonomy |
| [BLANK_GUARD.md](BLANK_GUARD.md) | Round 7 A/B |
| [DEPTH.md](DEPTH.md) | Why presets and holes matter |
| `rlv2/results/r6/` | Grade logs and failure notes |
| `rlv2/results/r7/` | Blank / hole / persist tables |
| `rlv2/hyperdrone_eval/results/` | Closed-loop markdown tables |

## What we do **not** claim

- Zero-contact flight in real rooms.
- The old path-planner “0 contacts on source missions” story (clean depth; does not hold under
  stereo holes).
- Fly-under-table competence (excluded from headlines).

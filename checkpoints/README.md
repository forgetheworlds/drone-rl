# Checkpoints

DAgger students (light PPO fine-tune) against a privileged teacher. Training depth used calibrated StereoBM presets matched to HyperDrone RGB stereo with OV-class camera effects.

| File | FOV | Notes |
|---|---|---|
| [`s120hd.bin`](s120hd.bin) | 120° | Best overall (ProcTHOR-1 + original missions) |
| [`s90hd.bin`](s90hd.bin) | 90° | Stronger on the small held-out ProcTHOR-26 set |

Obs: depth rays + ego + waypoint. Act: body velocity + yaw rate → RAPTOR @ 100 Hz.

Research / portfolio weights only — not flight-certified.

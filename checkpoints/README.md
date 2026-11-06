# Checkpoints

Student policies trained with DAgger (then light PPO fine-tune) against a privileged teacher
in the CPU voxel simulator. Depth during training used calibrated stereo-BM presets that
match HyperDrone RGB stereo with OV2640-style camera effects (JPEG, sync skew, exposure).

| File | FOV | Guards used when graded | Notes |
|---|---|---|---|
| `s120hd.bin` | 120° horizontal | memory + narrow (+ blank guard in round 7) | Best overall on ProcTHOR-1 and original missions |
| `s90hd.bin` | 90° horizontal | (graded with matching preset) | Stronger on the small held-out ProcTHOR-26 set |

Observation: depth rays + ego state + waypoint in body frame. Action: body-frame velocity
and yaw rate, tracked by the RAPTOR neural flight controller at 100 Hz.

These weights are for research and portfolio demonstration. They are **not** flight-certified.

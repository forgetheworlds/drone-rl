# Archived: observed-space path planner

This is the earlier map-and-route stack (A* over a rolling voxel map, certified stopping, view planning).

It scored well on **clean** depth and failed under realistic stereo holes (textureless walls, JPEG). The live tree uses the stereo RL student instead. Kept here only so the design is recoverable — it is **not** built or linked by the default targets.

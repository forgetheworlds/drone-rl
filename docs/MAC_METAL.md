# macOS Metal targets (optional)

These sources are included so the training stack is complete, but they **require Apple Metal**
and do not build on Linux:

| Target | Sources |
|---|---|
| `rlv2_metal` | `rlv2/rlv2.mm`, `rlv2/nav_env.metal`, … |
| `rlv2_train_metal` | `rlv2/train.mm`, `rlv2/train_env.metal`, … |

Root `CMakeLists.txt` only enables them when `APPLE` is set. Released checkpoints were trained
with this Metal path (or CPU `rlv2_train_cpu` for smaller runs) and can be **graded on Linux**
with `rlv2_grade` + OpenBLAS.

`rlv2/cpu/metal_cpu.h` is a CPU stand-in used by `train_cpu` / `grade` — it is not Metal.

"""Depth parity: HyperLux rendering of our voxel courses vs our ray caster.

For every live tick in an rlv2_reference recording (poses + the depth our Metal/C++ sim produced),
render the same 320 rays with HyperLux (a 20 x 16 camera whose pixel-centre rays are exactly ours,
voxmap.ray_grid_basis), convert planar depth to range, clamp to 12 m and compare.

  python depth_parity.py MISSIONS_DIR REF.bin [--glb N]

Movers are excluded from the comparison (rays whose reference hit is a mover are masked using the
numpy port), because the static scene is built once per course. --glb N also round-trips the first N
courses through an exported .glb (hyperlux.load_scene) to check the file path gives the same depth.
"""
import argparse
import json
import os
import tempfile

import numpy as np

import hyperlux
from hyperlux.camera import CameraSpec, Depth

import voxmap

p = argparse.ArgumentParser()
p.add_argument("missions")
p.add_argument("ref")
p.add_argument("--glb", type=int, default=3)
p.add_argument("--batch", type=int, default=1024)
a = p.parse_args()

missions = voxmap.load_missions(a.missions)
recs = voxmap.read_ref(a.ref)
by_mission = {}
for m, pos, q, t, d in voxmap.ref_poses(missions, recs):
    by_mission.setdefault(id(m), (m, []))[1].append((pos, q, t, d))

desc = Depth()


def render(scene, cams):
    r = hyperlux.HyperLux(width=20, height=16, batch_size=a.batch, num_probes=0,
                          cameras=(CameraSpec("front", (desc,)),), fidelity="low")
    r.init(scene)
    out = []
    for i in range(0, len(cams), a.batch):
        c = cams[i:i + a.batch]
        pad = np.zeros((a.batch, 12), np.float32)
        pad[:len(c)] = c
        pad[len(c):] = c[0]
        r.set_cameras(pad)
        r.render()
        r.synchronize()
        img = np.asarray(r.outputs[desc].read(), np.float32).reshape(a.batch, -1)[:len(c), :320]
        out.append(img)
    return np.concatenate(out)


rows = []
all_err_ref, all_err_np = [], []
glb_done = 0
for m, poses in by_mission.values():
    name = m.name
    cams = np.stack([voxmap.ray_grid_basis(pos, q) for pos, q, _, _ in poses])
    ref = np.stack([d for *_, d in poses])
    # static-only numpy depth, and full (with movers) numpy depth to mask mover hits
    static = voxmap.Task(**{**m.__dict__, "movers": []})
    np_static = np.stack([voxmap.depth_frame(static, pos, q, t) for pos, q, t, _ in poses])
    np_full = np.stack([voxmap.depth_frame(m, pos, q, t) for pos, q, t, _ in poses]) if m.movers else np_static
    mask = np.abs(np_full - np_static) < 1e-4
    hl = np.minimum(voxmap.planar_to_range(render(voxmap.hyperlux_scene(voxmap.world_quads(m, movers=False), lights=False), cams)), voxmap.MAX_RANGE)
    e_ref = (hl - ref)[mask]
    e_np = (hl - np_static).ravel()
    all_err_ref.append(e_ref); all_err_np.append(e_np)
    row = dict(mission=name, poses=len(poses), rays=int(mask.sum()), movers=len(m.movers),
               max_abs_vs_ref=float(np.abs(e_ref).max()), p999_abs_vs_ref=float(np.quantile(np.abs(e_ref), .999)),
               frac_gt_1cm=float((np.abs(e_ref) > .01).mean()))
    if glb_done < a.glb:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "course.glb")
            voxmap.write_glb(path, voxmap.world_quads(m, movers=False))
            row["glb_bytes"] = os.path.getsize(path)
            sc = hyperlux.load_scene(path, fidelity="low")
            hg = np.minimum(voxmap.planar_to_range(render(sc, cams)), voxmap.MAX_RANGE)
            row["glb_max_abs_vs_direct"] = float(np.abs(hg - hl).max())
            row["glb_frac_gt_1cm_vs_ref"] = float((np.abs(hg - ref)[mask] > .01).mean())
        glb_done += 1
    rows.append(row)
    print(json.dumps(row), flush=True)

er = np.abs(np.concatenate(all_err_ref)); en = np.abs(np.concatenate(all_err_np))
print(json.dumps(dict(summary=True, missions=len(rows), rays=int(er.size), mean_abs=float(er.mean()),
                      p99=float(np.quantile(er, .99)), p9999=float(np.quantile(er, .9999)), max=float(er.max()),
                      frac_gt_1mm=float((er > 1e-3).mean()), frac_gt_1cm=float((er > .01).mean()),
                      frac_gt_10cm=float((er > .1).mean()), vs_numpy_static_frac_gt_1cm=float((en > .01).mean()))))

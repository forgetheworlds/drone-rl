"""Can HyperLux probe rays stand in for our contact test?

Our contact: a 0.18 m sphere against voxels/movers, checked every 10 ms substep (exact).
HyperLux probes: rays from the camera origin along a fixed direction set; the minimum hit distance
is an upper bound on the true clearance (a probe can slip past a corner or edge).

  python probe_clearance.py MISSIONS_DIR [--tasks a.task b.task] [--n 4000]

Samples positions uniformly in each course's free space (plus a near-wall band), computes exact
clearance with voxmap.clearance, and probe clearance with 16/64/256/1024 probes. Reports the
overestimate and how often a real contact (clearance < 0.18) is missed.
"""
import argparse
import json
import os

import numpy as np

import hyperlux

import voxmap

p = argparse.ArgumentParser()
p.add_argument("missions")
p.add_argument("--tasks", nargs="*", default=["source/furniture-280000.task", "public/public-visible-Complex-3.task"])
p.add_argument("--n", type=int, default=4000)
a = p.parse_args()

for rel in a.tasks:
    task = voxmap.load_task(os.path.join(a.missions, rel))
    task = voxmap.Task(**{**task.__dict__, "movers": []})
    rng = np.random.default_rng(0)
    ext = np.array(task.occ.shape) * task.scale
    pts = []
    while len(pts) < a.n:
        c = rng.uniform(0.05, 1, (a.n, 3)) * (ext - 0.1)
        idx = np.floor(c / task.scale).astype(int)
        free = ~task.occ[idx[:, 0], idx[:, 1], idx[:, 2]]
        pts += list(c[free])
    pts = np.array(pts[:a.n], np.float32)
    exact = np.array([voxmap.clearance(task, x) for x in pts])
    # keep a balanced set: half within 0.5 m of something
    near = pts[exact < 0.5]; far = pts[exact >= 0.5]
    k = min(len(near), a.n // 2)
    pts = np.concatenate([near[:k], far[:a.n - k]])
    exact = np.array([voxmap.clearance(task, x) for x in pts])
    scene = voxmap.hyperlux_scene(voxmap.world_quads(task, movers=False), lights=False)
    contact = exact < voxmap.BODY_RADIUS
    for probes in (16, 64, 256, 1024):
        s = hyperlux.ClearanceSampler(scene, clearance=0.4, probes=probes)
        got = np.asarray(s.clearance_distances(pts), np.float64)
        del s
        over = got - exact
        pred = got < voxmap.BODY_RADIUS
        print(json.dumps(dict(task=rel, probes=probes, n=len(pts), contacts=int(contact.sum()),
                              under_estimates=int((over < -1e-3).sum()),
                              over_mean=float(over[np.isfinite(over)].mean()),
                              over_p50=float(np.quantile(over[np.isfinite(over)], .5)),
                              over_p95=float(np.quantile(over[np.isfinite(over)], .95)),
                              missed_contacts=int((contact & ~pred).sum()),
                              missed_contact_rate=float((contact & ~pred).sum() / max(1, contact.sum())),
                              false_contacts=int((~contact & pred).sum()))), flush=True)

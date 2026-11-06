"""Waypoint missions in a voxelized HyperDrone scene (procthor_voxel.py), as .task files.

One goal per mission, launch yaw facing the goal, goal within +-30 deg elevation (inside both the
90 and the 120 deg lens). Reachable: start and goal in the same connected free region with
>= 0.3 m clearance. Kinds (heights above the floor top):
  open      same height (+-0.5 m), clear line of sight, >= 0.5 m clearance along it
  gap       clear line of sight that passes within 0.35 m of something (doorways, furniture gaps)
  vertical  climb or descend >= 0.8 m (0.4-0.7 <-> 1.4-1.9 m), clear line of sight
  low_block start 0.35-0.6 m, goal at the same height behind furniture: the straight line is
            blocked; a point 0.25-1.2 m above the goal is visible from the start or 0.5 m above
            it (the VLM rule used for the counter missions)

  python procthor_missions.py VOXEL_DIR OUT_DIR [--per-kind 10] [--seed 1]
"""
import argparse, json, math, os
import numpy as np
from scipy import ndimage
import voxmap

p = argparse.ArgumentParser()
p.add_argument("vdir"); p.add_argument("out")
p.add_argument("--per-kind", type=int, default=10); p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
S = json.load(open(os.path.join(a.vdir, "origin.json")))["pitch"]
occ = voxmap.load_map(os.path.join(a.vdir, "procthor.3dmap"))
nz = occ.shape[2]
floor_z = int(np.argmax(occ.sum((0, 1)) > 0.5 * occ.sum((0, 1)).max()))      # first full layer
ceil_z = nz - 1 - int(np.argmax(occ.sum((0, 1))[::-1] > 0.5 * occ.sum((0, 1)).max()))
ftop = (floor_z + 1) * S
dist = ndimage.distance_transform_edt(~occ) * S
inside = np.zeros_like(occ); inside[:, :, floor_z + 1:ceil_z] = True
ok = (dist >= 0.3) & inside
lab, nl = ndimage.label(ok)
main = np.argmax(np.bincount(lab.ravel())[1:]) + 1
free_cells = np.argwhere(lab == main)
rng = np.random.default_rng(a.seed)


def cell(x):
    return tuple(np.floor(np.asarray(x) / S).astype(int))


def seg(p0, p1, step=0.05):
    n = max(2, int(np.linalg.norm(p1 - p0) / step))
    return p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]


def clear(p0, p1):   # line of sight through free cells
    c = np.floor(seg(p0, p1) / S).astype(int)
    return not occ[c[:, 0], c[:, 1], c[:, 2]].any()


def min_dist(p0, p1):
    c = np.floor(seg(p0, p1)[3:-3] / S).astype(int)
    return dist[c[:, 0], c[:, 1], c[:, 2]].min() if len(c) else 9


def sample(hlo, hhi):
    for _ in range(2000):
        c = free_cells[rng.integers(len(free_cells))]
        x = (c + rng.uniform(0, 1, 3)) * S
        if hlo <= x[2] - ftop <= hhi:
            return x
    return None


def reachable(x, g):
    return lab[cell(x)] == main and lab[cell(g)] == main


def elev_ok(x, g):
    d = g - x
    return abs(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))) <= 30


def try_kind(kind):
    if kind == "low_block":
        x = sample(.35, .6)
        if x is None: return None
        ang = rng.uniform(-math.pi, math.pi); r = rng.uniform(2, 5)
        g = x + np.array([r * math.cos(ang), r * math.sin(ang), rng.uniform(-.1, .1)])
        if not (0 <= g[0] < occ.shape[0] * S and 0 <= g[1] < occ.shape[1] * S): return None
        if not reachable(x, g) or clear(x, g): return None
        # blocked by furniture, not by a wall: something within 1.2 m above the line is free
        if not any(clear(x + [0, 0, up], g + [0, 0, up]) for up in (.6, .9, 1.2)): return None
        if not any(clear(x + [0, 0, s], g + [0, 0, u]) for s in (0, .5) for u in np.arange(.25, 1.21, .1)): return None
        return x, g
    if kind == "vertical":
        lo, hi = (.4, .7), (1.4, 1.9)
        if rng.uniform() < .5: lo, hi = hi, lo
        x = sample(*lo)
        if x is None: return None
        ang = rng.uniform(-math.pi, math.pi); r = rng.uniform(1.8, 5)
        g = x + np.array([r * math.cos(ang), r * math.sin(ang), 0]); g[2] = ftop + rng.uniform(*hi)
    else:
        x = sample(.4, 1.8)
        if x is None: return None
        ang = rng.uniform(-math.pi, math.pi); r = rng.uniform(2, 6)
        g = x + np.array([r * math.cos(ang), r * math.sin(ang), rng.uniform(-.5, .5)])
        g[2] = min(max(g[2], ftop + .4), ftop + 1.9)
    if not (0 <= g[0] < occ.shape[0] * S and 0 <= g[1] < occ.shape[1] * S): return None
    if not reachable(x, g) or not clear(x, g) or not elev_ok(x, g): return None
    md = min_dist(x, g)
    if kind == "open" and md < .5: return None
    if kind == "gap" and not (md < .35): return None
    if kind == "vertical" and abs(g[2] - x[2]) < .8: return None
    return x, g


os.makedirs(os.path.join(a.out, "procthor"), exist_ok=True)
mp = os.path.abspath(os.path.join(a.vdir, "procthor.3dmap"))
for kind in ("open", "gap", "vertical", "low_block"):
    n = tries = 0
    while n < a.per_kind and tries < 200000:
        tries += 1
        r = try_kind(kind)
        if r is None: continue
        x, g = r
        yaw = math.atan2(g[1] - x[1], g[0] - x[0])
        with open(os.path.join(a.out, "procthor", f"pt-{kind}-{n:02d}.task"), "w") as f:
            f.write(f"{mp} {S}\n{x[0]:.3f} {x[1]:.3f} {x[2]:.3f}\n1\n{g[0]:.3f} {g[1]:.3f} {g[2]:.3f}\n0\nyaw {yaw:.4f}\n")
        n += 1
    print(kind, n, "missions", tries, "tries")
print("floor_top", ftop, "ceiling", ceil_z * S)

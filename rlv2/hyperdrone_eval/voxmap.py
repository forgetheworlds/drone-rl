"""Our drone-rl voxel courses, in numpy, for comparing against HyperDrone.

- load_task(): reads a drone-rl .task and its .3dmap (same rules as rlv2/missions.hpp).
- ray_static()/ray(): a line-by-line numpy port of vx_ray_static/vx_ray in rlv2/voxel_world.h
  (Amanatides-Woo DDA, outside the grid is solid, 12 m cap, slab test for movers).
- camera_rays(): the 20 x 16 body-frame ray grid of vx_camera_ray (90 deg HFOV, slope 0.75).
- exposed_faces(): every face between an empty in-grid cell and a solid cell (out of grid
  counts as solid), so a mesh renderer sees exactly the surfaces our ray caster stops at.
- write_glb(): exports those faces (plus movers at a given time) as a .glb, in glTF's Y-up
  frame, so HyperLux's GLB loader (Y-up -> FLU) puts them back in our Z-up world frame.
- hyperlux_scene(): builds the same geometry directly as a hyperlux.Scene.
"""
import math
import os
from dataclasses import dataclass, field

import numpy as np

RAY_COLS, RAY_ROWS, MAX_RANGE, BODY_RADIUS = 20, 16, 12.0, 0.18


@dataclass
class Mover:
    center: np.ndarray
    extent: np.ndarray
    axis: int
    amplitude: float
    period: float
    phase: float

    def center_at(self, t):
        c = self.center.copy()
        c[self.axis] += self.amplitude * math.cos(2 * math.pi * t / self.period + self.phase)
        return c


@dataclass
class Task:
    name: str
    map_path: str
    scale: float
    start: np.ndarray
    goals: np.ndarray
    occ: np.ndarray  # bool [nx, ny, nz]
    movers: list = field(default_factory=list)
    launch_yaw: float = 0.0


def load_map(path):
    with open(path) as f:
        head = f.readline().split()
        assert head[0] == "voxel", path
        nx, ny, nz = map(int, head[1:4])
        cells = np.loadtxt(f, dtype=np.int64, ndmin=2)
    occ = np.zeros((nx, ny, nz), dtype=bool)
    if cells.size:
        occ[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return occ


def load_task(path):
    tokens = open(path).read().split()
    i = 0
    map_name = tokens[i]; i += 1
    scale = float(tokens[i]); i += 1
    start = np.array([float(t) for t in tokens[i:i + 3]]); i += 3
    n = int(tokens[i]); i += 1
    goals = np.array([[float(t) for t in tokens[i + 3 * k:i + 3 * k + 3]] for k in range(n)]); i += 3 * n
    nm = int(tokens[i]); i += 1
    movers = []
    for _ in range(nm):
        v = [float(t) for t in tokens[i:i + 10]]; i += 10
        movers.append(Mover(np.array(v[0:3]), np.array(v[3:6]), int(v[6]), v[7], v[8], v[9]))
    yaw = 0.0
    if i < len(tokens) and tokens[i] == "yaw":
        yaw = float(tokens[i + 1])
    map_path = map_name if os.path.isabs(map_name) else os.path.join(os.path.dirname(path), map_name)
    name = os.path.splitext(os.path.basename(path))[0]
    return Task(name, map_path, scale, start, goals, load_map(map_path), movers, yaw)


def camera_rays(slope=0.75):
    """(320, 3) unit body-frame (FLU) directions, ray k = row * 20 + col, as vx_camera_ray."""
    k = np.arange(RAY_COLS * RAY_ROWS)
    d = np.stack([np.ones(k.shape), 1 - ((k % RAY_COLS) + 0.5) / 10,
                  slope * (1 - ((k // RAY_COLS) + 0.5) / 8)], axis=1).astype(np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def rotation(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]], dtype=np.float32)


def ray_static(occ, scale, p, d, maximum=MAX_RANGE):
    """Vectorised vx_ray_static over rays: p (3,), d (N, 3). float32 like the Metal code."""
    p = np.asarray(p, np.float32)
    d = np.asarray(d, np.float32)
    n = len(d)
    s = np.float32(scale)
    nx, ny, nz = occ.shape
    shape = np.array([nx, ny, nz])
    cell = np.tile(np.floor(p / s).astype(np.int64), (n, 1))
    step = np.where(d >= 0, 1, -1)
    small = np.abs(d) < 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        tmax = np.where(small, np.float32(1e30), ((cell + (step > 0)).astype(np.float32) * s - p) / d).astype(np.float32)
        delta = np.where(small, np.float32(1e30), s / np.abs(d)).astype(np.float32)
    dist = np.zeros(n, np.float32)
    out = np.full(n, np.float32(maximum))
    live = np.ones(n, bool)
    rows = np.arange(n)
    while live.any():
        idx = rows[live]
        c = cell[idx]
        inside = np.all((c >= 0) & (c < shape), axis=1)
        solid = np.ones(len(idx), bool)
        ci = c[inside]
        solid[inside] = occ[ci[:, 0], ci[:, 1], ci[:, 2]]
        hit = solid & (dist[idx] < maximum)
        out[idx[hit]] = dist[idx[hit]]
        live[idx[hit]] = False
        idx = idx[~hit]
        if not len(idx):
            break
        t = tmax[idx]
        a = np.where(t[:, 0] < t[:, 1], np.where(t[:, 0] < t[:, 2], 0, 2), np.where(t[:, 1] < t[:, 2], 1, 2))
        dist[idx] = t[np.arange(len(idx)), a]
        tmax[idx, a] += delta[idx, a]
        cell[idx, a] += step[idx, a]
        done = dist[idx] >= maximum
        live[idx[done]] = False
    return out


def ray(task, p, d, t=0.0):
    """vx_ray: static grid then movers (slab test with drone-rl's edge cases)."""
    nearest = ray_static(task.occ, task.scale, p, d)
    p = np.asarray(p, np.float32)
    for m in task.movers:
        c = m.center_at(t)
        lo, hi = c - m.extent, c + m.extent
        entry = np.zeros(len(d), np.float32)
        exit_ = nearest.copy()
        for a in range(3):
            da = d[:, a]
            small = np.abs(da) < 1e-8
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = (lo[a] - p[a]) / da
                t1 = (hi[a] - p[a]) / da
            outside = small & ((p[a] < lo[a]) | (p[a] > hi[a]))
            exit_ = np.where(outside, -1, exit_)
            entry = np.where(small, entry, np.maximum(entry, np.minimum(t0, t1)))
            exit_ = np.where(small, exit_, np.minimum(exit_, np.maximum(t0, t1)))
        nearest = np.where(exit_ >= entry, entry, nearest).astype(np.float32)
    return nearest


def depth_frame(task, position, q, t=0.0, slope=0.75):
    r = rotation(q)
    return ray(task, position, camera_rays(slope) @ r.T, t)


def clearance(task, p, t=0.0):
    """Exact distance from p to the nearest solid voxel (out of grid solid) or mover."""
    s = task.scale
    occ = task.occ
    c = np.floor(np.asarray(p) / s).astype(int)
    best = np.inf
    for n in range(1, 64):
        lo = c - n
        hi = c + n + 1
        sub = np.ones((2 * n + 1,) * 3, bool)
        a0 = np.maximum(lo, 0); a1 = np.minimum(hi, occ.shape)
        if np.all(a1 > a0):
            sub[:] = True
            sub_slice = tuple(slice(a0[i] - lo[i], a1[i] - lo[i]) for i in range(3))
            sub[sub_slice] = occ[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
        idx = np.argwhere(sub) + lo
        if len(idx):
            q = np.clip(p, idx * s, (idx + 1) * s)
            best = min(best, np.sqrt(((q - p) ** 2).sum(1)).min())
        if best <= n * s:
            break
    for m in task.movers:
        cc = m.center_at(t)
        dd = np.maximum(0, np.abs(np.asarray(p) - cc) - m.extent)
        best = min(best, float(np.sqrt((dd ** 2).sum())))
    return best


def exposed_faces(occ):
    """Quads (F, 4, 3) in cell units where an empty in-grid cell meets solid (incl. outside)."""
    nx, ny, nz = occ.shape
    pad = np.ones((nx + 2, ny + 2, nz + 2), bool)
    pad[1:-1, 1:-1, 1:-1] = occ
    empty = ~occ
    quads = []
    corners = {  # unit-square corners of the face on the +side of axis a, ccw seen from the empty cell
        0: np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
        1: np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
        2: np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    }
    for a in range(3):
        for sign in (1, -1):
            off = [1, 1, 1]
            off[a] += sign
            nb = pad[off[0]:off[0] + nx, off[1]:off[1] + ny, off[2]:off[2] + nz]
            cells = np.argwhere(empty & nb)
            base = corners[a].copy()
            if sign < 0:
                base[:, a] = 0
                base = base[::-1]
            quads.append(cells[:, None, :] + base[None])
    return np.concatenate(quads).astype(np.float32)


def box_quads(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    return v[np.array(f)]


def world_quads(task, t=0.0, movers=True):
    q = exposed_faces(task.occ) * np.float32(task.scale)
    if movers and task.movers:
        q = np.concatenate([q] + [box_quads(m.center_at(t) - m.extent, m.center_at(t) + m.extent) for m in task.movers])
    return q


def face_colors(n, seed=0, palette=24):
    """Random flat colour per face (gives stereo something to match at voxel scale)."""
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.15, 0.95, size=(palette, 3)).astype(np.float32)
    return pal, rng.integers(0, palette, size=n)


def write_glb(path, quads, seed=0, palette=24):
    import trimesh
    pal, which = face_colors(len(quads), seed, palette)
    meshes = []
    for c in range(palette):
        sel = quads[which == c]
        if not len(sel):
            continue
        v = sel.reshape(-1, 3)
        base = np.arange(len(sel)) * 4
        # double sided, like hyperlux's own example quads
        f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1),
                            np.stack([base, base + 2, base + 1], 1), np.stack([base, base + 3, base + 2], 1)])
        yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # Z-up world -> glTF Y-up
        m = trimesh.Trimesh(vertices=yup, faces=f, process=False)
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile((np.append(pal[c], 1) * 255).astype(np.uint8), (len(f), 1)))
        meshes.append(m)
    scene = trimesh.Scene(meshes)
    scene.export(path)
    return len(quads)


def hyperlux_scene(quads, seed=0, palette=24, lights=True):
    import hyperlux
    pal, which = face_colors(len(quads), seed, palette)
    scene = hyperlux.Scene()
    obj = hyperlux.Object(name="course")
    for c in range(palette):
        sel = quads[which == c]
        if not len(sel):
            continue
        v = np.ascontiguousarray(sel.reshape(-1, 3), np.float32)
        base = np.arange(len(sel), dtype=np.int32) * 4
        f = np.concatenate([np.stack([base, base + 1, base + 2], 1), np.stack([base, base + 2, base + 3], 1),
                            np.stack([base, base + 2, base + 1], 1), np.stack([base, base + 3, base + 2], 1)]).astype(np.int32)
        obj.add_mesh(hyperlux.Mesh(v, np.ascontiguousarray(f), color=tuple(float(x) for x in pal[c])))
    scene.add_object(obj)
    if lights:
        scene.add_light(hyperlux.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
        scene.add_light(hyperlux.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.4, 0.4, 0.5)))
    return scene


def ray_grid_basis(position, q, slope=0.75):
    """HyperLux packed camera (12 floats) whose pixel-centre rays at 20 x 16 are exactly our 320
    rays: pixel (c, r) direction = dir_00 + (c+.5)/W du + (r+.5)/H dv."""
    r = rotation(q)
    dir00 = r @ np.array([1, 1, slope], np.float32)
    du = r @ np.array([0, -2, 0], np.float32)
    dv = r @ np.array([0, 0, -2 * slope], np.float32)
    return np.concatenate([position, dir00, du, dv]).astype(np.float32)


def planar_to_range(planar, slope=0.75):
    """HyperLux Depth is distance along the optical axis; ours is range along the ray."""
    d = camera_rays(slope)
    return planar / d[:, 0]


def load_missions(missions_dir):
    """All .task files in rlv2/missions.hpp order (source, public, fresh; sorted)."""
    out = []
    for portfolio in ("source", "public", "fresh"):
        d = os.path.join(missions_dir, portfolio)
        out += [load_task(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".task")]
    return out


REF_DTYPE = np.dtype([("position", "<f4", 3), ("velocity", "<f4", 3), ("orientation", "<f4", 4),
                      ("alive", "<u4"), ("contact", "<u4"), ("contact_substep", "<u4"), ("visible", "<u4"),
                      ("depth", "<f4", 320)])


def read_ref(path):
    """rlv2_reference output: (sequences, ticks) records. Depth of tick t was taken at the pose
    recorded for tick t-1 (tick 0: mission start, launch yaw) at time 0.05 t."""
    header = np.fromfile(path, dtype="<u4", count=4)
    assert header[0] == 0x31585652
    recs = np.fromfile(path, dtype=REF_DTYPE, offset=16)
    return recs.reshape(int(header[1]), int(header[2]))


def ref_poses(missions, recs):
    """Yields (mission, position, quaternion, time, recorded depth) for every live tick."""
    for s in range(recs.shape[0]):
        m = missions[s % len(missions)]
        for t in range(recs.shape[1]):
            r = recs[s, t]
            if not r["alive"]:
                break
            if t == 0:
                p, q = m.start.astype(np.float32), np.array([math.cos(m.launch_yaw / 2), 0, 0, math.sin(m.launch_yaw / 2)], np.float32)
            else:
                p, q = recs[s, t - 1]["position"], recs[s, t - 1]["orientation"]
            yield m, p, q, 0.05 * t, r["depth"]

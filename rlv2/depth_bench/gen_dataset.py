"""Generate the versioned depth-benchmark dataset with HyperDrone/HyperLux (Metal or OptiX).

Each sequence is 8 frames at 10 Hz of a drone flying through a scene. Per frame we store:
  clean  : left + right RGB at 320x240, rendered at the frame's nominal pose (global shutter)
  rs     : rolling shutter (30 ms readout, top->bottom) + 10 ms exposure motion blur
  rsdark : same with 40 ms exposure (low light needs long exposure -> more blur)
  rsunsync: right camera only, rs but started 12 ms after the left (free-running, unsynced pair)
for right cameras at baselines 6 cm and 12 cm (12 cm only clean+rs), plus for the left camera:
  depth  : GT planar depth (m, inf = no hit) at the nominal pose
  normal : GT surface class (0 none, 1 floor, 2 wall/vertical, 3 ceiling, 4 other)
  thin   : 1 where an added thin object (2-4 cm pole, 1-2 cm rod) is the visible surface
  pose   : 4x4 camera-to-world of the left camera (columns = forward, left, up; FLU world)
Rolling shutter / blur are composed from 15 sub-frames at 5 ms spacing (-15..55 ms around the
frame start), including 25 Hz roll/pitch vibration, so jello and smear are geometric, not 2D fakes.

Scenes: ProcTHOR houses (Train200 catalog) and our voxel courses with flat per-orientation colours
(whole walls are one colour -> textureless, unlike the random face colours in hyperdrone_eval).
Every sequence gets thin objects near its path.

  python gen_dataset.py --out DIR --split dev|test|nav [--seqs 8]

The dev/test splits alternate cruise and approach (wall close-up) sequences. The nav split has
only "nav" sequences: heading at an open direction (>= 2.2 m clear ahead), 0.8-1.5 m/s, gentle
yaw, so the frames look like normal flight toward a waypoint.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import hyperlux
from hyperlux.camera import CameraSpec, Depth, Normals, RGB

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hyperdrone_eval"))
import voxmap  # noqa: E402

VERSION = "v1"
W, H, HFOV = 320, 240, 66.0
BASELINES = (0.06, 0.12)
T, DT = 8, 0.1
SUB = np.arange(-15, 60, 5) * 1e-3          # 15 sub-frame times (s) relative to frame start
READOUT, EXP, EXP_DARK = 0.030, 0.010, 0.040
UNSYNC = 0.012                              # right camera started 12 ms late (no shared frame sync)
NOMINAL = 0.020                             # mid-exposure time of the middle row (normal exposure)
VIB_AMP, VIB_HZ = math.radians(0.8), 25.0
SPLITS = {
    "dev": ["procthor:ProcTHOR-Train-1", "procthor:ProcTHOR-Train-2",
            "voxel:source/furniture-280000", "voxel:source/overhang-282000"],
    "test": ["procthor:ProcTHOR-Train-150", "procthor:ProcTHOR-Train-151", "procthor:ProcTHOR-Train-152",
             "procthor:ProcTHOR-Train-153", "voxel:fresh/fresh-dense-00", "voxel:source/narrow_gap-281000",
             "voxel:source/gap_overhang-284000"],
    # Nav view: forward flight toward an open direction, as when heading for a waypoint along a
    # route. No wall close-ups: >= 2.2 m clear straight ahead at the start. Unseen houses.
    "nav": ["procthor:ProcTHOR-Train-154", "procthor:ProcTHOR-Train-155", "procthor:ProcTHOR-Train-156",
            "procthor:ProcTHOR-Train-157", "procthor:ProcTHOR-Train-158", "procthor:ProcTHOR-Train-159",
            "voxel:fresh/fresh-dense-00", "voxel:source/narrow_gap-281000", "voxel:source/gap_overhang-284000"],
}
F = (W / 2) / math.tan(math.radians(HFOV / 2))
TX = math.tan(math.radians(HFOV / 2)); TY = TX * H / W


def rot(yaw, pitch=0.0, roll=0.0):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx  # columns: forward, left, up


def cam_row(pos, r):
    f, l, u = r[:, 0], r[:, 1], r[:, 2]
    return np.concatenate([pos, f + TX * l + TY * u, -2 * TX * l, -2 * TY * u]).astype(np.float32)


def flat_voxel_scene(task, seed):
    """Voxel course, one flat colour per face orientation (6 near-neutral colours)."""
    rng = np.random.default_rng(seed)
    q = voxmap.world_quads(task, movers=False)
    n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    key = np.argmax(np.abs(n), 1) * 2 + (n[np.arange(len(n)), np.argmax(np.abs(n), 1)] > 0)
    base = rng.uniform(0.55, 0.85, 3)
    pal = np.clip(base[None] + rng.uniform(-0.12, 0.12, (6, 3)), 0.1, 0.95)
    scene = hyperlux.Scene()
    obj = hyperlux.Object(name="course")
    for c in range(6):
        sel = q[key == c]
        if not len(sel):
            continue
        add_quads(obj, sel, pal[c])
    scene.add_object(obj)
    scene.add_light(hyperlux.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.8, 1.8, 1.8)))
    scene.add_light(hyperlux.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.9, 0.9, 1.0)))
    return scene


def add_quads(obj, quads, color):
    v = np.ascontiguousarray(quads.reshape(-1, 3), np.float32)
    b = np.arange(len(quads), dtype=np.int32) * 4
    f = np.concatenate([np.stack([b, b + 1, b + 2], 1), np.stack([b, b + 2, b + 3], 1),
                        np.stack([b, b + 2, b + 1], 1), np.stack([b, b + 3, b + 2], 1)]).astype(np.int32)
    obj.add_mesh(hyperlux.Mesh(v, np.ascontiguousarray(f), color=tuple(float(x) for x in color)))


def load_base(spec, seed):
    kind, name = spec.split(":", 1)
    if kind == "procthor":
        ref = hyperlux.datasets.procthor.Train200()[name].reference
        return (lambda: hyperlux.load_scene(ref, fidelity="high")), None
    task = voxmap.load_task(os.path.join(os.environ["MISSIONS"], name + ".task"))
    return (lambda: flat_voxel_scene(task, seed)), task


def voxel_free_points(task, n, clearance, rng):
    occ, s = task.occ, task.scale
    k = int(math.ceil(clearance / s))
    pts = []
    for _ in range(200000):
        c = np.array([rng.integers(k, d - k) for d in occ.shape])
        if not occ[c[0] - k:c[0] + k + 1, c[1] - k:c[1] + k + 1, c[2] - k:c[2] + k + 1].any():
            p = (c + 0.5) * s
            if 0.5 < p[2] < 2.2:
                pts.append(p)
                if len(pts) == n:
                    break
    return np.array(pts)


THIN_COLORS = [(0.92, 0.92, 0.9), (0.08, 0.08, 0.08), (0.55, 0.4, 0.25), (0.5, 0.5, 0.52)]


def thin_objects(plans, rng):
    """One vertical pole per sequence 1.2-2.5 m ahead, laterally offset; half also get a rod."""
    quads = []
    for p in plans:
        r = rot(p["yaw"])
        # nav: the drone flies up to 1.05 m, so thin objects start beyond that and end >= 0.6 m away
        go = 1.05 if p["kind"] == "nav" else 0.0
        ahead = (go + rng.uniform(0.6, 2.0) if p["kind"] == "nav" else rng.uniform(1.2, 2.5) if p["kind"] == "cruise"
                 else min(p["d0"] - 0.3, rng.uniform(0.7, 1.6)))
        side = rng.choice([-1, 1]) * rng.uniform(0.15, 0.5)
        c = p["start"] + ahead * r[:, 0] + side * r[:, 1]
        w = rng.uniform(0.02, 0.04) / 2
        quads.append((voxmap.box_quads(c + [-w, -w, -1.5], c + [w, w, 1.5]), THIN_COLORS[rng.integers(4)]))
        if rng.random() < 0.5:
            c2 = p["start"] + (go + rng.uniform(0.6, 2.0) if p["kind"] == "nav" else rng.uniform(1.0, 2.2)) * r[:, 0] \
                + rng.uniform(-0.2, 0.3) * r[:, 2]
            t = rng.uniform(0.01, 0.02) / 2
            a = c2 - 0.5 * r[:, 1]; b = c2 + 0.5 * r[:, 1]
            lo = np.minimum(a, b) - t; hi = np.maximum(a, b) + t
            quads.append((voxmap.box_quads(lo, hi), THIN_COLORS[rng.integers(4)]))
    return quads


def thin_mask(row, depth, boxes):
    """1 where the pixel ray's first hit is one of the thin boxes (analytic slab test)."""
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    o, d00, du, dv = row[0:3], row[3:6], row[6:9], row[9:12]
    d = d00[None, None] + u[None, :, None] * du[None, None] + v[:, None, None] * dv[None, None]  # forward comp = 1
    out = np.zeros((H, W), np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for lo, hi in boxes:
            t0 = (lo - o) * inv; t1 = (hi - o) * inv
            tn = np.nanmax(np.minimum(t0, t1), -1); tf = np.nanmin(np.maximum(t0, t1), -1)
            hit = (tf >= np.maximum(tn, 0)) & (np.abs(np.maximum(tn, 0) - depth) < 0.03)
            out[hit] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    specs = a.scenes.split(",") if a.scenes else SPLITS[a.split]
    NB = len(BASELINES) + 1
    nsub = len(SUB)
    per_frame = NB * (nsub + 1)
    rgb_d, dep_d, nrm_d = RGB("rgba8"), Depth(), Normals()
    for si, spec in enumerate(specs):
        sname = spec.split(":", 1)[1].split("/")[-1]
        out_path = os.path.join(a.out, f"{a.split}_{sname}.npz")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        seed = 1000 * (si + 1) + {"dev": 0, "test": 50000, "nav": 90000}[a.split]
        rng = np.random.default_rng(seed)
        make, task = load_base(spec, seed)
        base = make()
        # pick starts + yaws on the base scene
        probe = hyperlux.HyperLux(width=64, height=48, batch_size=64, num_probes=0,
                                  cameras=(CameraSpec("front", (Depth(),)),), fidelity="high")
        probe.init(base)
        if task is not None:
            cand = voxel_free_points(task, 64, 0.5, rng)
        else:
            cand = hyperlux.ClearanceSampler(base, clearance=0.5).sample(64, seed=seed)
            cand = cand[(cand[:, 2] > 0.6) & (cand[:, 2] < 2.0)] if (cand[:, 2] > 0.6).sum() > 12 else cand
        plans = []
        yaws = np.arange(16) * (2 * math.pi / 16)
        for ci in range(0, len(cand), 4):
            if len(plans) >= a.seqs:
                break
            pts = cand[ci:ci + 4]
            rows = np.stack([cam_row(p, rot(y)) for p in pts for y in yaws])
            rows = np.concatenate([rows, np.repeat(rows[-1:], 64 - len(rows), 0)])
            probe.set_cameras(rows); probe.render(); probe.synchronize()
            d = np.asarray(probe.outputs[Depth()].read(), np.float32).reshape(64, 48, 64)
            for pi, p in enumerate(pts):
                dd = d[pi * 16:(pi + 1) * 16]
                if not np.isfinite(dd).mean() > 0.97 or np.nanmax(np.where(np.isfinite(dd), dd, 0)) > 25:
                    continue  # outside / open to the sky
                center = dd[:, 18:30, 26:38].reshape(16, -1).min(1)
                if a.split == "nav":
                    kind = "nav"
                    ok = np.where(center > 2.2)[0]
                else:
                    kind = "cruise" if len(plans) % 2 == 0 else "approach"
                    ok = np.where(center > 1.6)[0] if kind == "cruise" else np.where((center > 1.0) & (center < 1.7))[0]
                if not len(ok):
                    continue
                yi = int(rng.choice(ok))
                yaw_rate = {"cruise": rng.uniform(-0.7, 0.7), "approach": rng.uniform(-0.15, 0.15),
                            "nav": rng.uniform(-0.3, 0.3)}[kind]
                plans.append(dict(kind=kind, start=np.array(p, float), yaw=float(yaws[yi]), d0=float(center[yi]),
                                  yaw_rate=float(yaw_rate),
                                  vib_phase=rng.uniform(0, 2 * math.pi, 2)))
                if len(plans) >= a.seqs:
                    break
        # full scene with thin objects
        scene = make()
        thin = hyperlux.Object(name="thin")
        boxes = []
        for q, c in thin_objects(plans, rng):
            add_quads(thin, q, c)
            boxes.append((q.reshape(-1, 3).min(0), q.reshape(-1, 3).max(0)))
        scene.add_object(thin)
        nbatch = T * per_frame
        r = hyperlux.HyperLux(width=W, height=H, batch_size=nbatch, num_probes=0,
                              cameras=(CameraSpec("front", (rgb_d, dep_d, nrm_d)),), fidelity="high")
        r.init(scene)
        rec = {k: [] for k in ("L_clean", "R06_clean", "R12_clean", "L_rs", "R06_rs", "R12_rs",
                               "L_rsdark", "R06_rsdark", "R06_rsunsync", "depth", "normal", "thin", "pose")}
        meta = []
        for p in plans:
            if p["kind"] != "approach":
                speed = rng.uniform(0.8, 1.5)
            else:
                speed = (p["d0"] - 0.35) / ((T - 1) * DT)

            def state(t):
                yaw = p["yaw"] + p["yaw_rate"] * t
                if p["kind"] != "approach":  # integrate heading
                    pos = p["start"] + speed * (np.array([math.sin(yaw) - math.sin(p["yaw"]), -(math.cos(yaw) - math.cos(p["yaw"])), 0]) / p["yaw_rate"]
                                                if abs(p["yaw_rate"]) > 1e-3 else t * np.array([math.cos(yaw), math.sin(yaw), 0]))
                else:
                    pos = p["start"] + speed * t * np.array([math.cos(p["yaw"]), math.sin(p["yaw"]), 0])
                pitch = VIB_AMP * math.sin(2 * math.pi * VIB_HZ * t + p["vib_phase"][0]) + 0.05 * (speed / 1.5)
                roll = VIB_AMP * math.sin(2 * math.pi * VIB_HZ * t + p["vib_phase"][1])
                return pos, rot(yaw, pitch, roll)

            rows = []
            poses = []
            for k in range(T):
                tf = k * DT
                pos, R = state(tf + NOMINAL)
                M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pos
                poses.append(M)
                for bi, B in enumerate((0.0,) + BASELINES):
                    pc, Rc = state(tf + NOMINAL)
                    rows.append(cam_row(pc - B * Rc[:, 1], Rc))
                    for s in SUB:
                        ps, Rs = state(tf + s)
                        rows.append(cam_row(ps - B * Rs[:, 1], Rs))
            rows = np.stack(rows)
            r.set_cameras(rows); r.render(); r.synchronize()
            img = np.asarray(r.outputs[rgb_d].read()).reshape(nbatch, H, W, -1)[..., :3].astype(np.float32)
            dep = np.asarray(r.outputs[dep_d].read(), np.float32).reshape(nbatch, H, W)
            nrm = np.asarray(r.outputs[nrm_d].read(), np.float32).reshape(nbatch, H, W, -1)[..., :3]
            row_t = np.arange(H) / H * READOUT
            for k in range(T):
                o = k * per_frame
                for bi, nm in enumerate(("L", "R06", "R12")):
                    b = o + bi * (nsub + 1)
                    rec[f"{nm}_clean"].append(img[b].astype(np.uint8))
                    subs = img[b + 1:b + 1 + nsub]
                    for var, e, start in (("rs", EXP, 0.0), ("rsdark", EXP_DARK, -0.015), ("rsunsync", EXP, UNSYNC)):
                        if var != "rs" and nm != "R06" and not (var == "rsdark" and nm == "L"):
                            continue
                        outi = np.zeros((H, W, 3), np.float32)
                        for y in range(H):
                            lo = row_t[y] + start
                            sel = (SUB >= lo - 1e-6) & (SUB <= lo + e + 1e-6)
                            outi[y] = subs[sel, y].mean(0)
                        rec[f"{nm}_{var}"].append(np.clip(outi + 0.5, 0, 255).astype(np.uint8))
                d = dep[o]
                rec["depth"].append(d.astype(np.float32))
                n = nrm[o]
                nz = n[..., 2]  # normals are world-frame
                cls = np.full((H, W), 4, np.uint8)
                cls[nz > 0.8] = 1; cls[np.abs(nz) < 0.3] = 2; cls[nz < -0.8] = 3; cls[~np.isfinite(d)] = 0
                rec["normal"].append(cls)
                rec["thin"].append(thin_mask(rows[o], d, boxes))
                rec["pose"].append(poses[k])
            meta.append(dict(kind=p["kind"], yaw=p["yaw"], d0=p["d0"], yaw_rate=p["yaw_rate"], speed=float(speed)))
        arr = {k: np.stack(v).reshape((len(plans), T) + np.stack(v).shape[1:]) for k, v in rec.items()}
        np.savez_compressed(out_path, **arr, meta=json.dumps(dict(
            version=VERSION, scene=spec, split=a.split, W=W, H=H, hfov=HFOV, f_px=F, baselines=BASELINES,
            dt=DT, readout=READOUT, exp=EXP, exp_dark=EXP_DARK, unsync=UNSYNC, vib_amp_deg=math.degrees(VIB_AMP), vib_hz=VIB_HZ,
            seqs=meta)))
        print(f"{spec}: {len(plans)} seqs in {time.time() - t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

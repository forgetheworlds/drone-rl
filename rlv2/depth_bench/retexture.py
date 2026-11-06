"""Re-texture the benchmark renders with world-anchored albedo noise (a realism check, not a renderer).

Every visible surface gets a multiplicative albedo pattern 1 + amp*n(p), with n a 3-octave value
noise (2, 6 and 20 cm cells) evaluated at the surface point p in world coordinates, so the left and
right images see the same pattern at the right disparity. Left: p from GT depth + pose. Right (6 cm):
the left depth forward-warped into the right camera (z-buffer, occlusion holes filled from the
farther neighbour). Built on the *clean* (global shutter, no blur) renders; the output stores the
same textured image under L_rs / R06_rs, so the ov2640 profile adds noise, gain mismatch, rectification
error and JPEG but no rolling shutter or blur (slightly optimistic). amp=0 gives the matching
untextured control. The 6 and 12 cm pairs are written (no dark / unsync variants).

  python retexture.py --amp 0.25 --split nav --out /workspace/dbench_data/v1tex25
"""
import argparse, glob, os
import numpy as np

KEYS = ("depth", "normal", "thin", "pose", "meta")


def _hash(ix, iy, iz, table):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
    return table[h & (len(table) - 1)]


def value_noise(P, cell, table):
    q = P / cell
    i = np.floor(q).astype(np.int64); t = q - i
    t = t * t * (3 - 2 * t)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[..., 0] if dx else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1]) * (t[..., 2] if dz else 1 - t[..., 2])
                out = out + w * _hash(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, table)
    return out


def albedo(P, amp, seed):
    table = np.random.default_rng(seed).uniform(-1, 1, 1 << 20).astype(np.float32)
    n = 0.5 * value_noise(P, 0.02, table) + 0.3 * value_noise(P, 0.06, table) + 0.2 * value_noise(P, 0.20, table)
    return np.clip(1 + amp * n / 0.45, 0.25, 1.75)   # /0.45: n's std is ~0.45 of its range


def world_points(z, pose, f, cx, cy, y_shift=0.0):
    h, w = z.shape
    x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pc = np.stack([z, z * (cx - x) / f + y_shift, z * (cy - y) / f], -1)
    return pc @ pose[:3, :3].T + pose[:3, 3]


def right_depth(z, f, B):
    h, w = z.shape
    zr = np.full((h, w), np.inf, np.float32)
    ok = np.isfinite(z) & (z > 0.02)
    ys, xs = np.nonzero(ok)
    ur = xs + 0.5 - f * B / z[ys, xs]
    for off in (-0.5, 0.0, 0.5):                # splat a little wider than one pixel
        xi = np.floor(ur + off).astype(int)
        m = (xi >= 0) & (xi < w)
        np.minimum.at(zr, (ys[m], xi[m]), z[ys[m], xs[m]])
    # occlusion holes: take the farther horizontal neighbour (background shows through)
    for _ in range(8):
        hole = ~np.isfinite(zr)
        if not hole.any():
            break
        l = np.roll(zr, 1, 1); r = np.roll(zr, -1, 1)
        cand = np.where(np.isfinite(l) & np.isfinite(r), np.maximum(l, r), np.where(np.isfinite(l), l, r))
        zr[hole] = cand[hole]
    return zr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, required=True)
    ap.add_argument("--split", default="nav")
    ap.add_argument("--src", default=os.environ.get("DEPTH_BENCH_DATA", "/workspace/dbench_data/v1"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import json
    for path in sorted(glob.glob(os.path.join(a.src, f"{a.split}_*.npz"))):
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        f, W, H = meta["f_px"], meta["W"], meta["H"]
        cx, cy = W / 2, H / 2
        L = d["L_clean"].astype(np.float32); R = d["R06_clean"].astype(np.float32); R12 = d["R12_clean"].astype(np.float32)
        seed = sum(map(ord, os.path.basename(path)))   # one pattern per scene file
        for i in range(L.shape[0]):
            for k in range(L.shape[1]):
                z = d["depth"][i, k]; pose = d["pose"][i, k]
                zl = np.where(np.isfinite(z), z, 50.0)
                L[i, k] *= albedo(world_points(zl, pose, f, cx, cy), a.amp, seed)[..., None]
                zr = right_depth(z, f, 0.06); zr = np.where(np.isfinite(zr), zr, 50.0)
                R[i, k] *= albedo(world_points(zr, pose, f, cx, cy, y_shift=-0.06), a.amp, seed)[..., None]
                zr = right_depth(z, f, 0.12); zr = np.where(np.isfinite(zr), zr, 50.0)
                R12[i, k] *= albedo(world_points(zr, pose, f, cx, cy, y_shift=-0.12), a.amp, seed)[..., None]
        L, R, R12 = (np.clip(x, 0, 255).astype(np.uint8) for x in (L, R, R12))
        out = {k: d[k] for k in KEYS}
        out.update(L_clean=L, R06_clean=R, L_rs=L, R06_rs=R, R12_clean=R12, R12_rs=R12)
        np.savez(os.path.join(a.out, os.path.basename(path)), **out)
        print("wrote", os.path.basename(path), flush=True)


if __name__ == "__main__":
    main()

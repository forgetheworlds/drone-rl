"""Planner-facing metrics. All depths are planar (along the optical axis), metres.

Pixel level   tol(z) = max(0.10 m, 0.15 z). est valid & |e-z|<=tol -> ok; e<z-tol -> false near
              (false-occupied); e>z+tol -> false far. hole = no estimate.
Ray level     the image is cut into 16x12 blocks of 20x20 px (a ray each, like the policy's ray
              grid). GT ray = 5th percentile of GT depth in the block (nearest obstacle). Method ray
              = 5th percentile of its valid depths if >=10% of the block is valid, else UNKNOWN.
              FALSE-FREE  : est > gt + tol  (claims free space through an obstacle; dangerous)
              FALSE-OCC   : est < gt - tol  (phantom obstacle; stalls the planner)
              free coverage = sum(min(est,gt,4 m) over rays not false-free) / sum(min(gt,4 m)).
Surfaces      textureless (7x7 std of the clean image < 3 grey levels), textured, depth edge,
              thin (2-4 cm poles, 1-2 cm rods), floor / wall / ceiling (GT normal).
Voxel map     per sequence, depths are fused into a 0.125 m log-odds grid with the poses the
              method saw (VO, noisy); GT map = GT depth fused with GT poses. Region: voxels within
              3 m of the trajectory that the GT map observed. Free carving stops 0.15+0.05z before
              the hit; range cap 5 m.
"""
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

BX, BY = 20, 20
VOX = 0.125
L_OCC, L_FREE = 0.85, -0.4


def tol(z):
    return np.maximum(0.10, 0.15 * z)


def surface_masks(fr):
    z, g = fr["depth"], cv2.cvtColor(fr["left_clean"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.boxFilter(g, -1, (7, 7)); sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (7, 7)) - m * m, 0))
    zf = np.where(np.isfinite(z), z, 50.0)
    lz = np.log(zf)
    edge = (np.abs(cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3))) > 0.4
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    thin = fr["thin"].astype(bool)
    n = fr["normal"]
    return dict(textureless=(sd < 3) & ~thin & ~edge, textured=(sd >= 3) & ~thin & ~edge, edge=edge & ~thin, thin=thin,
                floor=n == 1, wall=(n == 2) & ~thin, ceiling=n == 3)


class Accum:
    def __init__(self):
        self.c = {}

    def add(self, k, num, den=1):
        a = self.c.setdefault(k, [0.0, 0.0])
        a[0] += float(num); a[1] += float(den)

    def rate(self, k):
        a = self.c.get(k, [0, 0])
        return a[0] / a[1] if a[1] else float("nan")


def frame_metrics(acc, est, fr, kvals):
    z = fr["depth"]
    gv = np.isfinite(z) & (z > 0.2) & (z < 12)
    ev = np.isfinite(est) & (est > 0)
    t = tol(np.where(gv, z, 1))
    ok = gv & ev & (np.abs(est - z) <= t)
    fn = gv & ev & (est < z - t)
    ff = gv & ev & (est > z + t)
    acc.add("px_hole", (gv & ~ev).sum(), gv.sum())
    acc.add("px_false_near", fn.sum(), (gv & ev).sum())
    acc.add("px_false_far", ff.sum(), (gv & ev).sum())
    for lo, hi in ((0.2, 1), (1, 2), (2, 4), (4, 8)):
        s = gv & (z >= lo) & (z < hi)
        acc.add(f"valid_{lo}-{hi}", (s & ev).sum(), s.sum())
        acc.add(f"ok_{lo}-{hi}", (s & ok).sum(), s.sum())
        acc.add(f"fnear_{lo}-{hi}", (s & fn).sum(), (s & ev).sum())
        if (s & ok).any():
            e = np.abs(est - z)[s & ok]
            acc.add(f"abserr_{lo}-{hi}", e.sum(), e.size)
    s = gv & ok & (z >= 1) & (z < 6)
    if s.any():
        kvals.append((np.abs(est - z)[s] / z[s] ** 2)[::7])
    for nm, m in surface_masks(fr).items():
        mm = m & gv
        acc.add(f"hole_{nm}", (mm & ~ev).sum(), mm.sum())
        acc.add(f"ok_{nm}", (mm & ok).sum(), mm.sum())
    # rays
    H, W = z.shape
    zb = np.where(gv, z, np.nan).reshape(H // BY, BY, W // BX, BX).transpose(0, 2, 1, 3).reshape(H // BY, W // BX, -1)
    eb = np.where(ev, est, np.nan).reshape(H // BY, BY, W // BX, BX).transpose(0, 2, 1, 3).reshape(H // BY, W // BX, -1)
    thb = fr["thin"].reshape(H // BY, BY, W // BX, BX).transpose(0, 2, 1, 3).reshape(H // BY, W // BX, -1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = np.nanpercentile(zb, 5, axis=-1)
            frac = np.isfinite(eb).mean(-1)
            e = np.nanpercentile(eb, 5, axis=-1)
    known = np.isfinite(g)
    unk = known & ((frac < 0.10) | ~np.isfinite(e))
    kn = known & ~unk
    tg = tol(np.nan_to_num(g, nan=1))
    rff = kn & (e > g + tg)
    rfo = kn & (e < g - tg)
    near = known & (g < 1.5)
    acc.add("ray_unknown", unk.sum(), known.sum())
    acc.add("ray_false_free", rff.sum(), known.sum())
    acc.add("ray_false_occ", rfo.sum(), known.sum())
    acc.add("ray_ok", (kn & ~rff & ~rfo).sum(), known.sum())
    acc.add("ray_false_free_near", (rff & near).sum(), near.sum())
    acc.add("ray_unknown_near", (unk & near).sum(), near.sum())
    acc.add("ray_false_occ_near", (rfo & near).sum(), near.sum())
    cap = 4.0
    gc = np.minimum(np.nan_to_num(g, nan=0), cap)
    cov = np.where(kn & ~rff, np.minimum(np.nan_to_num(e, nan=0), gc), 0)
    acc.add("free_coverage", cov[known].sum(), gc[known].sum())
    # thin rays: block's nearest GT surface is a thin object
    thin_ray = known & (thb.sum(-1) >= 8)
    zmin = np.nanmin(np.where(thb.astype(bool), zb, np.nan), axis=-1) if thin_ray.any() else None
    if thin_ray.any():
        thin_ray &= np.abs(zmin - g) < 0.1
        acc.add("thin_ray_detected", (thin_ray & kn & ~rff).sum(), thin_ray.sum())
        acc.add("thin_ray_false_free", (thin_ray & rff).sum(), thin_ray.sum())


# ---------------------------------------------------------------------------------------------- voxel fusion
def _rays_world(z, pose, f, cx, cy, step=4):
    H, W = z.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    zz = z[ys, xs]
    v = np.isfinite(zz) & (zz > 0.05)
    xs, ys, zz = xs[v] + 0.5, ys[v] + 0.5, zz[v]
    d = np.stack([np.ones_like(zz), (cx - xs) / f, (cy - ys) / f], -1)
    dw = d @ pose[:3, :3].T
    return pose[:3, 3], dw, zz


def fuse(depths, poses, calib, cap=5.0):
    f, cx, cy = calib["f"], calib["cx"], calib["cy"]
    lo = {}
    keys_all, vals_all = [], []
    for z, P in zip(depths, poses):
        o, dw, zz = _rays_world(z, P, f, cx, cy)
        if not len(zz):
            continue
        hit = zz <= cap
        carve = np.minimum(zz - (0.15 + 0.05 * zz), cap)
        ts = np.arange(0.05, cap, VOX / 2)
        m = ts[None, :] < carve[:, None]
        pts = o[None, None] + ts[None, :, None] * dw[:, None, :]
        fk = np.unique(_key(pts[m]))
        hp = o[None] + (zz[hit, None]) * dw[hit]
        hk = np.unique(_key(hp))
        fk = np.setdiff1d(fk, hk)
        keys_all += [fk, hk]; vals_all += [np.full(len(fk), L_FREE), np.full(len(hk), L_OCC)]
    if not keys_all:
        return np.zeros(0, np.int64), np.zeros(0)
    k = np.concatenate(keys_all); v = np.concatenate(vals_all)
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, v)


OFF = 1 << 20


def _key(p):
    i = np.floor(p / VOX).astype(np.int64) + 4096
    return (i[..., 0] << 26) | (i[..., 1] << 13) | i[..., 2]


def _unkey(k):
    return np.stack([(k >> 26) & 8191, (k >> 13) & 8191, k & 8191], -1).astype(np.float64) - 4096


def voxel_metrics(acc, est_depths, vo_poses, gt_depths, gt_poses, calib):
    gk, gl = fuse(gt_depths, gt_poses, calib)
    mk, ml = fuse(est_depths, vo_poses, calib)
    centers = (_unkey(gk) + 0.5) * VOX
    traj = gt_poses[:, :3, 3]
    dist = np.min(np.linalg.norm(centers[:, None] - traj[None], axis=-1), 1)
    reg = dist < 3.0
    g_occ = (gl > 0.5) & reg
    g_free = (gl < -0.3) & reg
    idx = np.searchsorted(mk, gk); idx = np.clip(idx, 0, max(len(mk) - 1, 0))
    present = (len(mk) > 0) & (mk[idx] == gk) if len(mk) else np.zeros(len(gk), bool)
    mlv = np.where(present, ml[idx] if len(mk) else 0, 0)
    m_occ = present & (mlv > 0.5)
    m_free = present & (mlv < -0.3)
    # occupied match tolerant to one voxel: dilate method occupancy
    occ_keys = mk[ml > 0.5] if len(mk) else np.zeros(0, np.int64)
    dil = set()
    if len(occ_keys):
        offs = [(a << 26) + (b << 13) + c for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        dil = np.unique((occ_keys[:, None] + np.array(offs)[None]).ravel())
    m_occ_d = np.isin(gk, dil) if len(occ_keys) else np.zeros(len(gk), bool)
    acc.add("vox_occ_recall", (g_occ & m_occ_d).sum(), g_occ.sum())
    acc.add("vox_occ_iou", (g_occ & m_occ).sum(), (g_occ | (m_occ & reg & (g_free | g_occ))).sum())
    acc.add("vox_free_iou", (g_free & m_free).sum(), (g_free | (m_free & reg & (g_free | g_occ))).sum())
    acc.add("vox_false_free", (g_occ & m_free).sum(), g_occ.sum())
    near = dist < 1.0
    acc.add("vox_false_free_1m", (g_occ & m_free & near).sum(), 1)  # count per sequence
    acc.add("vox_false_occ_1m", (g_free & m_occ & near & ~g_occ).sum(), (g_free & near).sum())
    acc.add("vox_unknown_1m", (g_free & near & ~m_free & ~m_occ).sum(), (g_free & near).sum())


# ---------------------------------------------------------------------------------------------- nav-view metrics
# Stopping distance: the shield's stop margin + distance flown during the frame's age + braking.
#   d_stop(v) = 0.3 m + v * 0.4 s + v^2 / (2 * 1.5 m/s^2)
# (NT_SHIELD_STOP, frame age ~ latency + half a period + Pi processing, NT_SHIELD_BRAKE in nav_task.h).
# The student's route speed is 1 m/s (NT_ROUTE_SPEED) -> 1.03 m; the command cap is 2 m/s -> 2.43 m.
STOP_MARGIN, STOP_AGE, STOP_BRAKE = 0.3, 0.4, 1.5
RANGE_CAP = 3.0
BANDS = ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99))


def d_stop(v):
    return STOP_MARGIN + v * STOP_AGE + v * v / (2 * STOP_BRAKE)


def _blocks(a, H, W):
    return a.reshape(H // BY, BY, W // BX, BX).transpose(0, 2, 1, 3).reshape(H // BY, W // BX, -1)


def nav_frame_metrics(acc, est, fr, cap=None):
    """Ray metrics for planning toward a waypoint.

    est: depth with NaN = unknown and +inf = "matched, but beyond the method's range cap" (known free
    up to `cap`). A far ray is false-free only if GT has an obstacle clearly inside the cap.
    Unknown rays are split by cause: textureless (>= half the block is textureless in the clean image),
    beyond range (GT > 3 m), else occluded / L-R / other; and by GT distance band.
    Stopping-distance metrics use only obstacle rays (the block's nearest surface is not floor/ceiling).
    """
    z = fr["depth"]
    H, W = z.shape
    gv = np.isfinite(z) & (z > 0.2) & (z < 12)
    ev = (np.isfinite(est) & (est > 0)) | np.isposinf(est)
    zb = _blocks(np.where(gv, z, np.nan), H, W)
    eb = _blocks(np.where(ev, est, np.nan), H, W)
    masks = surface_masks(fr)
    texless = _blocks(masks["textureless"].astype(np.float32), H, W).mean(-1) >= 0.5
    nb = _blocks(fr["normal"].astype(np.float32), H, W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = np.nanpercentile(zb, 5, axis=-1)
        frac = np.isfinite(eb).mean(-1) + np.isposinf(eb).mean(-1)
        e = np.nanpercentile(np.where(np.isposinf(eb), 1e9, eb), 5, axis=-1)
    # surface of the nearest GT pixels in the block
    near_px = zb <= (g[..., None] + 0.05)
    flc = ((nb == 1) | (nb == 3)) & near_px
    obstacle = ~(flc.sum(-1) > 0.5 * np.maximum(near_px.sum(-1), 1))
    known = np.isfinite(g)
    thin_ray = known & (_blocks(fr["thin"], H, W).sum(-1) >= 8)
    unk = known & ((frac < 0.10) | ~(e > 0))
    far = known & ~unk & (e >= 1e8)
    kn = known & ~unk & ~far
    gg = np.nan_to_num(g, nan=1)
    tg = tol(gg)
    capv = cap if cap else RANGE_CAP
    rff = (kn & (e > gg + tg)) | (far & (gg + tg < capv))
    rfo = kn & (e < gg - tg)
    acc.add("nav_unknown", unk.sum(), known.sum())
    acc.add("nav_far_free", far.sum(), known.sum())
    acc.add("nav_false_free", rff.sum(), known.sum())
    acc.add("nav_false_occ", rfo.sum(), known.sum())
    near15 = known & (g < 1.5)
    acc.add("nav_ff_1.5", (rff & near15).sum(), near15.sum())
    acc.add("nav_ff_1.5_solid", (rff & near15 & ~thin_ray).sum(), (near15 & ~thin_ray).sum())
    acc.add("nav_thin_detected", (thin_ray & ~unk & ~rff).sum(), thin_ray.sum())
    in3 = known & (g < 3.0)
    acc.add("nav_fo_3m", (rfo & in3).sum(), in3.sum())
    # unknown by cause (fractions of all known rays; they add up to nav_unknown)
    c_tex = unk & texless
    c_far = unk & ~texless & (g > RANGE_CAP)
    c_occ = unk & ~texless & ~(g > RANGE_CAP)
    acc.add("unk_textureless", c_tex.sum(), known.sum())
    acc.add("unk_beyond_range", c_far.sum(), known.sum())
    acc.add("unk_occluded_other", c_occ.sum(), known.sum())
    acc.add("rays_textureless", (known & texless).sum(), known.sum())
    for lo, hi in BANDS:
        s = known & (g >= lo) & (g < hi)
        acc.add(f"unk_band_{lo}-{hi}", (unk & s).sum(), s.sum())
        acc.add(f"share_band_{lo}-{hi}", s.sum(), known.sum())
    for v in (1.0, 2.0):
        ds = d_stop(v)
        s = known & obstacle & (g < ds)
        acc.add(f"stop{v:.0f}_ff", (rff & s).sum(), s.sum())
        acc.add(f"stop{v:.0f}_unknown", (unk & s).sum(), s.sum())
        acc.add(f"stop{v:.0f}_fo", (rfo & s).sum(), s.sum())
        acc.add(f"stop{v:.0f}_rays", s.sum(), known.sum())
        st = s & thin_ray
        acc.add(f"stop{v:.0f}_ff_thin", (rff & st).sum(), st.sum())
        acc.add(f"stop{v:.0f}_ff_solid", (rff & s & ~thin_ray).sum(), (s & ~thin_ray).sum())
        acc.add(f"stop{v:.0f}_unk_textureless", (unk & s & texless).sum(), s.sum())

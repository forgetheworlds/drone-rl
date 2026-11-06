"""Depth pipelines under test. Plugin interface:

    class MyMethod(Method):
        name = "my_method"; baseline = 0.06          # which stereo pair to receive (0.06 or 0.12)
        def reset(self, calib): ...                  # start of a sequence (stateful methods)
        def process(self, frame, calib) -> dict:     # frame: left, right (uint8 RGB HxWx3), pose (4x4 VO)
            return dict(depth=HxW float32 metres, NaN = unknown, conf=optional HxW 0..1)

The harness never passes ground truth to a method: `frame` only has left, right, pose (noisy VO,
camera-to-world, columns forward/left/up) and k. calib: f, cx, cy, baseline, W, H, dt.
Register with @register. Anything returning lower resolution is upsampled (nearest) by the harness.
"""
import math

import cv2
import numpy as np

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


class Method:
    name = "base"
    baseline = 0.06
    note = ""

    def reset(self, calib):
        pass

    def process(self, frame, calib):
        raise NotImplementedError


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def disp_to_depth(d, fB, dmin=0.5):
    z = np.full(d.shape, np.nan, np.float32)
    v = d > dmin
    z[v] = fB / d[v]
    return z


def texture_mask(g, thr, k=5):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    return cv2.boxFilter(np.abs(gx), -1, (k, k)) >= thr


# ----------------------------------------------------------------------------------------------
@register
class SGBMBaseline(Method):
    """hyperdrone_eval/stereo_sgbm.py settings: SGBM 3-way, 64 disp, block 5, no extra filtering."""
    name = "sgbm320"
    note = "baseline: OpenCV SGBM 3-way, 64 disp, block 5, uniq 10, speckle 50"

    def __init__(self):
        self.m = cv2.StereoSGBM_create(0, 64, 5, P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=1, uniquenessRatio=10,
                                       speckleWindowSize=50, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def process(self, frame, calib):
        d = self.m.compute(gray(frame["left"]), gray(frame["right"])).astype(np.float32) / 16
        d[:, :64] = 0  # no data in the first numDisparities columns (as stereo_sgbm.py's ROI)
        return dict(depth=disp_to_depth(d, calib["f"] * calib["baseline"]))


class FilteredSGBM(Method):
    """SGBM + left-right consistency + uniqueness + texture gate + min-disparity + speckle removal."""
    scale = 1.0
    nd = 64
    block = 5
    tex = 12.0
    lr_tol = 1.0
    uniq = 15
    dmin = 2.0       # disparity floor -> max range fB/dmin (7.4 m at 6 cm, 320 px)
    clahe = True

    def __init__(self):
        b = self.block
        self.m = cv2.StereoSGBM_create(0, self.nd, b, P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=-1,
                                       uniquenessRatio=self.uniq, speckleWindowSize=100, speckleRange=1,
                                       preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.r = cv2.ximgproc.createRightMatcher(self.m)
        self.cl = cv2.createCLAHE(2.0, (4, 4))

    def prep(self, img):
        g = gray(img)
        if self.scale != 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return self.cl.apply(g) if self.clahe else g

    def disparity(self, frame):
        L, R = self.prep(frame["left"]), self.prep(frame["right"])
        dl = self.m.compute(L, R).astype(np.float32) / 16
        dr = -self.r.compute(R, L).astype(np.float32) / 16
        return L, R, dl, dr

    def lr_check(self, dl, dr):
        h, w = dl.shape
        xs = np.arange(w)[None].repeat(h, 0)
        xr = np.clip(np.round(xs - dl).astype(int), 0, w - 1)
        back = dr[np.arange(h)[:, None], xr]
        return (dl > 0) & (np.abs(back - dl) <= self.lr_tol)

    def filtered(self, frame, calib):
        L, R, dl, dr = self.disparity(frame)
        ok = self.lr_check(dl, dr) & texture_mask(L, self.tex) & (dl >= self.dmin * self.scale)
        ok[:, :self.nd] = False
        d = np.where(ok, dl, 0).astype(np.float32)
        return L, d

    def process(self, frame, calib):
        L, d = self.filtered(frame, calib)
        return dict(depth=disp_to_depth(d, calib["f"] * self.scale * calib["baseline"]))


@register
class SGBMFiltered320(FilteredSGBM):
    name = "sgbm320_lrc_tex"
    note = "SGBM + CLAHE + L/R check (1 px) + uniq 15 + Sobel texture gate + d>=2"


@register
class SGBMFiltered160(FilteredSGBM):
    name = "sgbm160_lrc_tex"
    scale, nd, block, tex = 0.5, 32, 3, 10.0
    note = "same at 160x120 (4x cheaper)"


@register
class SGBMFiltered320B12(FilteredSGBM):
    name = "sgbm320_lrc_tex_b12"
    baseline, nd = 0.12, 96
    note = "12 cm baseline, 96 disp"


@register
class SGBMFiltered160B12(FilteredSGBM):
    name = "sgbm160_lrc_tex_b12"
    baseline, scale, nd, block, tex = 0.12, 0.5, 48, 3, 10.0
    note = "12 cm baseline at 160x120"


class BMBase(Method):
    """StereoBM (what a Pi Zero / P4 can run in real time) + L/R check, at a chosen resolution."""
    W, H, nd, block, tex, uniq = 160, 120, 32, 9, 10, 15

    def __init__(self):
        self.m = cv2.StereoBM_create(self.nd, self.block)
        self.m.setTextureThreshold(self.tex); self.m.setUniquenessRatio(self.uniq); self.m.setSpeckleWindowSize(50)
        self.m.setSpeckleRange(1); self.m.setPreFilterCap(31); self.m.setDisp12MaxDiff(-1)
        self.r = cv2.ximgproc.createRightMatcher(self.m)

    def process(self, frame, calib):
        L = cv2.resize(gray(frame["left"]), (self.W, self.H), interpolation=cv2.INTER_AREA)
        R = cv2.resize(gray(frame["right"]), (self.W, self.H), interpolation=cv2.INTER_AREA)
        dl = self.m.compute(L, R).astype(np.float32) / 16
        dr = -self.r.compute(R, L).astype(np.float32) / 16
        h, w = dl.shape
        xr = np.clip(np.round(np.arange(w)[None] - dl).astype(int), 0, w - 1)
        ok = (dl >= 1.0) & (np.abs(dr[np.arange(h)[:, None], xr] - dl) <= 1.0)
        ok[:, :self.nd] = False
        s = self.W / calib["W"]
        return dict(depth=disp_to_depth(np.where(ok, dl, 0), calib["f"] * s * calib["baseline"]))


@register
class BM160(BMBase):
    name = "bm160_lrc"
    note = "StereoBM 160x120, 32 disp, block 9, texture 10, uniq 15, L/R check"


@register
class BM160B12(BMBase):
    name = "bm160_lrc_b12"
    baseline, nd = 0.12, 48
    note = "StereoBM 160x120 at 12 cm baseline (48 disp)"


@register
class BM320(BMBase):
    name = "bm320_lrc"
    W, H, nd, block = 320, 240, 64, 11
    note = "StereoBM 320x240, 64 disp, block 11, L/R check"


@register
class WLS320(FilteredSGBM):
    """SGBM + DisparityWLSFilter (edge-aware fill), gated by the WLS confidence map."""
    name = "sgbm320_wls_conf"
    note = "SGBM + WLS filter (lambda 8000, sigma 1.5), keep WLS confidence > 0.6"
    gate = 0.6

    def process(self, frame, calib):
        L, R, _, _ = self.disparity(frame)
        dl16 = self.m.compute(L, R); dr16 = self.r.compute(R, L)
        wls = cv2.ximgproc.createDisparityWLSFilter(self.m)
        wls.setLambda(8000.0); wls.setSigmaColor(1.5)
        d = wls.filter(dl16, L, disparity_map_right=dr16).astype(np.float32) / 16
        conf = wls.getConfidenceMap().astype(np.float32) / 255.0
        ok = (conf > self.gate) & (d >= self.dmin)
        ok[:, :self.nd] = False
        return dict(depth=disp_to_depth(np.where(ok, d, 0), calib["f"] * calib["baseline"]), conf=conf)


@register
class WLS320Dense(WLS320):
    name = "sgbm320_wls_dense"
    note = "WLS filled output, no confidence gate (fills textureless walls by propagation)"
    gate = -1.0


# ---------------------------------------------------------------------------------------------- geometry helpers
def backproject(z, f, cx, cy):
    h, w = z.shape
    x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([z, z * (cx - x) / f, z * (cy - y) / f], -1)  # camera FLU


def project(p, f, cx, cy):
    z = p[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx - f * p[..., 1] / z
        v = cy - f * p[..., 2] / z
    return u, v, z


def warp_depth(z_prev, pose_prev, pose_cur, f, cx, cy, shape):
    """Splat a previous depth map into the current camera (nearest wins)."""
    h, w = shape
    v = np.isfinite(z_prev)
    p = backproject(np.where(v, z_prev, 0), f, cx, cy)[v]
    T = np.linalg.inv(pose_cur) @ pose_prev
    q = p @ T[:3, :3].T + T[:3, 3]
    u, vv, zz = project(q, f, cx, cy)
    ok = (zz > 0.1) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
    ui, vi, zz = u[ok].astype(int), vv[ok].astype(int), zz[ok]
    out = np.full((h, w), np.inf, np.float32)
    np.minimum.at(out, (vi, ui), zz)
    out[np.isinf(out)] = np.nan
    return out


@register
class TemporalVote(SGBMFiltered320):
    """Filtered SGBM, then keep a pixel only if a VO-warped previous frame agrees (kills one-off
    false near returns); fill holes from history when two past frames agree with each other."""
    name = "sgbm320_temporal"
    note = "filtered SGBM + 3-frame VO-warped vote (keep if >=1 past agrees) + history fill (2 agree)"
    K = 2
    rel = 0.12

    def reset(self, calib):
        self.hist = []

    def process(self, frame, calib):
        return dict(depth=self.vote(super().process(frame, calib)["depth"], frame, calib))

    def vote(self, z, frame, calib):
        f, cx, cy = calib["f"], calib["cx"], calib["cy"]
        warped = [warp_depth(zp, pp, frame["pose"], f, cx, cy, z.shape) for zp, pp in self.hist]
        warped = [cv2.dilate(np.nan_to_num(-w, nan=-1e3), np.ones((3, 3))) for w in warped]  # nearest-in-3x3
        warped = [np.where(w > -1e2, -w, np.nan) for w in warped]
        out = z.copy()
        if warped:
            agree = np.zeros(z.shape, bool); seen = np.zeros(z.shape, bool)
            for w in warped:
                vw = np.isfinite(w)
                seen |= vw
                agree |= vw & (np.abs(w - z) <= self.rel * np.maximum(z, 1.0))
            # contradicted: history saw something *farther* there and nothing agrees -> drop
            out[np.isfinite(z) & seen & ~agree] = np.nan
            if len(warped) >= 2:
                a, b = warped[-1], warped[-2]
                fill = ~np.isfinite(z) & np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= self.rel * np.maximum(a, 1.0))
                out[fill] = np.minimum(a, b)[fill]
        self.hist = (self.hist + [(z, frame["pose"])])[-self.K:]
        return out


@register
class PlaneFill(SGBMFiltered320):
    """Filtered SGBM + fill large textureless holes with a RANSAC plane fitted to the hole border."""
    name = "sgbm320_planefill"
    note = "filtered SGBM + RANSAC plane fill of textureless holes from their valid border"

    def process(self, frame, calib):
        z = super().process(frame, calib)["depth"]
        return dict(depth=plane_fill(z, gray(frame["left"]), calib))


def plane_fill(z, g, calib, min_area=300, rng=np.random.default_rng(0)):
    f, cx, cy = calib["f"], calib["cx"], calib["cy"]
    hole = (~np.isfinite(z)).astype(np.uint8)
    hole[:, :64] = 0
    smooth = ~texture_mask(g, 6.0, 7)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole & smooth.astype(np.uint8), connectivity=4)
    out = z.copy()
    P = backproject(np.nan_to_num(z), f, cx, cy)
    h, w = z.shape
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([np.ones_like(xs), (cx - xs) / f, (cy - ys) / f], -1)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = (lab == i).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((7, 7), np.uint8)).astype(bool) & ~m.astype(bool) & np.isfinite(z)
        pts = P[ring]
        if len(pts) < 40:
            continue
        best, bn = None, 0
        for _ in range(60):
            s = pts[rng.choice(len(pts), 3, replace=False)]
            nrm = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(nrm)
            if nn < 1e-6:
                continue
            nrm /= nn; dd = -nrm @ s[0]
            res = np.abs(pts @ nrm + dd)
            inl = (res < 0.03 + 0.01 * pts[:, 0] ** 2).sum()
            if inl > bn:
                bn, best = inl, (nrm, dd)
        if best is None or bn < 0.85 * len(pts):
            continue
        nrm, dd = best
        sel = m.astype(bool)
        denom = rays[sel] @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = -dd / denom
        lo, hi = np.percentile(pts[:, 0], 1), np.percentile(pts[:, 0], 99)
        okz = np.isfinite(zz) & (zz > 0.2) & (zz >= lo * 0.8) & (zz <= hi * 1.2)
        tmp = out[sel]; tmp[okz] = zz[okz]; out[sel] = tmp
    return out


# ---------------------------------------------------------------------------------------------- plane sweep
@register
class PlaneSweepMV(Method):
    """Multi-view plane sweep at 160x120: reference = left_k; sources = right_k, and left/right of
    frames k-1, k-2 via VO poses (motion stereo widens the baseline). ZNCC-like cost on
    zero-mean-normalised images, 5x5 box aggregation, winner-take-all + ratio test."""
    name = "planesweep_mv160"
    note = "multi-view plane sweep (stereo pair + 2 past frames, VO poses), 64 inv-depth planes"
    S = 0.5
    planes = 1.0 / np.linspace(1 / 0.3, 1 / 8.0, 64)

    def reset(self, calib):
        self.hist = []

    def prep(self, img):
        g = cv2.resize(gray(img), None, fx=self.S, fy=self.S, interpolation=cv2.INTER_AREA).astype(np.float32)
        m = cv2.boxFilter(g, -1, (7, 7)); s = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (7, 7)) - m * m, 4.0))
        return (g - m) / s, s

    def process(self, frame, calib):
        f = calib["f"] * self.S; cx = calib["cx"] * self.S; cy = calib["cy"] * self.S
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        # camera optical frame: x right, y down, z forward;  FLU -> optical
        C = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
        ref, sref = self.prep(frame["left"])
        Pl = frame["pose"]
        Pr = Pl.copy(); Pr[:3, 3] = Pl[:3, 3] - calib["baseline"] * Pl[:3, 1]
        views = [(self.prep(frame["right"])[0], Pr)]
        for (l, r, P) in self.hist:
            views.append((l, P))
            P2 = P.copy(); P2[:3, 3] = P[:3, 3] - calib["baseline"] * P[:3, 1]
            views.append((r, P2))
        h, w = ref.shape
        cost = np.zeros((len(self.planes), h, w), np.float32)
        cnt = np.zeros((len(self.planes), h, w), np.float32)
        n = np.array([0, 0, 1.0])  # plane normal in ref optical frame (fronto-parallel)
        for img, P in views:
            T = np.linalg.inv(P) @ Pl                       # ref FLU -> src FLU
            R = C @ T[:3, :3] @ C.T; t = C @ T[:3, 3]
            for i, d in enumerate(self.planes):
                Hm = K @ (R + np.outer(t, n) / d) @ np.linalg.inv(K)   # ref pixel -> src pixel for plane z=d
                wimg = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                v = np.isfinite(wimg)
                c = np.where(v, (ref - np.nan_to_num(wimg)) ** 2, 0)
                cost[i] += c; cnt[i] += v
        cost = cost / np.maximum(cnt, 1)
        cost[cnt < 1] = 1e3
        for i in range(len(self.planes)):
            cost[i] = cv2.boxFilter(cost[i], -1, (5, 5))
        best = np.argmin(cost, 0)
        c1 = np.take_along_axis(cost, best[None], 0)[0]
        cost2 = cost.copy()
        for o in (-2, -1, 0, 1, 2):
            idx = np.clip(best + o, 0, len(self.planes) - 1)
            np.put_along_axis(cost2, idx[None], 1e3, 0)
        c2 = cost2.min(0)
        ok = (c1 < 0.8 * c2) & (sref > 3.0) & (c1 < 0.6)
        z = np.where(ok, self.planes[best], np.nan).astype(np.float32)
        self.hist = (self.hist + [(ref, self.prep(frame["right"])[0], Pl)])[-2:]
        return dict(depth=z)


@register
class Fused(SGBMFiltered320):
    """Best-of combination: filtered SGBM + temporal vote/fill + plane fill (applied last)."""
    name = "sgbm320_temporal_planefill"
    note = "temporal vote/fill, then RANSAC plane fill"

    def reset(self, calib):
        self.t = TemporalVote(); self.t.reset(calib)

    def process(self, frame, calib):
        z = self.t.process(frame, calib)["depth"]
        return dict(depth=plane_fill(z, gray(frame["left"]), calib))


class TemporalMixin:
    """Wrap any per-frame method: keep a pixel only if a VO-warped past frame doesn't contradict it,
    fill holes where the last two warped frames agree (see TemporalVote)."""
    K, rel = 2, 0.12

    def reset(self, calib):
        self.hist = []

    def process(self, frame, calib):
        z = super().process(frame, calib)["depth"]
        if z.shape != (calib["H"], calib["W"]):
            z = cv2.resize(z, (calib["W"], calib["H"]), interpolation=cv2.INTER_NEAREST)
        return dict(depth=TemporalVote.vote(self, z, frame, calib))


@register
class BM160Temporal(TemporalMixin, BM160):
    name = "bm160_temporal"
    note = "bm160_lrc + 3-frame VO-warped vote / history fill"


@register
class BM160B12Temporal(TemporalMixin, BM160B12):
    name = "bm160_b12_temporal"
    note = "bm160_lrc_b12 + 3-frame VO-warped vote / history fill"


# ---------------------------------------------------------------------------------------------- map memory
class MapRaycastMixin:
    """Fuse every frame so far into the 0.125 m log-odds map (same fusion as the voxel metric, with VO
    poses) and fill the current frame's holes by raycasting that map: first occupied voxel -> depth;
    if the ray leaves known-free space into unknown first, it stays unknown. Current-frame depth wins
    where it exists. This is "what the policy sees" when its rays come from the map, not one frame."""
    step = 5            # raycast every 5 px (64x48 rays), nearest-upsampled
    occ_thr = 0.5
    free_thr = -0.3

    def reset(self, calib):
        super().reset(calib)
        self.mem = []

    def process(self, frame, calib):
        import metrics
        z = super().process(frame, calib)["depth"]
        if z.shape != (calib["H"], calib["W"]):
            z = cv2.resize(z, (calib["W"], calib["H"]), interpolation=cv2.INTER_NEAREST)
        self.mem.append((z, frame["pose"]))
        keys, lo = metrics.fuse([m[0] for m in self.mem], [m[1] for m in self.mem], calib)
        if not len(keys):
            return dict(depth=z)
        H, W, f, cx, cy = calib["H"], calib["W"], calib["f"], calib["cx"], calib["cy"]
        ys, xs = np.mgrid[self.step // 2:H:self.step, self.step // 2:W:self.step]
        d = np.stack([np.ones(xs.shape), (cx - xs - 0.5) / f, (cy - ys - 0.5) / f], -1).reshape(-1, 3)
        P = frame["pose"]
        dw = d @ P[:3, :3].T
        ts = np.arange(0.1, 5.0, metrics.VOX / 2)
        pk = metrics._key(P[:3, 3][None, None] + ts[None, :, None] * dw[:, None, :])
        idx = np.clip(np.searchsorted(keys, pk), 0, len(keys) - 1)
        l = np.where(keys[idx] == pk, lo[idx], 0.0)
        occ = l > self.occ_thr
        notfree = l >= self.free_thr                     # occupied or unknown
        first = np.argmax(notfree, 1)
        anyn = notfree.any(1)
        hit = anyn & occ[np.arange(len(first)), first]
        # depth = forward distance of the mean surface point stored in the hit voxel (not the voxel
        # boundary, which at grazing angles on floors/walls is far too near)
        pts = []
        for zm, Pm in self.mem:
            o, dwm, zz = metrics._rays_world(zm, Pm, f, cx, cy)
            pts.append(o[None] + zz[:, None] * dwm)
        pts = np.concatenate(pts)
        pk2 = metrics._key(pts)
        u, inv = np.unique(pk2, return_inverse=True)
        cnt = np.bincount(inv)
        mean = np.stack([np.bincount(inv, pts[:, i]) for i in range(3)], -1) / cnt[:, None]
        hk = pk[np.arange(len(first)), first]
        j = np.clip(np.searchsorted(u, hk), 0, len(u) - 1)
        have = hit & (u[j] == hk)
        tz = (mean[j] - P[:3, 3]) @ P[:3, 0]                # forward (planar) depth in the current camera
        rz = np.where(have, np.maximum(tz, 0.05), np.nan).reshape(xs.shape).astype(np.float32)
        mz = cv2.resize(rz, (W, H), interpolation=cv2.INTER_NEAREST)
        return dict(depth=np.where(np.isfinite(z), z, mz))


@register
class BM160Map(MapRaycastMixin, TemporalMixin, BM160):
    name = "bm160_temporal_map"
    note = "bm160_temporal + holes filled by raycasting the fused log-odds map (VO poses)"


@register
class SGBM320Map(MapRaycastMixin, SGBMFiltered320):
    name = "sgbm320_lrc_tex_map"
    note = "sgbm320_lrc_tex + holes filled by raycasting the fused log-odds map (VO poses)"


class RangeCapMixin:
    """Drop estimates beyond zmax: past ~3 m a 6 cm / 160 px pair is under 2.5 px of disparity and most
    'valid' pixels are false near (see fnear_4-8 in the leaderboard)."""
    zmax = 3.0

    def process(self, frame, calib):
        z = super().process(frame, calib)["depth"]
        # far: matched, but beyond the cap. The nav metrics read it as "free up to zmax".
        return dict(depth=np.where(z <= self.zmax, z, np.nan), far=np.isfinite(z) & (z > self.zmax))


@register
class BM160TemporalR3(RangeCapMixin, BM160Temporal):
    name = "bm160_temporal_r3"
    note = "bm160_temporal, estimates beyond 3 m dropped"


@register
class BM160B12R5(RangeCapMixin, BM160B12):
    name = "bm160_b12_r5"
    zmax = 5.0
    note = "bm160_lrc_b12, estimates beyond 5 m dropped"


@register
class SGBM320TemporalR3(RangeCapMixin, TemporalVote):
    name = "sgbm320_temporal_r3"
    note = "sgbm320_temporal, estimates beyond 3 m dropped"


# ---------------------------------------------------------------------------------------------- edge-anchored plane fill
def edge_plane_fill(z, g, calib, far=None, zmax=3.0, min_area=150, cover_min=0.6, sides_min=3,
                    inlier_min=0.9, rng=None, stats=None):
    """Fill a textureless hole with a plane only when its own border vouches for it.

    Holes = unknown pixels whose image is textureless (no matcher could have found them). Anchors =
    measured depth (<= zmax) within 4 px outside the hole. Fill only if
      - anchors cover >= cover_min of the hole's border (image-edge border excluded) and lie on
        >= sides_min of the 4 sides around the hole's centre,
      - one RANSAC plane explains >= inlier_min of the anchors (residual 2 cm + 1% z^2),
      - the plane is not grazing (|cos| between plane normal and the hole's centre ray > 0.25),
    and only write pixels whose plane depth is no farther than the farthest inlier anchor (and no
    nearer than 0.8x the nearest). A hole with an object in front of part of its border fails the
    single-plane test and stays unknown.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    f, cx, cy = calib["f"], calib["cx"], calib["cy"]
    h, w = z.shape
    valid = np.isfinite(z) & (z > 0.1) & (z <= zmax)
    hole = ~np.isfinite(z)
    if far is not None:
        hole &= ~far
    smooth = ~texture_mask(g, 6.0, 7)
    cand = (hole & smooth).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand, connectivity=4)
    out = z.copy()
    if n <= 1:
        return out
    P = backproject(np.nan_to_num(z), f, cx, cy)
    border_img = np.zeros((h, w), bool); border_img[[0, -1], :] = True; border_img[:, [0, -1]] = True
    k3, k9 = np.ones((3, 3), np.uint8), np.ones((9, 9), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x0, y0, bw, bh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        X0, Y0, X1, Y1 = max(0, x0 - 6), max(0, y0 - 6), min(w, x0 + bw + 6), min(h, y0 + bh + 6)
        m = (lab[Y0:Y1, X0:X1] == i).astype(np.uint8)
        mb = m.astype(bool)
        contour = cv2.dilate(m, k3).astype(bool) & ~mb
        on_edge = cv2.dilate((mb & border_img[Y0:Y1, X0:X1]).astype(np.uint8), k3).astype(bool)
        contour &= ~on_edge
        ring = cv2.dilate(m, k9).astype(bool) & ~mb
        anc = ring & valid[Y0:Y1, X0:X1]
        if stats is not None:
            stats["regions"] = stats.get("regions", 0) + 1
        if contour.sum() < 10 or anc.sum() < 30:
            continue
        covered = cv2.dilate(anc.astype(np.uint8), k9).astype(bool) & contour
        cover = covered.sum() / contour.sum()
        ys, xs = np.nonzero(anc)
        cxr, cyr = cen[i][0] - X0, cen[i][1] - Y0
        ang = np.arctan2(ys - cyr, xs - cxr)
        sides = len(np.unique(((ang + np.pi / 4) // (np.pi / 2)).astype(int) % 4))
        if cover < cover_min or sides < sides_min:
            continue
        pts = P[Y0:Y1, X0:X1][anc]
        if len(pts) > 400:
            pts = pts[rng.choice(len(pts), 400, replace=False)]
        best, bn = None, 0
        thr = 0.02 + 0.01 * pts[:, 0] ** 2
        for _ in range(50):
            s = pts[rng.choice(len(pts), 3, replace=False)]
            nrm = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(nrm)
            if nn < 1e-6:
                continue
            nrm /= nn; dd = -nrm @ s[0]
            inl = np.abs(pts @ nrm + dd) < thr
            if inl.sum() > bn:
                bn, best = inl.sum(), (nrm, dd, inl)
        if best is None or bn < inlier_min * len(pts):
            continue
        # refit on inliers (least squares through the centroid)
        q = pts[best[2]]
        c0 = q.mean(0)
        nrm = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        dd = -nrm @ c0
        ray_c = np.array([1.0, (cx - cen[i][0]) / f, (cy - cen[i][1]) / f]); ray_c /= np.linalg.norm(ray_c)
        if abs(nrm @ ray_c) < 0.25:
            continue
        sel_y, sel_x = np.nonzero(mb)
        rays = np.stack([np.ones(len(sel_x)), (cx - (sel_x + X0 + 0.5)) / f, (cy - (sel_y + Y0 + 0.5)) / f], -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = -dd / (rays @ nrm)
        zlo, zhi = q[:, 0].min(), q[:, 0].max()
        ok = np.isfinite(zz) & (zz >= 0.8 * zlo) & (zz <= zhi) & (zz > 0.2)
        out[sel_y[ok] + Y0, sel_x[ok] + X0] = zz[ok]
        if stats is not None:
            stats["filled"] = stats.get("filled", 0) + 1
            stats["px"] = stats.get("px", 0) + int(ok.sum())
    return out


class EdgePlaneMixin:
    """Edge-anchored plane fill on top of any method (outermost, so it sees the final depth)."""
    epf_zmax = 3.0
    epf_cover = 0.6

    def process(self, frame, calib):
        o = super().process(frame, calib)
        z = o["depth"]
        if z.shape != (calib["H"], calib["W"]):
            z = cv2.resize(z, (calib["W"], calib["H"]), interpolation=cv2.INTER_NEAREST)
        far = o.get("far")
        if far is not None and far.shape != z.shape:
            far = cv2.resize(far.astype(np.uint8), (calib["W"], calib["H"]), interpolation=cv2.INTER_NEAREST).astype(bool)
        out = dict(o)
        out["depth"] = edge_plane_fill(z, gray(frame["left"]), calib, far, zmax=getattr(self, "zmax", self.epf_zmax),
                                       cover_min=self.epf_cover)
        if far is not None:
            out["far"] = far
        return out


@register
class BM160EPF(EdgePlaneMixin, BM160):
    name = "bm160_lrc_epf"
    note = "bm160_lrc + edge-anchored plane fill (per frame)"


@register
class BM160TemporalR3EPF(EdgePlaneMixin, BM160TemporalR3):
    name = "bm160_temporal_r3_epf"
    note = "bm160_temporal_r3 + edge-anchored plane fill after the 3-frame vote"


@register
class SGBM320EPF(EdgePlaneMixin, SGBMBaseline):
    name = "sgbm320_epf"
    note = "sgbm320 baseline + edge-anchored plane fill"


@register
class SGBM320LrcTexEPF(EdgePlaneMixin, SGBMFiltered320):
    name = "sgbm320_lrc_tex_epf"
    note = "sgbm320_lrc_tex + edge-anchored plane fill"


@register
class BM160TemporalMapEPF(EdgePlaneMixin, BM160Map):
    name = "bm160_temporal_map_epf"
    note = "bm160_temporal + map raycast hole fill, then edge-anchored plane fill"


@register
class BM160TemporalR3EPFLoose(BM160TemporalR3EPF):
    name = "bm160_temporal_r3_epf30"
    epf_cover = 0.3
    note = "bm160_temporal_r3_epf with anchors on only 30% of the hole border (still >= 3 sides)"


@register
class BM160EPFLoose(BM160EPF):
    name = "bm160_lrc_epf30"
    epf_cover = 0.3
    note = "bm160_lrc_epf with anchors on only 30% of the hole border (still >= 3 sides)"

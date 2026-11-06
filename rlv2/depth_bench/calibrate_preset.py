"""Ray-level rates for the flight depth preset (rlv2/depth_preset.h), measured per surface class.

Rays as in the bench: 16x12 blocks of 20x20 px at 320x240; GT ray = 5th percentile of GT depth,
method ray = 5th percentile of its returns if >= 10 % of the block returned, else unknown. Only
rays right of the matcher's dead band (block columns >= 4) and with GT nearer than 3 m. Classes:
floor/ceiling (nearest GT surface), textureless (>= half the block textureless in the clean image),
textured (everything else; thin-object rays excluded). Rates:
  hole_*      P(unknown | class)
  phantom     P(return nearer than GT - tol | ray)          (preset: phantom, 0.3-2.3 m)
  false_free  P(return farther than GT + tol | returned, GT < 1.5 m)   (preset: false_free)
  k           sigma / z^2 of correct returns (robust: 1.4826 * median |e - g| / g^2)
Each also after `despeckle` (drop a ray with no 8-neighbour within max(0.3 m, 15 %)), as in
rlv2/depth_preset.h dp_despeckle. Real photos: the same on ETH3D indoor + Middlebury (from
texture_study.py), with GT disparity rescaled to the drone pair (f 246.4 px, 6 cm): every ray is
"obstacle"; there is no floor class.

  python calibrate_preset.py --data /workspace/dbench_data/tex_nav_0.3 --real /workspace/realstereo --out results/preset_calibration.json
"""
import argparse, json, os, warnings
import cv2
import numpy as np
import dataset, methods
from metrics import _blocks, surface_masks, tol

F, B = 246.4, 0.06


def despeckle(e):
    out = e.copy()
    H, W = e.shape
    for r in range(H):
        for c in range(W):
            d = e[r, c]
            if not np.isfinite(d):
                continue
            nb = e[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            sup = (np.abs(nb - d) < max(0.3, 0.15 * d)).sum() - 1
            if sup <= 0:
                out[r, c] = np.nan
    return out


class Acc:
    def __init__(self):
        self.n = {}
        self.k = []

    def add(self, key, num, den):
        a = self.n.setdefault(key, [0, 0]); a[0] += int(num); a[1] += int(den)

    def res(self):
        o = {k: (a[0] / a[1] if a[1] else None) for k, a in self.n.items()}
        o.update({f"n_{k}": a[1] for k, a in self.n.items()})
        o["k"] = float(np.median(self.k) * 1.4826) if self.k else None
        return o


def ray_stats(acc, est, z, texless_px, normal, thin):
    H, W = z.shape
    if est.shape != z.shape:   # 160x120 matchers: nearest-neighbour up to the GT grid
        est = cv2.resize(est, (W, H), interpolation=cv2.INTER_NEAREST)
    gv = np.isfinite(z) & (z > 0.2) & (z < 12)
    zb = _blocks(np.where(gv, z, np.nan), H, W)
    ev = np.isfinite(est) & (est > 0)
    eb = _blocks(np.where(ev, est, np.nan), H, W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = np.nanpercentile(zb, 5, axis=-1)
        frac = np.isfinite(eb).mean(-1)
        e = np.nanpercentile(eb, 5, axis=-1)
    e = np.where(frac >= 0.10, e, np.nan)
    tl = _blocks(texless_px.astype(np.float32), H, W).mean(-1) >= 0.5
    near = zb <= g[..., None] + 0.05
    nb = _blocks(normal.astype(np.float32), H, W)
    flc = (((nb == 1) | (nb == 3)) & near).sum(-1) > 0.5 * np.maximum(near.sum(-1), 1)
    th = _blocks(thin.astype(np.float32), H, W).sum(-1) >= 8
    base = np.isfinite(g) & (g < 3.0) & ~th
    base[:, :4] = False
    for tag, ee in (("", e), ("desp_", despeckle(e))):
        unk = ~np.isfinite(ee)
        gg = np.nan_to_num(g, nan=1.0)
        ret = base & ~unk
        for cname, cm in (("floor_ceiling", flc), ("textureless", tl & ~flc), ("textured", ~tl & ~flc)):
            m = base & cm
            acc.add(f"{tag}hole_{cname}", (m & unk).sum(), m.sum())
            ph = ret & (ee < gg - tol(gg))
            acc.add(f"{tag}phantom_{cname}", (ph & cm).sum(), m.sum())
            n15c = ret & (gg < 1.5) & cm
            acc.add(f"{tag}false_free_{cname}", (n15c & (ee > gg + tol(gg))).sum(), n15c.sum())
        acc.add(f"{tag}hole_all", (base & unk).sum(), base.sum())
        if not tag:   # phantom clumping: 8-connected blobs of phantom rays
            ph = (ret & (ee < gg - tol(gg))).astype(np.uint8)
            nlab, lab = cv2.connectedComponents(ph, connectivity=8)
            for i in range(1, nlab):
                acc.add("phantom_blob_rays", (lab == i).sum(), 1)
        acc.add(f"{tag}phantom", (ret & (ee < gg - tol(gg))).sum(), base.sum())
        acc.add(f"{tag}phantom_given_return", (ret & (ee < gg - tol(gg))).sum(), ret.sum())
        n15 = ret & (gg < 1.5)
        acc.add(f"{tag}false_free", (n15 & (ee > gg + tol(gg))).sum(), n15.sum())
        if not tag:
            ok = ret & (np.abs(ee - gg) <= tol(gg))
            acc.k.extend((np.abs(ee - gg)[ok] / gg[ok] ** 2).tolist())
    acc.add("textureless_share", (base & tl & ~flc).sum(), (base & ~flc).sum())


def sim(acc, data, method):
    M = methods.REGISTRY[method]()
    for seq in dataset.sequences("nav", "ov2640", M.baseline, data):
        M.reset(seq.calib)
        for k in range(seq.T):
            fr = seq.frame(k)
            est = M.process(dict(left=fr["left"], right=fr["right"], pose=fr["pose"], k=k), seq.calib)["depth"]
            est = np.where(np.isposinf(est), 50.0, est)
            ray_stats(acc, est.astype(np.float32), fr["depth"], surface_masks(fr)["textureless"], fr["normal"], fr["thin"])


def real(acc, root, method):
    import texture_study as T
    M = methods.REGISTRY[method]()
    for kind, name, d in T.real_pairs(root):
        if kind == "eth3d_outdoor":
            continue
        L = cv2.cvtColor(cv2.imread(os.path.join(d, "im0.png")), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(os.path.join(d, "im1.png")), cv2.COLOR_BGR2RGB)
        disp = T.read_pfm(os.path.join(d, "disp0GT.pfm"))
        m = os.path.join(d, "mask0nocc.png")
        if os.path.exists(m):
            disp[cv2.imread(m, 0) != 255] = np.inf
        h, w = disp.shape
        cw = min(w, h * 4 // 3); ch = cw * 3 // 4; x0 = (w - cw) // 2; y0 = (h - ch) // 2
        crop = lambda a: a[y0:y0 + ch, x0:x0 + cw]
        L2 = cv2.resize(crop(L), (320, 240), interpolation=cv2.INTER_AREA)
        R2 = cv2.resize(crop(R), (320, 240), interpolation=cv2.INTER_AREA)
        dg = crop(disp).copy(); dg[~np.isfinite(dg)] = np.nan
        d320 = cv2.resize(dg, (320, 240), interpolation=cv2.INTER_NEAREST) * 320.0 / cw
        z = np.where(d320 > 0, F * B / d320, np.nan).astype(np.float32)   # as if seen by the drone pair
        Lp, Rp = T.apply_profile(L2, R2, "ov2640", F, (kind, name, "tex", 0))
        calib = dict(f=F, cx=160, cy=120, baseline=M.baseline, W=320, H=240, dt=0.1)
        if M.baseline != B:   # 12 cm methods would need the scene at another scale; skip
            continue
        M.reset(calib)
        est = M.process(dict(left=Lp, right=Rp, pose=np.eye(4), k=0), calib)["depth"]
        est = np.where(np.isposinf(est), 50.0, est)
        g = cv2.cvtColor(L2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.boxFilter(g, -1, (7, 7)); sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (7, 7)) - mu * mu, 0))
        ray_stats(acc, est.astype(np.float32), z, sd < 3, np.full(z.shape, 2, np.uint8), np.zeros(z.shape, np.uint8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--real", default="")
    ap.add_argument("--methods", default="sgbm320,bm160_lrc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    res = {}
    for m in a.methods.split(","):
        acc = Acc(); sim(acc, a.data, m); res[f"sim:{os.path.basename(a.data.rstrip('/'))}:{m}"] = acc.res()
        if a.real:
            acc = Acc(); real(acc, a.real, m); res[f"real_indoor:{m}"] = acc.res()
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if not x.startswith("n_")})

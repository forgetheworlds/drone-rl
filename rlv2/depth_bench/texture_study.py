"""Texture realism check: are the HyperDrone renders smoother than real indoor imagery?

Same pipeline for both: 4:3 view at 320x240, the ov2640 camera profile (shot/read noise, right-camera
gain mismatch, rectification error, JPEG q50; sim frames also carry rendered rolling shutter + blur),
then 160x120 grey as the on-board StereoBM sees it. Per image:
  grad    mean Sobel gradient magnitude (grey levels / px) at 160x120
  smooth  fraction of pixels failing the bench texture gate (box-filtered |Sobel x| < 6, 7x7)
  hole    fraction of GT-valid pixels (columns >= 32) where bm160_lrc returns nothing
  ok      fraction of GT-valid pixels with a BM return within max(0.5 px, 15%) of GT disparity
plus P(hole | smooth) and P(hole | textured).
Real: ETH3D two-view training (indoor: delivery_area, electro, terrains; outdoor listed apart) and
Middlebury 2014 / MiddEval3 training at quarter resolution. Both are downloaded separately:
  https://www.eth3d.net/data/two_view_training{,_gt}.7z
  https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-{data,GT0}-Q.zip
Usage: python texture_study.py REAL_DIR [out.json]
"""
import glob, json, os, re, sys
import cv2
import numpy as np
from dataset import apply_profile, sequences
from methods import BM160, gray, texture_mask

bm = BM160()


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        d = np.frombuffer(f.read(), "<f4" if scale < 0 else ">f4").reshape(h, w)
    return np.flipud(d).astype(np.float32)


def bm_disp(L, R):
    """bm160_lrc disparity at 160x120 (0 = no return), L/R uint8 RGB 320x240."""
    calib = dict(f=1.0, W=320, baseline=1.0)
    z = bm.process(dict(left=L, right=R), calib)["depth"]  # = f*s*B / d with f*B = 1, s = .5
    return np.where(np.isfinite(z), 0.5 / np.where(np.isfinite(z), z, 1), 0)


def stats(L, R, gt_disp160, cls=None):
    g = cv2.resize(gray(L), (160, 120), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 8
    smooth = ~texture_mask(g, 6.0, 7)
    d = bm_disp(L, R)
    valid = np.isfinite(gt_disp160) & (gt_disp160 > 0)
    valid[:, :32] = False
    hole = valid & (d <= 0)
    ok = valid & (d > 0) & (np.abs(d - gt_disp160) <= np.maximum(0.5, 0.15 * gt_disp160))
    out = dict(grad=float(grad.mean()), smooth=float(smooth.mean()), n=int(valid.sum()),
               hole=int(hole.sum()), ok=int(ok.sum()),
               n_s=int((valid & smooth).sum()), hole_s=int((hole & smooth).sum()),
               n_t=int((valid & ~smooth).sum()), hole_t=int((hole & ~smooth).sum()))
    # ray level (the bench and the flight preset): 16x12 blocks of 10x10 px at 160x120, a ray is
    # UNKNOWN when < 10 % of its block has a BM return; textureless ray = >= half the block smooth.
    # Only blocks right of the BM dead band (block columns >= 4) and >= half GT-valid count.
    blk = lambda a: a.reshape(12, 10, 16, 10).swapaxes(1, 2).reshape(12, 16, 100)
    gvb = blk(np.isfinite(gt_disp160) & (gt_disp160 > 0)).mean(-1) >= 0.5
    gvb[:, :4] = False
    unk = blk(d > 0).mean(-1) < 0.10
    tl = blk(smooth).mean(-1) >= 0.5
    out.update(rays=int(gvb.sum()), rays_unk=int((gvb & unk).sum()), rays_tl=int((gvb & tl).sum()),
               rays_tl_unk=int((gvb & tl & unk).sum()))
    if cls is not None:
        wb = (blk(cls == 2).mean(-1) >= 0.5) & gvb
        out.update(rays_wall=int(wb.sum()), rays_wall_unk=int((wb & unk).sum()), rays_wall_tl=int((wb & tl).sum()),
                   rays_wall_tl_unk=int((wb & tl & unk).sum()))
    if cls is not None:
        for name, c in (("wall", 2), ("floor", 1)):
            m = valid & (cls == c)
            out[f"n_{name}"] = int(m.sum()); out[f"hole_{name}"] = int((hole & m).sum())
            out[f"smooth_{name}"] = int((smooth & m).sum())
            out[f"npx_{name}"] = int((cls == c).sum()); out[f"spx_{name}"] = int((smooth & (cls == c)).sum())
    return out


def real_pairs(root):
    for d in sorted(glob.glob(os.path.join(root, "eth3d", "*"))):
        name = os.path.basename(d)
        kind = "eth3d_indoor" if re.match(r"(delivery_area|electro|terrains)", name) else "eth3d_outdoor"
        yield kind, name, d
    for d in sorted(glob.glob(os.path.join(root, "MiddEval3", "trainingQ", "*"))):
        yield "middlebury", os.path.basename(d), d


def real(root):
    rows = []
    for kind, name, d in real_pairs(root):
        L = cv2.cvtColor(cv2.imread(os.path.join(d, "im0.png")), cv2.COLOR_BGR2RGB)
        R = cv2.cvtColor(cv2.imread(os.path.join(d, "im1.png")), cv2.COLOR_BGR2RGB)
        disp = read_pfm(os.path.join(d, "disp0GT.pfm"))
        m = os.path.join(d, "mask0nocc.png")
        if os.path.exists(m):
            disp[cv2.imread(m, 0) != 255] = np.inf
        h, w = disp.shape
        cw = min(w, h * 4 // 3); ch = cw * 3 // 4; x0 = (w - cw) // 2; y0 = (h - ch) // 2
        crop = lambda a: a[y0:y0 + ch, x0:x0 + cw]
        L2 = cv2.resize(crop(L), (320, 240), interpolation=cv2.INTER_AREA)
        R2 = cv2.resize(crop(R), (320, 240), interpolation=cv2.INTER_AREA)
        dg = crop(disp).copy(); dg[~np.isfinite(dg)] = np.nan
        dg160 = cv2.resize(dg, (160, 120), interpolation=cv2.INTER_NEAREST) * 160.0 / cw
        Lp, Rp = apply_profile(L2, R2, "ov2640", 246.4, (kind, name, "tex", 0))
        rows.append(dict(src=kind, name=name, **stats(Lp, Rp, dg160)))
    return rows


def sim(split, data_dir=None):
    rows = []
    for seq in sequences(split, "ov2640", 0.06, data_dir or os.environ.get("DEPTH_BENCH_DATA", "/workspace/dbench_data/v1")):
        f160 = seq.calib["f"] / 2
        kind = "sim_procthor" if "ProcTHOR" in seq.scene else "sim_voxel"
        for k in range(seq.T):
            fr = seq.frame(k)
            z = cv2.resize(fr["depth"], (160, 120), interpolation=cv2.INTER_NEAREST)
            cls = cv2.resize(fr["normal"], (160, 120), interpolation=cv2.INTER_NEAREST)
            with np.errstate(divide="ignore"):
                dg = np.where(np.isfinite(z) & (z > 0), f160 * 0.06 / z, np.nan).astype(np.float32)
            rows.append(dict(src=kind, name=f"{split}:{seq.scene}:{seq.idx}:{k}", **stats(fr["left"], fr["right"], dg, cls)))
    return rows


def summary(rows):
    out = {}
    for src in sorted({r["src"] for r in rows}):
        rr = [r for r in rows if r["src"] == src]
        s = lambda k: sum(r[k] for r in rr)
        o = dict(images=len(rr), grad=float(np.mean([r["grad"] for r in rr])),
                 grad_median=float(np.median([r["grad"] for r in rr])),
                 smooth=float(np.mean([r["smooth"] for r in rr])),
                 hole=s("hole") / max(1, s("n")), ok=s("ok") / max(1, s("n")),
                 hole_given_smooth=s("hole_s") / max(1, s("n_s")), hole_given_textured=s("hole_t") / max(1, s("n_t")))
        o.update(ray_unknown=s("rays_unk") / max(1, s("rays")), ray_textureless=s("rays_tl") / max(1, s("rays")),
                 ray_unknown_given_textureless=s("rays_tl_unk") / max(1, s("rays_tl")),
                 ray_unknown_given_textured=(s("rays_unk") - s("rays_tl_unk")) / max(1, s("rays") - s("rays_tl")))
        if "n_wall" in rr[0]:
            o.update(wall_ray_unknown=s("rays_wall_unk") / max(1, s("rays_wall")),
                     wall_ray_textureless=s("rays_wall_tl") / max(1, s("rays_wall")),
                     wall_ray_unknown_given_textureless=s("rays_wall_tl_unk") / max(1, s("rays_wall_tl")),
                     wall_ray_unknown_given_textured=(s("rays_wall_unk") - s("rays_wall_tl_unk")) / max(1, s("rays_wall") - s("rays_wall_tl")))
            for c in ("wall", "floor"):
                o[f"hole_{c}"] = s(f"hole_{c}") / max(1, s(f"n_{c}"))
                o[f"smooth_{c}"] = s(f"spx_{c}") / max(1, s(f"npx_{c}"))
        out[src] = o
    return out


if __name__ == "__main__":
    rows = real(sys.argv[1])
    for split in ("dev", "test", "nav"):
        rows += sim(split)
    res = dict(summary=summary(rows), rows=rows)
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "texture_study.json", "w"), indent=1)
    for k, v in res["summary"].items():
        print(k, json.dumps({a: round(b, 3) for a, b in v.items()}))

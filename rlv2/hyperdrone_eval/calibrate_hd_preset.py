"""Fit the hd_* depth presets (rlv2/depth_preset.h) to real StereoBM rays rendered in HyperDrone.

Reads the cal_<W>_<lens>_<thread>.npz files written by ext_depth_server.py --calib (and the main
camera's rays<t>.npz copied as cal_<W>_<lens>_main.npz) and prints, per camera model, the per-class
ray rates in the preset's terms, before despeckle (the preset adds its own despeckle); <model>_fx = with camera effects:
  hole_<class>    no return / rays with GT < 3 m (outside the StereoBM dead band)
  phantom_<class> return nearer than GT - tol / rays with GT < 3 m
  ff_<class>      return farther than GT + tol / returns with GT < 1.5 m
  k               1.4826 median |e - g| / g^2 over correct returns;  tol = max(0.10, 0.15 GT)
  textureless     share of wall/object rays that are textureless (preset: textureless tile share)
  dead_cols/rows  ray columns (left) / rows (top+bottom) that never return (dead band, outside render)
plus hole/phantom by GT distance band for reference.

  python calibrate_hd_preset.py CAL_DIR [--json out.json]
"""
import argparse, glob, json, os, re
import numpy as np

p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--json", default="")
a = p.parse_args()
groups = {}
for f in sorted(glob.glob(os.path.join(a.dir, "cal_*.npz"))):
    m = re.match(r"cal_(\d+)_(\w+?)_(\w+)\.npz", os.path.basename(f))
    groups.setdefault(f"{m.group(2)}_{m.group(1)}", []).append(f)
res = {}
for name, fs in groups.items():
    d = [np.load(f) for f in fs]
    for tag in ("", "_fx"):
        if tag and "fx" not in d[0]:
            continue
        est = np.concatenate([x["fx" if tag else "est"] for x in d]); gt = np.concatenate([x["gt"] for x in d])
        cls = np.concatenate([x["cls"] for x in d]); inside = d[0]["inside"]
        e = np.where(est >= 12, np.nan, est)
        tol = np.maximum(.10, .15 * np.nan_to_num(gt, nan=1))
        base = np.isfinite(gt) & (gt < 3) & inside[None]
        unk = ~np.isfinite(e); ret = base & ~unk
        ph = ret & (e < gt - tol); ok = ret & (np.abs(e - gt) <= tol)
        n15 = ret & (gt < 1.5); ff = n15 & (e > gt + tol)
        grid = inside.reshape(16, 20)
        r = dict(frames=int(len(est)), rays=int(base.sum()),
                 dead_cols=int(np.argmax(grid.any(0))), dead_rows=int(np.argmax(grid.any(1))),
                 k=round(float(np.median(np.abs(e - gt)[ok] / gt[ok] ** 2) * 1.4826), 4),
                 textureless=round(float((base & (cls == 2)).sum() / max(1, (base & (cls >= 2)).sum())), 3))
        for cn, c in (("floor", 1), ("textureless", 2), ("textured", 3)):
            b = base & (cls == c)
            r["share_" + cn] = round(float(b.sum() / base.sum()), 3)
            r["hole_" + cn] = round(float((b & unk).sum() / max(1, b.sum())), 3)
            r["phantom_" + cn] = round(float((b & ph).sum() / max(1, b.sum())), 4)
            r["ff_" + cn] = round(float((ff & (cls == c)).sum() / max(1, (n15 & (cls == c)).sum())), 4)
        for lo, hi in ((0, 1), (1, 2), (2, 3)):
            b = base & (gt >= lo) & (gt < hi)
            r[f"band_{lo}_{hi}"] = dict(n=int(b.sum()), hole=round(float((b & unk).sum() / max(1, b.sum())), 3),
                                        phantom=round(float((b & ph).sum() / max(1, b.sum())), 4))
        r["all"] = dict(hole=round(float((base & unk).sum() / base.sum()), 3), phantom=round(float(ph.sum() / base.sum()), 4),
                        ff=round(float(ff.sum() / max(1, n15.sum())), 4))
        res[name + tag] = r
print(json.dumps(res, indent=1))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

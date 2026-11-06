"""RGB stereo + OpenCV SGBM at OV2640-like resolution, vs our synthetic depth-noise model.

Renders a rectified pair (320 x 240, 66 deg HFOV, 6 cm baseline, f ~ 246 px, so f*B = 14.8 px*m,
our VX_VO_FOCAL_BASELINE) plus ground-truth planar depth with HyperLux, runs StereoSGBM and measures
what our sensors.h model parameterises:
  sigma/z^2 (depth_noise), hole rate (no disparity where truth < 12 m), near-false rate
  (estimate < 0.7 x truth), and the range where valid density falls below 50 %.

  python stereo_sgbm.py [--scene procthor|voxel:<task>] [--fidelity high] [--pairs 64] [--save DIR]
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

import hyperlux
from hyperlux.camera import CameraSpec, Depth, RGB

import voxmap

p = argparse.ArgumentParser()
p.add_argument("--scene", default="procthor")
p.add_argument("--fidelity", default="high")
p.add_argument("--pairs", type=int, default=64)
p.add_argument("--save", default="")
a = p.parse_args()
W, H, HFOV, B = 320, 240, 66.0, 0.06
F = (W / 2) / math.tan(math.radians(HFOV / 2))

if a.scene == "procthor":
    from hyperlux.examples.data import procthor_scene_path
    scene = hyperlux.load_scene(procthor_scene_path(), fidelity=a.fidelity)
else:
    task = voxmap.load_task(a.scene[6:])
    scene = voxmap.hyperlux_scene(voxmap.world_quads(task, movers=False))
rgb_d, dep_d = RGB("rgba8"), Depth()
r = hyperlux.HyperLux(width=W, height=H, batch_size=2 * a.pairs, num_probes=0,
                      cameras=(CameraSpec("front", (rgb_d, dep_d)),), fidelity=a.fidelity)
r.init(scene)
pos = hyperlux.ClearanceSampler(scene, clearance=0.4).sample(a.pairs, seed=1)
rng = np.random.default_rng(1)
yaw = rng.uniform(-math.pi, math.pi, a.pairs)
cams = np.zeros((2 * a.pairs, 12), np.float32)
tx = math.tan(math.radians(HFOV / 2)); ty = tx * H / W
for i, y in enumerate(yaw):
    f = np.array([math.cos(y), math.sin(y), 0]); left = np.array([-math.sin(y), math.cos(y), 0]); up = np.array([0, 0, 1.])
    for k, off in enumerate((0.0, -B)):  # left camera, right camera 6 cm to the right (-left)
        o = pos[i] + off * left
        cams[2 * i + k] = np.concatenate([o, f + tx * left + ty * up, -2 * tx * left, -2 * ty * up])
r.set_cameras(cams)
r.render(); r.synchronize()
rgb = np.asarray(r.outputs[rgb_d].read()).reshape(2 * a.pairs, H, W, -1)[..., :3]
dep = np.asarray(r.outputs[dep_d].read(), np.float32).reshape(2 * a.pairs, H, W)

nd = 64
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 3 * 25, P2=32 * 3 * 25,
                             disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
z_all, e_all, valid_all = [], [], []
for i in range(a.pairs):
    L = cv2.cvtColor(rgb[2 * i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    R = cv2.cvtColor(rgb[2 * i + 1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    d = sgbm.compute(L, R).astype(np.float32) / 16.0
    z = dep[2 * i]
    roi = np.zeros_like(z, bool); roi[:, nd:] = True  # SGBM has no data in the first numDisparities columns
    roi &= np.isfinite(z) & (z > F * B / (nd - 1)) & (z < 12)
    v = d > 0.5
    est = np.where(v, F * B / np.maximum(d, 1e-3), np.nan)
    z_all.append(z[roi]); e_all.append(est[roi]); valid_all.append(v[roi])
    if a.save and i < 4:
        os.makedirs(a.save, exist_ok=True)
        cv2.imwrite(os.path.join(a.save, f"{i}_left.png"), cv2.cvtColor(rgb[2 * i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(a.save, f"{i}_disp.png"), np.clip(d * 4, 0, 255).astype(np.uint8))
z = np.concatenate(z_all); e = np.concatenate(e_all); v = np.concatenate(valid_all)
bins = [0.3, 1, 2, 3, 4, 6, 8, 12]
per = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (z >= lo) & (z < hi)
    if s.sum() < 100:
        continue
    sv = s & v
    err = (e - z)[sv]
    good = np.abs(err) < 0.3 * z[sv]   # inliers for the Gaussian part
    sig = float(1.4826 * np.median(np.abs(err[good] - np.median(err[good])))) if good.sum() > 50 else float("nan")
    zm = float(np.median(z[sv])) if sv.sum() else float("nan")
    per.append(dict(z=f"{lo}-{hi}", px=int(s.sum()), valid=round(float(v[s].mean()), 3),
                    sigma_m=round(sig, 4), sigma_over_z2=round(sig / zm ** 2, 4) if sv.sum() else None,
                    near_false=round(float((e[sv] < 0.7 * z[sv]).mean()), 4),
                    far_false=round(float((e[sv] > 1.43 * z[sv]).mean()), 4)))
res = dict(scene=a.scene.split("/")[-1], fidelity=a.fidelity, pairs=a.pairs, f_px=round(F, 1), fB=round(F * B, 2),
           hole_rate=round(float(1 - v.mean()), 3),
           near_false_rate=round(float((e[v] < 0.7 * z[v]).mean()), 4), bins=per,
           ours_level1=dict(sigma_over_z2="0.01-0.03", hole_rate="0-0.08 (4x4 patches)", near_false="0-0.004",
                            max_range_m="4-6"))
print(json.dumps(res), flush=True)

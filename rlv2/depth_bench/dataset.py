"""Dataset loading + camera-realism profiles (deterministic, seeded per scene/seq/frame).

A profile turns the rendered images into what a cheap OV2640 pair would deliver:
  clean       global-shutter render, perfectly synced and calibrated (upper bound)
  ov2640      rolling shutter + 10 ms blur (rendered), shot/read noise, per-camera auto-exposure
              mismatch, 0.4 px residual rectification error, JPEG q50   <- the default
  ov2640_dark same, but 40 ms exposure (more blur), 5x less light with 5x analog gain
  ov2640_unsync  ov2640, right camera 12 ms late (free-running pair, no frame sync)
  ov2640_dots ov2640 + a cheap IR random-dot projector (850 nm DOE laser, ~5 k dots, co-located with
              the left camera) seen by OV2640s *without* IR-cut filter. Dots are rendered analytically
              from GT depth (left: fixed pattern; right: shifted by the true disparity), brightness
              ~1/z^2. Reference-only "active texture" upper bound; ignores occlusion of dots.
Baseline 6 cm unless a method asks for 12 cm (`baseline=0.12`; only clean/ov2640 exist at 12 cm).
Methods receive the *nominal* calibration; the rectification error is unknown to them.
VO pose noise: methods get poses with a random walk of 1 cm and 0.3 deg per frame (pose_gt is
kept for scoring), mimicking stereo VO drift over a 0.7 s sequence.
"""
import glob
import json
import math
import os
import zlib

import cv2
import numpy as np

DEFAULT_DATA = os.environ.get("DEPTH_BENCH_DATA", os.path.expanduser("/workspace/dbench_data/v1"))
PROFILES = ("clean", "ov2640", "ov2640_dark", "ov2640_unsync", "ov2640_dots")


def seeded(*keys):
    return np.random.default_rng(zlib.crc32("|".join(map(str, keys)).encode()))


def dot_layers(depth, f_px, baseline, n_dots=5000, amp=0.6, seed=7):
    """Linear-light dot images (H,W) for the left and right camera. amp = linear brightness at 1 m."""
    h, w = depth.shape
    rng = np.random.default_rng(seed)                       # the DOE pattern is fixed hardware
    u = rng.uniform(0, w, n_dots); v = rng.uniform(0, h, n_dots)
    z = depth[np.clip(v.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
    ok = np.isfinite(z) & (z > 0.05)
    u, v, z = u[ok], v[ok], z[ok]
    a = np.minimum(amp / z ** 2, 1.5)
    out = []
    for uu in (u, u - f_px * baseline / z):
        img = np.zeros((h, w), np.float32)
        x0 = np.floor(uu - 0.5).astype(int); y0 = np.floor(v - 0.5).astype(int)
        fx = uu - 0.5 - x0; fy = v - 0.5 - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            np.add.at(img, (yi[m], xi[m]), (a * wt)[m])
        out.append(cv2.GaussianBlur(img, (0, 0), 0.6) * 2 * math.pi * 0.36)
    return out


def _sensor(img, rng, light, gain, full_well=900.0, read_e=4.0, extra=None):
    lin = (img.astype(np.float32) / 255.0) ** 2.2 * light
    if extra is not None:
        lin = lin + extra[..., None] * light
    e = rng.poisson(np.clip(lin * full_well, 0, None)).astype(np.float32) + rng.normal(0, read_e, lin.shape)
    out = np.clip(e / full_well * gain, 0, 1) ** (1 / 2.2)
    return out * 255.0


def _jpeg(img, q=50):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_profile(L, R, profile, f_px, key, depth=None, baseline=0.06):
    if profile == "clean":
        return L, R
    rng = seeded(*key)
    seq_rng = seeded(*key[:-1])                          # per-sequence constants
    light, gain = (0.2, 5.0) if profile == "ov2640_dark" else (1.0, 1.0)
    gR = seq_rng.uniform(0.85, 1.15)                     # auto-exposure mismatch between the two cameras
    dots = dot_layers(depth, f_px, baseline) if profile == "ov2640_dots" else (None, None)
    out = []
    for img, g, ex in ((L, 1.0, dots[0]), (R, gR, dots[1])):
        x = _sensor(img, rng, light * g, gain, extra=ex)
        out.append(np.clip(x, 0, 255).astype(np.uint8))
    # residual rectification error on the right camera: ~0.4 px vertical + 0.1 deg roll
    dy = seq_rng.choice([-1, 1]) * 0.4
    roll = math.radians(seq_rng.uniform(-0.1, 0.1))
    h, w = R.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(roll), 1.0); M[1, 2] += dy
    out[1] = cv2.warpAffine(out[1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _jpeg(out[0]), _jpeg(out[1])


def _rand_rot(rng, sig):
    v = rng.normal(0, sig, 3)
    R, _ = cv2.Rodrigues(v)
    return R


class Sequence:
    def __init__(self, path, idx, data, meta, profile, baseline):
        self.path, self.idx, self.meta, self.profile, self.baseline = path, idx, meta, profile, baseline
        self.scene = os.path.basename(path)[:-4]
        self.d = data
        self.T = data["depth"].shape[1]
        self.calib = dict(f=meta["f_px"], cx=meta["W"] / 2, cy=meta["H"] / 2, baseline=baseline,
                          W=meta["W"], H=meta["H"], dt=meta["dt"])
        rng = seeded(self.scene, idx, "vo")
        self.pose_gt = data["pose"][idx]
        noisy = [self.pose_gt[0].copy()]
        drift = np.eye(4)
        for k in range(1, self.T):
            step = np.linalg.inv(self.pose_gt[k - 1]) @ self.pose_gt[k]
            n = np.eye(4); n[:3, :3] = _rand_rot(rng, math.radians(0.3)); n[:3, 3] = rng.normal(0, 0.01, 3)
            drift = drift @ n
            noisy.append(noisy[-1] @ step @ n)
        self.pose_vo = np.stack(noisy)

    def frame(self, k):
        d, i = self.d, self.idx
        b = "R06" if self.baseline < 0.09 else "R12"
        if self.profile == "clean":
            L, R = d["L_clean"][i, k], d[f"{b}_clean"][i, k]
        elif self.profile == "ov2640_dark":
            L, R = d["L_rsdark"][i, k], d["R06_rsdark"][i, k]
        elif self.profile == "ov2640_unsync":
            L, R = d["L_rs"][i, k], d["R06_rsunsync"][i, k]
        else:
            L, R = d["L_rs"][i, k], d[f"{b}_rs"][i, k]
        L, R = apply_profile(L, R, self.profile, self.calib["f"], (self.scene, self.idx, self.profile, b, k),
                             depth=d["depth"][i, k], baseline=self.baseline)
        return dict(left=L, right=R, left_clean=d["L_clean"][i, k], depth=d["depth"][i, k],
                    normal=d["normal"][i, k], thin=d["thin"][i, k], pose=self.pose_vo[k], pose_gt=self.pose_gt[k])


def sequences(split="test", profile="ov2640", baseline=0.06, data_dir=DEFAULT_DATA, max_seqs=None):
    for path in sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npz"))):
        z = np.load(path)
        data = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
        n = data["depth"].shape[0] if max_seqs is None else min(max_seqs, data["depth"].shape[0])
        for i in range(n):
            yield Sequence(path, i, data, meta, profile, baseline)

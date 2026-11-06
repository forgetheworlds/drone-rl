"""OV2640-like camera effects for rendered stereo pairs (applied before StereoBM).

Parameters are drawn once per flight (FlightFx(seed)) within the ranges below; per-frame effects use the
drone's velocity and body rates. Sources (see docs/HYPERDRONE.md, section 6):
  [DS]  OmniVision OV2640 datasheet / product brief: 1/4" UXGA, 2.2 um pixels, electronic rolling
        shutter, SNR 40 dB, dynamic range 50 dB, 15 fps UXGA / 30 fps SVGA / 60 fps CIF, on-chip
        AEC/AGC and JPEG compression, lens correction.
  [ESP] espressif/esp32-camera driver: jpeg_quality 0-63 (lower = better), examples use 10-12;
        each sensor runs its own AEC/AGC (two modules are not exposure- or frame-synchronized).
Assumptions where no datasheet number applies (marked "assumed"): read noise, gain range in dim
rooms, vignetting of cheap M12 / wide modules, rectification residual, sync offset.
"""
import math, os
import numpy as np
import cv2

# Profiles (CE_PROFILE): "best" (default) = a calibrated, rectified, hardware-synced pair with matched
# exposure in good indoor light: what we train and grade on. "harsh" = unsynced modules, rough
# rectification, independent AEC, dim rooms: a sensitivity row only.
PROFILES = dict(
    best=dict(LIGHT=(0.6, 1.0), READ_NOISE_DN=(0.5, 1.0), JPEG_Q=(85, 95), AE_MISMATCH=0.01,
              VIGNETTE=(0.10, 0.20), VIG_MATCHED=True, RECT_DY=(0.0, 0.2), K1_PX=0.15, SYNC=(0.0, 0.002)),
    harsh=dict(LIGHT=(0.12, 1.0), READ_NOISE_DN=(0.8, 1.6), JPEG_Q=(55, 80), AE_MISMATCH=0.10,
               VIGNETTE=(0.15, 0.40), VIG_MATCHED=False, RECT_DY=(0.3, 0.5), K1_PX=0.5, SYNC=(0.0, 0.020)),
    # Round 8 (sim-to-real, 65 deg lenses, 160x120 StereoBM on an ESP32-P4-class host). No JPEG on either
    # path (raw/YUV into the host). Assumed ranges, not measured on hardware:
    # ov2640_dvp: two OV2640 on DVP with a shared XCLK, started together but no frame-sync pin -> 0-8 ms
    #   capture offset; independent AEC/AGC (mismatch 3 %); 20-33 ms rolling readout; cheap 65 deg M12-ish lenses.
    # ov5647: OV5647 (Pi camera v1 class, 5 MP binned to 640x480 then 160x120) over MIPI CSI, 65 deg;
    #   rolling shutter ~ 30 ms readout at the 30 fps binned modes; FSIN not broken out on the cheap
    #   modules, so the pair is unsynced (0-16 ms, half a frame) unless both share one host trigger.
    ov2640_dvp=dict(LIGHT=(0.4, 1.0), READ_NOISE_DN=(0.6, 1.2), JPEG_Q=None, AE_MISMATCH=0.03,
                    VIGNETTE=(0.10, 0.25), VIG_MATCHED=False, RECT_DY=(0.1, 0.3), K1_PX=0.3, SYNC=(0.0, 0.008),
                    READOUT=(0.020, 0.033)),
    ov5647=dict(LIGHT=(0.4, 1.0), READ_NOISE_DN=(0.5, 1.0), JPEG_Q=None, AE_MISMATCH=0.03,
                VIGNETTE=(0.15, 0.30), VIG_MATCHED=False, RECT_DY=(0.1, 0.3), K1_PX=0.3, SYNC=(0.0, 0.016),
                READOUT=(0.028, 0.034)),
    # harsher robustness row for either: dim rooms, worse rectification, a full frame of sync offset.
    harsh65=dict(LIGHT=(0.12, 1.0), READ_NOISE_DN=(0.8, 1.6), JPEG_Q=None, AE_MISMATCH=0.10,
                 VIGNETTE=(0.15, 0.40), VIG_MATCHED=False, RECT_DY=(0.3, 0.6), K1_PX=0.6, SYNC=(0.0, 0.033),
                 READOUT=(0.028, 0.040)),
)
FRAME_READOUT = (0.020, 0.033)   # s, rolling-shutter top-to-bottom readout (30 fps SVGA-class timing [DS])
EXP_BRIGHT = (0.004, 0.010)      # s, exposure a well-lit room needs at unity gain (assumed)
EXP_MAX = 0.033                  # s, AEC cap at 30 fps; beyond that AGC raises gain [DS]
GAIN_MAX = 16.0                  # AGC ceiling used (OV2640 AGC goes to 32x) (assumed)
FULL_WELL_SNR_DB = 40.0          # [DS] -> shot noise sigma = 1 % of full scale at unity gain
# Per profile (assumed ranges): LIGHT room illuminance vs a bright office (log-uniform); READ_NOISE_DN at
# unity gain; JPEG_Q libjpeg quality (esp32-camera jpeg_quality ~4-6 best, 8-14 harsh); AE_MISMATCH
# +- L/R brightness; VIGNETTE corner light loss (matched L/R or not); RECT_DY vertical rectification
# residual px; K1_PX max radial residual at the corners px; SYNC right-camera capture offset s.
import os
# Ablation: CE_ONLY=a,b keeps only these effects (noise jpeg blur rs ae vig rect sync).
_ONLY = set(filter(None, os.environ.get("CE_ONLY", "").split(",")))
on = lambda name: not _ONLY or name in _ONLY


class FlightFx:
    def __init__(self, seed, profile=None):
        P = PROFILES[profile or os.environ.get("CE_PROFILE", "best")]
        LIGHT, READ_NOISE_DN, JPEG_Q, AE_MISMATCH = P["LIGHT"], P["READ_NOISE_DN"], P["JPEG_Q"], P["AE_MISMATCH"]
        VIGNETTE, RECT_DY, SYNC = P["VIGNETTE"], P["RECT_DY"], P["SYNC"]
        r = np.random.default_rng(seed)
        u = lambda lo_hi: float(r.uniform(*lo_hi))
        self.readout, self.exp_bright = u(P.get("READOUT", FRAME_READOUT)), u(EXP_BRIGHT)
        self.light = math.exp(r.uniform(math.log(LIGHT[0]), math.log(LIGHT[1])))
        self.read_dn = u(READ_NOISE_DN)
        self.jpeg_q = int(r.integers(JPEG_Q[0], JPEG_Q[1] + 1)) if JPEG_Q else 0   # 0: raw, no JPEG
        self.ae = 1 + r.uniform(-AE_MISMATCH, AE_MISMATCH, 2)        # L, R
        self.vig = r.uniform(*VIGNETTE, 2)
        if P["VIG_MATCHED"]:
            self.vig[1] = self.vig[0]
        self.dy = u(RECT_DY) * (1 if r.random() < .5 else -1)
        self.k1 = float(r.uniform(-1, 1)) * P["K1_PX"]   # px at the image corner
        self.sync = u(SYNC)
        self.rng = np.random.default_rng(seed + 7)
        if not on("ae"): self.ae[:] = 1
        if not on("vig"): self.vig[:] = 0
        if not on("rect"): self.dy = self.k1 = 0.0
        if not on("sync"): self.sync = 0.0
        if not on("rs"): self.readout = 0.0

    def summary(self):
        return dict(readout=self.readout, light=self.light, jpeg_q=self.jpeg_q, sync=self.sync, dy=self.dy, k1=self.k1)


def pixel_flow(z, f, W, H, v_cam, w_cam):
    """Image motion (px/s) of every pixel for camera velocity v_cam and rate w_cam (camera frame:
    x forward, y left, z up), planar depth z (forward)."""
    X = np.where(np.isfinite(z) & (z > 0), z, 50.0).astype(np.float32)
    uu, vv = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    Y = (W / 2 - uu) * X / f; Z = (H / 2 - vv) * X / f
    # dP/dt = -v - w x P
    dX = -v_cam[0] - (w_cam[1] * Z - w_cam[2] * Y)
    dY = -v_cam[1] - (w_cam[2] * X - w_cam[0] * Z)
    dZ = -v_cam[2] - (w_cam[0] * Y - w_cam[1] * X)
    du = -f * (dY * X - Y * dX) / X ** 2
    dv = -f * (dZ * X - Z * dX) / X ** 2
    return du.astype(np.float32), dv.astype(np.float32)


def apply(img, z, f, fx, cam, v_cam, w_cam, mean_clean):
    """img: HxWx3 uint8 clean render; cam: 0 left / 1 right. Returns the degraded uint8 image."""
    H, W = img.shape[:2]
    rng = fx.rng
    # auto exposure: exposure*gain needed for the AE target, exposure capped, rest is gain
    need = fx.exp_bright * 110.0 / max(mean_clean, 1.0) / fx.light
    t_exp = min(need, EXP_MAX); gain = min(GAIN_MAX, max(1.0, need / EXP_MAX))
    t_blur = t_exp if on("blur") else 0.0
    bright = (t_exp * gain) / need * 110.0 / max(mean_clean, 1.0) * fx.ae[cam]   # image scale after AE
    # rolling shutter + motion blur: sample along the flow at row time + exposure offsets
    du, dv = pixel_flow(z, f, W, H, v_cam, w_cam)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    t_row = ((vv + .5) / H - .5) * fx.readout
    n = 5
    acc = np.zeros((H, W, 3), np.float32)
    src = img.astype(np.float32)
    for s in np.linspace(-.5, .5, n):
        t = t_row + s * t_blur
        mx, my = uu + du * t, vv + dv * t
        if cam == 1:   # rectification residual: vertical offset and residual radial distortion
            xn, yn = (mx - W / 2) / (W / 2), (my - H / 2) / (W / 2)
            r2 = xn * xn + yn * yn
            kk = fx.k1 / (1.5625 * W / 2)   # corner (r^2 = 1.5625) shift = k1 px at the working resolution
            mx = W / 2 + xn * (1 + kk * r2) * (W / 2)
            my = H / 2 + yn * (1 + kk * r2) * (W / 2) + fx.dy
        acc += cv2.remap(src, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = acc / n
    # vignetting (cos^4-like radial falloff to 1 - vig at the corners)
    rr = ((uu - W / 2) ** 2 + (vv - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    x *= (1 - fx.vig[cam] * rr)[..., None]
    x *= bright
    # sensor noise in DN: shot (1 % of full scale at full well, scales with gain) + read noise
    k_shot = (255 * 10 ** (-FULL_WELL_SNR_DB / 20)) ** 2 / 255
    sig = np.sqrt(np.maximum(x, 0) * k_shot * gain + (fx.read_dn * gain) ** 2)
    if on("noise"):
        x = x + rng.standard_normal(x.shape).astype(np.float32) * sig
    x = np.clip(np.round(x), 0, 255).astype(np.uint8)
    if on("jpeg") and fx.jpeg_q > 0:
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, fx.jpeg_q])
        x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return x, dict(t_exp=t_exp, gain=gain)


def advance(pos, R, v_world, w_body, dt):
    """Pose dt later (constant velocity / body rate)."""
    th = np.asarray(w_body) * dt
    a = np.linalg.norm(th)
    if a < 1e-9:
        dR = np.eye(3)
    else:
        k = th / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        dR = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return pos + np.asarray(v_world) * dt, R @ dR

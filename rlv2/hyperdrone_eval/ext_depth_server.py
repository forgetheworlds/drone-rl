"""Closed-loop stereo depth for grade.cpp (RLV2_EXT_DEPTH): HyperLux renders, real StereoBM.

For every stereo frame the grader sends (pose in map metres, orientation wxyz, mission index),
render a rectified RGB pair (6 cm baseline, pinhole, 160 x 120 by default) at the drone pose,
run the bm160_lrc matcher from depth_bench/methods.py (StereoBM, 32 disparities, block 9, L/R
check; no ground truth goes in; with --fx 1, after the OV2640-like camera effects of
camera_effects.py), and turn it into our 320 rays exactly as the depth bench does:
per ray, the pixel window around its projection; estimate = 5th percentile of the returns if
>= 10 % of the window returned, else 12 (no return). The grader applies despeckle
(RLV2_PRESET_DESPECKLE=1). Ground-truth depth is rendered too, only for the ray-level metrics
written at exit (hole, phantom, false-free, as in depth_bench/calibrate_preset.py).

Camera models (--lens): pin90 = our default 20 x 16 pinhole grid (slope .75) rendered at 90 deg;
wide120 = our equiangular 120 x 90 deg grid (RLV2_HFOV=120), sampled from a 120 deg rectilinear
render (rays outside its +-52 deg vertical are no-return).

With --calib, every --calib-every frames the same pose is also rendered through other camera
models and their rays, ground truth and surface class are saved for calibrate_hd_preset.py.

  python ext_depth_server.py FIFO_DIR THREADS VOXEL_DIR --lens wide120 --metrics out.json
"""
import argparse, json, math, os, sys, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "depth_bench"))

p = argparse.ArgumentParser()
p.add_argument("fifo"); p.add_argument("threads", type=int); p.add_argument("vdir")
p.add_argument("--lens", default="pin90"); p.add_argument("--W", type=int, default=160); p.add_argument("--H", type=int, default=120)
p.add_argument("--fidelity", default="high"); p.add_argument("--metrics", default="")
p.add_argument("--res", type=int, default=0, help="320: render 320 x 240 and use bm320_lrc")
p.add_argument("--save", default="", help="save a few frames (left, disparity) here")
p.add_argument("--calib", default="", help="also render these camera models (e.g. 160:pin90,320:wide120) for preset calibration")
p.add_argument("--calib-every", type=int, default=2); p.add_argument("--calib-out", default=".")
p.add_argument("--floor", type=float, default=.15, help="floor top / ceiling in map metres (procthor_missions.py prints them)")
p.add_argument("--ceil", type=float, default=2.75)
p.add_argument("--fx", type=int, default=1, help="1: OV2640-like camera effects (camera_effects.py) before StereoBM")
a = p.parse_args()
if a.res == 320:
    a.W, a.H = 320, 240
B = 0.06


def body_rays(lens):
    c = (np.arange(320) % 20) + .5; r = (np.arange(320) // 20) + .5
    if lens == "pin90":
        d = np.stack([np.ones(320), 1 - c / 10, .75 * (1 - r / 8)], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True), 45.0
    # wideNN: equiangular grid with NN deg horizontal FOV, VFOV = 0.75 x HFOV (grade.cpp RLV2_HFOV=NN)
    hfov = float(lens[4:]) if lens.startswith("wide") and lens[4:].isdigit() else 120.0
    hh, hv = math.radians(hfov / 2), math.radians(.75 * hfov / 2)
    az, el = (1 - c / 10) * hh, (1 - r / 8) * hv
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1), hfov / 2


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Rig:
    """One stereo camera model: HyperLux renderer + StereoBM + the 320-ray conversion."""

    def __init__(self, W, lens, scene, fidelity, fx=False, clean=True):
        import hyperlux, methods
        from hyperlux.camera import CameraSpec, RGB, Depth
        H = W * 3 // 4
        self.W, self.H, self.lens = W, H, lens
        self.rays, half = body_rays(lens)
        rays = self.rays
        self.tx = tx = math.tan(math.radians(half)); self.ty = ty = tx * H / W
        f = (W / 2) / tx
        u = (1 - (rays[:, 1] / rays[:, 0]) / tx) * W / 2
        v = (1 - (rays[:, 2] / rays[:, 0]) / ty) * H / 2
        inside = (rays[:, 0] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hw = max(2, int(round(W / 40)))          # half window: one ray cell (8 px at 160)
        self.win = []
        for k in range(320):
            if not inside[k]:
                self.win.append(None); continue
            uc, vc = int(u[k]), int(v[k])
            self.win.append((slice(max(0, vc - hw), min(H, vc + hw)), slice(max(0, uc - hw), min(W, uc + hw))))
        # range = planar depth * |pixel ray| / its forward component, per pixel
        xs = (1 - (np.arange(W) + .5) / (W / 2)) * tx; ys = (1 - (np.arange(H) + .5) / (H / 2)) * ty
        self.scale = np.sqrt(1 + xs[None, :] ** 2 + ys[:, None] ** 2).astype(np.float32)
        self.rgb_d, self.dep_d = RGB("rgba8"), Depth()
        # cameras: left, right (synchronized, clean pair), right at the sync offset (effects pair)
        self.fx, self.clean = fx, clean or not fx
        self.views = ([0, 1] if self.clean else [0]) + ([2] if fx else [])
        self.r = hyperlux.HyperLux(width=W, height=H, batch_size=len(self.views), num_probes=0,
                                   cameras=(CameraSpec("front", (self.rgb_d, self.dep_d)),), fidelity=fidelity)
        self.r.init(scene)
        self.bm = methods.BM320() if W >= 320 else methods.BM160()   # bm320_lrc: 64 disparities, block 11
        self.bm.W, self.bm.H = W, H
        self.calib = dict(f=f, W=W, H=H, baseline=B)
        self.inside = inside & (u >= self.bm.nd)   # metrics skip StereoBM's dead band (never matches)

    def to_rays(self, z, rng_g=None):
        rng_e = z * self.scale
        out = np.full(320, 12.0, np.float32); gt = np.full(320, np.nan, np.float32)
        for k, w in enumerate(self.win):
            if w is None:
                continue
            e = rng_e[w]; ev = e[np.isfinite(e)]
            if ev.size >= .1 * e.size:
                out[k] = min(12.0, float(np.percentile(ev, 5)))
            if rng_g is not None:
                g = rng_g[w]; gv = g[np.isfinite(g)]
                if gv.size:
                    gt[k] = float(np.percentile(gv, 5))
        return out, gt

    def frame(self, pos, R, motion=None, flight=None):
        """-> (rays clean or None, rays with camera effects or None, gt). motion = (v_world, w_body)."""
        import camera_effects as ce
        W, H, tx, ty = self.W, self.H, self.tx, self.ty
        poses = [(pos, R), (pos, R)]
        if self.fx:
            poses.append(ce.advance(pos, R, motion[0], motion[1], flight.sync))
        cams = np.zeros((len(self.views), 12), np.float32)
        for i, k in enumerate(self.views):
            p_, R_ = poses[k]
            fwd, left, up = R_[:, 0], R_[:, 1], R_[:, 2]
            off = B / 2 if k == 0 else -B / 2
            cams[i] = np.concatenate([p_ + off * left, fwd + tx * left + ty * up, -2 * tx * left, -2 * ty * up])
        r = self.r
        r.set_cameras(cams); r.render(); r.synchronize()
        img = np.asarray(r.outputs[self.rgb_d].read()).reshape(len(self.views), H, W, -1)[..., :3].astype(np.uint8)
        gzs = np.asarray(r.outputs[self.dep_d].read(), np.float32).reshape(len(self.views), H, W)
        gz = gzs[0]
        rng_g = np.where(np.isfinite(gz) & (gz > 0), gz * self.scale, np.nan)
        out_c = out_f = None
        if self.clean:
            z = self.bm.process(dict(left=img[0], right=img[1]), self.calib)["depth"]
            out_c, gt = self.to_rays(z, rng_g)
        else:
            _, gt = self.to_rays(np.full((H, W), np.nan, np.float32), rng_g)
        left_img = img[0]
        if self.fx:
            v_cam = R.T @ np.asarray(motion[0]); w_cam = np.asarray(motion[1])
            f = self.calib["f"]; mean = float(img[0].mean())
            L, info = ce.apply(img[0], gz, f, flight, 0, v_cam, w_cam, mean)
            Rr, _ = ce.apply(img[-1], gzs[-1], f, flight, 1, v_cam, w_cam, mean)
            z = self.bm.process(dict(left=L, right=Rr), self.calib)["depth"]
            out_f, _ = self.to_rays(z)
            left_img = L
            self.info = info
        self.last = (img[:1] if not self.fx else np.stack([left_img, img[0]]), z, rng_g)
        return out_c, out_f, gt

    def classes(self, pos, R, gt):
        """Per-ray surface class as calibrate_preset.py: 1 floor/ceiling (hit height), 2 textureless
        (>= half the window has 7x7 grey sd < 3 away from depth edges), 3 textured; 0 no GT."""
        import cv2
        img, _, rng_g = self.last
        g = cv2.cvtColor(img[-1], cv2.COLOR_RGB2GRAY)   # texture class from the clean render.astype(np.float32)
        m = cv2.boxFilter(g, -1, (7, 7)); sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (7, 7)) - m * m, 0))
        lz = np.log(np.where(np.isfinite(rng_g), rng_g, 50.0))
        edge = (np.abs(cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3))) > 0.4
        tl_px = (sd < 3) & ~edge
        hz = pos[2] + gt * (self.rays @ R.T)[:, 2]
        cls = np.zeros(320, np.uint8)
        for k, w in enumerate(self.win):
            if w is None or not np.isfinite(gt[k]):
                continue
            if hz[k] < a.floor + .06 or hz[k] > a.ceil - .06:
                cls[k] = 1
            else:
                cls[k] = 2 if tl_px[w].mean() >= .5 else 3
        return cls


def read_pose(req):
    buf = b""
    while len(buf) < 64:
        chunk = req.read(64 - len(buf))
        if not chunk:
            return None
        buf += chunk
    return np.frombuffer(buf, np.float32)


def worker(t):
    import cv2, hyperlux, camera_effects as ce
    from hyperlux.examples.data import procthor_scene_path
    origin = np.array(json.load(open(os.path.join(a.vdir, "origin.json")))["origin"])
    scene = hyperlux.load_scene(procthor_scene_path(), fidelity=a.fidelity)
    main = Rig(a.W, a.lens, scene, a.fidelity, fx=a.fx, clean=False)
    extra = [Rig(int(v.split(":")[0]), v.split(":")[1], scene, a.fidelity, fx=a.fx, clean=True)
             for v in a.calib.split(",")] if a.calib else []
    req = open(os.path.join(a.fifo, f"req{t}"), "rb", buffering=0)
    resp = open(os.path.join(a.fifo, f"resp{t}"), "wb", buffering=0)
    est_all, gt_all, n = [], [], 0
    cal = [dict(est=[], fx=[], gt=[], cls=[]) for _ in extra]
    flight, key, fxlog = None, None, []
    while True:
        q = read_pose(req)
        if q is None:
            break
        pos = q[:3].astype(np.float64) + origin
        R = rot(q[3:7].astype(np.float64))
        motion = (q[8:11].astype(np.float64), q[11:14].astype(np.float64))
        if (int(q[7]), int(q[14])) != key:   # new flight: draw its camera parameters
            key = (int(q[7]), int(q[14]))
            flight = ce.FlightFx(key[0] * 1009 + key[1] * 7 + 12345)
            fxlog.append(flight.summary())
        oc, of, gt = main.frame(pos, R, motion, flight)
        out = of if a.fx else oc
        resp.write(out.tobytes())
        est_all.append(out); gt_all.append(gt); n += 1
        if extra and n % a.calib_every == 0:   # the same pose through the other camera models
            for rig, c in zip(extra, cal):
                ec, ef, g = rig.frame(pos, R, motion, flight)
                c["est"].append(ec); c["gt"].append(g); c["cls"].append(rig.classes(pos - origin, R, g))
                if ef is not None:
                    c["fx"].append(ef)
        if a.save and t == 0 and n % 200 == 1:
            os.makedirs(a.save, exist_ok=True)
            img, z, _ = main.last
            cv2.imwrite(os.path.join(a.save, f"{a.lens}_{n}_left.png"), cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(a.save, f"{a.lens}_{n}_depth.png"), np.clip(np.nan_to_num(z, nan=0) * 60, 0, 255).astype(np.uint8))
    if est_all:
        np.savez_compressed(os.path.join(a.fifo, f"rays{t}.npz"), est=np.array(est_all), gt=np.array(gt_all), inside=main.inside)
    for rig, c in zip(extra, cal):
        if c["est"]:
            kw = dict(fx=np.array(c["fx"])) if c["fx"] else {}
            np.savez_compressed(os.path.join(a.calib_out, f"cal_{rig.W}_{rig.lens}_{t}.npz"), est=np.array(c["est"]),
                                gt=np.array(c["gt"]), cls=np.array(c["cls"]), inside=rig.inside, **kw)
    if a.calib and fxlog:
        json.dump(fxlog, open(os.path.join(a.calib_out, f"fx_{t}.json"), "w"))


def despeckle(e):
    out = e.copy()
    for r in range(16):
        for c in range(20):
            d = e[r, c]
            if not np.isfinite(d):
                continue
            nb = e[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if (np.abs(nb - d) < max(0.3, 0.15 * d)).sum() - 1 <= 0:
                out[r, c] = np.nan
    return out


def metrics():
    est, gt = [], []
    inside = None
    for t in range(a.threads):
        f = os.path.join(a.fifo, f"rays{t}.npz")
        if os.path.exists(f):
            d = np.load(f); est.append(d["est"]); gt.append(d["gt"]); inside = d["inside"]
    if not est:
        return
    est = np.concatenate(est); gt = np.concatenate(gt)
    e = np.where(est >= 12, np.nan, est).reshape(-1, 16, 20)
    res = dict(lens=a.lens, frames=int(len(est)), rays_outside_render_or_dead_band=int((~inside).sum()))
    for tag, ee in (("", e), ("desp_", np.stack([despeckle(x) for x in e]))):
        ee = ee.reshape(len(est), 320); g = gt
        base = np.isfinite(g) & (g < 3.0) & inside[None]
        unk = ~np.isfinite(ee); ret = base & ~unk
        tl = np.maximum(.10, .15 * np.nan_to_num(g, nan=1))
        res[tag + "hole"] = float((base & unk).sum() / base.sum())
        res[tag + "phantom"] = float((ret & (ee < g - tl)).sum() / base.sum())
        n15 = ret & (g < 1.5)
        res[tag + "false_free"] = float((n15 & (ee > g + tl)).sum() / max(1, n15.sum()))
        ok = ret & (np.abs(ee - g) <= tl)
        res[tag + "k"] = float(np.median(np.abs(ee - g)[ok] / g[ok] ** 2) * 1.4826) if ok.any() else None
        res[tag + "rays_gt_lt3"] = int(base.sum())
    print(json.dumps(res), flush=True)
    if a.metrics:
        json.dump(res, open(a.metrics, "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(a.fifo, exist_ok=True)
    for t in range(a.threads):
        for nm in (f"req{t}", f"resp{t}"):
            pth = os.path.join(a.fifo, nm)
            if not os.path.exists(pth):
                os.mkfifo(pth)
    print("ready", flush=True)
    ps = [mp.Process(target=worker, args=(t,)) for t in range(a.threads)]
    for x in ps: x.start()
    for x in ps: x.join()
    metrics()

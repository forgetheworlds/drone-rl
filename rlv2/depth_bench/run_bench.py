"""Score depth pipelines on the benchmark.

  python run_bench.py --methods sgbm320,sgbm320_lrc_tex --split test --profile ov2640 [--max-seqs N]
Appends one JSON line per (method, split, profile) to results/<split>_<profile>.jsonl.
External methods: --plugin path/to/file.py (any @register'ed Method in it becomes available).
"""
import argparse
import importlib.util
import json
import os
import time

import cv2
import numpy as np

import dataset
import methods
import metrics

# Rough single-core slowdown vs this box's Xeon core for OpenCV-style integer image code.
# Pi Zero 2 W: Cortex-A53 @1 GHz w/ NEON (~8-12x); Pi Zero (v1/W): ARM1176 @1 GHz, no NEON (~30-50x);
# ESP32-P4: dual RISC-V @360-400 MHz with PIE SIMD, needs a C/ESP-DL port (~20-40x per core).
SLOWDOWN = {"pi_zero2w_1core": 10.0, "pi_zero_v1": 40.0, "esp32p4_1core": 30.0}


def run(method_name, split, profile, max_seqs=None, data_dir=dataset.DEFAULT_DATA):
    cv2.setNumThreads(1)
    M = methods.REGISTRY[method_name]()
    acc = metrics.Accum(); kvals = []; times = []
    nseq = 0
    for seq in dataset.sequences(split, profile, M.baseline, data_dir, max_seqs):
        M.reset(seq.calib)
        est_d, vo, gt_d = [], [], []
        for k in range(seq.T):
            fr = seq.frame(k)
            inp = dict(left=fr["left"], right=fr["right"], pose=fr["pose"], k=k)
            t0 = time.perf_counter()
            out = M.process(inp, seq.calib)
            times.append(time.perf_counter() - t0)
            z = out["depth"].astype(np.float32)
            far = out.get("far")
            if z.shape != fr["depth"].shape:
                z = cv2.resize(z, fr["depth"].shape[::-1], interpolation=cv2.INTER_NEAREST)
            if far is not None and far.shape != z.shape:
                far = cv2.resize(far.astype(np.uint8), z.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            metrics.frame_metrics(acc, z, fr, kvals)
            metrics.nav_frame_metrics(acc, np.where(far, np.inf, z) if far is not None else z, fr, getattr(M, "zmax", None))
            est_d.append(z); vo.append(fr["pose"]); gt_d.append(fr["depth"])
        metrics.voxel_metrics(acc, est_d, np.stack(vo), gt_d, seq.pose_gt, seq.calib)
        nseq += 1
    k = np.concatenate(kvals) if kvals else np.array([np.nan])
    r = {key: round(acc.rate(key), 4) for key in acc.c}
    r["vox_false_free_1m_per_seq"] = round(acc.c.get("vox_false_free_1m", [0, 1])[0] / max(nseq, 1), 2)
    r.pop("vox_false_free_1m", None)
    ms = 1000 * float(np.median(times))
    res = dict(method=method_name, note=M.note, split=split, profile=profile, baseline=M.baseline, seqs=nseq,
               frames=len(times), k_sigma_over_z2=round(float(1.4826 * np.nanmedian(k)), 4), ms_box=round(ms, 2),
               est_ms={kk: round(ms * v, 1) for kk, v in SLOWDOWN.items()},
               planner_cost=round(100 * (10 * r["ray_false_free_near"] + 3 * r["ray_false_occ"] + r["ray_unknown"]), 2),
               # Headline for planning toward a waypoint: errors inside the stopping distance at the
               # student's 1 m/s cap, false-free < 1.5 m, phantoms < 3 m. Far unknown is not charged.
               stop_cost=round(100 * (10 * r["nav_ff_1.5"] + 10 * r["stop1_ff"] + r["stop1_unknown"] + 3 * r["nav_fo_3m"]), 2),
               **r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="sgbm320")
    ap.add_argument("--split", default="test")
    ap.add_argument("--profile", default="ov2640")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--plugin", default="")
    ap.add_argument("--data", default=dataset.DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    if a.plugin:
        spec = importlib.util.spec_from_file_location("plugin", a.plugin)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    os.makedirs(a.out, exist_ok=True)
    for prof in a.profile.split(","):
        for m in a.methods.split(","):
            t0 = time.time()
            res = run(m, a.split, prof, a.max_seqs, a.data)
            res["wall_s"] = round(time.time() - t0, 1)
            with open(os.path.join(a.out, f"{a.split}_{prof}.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps({k: res[k] for k in ("method", "profile", "stop_cost", "stop1_ff", "stop1_unknown", "nav_ff_1.5", "nav_fo_3m", "planner_cost", "ray_false_free_near", "ray_false_occ",
                                                  "ray_unknown", "px_hole", "px_false_near", "vox_false_free_1m_per_seq",
                                                  "ms_box", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()

"""HyperDrone/HyperLux throughput on this Mac, in the shapes our simulator uses.

One configuration per process run (each HyperLux config is JIT-compiled once and cached):

  python bench_hyperdrone.py --scene procthor --output depth --width 64 --height 64 --envs 4096 --mode physics

scene:  simple20  - a 16 x 16 x 4 m room with 20 random boxes (the paper's "20 primitive objects" scene type)
        procthor  - ProcTHOR-Train-1 (the paper's indoor scene, fetched by conta)
        voxel:<task path> - one of our drone-rl courses exported as exposed voxel faces
mode:   render    - cameras fixed, timed render loop (the paper's render-only protocol)
        physics   - every step: L2F dynamics step for all drones (CPU, one thread, as on Metal),
                    camera bases from the new poses, upload, render. One rendered frame per 10 ms step.
        physics50 - our tick: 5 L2F substeps (50 ms) per rendered frame, like rlv2's nav tick.
Prints one JSON line with frames/s, env-steps/s, rays/s.
"""
import argparse
import json
import math
import platform
import subprocess
import time

import numpy as np

import hyperlux
from hyperlux.camera import CameraSpec, Depth, RGB

import voxmap

p = argparse.ArgumentParser()
p.add_argument("--scene", default="procthor")
p.add_argument("--output", choices=("depth", "rgb"), default="depth")
p.add_argument("--width", type=int, default=64)
p.add_argument("--height", type=int, default=64)
p.add_argument("--envs", type=int, default=4096)
p.add_argument("--fidelity", default="low")
p.add_argument("--mode", choices=("render", "physics", "physics50"), default="render")
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--fov", type=float, default=90.0)
p.add_argument("--readback", action="store_true", help="copy the frames to numpy every step (what a host-side trainer needs)")
a = p.parse_args()


def simple20(rng):
    scene = hyperlux.Scene()
    room = hyperlux.Object(name="room")
    half, height = 8.0, 4.0
    quads = [voxmap.box_quads((-half, -half, -0.1), (half, half, 0.0)),
             voxmap.box_quads((-half, -half, height), (half, half, height + 0.1)),
             voxmap.box_quads((-half - 0.1, -half, 0), (-half, half, height)),
             voxmap.box_quads((half, -half, 0), (half + 0.1, half, height)),
             voxmap.box_quads((-half, -half - 0.1, 0), (half, -half, height)),
             voxmap.box_quads((-half, half, 0), (half, half + 0.1, height))]
    for _ in range(20):
        c = np.append(rng.uniform(-half * 0.8, half * 0.8, 2), rng.uniform(0.3, 3.0))
        e = rng.uniform(0.2, 0.8, 3)
        quads.append(voxmap.box_quads(c - e, c + e))
    q = np.concatenate(quads)
    return voxmap.hyperlux_scene(q, seed=1, palette=12)


rng = np.random.default_rng(0)
t_build = time.perf_counter()
if a.scene == "simple20":
    scene = simple20(rng)
elif a.scene == "procthor":
    from hyperlux.examples.data import procthor_scene_path
    scene = hyperlux.load_scene(procthor_scene_path(), fidelity=a.fidelity)
elif a.scene.startswith("voxel:"):
    task = voxmap.load_task(a.scene[6:])
    scene = voxmap.hyperlux_scene(voxmap.world_quads(task))
else:
    raise SystemExit("unknown scene")

out_desc = Depth() if a.output == "depth" else RGB("rgba8")
renderer = hyperlux.HyperLux(width=a.width, height=a.height, batch_size=a.envs, num_probes=0,
                             cameras=(CameraSpec("front", (out_desc,)),), fidelity=a.fidelity)
renderer.init(scene)
t_init = time.perf_counter() - t_build
sampler = hyperlux.ClearanceSampler(scene, clearance=0.4)
positions = sampler.sample(a.envs, seed=0)
del sampler

aspect = a.width / a.height


def static_cameras():
    yaw = rng.uniform(-math.pi, math.pi, a.envs)
    pitch = rng.uniform(-0.2, 0.2, a.envs)
    f = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], 1)
    up = np.stack([-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch)], 1)
    scale = 2 * math.tan(math.radians(a.fov) / 2)
    du = np.cross(f, up); du /= np.linalg.norm(du, axis=1, keepdims=True); du *= scale
    dv = np.cross(du, f); dv /= np.linalg.norm(dv, axis=1, keepdims=True); dv *= scale / aspect
    d00 = f - 0.5 * du + 0.5 * dv
    return np.concatenate([positions, d00, du, -dv], 1).astype(np.float32)


frames = 0
steps = 0
host = None
if a.mode == "render":
    renderer.set_cameras(static_cameras())
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        renderer.render()
    renderer.synchronize()
    t0 = time.perf_counter()
    it = 0
    while True:
        renderer.render_launch()
        it += 1
        if it % 10 == 0:
            renderer.render_sync()
            if a.readback:
                host = renderer.outputs[out_desc].read()
            if time.perf_counter() - t0 >= a.seconds:
                break
    renderer.synchronize()
    wall = time.perf_counter() - t0
    frames = it * a.envs
    steps = frames
    substeps = 1
else:
    from hyperdrone import dynamics
    sim = dynamics.Sim(num_drones=a.envs, parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, device="cpu")
    sim.reset()
    sim.state["position"] = positions
    hover = sim.parameters["dynamics"]["dynamics"]["hovering_throttle_relative"]
    base = np.full((a.envs, sim.action_dim), 2 * hover - 1, np.float32)
    substeps = 5 if a.mode == "physics50" else 1
    yaw_diff = (rng.uniform(0.01, 0.03, a.envs) * rng.choice((-1, 1), a.envs))[:, None] * np.array([1, -1, 1, -1], np.float32)
    z0 = positions[:, 2].copy()

    def actions():
        s = sim.state
        corr = 0.4 * (z0 - s["position"][:, 2]) - 0.25 * s["linear_velocity"][:, 2]
        return np.clip(base + yaw_diff + corr[:, None], -1, 1).astype(np.float32)

    def tick():
        for _ in range(substeps):
            sim.step(actions())
        renderer.set_cameras(sim.camera_bases(fov=a.fov, aspect=aspect))
        renderer.render()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        tick()
    renderer.synchronize()
    # time the pieces separately too
    t_phys = t_rend = 0.0
    t0 = time.perf_counter()
    it = 0
    while time.perf_counter() - t0 < a.seconds:
        ta = time.perf_counter()
        for _ in range(substeps):
            sim.step(actions())
        cams = sim.camera_bases(fov=a.fov, aspect=aspect)
        tb = time.perf_counter()
        renderer.set_cameras(cams)
        renderer.render()
        renderer.render_sync()
        if a.readback:
            host = renderer.outputs[out_desc].read()
        tc = time.perf_counter()
        t_phys += tb - ta
        t_rend += tc - tb
        it += 1
    renderer.synchronize()
    wall = time.perf_counter() - t0
    frames = it * a.envs
    steps = frames

rays_per_frame = a.width * a.height
res = dict(scene=a.scene.split("/")[-1], output=a.output, res=f"{a.width}x{a.height}", envs=a.envs, fidelity=a.fidelity,
           mode=a.mode, readback=a.readback, backend=renderer.backend, init_s=round(t_init, 2),
           wall_s=round(wall, 2), frames_per_s=round(frames / wall), env_steps_per_s=round(steps / wall),
           sim_seconds_per_s=round(steps * substeps * 0.01 / wall), rays_per_s=round(frames * rays_per_frame / wall))
if a.mode != "render":
    res["physics_ms_per_step"] = round(t_phys / it * 1e3, 3)
    res["render_ms_per_step"] = round(t_rend / it * 1e3, 3)
print(json.dumps(res), flush=True)

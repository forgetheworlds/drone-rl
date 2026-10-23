#!/usr/bin/env python3
"""Render a flown mission as a 3D replay (GIF or MP4) from its recorded traces.

Everything drawn comes from the run: the drone pose and depth frames from depth-pose.bin, the
flown path and current waypoint from trace.csv, and the moving boxes from the task file.
Static surfaces are drawn faintly until the drone's depth camera has hit them.
"""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAY_SLOPE = 0.75
COLORS = dict(background="#f4f1ea", ghost="#8d99a6", seen="#3d5a80", path="#ee6c4d", drone="#1b1b1e",
              rotor="#f77f00", goal="#f2a541", reached="#2a9d8f", mover="#c1121f", ray="#ffd166")


def read_task(path):
    tokens = path.read_text().split()
    scale = float(tokens[1])
    count = int(tokens[5])
    goals = np.array(tokens[6:6 + 3 * count], dtype=float).reshape(count, 3)
    rest = tokens[6 + 3 * count:]
    movers = []
    for i in range(int(rest[0])):
        v = [float(x) for x in rest[1 + 10 * i:11 + 10 * i]]
        movers.append(dict(center=np.array(v[0:3]), half=np.array(v[3:6]), axis=int(v[6]),
                           amplitude=v[7], period=v[8], phase=v[9]))
    return path.parent / tokens[0], scale, goals, movers


def read_grid(path):
    with path.open() as f:
        shape = tuple(int(x) for x in f.readline().split()[1:])
        cells = np.loadtxt(f, dtype=int, ndmin=2)
    grid = np.zeros(shape, dtype=bool)
    grid[tuple(cells.T)] = True
    return grid


def merged_boxes(grid):
    grid = grid.copy()
    boxes = []
    while grid.any():
        x, y, z = np.argwhere(grid)[0]
        X = x + 1
        while X < grid.shape[0] and grid[X, y, z]:
            X += 1
        Y = y + 1
        while Y < grid.shape[1] and grid[x:X, Y, z].all():
            Y += 1
        Z = z + 1
        while Z < grid.shape[2] and grid[x:X, y:Y, Z].all():
            Z += 1
        grid[x:X, y:Y, z:Z] = False
        boxes.append((np.array([x, y, z]), np.array([X, Y, Z])))
    return boxes


def box_mesh(boxes, scale):
    if not boxes:
        return None
    return pv.merge([pv.Box(bounds=(a[0] * scale, b[0] * scale, a[1] * scale, b[1] * scale,
                                    a[2] * scale, b[2] * scale)) for a, b in boxes])


def ray_directions():
    k = np.arange(320)
    d = np.stack([np.ones(320), 1 - (k % 20 + 0.5) / 10, RAY_SLOPE * (1 - (k // 20 + 0.5) / 8)], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def drone_model(radius):
    arm = radius * 0.75
    k = radius / 0.18
    parts = [pv.Box(bounds=(-0.05 * k, 0.05 * k, -0.035 * k, 0.035 * k, -0.02 * k, 0.02 * k)),
             pv.Cone(center=(0.07 * k, 0, 0), direction=(1, 0, 0), height=0.05 * k, radius=0.025 * k, resolution=12)]
    for angle in np.radians([45, 135, 225, 315]):
        tip = np.array([np.cos(angle), np.sin(angle), 0]) * arm
        parts.append(pv.Cylinder(center=tip / 2, direction=tip, radius=0.008 * k, height=arm, resolution=8))
    body = pv.merge(parts)
    rotors = pv.merge([pv.Disc(center=(np.cos(a) * arm, np.sin(a) * arm, 0.025 * k), inner=0, outer=radius * 0.42,
                               c_res=24) for a in np.radians([45, 135, 225, 315])])
    return body, rotors


def posed(mesh, position, rotation):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = position
    return mesh.transform(m, inplace=False)


def frustum(position, rotation, length):
    corners = np.array([[1, s, RAY_SLOPE * t] for s, t in ((1, 1), (-1, 1), (-1, -1), (1, -1))]) * 0.95
    points = np.vstack([position, position + (rotation @ corners.T).T * length])
    faces = np.hstack([[3, 0, i, i % 4 + 1] for i in range(1, 5)])
    return pv.PolyData(points, faces)


def mover_center(mover, t):
    center = mover["center"].copy()
    center[mover["axis"]] += mover["amplitude"] * np.cos(2 * np.pi * t / mover["period"] + mover["phase"])
    return center


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def depth_inset(depth, height):
    image = (np.clip(1 - depth.reshape(16, 20) / 6, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([image * 0.93, image * 0.82, image * 0.4 + 30], -1).astype(np.uint8)
    return Image.fromarray(rgb).resize((height * 20 // 16, height), Image.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("portfolio")
    parser.add_argument("mission")
    parser.add_argument("output", type=Path, help=".gif or .mp4")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--speed", type=float, default=5, help="playback speed-up")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--zoom", type=float, default=1.0, help="camera distance as a fraction of the scene size")
    parser.add_argument("--elevation", type=float, default=38, help="camera elevation (degrees)")
    parser.add_argument("--azimuth", type=float, default=-60, help="starting camera azimuth (degrees)")
    parser.add_argument("--orbit", type=float, default=0, help="camera orbit over the clip (degrees)")
    parser.add_argument("--ceiling", type=float, default=1.2, help="hide geometry this far above the path (m)")
    parser.add_argument("--drone-scale", type=float, default=2, help="drawn size of the drone model")
    parser.add_argument("--title", help="caption text (default: mission name)")
    parser.add_argument("--still", type=float, nargs="*", help="only save PNG stills at these mission times")
    args = parser.parse_args()

    task = ROOT / "missions" / args.portfolio / f"{args.mission}.task"
    report = args.results / args.portfolio / f"{args.mission}.json"
    map_path, scale, goals, movers = read_task(task)
    result = json.loads(report.read_text())
    rows = list(csv.DictReader(open(f"{report}.trace.csv")))
    times = np.array([float(r["time_s"]) for r in rows])
    path = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
    legs = np.array([int(r["leg"]) for r in rows])
    captures = np.fromfile(f"{report}.depth-pose.bin", dtype="<f4").reshape(-1, 333)
    capture_times, capture_pos = captures[:, 0], captures[:, 1:4]
    capture_rot, capture_depth = captures[:, 4:13].reshape(-1, 3, 3), captures[:, 13:]

    grid = read_grid(map_path)
    shape = np.array(grid.shape)
    points = np.vstack([path, goals])
    lo = np.floor((points.min(0) - [1.5, 1.5, 3.0]) / scale).astype(int).clip(0)
    hi = np.minimum(np.ceil((points.max(0) + 1.5) / scale).astype(int), shape)
    hi[2] = min(shape[2], int(np.ceil((points[:, 2].max() + args.ceiling) / scale)))
    window = np.zeros_like(grid)
    window[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    shell = np.zeros_like(grid)
    shell[1:-1, 1:-1, 1:-1] = True
    static = grid & window & shell
    floor_z = None
    if lo[2] == 0 and grid[:, :, 0].mean() > 0.5:
        floor_z = scale
        static[:, :, 0] = False

    directions = ray_directions()
    hits = []
    for position, rotation, depth in zip(capture_pos, capture_rot, capture_depth):
        mask = depth < 11.9
        world = position + (rotation @ directions[mask].T).T * (depth[mask, None] + scale * 0.25)
        hits.append(np.floor(world / scale).astype(int))
    first_seen = np.full(grid.shape, np.inf)
    for t, cells in zip(capture_times, hits):
        cells = cells[(cells >= 0).all(1) & (cells < shape).all(1)]
        cells = cells[static[tuple(cells.T)]]
        idx = tuple(cells.T)
        first_seen[idx] = np.minimum(first_seen[idx], t)

    height = int(args.width * 9 / 16) // 2 * 2
    plotter = pv.Plotter(off_screen=True, window_size=(args.width * 2, height * 2), lighting="three lights")
    plotter.set_background(COLORS["background"])
    plotter.enable_anti_aliasing("ssaa")
    ghost = box_mesh(merged_boxes(static), scale)
    if ghost is not None:
        plotter.add_mesh(ghost, color=COLORS["ghost"], opacity=0.2, show_edges=False)
    if floor_z is not None:
        lo_m, hi_m = lo * scale, hi * scale
        floor = pv.Plane(center=((lo_m[0] + hi_m[0]) / 2, (lo_m[1] + hi_m[1]) / 2, floor_z),
                         i_size=hi_m[0] - lo_m[0], j_size=hi_m[1] - lo_m[1], i_resolution=int((hi_m[0] - lo_m[0]) / 0.5),
                         j_resolution=int((hi_m[1] - lo_m[1]) / 0.5))
        plotter.add_mesh(floor, color="#e4dfd3", show_edges=True, edge_color="#d3cdbf", line_width=1)
    body, rotors = drone_model(0.18 * args.drone_scale)
    goal_actors = [plotter.add_mesh(pv.Sphere(radius=0.14, center=g), color=COLORS["goal"], smooth_shading=True)
                   for g in goals]
    for g in goals:
        plotter.add_mesh(pv.Line((g[0], g[1], floor_z if floor_z is not None else g[2] - 1), g),
                         color=COLORS["goal"], line_width=2, opacity=0.6)

    seen_cells = np.argwhere(np.isfinite(first_seen))
    seen_times = first_seen[tuple(seen_cells.T)]
    order = np.argsort(seen_times)
    seen_cells, seen_times = seen_cells[order], seen_times[order]
    cube = pv.Cube(x_length=scale * 0.96, y_length=scale * 0.96, z_length=scale * 0.96)

    duration = times[-1]
    frame_times = np.append(np.arange(0, duration, args.speed / args.fps), [duration] * int(args.fps * 1.5))
    if args.still:
        frame_times = np.array(args.still)
    center = (points.min(0) + points.max(0)) / 2
    extent = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 4.0)
    distance = extent * 1.15 * args.zoom
    dynamic = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for n, t in enumerate(frame_times):
            for actor in dynamic:
                plotter.remove_actor(actor)
            dynamic = []
            i = min(np.searchsorted(times, t), len(times) - 1)
            c = int(np.argmin(np.abs(capture_times - t)))
            position = np.array([np.interp(t, times, path[:, k]) for k in range(3)])
            rotation = capture_rot[c]

            count = np.searchsorted(seen_times, t, side="right")
            if count:
                cloud = pv.PolyData((seen_cells[:count] + 0.5) * scale)
                cloud["z"] = cloud.points[:, 2]
                dynamic.append(plotter.add_mesh(cloud.glyph(geom=cube, orient=False, scale=False), scalars="z",
                                                cmap=["#a8dadc", "#5e8fb5", "#3d5a80"], show_scalar_bar=False,
                                                smooth_shading=False))
            trail = np.vstack([path[:i + 1], position])
            if len(trail) > 2:
                dynamic.append(plotter.add_mesh(pv.Spline(trail, len(trail)).tube(radius=0.03), color=COLORS["path"],
                                                smooth_shading=True))
            dynamic.append(plotter.add_mesh(posed(body, position, rotation), color=COLORS["drone"], smooth_shading=True))
            dynamic.append(plotter.add_mesh(pv.Sphere(radius=0.18, center=position), color=COLORS["path"],
                                            opacity=0.35, smooth_shading=True))
            dynamic.append(plotter.add_mesh(posed(rotors, position, rotation), color=COLORS["rotor"], opacity=0.8))
            dynamic.append(plotter.add_mesh(frustum(position, rotation, 1.2), color=COLORS["ray"], opacity=0.18))
            depth = capture_depth[c]
            mask = depth < 11.9
            ends = capture_pos[c] + (rotation @ directions[mask].T).T * depth[mask, None]
            cells = np.floor((ends + (rotation @ directions[mask].T).T * scale * 0.25) / scale).astype(int)
            inside = (cells >= 0).all(1) & (cells < shape).all(1)
            keep = np.zeros(len(ends), dtype=bool)
            keep[inside] = static[tuple(cells[inside].T)]
            for mover in movers:
                keep |= (np.abs(ends - mover_center(mover, capture_times[c])) <= mover["half"] + scale).all(1)
            ends = ends[keep]
            if len(ends):
                dynamic.append(plotter.add_mesh(pv.PolyData(ends), color=COLORS["ray"], point_size=7,
                                                render_points_as_spheres=True))
            for mover in movers:
                middle = mover_center(mover, t)
                lo_b, hi_b = middle - mover["half"], middle + mover["half"]
                dynamic.append(plotter.add_mesh(pv.Box(bounds=(lo_b[0], hi_b[0], lo_b[1], hi_b[1], lo_b[2], hi_b[2])),
                                                color=COLORS["mover"], opacity=0.85))
            done = t >= duration and result["outcome"] == "success"
            for k, actor in enumerate(goal_actors):
                passed = k < legs[i] or done
                actor.prop.color = COLORS["reached"] if passed else COLORS["goal"]

            if floor_z is not None:
                shadow = pv.Disc(center=(position[0], position[1], floor_z + 0.01), inner=0, outer=0.12 * args.drone_scale, c_res=24)
                dynamic.append(plotter.add_mesh(shadow, color="#1b1b1e", opacity=0.25))
                dynamic.append(plotter.add_mesh(pv.Line((position[0], position[1], floor_z), position),
                                                color="#1b1b1e", opacity=0.3, line_width=1))
            azimuth = np.radians(args.azimuth + args.orbit * min(t, duration) / max(duration, 1e-6))
            elevation = np.radians(args.elevation)
            offset = np.array([np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation),
                               np.sin(elevation)]) * distance
            plotter.camera.position = center + offset
            plotter.camera.focal_point = center
            plotter.camera.up = (0, 0, 1)
            plotter.camera.view_angle = 30

            frame = Image.fromarray(plotter.screenshot(return_img=True)).resize((args.width, height), Image.LANCZOS)
            banner = Image.new("RGBA", (args.width, int(height * 0.13)), (244, 241, 234, 215))
            frame.paste(banner, (0, 0), banner)
            draw = ImageDraw.Draw(frame)
            speed = np.linalg.norm(np.gradient(path, times, axis=0)[i])
            title = args.title or args.mission
            draw.text((14, 10), title, fill="#1b1b1e", font=font(max(14, args.width // 40)))
            reached = len(goals) if done else min(int(legs[i]), len(goals))
            status = {"success": "all waypoints reached"}.get(result["outcome"], result["outcome"].replace("_", " "))
            line = f"t {min(t, duration):5.1f} s   {speed:4.2f} m/s   waypoints {reached}/{len(goals)}   {args.speed:g}x"
            if t >= duration:
                line = f"t {duration:5.1f} s   {status}   contacts {int(result['contact'])}"
            draw.text((14, 14 + max(14, args.width // 40) + 4), line, fill="#3a3a3a", font=font(max(11, args.width // 56)))
            inset_h = height // 5
            inset = depth_inset(depth, inset_h)
            label_font = font(max(10, args.width // 64))
            label_width = int(draw.textlength("depth camera 16x20", font=label_font))
            x0 = args.width - max(inset.width, label_width) - 14
            y0 = height - inset_h - 30
            frame.paste(inset, (x0, y0))
            draw.rectangle((x0 - 1, y0 - 1, x0 + inset.width, y0 + inset_h), outline="#1b1b1e")
            draw.text((x0, y0 + inset_h + 4), "depth camera 16x20", fill="#3a3a3a", font=label_font)
            if args.still:
                frame.save(args.output.with_name(f"{args.output.stem}-{t:05.1f}.png"))
                continue
            frame.save(f"{tmp}/{n:05d}.png")

        if args.still:
            return
        source = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", f"{tmp}/%05d.png"]
        if args.output.suffix == ".gif":
            palette = "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
            subprocess.run([*source, "-vf", palette, str(args.output)], check=True)
        else:
            subprocess.run([*source, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(args.output)], check=True)
    plotter.close()
    print(args.output)


if __name__ == "__main__":
    main()

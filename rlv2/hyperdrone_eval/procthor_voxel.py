"""Voxelize a HyperDrone scene (.glb, Y-up) into our .3dmap format in HyperLux's Z-up frame.

HyperLux loads glTF Y-up as (x, -z, y). Surfaces are sampled densely and every 0.1 m cell a
sample falls in is solid, so surfaces are up to one cell thick (contact geometry is at most
~0.1 m conservative vs the mesh).

  python procthor_voxel.py OUT_DIR [--pitch 0.1]   -> OUT_DIR/procthor.3dmap + origin.json
"""
import argparse, json, os
import numpy as np, trimesh

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--pitch", type=float, default=0.1)
p.add_argument("--glb", default="")
a = p.parse_args()
if not a.glb:
    from hyperlux.examples.data import procthor_scene_path
    a.glb = str(procthor_scene_path())
mesh = trimesh.load(a.glb, file_type="glb", force="scene").to_geometry()
v = np.asarray(mesh.vertices)
mesh.vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
n = int(mesh.area / (a.pitch / 4) ** 2)
pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
pts = np.concatenate([pts, mesh.vertices])
lo = mesh.bounds[0] - 2 * a.pitch
idx = np.floor((pts - lo) / a.pitch).astype(int)
dims = idx.max(0) + 3
occ = np.zeros(dims, bool)
occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
os.makedirs(a.out, exist_ok=True)
cells = np.argwhere(occ)
with open(os.path.join(a.out, "procthor.3dmap"), "w") as f:
    f.write(f"voxel {dims[0]} {dims[1]} {dims[2]}\n")
    np.savetxt(f, cells, fmt="%d")
json.dump(dict(origin=lo.tolist(), pitch=a.pitch, dims=dims.tolist(), area=float(mesh.area), samples=n),
          open(os.path.join(a.out, "origin.json"), "w"))
print(json.dumps(dict(origin=lo.tolist(), dims=dims.tolist(), solid=int(occ.sum()), area=float(mesh.area))))

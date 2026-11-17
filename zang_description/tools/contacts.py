"""Mesh-based contact graph between leaf parts (surface samples + KD-tree)."""
import sys, pickle, itertools
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SPACING = 0.3  # mm between surface samples
TOL = 0.5      # mm gap still counted as contact

data = pickle.load(open(sys.argv[1], "rb"))
parts = data["parts"]
rng = np.random.default_rng(0)
trees = []
for p in parts:
    m = trimesh.Trimesh(p["V"], p["F"], process=False)
    n = int(np.clip(m.area / SPACING**2, 200, 400_000))
    pts, _ = trimesh.sample.sample_surface(m, n, seed=0)
    pts = np.vstack([pts, p["V"]])
    p["_pts"] = pts
    trees.append(cKDTree(pts))

edges = []
for i, j in itertools.combinations(range(len(parts)), 2):
    a, b = parts[i]["bbox"], parts[j]["bbox"]
    if np.any(a[0] - b[1] > TOL) or np.any(b[0] - a[1] > TOL):
        continue
    small, big = (i, j) if len(parts[i]["_pts"]) < len(parts[j]["_pts"]) else (j, i)
    d, _ = trees[big].query(parts[small]["_pts"], distance_upper_bound=TOL)
    dmin = d.min()
    if dmin <= TOL:
        edges.append((i, j, float(dmin), int((d <= TOL).sum())))

for p in parts:
    del p["_pts"]
data["edges"] = edges
pickle.dump(data, open(sys.argv[1], "wb"))
print(f"{len(edges)} contacts")

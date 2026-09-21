"""Record the irregular-density ground-truth counterexample from the shipped functions."""
import json, hashlib, subprocess, numpy as np, open3d as o3d
from npa.smoke.test_open3d_functional import _unit_cube, _CUBE_FACES
from npa.workbench.open3d.runner import _support, _crop_justification

rng = np.random.default_rng(7)
mesh = _unit_cube(o3d)
verts = np.asarray(mesh.vertices)
tris = np.asarray(mesh.triangles)
axes = np.eye(3)

# Ground truth: distance from every vertex to the analytic unit cube surface.
def distance_to_cube(p):
    inside = np.all((p >= 0) & (p <= 1), axis=1)
    d_faces = np.minimum(np.abs(p), np.abs(1 - p)).min(axis=1)
    out = np.maximum(np.maximum(-p, p - 1), 0.0)
    return np.where(inside, d_faces, np.linalg.norm(out, axis=1))

gt = distance_to_cube(verts)
# Triangle centroids too, so the claim covers area and not only vertices.
gt_tri = distance_to_cube(verts[tris].mean(axis=1))

def observe(dense, sparse):
    pts = []
    for i, (o, u, v) in enumerate(_CUBE_FACES):
        k = sparse if i == len(_CUBE_FACES) - 1 else dense
        st = rng.random((k, 2))
        pts.append(np.asarray(o, dtype=float) + np.outer(st[:, 0], axes[u]) + np.outer(st[:, 1], axes[v]))
    return np.vstack(pts)

def measure(label, dense, sparse):
    pts = observe(dense, sparse)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    tree = o3d.geometry.KDTreeFlann(cloud)
    nn = [np.sqrt(tree.search_knn_vector_3d(p, 2)[2][1]) for p in pts]
    voxel = 2.0 * float(np.median(nn))
    before = _support(o3d, mesh, cloud, voxel)
    just = _crop_justification(before, 0, mesh)
    return {
        "case": label,
        "observations": int(len(pts)),
        "per_face_counts": {"five_faces_each": dense, "sixth_face": sparse},
        "median_sample_spacing": round(float(np.median(nn)), 6),
        "voxel_size": round(voxel, 6),
        "voxel_rule": "2x global median nearest-neighbour sample spacing (module recommendation)",
        "unsupported_area_fraction": round(before["unsupported_area_fraction"], 6),
        "unsupported_area_beyond_1_5_voxels": round(before["unsupported_area_beyond_1_5_voxels"], 6),
        "unsupported_area_beyond_3_voxels": round(before["unsupported_area_beyond_3_voxels"], 6),
        "far_band_share_of_unsupported_area": round(just["unsupported_area_share_beyond_3_voxels"], 6),
        "removed_surface_reads_as": just["removed_surface_reads_as"],
    }

record = {
  "what_this_measures":
    "Whether crop_justification's far reading can fire on a surface with zero error. It can, "
    "when the observations cover the surface unevenly, because the distance it reads is "
    "distance from the nearest observation and not distance from the truth.",
  "why_it_matters":
    "Earlier heads of this lane claimed the reading's error was one-sided and that an area it "
    "called invented was invented. That claim is false and is now removed from the source, the "
    "operator-facing note, the skill and the tests.",
  "found_by":
    "An independent root audit, which reproduced it on extracted functions with a SciPy "
    "cKDTree adapter. This record is the lane's own reproduction through the shipped "
    "functions and real Open3D, so neither rests on the other.",
  "geometry": {
    "shape": "closed triangulated unit cube",
    "vertices": int(len(verts)),
    "triangles": int(len(tris)),
    "is_its_own_ground_truth": True,
    "max_vertex_distance_to_analytic_cube": float(gt.max()),
    "max_triangle_centroid_distance_to_analytic_cube": float(gt_tri.max()),
    "invented_area_fraction": 0.0,
    "note":
      "Every vertex is constructed on a face of the analytic cube, and every observation is "
      "drawn on that same cube, so there is nothing anywhere for a fabrication reading to be "
      "right about.",
  },
  "results": [
    measure("even coverage of all six faces", 4000, 4000),
    measure("dense on five faces, sparse on the sixth", 4000, 40),
  ],
  "conclusion":
    "Evenness of coverage, not correctness of geometry, decides which side of the threshold "
    "this lands on. A high reading therefore cannot establish fabrication, and a low one "
    "cannot establish cleanliness. Both are statements about distance from observations.",
  "superseded_conclusions_kept_immutable": [
    "evidence/open3d/poisson-procedure-and-overcall-audit.json: twelve zero-error cells all "
    "read near-threshold. The measurements stand. The inference that the reading cannot "
    "over-call does not: every cell sampled convex geometry evenly.",
  ],
  "generator": "harness/irregular_density_counterexample.py",
  "reproduce":
    "docker run --rm --network none --entrypoint python3 <open3d image> "
    "harness/irregular_density_counterexample.py",
  "open3d_version": o3d.__version__,
  "numpy_seed": 7,
}
print(json.dumps(record, indent=2))

"""Recompute every published Open3D number from the delivered bytes.

Independent of the producing code path: this reads the artifacts back, measures
them again from scratch, and compares against what the run reported. It also runs
the held-out generalization test the evaluation plan froze, which the pipeline
itself does not perform: reconstruct from two fragments and measure how far the
third fragment's points fall from that surface.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

VOXEL = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support(mesh, cloud, voxel=VOXEL):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    tree = o3d.geometry.KDTreeFlann(cloud)
    dv = np.array([np.sqrt(tree.search_knn_vector_3d(v, 1)[2][0]) for v in verts])
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    far = dv[tris].max(axis=1) > voxel
    return {
        "unsupported_area_fraction": float(area[far].sum() / area.sum()),
        "max_vertex_distance_to_sample": float(dv.max()),
        "surface_area": float(area.sum()),
        "triangles": int(len(tris)),
        "vertices": int(len(verts)),
        "extent": [float(x) for x in mesh.get_axis_aligned_bounding_box().get_extent()],
    }


def coverage(mesh, cloud, voxel=VOXEL):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    pts = np.asarray(cloud.points).astype(np.float32)
    d = scene.compute_distance(o3d.core.Tensor(pts)).numpy()
    return {
        "samples": int(len(d)),
        "fraction_within_voxel": float((d < voxel).mean()),
        "fraction_within_two_voxels": float((d < 2 * voxel).mean()),
        "sample_to_surface_rmse": float(np.sqrt((d**2).mean())),
        "sample_to_surface_p95": float(np.percentile(d, 95)),
    }


def reconstruct(cloud, support_factor):
    work = o3d.geometry.PointCloud(cloud)
    work.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2, max_nn=30)
    )
    work.orient_normals_consistent_tangent_plane(k=30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        work, depth=9
    )
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.02))
    if support_factor > 0:
        tree = o3d.geometry.KDTreeFlann(work)
        dv = np.array(
            [
                np.sqrt(tree.search_knn_vector_3d(v, 1)[2][0])
                for v in np.asarray(mesh.vertices)
            ]
        )
        mesh.remove_vertices_by_mask(dv > VOXEL * support_factor)
    mesh.compute_vertex_normals()
    return mesh


def held_out(root: Path):
    """G5: does the surface predict a fragment it was never built from?"""
    graph = json.loads((root / "multiway/pose_graph.json").read_text())
    clouds = {}
    for node in graph["nodes"]:
        key = node["fragment_id"]
        pc = o3d.io.read_point_cloud(str(root / f"prepared/fragments/{key}.pcd"))
        pc.transform(np.asarray(node["pose"], dtype=float))
        clouds[key] = pc.voxel_down_sample(VOXEL)
    out = []
    for holdout in clouds:
        train = o3d.geometry.PointCloud()
        for key, pc in clouds.items():
            if key != holdout:
                train += pc
        train = train.voxel_down_sample(VOXEL)
        row = {"held_out_fragment": holdout, "train_points": len(train.points)}
        for label, factor in (("baseline", 0.0), ("candidate", 1.0)):
            mesh = reconstruct(train, factor)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            pts = np.asarray(clouds[holdout].points).astype(np.float32)
            d = scene.compute_distance(o3d.core.Tensor(pts)).numpy()
            row[label] = {
                "median_distance": float(np.median(d)),
                "p95_distance": float(np.percentile(d, 95)),
                "fraction_within_voxel": float((d < VOXEL).mean()),
                "triangles": int(len(mesh.triangles)),
            }
        row["median_delta"] = row["candidate"]["median_distance"] - row["baseline"][
            "median_distance"
        ]
        out.append(row)
    return out


def main() -> int:
    root = Path(sys.argv[1])
    fused = o3d.io.read_point_cloud(str(root / "multiway/fused.ply"))
    report = {
        "recomputed_from": "delivered artifact bytes",
        "hashes": {
            str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
        "fused": {
            "points": len(fused.points),
            "extent": [
                float(x)
                for x in fused.get_axis_aligned_bounding_box().get_extent()
            ],
        },
    }
    for label, rel in (
        ("candidate", "mesh/mesh.ply"),
        ("baseline", "mesh-baseline/mesh.ply"),
        ("candidate_uncropped", "mesh/mesh_uncropped.ply"),
    ):
        mesh = o3d.io.read_triangle_mesh(str(root / rel))
        mesh.compute_vertex_normals()
        block = {"artifact": rel, **support(mesh, fused), **coverage(mesh, fused)}
        ratio = max(
            m / c if c > 1e-9 else float("inf")
            for m, c in zip(block["extent"], report["fused"]["extent"])
        )
        block["mesh_to_cloud_bbox_worst_axis_ratio"] = ratio
        report[label] = block
    report["held_out"] = held_out(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure boundary topology and statistical-outlier distances on bound inputs.

This harness deliberately publishes no retained result in the repository. A prior
run used an unhashbound ephemeral bundle and a vertex-once boundary walker that was
not valid for bowtie topology. Its counts are withdrawn. A future run must retain
the exact input hashes and this harness hash alongside its output.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/holes_and_outliers.py
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d

BUNDLE = Path(os.environ.get("OPEN3D_BUNDLE", "/data/final-bundle"))
OUT = Path(os.environ.get("OPEN3D_OUT", "/data/holes-and-outliers.json"))
VARIANTS = {"baseline": "mesh-baseline/mesh.ply", "candidate": "mesh/mesh.ply"}
CLOUD = "multiway/fused.ply"
NB_NEIGHBORS = 20
STD_RATIO = 2.0
CONTROL_SAMPLES = 20000
CONTROL_SEED = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    """Return undirected edges used by exactly one triangle."""

    uses: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        a, b, c = (int(value) for value in triangle)
        uses.update((min(u, v), max(u, v)) for u, v in ((a, b), (b, c), (c, a)))
    return {edge for edge, count in uses.items() if count == 1}


def _find_simple_cycle(
    unused: set[tuple[int, int]], start_edge: tuple[int, int]
) -> tuple[list[int], set[tuple[int, int]]] | None:
    """Find one simple cycle containing ``start_edge`` without consuming edges."""

    start, second = start_edge
    path = [start, second]
    path_edges = {start_edge}

    def visit(current: int) -> tuple[list[int], set[tuple[int, int]]] | None:
        for edge in sorted(unused - path_edges):
            if current not in edge:
                continue
            following = edge[1] if edge[0] == current else edge[0]
            if following == start and len(path) >= 3:
                return list(path), {*path_edges, edge}
            if following in path:
                continue
            path.append(following)
            path_edges.add(edge)
            found = visit(following)
            if found is not None:
                return found
            path_edges.remove(edge)
            path.pop()
        return None

    return visit(second)


def closed_boundary_loops(triangles: np.ndarray) -> tuple[list[list[int]], dict]:
    """Decompose boundary edges into simple closed loops, using each edge once."""

    unused = boundary_edges(triangles)
    total = len(unused)
    degree: Counter[int] = Counter(vertex for edge in unused for vertex in edge)
    loops: list[list[int]] = []
    while unused:
        start_edge = min(unused)
        found = _find_simple_cycle(unused, start_edge)
        if found is None:
            raise ValueError("boundary contains an open fragment")
        loop, consumed = found
        unused -= consumed
        loops.append(loop)
    consumed_count = sum(len(loop) for loop in loops)
    if consumed_count != total:
        raise ValueError("boundary loop closure did not consume each edge exactly once")
    return loops, {
        "boundary_edges": total,
        "consumed_boundary_edges": consumed_count,
        "boundary_vertices_degree_not_two": sum(
            1 for value in degree.values() if value != 2
        ),
    }


def projected_loop_area(vertices: np.ndarray, loop: list[int]) -> dict[str, float]:
    """Return Newell projected area and a best-fit-plane nonplanarity measure."""

    points = vertices[loop].astype(float)
    centre = points.mean(axis=0)
    normal = np.zeros(3)
    for index, point in enumerate(points):
        normal += np.cross(point - centre, points[(index + 1) % len(points)] - centre)
    projected_area = float(0.5 * np.linalg.norm(normal))
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    plane_normal = axes[-1]
    nonplanarity = float(np.abs((points - centre) @ plane_normal).max())
    return {
        "projected_boundary_area_m2": round(projected_area, 6),
        "max_distance_from_best_fit_plane_m": round(nonplanarity, 6),
    }


def topology(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Describe boundary loops without deciding which loop is an interior hole."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles):
        raise ValueError("mesh is empty")
    loops, closure = closed_boundary_loops(triangles)
    return {
        **closure,
        "closed_boundary_loops": len(loops),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "loops": [
            {"vertices": len(loop), **projected_loop_area(vertices, loop)}
            for loop in loops
        ],
        "interpretation": (
            "Boundary loops are reported without classifying the outer patch edge as an "
            "interior hole. Projected boundary area is not hole surface area."
        ),
    }


def detector_validation() -> dict:
    """Exercise closed, one-loop, bowtie, and open-fragment behavior."""

    tetrahedron = np.array([(0, 1, 2), (0, 3, 1), (1, 3, 2), (0, 2, 3)], dtype=int)
    triangle = np.array([(0, 1, 2)], dtype=int)
    bowtie = np.array([(0, 1, 2), (0, 3, 4)], dtype=int)
    closed = closed_boundary_loops(tetrahedron)[0]
    one = closed_boundary_loops(triangle)[0]
    two, bowtie_closure = closed_boundary_loops(bowtie)
    try:
        _find_simple_cycle({(0, 1), (1, 2)}, (0, 1))
        open_rejected = _find_simple_cycle({(0, 1), (1, 2)}, (0, 1)) is None
    except ValueError:
        open_rejected = True
    valid = (
        len(closed) == 0
        and len(one) == 1
        and len(two) == 2
        and bowtie_closure["boundary_edges"]
        == bowtie_closure["consumed_boundary_edges"]
        and bowtie_closure["boundary_vertices_degree_not_two"] == 1
        and open_rejected
    )
    return {
        "closed_tetrahedron_loops": len(closed),
        "single_triangle_loops": len(one),
        "bowtie_loops": len(two),
        "bowtie_closure": bowtie_closure,
        "open_fragment_rejected": open_rejected,
        "detector_is_valid": valid,
    }


def outlier_distances(
    cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> dict:
    """Return descriptive distances; do not infer fitted/rejected causality."""

    points = np.asarray(cloud.points)
    if not len(points):
        raise ValueError("point cloud is empty")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def distances(sample: np.ndarray) -> np.ndarray:
        return scene.compute_distance(
            o3d.core.Tensor(sample.astype(np.float32))
        ).numpy()

    _, kept = cloud.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
    )
    retained = np.zeros(len(points), dtype=bool)
    retained[list(kept)] = True
    if not retained.any() or retained.all():
        raise ValueError("statistical split must contain retained and flagged points")
    rng = np.random.default_rng(CONTROL_SEED)
    control = rng.uniform(points.min(axis=0), points.max(axis=0), (CONTROL_SAMPLES, 3))
    return {
        "source_points": len(points),
        "flagged_points": int((~retained).sum()),
        "setting": {"nb_neighbors": NB_NEIGHBORS, "std_ratio": STD_RATIO},
        "median_distance_m": {
            "retained": round(float(np.median(distances(points[retained]))), 6),
            "flagged": round(float(np.median(distances(points[~retained]))), 6),
            "random_bbox_control": round(float(np.median(distances(control))), 6),
        },
        "interpretation": (
            "Descriptive distances only. They do not establish whether a flagged point "
            "caused, was fitted by, or was rejected by the reconstruction."
        ),
    }


def build() -> dict:
    """Build a hash-bound report or fail before publication."""

    paths = {"cloud": BUNDLE / CLOUD}
    paths.update({name: BUNDLE / relative for name, relative in VARIANTS.items()})
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} input is missing or empty: {path}")
    cloud = o3d.io.read_point_cloud(str(paths["cloud"]))
    meshes = {name: o3d.io.read_triangle_mesh(str(paths[name])) for name in VARIANTS}
    topologies = {name: topology(mesh) for name, mesh in meshes.items()}
    return {
        "schema": "npa.open3d.holes-and-outliers.measurement.v2",
        "status": "measured",
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
        "detector_validation": detector_validation(),
        "boundary_topology": topologies,
        "outlier_distances": outlier_distances(cloud, meshes["candidate"]),
        "claims": {
            "interior_hole_count": "not established",
            "interior_hole_surface_area": "not established",
            "outliers_fitted_or_rejected": "not established",
        },
    }


def publication(report: dict, out: Path) -> tuple[Path, bool]:
    """Publish only a controlled report with nonempty SHA-256 input identities."""

    valid = report.get("detector_validation", {}).get("detector_is_valid") is True
    inputs = report.get("inputs")
    bound = (
        isinstance(inputs, dict)
        and inputs
        and all(
            isinstance(value, dict)
            and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
            for value in inputs.values()
        )
    )
    if valid and bound:
        return out, True
    return out.with_suffix(".invalid.json"), False


def main() -> int:
    try:
        report = build()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Measurement failed closed: {exc}")
        return 1
    destination, published = publication(report, OUT)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not published:
        print(f"Measurement controls failed; quarantined at {destination}")
        return 1
    print(f"Wrote hash-bound descriptive measurement to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

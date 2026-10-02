"""Index exact source triangles and unsafe topology for local foot support queries."""

import hashlib

import numpy as np


def _topology(points, triangles):
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    pairs = starts[counts == 2]
    first, second = order[pairs], order[pairs + 1]
    opposite = np.all(edges[first] == edges[second, ::-1], axis=1)
    opposite &= first // 3 != second // 3
    first, second = first[opposite], second[opposite]
    neighbors = np.full(len(edges), -1, dtype=np.int32)
    neighbors[first], neighbors[second] = second // 3, first // 3
    face_points = points.astype(np.float64)[triangles]
    normals = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths > 0, lengths, 1)[:, None]
    if not np.isfinite(normals).all():
        raise ValueError("Derived source normals must be finite")
    return neighbors.reshape(-1, 3), normals.astype(np.float32)


def _validate_arrays(points, triangles):
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Support vertices must have nonempty shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Support faces must have nonempty shape (N, 3)")
    if triangles.dtype.kind not in "iu" or (triangles < 0).any():
        raise ValueError("Support face indices must be nonnegative integers")
    if (triangles >= len(points)).any() or not np.isfinite(points).all():
        raise ValueError("Support source triangles must be finite and in range")
    if not np.isfinite(points.astype(np.float32)).all():
        raise ValueError("Support vertices must remain finite in native precision")


def _world_mesh(sensor, stage):
    from pxr import Usd, UsdGeom

    if len(sensor.cfg.mesh_prim_paths) != 1:
        raise ValueError("Support queries require exactly one static source mesh")
    path = sensor.cfg.mesh_prim_paths[0]
    mesh = sensor.meshes[(path, sensor.device)]
    authored = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    counts = np.asarray(authored.GetFaceVertexCountsAttr().Get())
    if not len(counts) or not np.all(counts == 3):
        raise ValueError("Support queries require authored triangular faces")
    points = np.asarray(authored.GetPointsAttr().Get())
    transform = np.asarray(
        UsdGeom.Xformable(authored).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T
    expected = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    indices = np.asarray(authored.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    actual = mesh.points.numpy()
    if not np.array_equal(actual, expected) or not np.array_equal(
        mesh.indices.numpy(), indices
    ):
        raise ValueError(
            "Native static ray mesh differs from the authored world geometry"
        )
    return mesh, actual, indices.reshape(-1, 3)


class SurfaceMesh:
    """Hold immutable world triangles and edge adjacency for support classification.

    Args:
        mesh: Existing native Warp mesh, also used by the static ray sensors.
        points: Exact world-space mesh vertices.
        triangles: Exact source face vertex indices.
    Returns:
        A source-bound topology index without altering collision geometry.
    Raises:
        ValueError: Geometry is nonfinite or malformed.
    """

    def __init__(self, mesh, points, triangles):
        import warp as wp

        _validate_arrays(points, triangles)
        neighbors, normals = _topology(points, triangles)
        self.mesh = mesh
        self.neighbors = wp.array(neighbors, dtype=wp.int32, device=mesh.device)
        self.normals = wp.array(normals, dtype=wp.vec3, device=mesh.device)
        # Query kernels use PyTorch's stream; finish these one-time topology uploads.
        if mesh.device.is_cuda:
            wp.synchronize_device(mesh.device)
        digest = hashlib.sha256(points.astype(np.float32).tobytes())
        digest.update(triangles.astype(np.int32).tobytes())
        self.metadata = {
            "world_mesh_sha256": digest.hexdigest(),
            "vertices": len(points),
            "triangles": len(triangles),
            "unsafe_directed_edges": int((neighbors < 0).sum()),
            "neighborhood_capacity": 64,
            "capacity_scope": "Exceeding local query capacity preserves obstacle classification",
        }

    @classmethod
    def from_sensor(cls, sensor, stage):
        """Bind the sensor's world mesh to the current authored source geometry.

        Args:
            sensor: Initialized native static ray sensor.
            stage: Current native USD stage.
        Returns:
            Exact source topology for the existing sensor mesh.
        Raises:
            ValueError: Cached mesh bytes differ from the current world geometry.
        """
        return cls(*_world_mesh(sensor, stage))

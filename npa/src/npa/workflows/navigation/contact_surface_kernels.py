"""Classify geometrically unambiguous local support without changing native contacts."""

import warp as wp

_Faces = wp.types.vector(length=64, dtype=wp.int32)


@wp.func
def _segment_point(point: wp.vec3, first: wp.vec3, second: wp.vec3):
    edge = second - first
    length = wp.dot(edge, edge)
    if length == 0.0:
        return first
    fraction = wp.clamp(wp.dot(point - first, edge) / length, 0.0, 1.0)
    return first + fraction * edge


@wp.func
def _triangle_point(point: wp.vec3, first: wp.vec3, second: wp.vec3, third: wp.vec3):
    edge0, edge1 = second - first, third - first
    offset = point - first
    aa, ab, bb = wp.dot(edge0, edge0), wp.dot(edge0, edge1), wp.dot(edge1, edge1)
    determinant = aa * bb - ab * ab
    if determinant > 0.0:
        ad, bd = wp.dot(edge0, offset), wp.dot(edge1, offset)
        u, v = (bb * ad - ab * bd) / determinant, (aa * bd - ab * ad) / determinant
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            return first + u * edge0 + v * edge1
    closest = _segment_point(point, first, second)
    candidate = _segment_point(point, second, third)
    if wp.length_sq(point - candidate) < wp.length_sq(point - closest):
        closest = candidate
    candidate = _segment_point(point, third, first)
    if wp.length_sq(point - candidate) < wp.length_sq(point - closest):
        closest = candidate
    return closest


@wp.func
def _face_point(mesh: wp.uint64, face: int, point: wp.vec3):
    # Warp mesh_get_point already resolves the face-vertex index through indices.
    first = wp.mesh_get_point(mesh, face * 3)
    second = wp.mesh_get_point(mesh, face * 3 + 1)
    third = wp.mesh_get_point(mesh, face * 3 + 2)
    return _triangle_point(point, first, second, third)


@wp.func
def _edge_near(mesh: wp.uint64, face: int, edge: int, point: wp.vec3, radius: float):
    first = wp.mesh_get_point(mesh, face * 3 + edge)
    second = wp.mesh_get_point(mesh, face * 3 + (edge + 1) % 3)
    return wp.length(point - _segment_point(point, first, second)) <= radius


@wp.func
def _local_index(faces: _Faces, count: int, face: int):
    for index in range(count):
        if faces[index] == face:
            return index
    return -1


@wp.func
def _connected(
    mesh: wp.uint64,
    faces: _Faces,
    count: int,
    seed: int,
    neighbors: wp.array2d(dtype=wp.int32),
    point: wp.vec3,
    radius: float,
):
    first = _local_index(faces, count, seed)
    if first < 0:
        return False
    reached = _Faces(0)
    queue = _Faces(-1)
    reached[first] = 1
    queue[0] = first
    scanned, total = int(0), int(1)
    while scanned < total:
        index = queue[scanned]
        scanned += 1
        for edge in range(3):
            other = _local_index(faces, count, neighbors[faces[index], edge])
            if other < 0:
                continue
            if reached[other] == 1:
                continue
            if _edge_near(mesh, faces[index], edge, point, radius):
                reached[other] = 1
                queue[total] = other
                total += 1
    return total == count


@wp.func
def _neighborhood(
    mesh: wp.uint64,
    point: wp.vec3,
    radius: float,
    seed: int,
    normals: wp.array(dtype=wp.vec3),
    neighbors: wp.array2d(dtype=wp.int32),
):
    query = wp.mesh_query_aabb(mesh, point - wp.vec3(radius), point + wp.vec3(radius))
    face, count = int(0), int(0)
    faces = _Faces(-1)
    while wp.mesh_query_aabb_next(query, face):
        if wp.length(point - _face_point(mesh, face, point)) > radius:
            continue
        if normals[face][2] < 0.7:
            return 6
        if wp.dot(normals[seed], normals[face]) < 0.7:
            return 9
        for edge in range(3):
            if neighbors[face, edge] < 0 and _edge_near(
                mesh, face, edge, point, radius
            ):
                return 8
        if count == 64:
            return 10
        faces[count] = face
        count += 1
    if not _connected(mesh, faces, count, seed, neighbors, point, radius):
        return 11
    return 1


@wp.kernel
def _classify(
    mesh: wp.uint64,
    normals: wp.array(dtype=wp.vec3),
    neighbors: wp.array2d(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    separations: wp.array(dtype=float),
    roots: wp.array(dtype=wp.vec3),
    reasons: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    closest: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    index = wp.tid()
    point, radius = points[index], radii[index]
    reasons[index] = 3
    if radius <= 0.0:
        return
    reasons[index] = 5
    query = wp.mesh_query_point_no_sign(mesh, point, radius)
    if not query.result:
        return
    projected = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    faces[index] = query.face
    closest[index] = projected
    distances[index] = wp.length(point - projected)
    if distances[index] > radius:
        return
    reasons[index] = 7
    if roots[index][2] <= projected[2]:
        return
    reasons[index] = _neighborhood(mesh, point, radius, query.face, normals, neighbors)

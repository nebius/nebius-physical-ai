"""Recognize only geometrically unambiguous native foot contacts as floor support."""

_FEET = {"LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"}
_REASONS = {
    1: "recognized_floor",
    2: "not_an_oblique_foot_candidate",
    3: "unavailable_single_shape_margin",
    4: "separation_exceeds_native_margin",
    5: "no_source_face_within_margin",
    6: "nearby_non_upward_or_degenerate_face",
    7: "root_not_above_source_surface",
    8: "nearby_open_nonmanifold_or_winding_edge",
    9: "conflicting_surface_normals",
    10: "local_face_capacity_exceeded",
    11: "disconnected_local_surfaces",
}


def _tensor(value):
    from npa.workflows.navigation.reference_contacts import _tensor as convert

    return convert(value)


def _native_foot_view(env):
    from isaaclab_physx.physics import PhysxManager

    expression = "(" + "|".join(sorted(_FEET)) + ")"
    view = PhysxManager.get_physics_sim_view().create_rigid_body_view(
        f"/World/envs/env_*/Robot/{expression}"
    )
    if view.count != env.num_envs * len(_FEET):
        raise ValueError("Native support view does not identify every reference foot")
    return view


def _resolved_margins(view):
    import torch

    if view.max_shapes != 1:
        return torch.zeros(view.count), torch.zeros(view.count)
    contact = _tensor(view.get_contact_offsets()).cpu().reshape(-1).clone()
    rest = _tensor(view.get_rest_offsets()).cpu().reshape(-1).clone()
    if not torch.isfinite(contact).all() or not torch.isfinite(rest).all():
        raise ValueError("Resolved native foot offsets must be finite")
    if not ((contact > 0) & (rest < contact)).all():
        raise ValueError(
            "Resolved native foot offsets violate the shape margin contract"
        )
    return contact, rest


def _sensor_tables(env, measurements, view):
    from npa.workflows.navigation.contact_evidence import _robot_index

    paths = list(view.prim_paths)
    if len(set(paths)) != len(paths) or len(paths) != env.num_envs * len(_FEET):
        raise ValueError("Native foot paths are ambiguous")
    feet = {path: index for index, path in enumerate(paths)}
    roots = _root_indices(env)
    contact, rest = _resolved_margins(view)
    rows, offsets, rests, root_indices = [], [], [], []
    for sensor, path in enumerate(measurements.view.sensor_paths):
        robot, body = _robot_index(path), path.rsplit("/", 1)[1]
        if (
            robot != sensor // measurements.body_count
            or body != measurements.body_names[sensor % measurements.body_count]
        ):
            raise ValueError(
                "Native contact sensor ordering differs from classification"
            )
        foot = feet.get(path) if body in _FEET else None
        if body in _FEET and foot is None:
            raise ValueError("Native foot view does not cover the contact sensor")
        offsets.append(float(contact[foot]) if foot is not None else 0.0)
        rests.append(float(rest[foot]) if foot is not None else 0.0)
        root_indices.append(roots[robot])
        if foot is not None:
            rows.append(
                {
                    "sensor_index": sensor,
                    "native_path": path,
                    "contact_offset_m": offsets[-1],
                    "rest_offset_m": rests[-1],
                }
            )
    return offsets, rests, root_indices, rows


def _root_indices(env):
    from npa.workflows.navigation.contact_evidence import _robot_index

    paths = list(env.scene["robot"].root_view.prim_paths)
    roots = {_robot_index(path): index for index, path in enumerate(paths)}
    if len(paths) != env.num_envs or len(roots) != len(paths):
        raise ValueError("Native root paths must identify each robot exactly once")
    if set(roots) != set(range(env.num_envs)) or any(
        path != f"/World/envs/env_{robot}/Robot"
        for robot, index in roots.items()
        for path in [paths[index]]
    ):
        raise ValueError("Native root paths do not cover the reference population")
    return roots


def _query(surface, points, radii, separation, roots):
    import torch

    count, device = len(points), points.device
    reason = torch.zeros(count, dtype=torch.int32, device=device)
    face = torch.full((count,), -1, dtype=torch.int32, device=device)
    closest = torch.zeros_like(points)
    distance = torch.full((count,), -1.0, device=device)
    _launch(surface, points, radii, separation, roots, reason, face, closest, distance)
    return {
        "support_reason": reason,
        "source_face_index": face,
        "source_point_world_m": closest,
        "source_distance_m": distance,
        "support_radius_m": radii,
        "native_separation_m": separation,
    }


def _launch(surface, points, radii, separation, roots, reason, face, closest, distance):
    import torch
    import warp as wp
    from npa.workflows.navigation.contact_surface_kernels import _classify

    arguments = [
        surface.mesh.id,
        surface.normals,
        surface.neighbors,
        wp.from_torch(points, dtype=wp.vec3),
        wp.from_torch(radii),
        wp.from_torch(separation),
        wp.from_torch(roots, dtype=wp.vec3),
        wp.from_torch(reason),
        wp.from_torch(face),
        wp.from_torch(closest, dtype=wp.vec3),
        wp.from_torch(distance),
    ]
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(points.device))
        if points.device.type == "cuda"
        else None
    )
    wp.launch(
        _classify,
        dim=len(points),
        inputs=arguments,
        device=surface.mesh.device,
        stream=stream,
    )


class FootSupport:
    """Apply source geometry and resolved native margins to oblique foot contacts.

    Args:
        env: Initialized native reference environment.
        measurements: Existing contact view with original classification metadata.
    Returns:
        A read-only support query that never modifies shapes or physics settings.
    Raises:
        ValueError: Native identities, margins, or source mesh identity are invalid.
    """

    def __init__(self, env, measurements):
        import torch
        from npa.workflows.navigation.contact_surface_geometry import SurfaceMesh

        self.env, self.measurements = env, measurements
        self.view = _native_foot_view(env)
        margins, rests, roots, rows = _sensor_tables(env, measurements, self.view)
        self.radii = torch.tensor(margins, dtype=torch.float32, device=env.device)
        self.rests = torch.tensor(rests, dtype=torch.float32, device=env.device)
        self.roots = torch.tensor(roots, dtype=torch.long, device=env.device)
        self.feet = torch.tensor(
            [
                path.rsplit("/", 1)[1] in _FEET
                for path in measurements.view.sensor_paths
            ],
            dtype=torch.bool,
            device=env.device,
        )
        self.surface = SurfaceMesh.from_sensor(
            env.scene["navigation_ranges"], env.sim.stage
        )
        self.metadata = {
            **self.surface.metadata,
            "foot_max_shapes": self.view.max_shapes,
            "native_foot_offsets": rows,
            "native_foot_offset_status": (
                "resolved_single_shape"
                if self.view.max_shapes == 1
                else "unavailable_multi_shape_zero_sentinel"
            ),
            "reason_codes": _REASONS,
            "radius_scope": "Resolved single-shape FOOT contact offset only; no inferred terrain margin or fitted distance",
            "slope_threshold": 0.7,
        }

    def recognize(self, data, pairs, indices, original, retain=False):
        """Recognize a bounded, connected upward source patch around actual contacts.

        Args:
            data: Unmodified native six-array contact tuple.
            pairs: Sensor/filter pair index for each contact.
            indices: Exact native contact buffer indices.
            original: Original steep-normal or base obstacle mask.
            retain: Retain per-candidate geometry for probe evidence.
        Returns:
            Support mask and optional copied-by-recorder geometry fields.
        Raises:
            ValueError: Native contact inputs are nonfinite.
        """
        import torch

        sensors = pairs // self.measurements.view.filter_count
        candidates = torch.nonzero(original & self.feet[sensors]).flatten()
        supported = torch.zeros_like(original)
        if not len(candidates):
            return supported, None
        inputs = self._inputs(data, indices[candidates], sensors[candidates])
        fields = _query(self.surface, *inputs)
        supported[candidates] = fields["support_reason"] == 1
        if not retain:
            return supported, None
        fields["support_rest_offset_m"] = self.rests[sensors[candidates]]
        return supported, _expanded(fields, candidates, len(indices))

    def _inputs(self, data, indices, sensors):
        import torch

        points = _tensor(data[1])[indices].contiguous()
        separation = _tensor(data[3]).flatten()[indices].contiguous()
        roots = _tensor(self.env.scene["robot"].root_view.get_root_transforms())
        roots = roots[self.roots[sensors], :3].contiguous()
        if not all(
            torch.isfinite(value).all() for value in (points, separation, roots)
        ):
            raise ValueError("Native foot support inputs are nonfinite")
        return points, self.radii[sensors].contiguous(), separation, roots


def _expanded(fields, candidates, count):
    import torch

    result = {}
    for name, values in fields.items():
        default = 2 if name == "support_reason" else -1
        result[name] = torch.full(
            (count, *values.shape[1:]),
            default,
            dtype=values.dtype,
            device=values.device,
        )
        result[name][candidates] = values
    return result

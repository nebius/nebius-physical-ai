"""Measure static obstacle impacts and unexpected contacts from real PhysX tensors."""


def _tensor(value):
    import warp as wp

    return wp.to_torch(value) if isinstance(value, wp.array) else value


def _scene_filters(stage, root):
    from pxr import Sdf, Usd, UsdPhysics

    paths = [
        str(prim.GetPath())
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.GetPath().HasPrefix(Sdf.Path(root))
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    ]
    if not paths:
        raise RuntimeError("shared scene has no enabled collision filters")
    # PhysX broadcasts each single-prim filter to every sensor. A wildcard
    # matching several scene prims instead requires one match per sensor.
    return sorted(paths)


class ContactMeasurements:
    """Accumulate physical contact evidence over each high-level control interval.

    Args:
        env: Native PhysX environment with one static shared scene.
    Returns:
        Contact views and per-robot accumulated force buffers.
    Raises:
        RuntimeError: PhysX contact views are unavailable or malformed.
    """

    def __init__(self, env):
        import torch
        from isaaclab_physx.physics import PhysxManager

        self.env = env
        self.evidence = None
        self.obstacle = torch.zeros(env.num_envs, device=env.device)
        self.peer = torch.zeros_like(self.obstacle)
        physics = PhysxManager.get_physics_sim_view()
        bodies = env.scene["contact_forces"].body_names
        self.body_names = list(bodies)
        self.body_count = len(bodies)
        self.base_index = bodies.index("base")
        names = "(" + "|".join(bodies) + ")"
        filters = _scene_filters(env.sim.stage, env.cfg.npa_scene_prim)
        self.filter_paths = filters
        self.view = physics.create_rigid_contact_view(
            f"/World/envs/env_*/Robot/{names}",
            filter_patterns=filters,
            max_contact_data_count=64 * len(bodies) * env.num_envs,
        )
        if self.view.sensor_count != env.num_envs * len(
            bodies
        ) or self.view.filter_count != len(filters):
            raise RuntimeError(
                "reference contact view does not cover every robot and scene"
            )

    def reset(self, env_ids=None):
        """Clear accumulated physical measurements for the requested robots.

        Args:
            env_ids: Native indices or None for every robot.
        Returns:
            None.
        Raises:
            None.
        """
        indices = slice(None) if env_ids is None else env_ids
        self.obstacle[indices] = 0
        self.peer[indices] = 0

    def update(self):
        """Accumulate actual obstacle normals and unexplained external contact forces.

        Args:
            None.
        Returns:
            None.
        Raises:
            RuntimeError: Native contact data contains nonfinite values.
        """
        import torch

        static = _tensor(self.view.get_contact_force_matrix(dt=self.env.physics_dt))
        total = _tensor(self.view.get_net_contact_forces(dt=self.env.physics_dt))
        residual = torch.linalg.vector_norm(total - static.sum(dim=1), dim=-1)
        residual = residual.reshape(self.env.num_envs, self.body_count).amax(dim=1)
        residual = torch.where(residual > 0.02, residual, 0.0)
        self.peer[:] = torch.maximum(self.peer, residual)
        self.obstacle[:] = torch.maximum(self.obstacle, self._obstacle_forces())
        if (
            not torch.isfinite(self.obstacle).all()
            or not torch.isfinite(self.peer).all()
        ):
            raise RuntimeError("PhysX returned nonfinite contact evidence")

    def _obstacle_forces(self):
        import torch

        view = self.view
        data = view.get_contact_data(dt=self.env.physics_dt)
        force, _, normals, _, counts, starts = data
        counts = _tensor(counts).long().flatten()
        starts = _tensor(starts).long().flatten()
        pairs = torch.repeat_interleave(
            torch.arange(len(counts), device=counts.device), counts
        )
        offsets = torch.cumsum(counts, 0) - counts
        local = torch.arange(
            len(pairs), device=counts.device
        ) - torch.repeat_interleave(offsets, counts)
        indices = torch.repeat_interleave(starts, counts) + local
        magnitudes = _tensor(force).flatten()[indices].abs()
        body_ids = (pairs // view.filter_count) % self.body_count
        obstacle = (_tensor(normals)[indices, 2].abs() < 0.7) | (
            body_ids == self.base_index
        )
        magnitudes = magnitudes * obstacle
        result = torch.zeros_like(self.obstacle)
        result.scatter_add_(
            0, pairs // (view.filter_count * self.body_count), magnitudes
        )
        if self.evidence is not None:
            self.evidence.sample(data, pairs, indices, obstacle, result)
        return torch.where(result > 0.02, result, 0.0)

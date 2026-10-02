"""Measure physical validity on explicitly flat, supported reference routes."""


def footprint_pattern(cfg, device):
    """Cast downward at the center and four corners of the reference footprint.

    Args:
        cfg: Native pattern configuration.
        device: Native tensor device.
    Returns:
        Five ray origins and downward directions in the sensor frame.
    Raises:
        RuntimeError: Native tensor allocation fails.
    """
    import torch

    origins = torch.tensor(
        [
            [0, 0, 0],
            [-0.35, -0.23, 0],
            [-0.35, 0.23, 0],
            [0.35, -0.23, 0],
            [0.35, 0.23, 0],
        ],
        device=device,
    )
    directions = torch.zeros_like(origins)
    directions[:, 2] = -1
    return origins, directions


def physical_state(env):
    """Read real floor intersections, root clearance and upright orientation.

    Args:
        env: Native reference environment with downward static-scene rays.
    Returns:
        Measured failure, upright cosine, clearance and support diagnostics.
    Raises:
        RuntimeError: Native root orientation is nonfinite.
    """
    robot = env.scene["robot"]
    hits = env.scene["ground_support"].data.ray_hits_w.torch
    clearance = robot.data.root_pos_w.torch[:, 2, None] - hits[:, :, 2]
    upright = -robot.data.projected_gravity_b.torch[:, 2]
    return validity_from_measurements(clearance, upright)


def validity_from_measurements(clearance, upright):
    """Apply the public flat-route support envelope to actual measured tensors.

    Args:
        clearance: Five downward floor distances per robot; misses are nonfinite.
        upright: Root vertical alignment measured from projected gravity.
    Returns:
        Finite auditable measurements; missing support is an explicit failure.
    Raises:
        RuntimeError: Orientation or sensor shapes are invalid.
    """
    import torch

    if clearance.shape != (len(upright), 5) or not torch.isfinite(upright).all():
        raise RuntimeError("invalid native physical-validity measurements")
    supported = torch.isfinite(clearance)
    distances = torch.where(supported, clearance, 1.2)
    low, high = distances.amin(dim=1), distances.amax(dim=1)
    valid = supported.all(dim=1) & (low >= 0.25) & (high <= 0.85)
    valid &= (high - low <= 0.15) & (upright >= 0.5)
    return {
        "physical_failure": ~valid,
        "upright_cosine": upright,
        "ground_clearance_m": high,
        "minimum_ground_clearance_m": low,
        "ground_support_fraction": supported.float().mean(dim=1),
    }

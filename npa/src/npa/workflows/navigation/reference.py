"""Adapt the public native quadruped task to sealed shared-scene navigation cases."""

from contextlib import contextmanager

TASK = "NPA-Shared-Scene-Navigation-v0"


def configure(*, task, scene_file, scene_prim, num_envs, cases, training):
    """Configure real Isaac physics, public locomotion and obstacle-aware PPO.

    Args:
        task: Public reference task identifier.
        scene_file: Local sealed metric Z-up USDZ containing static collision meshes.
        scene_prim: One global scene reference path.
        num_envs: Number of simultaneous robots sharing that scene.
        cases: Explicit training or evaluation reset cases, never both.
        training: Enable training episode resets when true.
    Returns:
        Native Isaac Lab environment configuration.
    Raises:
        ValueError: Task identity or scene geometry is unsupported.
        ImportError: The pinned native runtime is unavailable.
    """
    from npa.workflows.navigation.reference_config import build_config

    if task != TASK:
        raise ValueError(f"public reference requires task {TASK}")
    return build_config(scene_file, scene_prim, num_envs, cases, training)


def reset(env, cases):
    """Reset native articulation, controller, contact and observation histories.

    Args:
        env: Real Isaac environment.
        cases: One explicit world-frame reset per robot.
    Returns:
        None.
    Raises:
        ValueError: The reset count differs from the robot count.
    """
    native = env.unwrapped
    if len(cases) != native.num_envs:
        raise ValueError("reference reset requires one case per robot")
    native.npa_override_cases = cases
    native.reset(seed=cases[0]["seed"])
    native.npa_override_cases = None


def measure(env):
    """Read actual native poses, fixed goals and accumulated PhysX contact forces.

    Args:
        env: Real Isaac environment.
    Returns:
        NumPy snapshots in the navigation measurement contract.
    Raises:
        RuntimeError: Native contact instrumentation is not initialized.
    """
    native = env.unwrapped
    from npa.workflows.navigation.reference_validity import physical_state

    robot = native.scene["robot"]
    command = native.command_manager.get_term("pose_command")
    values = {
        "position_m": robot.data.root_pos_w.torch,
        "heading_rad": robot.data.heading_w.torch,
        "goal_m": command.pos_command_w[:, :2],
        "obstacle_contact": native.npa_contacts.obstacle,
        "peer_contact": native.npa_contacts.peer,
        **physical_state(native),
    }
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


@contextmanager
def probe_mode(env):
    """Disable auto-resets while measuring the physical isolation controls.

    Args:
        env: Native reference environment.
    Returns:
        Context restoring the original termination behavior on exit.
    Raises:
        None.
    """
    native = env.unwrapped
    original = native.npa_probe
    native.npa_probe = True
    try:
        yield
    finally:
        native.npa_probe = original

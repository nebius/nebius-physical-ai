"""Initialize fresh Cartesian Franka actors at their measured native ready pose."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.isaac_franka_policy import SCENARIO_TASK_ID


def _tensor(value: Any) -> Any:
    return getattr(value, "torch", value)


def _root_tcp_pose(task: Any) -> Any:
    import torch
    from isaaclab.utils.math import subtract_frame_transforms

    robot_root = _tensor(task.scene["robot"].data.root_pose_w)
    tcp = task.scene["ee_frame"].data
    position, orientation = subtract_frame_transforms(
        robot_root[:, :3],
        robot_root[:, 3:7],
        _tensor(tcp.target_pos_w)[:, 0],
        _tensor(tcp.target_quat_w)[:, 0],
    )
    return torch.cat((position, orientation), dim=-1)


def _canonical_ready_action(pose: Any, scale: Any) -> tuple[Any, float]:
    import torch

    pose = pose.detach().clone()
    if pose.ndim != 2 or pose.shape[0] < 1 or pose.shape[1] != 7:
        raise ValueError("fresh Franka actor requires measured batched TCP poses")
    if not torch.isfinite(pose).all():
        raise ValueError("fresh Franka ready pose must be finite")
    lengths = torch.linalg.vector_norm(pose[:, 3:7], dim=-1, keepdim=True)
    if (lengths < 1e-6).any():
        raise ValueError("fresh Franka ready orientation must be nonzero")
    pose[:, 3:7] /= lengths
    signs = (pose[:, 3:7] * pose[0, 3:7]).sum(dim=-1, keepdim=True)
    pose[:, 3:7] *= torch.where(signs < 0, -1.0, 1.0)
    spread = torch.linalg.vector_norm(pose[:, :3] - pose[0, :3], dim=-1).max()
    if spread > 1e-4 or not torch.allclose(
        pose[:, 3:7], pose[0, 3:7].expand_as(pose[:, 3:7]), atol=1e-4, rtol=0
    ):
        raise ValueError("fresh Franka actors require a common robot-frame ready pose")
    scale = torch.as_tensor(scale, device=pose.device, dtype=pose.dtype)
    if scale.shape[-1:] != (7,) or not torch.isfinite(scale).all():
        raise ValueError("fresh Franka action scales must have seven finite values")
    if (scale <= 0).any():
        raise ValueError("fresh Franka action scales must be positive")
    actions = torch.cat((pose / scale, torch.ones_like(pose[:, :1])), dim=-1)
    if actions.shape != (len(pose), 8) or not torch.isfinite(actions).all():
        raise ValueError("fresh Franka ready actions must be finite eight-vectors")
    if not torch.allclose(actions, actions[0].expand_as(actions), atol=1e-3, rtol=0):
        raise ValueError("fresh Franka action scaling differs across ready poses")
    return actions[0], float(spread)


def _fresh_actor_output(actor: Any, *, checkpoint_loaded: bool) -> Any:
    import torch

    if checkpoint_loaded is not False:
        raise RuntimeError(
            "ready-pose initialization cannot replace a loaded checkpoint"
        )
    if getattr(actor, "_npa_ready_pose_initialized", False):
        raise RuntimeError("fresh Franka actor was already initialized")
    output = actor.mlp[-1]
    if not actor.is_recurrent or not isinstance(output, torch.nn.Linear):
        raise RuntimeError(
            "ready-pose initialization requires the native recurrent actor"
        )
    if output.out_features != 8 or output.bias is None:
        raise RuntimeError("ready-pose initialization requires eight affine actions")
    return output


def initialize_fresh_stock_franka_actor(
    actor: Any, task: Any, *, checkpoint_loaded: bool
) -> dict[str, Any]:
    """Start a fresh learned actor at its measured pose without overriding inference.

    Args:
        actor: Newly constructed native recurrent actor, before any training.
        task: Reset stock-Franka scenario environment with native TCP measurements.
        checkpoint_loaded: Whether any trained checkpoint has been loaded.
    Returns:
        Measured initialization evidence; this does not establish task competence.
    Raises:
        ValueError: The task, measured poses, or native action scales differ.
        RuntimeError: The actor was loaded, initialized, or has incompatible outputs.
    """
    import torch

    if task.spec.id != SCENARIO_TASK_ID:
        raise ValueError("ready-pose initialization requires the stock scenario task")
    output = _fresh_actor_output(actor, checkpoint_loaded=checkpoint_loaded)
    scale = _tensor(task.action_manager.get_term("arm_action")._scale)
    action, spread = _canonical_ready_action(_root_tcp_pose(task), scale)
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(action.to(device=output.bias.device, dtype=output.bias.dtype))
    actor._npa_ready_pose_initialized = True
    return {
        "profile": "fresh-franka-measured-ready-pose-v1",
        "checkpoint_loaded": False,
        "inference_composition": "learned_actor_only",
        "task_competence_established": False,
        "initial_action": action.tolist(),
        "maximum_ready_pose_position_spread_m": spread,
    }

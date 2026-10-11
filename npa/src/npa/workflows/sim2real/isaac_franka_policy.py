"""Share the native stock-Franka Cartesian recurrent policy across train and inference."""

from __future__ import annotations

from typing import Any

SCENARIO_TASK_ID = "NPA-Lift-Cube-Franka-Scenarios-v0"
BOOTSTRAP_PROFILE = "franka-cartesian-gru-distillation-v2"


def stock_franka_policy_contract() -> dict[str, Any]:
    """Describe the action semantics and observed state of the stock scenario policy.

    Args:
        None.
    Returns:
        The shared native action, observation, and recurrent-model contract.
    Raises:
        None.
    """
    return {
        "profile": "stock-franka-cartesian-gru-v2",
        "class_name": "RNNModel",
        "rnn_type": "gru",
        "rnn_hidden_dim": 128,
        "rnn_num_layers": 1,
        "observation_normalization": True,
        "observation_dim": 39,
        "action_dim": 8,
        "native_ik_mode": "absolute_pose",
        "native_action_scale": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0],
        "native_episode_clock_max_s": 5.0,
        "expert_phase_is_observation": False,
    }


def _tensor(value: Any) -> Any:
    return getattr(value, "torch", value)


def reset_stock_franka_actor(actor: Any, dones: Any = None) -> None:
    """Reset native recurrent state in the context that created inference tensors.

    Args:
        actor: The native recurrent actor used by stock-Franka inference.
        dones: Completed-environment mask, or None to clear all hidden state.
    Returns:
        None; only the native actor's recurrent state is reset.
    Raises:
        RuntimeError: The native actor rejects its hidden state or reset mask.
        AttributeError: The actor lacks the native reset API.
    """
    import torch

    with torch.inference_mode():
        actor.reset(dones)


def _episode_clock(task: Any) -> Any:
    import torch

    elapsed = _tensor(task.episode_length_buf).to(dtype=torch.float32)
    return (elapsed.unsqueeze(-1) * float(task.step_dt)).clamp(max=5.0)


def _tcp_geometry(task: Any) -> Any:
    import torch
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    root = _tensor(task.scene["robot"].data.root_pose_w)
    sensor = task.scene["ee_frame"]
    position = _tensor(sensor.data.target_pos_w)[:, 0]
    orientation = _tensor(sensor.data.target_quat_w)[:, 0]
    tcp_root, orientation_root = subtract_frame_transforms(
        root[:, :3], root[:, 3:7], position, orientation
    )
    object_position = _tensor(task.scene["object"].data.root_pos_w)
    object_root = quat_apply_inverse(root[:, 3:7], object_position - root[:, :3])
    goal = task.command_manager.get_command("object_pose")[:, :3]
    return torch.cat(
        (object_root - tcp_root, goal - tcp_root, orientation_root), dim=-1
    )


def configure_stock_franka_policy(env_cfg: Any) -> None:
    """Configure native Cartesian actions and physical observations before creation.

    Args:
        env_cfg: Stock-Franka scenario configuration shared by all policy consumers.
    Returns:
        None; the configuration is updated before native environment creation.
    Raises:
        ImportError: The pinned Isaac Lab runtime is unavailable.
        AttributeError: The native robot or TCP sensor contract is missing.
    """
    from isaaclab.controllers import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import (
        DifferentialInverseKinematicsActionCfg,
    )
    from isaaclab.managers import ObservationTermCfg

    env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        scale=tuple(stock_franka_policy_contract()["native_action_scale"]),
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=tuple(env_cfg.scene.ee_frame.target_frames[0].offset.pos)
        ),
    )
    env_cfg.observations.policy.actions = None
    env_cfg.observations.policy.episode_clock = ObservationTermCfg(func=_episode_clock)
    env_cfg.observations.policy.tcp_geometry = ObservationTermCfg(func=_tcp_geometry)


def configure_stock_franka_actor(config: dict[str, Any], *, task: str) -> bool:
    """Select the native GRU only for the stock-Franka scenario task.

    Args:
        config: Complete migrated RSL-RL runner configuration.
        task: The actual registered native task identifier.
    Returns:
        Whether the stock scenario actor profile was applied.
    Raises:
        RuntimeError: The native runner configuration lacks its actor dictionary.
    """
    if task != SCENARIO_TASK_ID:
        return False
    actor = config.get("actor")
    if not isinstance(actor, dict):
        raise RuntimeError("native stock-Franka runner lacks its actor configuration")
    contract = stock_franka_policy_contract()
    actor.update(
        {
            name: contract[name]
            for name in ("class_name", "rnn_type", "rnn_hidden_dim", "rnn_num_layers")
        }
    )
    actor["obs_normalization"] = contract["observation_normalization"]
    return True


def create_stock_franka_runner(
    runner_type: Any,
    env: Any,
    config: dict[str, Any],
    *,
    task: str,
    log_dir: str | None,
    device: str,
) -> Any:
    """Use the native runner's save API to bind every checkpoint to its policy semantics.

    Args:
        runner_type: The pinned native RSL-RL runner class.
        env: Its native vector environment.
        config: Complete migrated and configured native runner configuration.
        task: The actual registered native task identifier.
        log_dir: Native checkpoint and training log directory, or None for inference.
        device: The native Torch device.
    Returns:
        A native runner that records the stock profile on every saved checkpoint.
    Raises:
        RuntimeError: Supplied checkpoint metadata contradicts the stock profile.
    """
    selected_type = runner_type
    if task == SCENARIO_TASK_ID:
        selected_type = _profiled_runner_type(runner_type)
    return selected_type(env, config, log_dir=log_dir, device=device)


def _profiled_runner_type(runner_type: Any) -> Any:
    class ProfiledRunner(runner_type):
        def save(self, path, infos=None):
            _validate_actor_model(self.alg.actor)
            metadata = dict(infos or {})
            contract = stock_franka_policy_contract()
            previous = metadata.get("npa_stock_franka_policy", contract)
            if previous != contract:
                raise RuntimeError(
                    "checkpoint save contradicts the stock-Franka profile"
                )
            metadata["npa_stock_franka_policy"] = contract
            return super().save(path, infos=metadata)

    return ProfiledRunner


def _validate_actor_model(actor: Any) -> None:
    contract = stock_franka_policy_contract()
    recurrent = getattr(getattr(actor, "rnn", None), "rnn", None)
    normalizer = getattr(actor, "obs_normalizer", None)
    if recurrent is None or (
        type(recurrent).__name__ != "GRU"
        or recurrent.input_size != contract["observation_dim"]
        or recurrent.hidden_size != contract["rnn_hidden_dim"]
        or recurrent.num_layers != contract["rnn_num_layers"]
        or actor.obs_groups != ["policy"]
    ):
        raise RuntimeError("native checkpoint actor differs from the stock-Franka GRU")
    if (
        type(normalizer).__name__ != "EmpiricalNormalization"
        or normalizer.mean.numel() != contract["observation_dim"]
        or int(normalizer.count) <= 0
        or actor.mlp[-1].out_features != contract["action_dim"]
    ):
        raise RuntimeError(
            "native checkpoint lacks initialized stock-Franka observations"
        )


def validate_stock_franka_checkpoint(infos: Any, *, task: str) -> None:
    """Reject a stock checkpoint with unbound or changed action and model semantics.

    Args:
        infos: Metadata returned by the native runner's checkpoint load API.
        task: The actual registered native task identifier.
    Returns:
        None when the checkpoint profile is exact, or for a different task.
    Raises:
        RuntimeError: The stock profile is missing, different, or has coerced values.
    """
    if task != SCENARIO_TASK_ID:
        return
    actual = infos.get("npa_stock_franka_policy") if isinstance(infos, dict) else None
    contract = stock_franka_policy_contract()
    if not isinstance(actual, dict) or actual != contract:
        raise RuntimeError("checkpoint lacks the exact stock-Franka policy profile")
    if any(type(actual[name]) is not type(value) for name, value in contract.items()):
        raise RuntimeError("checkpoint policy profile requires literal metadata")
    if any(type(value) is not float for value in actual["native_action_scale"]):
        raise RuntimeError("checkpoint policy action scales require literal metadata")

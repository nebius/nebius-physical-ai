"""Initialize a learned Franka actor from training-only native Isaac demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

_PROFILE = "franka-ik-distillation-v1"
_BOOTSTRAP_ROUNDS = 2
_FITTING_EPOCHS = 8
_MINIBATCH_SIZE = 4096


def resolve_training_bootstrap(configured: str, *, eligible: bool) -> str:
    """Select a training bootstrap only for the explicitly eligible native task.

    Args:
        configured: ``auto``, ``none``, or the named Franka training profile.
        eligible: Whether fresh PPO consumes the sealed training split on stock Franka.
    Returns:
        The named native training profile, or ``none`` for compatibility callers.
    Raises:
        ValueError: The setting is unknown or an explicit profile is ineligible.
    """
    if configured not in {"auto", "none", _PROFILE}:
        raise ValueError(f"unsupported Isaac training bootstrap: {configured!r}")
    if configured == "none":
        return "none"
    if configured == "auto":
        return _PROFILE if eligible else "none"
    if not eligible:
        raise ValueError(
            "Franka bootstrap requires fresh native PPO on sealed training data"
        )
    return _PROFILE


def _tensor(value: Any) -> Any:
    """Read the Torch view of a native Isaac Lab 3 buffer."""
    return getattr(value, "torch", value)


def _teacher_target(phase, object_position, goals, lift_targets, grasp_offsets):
    """Keep grasp approach, object lift, and exact goal transport distinct."""
    target = object_position.clone()
    target[phase == 0, 2] += 0.1
    target[phase == 3] = lift_targets[phase == 3]
    target[phase >= 4] = (goals + grasp_offsets)[phase >= 4]
    return target


def _next_teacher_phase(phase, elapsed, distance, finger_gap):
    """Advance from measured approach and finger closure before transport."""
    advance = (phase == 0) & (distance < 0.02) & (elapsed >= 0.1)
    advance |= (phase == 1) & (distance < 0.012) & (elapsed >= 0.1)
    advance |= (phase == 2) & (finger_gap < 0.07) & (elapsed >= 0.25)
    advance |= (phase == 3) & (distance < 0.025)
    return phase + advance.to(phase.dtype), advance


def _native_joint_contract(task):
    """Use the live task's joint mapping and action scaling for demonstration labels."""
    from isaaclab.managers import SceneEntityCfg

    robot = task.scene["robot"]
    entity = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    entity.resolve(task.scene)
    arm = task.action_manager.get_term("arm_action")
    if not robot.is_fixed_base or len(entity.joint_ids) != 7:
        raise RuntimeError(
            "Franka bootstrap requires its native seven-joint fixed base"
        )
    if list(arm._joint_ids) != list(entity.joint_ids):
        raise RuntimeError(
            "Franka bootstrap arm labels differ from the task action map"
        )
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    if len(fingers) != 2:
        raise RuntimeError("Franka bootstrap requires the native two-finger gripper")
    return robot, entity, arm, fingers


class _FrankaTeacher:
    """Generate native IK demonstrations exclusively inside the training task."""

    def __init__(self, task):
        import torch
        from isaaclab.controllers import (
            DifferentialIKController,
            DifferentialIKControllerCfg,
        )

        self.task = task
        self.robot, self.entity, self.arm, self.fingers = _native_joint_contract(task)
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=task.num_envs,
            device=task.device,
        )
        self.phase = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        self.elapsed = torch.zeros(task.num_envs, device=task.device)
        self.stable_steps = torch.zeros_like(self.phase)
        self.successful_rows = torch.zeros(
            len(task.npa_scenario_rows), dtype=torch.bool, device=task.device
        )
        self.grasp_offsets = torch.zeros((task.num_envs, 3), device=task.device)
        self.lift_targets = torch.zeros_like(self.grasp_offsets)
        self.orientation = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=task.device
        ).repeat(task.num_envs, 1)
        sensor = task.scene["ee_frame"]
        self.offset = torch.tensor(
            sensor.cfg.target_frames[0].offset.pos, device=task.device
        ).repeat(task.num_envs, 1)

    def _world_goals(self):
        from isaaclab.utils.math import combine_frame_transforms

        root = _tensor(self.robot.data.root_pose_w)
        command = self.task.command_manager.get_command("object_pose")
        goals, _ = combine_frame_transforms(
            root[:, :3], root[:, 3:7], command[:, :3], command[:, 3:7]
        )
        return goals

    def _record_stable_placements(self):
        import torch
        import isaac_scenario_task

        distance, speed, _, _ = isaac_scenario_task._placement_state(
            self.task, command_name="object_pose", object_name="object"
        )
        stable = (distance < isaac_scenario_task.STABLE_PLACEMENT_DISTANCE_M) & (
            speed < isaac_scenario_task.STABLE_PLACEMENT_SPEED_MPS
        )
        self.stable_steps = torch.where(
            stable, self.stable_steps + 1, torch.zeros_like(self.stable_steps)
        )
        rows = self.task.npa_scenario_indices[
            self.stable_steps >= isaac_scenario_task.STABLE_PLACEMENT_STEPS
        ]
        self.successful_rows[rows] = True

    def _advance(self, tcp, object_position, target):
        import torch

        self.elapsed += float(self.task.step_dt)
        distance = torch.linalg.vector_norm(tcp - target, dim=1)
        joints = _tensor(self.robot.data.joint_pos)
        finger_gap = joints[:, self.fingers].sum(dim=1)
        next_phase, advance = _next_teacher_phase(
            self.phase, self.elapsed, distance, finger_gap
        )
        grasped = (self.phase == 2) & advance
        self.grasp_offsets[grasped] = (tcp - object_position)[grasped]
        self.lift_targets[grasped] = tcp[grasped]
        self.lift_targets[grasped, 2] += 0.08
        self.phase = next_phase
        self.elapsed[advance] = 0

    def _joint_labels(self, tcp_target):
        import torch
        from isaaclab.utils.math import quat_apply

        wrist_target = tcp_target - quat_apply(self.orientation, self.offset)
        self.controller.set_command(torch.cat([wrist_target, self.orientation], dim=1))
        pose = _tensor(self.robot.data.body_pose_w)[:, self.entity.body_ids[0]]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)
        jacobian = jacobian[:, self.entity.body_ids[0] - 1, :, self.entity.joint_ids]
        joints = _tensor(self.robot.data.joint_pos)[:, self.entity.joint_ids]
        desired = self.controller.compute(pose[:, :3], pose[:, 3:7], jacobian, joints)
        desired = joints + (desired - joints).clamp(-0.1, 0.1)
        limits = _tensor(self.robot.data.soft_joint_pos_limits)[
            :, self.entity.joint_ids
        ]
        desired = torch.maximum(limits[..., 0], torch.minimum(limits[..., 1], desired))
        arm_labels = (desired - _tensor(self.arm._offset)) / _tensor(self.arm._scale)
        gripper = torch.where(self.phase >= 2, -1.0, 1.0).unsqueeze(1)
        return torch.cat([arm_labels, gripper], dim=1)

    def actions(self):
        import torch

        self._record_stable_placements()
        tcp = _tensor(self.task.scene["ee_frame"].data.target_pos_w)[:, 0]
        obj = _tensor(self.task.scene["object"].data.root_pos_w)
        target = _teacher_target(
            self.phase, obj, self._world_goals(), self.lift_targets, self.grasp_offsets
        )
        self._advance(tcp, obj, target)
        delta = target - tcp
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        step = delta * torch.minimum(
            torch.full_like(distance, 0.7), 0.01 / distance.clamp_min(1e-8)
        )
        return self._joint_labels(tcp + step)

    def reset(self, dones):
        self.phase[dones.bool()] = 0
        self.elapsed[dones.bool()] = 0
        self.stable_steps[dones.bool()] = 0
        self.controller.reset(dones.nonzero(as_tuple=False).flatten())


def _training_round(env, actor, *, learner_fraction):
    """Collect supervised native actions over every consumed training scenario."""
    import torch
    from tensordict import TensorDict

    task = env.unwrapped
    teacher = _FrankaTeacher(task)
    row_count = len(task.npa_scenario_rows)
    cycles = math.ceil(row_count / task.num_envs) + 1
    steps = cycles * int(task.max_episode_length)
    observations = {name: [] for name in actor.obs_groups}
    targets = []
    with torch.no_grad():
        for _ in range(steps):
            obs = env.get_observations()
            labels = teacher.actions()
            for name in observations:
                observations[name].append(obs[name].clone())
            targets.append(labels.clone())
            actions = labels
            if learner_fraction:
                actions = labels.lerp(actor(obs), learner_fraction)
            _, _, dones, _ = env.step(actions)
            teacher.reset(dones)
    combined = {name: torch.cat(rows) for name, rows in observations.items()}
    batch = TensorDict(combined, batch_size=[steps * task.num_envs])
    return batch, torch.cat(targets), teacher.successful_rows


def _fit_actor(actor, observations, labels, optimizer):
    """Fit the actor mean while preserving the native inference and noise modules."""
    import torch

    actor.update_normalization(observations)
    actor.train()
    losses = []
    for _ in range(_FITTING_EPOCHS):
        order = torch.randperm(len(labels), device=labels.device)
        for indices in order.split(_MINIBATCH_SIZE):
            error = actor(observations[indices]) - labels[indices]
            loss = error[:, :7].square().mean() + 2 * error[:, 7].square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Franka bootstrap actor loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    actor.eval()
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "optimizer_updates": len(losses),
    }


def _validate_bootstrap(env, runner, profile, training_split):
    if training_split != "train":
        raise ValueError("Franka demonstrations may consume only the training split")
    if profile != _PROFILE:
        raise ValueError(f"unsupported Isaac training bootstrap: {profile!r}")
    if os.environ.get("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0") != "0":
        raise RuntimeError("Franka bootstrap requires exact training goals")
    task = env.unwrapped
    if task.spec.id != "NPA-Lift-Cube-Franka-Scenarios-v0":
        raise RuntimeError("Franka bootstrap requires the curated native scenario task")
    if not getattr(task, "npa_scenario_rows", None) or env.num_actions != 8:
        raise RuntimeError(
            "Franka bootstrap requires consumed scenarios and eight actions"
        )
    if int(runner.current_learning_iteration) != 0:
        raise RuntimeError("Franka bootstrap cannot replace a resumed PPO checkpoint")
    actor = getattr(runner.alg, "actor", None)
    if actor is None or actor.is_recurrent:
        raise RuntimeError(
            "Franka bootstrap requires the native feed-forward RSL-RL actor"
        )
    return task, actor


def _bootstrap_round(env, actor, optimizer, *, index):
    batch, labels, successes = _training_round(
        env, actor, learner_fraction=0.0 if index == 0 else 0.5
    )
    count = int(successes.sum())
    if count * 2 < len(successes):
        raise RuntimeError(
            "native training demonstrations did not establish stable placement"
        )
    fit = _fit_actor(actor, batch, labels, optimizer)
    audit = {
        "round": index,
        "training_scenarios": len(successes),
        "teacher_stable_placements": count,
        "examples": len(labels),
        **fit,
    }
    print("ROBOT_BOOTSTRAP_ROUND " + json.dumps(audit, sort_keys=True), flush=True)
    return audit


def _publish_bootstrap_audit(task, runner, rounds, profile, output_dir):
    """Keep demonstration learning distinct from measured PPO updates."""
    import isaac_scenario_task

    if int(runner.current_learning_iteration) != 0:
        raise RuntimeError("training bootstrap changed the PPO update counter")
    audit = {
        "schema": "npa.sim2real.franka_bootstrap.v1",
        "profile": profile,
        "scope": "sealed_training_only",
        "gold_used": False,
        "inference_composition": "learned_actor_only",
        "ppo_updates": 0,
        "training_goals_exact": True,
        "training_scenarios": len(task.npa_scenario_rows),
        "rounds": rounds,
        "training_scenarios_sha256": hashlib.sha256(
            Path(os.environ[isaac_scenario_task.SCENARIO_ENV]).read_bytes()
        ).hexdigest(),
    }
    destination = Path(output_dir)
    (destination / "bootstrap-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    runner.save(str(destination / "bootstrap-checkpoint.pt"))
    return audit


def bootstrap_franka_actor(
    env: Any, runner: Any, *, profile: str, training_split: str, output_dir: str
) -> dict[str, Any]:
    """Distill training-only native IK demonstrations before the full PPO pass.

    Args:
        env: Native RSL-RL wrapper consuming only the sealed training scenarios.
        runner: Fresh RSL-RL runner whose learned actor receives supervised updates.
        profile: Explicit ``franka-ik-distillation-v1`` training configuration.
        training_split: The caller's verified sealed ``train`` split binding.
        output_dir: Native training directory for the audit and initial checkpoint.
    Returns:
        Training-only demonstration, fit, checkpoint, and unchanged PPO-count audit.
    Raises:
        ValueError: The profile is unsupported.
        RuntimeError: The task, actor, demonstrations, or learning evidence is invalid.
        OSError: Training evidence cannot be written.
    """
    import torch

    task, actor = _validate_bootstrap(env, runner, profile, training_split)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    rounds = [
        _bootstrap_round(env, actor, optimizer, index=index)
        for index in range(_BOOTSTRAP_ROUNDS)
    ]
    audit = _publish_bootstrap_audit(task, runner, rounds, profile, output_dir)
    env.reset()
    return audit

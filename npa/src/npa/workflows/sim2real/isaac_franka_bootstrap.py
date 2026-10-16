"""Initialize a learned Franka actor from training-only native Isaac demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.isaac_franka_policy import (
    BOOTSTRAP_PROFILE,
    stock_franka_policy_contract,
)
from npa.workflows.sim2real.isaac_recurrent_imitation import (
    RecurrentDemonstrationReplay,
)

_PROFILE = BOOTSTRAP_PROFILE
_BOOTSTRAP_ROUNDS = 2
_FITTING_EPOCHS = 32
_TEACHER_TCP_STEP_M = 0.1


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

        self.task = task
        self.robot, self.entity, self.arm, self.fingers = _native_joint_contract(task)
        self.phase = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        self.elapsed = torch.zeros(task.num_envs, device=task.device)
        self.stable_steps = torch.zeros_like(self.phase)
        self.successful_rows = torch.zeros(
            len(task.npa_scenario_rows), dtype=torch.bool, device=task.device
        )
        self.grasp_offsets = torch.zeros((task.num_envs, 3), device=task.device)
        self.lift_targets = torch.zeros_like(self.grasp_offsets)
        self.orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=task.device
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

    def _cartesian_labels(self, tcp_target):
        import torch
        from isaaclab.utils.math import subtract_frame_transforms

        root = _tensor(self.robot.data.root_pose_w)
        target, orientation = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], tcp_target, self.orientation
        )
        pose = torch.cat((target, orientation), dim=1) / _tensor(self.arm._scale)
        gripper = torch.where(self.phase >= 2, -1.0, 1.0).unsqueeze(1)
        return torch.cat((pose, gripper), dim=1)

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
            torch.full_like(distance, 0.7),
            _TEACHER_TCP_STEP_M / distance.clamp_min(1e-8),
        )
        return self._cartesian_labels(tcp + step)

    def reset(self, dones):
        self.phase[dones.bool()] = 0
        self.elapsed[dones.bool()] = 0
        self.stable_steps[dones.bool()] = 0


def _training_round(env, actor, *, learner_fraction):
    """Collect native observations, separate fit annotations, and strict actor outcomes."""
    import torch

    env.reset()
    task = env.unwrapped
    teacher = _FrankaTeacher(task)
    assessment = _TerminalPlacementAssessment(task)
    cycles = math.ceil(len(task.npa_scenario_rows) / task.num_envs) + 1
    steps = cycles * int(task.max_episode_length)
    collection = _DemonstrationCollection(actor)
    actor.reset()
    actor.eval()
    with torch.no_grad():
        for step in range(steps):
            observations = env.get_observations()
            labels = teacher.actions()
            prediction = actor(observations)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("native Franka actor produced nonfinite actions")
            collection.record(observations, labels, prediction, teacher.phase)
            actions = prediction
            if learner_fraction != 1.0:
                actions = labels.lerp(prediction, learner_fraction)
            if (step + 1) % int(task.max_episode_length) == 0:
                _report_teacher_progress(teacher, step + 1, steps, learner_fraction)
            _, _, dones, _ = env.step(actions)
            assessment.advance(task, dones)
            collection.resets.append(dones.bool().clone().unsqueeze(1))
            teacher.reset(dones)
            actor.reset(dones)
    actor.reset()
    return collection.finish(steps, assessment, teacher.successful_rows)


class _TerminalPlacementAssessment:
    """Seal each scenario's first physical episode using its last pre-reset state."""

    def __init__(self, task):
        import torch

        self.stable_steps = torch.zeros(
            task.num_envs, dtype=torch.long, device=task.device
        )
        self.scenario_indices = task.npa_scenario_indices.clone()
        self.completed_rows = torch.zeros(
            len(task.npa_scenario_rows), dtype=torch.bool, device=task.device
        )
        self.successful_rows = torch.zeros_like(self.completed_rows)

    def advance(self, task, dones):
        import torch
        import isaac_scenario_task

        completed = dones.bool()
        self._seal_completed(completed, isaac_scenario_task.STABLE_PLACEMENT_STEPS)
        distance, speed, _, _ = isaac_scenario_task._placement_state(
            task, command_name="object_pose", object_name="object"
        )
        stable = (distance < isaac_scenario_task.STABLE_PLACEMENT_DISTANCE_M) & (
            speed < isaac_scenario_task.STABLE_PLACEMENT_SPEED_MPS
        )
        # A done environment already exposes its next episode's reset state.
        self.stable_steps = torch.where(stable & ~completed, self.stable_steps + 1, 0)
        self.scenario_indices = task.npa_scenario_indices.clone()

    def _seal_completed(self, completed, required_steps):
        rows = self.scenario_indices[completed]
        first_episode = ~self.completed_rows[rows]
        rows = rows[first_episode]
        stable = self.stable_steps[completed][first_episode] >= required_steps
        self.successful_rows[rows] = True
        # If identical scenarios finish together, any failure remains a failure.
        self.successful_rows[rows[~stable]] = False
        self.completed_rows[rows] = True

    def evidence(self, ever_successful_rows):
        return {
            "placement_criterion": "first_episode_last_pre_reset",
            "terminal_training_scenarios": int(self.completed_rows.sum()),
            "ever_training_placements": int(ever_successful_rows.sum()),
        }


def _report_teacher_progress(teacher, step, total_steps, learner_fraction):
    """Expose measured collection progress before any supervised fitting starts."""
    import torch

    progress = {
        "collection_step": step,
        "collection_steps": total_steps,
        "learner_fraction": learner_fraction,
        "phase_counts": torch.bincount(teacher.phase, minlength=5).tolist(),
        "training_scenarios": len(teacher.successful_rows),
        "stable_training_placements": int(teacher.successful_rows.sum()),
        "placement_criterion": "any_episode_any_stable_event_diagnostic_only",
        "gold_used": False,
    }
    print(
        "ROBOT_BOOTSTRAP_PROGRESS " + json.dumps(progress, sort_keys=True), flush=True
    )


def _fit_actor(actor, observations, labels, optimizer, *, replay):
    """Keep native GRU inference, noise, reset boundaries, and retained demonstrations."""
    return replay.fit(actor, observations, labels, optimizer, epochs=_FITTING_EPOCHS)


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
    if actor is None or not actor.is_recurrent:
        raise RuntimeError(
            "Franka bootstrap requires the native recurrent RSL-RL actor"
        )
    from npa.workflows.sim2real.isaac_franka_control import stock_franka_control_audit

    stock_franka_control_audit(task.cfg)
    return task, actor


def _bootstrap_round(env, actor, optimizer, *, index, replay=None):
    batch, labels, successes, diagnostics, recovery = _guarded_demonstrations(
        env, actor, optimizer, index=index, replay=replay
    )
    count = int(successes.sum())
    fit = _fit_actor(actor, batch, labels, optimizer, replay=replay)
    audit = {
        "round": index,
        "training_scenarios": len(successes),
        "teacher_stable_placements": count,
        "examples": len(labels),
        "warmup_recovery": recovery,
        **diagnostics,
        **fit,
    }
    print("ROBOT_BOOTSTRAP_ROUND " + json.dumps(audit, sort_keys=True), flush=True)
    return audit


def _require_successful_demonstrations(successes, *, index):
    count = int(successes.sum())
    if count * 2 < len(successes):
        raise RuntimeError(
            "native training demonstrations did not establish stable placement: "
            f"round {index}, {count}/{len(successes)} training scenarios"
        )


def _guarded_demonstrations(env, actor, optimizer, *, index, replay):
    recovery = []
    fraction = 0.0 if index == 0 else 0.5
    while True:
        batch, labels, successes, diagnostics = _training_round(
            env, actor, learner_fraction=fraction
        )
        if int(successes.sum()) * 2 >= len(successes):
            return batch, labels, successes, diagnostics, recovery
        if index == 0:
            _require_successful_demonstrations(successes, index=index)
        failed = _collection_evidence(successes, diagnostics, fraction=fraction)
        del batch, labels
        recovery.append(
            _recover_mixed_demonstrations(env, actor, optimizer, replay, failed)
        )


def _collection_evidence(successes, diagnostics, *, fraction):
    return {
        "scope": "sealed_training_only",
        "training_scenarios": len(successes),
        "strict_training_placements": int(successes.sum()),
        "learner_fraction": fraction,
        "inference_composition": "training_demonstration_mixture",
        "gold_used": False,
        "ppo_updates": 0,
        **diagnostics,
    }


def _recover_mixed_demonstrations(env, actor, optimizer, replay, failed):
    batch, labels, successes, diagnostics = _training_round(
        env, actor, learner_fraction=0.0
    )
    _require_successful_demonstrations(successes, index="teacher recovery")
    fit = _fit_actor(actor, batch, labels, optimizer, replay=replay)
    recovery = {
        "failed_mixed_collection": failed,
        "teacher_recovery_fit": {
            **_collection_evidence(successes, diagnostics, fraction=0.0),
            "examples": len(labels),
            "fit": fit,
        },
    }
    print(
        "ROBOT_BOOTSTRAP_WARMUP_RECOVERY " + json.dumps(recovery, sort_keys=True),
        flush=True,
    )
    return recovery


def _publish_bootstrap_audit(task, runner, rounds, profile, output_dir, evidence):
    """Keep demonstration learning distinct from measured PPO updates."""
    import isaac_scenario_task
    from npa.workflows.sim2real.isaac_franka_control import stock_franka_control_audit

    if int(runner.current_learning_iteration) != 0:
        raise RuntimeError("training bootstrap changed the PPO update counter")
    audit = {
        "schema": "npa.sim2real.franka_bootstrap.v3",
        "profile": profile,
        "scope": "sealed_training_only",
        "gold_used": False,
        "inference_composition": "learned_actor_only",
        "ppo_updates": 0,
        "training_goals_exact": True,
        "robot_control": stock_franka_control_audit(task.cfg),
        "training_scenarios": len(task.npa_scenario_rows),
        "rounds": rounds,
        "policy_model": stock_franka_policy_contract(),
        **evidence,
        "training_scenarios_sha256": hashlib.sha256(
            Path(os.environ[isaac_scenario_task.SCENARIO_ENV]).read_bytes()
        ).hexdigest(),
    }
    destination = Path(output_dir)
    (destination / "bootstrap-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    # RSL-RL initializes its checkpoint logger inside learn(); PPO owns saving.
    return audit


def bootstrap_franka_actor(
    env: Any, runner: Any, *, profile: str, training_split: str, output_dir: str
) -> dict[str, Any]:
    """Establish an actor-only training handoff before the complete native PPO pass.

    Args:
        env: Native vector wrapper consuming only the sealed training scenarios.
        runner: Fresh RSL-RL runner whose GRU receives supervised updates.
        profile: Explicit ``franka-cartesian-gru-distillation-v2`` configuration.
        training_split: The caller's verified sealed ``train`` split binding.
        output_dir: Native output directory for the measured bootstrap audit.
    Returns:
        Demonstration, corrective fitting, and strict actor-only handoff evidence.
    Raises:
        ValueError: The profile or training split is unsupported.
        RuntimeError: The native task, fitting, or actor-only handoff is invalid.
        OSError: Training evidence cannot be written.
    """
    import torch

    task, actor = _validate_bootstrap(env, runner, profile, training_split)
    replay = RecurrentDemonstrationReplay(task.num_envs)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    rounds = [
        _bootstrap_round(env, actor, optimizer, index=index, replay=replay)
        for index in range(_BOOTSTRAP_ROUNDS)
    ]
    evidence = _corrective_training_handoff(env, actor, replay)
    _freeze_training_handoff(actor, evidence)
    audit = _publish_bootstrap_audit(
        task, runner, rounds, profile, output_dir, evidence
    )
    env.reset()
    actor.reset()
    return audit


def _freeze_training_handoff(actor, evidence):
    from npa.workflows.sim2real.isaac_franka_normalization import (
        freeze_training_normalization,
    )

    evidence["normalization_freeze"] = freeze_training_normalization(actor)
    print(
        "ROBOT_NORMALIZATION_FROZEN "
        + json.dumps(evidence["normalization_freeze"], sort_keys=True),
        flush=True,
    )


class _DemonstrationCollection:
    def __init__(self, actor):
        self.observations = {name: [] for name in actor.obs_groups}
        self.targets = []
        self.resets = []
        self.phases = []
        self.arm_error = 0.0
        self.gripper_error = 0.0

    def record(self, observations, labels, prediction, phase):
        for name, rows in self.observations.items():
            rows.append(observations[name].clone())
        self.targets.append(labels.clone())
        self.phases.append(phase.clone().unsqueeze(1))
        error = prediction - labels
        self.arm_error += error[:, :-1].square().mean()
        self.gripper_error += error[:, -1].square().mean()

    def finish(self, steps, assessment, ever_successful_rows):
        import torch
        from tensordict import TensorDict

        combined = {name: torch.cat(rows) for name, rows in self.observations.items()}
        combined["__done"] = torch.cat(self.resets)
        combined["__phase"] = torch.cat(self.phases)
        observations = TensorDict(combined, batch_size=[len(combined["__done"])])
        diagnostics = {
            "arm_label_mse": float(self.arm_error / steps),
            "gripper_label_mse": float(self.gripper_error / steps),
            **assessment.evidence(ever_successful_rows),
        }
        return (
            observations,
            torch.cat(self.targets),
            assessment.successful_rows,
            diagnostics,
        )


def _actor_collection_evidence(env, actor, *, learner_fraction):
    batch, labels, successes, diagnostics = _training_round(
        env, actor, learner_fraction=learner_fraction
    )
    evidence = {
        "scope": "sealed_training_only",
        "training_scenarios": len(successes),
        "strict_training_placements": int(successes.sum()),
        "learner_fraction": learner_fraction,
        "inference_composition": "learned_actor_only"
        if learner_fraction == 1.0
        else "training_demonstration_mixture",
        "gold_used": False,
        "ppo_updates": 0,
        **diagnostics,
    }
    return batch, labels, evidence


def _corrective_training_handoff(env, actor, replay):
    import torch

    batch, labels, baseline = _actor_collection_evidence(
        env, actor, learner_fraction=1.0
    )
    del batch, labels
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    corrective = _fit_corrective_collection(env, actor, replay, optimizer)
    additional = []
    assessments = []
    while True:
        batch, labels, handoff = _actor_collection_evidence(
            env, actor, learner_fraction=1.0
        )
        del batch, labels
        assessments.append(handoff)
        print(
            "ROBOT_ACTOR_HANDOFF_ASSESSMENT " + json.dumps(handoff, sort_keys=True),
            flush=True,
        )
        if handoff["strict_training_placements"] * 2 >= handoff["training_scenarios"]:
            return {
                "actor_baseline": baseline,
                "corrective_fit": corrective,
                "additional_corrective_fits": additional,
                "actor_handoff_assessments": assessments,
                "actor_handoff": handoff,
            }
        additional.append(_fit_corrective_collection(env, actor, replay, optimizer))


def _fit_corrective_collection(env, actor, replay, optimizer):
    batch, labels, evidence = _actor_collection_evidence(
        env, actor, learner_fraction=0.75
    )
    evidence["fit"] = _fit_actor(actor, batch, labels, optimizer, replay=replay)
    print(
        "ROBOT_ACTOR_CORRECTIVE_FIT " + json.dumps(evidence, sort_keys=True), flush=True
    )
    return evidence

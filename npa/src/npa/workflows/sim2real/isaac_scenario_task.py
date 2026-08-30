"""Apply curated Sim2Real scenarios to vectorized Isaac Lift environments.

This module is shipped into Isaac sibling jobs and imported only after
``AppLauncher`` boots.  Its pure parsing/digest helpers remain importable in CPU
tests.  The reset event binds each Isaac env index to one scenario record and
applies the exact object pose, goal, friction, and mass scale.  Global scene and
lighting fields are accepted only at their stock fixed values, avoiding the old
label-only fiction that each cloned env had an independently configurable dome.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCENARIO_ENV = "NPA_SIM2REAL_SCENARIOS_JSONL"
SCENARIO_TASK_ID = "NPA-Lift-Cube-Franka-Scenarios-v0"
STOCK_TASK_ID = "Isaac-Lift-Cube-Franka-v0"
EXPECTED_SCENE_ID = "isaac://Isaac-Lift-Cube-Franka-v0/stock-table-v1"
EXPECTED_LIGHT_INTENSITY = 3000.0
STABLE_PLACEMENT_DISTANCE_M = 0.05
STABLE_PLACEMENT_SPEED_MPS = 0.03
STABLE_PLACEMENT_STEPS = 3
# A real nine-pass Franka run drove deterministic held-out reach from 0/64 to
# 64/64 while contact stayed 0/64 throughout. PPO's sampled training episodes
# still reported lift reward, but the actor mean kept the binary gripper open.
# Install the already-proven robot-agnostic grasp terms on the stock scenario
# task as well: closure is rewarded only near the object, and lift intent remains
# gated by near + closed so raising an empty hand cannot retain the signal.
GRASP_CLOSURE_REWARD_WEIGHT = 16.0
GRASP_CLOSURE_STD_M = 0.06
GRASP_LIFT_ATTEMPT_REWARD_WEIGHT = 32.0
GRASP_LIFT_ATTEMPT_STD_M = 0.05
# The first real stock-Franka run with the grasp precursors converted 0/64
# contact into 64/64 contact and 53/64 stable grasps, but deterministic rollouts
# repeatedly stopped at 0.040-0.044 m object lift.  The stock Lift reward is a
# step at its minimal-height boundary, so the last centimetre before the strict
# 5 cm evaluator threshold still had no object-space gradient.  Reuse the
# existing robot-agnostic continuous object-height term on the stock task.  Its
# weight matches the lift-attempt precursor: the hand signal bootstraps motion,
# then real object displacement—not a scripted action—must retain the reward.
STOCK_DENSE_LIFT_REWARD_WEIGHT = 32.0
STOCK_DENSE_LIFT_STD_M = 0.08
STOCK_GRIPPER_JOINT_NAMES = (
    "panda_finger_joint1",
    "panda_finger_joint2",
)
STOCK_GRIPPER_OPEN_POSITION = 0.04
STOCK_GRIPPER_CLOSED_POSITION = 0.0
# The first validation canary learned reach/contact and 2/3 grasp+lift, but the
# closest goal distance remained 0.205-0.364 m.  A 0.15 m tanh scale multiplied
# by weight 8 is effectively flat at that boundary, so lifting remains the much
# easier dominant objective.  Keep the strict verdict at 5 cm while making the
# curriculum genuinely dense across the observed post-lift placement basin.
PLACEMENT_APPROACH_STD_M = 0.35
PLACEMENT_NEAR_STD_M = 0.08
PLACEMENT_HOLD_STD_M = 0.15
PLACEMENT_HOLD_REWARD_FLOOR = 0.20
PLACEMENT_HOLD_MAX_DISTANCE_M = 0.30
PLACEMENT_DWELL_SCALE = 2.0
PLACEMENT_SETTLING_SPEED_MPS = 0.20
# Train12 crossed the strict basin at 0.10-0.12 m/s and managed only one
# sub-0.03 m/s sample before leaving. A 1 cm logistic transition starts the
# braking reward only after the policy is effectively at the 5 cm boundary,
# which is physically too late. Keep the verdict at 5 cm, but expose its
# positive-only settling gradient over the preceding 5 cm of approach.
PLACEMENT_BASIN_WIDTH_M = 0.05
PLACEMENT_GOAL_CURRICULUM_LIFT_M = 0.08
PLACEMENT_PROGRESS_SCALE_M = 0.02
PLACEMENT_PROGRESS_REWARD_WEIGHT = 512.0
# Isaac RewardManager multiplies every weight by the environment dt.  Train5's
# nominal -50 terminal weight contributed only about -0.16 to late episode
# summaries while throw/drop returns exceeded 18, so terminal consequences must
# be expressed on that time-scaled basis. Train6 then showed that -5000 from the
# first step suppressed grasp/lift discovery; the bounded penalty below ramps with
# the goal curriculum while the rarer exact completion retains the larger bonus.
PLACEMENT_DROP_PENALTY_WEIGHT = -2000.0
PLACEMENT_COMPLETION_REWARD_WEIGHT = 5000.0
STABLE_PLACEMENT_REWARD_WEIGHT = 32.0
PLACEMENT_BASIN_SETTLING_REWARD_WEIGHT = 256.0
# Eval34 left 51/64 objects inside the strict distance basin but produced no
# terminal three-step stillness. Object velocity is a delayed physical outcome;
# near-goal arm joint speed is the directly controllable cause. Train32 proved
# that a 0.15 rad/s tanh scale saturates at ordinary arm speed: its aggregate
# arm-stillness reward stayed effectively zero and only checkpoint 5300 reached
# one actor-only strict validation success. Keep this positive-only and confined
# to the same braking envelope, but expose a gradient over normal joint motion.
# This scale is curriculum, not the unchanged object-space success threshold.
PLACEMENT_ARM_SETTLING_SPEED_RADPS = 1.0
# Train33 made the widened signal observable (about 10 reward at the final
# iteration), put 57/64 validation objects inside 5 cm, and produced a genuine
# actor-only terminal placement. Train34 tested weight 4096; the term dominated
# at about 70 reward, regressed transport, and all seven validation checkpoints
# fell to 0/64. Retain the last empirically credible balanced weight.
PLACEMENT_ARM_STILLNESS_REWARD_WEIGHT = 512.0
# Train14 put 49/64 validation objects inside the unchanged 5 cm basin and left
# 30 there, yet only two ever crossed 0.03 m/s and neither held it for the three
# required steps.  Reward the exact consecutive-step event itself so PPO can
# distinguish a transient slow sample from the required dwell.  This does not
# change any success threshold and remains zero outside the strict event.
PLACEMENT_STRICT_DWELL_REWARD_WEIGHT = 4096.0
PLACEMENT_DWELL_REWARD_EXPONENT = 2.0
# Train23 proved that even a transition-gated negative break consequence made
# the exact stable state avoidable: every validation checkpoint regressed to a
# one-step maximum. Use only positive quadratic progression for unfinished dwell.
# The first step remains a waypoint while the second, third, and continued stable
# steps dominate a fly-through without making approach negative anywhere.
# Train25 then isolated the same delayed-penalty failure after completion: one
# PPO update erased Train21's 16-step placement because all later unstable steps
# made the return for reaching that rare success strongly negative. Retention is
# therefore positive-only too. Saturated dwell pays every continued stable step;
# the existing signed physical progress term still penalizes actual departure.
PLACEMENT_MINIMAL_LIFT_M = 0.04
_COMMAND_TYPE: type | None = None
_DROP_PENALTY_TYPE: type | None = None
_SCENARIO_CACHE: tuple[str, int, str, list[dict[str, Any]]] | None = None


class ScenarioContractError(ValueError):
    """Raised before simulation when a scenario cannot be applied honestly."""


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def read_scenarios(path: str) -> list[dict[str, Any]]:
    """Read and validate a curated JSONL split."""

    global _SCENARIO_CACHE
    source = Path(path)
    if not source.is_file():
        raise ScenarioContractError(f"scenario JSONL does not exist: {source}")
    cache_key = str(source.resolve())
    cache_mtime = source.stat().st_mtime_ns
    expected_contract_digest = os.environ.get(
        "NPA_SIM2REAL_TASK_CONTRACT_DIGEST", ""
    ).strip()
    if _SCENARIO_CACHE and _SCENARIO_CACHE[:3] == (
        cache_key,
        cache_mtime,
        expected_contract_digest,
    ):
        return _SCENARIO_CACHE[3]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioContractError(
                f"invalid scenario JSON on line {number}: {exc}"
            ) from exc
        applied = row.get("applied_config") or {}
        digest = str(row.get("scenario_config_digest") or "")
        if not digest or _digest(applied) != digest:
            raise ScenarioContractError(
                f"scenario {row.get('env_id', number)!r} has an invalid config digest"
            )
        if digest in seen:
            raise ScenarioContractError(f"duplicate scenario config digest: {digest}")
        if applied.get("scene_id") != EXPECTED_SCENE_ID:
            raise ScenarioContractError(
                f"unsupported scene {applied.get('scene_id')!r}; the stock Lift task "
                f"can apply only {EXPECTED_SCENE_ID!r}"
            )
        light = float((applied.get("lighting") or {}).get("dome_intensity", -1))
        if light != EXPECTED_LIGHT_INTENSITY:
            raise ScenarioContractError(
                "per-env lighting is not supported by the cloned stock Lift scene; "
                f"expected fixed {EXPECTED_LIGHT_INTENSITY}, got {light}"
            )
        if (
            row.get("task_id") != STOCK_TASK_ID
            or applied.get("task_id") != STOCK_TASK_ID
        ):
            raise ScenarioContractError(
                f"scenario {row.get('env_id', number)!r} is not bound to {STOCK_TASK_ID}"
            )
        contract_digest = str(row.get("task_contract_digest") or "")
        if (
            not contract_digest
            or applied.get("task_contract_digest") != contract_digest
        ):
            raise ScenarioContractError(
                f"scenario {row.get('env_id', number)!r} has no matching task-contract digest"
            )
        rows.append(row)
        seen.add(digest)
    if not rows:
        raise ScenarioContractError(f"scenario JSONL is empty: {source}")
    if len({str(row["task_contract_digest"]) for row in rows}) != 1:
        raise ScenarioContractError("scenario split mixes task-contract digests")
    if (
        expected_contract_digest
        and rows[0]["task_contract_digest"] != expected_contract_digest
    ):
        raise ScenarioContractError(
            "scenario split task-contract digest does not match the authoritative "
            "Stage 2 contract"
        )
    _SCENARIO_CACHE = (cache_key, cache_mtime, expected_contract_digest, rows)
    return rows


def scenarios_from_env(env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    source = os.environ if env is None else env
    path = str(source.get(SCENARIO_ENV) or "").strip()
    return read_scenarios(path) if path else []


def scenario_contract_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    difficulties: dict[str, int] = {}
    for row in rows:
        name = str(row.get("difficulty") or "unknown")
        difficulties[name] = difficulties.get(name, 0) + 1
    return {
        "scenario_count": len(rows),
        "unique_config_digests": len(
            {str(row["scenario_config_digest"]) for row in rows}
        ),
        "difficulty": difficulties,
        "task_contract_digests": sorted(
            {str(row.get("task_contract_digest") or "") for row in rows}
        ),
        "scene_id": EXPECTED_SCENE_ID,
        "lighting_intensity": EXPECTED_LIGHT_INTENSITY,
    }


def scenario_assignment_indices(
    *, count: int, row_count: int, cursor: int = 0, offset: int = 0
) -> list[int]:
    """Return a deterministic round-robin window over curated scenarios.

    Reset callbacks can contain only a subset of vector environments.  Advancing
    each environment independently can therefore revisit already-seen records
    before the tail of the curated split is ever assigned.  A single monotonic
    cursor guarantees complete coverage after ``row_count`` assignments while
    retaining deterministic wraparound and an operator-selected offset.
    """

    if count < 0:
        raise ValueError("scenario assignment count must be non-negative")
    if row_count <= 0:
        raise ValueError("scenario assignment requires at least one row")
    return [int((cursor + offset + index) % row_count) for index in range(count)]


def _ensure_runtime_buffers(env: Any, rows: list[dict[str, Any]]) -> None:
    import torch

    if not hasattr(env, "npa_scenario_indices"):
        env.npa_scenario_indices = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env.npa_scenario_assignment_cursor = 0
        env.npa_scenario_applied_counts = torch.zeros(
            len(rows), dtype=torch.long, device=env.device
        )
        env.npa_scenario_rows = rows
    if not hasattr(env, "npa_scenario_assignment_cursor"):
        env.npa_scenario_assignment_cursor = 0
    if not hasattr(env, "npa_scenario_episode_counts"):
        env.npa_scenario_episode_counts = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "npa_stable_placement_steps"):
        env.npa_stable_placement_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "npa_stable_placement_reward_steps"):
        env.npa_stable_placement_reward_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "npa_stable_placement_achieved"):
        env.npa_stable_placement_achieved = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "npa_stable_placement_newly_achieved"):
        env.npa_stable_placement_newly_achieved = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "npa_previous_placement_distance"):
        env.npa_previous_placement_distance = torch.full(
            (env.num_envs,), float("inf"), dtype=torch.float, device=env.device
        )


def _assign(env: Any, env_ids: Any, rows: list[dict[str, Any]]) -> Any:
    import torch

    _ensure_runtime_buffers(env, rows)
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    rotate = os.environ.get("NPA_SIM2REAL_SCENARIO_ROTATE_ON_RESET", "1") != "0"
    offset = int(os.environ.get("NPA_SIM2REAL_SCENARIO_OFFSET", "0") or 0)
    if rotate:
        cursor = int(env.npa_scenario_assignment_cursor)
        scenario_ids = torch.tensor(
            scenario_assignment_indices(
                count=int(ids.numel()),
                row_count=len(rows),
                cursor=cursor,
                offset=offset,
            ),
            dtype=torch.long,
            device=env.device,
        )
        env.npa_scenario_assignment_cursor = cursor + int(ids.numel())
        env.npa_scenario_episode_counts[ids] += 1
    else:
        scenario_ids = (ids + offset) % len(rows)
    env.npa_scenario_indices[ids] = scenario_ids
    env.npa_stable_placement_steps[ids] = 0
    env.npa_stable_placement_reward_steps[ids] = 0
    env.npa_stable_placement_achieved[ids] = False
    env.npa_stable_placement_newly_achieved[ids] = False
    env.npa_previous_placement_distance[ids] = float("inf")
    env.npa_scenario_applied_counts += torch.bincount(scenario_ids, minlength=len(rows))
    return ids, scenario_ids


def apply_scenario_reset(env: Any, env_ids: Any, asset_cfg: Any) -> None:
    """Isaac reset event applying exact point configuration per environment."""

    import torch

    rows = scenarios_from_env()
    ids, scenario_ids = _assign(env, env_ids, rows)
    selected = [rows[int(index)] for index in scenario_ids.detach().cpu().tolist()]
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[ids].clone()
    offsets = torch.tensor(
        [
            [
                float(row["object_placement"]["x"]),
                float(row["object_placement"]["y"]),
                float(row["object_placement"]["z"]),
            ]
            for row in selected
        ],
        device=asset.device,
        dtype=root_state.dtype,
    )
    positions = root_state[:, :3] + env.scene.env_origins[ids] + offsets
    asset.write_root_pose_to_sim(
        torch.cat((positions, root_state[:, 3:7]), dim=-1), env_ids=ids
    )
    asset.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:13]), env_ids=ids)

    ids_cpu = ids.cpu()
    # Exact per-env material values.  This mirrors Isaac Lab's own material event
    # setter but supplies the curated values instead of resampling buckets.
    materials = asset.root_physx_view.get_material_properties()
    friction = torch.tensor(
        [float(row["physics"]["friction"]) for row in selected],
        dtype=materials.dtype,
        device=materials.device,
    )
    materials[ids_cpu, :, 0] = friction[:, None]
    materials[ids_cpu, :, 1] = friction[:, None]
    materials[ids_cpu, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, ids_cpu)

    masses = asset.root_physx_view.get_masses()
    default_mass = asset.data.default_mass.cpu()
    mass_scale = torch.tensor(
        [float(row["physics"]["mass_scale"]) for row in selected],
        dtype=masses.dtype,
        device=masses.device,
    )
    masses[ids_cpu] = default_mass[ids_cpu].to(masses.device) * mass_scale[:, None]
    asset.root_physx_view.set_masses(masses, ids_cpu)

    if not getattr(env, "npa_scenario_first_reset_logged", False):
        print(
            "SCENARIO_APPLIED_FIRST_RESET",
            json.dumps(
                [
                    {
                        "isaac_env_index": int(ids[i].item()),
                        "env_id": row["env_id"],
                        "config_digest": row["scenario_config_digest"],
                    }
                    for i, row in enumerate(selected[:8])
                ],
                sort_keys=True,
            ),
            flush=True,
        )
        env.npa_scenario_first_reset_logged = True


def goal_curriculum_fraction(step: int, full_goal_step: int) -> float:
    """Return the bounded easy-to-exact scenario-goal curriculum fraction."""

    if full_goal_step <= 0:
        raise ScenarioContractError("goal curriculum full step must be positive")
    return min(1.0, max(0.0, float(step) / float(full_goal_step)))


def drop_penalty_schedule_fraction(
    step: int, full_goal_step: int, *, curriculum_enabled: bool
) -> float:
    """Ramp drop consequences with first-pass goal difficulty.

    Resumed passes use the exact goal from their first step and therefore apply
    the complete consequence immediately. A from-scratch pass starts at zero so
    PPO can discover grasp/lift before dropping becomes increasingly expensive.
    """

    if not curriculum_enabled:
        return 1.0
    return goal_curriculum_fraction(step, full_goal_step)


def _scheduled_drop_penalty_type(base_type: type | None = None) -> type:
    """Wrap Isaac's stateful structured termination term with phase scaling.

    Isaac Lab 2.3 implements ``mdp.is_terminated_term`` as ``ManagerTermBase``:
    RewardManager constructs it with ``(RewardTermCfg, env)`` and only then calls
    the instance with ``(env, term_keys)``.  Subclassing that contract preserves
    its resolved termination names and timeout filtering instead of incorrectly
    invoking the class as a free function. ``base_type`` is an injection seam for
    CPU contract tests; production always resolves the pinned Isaac class.
    """

    global _DROP_PENALTY_TYPE
    injected = base_type is not None
    if not injected and _DROP_PENALTY_TYPE is not None:
        return _DROP_PENALTY_TYPE
    if base_type is None:
        from isaaclab.envs import mdp

        base_type = mdp.is_terminated_term

    base_call: Any = base_type.__call__

    def scheduled_call(
        self: Any,
        env: Any,
        term_keys: str | list[str] = "object_dropping",
        full_goal_step: int = 1,
    ) -> Any:
        env_step = int(getattr(env, "_sim_step_counter", 0)) // int(env.cfg.decimation)
        scale = drop_penalty_schedule_fraction(
            env_step,
            full_goal_step,
            curriculum_enabled=(
                os.environ.get("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0") == "1"
            ),
        )
        structured = base_call(self, env, term_keys=term_keys)
        return structured * float(scale)

    ScheduledDropPenalty = type(
        "ScheduledDropPenalty",
        (base_type,),
        {"__call__": scheduled_call, "__module__": __name__},
    )

    if not injected:
        _DROP_PENALTY_TYPE = ScheduledDropPenalty
    return ScheduledDropPenalty


def _scenario_command_type() -> type:
    global _COMMAND_TYPE
    if _COMMAND_TYPE is not None:
        return _COMMAND_TYPE

    import torch
    from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

    class ScenarioPoseCommand(UniformPoseCommand):
        def _resample_command(self, env_ids: Any) -> None:
            rows = scenarios_from_env()
            _ensure_runtime_buffers(self._env, rows)
            ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            ).flatten()
            scenario_ids = self._env.npa_scenario_indices[ids]
            selected = [rows[int(index)] for index in scenario_ids.cpu().tolist()]
            goals = torch.tensor(
                [
                    [
                        float(row["goal_placement"]["x"]),
                        float(row["goal_placement"]["y"]),
                        float(row["goal_placement"]["z"]),
                    ]
                    for row in selected
                ],
                dtype=self.pose_command_b.dtype,
                device=self.device,
            )
            curriculum_enabled = (
                os.environ.get("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0") == "1"
            )
            full_goal_step = int(
                os.environ.get("NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP", "1") or 1
            )
            env_step = int(getattr(self._env, "_sim_step_counter", 0)) // int(
                self._env.cfg.decimation
            )
            fraction = (
                goal_curriculum_fraction(env_step, full_goal_step)
                if curriculum_enabled
                else 1.0
            )
            if fraction < 1.0:
                object_offsets = torch.tensor(
                    [
                        [
                            float(row["object_placement"]["x"]),
                            float(row["object_placement"]["y"]),
                            float(row["object_placement"]["z"]),
                        ]
                        for row in selected
                    ],
                    dtype=goals.dtype,
                    device=self.device,
                )
                object_default = (
                    self._env.scene["object"]
                    .data.default_root_state[ids, :3]
                    .to(goals.dtype)
                )
                easy_goals = object_default + object_offsets
                easy_goals[:, 2] += PLACEMENT_GOAL_CURRICULUM_LIFT_M
                goals = easy_goals + fraction * (goals - easy_goals)
            assignment_count = int(ids.numel())
            self._env.npa_goal_curriculum_assignment_count = (
                int(getattr(self._env, "npa_goal_curriculum_assignment_count", 0))
                + assignment_count
            )
            self._env.npa_goal_curriculum_true_assignments = int(
                getattr(self._env, "npa_goal_curriculum_true_assignments", 0)
            ) + (assignment_count if fraction >= 1.0 else 0)
            self._env.npa_goal_curriculum_max_fraction = max(
                float(getattr(self._env, "npa_goal_curriculum_max_fraction", 0.0)),
                fraction,
            )
            self.pose_command_b[ids, :3] = goals
            self.pose_command_b[ids, 3:] = 0.0
            self.pose_command_b[ids, 3] = 1.0

    _COMMAND_TYPE = ScenarioPoseCommand
    return ScenarioPoseCommand


def placement_curriculum_signal(
    distance: Any,
    speed: Any,
    hold_distance: Any,
    *,
    tanh: Any,
    approach_std_m: float = PLACEMENT_APPROACH_STD_M,
    near_std_m: float = PLACEMENT_NEAR_STD_M,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
    dwell_scale: float = PLACEMENT_DWELL_SCALE,
    settling_speed_mps: float = PLACEMENT_SETTLING_SPEED_MPS,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
) -> Any:
    """Return held-object proximity plus a near-target settling objective.

    Train3 exposed that velocity-gating the whole approach signal suppresses
    deliberate transport. Train4 then showed that a signed step delta is
    exploitable through drop/reset cycles. Absolute proximity therefore remains
    independent of velocity but is gated by end-effector/object proximity. This
    attenuates the Train5 loophole where throwing the object earned placement
    reward before a later drop. Train8 then entered the strict 5 cm basin without
    arresting motion because fast near-target motion merely lost the positive
    dwell bonus. Train9 made that basin repulsive with a negative fast-motion
    term and regressed medium/hard transport. A smooth settling term now supplies
    a learnable braking gradient before the exact velocity boundary. Departure is
    penalized by the signed potential-progress term below. A bounded floor
    preserves grasp/lift exploration and broad transport remains positive.
    ``tanh`` is injected so this contract is testable without Isaac's Torch runtime.
    """

    approach = 1.0 - tanh(distance / float(approach_std_m))
    near = 1.0 - tanh(distance / float(near_std_m))
    held = float(hold_reward_floor) + (1.0 - float(hold_reward_floor)) * (
        1.0 - tanh(hold_distance / float(hold_std_m))
    )
    settling = 1.0 - tanh(speed / float(settling_speed_mps))
    strict_stillness = 1.0 - tanh(speed / float(stable_speed_mps))
    braking = 0.5 * settling + 0.5 * strict_stillness
    return held * (approach + float(dwell_scale) * near * braking)


def strict_basin_settling_signal(
    distance: Any,
    speed: Any,
    hold_distance: Any,
    *,
    tanh: Any,
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    basin_width_m: float = PLACEMENT_BASIN_WIDTH_M,
    settling_speed_mps: float = PLACEMENT_SETTLING_SPEED_MPS,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
) -> Any:
    """Reward braking only after reaching the unchanged strict target basin.

    Exact c8a1980 validation put every scenario inside 5 cm but none below
    0.03 m/s for three steps. Train11 proved that a negative basin reward teaches
    avoidance instead of braking. This signal is therefore effectively zero
    during broad transport and becomes positive over the final 5 cm of approach,
    with a smooth velocity gradient plus an exact-boundary-focused component. The
    wider positive-only envelope gives a moving policy enough distance to brake;
    it does not alter the strict success boundary. Sustained stillness earns more
    than a fast crossing without making the target repulsive.
    """

    if basin_width_m <= 0 or settling_speed_mps <= 0 or stable_speed_mps <= 0:
        raise ValueError("basin width and settling speeds must be positive")
    basin = 0.5 * (
        1.0 + tanh((float(success_distance_m) - distance) / float(basin_width_m))
    )
    held = float(hold_reward_floor) + (1.0 - float(hold_reward_floor)) * (
        1.0 - tanh(hold_distance / float(hold_std_m))
    )
    speed_signal = 0.5 * (1.0 - tanh(speed / float(settling_speed_mps))) + 0.5 * (
        1.0 - tanh(speed / float(stable_speed_mps))
    )
    return held * basin * speed_signal


def near_goal_arm_stillness_signal(
    distance: Any,
    arm_speed: Any,
    hold_distance: Any,
    *,
    tanh: Any,
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    basin_width_m: float = PLACEMENT_BASIN_WIDTH_M,
    arm_settling_speed_radps: float = PLACEMENT_ARM_SETTLING_SPEED_RADPS,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
) -> Any:
    """Reward a motionless arm near goal without making approach repulsive."""

    if basin_width_m <= 0 or arm_settling_speed_radps <= 0:
        raise ValueError("basin width and arm settling speed must be positive")
    basin = 0.5 * (
        1.0 + tanh((float(success_distance_m) - distance) / float(basin_width_m))
    )
    held = float(hold_reward_floor) + (1.0 - float(hold_reward_floor)) * (
        1.0 - tanh(hold_distance / float(hold_std_m))
    )
    stillness = 1.0 - tanh(arm_speed / float(arm_settling_speed_radps))
    return held * basin * stillness


def _placement_state(
    env: Any,
    *,
    command_name: str,
    object_name: str,
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
) -> tuple[Any, Any, Any, Any]:
    """Return goal distance, speed, position, and end-effector hold distance."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    obj = env.scene[object_name]
    robot = env.scene[robot_name]
    position = obj.data.root_pos_w[:, :3]
    command = env.command_manager.get_command(command_name)
    from isaaclab.utils.math import combine_frame_transforms

    goal, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
    )
    distance = torch.linalg.norm(position - goal, dim=1)
    speed = torch.linalg.norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    ee_position = env.scene[ee_frame_name].data.target_pos_w[..., 0, :]
    hold_distance = torch.linalg.norm(position - ee_position, dim=1)
    return distance, speed, position, hold_distance


def placement_progress_signal(
    previous_distance: float,
    distance: float,
    *,
    progress_scale_m: float = PLACEMENT_PROGRESS_SCALE_M,
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    braking_width_m: float = PLACEMENT_BASIN_WIDTH_M,
) -> float:
    """Return signed progress with positive drive tapered before the goal.

    Negative departure remains fully penalized. Positive approach falls to zero
    at the strict-distance boundary so discounted PPO cannot profit from flying
    through the target before paying the later departure penalty.
    """

    if progress_scale_m <= 0 or braking_width_m <= 0:
        raise ValueError("progress scale and braking width must be positive")
    if not math.isfinite(previous_distance):
        return 0.0
    progress = max(
        -1.0,
        min(1.0, (previous_distance - distance) / float(progress_scale_m)),
    )
    if progress <= 0:
        return progress
    approach_scale = max(
        0.0,
        min(
            1.0,
            (distance - float(success_distance_m)) / float(braking_width_m),
        ),
    )
    return progress * approach_scale


def potential_placement_progress(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    ee_frame_name: str = "ee_frame",
    progress_scale_m: float = PLACEMENT_PROGRESS_SCALE_M,
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    braking_width_m: float = PLACEMENT_BASIN_WIDTH_M,
    hold_max_distance_m: float = PLACEMENT_HOLD_MAX_DISTANCE_M,
    minimal_lift_m: float = PLACEMENT_MINIMAL_LIFT_M,
) -> Any:
    """Reward held-object approach and penalize departure without reset exploits.

    The prior best-distance reward paid for reaching a new closest point but made
    leaving that point free. Live Train8/Train9 validation repeatedly crossed the
    5 cm basin and drifted away without three stable steps. This signed potential
    delta makes approach positive and departure negative. Train14 then showed the
    full positive drive still encouraged high-speed crossings in 49/64 validation
    cases. Positive progress now tapers over the pre-arrival braking envelope and
    reaches zero at the unchanged strict boundary; negative departure remains
    full strength. The first eligible sample after reset or regrasp earns zero,
    so reset/drop cycles cannot synthesize progress. Every step is bounded.
    """

    import torch  # noqa: WPS433 - Isaac runtime dependency

    rows = scenarios_from_env()
    _ensure_runtime_buffers(env, rows)
    obj = env.scene[object_name]
    distance, _, position, hold_distance = _placement_state(
        env,
        command_name=command_name,
        object_name=object_name,
        ee_frame_name=ee_frame_name,
    )
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = position[:, 2] - initial_z >= float(minimal_lift_m)
    eligible = lifted & (hold_distance < float(hold_max_distance_m))
    previous = env.npa_previous_placement_distance
    initialized = torch.isfinite(previous)
    raw_progress = torch.where(
        eligible & initialized,
        torch.clamp((previous - distance) / float(progress_scale_m), min=-1.0, max=1.0),
        torch.zeros_like(distance),
    )
    if braking_width_m <= 0:
        raise ValueError("braking_width_m must be positive")
    approach_scale = torch.clamp(
        (distance - float(success_distance_m)) / float(braking_width_m),
        min=0.0,
        max=1.0,
    )
    progress = torch.where(
        raw_progress > 0,
        raw_progress * approach_scale,
        raw_progress,
    )
    env.npa_previous_placement_distance = torch.where(
        eligible,
        distance,
        torch.full_like(distance, float("inf")),
    )
    return progress


def stable_placement_dwell_signal(
    is_stable: bool,
    previous_steps: int,
    *,
    required_steps: int = STABLE_PLACEMENT_STEPS,
    reward_exponent: float = PLACEMENT_DWELL_REWARD_EXPONENT,
) -> tuple[int, float]:
    """Advance the exact strict-event dwell with positive-only progression."""

    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if reward_exponent <= 0:
        raise ValueError("reward_exponent must be positive")
    steps = min(int(required_steps), int(previous_steps) + 1) if is_stable else 0
    fraction = float(steps) / float(required_steps)
    return steps, fraction ** float(reward_exponent)


def stable_placement_retention_signal(
    is_stable: bool,
    previous_steps: int,
    achieved: bool,
    *,
    required_steps: int = STABLE_PLACEMENT_STEPS,
) -> tuple[int, float, bool, float]:
    """Advance positive-only dwell and expose one-shot completion.

    Returns ``(steps, dwell, achieved, completion)``. Completion is one only on
    the first exact three-step event. Later unstable steps earn zero here; they
    do not make the earlier success return negative.
    """

    steps, dwell = stable_placement_dwell_signal(
        is_stable, previous_steps, required_steps=required_steps
    )
    newly_achieved = bool(is_stable and steps >= required_steps and not achieved)
    achieved_now = bool(achieved or newly_achieved)
    return steps, dwell, achieved_now, float(newly_achieved)


def stable_placement_dwell(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
    required_steps: int = STABLE_PLACEMENT_STEPS,
    reward_exponent: float = PLACEMENT_DWELL_REWARD_EXPONENT,
) -> Any:
    """Reward quadratic strict dwell progression, resetting on a miss."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if reward_exponent <= 0:
        raise ValueError("reward_exponent must be positive")
    rows = scenarios_from_env()
    _ensure_runtime_buffers(env, rows)
    distance, speed, position, _ = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    obj = env.scene[object_name]
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = position[:, 2] - initial_z >= float(PLACEMENT_MINIMAL_LIFT_M)
    stable = (
        lifted
        & (distance < float(success_distance_m))
        & (speed < float(stable_speed_mps))
    )
    previous_steps = env.npa_stable_placement_reward_steps
    next_steps = torch.where(
        stable,
        torch.clamp(
            previous_steps + 1,
            max=int(required_steps),
        ),
        torch.zeros_like(previous_steps),
    )
    newly_achieved = (
        stable
        & (next_steps >= int(required_steps))
        & ~env.npa_stable_placement_achieved
    )
    env.npa_stable_placement_reward_steps = next_steps
    env.npa_stable_placement_newly_achieved = newly_achieved
    env.npa_stable_placement_achieved |= newly_achieved
    fraction = env.npa_stable_placement_reward_steps.to(position.dtype) / float(
        required_steps
    )
    return fraction ** float(reward_exponent)


def stable_placement_completion(env: Any) -> Any:
    """Reward the first exact dwell event without ending the episode."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    return env.npa_stable_placement_newly_achieved.to(dtype=torch.float32)


def strict_basin_settling(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    basin_width_m: float = PLACEMENT_BASIN_WIDTH_M,
    settling_speed_mps: float = PLACEMENT_SETTLING_SPEED_MPS,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
    minimal_lift_m: float = PLACEMENT_MINIMAL_LIFT_M,
) -> Any:
    """Apply the strict-basin braking signal only after a genuine lift."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    obj = env.scene[object_name]
    distance, speed, position, hold_distance = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = (position[:, 2] - initial_z >= float(minimal_lift_m)).to(position.dtype)
    signal = strict_basin_settling_signal(
        distance,
        speed,
        hold_distance,
        tanh=torch.tanh,
        success_distance_m=success_distance_m,
        basin_width_m=basin_width_m,
        settling_speed_mps=settling_speed_mps,
        stable_speed_mps=stable_speed_mps,
        hold_std_m=hold_std_m,
        hold_reward_floor=hold_reward_floor,
    )
    return lifted * signal


def near_goal_arm_stillness(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    basin_width_m: float = PLACEMENT_BASIN_WIDTH_M,
    arm_settling_speed_radps: float = PLACEMENT_ARM_SETTLING_SPEED_RADPS,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
    minimal_lift_m: float = PLACEMENT_MINIMAL_LIFT_M,
) -> Any:
    """Reward low RMS arm-joint speed near goal after a genuine lift."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    obj = env.scene[object_name]
    distance, _, position, hold_distance = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    try:
        arm_term = env.action_manager.get_term("arm_action")
        joint_velocity = arm_term._asset.data.joint_vel[:, arm_term._joint_ids]
    except Exception as exc:
        raise RuntimeError(
            "near-goal arm stillness requires the live arm_action joint mapping"
        ) from exc
    if joint_velocity.ndim != 2 or joint_velocity.shape[1] <= 0:
        raise RuntimeError("near-goal arm stillness received invalid joint velocity")
    arm_speed = torch.sqrt(torch.mean(torch.square(joint_velocity), dim=1))
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = (position[:, 2] - initial_z >= float(minimal_lift_m)).to(position.dtype)
    signal = near_goal_arm_stillness_signal(
        distance,
        arm_speed,
        hold_distance,
        tanh=torch.tanh,
        success_distance_m=success_distance_m,
        basin_width_m=basin_width_m,
        arm_settling_speed_radps=arm_settling_speed_radps,
        hold_std_m=hold_std_m,
        hold_reward_floor=hold_reward_floor,
    )
    return lifted * signal


def stable_placement_curriculum(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    approach_std_m: float = PLACEMENT_APPROACH_STD_M,
    near_std_m: float = PLACEMENT_NEAR_STD_M,
    hold_std_m: float = PLACEMENT_HOLD_STD_M,
    hold_reward_floor: float = PLACEMENT_HOLD_REWARD_FLOOR,
    dwell_scale: float = PLACEMENT_DWELL_SCALE,
    settling_speed_mps: float = PLACEMENT_SETTLING_SPEED_MPS,
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
    minimal_lift_m: float = PLACEMENT_MINIMAL_LIFT_M,
) -> Any:
    """Shape the last mile without changing the strict placement verdict.

    The stock Lift task rewards proximity to the goal but does not explicitly
    reward arresting object motion. Real attempt evidence consequently learned
    reach, grasp, and lift while repeatedly dropping or carrying through the
    target. This term activates only after a genuine lift and combines:

    * broad absolute proximity that remains active during transport;
    * a narrow near-goal stillness signal; and
    * a sparse bonus at the exact 5 cm / 0.03 m/s stable-placement boundary.

    The authoritative evaluator remains the source of success and still requires
    three consecutive stable steps; this is curriculum, not a relaxed metric.
    """

    import torch  # noqa: WPS433 - Isaac runtime dependency

    obj = env.scene[object_name]
    distance, speed, position, hold_distance = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = (position[:, 2] - initial_z >= float(minimal_lift_m)).to(position.dtype)
    dense = placement_curriculum_signal(
        distance,
        speed,
        hold_distance,
        tanh=torch.tanh,
        approach_std_m=approach_std_m,
        near_std_m=near_std_m,
        hold_std_m=hold_std_m,
        hold_reward_floor=hold_reward_floor,
        dwell_scale=dwell_scale,
        settling_speed_mps=settling_speed_mps,
        stable_speed_mps=stable_speed_mps,
    )
    strict = (
        (distance < float(success_distance_m)) & (speed < float(stable_speed_mps))
    ).to(position.dtype)
    return lifted * (dense + strict)


def stable_placement_success(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    success_distance_m: float = STABLE_PLACEMENT_DISTANCE_M,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
    required_steps: int = STABLE_PLACEMENT_STEPS,
) -> Any:
    """Terminate training after the evaluator's exact stable-placement event."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    rows = scenarios_from_env()
    _ensure_runtime_buffers(env, rows)
    distance, speed, _, _ = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    stable = (distance < float(success_distance_m)) & (speed < float(stable_speed_mps))
    env.npa_stable_placement_steps = torch.where(
        stable,
        env.npa_stable_placement_steps + 1,
        torch.zeros_like(env.npa_stable_placement_steps),
    )
    return env.npa_stable_placement_steps >= int(required_steps)


def install_env_cfg(env_cfg: Any) -> bool:
    """Install exact scenario reset/goal terms into a Lift env config."""

    rows = scenarios_from_env()
    if not rows:
        return False
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import RewardTermCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    env_cfg.events.reset_object_position = EventTerm(
        func=apply_scenario_reset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    env_cfg.commands.object_pose.class_type = _scenario_command_type()
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # The stock task's sparse lift reward did not teach the deterministic actor
    # mean to close its gripper on the curated distribution. Reuse the same
    # embodiment-aware shaping functions as BYO robots, with the stock Franka
    # finger contract made explicit. Both terms are precursors only; strict
    # placement remains the unchanged object-space verdict below.
    import isaac_byo_robot_task as robot_task  # noqa: WPS433 - baked runtime module

    env_cfg.rewards.grasp_closure_curriculum = RewardTermCfg(
        func=robot_task.grasp_shaping,
        weight=GRASP_CLOSURE_REWARD_WEIGHT,
        params={
            "std": GRASP_CLOSURE_STD_M,
            "object_name": "object",
            "ee_frame_name": "ee_frame",
            "gripper_joint_names": STOCK_GRIPPER_JOINT_NAMES,
            "gripper_open": STOCK_GRIPPER_OPEN_POSITION,
            "gripper_close": STOCK_GRIPPER_CLOSED_POSITION,
        },
    )
    env_cfg.rewards.grasp_lift_attempt_curriculum = RewardTermCfg(
        func=robot_task.grasp_lift_hold,
        weight=GRASP_LIFT_ATTEMPT_REWARD_WEIGHT,
        params={
            "std": GRASP_LIFT_ATTEMPT_STD_M,
            "object_name": "object",
            "ee_frame_name": "ee_frame",
            "gripper_joint_names": STOCK_GRIPPER_JOINT_NAMES,
            "gripper_open": STOCK_GRIPPER_OPEN_POSITION,
            "gripper_close": STOCK_GRIPPER_CLOSED_POSITION,
        },
    )
    env_cfg.rewards.dense_object_lift_curriculum = RewardTermCfg(
        func=robot_task.object_lift_progress,
        weight=STOCK_DENSE_LIFT_REWARD_WEIGHT,
        params={
            "std": STOCK_DENSE_LIFT_STD_M,
            "object_name": "object",
        },
    )
    env_cfg.rewards.stable_placement_curriculum = RewardTermCfg(
        func=stable_placement_curriculum,
        weight=STABLE_PLACEMENT_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "approach_std_m": PLACEMENT_APPROACH_STD_M,
            "near_std_m": PLACEMENT_NEAR_STD_M,
            "hold_std_m": PLACEMENT_HOLD_STD_M,
            "hold_reward_floor": PLACEMENT_HOLD_REWARD_FLOOR,
            "dwell_scale": PLACEMENT_DWELL_SCALE,
            "settling_speed_mps": PLACEMENT_SETTLING_SPEED_MPS,
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
            "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
        },
    )
    env_cfg.rewards.potential_placement_progress = RewardTermCfg(
        func=potential_placement_progress,
        weight=PLACEMENT_PROGRESS_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "ee_frame_name": "ee_frame",
            "progress_scale_m": PLACEMENT_PROGRESS_SCALE_M,
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "braking_width_m": PLACEMENT_BASIN_WIDTH_M,
            "hold_max_distance_m": PLACEMENT_HOLD_MAX_DISTANCE_M,
            "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
        },
    )
    env_cfg.rewards.strict_basin_settling = RewardTermCfg(
        func=strict_basin_settling,
        weight=PLACEMENT_BASIN_SETTLING_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "basin_width_m": PLACEMENT_BASIN_WIDTH_M,
            "settling_speed_mps": PLACEMENT_SETTLING_SPEED_MPS,
            "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
            "hold_std_m": PLACEMENT_HOLD_STD_M,
            "hold_reward_floor": PLACEMENT_HOLD_REWARD_FLOOR,
            "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
        },
    )
    env_cfg.rewards.near_goal_arm_stillness = RewardTermCfg(
        func=near_goal_arm_stillness,
        weight=PLACEMENT_ARM_STILLNESS_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "basin_width_m": PLACEMENT_BASIN_WIDTH_M,
            "arm_settling_speed_radps": PLACEMENT_ARM_SETTLING_SPEED_RADPS,
            "hold_std_m": PLACEMENT_HOLD_STD_M,
            "hold_reward_floor": PLACEMENT_HOLD_REWARD_FLOOR,
            "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
        },
    )
    env_cfg.rewards.stable_placement_dwell = RewardTermCfg(
        func=stable_placement_dwell,
        weight=PLACEMENT_STRICT_DWELL_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
            "required_steps": STABLE_PLACEMENT_STEPS,
            "reward_exponent": PLACEMENT_DWELL_REWARD_EXPONENT,
        },
    )
    env_cfg.rewards.stable_placement_completion = RewardTermCfg(
        func=stable_placement_completion,
        weight=PLACEMENT_COMPLETION_REWARD_WEIGHT,
    )

    env_cfg.rewards.object_drop_penalty = RewardTermCfg(
        func=_scheduled_drop_penalty_type(),
        weight=PLACEMENT_DROP_PENALTY_WEIGHT,
        params={
            "term_keys": "object_dropping",
            "full_goal_step": int(
                os.environ.get("NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP", "1") or 1
            ),
        },
    )
    success_termination_enabled = (
        os.environ.get("NPA_SIM2REAL_ENABLE_SUCCESS_TERMINATION", "0") == "1"
    )
    if success_termination_enabled:
        env_cfg.terminations.stable_placement_success = DoneTerm(
            func=stable_placement_success,
            params={
                "command_name": "object_pose",
                "object_name": "object",
                "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
                "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
                "required_steps": STABLE_PLACEMENT_STEPS,
            },
        )
    # The light is global across cloned envs.  All curated records use this exact
    # value, so setting it once is an applied contract, not a per-env label.
    if hasattr(env_cfg.scene.light.spawn, "intensity"):
        env_cfg.scene.light.spawn.intensity = EXPECTED_LIGHT_INTENSITY
    try:
        env_cfg.scene.object.spawn.activate_contact_sensors = True
        from isaaclab.sensors import ContactSensorCfg

        env_cfg.scene.object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            history_length=3,
            track_air_time=True,
        )
    except Exception as exc:  # noqa: BLE001 - eval records the availability
        print("SCENARIO_CONTACT_SENSOR_SKIP", repr(exc), flush=True)
    print(
        "SCENARIO_CFG_INSTALLED",
        json.dumps(
            {
                **scenario_contract_summary(rows),
                "grasp_curriculum": {
                    "closure_reward_weight": GRASP_CLOSURE_REWARD_WEIGHT,
                    "closure_std_m": GRASP_CLOSURE_STD_M,
                    "lift_attempt_reward_weight": GRASP_LIFT_ATTEMPT_REWARD_WEIGHT,
                    "lift_attempt_std_m": GRASP_LIFT_ATTEMPT_STD_M,
                    "gripper_joint_names": list(STOCK_GRIPPER_JOINT_NAMES),
                    "gripper_open_position": STOCK_GRIPPER_OPEN_POSITION,
                    "gripper_closed_position": STOCK_GRIPPER_CLOSED_POSITION,
                },
                "placement_curriculum": {
                    "weight": STABLE_PLACEMENT_REWARD_WEIGHT,
                    "approach_std_m": PLACEMENT_APPROACH_STD_M,
                    "near_std_m": PLACEMENT_NEAR_STD_M,
                    "hold_std_m": PLACEMENT_HOLD_STD_M,
                    "hold_reward_floor": PLACEMENT_HOLD_REWARD_FLOOR,
                    "hold_max_distance_m": PLACEMENT_HOLD_MAX_DISTANCE_M,
                    "dwell_scale": PLACEMENT_DWELL_SCALE,
                    "progress_scale_m": PLACEMENT_PROGRESS_SCALE_M,
                    "progress_reward_weight": PLACEMENT_PROGRESS_REWARD_WEIGHT,
                    "arm_settling_speed_radps": (PLACEMENT_ARM_SETTLING_SPEED_RADPS),
                    "arm_stillness_reward_weight": (
                        PLACEMENT_ARM_STILLNESS_REWARD_WEIGHT
                    ),
                    "strict_dwell_reward_weight": (
                        PLACEMENT_STRICT_DWELL_REWARD_WEIGHT
                    ),
                    "strict_dwell_reward_exponent": (PLACEMENT_DWELL_REWARD_EXPONENT),
                    "goal_curriculum_enabled": os.environ.get(
                        "NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0"
                    )
                    == "1",
                    "goal_curriculum_full_step": int(
                        os.environ.get("NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP", "1")
                        or 1
                    ),
                    "goal_curriculum_lift_m": PLACEMENT_GOAL_CURRICULUM_LIFT_M,
                    "drop_penalty_weight": PLACEMENT_DROP_PENALTY_WEIGHT,
                    "completion_reward_weight": (PLACEMENT_COMPLETION_REWARD_WEIGHT),
                    "strict_distance_m": STABLE_PLACEMENT_DISTANCE_M,
                    "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
                    "stable_steps": STABLE_PLACEMENT_STEPS,
                    "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
                    "success_termination_enabled": success_termination_enabled,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return True


def register_stock() -> str | None:
    """Register a stock-Franka task variant that consumes curated scenarios."""

    rows = scenarios_from_env()
    if not rows:
        return None
    import gymnasium as gym
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka import (
        joint_pos_env_cfg as franka_lift,
    )

    class ScenarioLiftEnvCfg(franka_lift.FrankaCubeLiftEnvCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            if not install_env_cfg(self):
                raise ScenarioContractError(
                    "scenario task registered without scenarios"
                )

    stock_kwargs = gym.spec(STOCK_TASK_ID).kwargs
    gym.register(
        id=SCENARIO_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": ScenarioLiftEnvCfg,
            "rsl_rl_cfg_entry_point": stock_kwargs.get("rsl_rl_cfg_entry_point"),
        },
    )
    return SCENARIO_TASK_ID


def runtime_audit(env: Any) -> dict[str, Any]:
    rows = getattr(env, "npa_scenario_rows", scenarios_from_env())
    counts_tensor = getattr(env, "npa_scenario_applied_counts", None)
    counts = counts_tensor.detach().cpu().tolist() if counts_tensor is not None else []
    seen = sum(1 for count in counts if int(count) > 0)
    return {
        "schema": "npa.sim2real.applied_scenarios.v1",
        **scenario_contract_summary(rows),
        "applied_unique_config_digests": seen,
        "coverage_rate": round(seen / len(rows), 6) if rows else 0.0,
        "total_reset_assignments": sum(int(count) for count in counts),
        "goal_curriculum": {
            "enabled": os.environ.get("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0")
            == "1",
            "full_goal_step": int(
                os.environ.get("NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP", "1") or 1
            ),
            "assignment_count": int(
                getattr(env, "npa_goal_curriculum_assignment_count", 0)
            ),
            "true_goal_assignments": int(
                getattr(env, "npa_goal_curriculum_true_assignments", 0)
            ),
            "max_fraction": round(
                float(getattr(env, "npa_goal_curriculum_max_fraction", 0.0)), 6
            ),
        },
        "records": [
            {
                "env_id": row["env_id"],
                "seed": row["seed"],
                "scenario_config_digest": row["scenario_config_digest"],
                "applied_count": int(counts[index]) if index < len(counts) else 0,
            }
            for index, row in enumerate(rows)
        ],
    }


def module_source() -> str:
    return Path(__file__).read_text(encoding="utf-8")

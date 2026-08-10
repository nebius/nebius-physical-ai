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
# The first validation canary learned reach/contact and 2/3 grasp+lift, but the
# closest goal distance remained 0.205-0.364 m.  A 0.15 m tanh scale multiplied
# by weight 8 is effectively flat at that boundary, so lifting remains the much
# easier dominant objective.  Keep the strict verdict at 5 cm while making the
# curriculum genuinely dense across the observed post-lift placement basin.
PLACEMENT_APPROACH_STD_M = 0.35
PLACEMENT_NEAR_STD_M = 0.08
PLACEMENT_DWELL_SCALE = 2.0
PLACEMENT_GOAL_CURRICULUM_LIFT_M = 0.08
PLACEMENT_DROP_PENALTY_WEIGHT = -50.0
STABLE_PLACEMENT_REWARD_WEIGHT = 32.0
PLACEMENT_MINIMAL_LIFT_M = 0.04
_COMMAND_TYPE: type | None = None
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


def _ensure_runtime_buffers(env: Any, rows: list[dict[str, Any]]) -> None:
    import torch

    if not hasattr(env, "npa_scenario_indices"):
        env.npa_scenario_indices = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env.npa_scenario_episode_counts = torch.zeros_like(env.npa_scenario_indices)
        env.npa_scenario_applied_counts = torch.zeros(
            len(rows), dtype=torch.long, device=env.device
        )
        env.npa_scenario_rows = rows
    if not hasattr(env, "npa_stable_placement_steps"):
        env.npa_stable_placement_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )


def _assign(env: Any, env_ids: Any, rows: list[dict[str, Any]]) -> Any:
    import torch

    _ensure_runtime_buffers(env, rows)
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    rotate = os.environ.get("NPA_SIM2REAL_SCENARIO_ROTATE_ON_RESET", "1") != "0"
    offset = int(os.environ.get("NPA_SIM2REAL_SCENARIO_OFFSET", "0") or 0)
    if rotate:
        # A prime stride makes complete passes through non-prime split sizes while
        # different vector env indices cover adjacent records in parallel.
        scenario_ids = (
            ids + offset + env.npa_scenario_episode_counts[ids] * 104729
        ) % len(rows)
        env.npa_scenario_episode_counts[ids] += 1
    else:
        scenario_ids = (ids + offset) % len(rows)
    env.npa_scenario_indices[ids] = scenario_ids
    env.npa_stable_placement_steps[ids] = 0
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
    *,
    tanh: Any,
    approach_std_m: float = PLACEMENT_APPROACH_STD_M,
    near_std_m: float = PLACEMENT_NEAR_STD_M,
    dwell_scale: float = PLACEMENT_DWELL_SCALE,
    stable_speed_mps: float = STABLE_PLACEMENT_SPEED_MPS,
) -> Any:
    """Return unsuppressed proximity plus a near-target dwell incentive.

    Train3 exposed that velocity-gating the whole approach signal suppresses
    deliberate transport. Train4 then showed that a signed step delta is
    exploitable through drop/reset cycles. Absolute proximity is therefore kept
    dense and independent of velocity, while stillness is valuable only in the
    narrow target basin. ``tanh`` is injected so this contract is testable with
    scalar math without importing Isaac's Torch runtime.
    """

    approach = 1.0 - tanh(distance / float(approach_std_m))
    near = 1.0 - tanh(distance / float(near_std_m))
    strict_stillness = 1.0 - tanh(speed / float(stable_speed_mps))
    return approach + float(dwell_scale) * near * strict_stillness


def _placement_state(
    env: Any, *, command_name: str, object_name: str
) -> tuple[Any, Any, Any]:
    """Return object distance, linear speed, and position for placement terms."""

    import torch  # noqa: WPS433 - Isaac runtime dependency

    obj = env.scene[object_name]
    position = obj.data.root_pos_w[:, :3]
    goal = (
        env.command_manager.get_command(command_name)[:, :3]
        + env.scene.env_origins[:, :3]
    )
    distance = torch.linalg.norm(position - goal, dim=1)
    speed = torch.linalg.norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    return distance, speed, position


def stable_placement_curriculum(
    env: Any,
    *,
    command_name: str = "object_pose",
    object_name: str = "object",
    approach_std_m: float = PLACEMENT_APPROACH_STD_M,
    near_std_m: float = PLACEMENT_NEAR_STD_M,
    dwell_scale: float = PLACEMENT_DWELL_SCALE,
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
    distance, speed, position = _placement_state(
        env, command_name=command_name, object_name=object_name
    )
    initial_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    lifted = (position[:, 2] - initial_z >= float(minimal_lift_m)).to(position.dtype)
    dense = placement_curriculum_signal(
        distance,
        speed,
        tanh=torch.tanh,
        approach_std_m=approach_std_m,
        near_std_m=near_std_m,
        dwell_scale=dwell_scale,
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
    distance, speed, _ = _placement_state(
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
    env_cfg.rewards.stable_placement_curriculum = RewardTermCfg(
        func=stable_placement_curriculum,
        weight=STABLE_PLACEMENT_REWARD_WEIGHT,
        params={
            "command_name": "object_pose",
            "object_name": "object",
            "approach_std_m": PLACEMENT_APPROACH_STD_M,
            "near_std_m": PLACEMENT_NEAR_STD_M,
            "dwell_scale": PLACEMENT_DWELL_SCALE,
            "success_distance_m": STABLE_PLACEMENT_DISTANCE_M,
            "stable_speed_mps": STABLE_PLACEMENT_SPEED_MPS,
            "minimal_lift_m": PLACEMENT_MINIMAL_LIFT_M,
        },
    )
    from isaaclab.envs import mdp

    env_cfg.rewards.object_drop_penalty = RewardTermCfg(
        func=mdp.is_terminated_term,
        weight=PLACEMENT_DROP_PENALTY_WEIGHT,
        params={"term_keys": "object_dropping"},
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
                "placement_curriculum": {
                    "weight": STABLE_PLACEMENT_REWARD_WEIGHT,
                    "approach_std_m": PLACEMENT_APPROACH_STD_M,
                    "near_std_m": PLACEMENT_NEAR_STD_M,
                    "dwell_scale": PLACEMENT_DWELL_SCALE,
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

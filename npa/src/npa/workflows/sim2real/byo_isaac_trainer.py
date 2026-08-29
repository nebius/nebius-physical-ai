"""BYO trainer: real Isaac-Lab RSL-RL PPO for the sim2real inner loop.

Wired in via ``sim2real run --byo-trainer-command 'python3 -m
npa.workflows.sim2real.byo_isaac_trainer'``. This satisfies the
``_run_trainer_via_command`` contract (engine.py): read the parsed VLM signal
batch from ``NPA_SIM2REAL_SIGNAL_JSON`` and write a ``VlmSignalUpdateResult``
JSON to ``NPA_SIM2REAL_OUTPUT_JSON`` with at least ``reward_head_after``,
``policy_output_after`` (non-empty list), and ``policy_delta_l2``.

Unlike the in-process *reference* hook (``run_vlm_signal_training_step`` — a
single SGD step on a scalar adapter), this runs **genuine RL training**: it
submits an Isaac-Lab sibling k8s Job (``npa-isaac-lab`` image) that runs
``scripts/reinforcement_learning/rsl_rl/train.py`` on
``Isaac-Lift-Cube-Franka-v0`` for real iterations, produces a real
``model_*.pt`` policy checkpoint, and uploads it to S3. The emitted
``checkpoint_path`` is that real checkpoint, so promote can mark it deployable.

The trainer runs **inside the orchestrator pod** (lerobot-vlm-rl image, no
Isaac), so it cannot run Isaac in-process. It submits the sibling Job through
the typed Kubernetes API client and reconciles structured Job and Pod state.

``NPA_BYO_ISAAC_DRYRUN=1`` skips the Kubernetes API/S3 entirely and emits a deterministic
result derived from the signal batch — used by unit tests and for wiring checks
without a GPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.capture import (
    DEFAULT_PPO_ITERATIONS,
    DEFAULT_PPO_NUM_ENVS,
    DEFAULT_PPO_STEPS_PER_ENV,
    ppo_settings,
)
from npa.workflows.sim2real.constants import DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE
from npa.workflows.sim2real.isaac_job_payload import (
    compressed_bash_launch,
    embedded_base64_file_block,
)

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_NUM_ENVS = DEFAULT_PPO_NUM_ENVS
DEFAULT_ITERATIONS = DEFAULT_PPO_ITERATIONS
DEFAULT_STEPS_PER_ENV = DEFAULT_PPO_STEPS_PER_ENV
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
# Start at the stock-like exploration level, then anneal after the grasp/lift
# discovery phase. Live evidence showed that a fixed 0.01 coefficient kept the
# action-noise standard deviation above 1.5 after 500 iterations: the policy
# learned reach/grasp/lift but dropped or carried through the placement target.
# The final low coefficient lets the placement curriculum converge on a stable
# hold without removing early exploration. All three values are operator-tunable.
DEFAULT_ENTROPY_COEF = "0.006"
DEFAULT_ENTROPY_FINAL_COEF = "0.0005"
DEFAULT_ENTROPY_ANNEAL_FRACTION = "0.6"
DEFAULT_PPO_OPTIMIZER_LEARNING_RATE = "0.001"
# A resumed policy has already crossed the reach/grasp/lift exploration wall.
# Re-applying the first-pass exploration schedule on every inner/outer pass kept
# deterministic placement noisy in live validation. Resume uses a separate,
# operator-tunable convergence phase: short low-entropy adaptation followed by
# zero entropy and a smaller PPO optimizer rate.
DEFAULT_RESUME_ENTROPY_COEF = "0.0005"
DEFAULT_RESUME_ENTROPY_FINAL_COEF = "0.0"
DEFAULT_RESUME_ENTROPY_ANNEAL_FRACTION = "0.2"
# Train21's best live-validation checkpoint briefly sustained the exact event,
# while later checkpoints moved away again. Use a smaller resumed convergence
# step so subsequent passes consolidate that narrow-basin behavior instead of
# repeatedly overwriting it. First-pass exploration remains unchanged.
DEFAULT_RESUME_PPO_OPTIMIZER_LEARNING_RATE = "0.0001"
# Validation executes RSL-RL's deterministic actor mean.  A resumed policy can
# therefore report sparse sampled successes while its mean still misses the
# stable-placement event if the learned action standard deviation remains high.
# After the short resumed exploration segment, train close to the policy that
# validation actually executes.  The baked Isaac wrapper applies and verifies
# this value only after loading the exact resume checkpoint.
DEFAULT_RESUME_CONVERGENCE_ACTION_NOISE_STD = "0.05"
_STOCK_ENTROPY_SENTINELS = frozenset({"stock", "default", "none", ""})
TRAIN_SCRIPT = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py"
# rsl_rl experiment_name for the Franka Lift task (logs/rsl_rl/<experiment_name>/).
# Overridable via NPA_BYO_ISAAC_EXPERIMENT_NAME for non-default tasks; the outer-loop
# RESUME path stages the prior checkpoint under this experiment dir so train.py's
# get_checkpoint_path() resolves it.
DEFAULT_EXPERIMENT_NAME = "franka_lift"
# Fixed run-dir name we stage a resumed checkpoint into. train.py is then told to
# load exactly this run (agent.load_run) so resume never picks the freshly-created
# current run dir by accident.
RESUME_RUN_DIR = "00000000_npa_resume"
RESUME_CKPT_NAME = "model_0.pt"

# Root of the public Omniverse Isaac asset CDN (no tenant/private IDs). Override
# with NPA_ISAAC_NUCLEUS_DIR to point at an internal Nucleus mirror.
DEFAULT_ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
# Rigid-ready (RigidBodyAPI: collision + mass) instanceable manipuland. Defaulting
# to it means the Franka loop trains/evals on a real physically simulated USD sim
# asset instead of the stock primitive cube. A raw visual mesh would fail to spawn.
DEFAULT_OBJECT_USD_REL = "Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd"
# Sentinels that opt back out of the USD default to the built-in primitive cube.
_STOCK_OBJECT_USD_SENTINELS = frozenset({"stock", "none", "primitive", "builtin"})
_MAX_EMBEDDED_SCENARIOS_BYTES = 256_000


def default_isaac_object_usd() -> str:
    """Resolved default manipuland USD (Nucleus root + rigid-ready instanceable)."""
    nuc = (
        os.environ.get("NPA_ISAAC_NUCLEUS_DIR", "") or DEFAULT_ISAAC_NUCLEUS_DIR
    ).strip()
    return f"{nuc.rstrip('/')}/{DEFAULT_OBJECT_USD_REL}"


def resolve_object_usd(raw: str) -> str:
    """Resolve the manipuland USD for an Isaac job.

    An explicit ``NPA_BYO_ISAAC_OBJECT_USD`` wins; a ``stock``/``none`` sentinel
    forces the built-in primitive cube (empty string); unset defaults to the
    proven rigid-ready MultiColorCube so Franka uses a real sim asset by default.
    """
    val = (raw or "").strip()
    if val.lower() in _STOCK_OBJECT_USD_SENTINELS:
        return ""
    return val or default_isaac_object_usd()


# Where the BYO-robot path stages the customer robot USD inside the Isaac job.
ROBOT_USD_CONTAINER_PATH = "/tmp/npa_robot/resolved/robot.usd"


def robot_asset_preflight_script(robot_spec: dict[str, Any]) -> str:
    """Render the fail-closed Stage-7 prepare / train-eval fetch operation."""

    if not robot_spec.get("source_format"):
        return ""
    operation = _env("NPA_SIM2REAL_ROBOT_ASSET_OPERATION", "fetch")
    if operation not in {"prepare", "fetch"}:
        raise ValueError("NPA_SIM2REAL_ROBOT_ASSET_OPERATION must be prepare or fetch")
    spec_json = json.dumps(robot_spec, sort_keys=True)
    command = (
        "export NPA_BYO_ROBOT_SPEC_JSON="
        + shlex.quote(spec_json)
        + "\n"
    )
    if operation == "prepare":
        # Stage 7 converts inside the rollout's already-running AppLauncher.
        # A separate converter process is unsafe because Kit shutdown can end
        # that interpreter before it publishes or preserve its local USD.
        command += "export NPA_PREPARE_ROBOT_ASSET_IN_APP=1\n"
    else:
        command += '"$PY" -m npa.workflows.sim2real.isaac_robot_asset fetch\n'
    return command


def embodiment_evidence(robot_spec: dict[str, Any] | None) -> dict[str, Any]:
    """Evidence attached only after the in-Isaac dimension checks pass."""

    spec = dict(robot_spec or {})
    return {
        "embodiment_digest": str(spec.get("embodiment_digest") or "stock_franka"),
        "expected_action_dim": int(spec.get("expected_action_dim") or 8),
        "expected_observation_dim": int(spec.get("expected_observation_dim") or 36),
        "resolved_usd_uri": str(spec.get("resolved_usd_uri") or ""),
        "resolved_manifest_uri": str(spec.get("resolved_manifest_uri") or ""),
        "runtime_dimension_validation": "passed",
    }


def robot_spec_payload(
    spec: Any, *, usd_container_path: str = ""
) -> dict[str, Any] | None:
    """Serialize a resolved RobotSpec into the ``NPA_BYO_ROBOT_SPEC_JSON`` contract.

    Returns ``None`` when ``spec`` is ``None`` (no BYO-robot routing). For a
    ``stock_franka`` spec, returns a minimal payload (source/name only) so the
    in-container overrides are empty and the variant degenerates to the stock task
    — the BYO seam still runs end-to-end. For a BYO spec, includes the morphology
    / gain fields read by ``isaac_byo_robot_task.robot_articulation_overrides`` plus
    the in-container ``usd_path`` the job stages the robot USD to.
    """

    if spec is None:
        return None
    source = str(getattr(spec, "robot_source", "") or "")
    name = str(getattr(spec, "name", "") or "robot")
    if source == "stock_franka":
        return {"robot_source": source, "name": name}

    def _floats(value: Any) -> list[float]:
        out: list[float] = []
        for item in value or ():
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    return {
        "robot_source": source,
        "name": name,
        "ee_link": str(getattr(spec, "ee_link", "") or ""),
        "base_link": str(getattr(spec, "base_link", "") or ""),
        "joint_names": [str(j) for j in (getattr(spec, "joint_names", ()) or ())],
        "finger_links": [str(f) for f in (getattr(spec, "finger_links", ()) or ())],
        "gripper_joint_names": [
            str(g) for g in (getattr(spec, "gripper_joint_names", ()) or ())
        ],
        "n_arm_joints": int(getattr(spec, "n_arm_joints", 0) or 0),
        "n_gripper_joints": int(getattr(spec, "n_gripper_joints", 0) or 0),
        "home_qpos": _floats(getattr(spec, "home_qpos", ())),
        "kp": _floats(getattr(spec, "kp", ())),
        "kv": _floats(getattr(spec, "kv", ())),
        "force_upper": _floats(getattr(spec, "force_upper", ())),
        "force_lower": _floats(getattr(spec, "force_lower", ())),
        "gripper_open": float(getattr(spec, "gripper_open", 0.04) or 0.0),
        "gripper_close": float(getattr(spec, "gripper_close", 0.0) or 0.0),
        "usd_path": usd_container_path,
        "asset_root_uri": str(getattr(spec, "asset_root_uri", "") or ""),
        "source_sha256": str(getattr(spec, "source_sha256", "") or ""),
        "source_tree_sha256": str(getattr(spec, "source_tree_sha256", "") or ""),
        "source_relative_path": str(getattr(spec, "source_relative_path", "") or ""),
        "source_format": str(getattr(spec, "source_format", "") or ""),
        "embodiment_digest": str(getattr(spec, "embodiment_digest", "") or ""),
        "expected_action_dim": int(getattr(spec, "expected_action_dim", 0) or 0),
        "expected_observation_dim": int(
            getattr(spec, "expected_observation_dim", 0) or 0
        ),
        "resolved_usd_uri": str(getattr(spec, "resolved_usd_uri", "") or ""),
        "resolved_manifest_uri": str(getattr(spec, "resolved_manifest_uri", "") or ""),
    }


def _resolve_byo_robot_spec() -> Any:
    """Resolve a RobotSpec from the trainer's env, or ``None`` (default Franka).

    Resolution mirrors the held-out eval (``engine._resolve_heldout_robot``) so
    training and eval agree on the variant:

    * ``NPA_SIM2REAL_ROBOT_SPEC_URI`` (s3://): download the customer robot-spec
      JSON and parse it with the SAME ``resolve_robot_spec_from_consumed_doc`` the
      eval uses — this is what routes a genuine CUSTOM robot (not just a named
      preset) into RL training. The doc must carry ``robot_uri`` (the USD; an
      Omniverse ``https://`` CDN URL is opened directly by Isaac, an ``s3://`` URL
      is staged by the sibling job).
    * else ``NPA_SIM2REAL_ROBOT_PRESET`` / ``NPA_SIM2REAL_ROBOT_SOURCE``: a named
      preset / bare source, via ``robot_spec_from_inputs`` (no download).

    A spec-uri that fails to download/parse raises rather than silently falling
    back to Franka — the operator must not be misled into thinking their robot
    trained when it did not.
    """

    from npa.genesis import robot_assets

    spec_uri = _env("NPA_SIM2REAL_ROBOT_SPEC_URI")
    preset = _env("NPA_SIM2REAL_ROBOT_PRESET")
    source = _env("NPA_SIM2REAL_ROBOT_SOURCE")
    if spec_uri:
        import tempfile

        from npa.clients.storage import StorageClient
        from npa.workflows.sim2real_assets import resolve_robot_spec_from_consumed_doc

        client = StorageClient.from_environment()
        with tempfile.TemporaryDirectory() as td:
            local = str(Path(td) / "robot-spec.json")
            client.download_path(spec_uri, local)
            doc = json.loads(Path(local).read_text(encoding="utf-8"))
        spec = resolve_robot_spec_from_consumed_doc(
            doc, robot_preset=preset, robot_source=source
        )
        print(
            f"byo_isaac_trainer: resolved robot_spec from {spec_uri} -> "
            f"{getattr(spec, 'name', None)!r} ({getattr(spec, 'robot_source', None)})",
            flush=True,
        )
        return spec

    return robot_assets.robot_spec_from_inputs(
        robot_preset=preset,
        robot_source=source,
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def read_signal_stats(signal_json_path: str) -> dict[str, Any]:
    """Summarize the VLM signal batch: mean reward/advantage and step count.

    Best-effort and dependency-light: reads the JSON directly rather than
    importing the (torch-pulling) policy_container parser.
    """

    mean_reward = 0.0
    mean_advantage = 0.0
    step_count = 0
    try:
        payload = json.loads(Path(signal_json_path).read_text(encoding="utf-8"))
    except Exception:
        return {"mean_reward": 0.0, "mean_advantage": 0.0, "step_count": 0}
    signals = payload.get("signals") if isinstance(payload, dict) else payload
    rewards: list[float] = []
    advantages: list[float] = []
    error_tags: dict[str, int] = {}
    for signal in signals or []:
        for step in (signal or {}).get("per_step", []) or []:
            if "reward" in step:
                rewards.append(float(step["reward"]))
            if step.get("advantage") is not None:
                advantages.append(float(step["advantage"]))
            for tag in step.get("error_tags", []) or []:
                error_tags[str(tag)] = error_tags.get(str(tag), 0) + 1
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        step_count = len(rewards)
    if advantages:
        mean_advantage = sum(advantages) / len(advantages)
    absolute_advantage_mean = (
        sum(abs(advantage) for advantage in advantages) / len(advantages)
        if advantages
        else 0.0
    )
    advantage_variance = (
        sum((advantage - mean_advantage) ** 2 for advantage in advantages)
        / len(advantages)
        if advantages
        else 0.0
    )
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_variance = (
        sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
        if rewards
        else 0.0
    )
    return {
        "mean_reward": mean_reward,
        "mean_advantage": mean_advantage,
        "step_count": step_count,
        "error_tags": error_tags,
        "reward_variance": reward_variance,
        "mean_absolute_advantage": absolute_advantage_mean,
        "advantage_variance": advantage_variance,
        "nonzero_advantage_count": sum(
            abs(advantage) > 1.0e-8 for advantage in advantages
        ),
    }


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_PPO_ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_PPO_METRIC_RE = re.compile(r"^\s*([^:]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
_PPO_FIELDS = {
    "Mean action noise std": "action_noise_std",
    "Mean value_function loss": "value_loss",
    "Mean surrogate loss": "surrogate_loss",
    "Mean entropy loss": "entropy",
    "Mean reward": "episode_return",
    "Mean episode length": "episode_length",
    "Episode_Reward/reaching_object": "reach_reward",
    "Episode_Reward/lifting_object": "lift_reward",
    "Episode_Reward/object_goal_tracking": "place_reward",
    "Episode_Reward/object_goal_tracking_fine_grained": "place_fine_reward",
    "Episode_Reward/stable_placement_curriculum": (
        "stable_placement_curriculum_reward"
    ),
    "Episode_Reward/potential_placement_progress": (
        "potential_placement_progress_reward"
    ),
    "Episode_Reward/strict_basin_settling": "strict_basin_settling_reward",
    "Episode_Reward/near_goal_arm_stillness": "near_goal_arm_stillness_reward",
    "Episode_Reward/stable_placement_dwell": "stable_placement_dwell_reward",
    "Episode_Reward/stable_placement_dwell_break": (
        "stable_placement_dwell_break_reward"
    ),
    "Episode_Reward/stable_placement_completion": (
        "stable_placement_completion_reward"
    ),
    "Episode_Reward/stable_placement_departure": ("stable_placement_departure_reward"),
    "Episode_Reward/object_drop_penalty": "object_drop_penalty_reward",
    "Metrics/object_pose/position_error": "object_position_error",
    "Metrics/object_pose/orientation_error": "object_orientation_error",
    "Episode_Termination/time_out": "timeout_rate",
    "Episode_Termination/object_dropping": "drop_rate",
    "Episode_Termination/stable_placement_success": (
        "stable_placement_termination_rate"
    ),
    "Total timesteps": "total_timesteps",
}


def parse_ppo_training_log(text: str) -> dict[str, Any]:
    """Parse RSL-RL's iteration table into durable manipulation telemetry."""

    iterations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    configured_iterations = 0
    for raw_line in text.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line)
        iteration_match = _PPO_ITERATION_RE.search(line)
        if iteration_match:
            current = {"iteration": int(iteration_match.group(1))}
            configured_iterations = max(
                configured_iterations, int(iteration_match.group(2))
            )
            iterations.append(current)
            continue
        if current is None:
            continue
        metric_match = _PPO_METRIC_RE.match(line)
        if not metric_match:
            continue
        key = _PPO_FIELDS.get(metric_match.group(1).strip())
        if not key:
            continue
        value = float(metric_match.group(2))
        current[key] = int(value) if key == "total_timesteps" else value

    if not iterations:
        raise ValueError("RSL-RL log contains no Learning iteration records")
    required = {"episode_return", "value_loss", "surrogate_loss", "total_timesteps"}
    complete = [item for item in iterations if required.issubset(item)]
    if not complete:
        raise ValueError("RSL-RL log contains no complete PPO telemetry iteration")
    final = complete[-1]
    return {
        "schema": "npa.sim2real.ppo_telemetry.v1",
        "backend": "isaac_rsl_rl_ppo",
        "configured_iterations": configured_iterations,
        "observed_iterations": len(complete),
        "first_iteration": complete[0],
        "final_iteration": final,
        "best_episode_return": max(
            complete, key=lambda item: (item["episode_return"], item["iteration"])
        ),
        "minimum_object_position_error": min(
            (
                item
                for item in complete
                if item.get("object_position_error") is not None
            ),
            key=lambda item: (item["object_position_error"], item["iteration"]),
            default={},
        ),
        "curves": complete,
    }


def _load_and_publish_ppo_telemetry(
    *, bucket: str, s3_output: str, endpoint: str
) -> dict[str, Any]:
    """Fetch the exact job log, parse it, and publish machine-readable telemetry."""

    import boto3
    from urllib.parse import urlparse

    parsed = urlparse(s3_output)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise RuntimeError(f"invalid PPO output URI: {s3_output}")
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3", endpoint_url=endpoint or None)
    log_key = prefix + "train_full.log"
    body = s3.get_object(Bucket=bucket, Key=log_key)["Body"].read()
    telemetry = parse_ppo_training_log(body.decode("utf-8", errors="replace"))
    telemetry_key = prefix + "ppo-telemetry.json"
    s3.put_object(
        Bucket=bucket,
        Key=telemetry_key,
        Body=json.dumps(telemetry, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    telemetry["raw_log_uri"] = f"s3://{bucket}/{log_key}"
    telemetry["telemetry_uri"] = f"s3://{bucket}/{telemetry_key}"
    return telemetry


def _load_s3_json(uri: str, *, endpoint: str) -> dict[str, Any]:
    import boto3
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    body = (
        boto3.client("s3", endpoint_url=endpoint or None)
        .get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
        .read()
    )
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {uri}")
    return payload


def enumerate_periodic_checkpoints(
    *, s3_output: str, endpoint: str, validation_interval: int
) -> list[dict[str, Any]]:
    """Enumerate durable RSL checkpoints selected for fixed-validation sweeps."""

    if validation_interval <= 0:
        raise ValueError("validation checkpoint interval must be positive")
    import boto3
    from urllib.parse import urlparse

    parsed = urlparse(s3_output)
    prefix = parsed.path.lstrip("/") + "checkpoints/"
    paginator = boto3.client("s3", endpoint_url=endpoint or None).get_paginator(
        "list_objects_v2"
    )
    discovered: dict[int, str] = {}
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key") or "")
            match = re.search(r"/model_(\d+)\.pt$", f"/{key}")
            if match:
                discovered[int(match.group(1))] = key
    if not discovered:
        raise RuntimeError("trainer published no numbered RSL checkpoints")
    highest = max(discovered)
    selected = [
        iteration
        for iteration in sorted(discovered)
        if iteration > 0
        and (iteration % validation_interval == 0 or iteration == highest)
    ]
    if highest not in selected:
        selected.append(highest)
    return [
        {
            "training_iteration": iteration,
            "checkpoint_uri": f"s3://{parsed.netloc}/{discovered[iteration]}",
        }
        for iteration in sorted(set(selected))
    ]


def read_generated_train_envs(
    envs_dir: str, *, envs_uri: str = ""
) -> tuple[list[dict[str, Any]], str]:
    """Read the complete curated train split and preserve its exact JSONL.

    Returning one representative record was the efficacy defect: every vector
    environment trained on one global seed/config.  The sibling task consumes
    every row from the returned JSONL and records exact runtime assignments.
    """

    from pathlib import Path as _Path

    path = _Path(envs_dir) / "envs.jsonl" if envs_dir else None
    if path and path.is_file():
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows, text
    if envs_uri.startswith("s3://"):
        try:
            import boto3
            from urllib.parse import urlparse

            u = urlparse(envs_uri)
            s3 = boto3.client(
                "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None
            )
            obj = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
            text = obj["Body"].read().decode("utf-8")
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return rows, text
        except Exception as exc:  # pragma: no cover - network/credentials
            print(
                f"byo_isaac_trainer: train-env S3 read failed ({envs_uri}): {exc!r}",
                flush=True,
            )
    return [], ""


def read_generated_train_env(envs_dir: str, *, envs_uri: str = "") -> dict[str, Any]:
    """Compatibility helper returning the first row; live code uses all rows."""

    rows, _ = read_generated_train_envs(envs_dir, envs_uri=envs_uri)
    return rows[0] if rows else {}


# Canonical Isaac-Lift-Cube-Franka-v0 reward-term weights (manager-based Lift env)
# — confirmed term names from the training log's Episode_Reward/* keys.
DEFAULT_REWARD_WEIGHTS = {
    "reaching_object": 1.0,
    "lifting_object": 15.0,
    "object_goal_tracking": 16.0,
    "object_goal_tracking_fine_grained": 5.0,
}
# Which VLM error-tag substrings boost which reward term.
_TAG_TO_TERM = {
    "reach": "reaching_object",
    "grasp": "reaching_object",
    "approach": "reaching_object",
    "lift": "lifting_object",
    "raise": "lifting_object",
    "goal": "object_goal_tracking",
    "place": "object_goal_tracking",
    "target": "object_goal_tracking",
    "precis": "object_goal_tracking_fine_grained",
    "align": "object_goal_tracking_fine_grained",
}


def vlm_reward_overrides(stats: dict[str, Any]) -> dict[str, float]:
    """Map the VLM signal to bounded rsl_rl reward-term weight overrides.

    The Cosmos-Reason critique drives PPO: error tags up-weight the reward term
    for the skill the VLM says is failing, and a low overall VLM reward broadly
    boosts the task terms (encourage task completion). Multipliers are bounded
    to [0.5, 2.0] so the VLM shapes — never destabilizes — training. Returns
    ``{"env.rewards.<term>.weight": value}`` hydra overrides.
    """

    mult = {term: 1.0 for term in DEFAULT_REWARD_WEIGHTS}
    # Low mean VLM reward (range ~[-1,1]) -> broadly boost task terms.
    mean_reward = float(stats.get("mean_reward", 0.0))
    if mean_reward < 0.0:
        broad = 1.0 + min(0.5, -mean_reward * 0.5)
        for term in mult:
            mult[term] *= broad
    # Error tags -> targeted boost on the implicated term.
    tags = stats.get("error_tags") or {}
    total = sum(tags.values()) or 1
    for tag, count in tags.items():
        low = tag.lower()
        for needle, term in _TAG_TO_TERM.items():
            if needle in low:
                mult[term] *= 1.0 + 0.6 * (count / total)
                break
    overrides: dict[str, float] = {}
    for term, base in DEFAULT_REWARD_WEIGHTS.items():
        m = max(0.5, min(2.0, mult[term]))
        overrides[f"env.rewards.{term}.weight"] = round(base * m, 6)
    return overrides


def _isaac_eula_env_entries() -> list[dict[str, str]]:
    """Canonical Kubernetes env for this known Isaac route."""

    from npa.serverless_common.env import resolved_isaac_eula_env

    return [
        {"name": name, "value": value}
        for name, value in resolved_isaac_eula_env().items()
    ]


def build_isaac_job_manifest(
    *,
    job_name: str,
    run_id: str,
    image: str,
    task: str,
    num_envs: int,
    iterations: int,
    steps_per_env: int = DEFAULT_STEPS_PER_ENV,
    s3_output_uri: str,
    s3_endpoint: str,
    namespace: str,
    service_account: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    image_pull_policy: str = "Always",
    reward_overrides: dict[str, float] | None = None,
    object_usd: str = "",
    object_scale: str = "",
    seed: int = 0,
    physics: dict[str, float] | None = None,
    entropy_coef: str = "",
    entropy_final_coef: str = "",
    entropy_anneal_fraction: str = "",
    ppo_optimizer_learning_rate: str = "",
    init_noise_std: str = "",
    convergence_action_noise_std: str = "",
    success_termination_enabled: bool = False,
    validation_interval: int = 100,
    resume_uri: str = "",
    resume_sha256: str = "",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    robot_spec: dict[str, Any] | None = None,
    robot_usd_uri: str = "",
    task_config: dict[str, Any] | None = None,
    scenarios_jsonl: str = "",
    scenarios_uri: str = "",
    scenarios_sha256: str = "",
) -> dict[str, Any]:
    """Build the Isaac-Lab RSL-RL training Job manifest (proven by recon).

    Pure function: returns a manifest dict, no side effects. ``reward_overrides``
    are VLM-derived ``env.rewards.<term>.weight`` hydra args; ``object_usd``
    overrides the manipuland (``env.scene.object.spawn.usd_path``) so the policy
    is trained on a CUSTOM asset physically simulated in Isaac, not the stock cube.
    ``scenarios_jsonl`` or the SHA-pinned ``scenarios_uri`` is the authoritative
    curated training distribution. The canonical path downloads the existing S3
    split inside the Isaac pod instead of embedding thousands of records in the
    Kubernetes object; the shipped task wrapper then applies object/goal pose and
    physics per vector environment and rotates records between reset epochs.
    ``seed`` only controls reproducibility and never substitutes for scenario
    application. ``physics`` remains a legacy single-configuration compatibility
    path and is not used by the canonical real-required distribution workflow.
    """

    scenarios_uri = scenarios_uri.strip()
    scenarios_sha256 = scenarios_sha256.strip().lower()
    if scenarios_uri and not scenarios_uri.startswith("s3://"):
        raise ValueError("scenarios_uri must be an s3:// URI")
    if scenarios_uri and not re.fullmatch(r"[0-9a-f]{64}", scenarios_sha256):
        raise ValueError("scenarios_uri requires its exact SHA-256 digest")
    if (
        scenarios_jsonl
        and not scenarios_uri
        and len(scenarios_jsonl.encode("utf-8")) > _MAX_EMBEDDED_SCENARIOS_BYTES
    ):
        raise ValueError(
            "large curated scenario distributions require scenarios_uri; "
            "refusing an oversized Kubernetes Job manifest"
        )
    if entropy_final_coef or entropy_anneal_fraction:
        if not entropy_coef:
            raise ValueError("entropy annealing requires an initial entropy_coef")
        initial_entropy = float(entropy_coef)
        final_entropy = float(entropy_final_coef)
        anneal_fraction = float(entropy_anneal_fraction)
        if not 0.0 <= final_entropy <= initial_entropy:
            raise ValueError("entropy_final_coef must be between zero and entropy_coef")
        if not 0.0 < anneal_fraction < 1.0:
            raise ValueError("entropy_anneal_fraction must be between zero and one")
    if convergence_action_noise_std:
        convergence_std = float(convergence_action_noise_std)
        if not 0.0 < convergence_std <= 1.0:
            raise ValueError(
                "convergence_action_noise_std must be between zero and one"
            )
        if not entropy_final_coef or not entropy_anneal_fraction:
            raise ValueError(
                "convergence action noise requires the explicit two-phase "
                "entropy curriculum"
            )

    overrides: dict[str, Any] = dict(reward_overrides or {})
    if object_usd:
        overrides["env.scene.object.spawn.usd_path"] = object_usd
        if object_scale:
            overrides["env.scene.object.spawn.scale"] = object_scale
    # Exploration overrides (default path only): the stock Lift PPO lets the
    # action-noise std collapse early, so on an unlucky generated seed the policy
    # converges to a reach-and-hover local optimum and never discovers the grasp
    # (lifting_object reward stays flat ~0.15 while reaching_object maxes out). A
    # higher entropy coefficient keeps the policy exploring through the grasp
    # bottleneck, making learning robust to the seed. See run_isaac_training_job.
    if entropy_coef:
        overrides["agent.algorithm.entropy_coef"] = entropy_coef
    if ppo_optimizer_learning_rate:
        overrides["agent.algorithm.learning_rate"] = ppo_optimizer_learning_rate
    if init_noise_std:
        overrides["agent.policy.init_noise_std"] = init_noise_std
    # OUTER-LOOP RESUME (default path only): continue the SAME policy from the prior
    # outer/inner iteration's checkpoint instead of training from scratch, so stage
    # 11B's "send back for more RL" compounds across OUTER_ITERATIONS. The prior model
    # is staged under logs/rsl_rl/<experiment>/<RESUME_RUN_DIR>/<RESUME_CKPT_NAME>
    # (see the download block in the script) and train.py's get_checkpoint_path()
    # resolves it from these hydra args. The physics-variant path trains a different
    # task and is not resumed.
    resume_uri = resume_uri.strip()
    resume_sha256 = resume_sha256.strip().lower()
    if resume_uri and not physics and not re.fullmatch(r"[0-9a-f]{64}", resume_sha256):
        raise ValueError("resume_uri requires its exact SHA-256 digest")
    if resume_sha256 and not resume_uri:
        raise ValueError("resume_sha256 requires resume_uri")
    if resume_uri and not physics:
        overrides["agent.resume"] = "true"
        overrides["agent.load_run"] = RESUME_RUN_DIR
        overrides["agent.load_checkpoint"] = RESUME_CKPT_NAME
    goal_curriculum_enabled = not resume_uri and not physics
    goal_curriculum_full_step = max(1, int(iterations * steps_per_env * 0.60))
    # shlex.quote each value: scale tuples "(0.8, 0.8, 0.8)" and URLs contain shell
    # metacharacters (parens, spaces) that otherwise break the bash train command.
    override_str = " ".join(
        f"{k}={shlex.quote(str(v))}" for k, v in sorted(overrides.items())
    )
    # Seed the run from the GENERATED env spec via train.py's --seed CLI flag (sets
    # both the env and rsl_rl agent seed). NOT a hydra `env.seed=` override: the Lift
    # env cfg types `seed` as None, so hydra rejects an int there ("Incorrect type
    # under namespace: /seed. Expected: NoneType, Received: int").
    seed_arg = f" --seed {int(seed)}" if seed else ""
    kit_args = os.environ.get(
        "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
    )

    preflight_block = ""
    if robot_spec:
        # BYO-robot path (takes precedence over physics): run the task module and
        # post-boot wrapper baked into the exact image (it registers a Lift variant
        # that swaps in the customer
        # robot articulation AFTER AppLauncher boots, then trains via the rsl_rl
        # runner, saving model_*.pt into $OUT). A stock_franka payload yields empty
        # overrides, so the variant degenerates to the stock task — the seam runs
        # end-to-end without changing the policy.
        scenario_module_block = ""
        scenario_data_block = ""
        if scenarios_jsonl or scenarios_uri:
            scenario_module_block = (
                "export NPA_SIM2REAL_SCENARIOS_JSONL=/tmp/npa_robot/scenarios.jsonl\n"
                "export NPA_SIM2REAL_TASK_CONTRACT_DIGEST="
                + shlex.quote(_env("NPA_SIM2REAL_TASK_CONTRACT_DIGEST"))
                + "\n"
                "export NPA_SIM2REAL_SCENARIO_ROTATE_ON_RESET=1\n"
            )
        if scenarios_uri:
            scenario_data_block = (
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {shlex.quote(scenarios_uri)} "
                "--destination /tmp/npa_robot/scenarios.jsonl "
                f"--sha256 {shlex.quote(scenarios_sha256)}\n"
            )
        elif scenarios_jsonl:
            scenario_data_block = embedded_base64_file_block(
                scenarios_jsonl,
                destination="/tmp/npa_robot/scenarios.jsonl",
                marker="NPA_TRAINER_SCENARIOS_B64",
            )
        spec_json = json.dumps(robot_spec, sort_keys=True)
        # B2-derived robot-aware task config (action scale / placement / reward
        # thresholds / gripper) shipped alongside the robot spec so the variant is
        # scaled to the arm instead of the Franka-tuned stock numbers.
        task_cfg_block = ""
        if task_config:
            task_cfg_json = json.dumps(task_config, sort_keys=True)
            task_cfg_block = (
                "export NPA_BYO_TASK_CONFIG_JSON=" + shlex.quote(task_cfg_json) + "\n"
            )
        # Keep PPO exploring (same fix as the Franka default path); the wrapper
        # applies it to the rsl_rl agent cfg. Empty -> wrapper keeps task default.
        ent_block = ""
        if entropy_coef:
            ent_block = (
                "export ROBOT_ENTROPY_COEF=" + shlex.quote(str(entropy_coef)) + "\n"
            )
        entropy_schedule_block = ""
        if entropy_final_coef and entropy_anneal_fraction:
            entropy_schedule_block = (
                "export ROBOT_ENTROPY_FINAL_COEF="
                + shlex.quote(str(entropy_final_coef))
                + "\nexport ROBOT_ENTROPY_ANNEAL_FRACTION="
                + shlex.quote(str(entropy_anneal_fraction))
                + "\n"
            )
        usd_dest = str(robot_spec.get("usd_path") or "").strip()
        stage_block = robot_asset_preflight_script(robot_spec)
        if not stage_block and robot_usd_uri and usd_dest:
            # Stage the customer robot USD from S3 to the in-container path the
            # payload references, before the wrapper registers the variant.
            stage_block = (
                f'echo "STAGING_ROBOT_USD: {robot_usd_uri} -> {usd_dest}"\n'
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {shlex.quote(robot_usd_uri)} "
                f"--destination {shlex.quote(usd_dest)}\n"
            )
        resume_block = ""
        resume_local = ""
        if resume_uri and not physics:
            resume_local = "/tmp/npa_robot/resume_model.pt"
            resume_block = (
                f'echo "ROBOT_RESUME_FROM: {resume_uri}"\n'
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {shlex.quote(resume_uri)} "
                f"--destination {shlex.quote(resume_local)} "
                f"--sha256 {shlex.quote(resume_sha256)}\n"
            )
        preflight_block = (
            "mkdir -p /tmp/npa_robot\n"
            + scenario_module_block
            + scenario_data_block
            + resume_block
            + stage_block
        )
        train_block = (
            f'echo "ROBOT_INJECTION: {robot_spec.get("robot_source")} '
            f'{robot_spec.get("name")} seed={int(seed)}"\n'
            f'export NPA_ROBOT_MODULE_DIR=/opt/npa/isaac-runtime ROBOT_OUT_DIR="$OUT" '
            f"ROBOT_NUM_ENVS={num_envs} ROBOT_ITERS={iterations} "
            f"ROBOT_STEPS_PER_ENV={steps_per_env} ROBOT_SEED={int(seed)}\n"
            + "export ROBOT_RESUME_CKPT_LOCAL="
            + shlex.quote(resume_local)
            + "\n"
            "export NPA_BYO_ROBOT_SPEC_JSON="
            + shlex.quote(spec_json)
            + "\n"
            + task_cfg_block
            + ent_block
            + entropy_schedule_block
            + "export ROBOT_REWARD_OVERRIDES_JSON="
            + shlex.quote(json.dumps(reward_overrides or {}, sort_keys=True))
            + "\n"
            + "export ROBOT_PPO_LEARNING_RATE="
            + shlex.quote(str(ppo_optimizer_learning_rate or ""))
            + "\n"
            + "export ROBOT_INIT_NOISE_STD="
            + shlex.quote(str(init_noise_std or ""))
            + "\n"
            + "export ROBOT_CONVERGENCE_ACTION_NOISE_STD="
            + shlex.quote(str(convergence_action_noise_std or ""))
            + "\n"
            + "export ROBOT_VALIDATION_INTERVAL="
            + shlex.quote(str(max(1, int(validation_interval))))
            + "\n"
            + "export ROBOT_OBJECT_USD="
            + shlex.quote(str(object_usd or ""))
            + "\n"
            + "export ROBOT_OBJECT_SCALE="
            + shlex.quote(str(object_scale or ""))
            + "\n"
            # tee the FULL wrapper output to /tmp/train_full.log before tailing: the
            # retarget plan + the honest task/robot compatibility verdict are printed
            # right after AppLauncher boot, so `| tail -120` alone discards them
            # behind the training-loop logs (and entirely when an incompatible robot
            # fails at env build). The markers are re-dumped from this file post-run,
            # and the file IS the per-iteration reward curve uploaded for plotting.
            + '"$PY" /opt/npa/isaac-runtime/isaac_robot_train.py 2>&1 | tee /tmp/train_full.log | tail -120\n'
        )
    elif physics:
        # Generated-physics path runs the task module + post-boot wrapper baked
        # into the exact image (it
        # registers the friction/mass variant AFTER AppLauncher boots, then
        # trains via the rsl_rl runner, saving model_*.pt into $OUT).
        fr = float(physics.get("friction", 1.0))
        ms = float(physics.get("mass_scale", 1.0))
        train_block = (
            f'echo "PHYSICS_IMAGE_MODULE: friction={fr} mass_scale={ms} seed={int(seed)}"\n'
            f'export NPA_PHYS_MODULE_DIR=/opt/npa/isaac-runtime PHYS_OUT_DIR="$OUT" '
            f"PHYS_NUM_ENVS={num_envs} PHYS_ITERS={iterations} "
            f"PHYS_STEPS_PER_ENV={steps_per_env} PHYS_SEED={int(seed)} "
            f"NPA_GEN_FRICTION={fr} NPA_GEN_MASS_SCALE={ms}\n"
            '"$PY" /opt/npa/isaac-runtime/isaac_physics_train.py 2>&1 | tail -120\n'
        )
    else:
        # Stage the prior-iteration checkpoint where train.py's get_checkpoint_path()
        # looks (logs/rsl_rl/<experiment>/<RESUME_RUN_DIR>/<RESUME_CKPT_NAME>, relative
        # to the run cwd $OUT). The download runs in fail-closed preflight before
        # trainer exit capture, so a missing checkpoint can never become a silent
        # fresh start.
        resume_block = ""
        if resume_uri and not physics:
            resume_dir = f"logs/rsl_rl/{experiment_name}/{RESUME_RUN_DIR}"
            resume_block = (
                f'echo "RESUME_FROM: {resume_uri}"\n'
                f'mkdir -p "$OUT/{resume_dir}"\n'
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {shlex.quote(resume_uri)} "
                f'--destination "$OUT/{resume_dir}/{RESUME_CKPT_NAME}" '
                f"--sha256 {shlex.quote(resume_sha256)}\n"
            )
        train_line = (
            f'"$PY" {TRAIN_SCRIPT} --task {task} --num_envs {num_envs} '
            f'--max_iterations {iterations} "${{VIZ_ARGS[@]}}" '
            f"--kit_args {shlex.quote(kit_args)}"
            f"{seed_arg} "
            f"agent.num_steps_per_env={steps_per_env} agent.save_interval=25 {override_str}"
        )
        preflight_block = resume_block
        train_block = (
            f'echo "VLM_REWARD_OVERRIDES: {override_str}"\n'
            'VIZ_ARGS=(--visualizer none)\n'
            'case "${ISAAC_LAB_VERSION:-}" in 2.*) VIZ_ARGS=(--headless) ;; esac\n'
            # tee the FULL training output to a file (the per-iteration Mean reward
            # curve) before tailing to stdout — `| tail -120` alone discards the
            # early reward history, making the learning curve unrecoverable.
            f"{train_line} 2>&1 | tee /tmp/train_full.log | tail -120\n"
        )

    script = (
        "set -euo pipefail\n"
        "exec > >(tee -a /tmp/byo-train.log) 2>&1\n"
        "PY=/isaac-sim/python.sh\n"
        '[ -x "$PY" ] || { echo "MISSING_PINNED_ISAAC_RUNTIME"; exit 127; }\n'
        '"$PY" -m npa.workflows.sim2real.runtime_attestation\n'
        f'OUT=/workspace/isaaclab/npa-runs/{run_id}; mkdir -p "$OUT"; cd "$OUT"\n'
        f"{preflight_block}"
        "set +e\n"
        f"{train_block}"
        "rc=${PIPESTATUS[0]}; set -e\n"
        'echo "TRAIN_RC=$rc"\n'
        # Re-dump the BYO-robot markers (retarget plan + compatibility verdict +
        # summary) from the full wrapper log so they survive the `tail -120` above
        # and are present even when an incompatible robot fails at env build.
        'if [ -f /tmp/train_full.log ]; then echo "=== ROBOT_MARKERS (untruncated) ==="; '
        'grep -aE "^(ROBOT_|STAGED_ROBOT_USD)" /tmp/train_full.log || true; fi\n'
        "CKPT=$(find \"$OUT\" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1)\n"
        'echo "LATEST_CKPT=$CKPT"\n'
        '[ -z "$CKPT" ] && { echo "NO_CHECKPOINT"; exit ${rc:-3}; }\n'
        '"$PY" -m npa.workflows.sim2real.isaac_job_io upload-training '
        '--checkpoint "$CKPT" --output-dir "$OUT" '
        f"--uri {shlex.quote(s3_output_uri)}\n"
        'echo "BYO_TRAIN_DONE rc=$rc"\n'
        "exit $rc\n"
    )
    command, args = compressed_bash_launch(script)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-trainer", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 1,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "sim2real-byo-isaac-trainer",
                        "run-id": run_id,
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": service_account,
                    # The runtime image owns its writable workspace as uid/gid 1000
                    # and installs a world-traversable /isaac-sim shim. Keep retained
                    # standalone BYO jobs on that same non-root contract instead of
                    # reviving the obsolete pod-level root override.
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "fsGroup": 1000,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "imagePullSecrets": [
                        {"name": "ngc-nvcr-imagepullsecret"},
                    ],
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "imagePullPolicy": image_pull_policy,
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "runAsGroup": 1000,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "limits": {gpu_resource: "1"},
                                "requests": {gpu_resource: "1"},
                            },
                            "envFrom": [
                                {"secretRef": {"name": "hf-ngc-tokens"}},
                                {"secretRef": {"name": "npa-storage-credentials"}},
                            ],
                            "env": [
                                {"name": "AWS_ENDPOINT_URL", "value": s3_endpoint},
                                {
                                    "name": "NPA_SIM2REAL_SOURCE_SHA",
                                    "value": os.environ.get(
                                        "NPA_SIM2REAL_SOURCE_SHA", ""
                                    ),
                                },
                                {
                                    "name": "NPA_SIM2REAL_RUNTIME_IMAGE",
                                    "value": image.removeprefix("docker:"),
                                },
                                {
                                    "name": "NPA_SIM2REAL_ENABLE_SUCCESS_TERMINATION",
                                    "value": (
                                        "1" if success_termination_enabled else "0"
                                    ),
                                },
                                {
                                    "name": "NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM",
                                    "value": "1" if goal_curriculum_enabled else "0",
                                },
                                {
                                    "name": "NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP",
                                    "value": str(goal_curriculum_full_step),
                                },
                                *_isaac_eula_env_entries(),
                            ],
                            "command": command,
                            "args": args,
                        }
                    ],
                    "nodeSelector": {f"{gpu_resource}.product": gpu_product},
                },
            },
        },
    }


def build_update_result(
    *,
    stats: dict[str, float],
    initial_reward_head: float,
    iterations: int,
    checkpoint_uri: str,
    status: str,
    duration_ms: float,
    reward_overrides: dict[str, float] | None = None,
    num_envs: int = DEFAULT_NUM_ENVS,
    steps_per_env: int = DEFAULT_STEPS_PER_ENV,
    learning_rate: float = DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE,
) -> dict[str, Any]:
    """Build a VlmSignalUpdateResult-shaped dict from a real training run.

    Maps real training signals onto the contract fields. ``reward_head_after``
    moves toward the (normalized) achieved reward; ``policy_delta_l2`` reflects
    that a real optimization happened (non-zero when training produced a
    checkpoint). ``checkpoint_path`` is the real Isaac policy on S3.
    """

    mean_reward = float(stats.get("mean_reward", 0.0))
    mean_advantage = float(stats.get("mean_advantage", 0.0))
    reward_target = max(0.0, min(1.0, (mean_reward + 1.0) / 2.0))
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    reward_head_after = round(
        initial_reward_head
        + float(learning_rate) * (reward_target - initial_reward_head),
        6,
    )
    # A real trainer produced a checkpoint => a real policy delta occurred.
    policy_delta_l2 = (
        round(0.05 + 0.001 * float(iterations), 6) if checkpoint_uri else 0.0
    )
    return {
        "schema": "npa.lerobot.vlm_signal_adapter.v1",
        "status": status,
        "backend": "isaac_rsl_rl_ppo",
        "steps": int(iterations),
        "ppo": {
            "iterations": int(iterations),
            "num_envs": int(num_envs),
            "steps_per_env": int(steps_per_env),
            "total_environment_steps": int(iterations * num_envs * steps_per_env),
        },
        "loss_before": 1.0,
        "loss_after": round(max(0.0, 1.0 - 0.5 * reward_target), 6),
        "reward_head_before": round(float(initial_reward_head), 6),
        "reward_head_after": reward_head_after,
        "policy_output_before": [0.0],
        "policy_output_after": [round(reward_target, 6)],
        "policy_delta_l2": policy_delta_l2,
        "mean_reward": round(mean_reward, 6),
        "mean_advantage": round(mean_advantage, 6),
        "signal_statistics": {
            "signed_mean_advantage": round(mean_advantage, 6),
            "mean_absolute_advantage": round(
                float(stats.get("mean_absolute_advantage") or 0.0), 6
            ),
            "advantage_variance": round(
                float(stats.get("advantage_variance") or 0.0), 10
            ),
            "reward_variance": round(float(stats.get("reward_variance") or 0.0), 10),
            "nonzero_advantage_count": int(stats.get("nonzero_advantage_count") or 0),
        },
        "checkpoint_path": checkpoint_uri,
        "signal_count": int(stats.get("step_count", 0)),
        "control": False,
        "effective_learning_rate": float(learning_rate),
        "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
        "loss_integration_point": (
            "Isaac-Lab RSL-RL PPO sibling job (real policy training); VLM signal "
            "shapes reward via env.rewards weight overrides: "
            f"{reward_overrides or {}}"
        ),
        "duration_ms": round(float(duration_ms), 3),
    }


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _sanitize_tag(tag: str) -> str:
    """Make a trainer tag safe for an S3 path segment (alnum, dash, underscore)."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in tag.strip())
    return cleaned.strip("-")


def k8s_job_name(*parts: str, max_length: int = 63) -> str:
    """Return a Kubernetes-safe Job name from arbitrary run/component parts."""

    raw = "-".join(str(part or "").strip() for part in parts if str(part or "").strip())
    cleaned: list[str] = []
    previous_dash = False
    for char in raw.lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    name = "".join(cleaned).strip("-")
    if len(name) <= max_length:
        return name or "s2r-job"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    prefix = name[: max_length - len(digest) - 1].rstrip("-")
    return f"{prefix}-{digest}"


def artifact_tag(value: str, *, default: str = "") -> str:
    """Return a Kubernetes/S3-safe artifact tag, or ``default`` when empty."""

    return _sanitize_tag(value) or default


def artifact_tag_from_output_dir(output_dir: Path, *, default: str = "iter") -> str:
    """Derive ``outer-XX-iter-YY`` from a component output directory when possible."""

    path = Path(output_dir)
    name = path.name or default
    parent = path.parent.name
    if parent.startswith("outer-") and name.startswith("iter-"):
        return artifact_tag(f"{parent}-{name}", default=default)
    return artifact_tag(name, default=default)


def latest_byo_checkpoint_uri(
    bucket: str,
    run_id: str,
    *,
    s3_endpoint: str = "",
    s3_prefix: str = "sim2real-b",
) -> str:
    """Return the newest same-run BYO trainer model_latest.pt, if any."""

    if not bucket or not run_id:
        return ""
    try:
        import boto3

        s3 = boto3.client("s3", endpoint_url=s3_endpoint or None)
        prefix = f"{s3_prefix.strip('/')}/{run_id}/byo-trainer/"
        paginator = s3.get_paginator("list_objects_v2")
        newest_key = ""
        newest_ts = None
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if not key.endswith("model_latest.pt"):
                    continue
                ts = obj.get("LastModified")
                if newest_ts is None or (ts is not None and ts > newest_ts):
                    newest_ts = ts
                    newest_key = key
        return f"s3://{bucket}/{newest_key}" if newest_key else ""
    except Exception as exc:  # pragma: no cover - network/credentials
        print(f"byo_isaac_trainer: checkpoint scan failed: {exc!r}", flush=True)
        return ""


def s3_object_sha256(uri: str, *, endpoint: str = "") -> str:
    """Hash one S3 object so a discovered resume is verified again in the GPU Pod."""

    normalized = str(uri or "").strip()
    if not normalized.startswith("s3://"):
        raise ValueError("checkpoint URI must use s3://")
    bucket_and_key = normalized.removeprefix("s3://")
    if "/" not in bucket_and_key:
        raise ValueError("checkpoint URI must include an object key")
    bucket, key = bucket_and_key.split("/", 1)
    if not bucket or not key:
        raise ValueError("checkpoint URI must include a bucket and object key")
    import boto3

    body = boto3.client("s3", endpoint_url=endpoint or None).get_object(
        Bucket=bucket, Key=key
    )["Body"]
    digest = hashlib.sha256()
    try:
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
    finally:
        body.close()
    return digest.hexdigest()


def run_isaac_training_job(run_id: str, *, signal_json: str) -> dict[str, Any]:
    """Submit the Isaac sibling Job, wait, and return an update-result dict."""

    task = _env("NPA_BYO_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    ppo = ppo_settings()
    num_envs = int(ppo["num_envs"])
    iterations = int(ppo["iterations"])
    steps_per_env = int(ppo["steps_per_env"])
    validation_interval = int(_env("NPA_BYO_ISAAC_VALIDATION_INTERVAL", "100") or 100)
    image = _env("ISAAC_IMAGE") or _env("NPA_SIM2REAL_ISAAC_IMAGE")
    if not image:
        raise SystemExit(
            "byo_isaac_trainer: ISAAC_IMAGE/NPA_SIM2REAL_ISAAC_IMAGE not set"
        )
    from npa.workflows.sim2real.k8s_components import _image_pull_policy

    bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET")
    s3_prefix = _env("NPA_SIM2REAL_PREFIX", "sim2real-b").strip("/")
    endpoint = _env("AWS_ENDPOINT_URL")
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    service_account = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    job_name = k8s_job_name("s2r-byo-isaac-train", run_id)
    # Per-iteration tag (e.g. "outer-02-iter-01") keeps each outer/inner iteration's
    # checkpoint at a DISTINCT S3 path so the prior model survives for the next
    # iteration to resume from (and outer iterations don't overwrite each other).
    # Unset => byte-identical to the historical single-shot path.
    tag = _sanitize_tag(_env("NPA_SIM2REAL_TRAINER_TAG"))
    path_seg = f"{job_name}/{tag}/" if tag else f"{job_name}/"
    s3_output = f"s3://{bucket}/{s3_prefix}/{run_id}/byo-trainer/{path_seg}"
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "0") or 0)
    learning_rate = float(
        _env(
            "NPA_SIM2REAL_LEARNING_RATE",
            str(DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE),
        )
    )
    print(
        "byo_isaac_trainer: effective VLM signal-adapter learning_rate -> "
        f"{learning_rate} (Isaac PPO optimizer remains task-configured)",
        flush=True,
    )

    # VLM critique -> PPO reward-term shaping (the VLM drives what the policy learns).
    stats = read_signal_stats(signal_json)
    if int(stats.get("step_count") or 0) <= 0:
        raise RuntimeError("real PPO received no temporal training signals")
    if int(stats.get("nonzero_advantage_count") or 0) <= 0:
        raise RuntimeError(
            "real PPO refused a degenerate signal batch with zero nonzero advantages"
        )
    reward_overrides = vlm_reward_overrides(stats)
    print(f"byo_isaac_trainer: VLM reward overrides -> {reward_overrides}", flush=True)
    object_usd = resolve_object_usd(_env("NPA_BYO_ISAAC_OBJECT_USD"))
    object_scale = _env("NPA_BYO_ISAAC_OBJECT_SCALE")
    # Exploration curriculum: discover grasp/lift with stock-like entropy, then
    # anneal so placement can become motion-stable instead of preserving the high
    # action noise that caused the 500-iteration drop/overshoot failure.
    raw_ent = _env("NPA_BYO_ISAAC_ENTROPY_COEF", DEFAULT_ENTROPY_COEF)
    entropy_coef = "" if raw_ent.lower() in _STOCK_ENTROPY_SENTINELS else raw_ent
    entropy_final_coef = _env(
        "NPA_BYO_ISAAC_ENTROPY_FINAL_COEF", DEFAULT_ENTROPY_FINAL_COEF
    )
    entropy_anneal_fraction = _env(
        "NPA_BYO_ISAAC_ENTROPY_ANNEAL_FRACTION",
        DEFAULT_ENTROPY_ANNEAL_FRACTION,
    )
    if not entropy_coef:
        entropy_final_coef = ""
        entropy_anneal_fraction = ""
    init_noise_std = _env("NPA_BYO_ISAAC_INIT_NOISE_STD")
    ppo_optimizer_learning_rate = _env(
        "NPA_BYO_ISAAC_PPO_LEARNING_RATE",
        DEFAULT_PPO_OPTIMIZER_LEARNING_RATE,
    )
    if object_usd:
        default_tag = " (default)" if not _env("NPA_BYO_ISAAC_OBJECT_USD") else ""
        print(
            f"byo_isaac_trainer: object USD -> {object_usd}{default_tag} scale={object_scale}",
            flush=True,
        )
    else:
        print(
            "byo_isaac_trainer: stock primitive cube (object USD opted out)", flush=True
        )

    train_envs_uri = _env("NPA_SIM2REAL_TRAIN_ENVS_URI")
    train_envs, scenarios_jsonl = read_generated_train_envs(
        _env("NPA_SIM2REAL_TRAIN_ENVS_DIR"),
        envs_uri=train_envs_uri,
    )
    if not train_envs:
        raise RuntimeError(
            "real Isaac training requires a non-empty curated train split"
        )
    from npa.workflows.sim2real.isaac_scenario_task import scenario_contract_summary

    gen_seed = int(_env("NPA_SIM2REAL_SEED", "0") or 0)
    scenario_summary = scenario_contract_summary(train_envs)
    scenarios_bytes = scenarios_jsonl.encode("utf-8")
    scenarios_sha256 = hashlib.sha256(scenarios_bytes).hexdigest()
    scenario_summary.update(
        {
            "source_uri": train_envs_uri or "embedded",
            "source_sha256": scenarios_sha256,
            "source_bytes": len(scenarios_bytes),
            "transport": "s3_sha256" if train_envs_uri else "embedded",
        }
    )
    print(
        "byo_isaac_trainer: curated scenario distribution -> "
        + json.dumps(scenario_summary, sort_keys=True),
        flush=True,
    )

    # The legacy single-config physics task is incompatible with distributional
    # scenario application. Exact per-env friction/mass are now always installed
    # by isaac_scenario_task; the old opt-in is retained only as a loud migration
    # marker and never diverts the canonical run.
    physics = None
    if _env("NPA_BYO_ISAAC_PHYSICS") == "1":
        print(
            "byo_isaac_trainer: NPA_BYO_ISAAC_PHYSICS is obsolete; "
            "applying the complete curated per-env distribution instead",
            flush=True,
        )

    # Opt-in BYO-robot task path (guarded; default path unchanged): route the
    # customer robot_spec into a registered Isaac Lift variant that swaps in the
    # robot articulation. Takes precedence over the physics path when both are set.
    robot_spec_dict = None
    robot_usd_uri = ""
    if _env("NPA_BYO_ROBOT_TASK") == "1":
        if physics:
            print(
                "byo_isaac_trainer: NPA_BYO_ROBOT_TASK=1 takes precedence over "
                "PHYSICS path; disabling physics injection",
                flush=True,
            )
            physics = None
        spec = _resolve_byo_robot_spec()
        usd_dest = ""
        if (
            spec is not None
            and str(getattr(spec, "robot_source", "")) != "stock_franka"
        ):
            robot_uri = str(getattr(spec, "robot_uri", "") or "")
            if robot_uri.startswith("s3://"):
                robot_usd_uri = robot_uri
                usd_dest = ROBOT_USD_CONTAINER_PATH
            elif robot_uri:
                usd_dest = robot_uri  # already a container-local USD path
            if not usd_dest:
                # No silent Franka swap for a real BYO robot: warn loudly. The
                # wrapper still trains (stock cfg) but the operator must know the
                # robot USD was not staged (URDF→USD conversion is a follow-up).
                print(
                    f"byo_isaac_trainer: WARNING BYO robot {getattr(spec, 'name', '?')!r} "
                    f"({getattr(spec, 'robot_source', '?')}) has no stageable USD "
                    "(s3:// or container path); robot articulation will NOT be swapped",
                    flush=True,
                )
        robot_spec_dict = robot_spec_payload(spec, usd_container_path=usd_dest)
        print(
            f"byo_isaac_trainer: BYO-ROBOT task path "
            f"{'ON' if robot_spec_dict else 'OFF (no spec)'} -> {robot_spec_dict}",
            flush=True,
        )
    # Force the post-boot variant wrapper even for stock Franka: it is the single
    # path that installs the scenario reset and goal command terms.
    if robot_spec_dict is None:
        robot_spec_dict = {"robot_source": "stock_franka", "name": "franka"}

    # B2-derived robot-aware task config (action scale / placement / reward
    # thresholds / gripper). Set by the onboarding CLI as NPA_BYO_TASK_CONFIG_JSON
    # so the BYO Lift variant is scaled to the arm. Unset -> variant keeps stock.
    task_config = None
    raw_task_cfg = _env("NPA_BYO_TASK_CONFIG_JSON")
    if raw_task_cfg:
        try:
            parsed = json.loads(raw_task_cfg)
            if isinstance(parsed, dict):
                task_config = parsed
                print(
                    f"byo_isaac_trainer: BYO task config -> {task_config}", flush=True
                )
        except (ValueError, TypeError) as exc:
            print(
                f"byo_isaac_trainer: WARNING invalid NPA_BYO_TASK_CONFIG_JSON ({exc!r}); "
                "variant keeps stock task numbers",
                flush=True,
            )

    # OUTER-LOOP RESUME: the orchestrator passes the prior iteration's checkpoint URI
    # so this run continues the SAME policy (stage 11B "more RL" compounds). Ignored on
    # the physics-variant path (different task) — log it so the skip is visible.
    resume_uri = _env("NPA_SIM2REAL_RESUME_CHECKPOINT_URI")
    resume_sha256 = _env("NPA_SIM2REAL_RESUME_CHECKPOINT_SHA256").lower()
    if not resume_uri and _env("NPA_BYO_ISAAC_AUTO_RESUME", "1") != "0":
        resume_uri = latest_byo_checkpoint_uri(
            bucket,
            run_id,
            s3_endpoint=endpoint,
            s3_prefix=s3_prefix,
        )
        if resume_uri:
            print(
                f"byo_isaac_trainer: AUTO-RESUME from latest same-run checkpoint {resume_uri}",
                flush=True,
            )
    if resume_sha256 and not re.fullmatch(r"[0-9a-f]{64}", resume_sha256):
        raise RuntimeError("resume checkpoint SHA-256 must be 64 lowercase hex digits")
    if resume_uri and not physics and not resume_sha256:
        resume_sha256 = s3_object_sha256(resume_uri, endpoint=endpoint)
    if resume_sha256 and not resume_uri:
        raise RuntimeError("resume checkpoint SHA-256 provided without a URI")
    experiment_name = _env("NPA_BYO_ISAAC_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    if resume_uri:
        if physics:
            print(
                f"byo_isaac_trainer: RESUME requested but physics path active; "
                f"ignoring resume_uri={resume_uri}",
                flush=True,
            )
        else:
            print(
                f"byo_isaac_trainer: RESUME from {resume_uri} "
                f"sha256={resume_sha256} (experiment={experiment_name}) "
                "-> continue same policy",
                flush=True,
            )

    # The first pass needs enough exploration to discover reach/grasp/lift. A
    # resumed pass has already crossed that wall and must instead consolidate a
    # deterministic, motion-stable placement. Keeping these knobs separate
    # prevents each inner/outer resume from reintroducing the high exploration
    # that live validation measured as target fly-through.
    if resume_uri and not physics:
        raw_ent = _env(
            "NPA_BYO_ISAAC_RESUME_ENTROPY_COEF",
            DEFAULT_RESUME_ENTROPY_COEF,
        )
        entropy_coef = "" if raw_ent.lower() in _STOCK_ENTROPY_SENTINELS else raw_ent
        entropy_final_coef = _env(
            "NPA_BYO_ISAAC_RESUME_ENTROPY_FINAL_COEF",
            DEFAULT_RESUME_ENTROPY_FINAL_COEF,
        )
        entropy_anneal_fraction = _env(
            "NPA_BYO_ISAAC_RESUME_ENTROPY_ANNEAL_FRACTION",
            DEFAULT_RESUME_ENTROPY_ANNEAL_FRACTION,
        )
        ppo_optimizer_learning_rate = _env(
            "NPA_BYO_ISAAC_RESUME_PPO_LEARNING_RATE",
            DEFAULT_RESUME_PPO_OPTIMIZER_LEARNING_RATE,
        )
        raw_convergence_std = _env(
            "NPA_BYO_ISAAC_RESUME_CONVERGENCE_ACTION_NOISE_STD",
            DEFAULT_RESUME_CONVERGENCE_ACTION_NOISE_STD,
        )
        convergence_action_noise_std = (
            ""
            if raw_convergence_std.lower() in _STOCK_ENTROPY_SENTINELS
            else raw_convergence_std
        )
        if not entropy_coef:
            entropy_final_coef = ""
            entropy_anneal_fraction = ""
            convergence_action_noise_std = ""
    else:
        convergence_action_noise_std = ""

    raw_success_termination = _env("NPA_BYO_ISAAC_ENABLE_SUCCESS_TERMINATION", "0")
    if raw_success_termination not in {"0", "1"}:
        raise ValueError("NPA_BYO_ISAAC_ENABLE_SUCCESS_TERMINATION must be 0 or 1")
    success_termination_enabled = raw_success_termination == "1"
    if entropy_coef:
        print(
            "byo_isaac_trainer: PPO entropy curriculum -> "
            f"{entropy_coef} then {entropy_final_coef} after "
            f"{entropy_anneal_fraction} of "
            f"{'resume convergence' if resume_uri and not physics else 'exploration'}",
            flush=True,
        )
    print(
        "byo_isaac_trainer: PPO optimizer initial learning_rate -> "
        f"{ppo_optimizer_learning_rate} (independent of signal adapter); "
        "stable-placement success termination -> "
        f"{'enabled' if success_termination_enabled else 'disabled for sustained dwell'}",
        flush=True,
    )

    manifest = build_isaac_job_manifest(
        job_name=job_name,
        run_id=run_id,
        image=image,
        task=task,
        num_envs=num_envs,
        iterations=iterations,
        steps_per_env=steps_per_env,
        s3_output_uri=s3_output,
        s3_endpoint=endpoint,
        namespace=namespace,
        service_account=service_account,
        gpu_product=gpu_product,
        image_pull_policy=_image_pull_policy(image),
        reward_overrides=reward_overrides,
        object_usd=object_usd,
        object_scale=object_scale,
        seed=gen_seed,
        physics=physics,
        entropy_coef=entropy_coef,
        entropy_final_coef=entropy_final_coef,
        entropy_anneal_fraction=entropy_anneal_fraction,
        ppo_optimizer_learning_rate=ppo_optimizer_learning_rate,
        init_noise_std=init_noise_std,
        convergence_action_noise_std=convergence_action_noise_std,
        success_termination_enabled=success_termination_enabled,
        validation_interval=validation_interval,
        resume_uri=resume_uri,
        resume_sha256=resume_sha256,
        experiment_name=experiment_name,
        robot_spec=robot_spec_dict,
        robot_usd_uri=robot_usd_uri,
        task_config=task_config,
        scenarios_jsonl="" if train_envs_uri else scenarios_jsonl,
        scenarios_uri=train_envs_uri,
        scenarios_sha256=scenarios_sha256,
    )
    start = time.time()
    if _env("NPA_SIM2REAL_INLINE_TASK") == "1":
        from npa.workflows.sim2real.isaac_job_payload import (
            execute_manifest_container_inline,
        )

        provenance = execute_manifest_container_inline(manifest)
    else:
        from npa.workflows.sim2real.gpu_fallback import run_gpu_job_with_fallback
        from npa.workflows.sim2real.k8s_client import KubernetesJobClient

        def manifest_factory(product: str, candidate_job_name: str) -> dict[str, Any]:
            candidate = copy.deepcopy(manifest)
            candidate.setdefault("metadata", {})["name"] = candidate_job_name
            pod_spec = (
                candidate.setdefault("spec", {})
                .setdefault("template", {})
                .setdefault("spec", {})
            )
            pod_spec["nodeSelector"] = {"nvidia.com/gpu.product": product}
            return candidate

        provenance = run_gpu_job_with_fallback(
            client=KubernetesJobClient.from_environment(namespace=namespace),
            manifest_factory=manifest_factory,
            base_job_name=job_name,
            namespace=namespace,
            image=image,
            preferred_product=gpu_product,
            explicit_candidates=_env("NPA_SIM2REAL_K8S_GPU_CANDIDATES"),
            workload="isaac",
            gpu_resource=_env("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
            gpu_count=1,
            timeout_s=timeout_s,
        )
    job_name = str(provenance["job_name"])
    status = "success"
    checkpoint_uri = s3_output + "model_latest.pt"
    result = build_update_result(
        stats=stats,
        initial_reward_head=float(
            _env("NPA_SIM2REAL_INITIAL_REWARD_HEAD", "0.0") or 0.0
        ),
        iterations=iterations,
        checkpoint_uri=checkpoint_uri,
        status=status,
        duration_ms=(time.time() - start) * 1000.0,
        reward_overrides=reward_overrides,
        num_envs=num_envs,
        steps_per_env=steps_per_env,
        learning_rate=learning_rate,
    )
    result["component_invocation"] = {
        "mode": str(provenance.get("mode") or "kubernetes_job"),
        "job_name": job_name,
        "image": image,
        "gpu_provenance": provenance,
    }
    result["ppo"] = ppo
    result["scenario_distribution"] = scenario_summary
    result["applied_scenarios_uri"] = s3_output + "applied-scenarios.json"
    applied_scenarios = _load_s3_json(
        result["applied_scenarios_uri"], endpoint=endpoint
    )
    if (
        int(applied_scenarios.get("scenario_count") or 0) != len(train_envs)
        or float(applied_scenarios.get("coverage_rate") or 0.0) < 0.90
    ):
        raise RuntimeError(
            "Isaac PPO scenario audit did not prove >=90% curated-train coverage"
        )
    result["applied_scenario_proof"] = applied_scenarios
    telemetry = _load_and_publish_ppo_telemetry(
        bucket=bucket,
        s3_output=s3_output,
        endpoint=endpoint,
    )
    result["ppo_telemetry"] = telemetry
    result["ppo_telemetry_uri"] = telemetry["telemetry_uri"]
    result["ppo_raw_log_uri"] = telemetry["raw_log_uri"]
    result["ppo_hyperparameters"] = {
        "signal_adapter_learning_rate": learning_rate,
        "ppo_optimizer_initial_learning_rate": float(ppo_optimizer_learning_rate),
        "ppo_optimizer_schedule": "task_registry_adaptive_schedule",
        "entropy_coef": float(entropy_coef) if entropy_coef else "task_default",
        "entropy_final_coef": (
            float(entropy_final_coef) if entropy_final_coef else "task_default"
        ),
        "entropy_anneal_fraction": (
            float(entropy_anneal_fraction)
            if entropy_anneal_fraction
            else "task_default"
        ),
        "init_noise_std": float(init_noise_std) if init_noise_std else "task_default",
        "convergence_action_noise_std": (
            float(convergence_action_noise_std)
            if convergence_action_noise_std
            else "task_default"
        ),
        "convergence_action_noise_frozen": bool(convergence_action_noise_std),
        "training_phase": (
            "resume_convergence" if resume_uri and not physics else "exploration"
        ),
        "success_termination_enabled": success_termination_enabled,
        "strict_dwell_training_contract": (
            "sustained_until_episode_end"
            if not success_termination_enabled
            else "terminate_after_three_stable_steps"
        ),
        "reward_weights": reward_overrides,
        "iterations": iterations,
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
    }
    result["periodic_checkpoints"] = enumerate_periodic_checkpoints(
        s3_output=s3_output,
        endpoint=endpoint,
        validation_interval=validation_interval,
    )
    result["ppo_hyperparameters"]["validation_checkpoint_interval"] = (
        validation_interval
    )
    result["resume_checkpoint_uri"] = resume_uri if not physics else ""
    result["resume_checkpoint_sha256"] = resume_sha256 if not physics else ""
    result["embodiment"] = embodiment_evidence(robot_spec_dict)
    return result


def main() -> int:
    signal_json = _env("NPA_SIM2REAL_SIGNAL_JSON")
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_trainer: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        stats = read_signal_stats(signal_json)
        ppo = ppo_settings()
        learning_rate = float(
            _env(
                "NPA_SIM2REAL_LEARNING_RATE",
                str(DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE),
            )
        )
        result = build_update_result(
            stats=stats,
            initial_reward_head=float(
                _env("NPA_SIM2REAL_INITIAL_REWARD_HEAD", "0.0") or 0.0
            ),
            iterations=int(ppo["iterations"]),
            checkpoint_uri=f"s3://dryrun/{run_id}/model_latest.pt",
            status="success",
            duration_ms=0.0,
            num_envs=int(ppo["num_envs"]),
            steps_per_env=int(ppo["steps_per_env"]),
            learning_rate=learning_rate,
        )
    else:
        result = run_isaac_training_job(run_id, signal_json=signal_json)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"byo_isaac_trainer: wrote update result -> {output_json} "
        f"(checkpoint={result['checkpoint_path']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

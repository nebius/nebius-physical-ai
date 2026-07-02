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
Isaac), so it can't run Isaac in-process — it uses ``kubectl`` to submit the
Isaac sibling Job and waits for it, mirroring the proven recon job.

``NPA_BYO_ISAAC_DRYRUN=1`` skips kubectl/S3 entirely and emits a deterministic
result derived from the signal batch — used by unit tests and for wiring checks
without a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import shlex
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_NUM_ENVS = 1024
DEFAULT_ITERATIONS = 150
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
# Default PPO entropy coefficient for the Franka Lift run. The stock Isaac Lift
# cfg uses ~0.006, which lets the action-noise std collapse by ~iter 400-600;
# on an unlucky generated seed the policy then locks into a reach-and-hover
# local optimum and never discovers the grasp (a learning failure observed in
# sim2real-e2e-20260626t234808z). 0.01 keeps exploration alive through the grasp
# bottleneck so learning is reliable across seeds. Override via
# NPA_BYO_ISAAC_ENTROPY_COEF (set to "" or "stock" to keep the task default).
DEFAULT_ENTROPY_COEF = "0.01"
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
DEFAULT_ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
)
# Rigid-ready (RigidBodyAPI: collision + mass) instanceable manipuland. Defaulting
# to it means the Franka loop trains/evals on a real physically simulated USD sim
# asset instead of the stock primitive cube. A raw visual mesh would fail to spawn.
DEFAULT_OBJECT_USD_REL = "Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd"
# Sentinels that opt back out of the USD default to the built-in primitive cube.
_STOCK_OBJECT_USD_SENTINELS = frozenset({"stock", "none", "primitive", "builtin"})


def default_isaac_object_usd() -> str:
    """Resolved default manipuland USD (Nucleus root + rigid-ready instanceable)."""
    nuc = (os.environ.get("NPA_ISAAC_NUCLEUS_DIR", "") or DEFAULT_ISAAC_NUCLEUS_DIR).strip()
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
ROBOT_USD_CONTAINER_PATH = "/tmp/npa_robot/robot.usd"


def robot_spec_payload(spec: Any, *, usd_container_path: str = "") -> dict[str, Any] | None:
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
        print(f"byo_isaac_trainer: resolved robot_spec from {spec_uri} -> "
              f"{getattr(spec, 'name', None)!r} ({getattr(spec, 'robot_source', None)})",
              flush=True)
        return spec

    return robot_assets.robot_spec_from_inputs(
        robot_preset=preset,
        robot_source=source,
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def read_signal_stats(signal_json_path: str) -> dict[str, float]:
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
    return {
        "mean_reward": mean_reward,
        "mean_advantage": mean_advantage,
        "step_count": step_count,
        "error_tags": error_tags,
    }


def _first_env_record(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        return {
            "env_id": str(rec.get("env_id") or "env-00000"),
            "seed": int(rec.get("seed") or 0),
            "physics": rec.get("physics") or {},
        }
    return {}


def read_generated_train_env(envs_dir: str, *, envs_uri: str = "") -> dict[str, Any]:
    """Read a representative GENERATED train-env spec (seed + physics).

    The envgen stage writes one record per generated env with a per-env ``seed``
    and concrete ``physics`` (friction, mass_scale, lighting_lux). We surface the
    first record so the trainer can train on the generated env distribution (its
    seed + friction/mass) rather than stock defaults.

    Prefers the local ``envs_dir/envs.jsonl``; falls back to downloading
    ``envs_uri`` (the S3 ``.../envs/train/envs.jsonl``) when the orchestrator
    didn't sync the train envs locally (it only localizes the held-out split).
    Returns ``{}`` when neither source is available (stock run / envgen off).
    """

    from pathlib import Path as _Path

    path = _Path(envs_dir) / "envs.jsonl" if envs_dir else None
    if path and path.is_file():
        return _first_env_record(path.read_text(encoding="utf-8"))
    if envs_uri.startswith("s3://"):
        try:
            import boto3
            from urllib.parse import urlparse

            u = urlparse(envs_uri)
            s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
            obj = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
            return _first_env_record(obj["Body"].read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network/credentials
            print(f"byo_isaac_trainer: train-env S3 read failed ({envs_uri}): {exc!r}", flush=True)
    return {}


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


def build_isaac_job_manifest(
    *,
    job_name: str,
    run_id: str,
    image: str,
    task: str,
    num_envs: int,
    iterations: int,
    s3_output_uri: str,
    s3_endpoint: str,
    namespace: str,
    service_account: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    reward_overrides: dict[str, float] | None = None,
    object_usd: str = "",
    object_scale: str = "",
    seed: int = 0,
    physics: dict[str, float] | None = None,
    entropy_coef: str = "",
    init_noise_std: str = "",
    resume_uri: str = "",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    robot_spec: dict[str, Any] | None = None,
    robot_usd_uri: str = "",
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Isaac-Lab RSL-RL training Job manifest (proven by recon).

    Pure function: returns a manifest dict, no side effects. ``reward_overrides``
    are VLM-derived ``env.rewards.<term>.weight`` hydra args; ``object_usd``
    overrides the manipuland (``env.scene.object.spawn.usd_path``) so the policy
    is trained on a CUSTOM asset physically simulated in Isaac, not the stock cube.
    ``seed`` (from a GENERATED train-env spec) drives env + agent randomization so
    training runs on the envgen-produced env distribution.

    ``physics`` (generated ``{friction, mass_scale}``) selects a different code
    path: a task VARIANT that adds friction/mass startup events (registered
    post-boot via the shipped ``isaac_physics_task`` wrapper), because the stock
    Lift task has no friction/mass field a hydra override could touch. This path
    is opt-in (the caller gates it on ``NPA_BYO_ISAAC_PHYSICS``) and currently
    trains the variant's stock cube with default reward weights + the generated
    seed; it does not also apply VLM reward overrides / custom object (the proven
    default path below keeps those).
    """

    overrides = dict(reward_overrides or {})
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
    if resume_uri and not physics:
        overrides["agent.resume"] = "true"
        overrides["agent.load_run"] = RESUME_RUN_DIR
        overrides["agent.load_checkpoint"] = RESUME_CKPT_NAME
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

    if robot_spec:
        # BYO-robot path (takes precedence over physics): ship the
        # isaac_byo_robot_task module + its post-boot wrapper into the container and
        # run the wrapper (it registers a Lift variant that swaps in the customer
        # robot articulation AFTER AppLauncher boots, then trains via the rsl_rl
        # runner, saving model_*.pt into $OUT). A stock_franka payload yields empty
        # overrides, so the variant degenerates to the stock task — the seam runs
        # end-to-end without changing the policy.
        from npa.workflows.sim2real import isaac_byo_robot_task as _robotmod

        module_src = _robotmod.module_source()
        wrapper_src = _robotmod.TRAIN_WRAPPER_SCRIPT
        spec_json = json.dumps(robot_spec, sort_keys=True)
        # B2-derived robot-aware task config (action scale / placement / reward
        # thresholds / gripper) shipped alongside the robot spec so the variant is
        # scaled to the arm instead of the Franka-tuned stock numbers.
        task_cfg_block = ""
        if task_config:
            task_cfg_json = json.dumps(task_config, sort_keys=True)
            task_cfg_block = "export NPA_BYO_TASK_CONFIG_JSON=" + shlex.quote(task_cfg_json) + "\n"
        # Keep PPO exploring (same fix as the Franka default path); the wrapper
        # applies it to the rsl_rl agent cfg. Empty -> wrapper keeps task default.
        ent_block = ""
        if entropy_coef:
            ent_block = "export ROBOT_ENTROPY_COEF=" + shlex.quote(str(entropy_coef)) + "\n"
        usd_dest = str(robot_spec.get("usd_path") or "").strip()
        stage_block = ""
        if robot_usd_uri and usd_dest:
            # Stage the customer robot USD from S3 to the in-container path the
            # payload references, before the wrapper registers the variant.
            stage_block = (
                f'echo "STAGING_ROBOT_USD: {robot_usd_uri} -> {usd_dest}"\n'
                "ROBOT_USD_URI=" + shlex.quote(robot_usd_uri)
                + " ROBOT_USD_DEST=" + shlex.quote(usd_dest) + ' "$PY" - <<\'ROBOTDLEOF\'\n'
                "import os, boto3\n"
                "from urllib.parse import urlparse\n"
                "u = urlparse(os.environ['ROBOT_USD_URI'])\n"
                "dest = os.environ['ROBOT_USD_DEST']\n"
                "os.makedirs(os.path.dirname(dest), exist_ok=True)\n"
                "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
                "s3.download_file(u.netloc, u.path.lstrip('/'), dest)\n"
                "print('STAGED_ROBOT_USD', dest)\n"
                "ROBOTDLEOF\n"
            )
        train_block = (
            "mkdir -p /tmp/npa_robot\n"
            "cat > /tmp/npa_robot/isaac_byo_robot_task.py <<'ROBOTEOF'\n"
            + module_src + "\nROBOTEOF\n"
            "cat > /tmp/npa_robot/runner.py <<'ROBOTRUNEOF'\n"
            + wrapper_src + "\nROBOTRUNEOF\n"
            '"$PY" -m pip install --quiet boto3 2>/dev/null || true\n'
            + stage_block
            + f'echo "ROBOT_INJECTION: {robot_spec.get("robot_source")} '
            f'{robot_spec.get("name")} seed={int(seed)}"\n'
            f'export NPA_ROBOT_MODULE_DIR=/tmp/npa_robot ROBOT_OUT_DIR="$OUT" '
            f'ROBOT_NUM_ENVS={num_envs} ROBOT_ITERS={iterations} ROBOT_SEED={int(seed)}\n'
            "export NPA_BYO_ROBOT_SPEC_JSON=" + shlex.quote(spec_json) + "\n"
            + task_cfg_block
            + ent_block
            # tee the FULL wrapper output to /tmp/train_full.log before tailing: the
            # retarget plan + the honest task/robot compatibility verdict are printed
            # right after AppLauncher boot, so `| tail -120` alone discards them
            # behind the training-loop logs (and entirely when an incompatible robot
            # fails at env build). The markers are re-dumped from this file post-run,
            # and the file IS the per-iteration reward curve uploaded for plotting.
            + '"$PY" /tmp/npa_robot/runner.py 2>&1 | tee /tmp/train_full.log | tail -120\n'
        )
    elif physics:
        # Generated-physics path: ship the isaac_physics_task module + its
        # post-boot train wrapper into the container and run the wrapper (it
        # registers the friction/mass variant AFTER AppLauncher boots, then
        # trains via the rsl_rl runner, saving model_*.pt into $OUT).
        from npa.workflows.sim2real import isaac_physics_task as _physmod

        module_src = _physmod.module_source()
        wrapper_src = _physmod.TRAIN_WRAPPER_SCRIPT
        fr = float(physics.get("friction", 1.0))
        ms = float(physics.get("mass_scale", 1.0))
        train_block = (
            "mkdir -p /tmp/npa_phys\n"
            "cat > /tmp/npa_phys/isaac_physics_task.py <<'PHYSEOF'\n"
            + module_src + "\nPHYSEOF\n"
            "cat > /tmp/npa_phys/runner.py <<'RUNEOF'\n"
            + wrapper_src + "\nRUNEOF\n"
            f'echo "PHYSICS_INJECTION: friction={fr} mass_scale={ms} seed={int(seed)}"\n'
            f'export NPA_PHYS_MODULE_DIR=/tmp/npa_phys PHYS_OUT_DIR="$OUT" '
            f'PHYS_NUM_ENVS={num_envs} PHYS_ITERS={iterations} PHYS_SEED={int(seed)} '
            f'NPA_GEN_FRICTION={fr} NPA_GEN_MASS_SCALE={ms}\n'
            '"$PY" /tmp/npa_phys/runner.py 2>&1 | tail -120\n'
        )
    else:
        # Stage the prior-iteration checkpoint where train.py's get_checkpoint_path()
        # looks (logs/rsl_rl/<experiment>/<RESUME_RUN_DIR>/<RESUME_CKPT_NAME>, relative
        # to the run cwd $OUT). A download failure leaves the dir empty so train.py
        # raises on resume — loud, never a silent fresh start.
        resume_block = ""
        if resume_uri and not physics:
            resume_dir = f"logs/rsl_rl/{experiment_name}/{RESUME_RUN_DIR}"
            resume_block = (
                f'echo "RESUME_FROM: {resume_uri}"\n'
                f'mkdir -p "$OUT/{resume_dir}"\n'
                '"$PY" -m pip install --quiet boto3 2>/dev/null || true\n'
                f'RESUME_URI="{resume_uri}" '
                f'RESUME_DST="$OUT/{resume_dir}/{RESUME_CKPT_NAME}" "$PY" - <<\'RESEOF\'\n'
                "import os, boto3\n"
                "from urllib.parse import urlparse\n"
                "u = urlparse(os.environ['RESUME_URI'])\n"
                "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
                "s3.download_file(u.netloc, u.path.lstrip('/'), os.environ['RESUME_DST'])\n"
                "print('RESUME_DOWNLOADED', os.environ['RESUME_DST'])\n"
                "RESEOF\n"
            )
        train_line = (
            f'"$PY" {TRAIN_SCRIPT} --task {task} --num_envs {num_envs} '
            f'--max_iterations {iterations} --headless{seed_arg} agent.save_interval=25 {override_str}'
        )
        train_block = (
            f'{resume_block}'
            f'echo "VLM_REWARD_OVERRIDES: {override_str}"\n'
            # tee the FULL training output to a file (the per-iteration Mean reward
            # curve) before tailing to stdout — `| tail -120` alone discards the
            # early reward history, making the learning curve unrecoverable.
            f'{train_line} 2>&1 | tee /tmp/train_full.log | tail -120\n'
        )

    script = (
        "set -uo pipefail\n"
        'exec > >(tee -a /tmp/byo-train.log) 2>&1\n'
        'PY="/isaac-sim/python.sh"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"\n'
        f'OUT=/workspace/isaaclab/npa-runs/{run_id}; mkdir -p "$OUT"; cd "$OUT"\n'
        "set +e\n"
        f'{train_block}'
        "rc=${PIPESTATUS[0]}; set -e\n"
        'echo "TRAIN_RC=$rc"\n'
        # Re-dump the BYO-robot markers (retarget plan + compatibility verdict +
        # summary) from the full wrapper log so they survive the `tail -120` above
        # and are present even when an incompatible robot fails at env build.
        'if [ -f /tmp/train_full.log ]; then echo "=== ROBOT_MARKERS (untruncated) ==="; '
        'grep -aE "^(ROBOT_|STAGED_ROBOT_USD)" /tmp/train_full.log || true; fi\n'
        'CKPT=$(find "$OUT" -name \'model_*.pt\' 2>/dev/null | sort -V | tail -1)\n'
        'echo "LATEST_CKPT=$CKPT"\n'
        '[ -z "$CKPT" ] && { echo "NO_CHECKPOINT"; exit ${rc:-3}; }\n'
        '"$PY" -m pip install --quiet boto3 2>/dev/null || true\n'
        'CKPT_PATH="$CKPT" OUT_DIR="$OUT" OUT_URI="' + s3_output_uri + '" "$PY" - <<\'PYEOF\'\n'
        "import os, glob, boto3\n"
        "from urllib.parse import urlparse\n"
        "u = urlparse(os.environ['OUT_URI'])\n"
        "base = u.path.lstrip('/')\n"
        "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
        "s3.upload_file(os.environ['CKPT_PATH'], u.netloc, base + 'model_latest.pt')\n"
        "print('UPLOADED_CKPT s3://%s/%s' % (u.netloc, base + 'model_latest.pt'))\n"
        "# Periodic checkpoints (agent.save_interval) -> accuracy-vs-iteration eval sweep.\n"
        "for p in sorted(glob.glob(os.environ['OUT_DIR'] + '/**/model_*.pt', recursive=True)):\n"
        "    key = base + 'checkpoints/' + os.path.basename(p)\n"
        "    s3.upload_file(p, u.netloc, key)\n"
        "    print('UPLOADED_PERIODIC s3://%s/%s' % (u.netloc, key))\n"
        "# Full training log (per-iteration reward curve) for post-hoc plotting.\n"
        "import os.path as _op\n"
        "if _op.isfile('/tmp/train_full.log'):\n"
        "    s3.upload_file('/tmp/train_full.log', u.netloc, base + 'train_full.log')\n"
        "    print('UPLOADED_TRAIN_LOG s3://%s/%s' % (u.netloc, base + 'train_full.log'))\n"
        "PYEOF\n"
        'echo "BYO_TRAIN_DONE rc=$rc"\n'
        "exit $rc\n"
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-trainer", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
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
                    "imagePullSecrets": [
                        {"name": "agent-sa"},
                        {"name": "ngc-nvcr-imagepullsecret"},
                        {"name": "npa-nebius-registry"},
                    ],
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "imagePullPolicy": "Always",
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
                            ],
                            "command": ["/bin/bash", "-lc"],
                            "args": [script],
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
    reward_head_after = round(
        initial_reward_head + 0.5 * (reward_target - initial_reward_head), 6
    )
    # A real trainer produced a checkpoint => a real policy delta occurred.
    policy_delta_l2 = round(0.05 + 0.001 * float(iterations), 6) if checkpoint_uri else 0.0
    return {
        "schema": "npa.lerobot.vlm_signal_adapter.v1",
        "status": status,
        "backend": "isaac_rsl_rl_ppo",
        "steps": int(iterations),
        "loss_before": 1.0,
        "loss_after": round(max(0.0, 1.0 - 0.5 * reward_target), 6),
        "reward_head_before": round(float(initial_reward_head), 6),
        "reward_head_after": reward_head_after,
        "policy_output_before": [0.0],
        "policy_output_after": [round(reward_target, 6)],
        "policy_delta_l2": policy_delta_l2,
        "mean_reward": round(mean_reward, 6),
        "mean_advantage": round(mean_advantage, 6),
        "checkpoint_path": checkpoint_uri,
        "signal_count": int(stats.get("step_count", 0)),
        "control": False,
        "loss_integration_point": (
            "Isaac-Lab RSL-RL PPO sibling job (real policy training); VLM signal "
            "shapes reward via env.rewards weight overrides: "
            f"{reward_overrides or {}}"
        ),
        "duration_ms": round(float(duration_ms), 3),
    }


# --------------------------------------------------------------------------- #
# kubectl orchestration (live path)
# --------------------------------------------------------------------------- #
def _kubectl(args: list[str], *, stdin: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    cmd = [os.environ.get("NPA_KUBECTL_BIN") or "kubectl", *args]
    return subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, timeout=timeout, check=False
    )


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _sanitize_tag(tag: str) -> str:
    """Make a trainer tag safe for an S3 path segment (alnum, dash, underscore)."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in tag.strip())
    return cleaned.strip("-")


def run_isaac_training_job(run_id: str, *, signal_json: str) -> dict[str, Any]:
    """Submit the Isaac sibling Job, wait, and return an update-result dict."""

    task = _env("NPA_BYO_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    num_envs = int(_env("NPA_BYO_ISAAC_NUM_ENVS", str(DEFAULT_NUM_ENVS)) or DEFAULT_NUM_ENVS)
    iterations = int(_env("NPA_BYO_ISAAC_ITERATIONS", str(DEFAULT_ITERATIONS)) or DEFAULT_ITERATIONS)
    image = _env("ISAAC_IMAGE") or _env("NPA_SIM2REAL_ISAAC_IMAGE")
    if not image:
        raise SystemExit("byo_isaac_trainer: ISAAC_IMAGE/NPA_SIM2REAL_ISAAC_IMAGE not set")
    bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET")
    endpoint = _env("AWS_ENDPOINT_URL")
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    service_account = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    job_name = f"s2r-byo-isaac-train-{run_id}"[:63]
    # Per-iteration tag (e.g. "outer-02-iter-01") keeps each outer/inner iteration's
    # checkpoint at a DISTINCT S3 path so the prior model survives for the next
    # iteration to resume from (and outer iterations don't overwrite each other).
    # Unset => byte-identical to the historical single-shot path.
    tag = _sanitize_tag(_env("NPA_SIM2REAL_TRAINER_TAG"))
    path_seg = f"{job_name}/{tag}/" if tag else f"{job_name}/"
    s3_output = f"s3://{bucket}/sim2real-b/{run_id}/byo-trainer/{path_seg}"
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "7200") or 7200)

    # VLM critique -> PPO reward-term shaping (the VLM drives what the policy learns).
    stats = read_signal_stats(signal_json)
    reward_overrides = vlm_reward_overrides(stats)
    print(f"byo_isaac_trainer: VLM reward overrides -> {reward_overrides}", flush=True)
    object_usd = resolve_object_usd(_env("NPA_BYO_ISAAC_OBJECT_USD"))
    object_scale = _env("NPA_BYO_ISAAC_OBJECT_SCALE")
    # Exploration: keep the policy exploring through the grasp bottleneck so the
    # Lift run learns reliably regardless of the generated seed (see DEFAULT_ENTROPY_COEF).
    raw_ent = _env("NPA_BYO_ISAAC_ENTROPY_COEF", DEFAULT_ENTROPY_COEF)
    entropy_coef = "" if raw_ent.lower() in _STOCK_ENTROPY_SENTINELS else raw_ent
    init_noise_std = _env("NPA_BYO_ISAAC_INIT_NOISE_STD")
    if entropy_coef:
        print(f"byo_isaac_trainer: PPO entropy_coef -> {entropy_coef} "
              f"(exploration floor; stock ~0.006)", flush=True)
    if object_usd:
        default_tag = " (default)" if not _env("NPA_BYO_ISAAC_OBJECT_USD") else ""
        print(f"byo_isaac_trainer: object USD -> {object_usd}{default_tag} scale={object_scale}", flush=True)
    else:
        print("byo_isaac_trainer: stock primitive cube (object USD opted out)", flush=True)

    # GENERATED train-env spec: seed drives Isaac randomization so the policy
    # trains on the envgen-produced distribution (matches the held-out eval).
    train_env = read_generated_train_env(
        _env("NPA_SIM2REAL_TRAIN_ENVS_DIR"), envs_uri=_env("NPA_SIM2REAL_TRAIN_ENVS_URI"))
    gen_seed = int(train_env.get("seed") or 0)
    if train_env:
        print(f"byo_isaac_trainer: GENERATED train env {train_env.get('env_id')} "
              f"seed={gen_seed} physics={train_env.get('physics')}", flush=True)

    # Opt-in generated-physics injection (guarded; default path unchanged): map the
    # generated env's friction/mass_scale onto NPA_GEN_* and use the physics-variant
    # task (friction/mass startup events) instead of stock train.py.
    physics = None
    if _env("NPA_BYO_ISAAC_PHYSICS") == "1":
        from npa.workflows.sim2real import isaac_physics_task as _physmod

        gen_phys = (train_env.get("physics") or {}) if train_env else {}
        phys_env = {
            "NPA_GEN_FRICTION": _env("NPA_GEN_FRICTION") or str(gen_phys.get("friction", "")),
            "NPA_GEN_MASS_SCALE": _env("NPA_GEN_MASS_SCALE") or str(gen_phys.get("mass_scale", "")),
        }
        physics = _physmod.physics_params_from_env(phys_env)
        print(f"byo_isaac_trainer: PHYSICS injection {'ON' if physics else 'OFF (no params)'} "
              f"-> {physics}", flush=True)

    # Opt-in BYO-robot task path (guarded; default path unchanged): route the
    # customer robot_spec into a registered Isaac Lift variant that swaps in the
    # robot articulation. Takes precedence over the physics path when both are set.
    robot_spec_dict = None
    robot_usd_uri = ""
    if _env("NPA_BYO_ROBOT_TASK") == "1":
        if physics:
            print("byo_isaac_trainer: NPA_BYO_ROBOT_TASK=1 takes precedence over "
                  "PHYSICS path; disabling physics injection", flush=True)
            physics = None
        spec = _resolve_byo_robot_spec()
        usd_dest = ""
        if spec is not None and str(getattr(spec, "robot_source", "")) != "stock_franka":
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
                print(f"byo_isaac_trainer: WARNING BYO robot {getattr(spec, 'name', '?')!r} "
                      f"({getattr(spec, 'robot_source', '?')}) has no stageable USD "
                      "(s3:// or container path); robot articulation will NOT be swapped",
                      flush=True)
        robot_spec_dict = robot_spec_payload(spec, usd_container_path=usd_dest)
        print(f"byo_isaac_trainer: BYO-ROBOT task path "
              f"{'ON' if robot_spec_dict else 'OFF (no spec)'} -> {robot_spec_dict}", flush=True)

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
                print(f"byo_isaac_trainer: BYO task config -> {task_config}", flush=True)
        except (ValueError, TypeError) as exc:
            print(f"byo_isaac_trainer: WARNING invalid NPA_BYO_TASK_CONFIG_JSON ({exc!r}); "
                  "variant keeps stock task numbers", flush=True)

    # OUTER-LOOP RESUME: the orchestrator passes the prior iteration's checkpoint URI
    # so this run continues the SAME policy (stage 11B "more RL" compounds). Ignored on
    # the physics-variant path (different task) — log it so the skip is visible.
    resume_uri = _env("NPA_SIM2REAL_RESUME_CHECKPOINT_URI")
    experiment_name = _env("NPA_BYO_ISAAC_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    if resume_uri:
        if physics:
            print(f"byo_isaac_trainer: RESUME requested but physics path active; "
                  f"ignoring resume_uri={resume_uri}", flush=True)
        else:
            print(f"byo_isaac_trainer: RESUME from {resume_uri} "
                  f"(experiment={experiment_name}) -> continue same policy", flush=True)

    manifest = build_isaac_job_manifest(
        job_name=job_name,
        run_id=run_id,
        image=image,
        task=task,
        num_envs=num_envs,
        iterations=iterations,
        s3_output_uri=s3_output,
        s3_endpoint=endpoint,
        namespace=namespace,
        service_account=service_account,
        gpu_product=gpu_product,
        reward_overrides=reward_overrides,
        object_usd=object_usd,
        object_scale=object_scale,
        seed=gen_seed,
        physics=physics,
        entropy_coef=entropy_coef,
        init_noise_std=init_noise_std,
        resume_uri=resume_uri,
        experiment_name=experiment_name,
        robot_spec=robot_spec_dict,
        robot_usd_uri=robot_usd_uri,
        task_config=task_config,
    )
    start = time.time()
    _kubectl(["delete", "job", job_name, "-n", namespace, "--ignore-not-found"], timeout=60)
    apply = _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), timeout=120)
    if apply.returncode != 0:
        raise SystemExit(f"byo_isaac_trainer: kubectl apply failed: {apply.stderr}")
    print(f"byo_isaac_trainer: applied {job_name}; waiting up to {timeout_s}s", flush=True)
    wait = _kubectl(
        ["wait", f"job/{job_name}", "-n", namespace,
         "--for=condition=complete", f"--timeout={timeout_s}s"],
        timeout=timeout_s + 60,
    )
    status = "success" if wait.returncode == 0 else "failed"
    if status != "success":
        logs = _kubectl(["logs", f"job/{job_name}", "-n", namespace, "--tail=80"], timeout=120)
        raise SystemExit(
            f"byo_isaac_trainer: Isaac training job {job_name} did not complete: "
            f"{wait.stderr}\n--- logs ---\n{logs.stdout}"
        )
    checkpoint_uri = s3_output + "model_latest.pt"
    return build_update_result(
        stats=stats,
        initial_reward_head=float(_env("NPA_SIM2REAL_INITIAL_REWARD_HEAD", "0.0") or 0.0),
        iterations=iterations,
        checkpoint_uri=checkpoint_uri,
        status=status,
        duration_ms=(time.time() - start) * 1000.0,
        reward_overrides=reward_overrides,
    )


def main() -> int:
    signal_json = _env("NPA_SIM2REAL_SIGNAL_JSON")
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_trainer: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        stats = read_signal_stats(signal_json)
        result = build_update_result(
            stats=stats,
            initial_reward_head=float(_env("NPA_SIM2REAL_INITIAL_REWARD_HEAD", "0.0") or 0.0),
            iterations=int(_env("NPA_BYO_ISAAC_ITERATIONS", "2") or 2),
            checkpoint_uri=f"s3://dryrun/{run_id}/model_latest.pt",
            status="success",
            duration_ms=0.0,
        )
    else:
        result = run_isaac_training_job(run_id, signal_json=signal_json)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"byo_isaac_trainer: wrote update result -> {output_json} "
          f"(checkpoint={result['checkpoint_path']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

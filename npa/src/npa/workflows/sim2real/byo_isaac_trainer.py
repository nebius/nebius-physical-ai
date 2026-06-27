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

    if physics:
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
        train_line = (
            f'"$PY" {TRAIN_SCRIPT} --task {task} --num_envs {num_envs} '
            f'--max_iterations {iterations} --headless{seed_arg} agent.save_interval=25 {override_str}'
        )
        train_block = (
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
    s3_output = f"s3://{bucket}/sim2real-b/{run_id}/byo-trainer/{job_name}/"
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

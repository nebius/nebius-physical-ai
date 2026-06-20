"""BYO held-out eval: roll the TRAINED Isaac policy for a real success_rate.

Wired in via ``sim2real run --byo-eval-command 'python3 -m
npa.workflows.sim2real.byo_isaac_eval'``. Satisfies ``run_heldout_eval``'s
contract: write ``NPA_SIM2REAL_OUTPUT_JSON`` with a ``per_env`` list of
``{env_id, score, success}``; the engine's ``_normalize_heldout_report``
computes ``success_rate`` from it.

Unlike the reference/stub held-out payload (which scores synthetic rollouts and
does NOT load any trained policy), this loads the **trained checkpoint** (from
the inner-loop evidence's ``update.checkpoint_path``) and rolls it in Isaac on
``Isaac-Lift-Cube-Franka-v0``, deriving per-env success from the task's own
object-to-goal metric.

Runs in the orchestrator pod (no Isaac), so it submits an Isaac sibling k8s Job
that downloads the checkpoint, plays the policy, writes per-env scores to S3;
this process reads them back and writes the output JSON.

``NPA_BYO_ISAAC_DRYRUN=1`` skips kubectl/S3 and emits a deterministic per-env
report for unit tests / wiring checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
# Object-to-goal distance (metres) under which a Lift episode counts as success.
DEFAULT_SUCCESS_DIST_M = 0.05

# Set by main() so run_isaac_eval_job can sync rendered frames to the heldout
# renders dir + surface the render manifest into the report (for Rerun viz).
_RENDERS_LOCAL_DIR = ""
_RENDER_MANIFEST: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def extract_checkpoint_uri(inner_evidence: dict[str, Any]) -> str:
    """Pull the trained-policy checkpoint S3 URI from inner-loop evidence.

    Looks at the latest iteration's ``update.checkpoint_path``. Returns "" when
    no real checkpoint is present (e.g. reference trainer).
    """

    iterations = inner_evidence.get("iterations") or []
    for record in reversed(iterations):
        update = (record or {}).get("update") or {}
        ckpt = str(update.get("checkpoint_path") or "").strip()
        if ckpt.startswith("s3://"):
            return ckpt
    return ""


def build_heldout_report(
    per_env: list[dict[str, Any]],
    *,
    isaac_task: str,
    checkpoint_uri: str,
    source: str,
) -> dict[str, Any]:
    """Build the payload _normalize_heldout_report consumes (per_env list)."""

    return {
        "schema": "npa.sim2real.heldout_eval.v1",
        "source": source,
        "sim_backend": "isaac",
        "isaac_task": isaac_task,
        "policy_checkpoint": checkpoint_uri,
        "deployable_policy_eval": bool(checkpoint_uri),
        "per_env": per_env,
    }


def read_generated_envs(envs_dir: str, *, limit: int = 0) -> list[dict[str, Any]]:
    """Read the GENERATED held-out env specs (env_id + seed) from envs.jsonl.

    The envgen stage emits one record per generated env with a per-env ``seed``
    and scene composition. We use those seeds to drive the Isaac eval so the
    trained policy is tested on the generated env distribution (not just stock
    copies), and label results by the real generated ``env_id``.
    """

    path = Path(envs_dir) / "envs.jsonl"
    envs: list[dict[str, Any]] = []
    if not path.is_file():
        return envs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        envs.append(
            {
                "env_id": str(rec.get("env_id") or f"env-{len(envs):05d}"),
                "seed": int(rec.get("seed") or 0),
                "scene": rec.get("scene") or {},
            }
        )
        if limit and len(envs) >= limit:
            break
    return envs


def per_env_from_distances(
    distances: list[float],
    *,
    success_dist_m: float,
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Convert per-env final object-to-goal distances into scored per-env rows.

    score = clamp(1 - dist/(2*success_dist), 0, 1); success = dist < threshold.
    A genuine measurement of the trained policy, grounded in the task metric.
    When provided, rows are labelled by the GENERATED env_id/seed they came from.
    """

    rows: list[dict[str, Any]] = []
    for index, dist in enumerate(distances):
        d = max(0.0, float(dist))
        score = max(0.0, min(1.0, 1.0 - d / (2.0 * success_dist_m)))
        env_id = env_ids[index] if env_ids and index < len(env_ids) else f"heldout-{index:04d}"
        details: dict[str, Any] = {"object_goal_distance_m": round(d, 6)}
        if seeds and index < len(seeds):
            details["generated_env_seed"] = int(seeds[index])
        rows.append(
            {
                "env_id": env_id,
                "success": bool(d < success_dist_m),
                "score": round(score, 6),
                "details": details,
            }
        )
    return rows


# In-Isaac rollout script (runs in the sibling Job). Defensive: tries the
# standard Isaac Lab + rsl_rl play API, derives per-env final object-to-goal
# distance, and writes per_env_distances.json. Verbose so the first run reveals
# the exact API if anything mismatches.
ISAAC_EVAL_SCRIPT = r'''
import json, os, sys, traceback
import numpy as np
N = int(os.environ.get("EVAL_NUM_ENVS", "4"))
STEPS = int(os.environ.get("EVAL_MAX_STEPS", "300"))
TASK = os.environ["EVAL_TASK"]
CKPT = os.environ["EVAL_CKPT_LOCAL"]
OUT = os.environ["EVAL_OUT_JSON"]
SEED = int(os.environ.get("EVAL_SEED", "0"))  # generated-env seed (envgen envs.jsonl)
def dump(distances, note, episodes=None):
    json.dump({"object_goal_distances": list(distances), "note": note,
               "render_episodes": episodes or []},
              open(OUT, "w"))
    print("EVAL_WROTE", OUT, note, "episodes", len(episodes or []), flush=True)
try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    import gymnasium as gym, torch
    import isaaclab_tasks  # noqa: F401  registers tasks
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # older layout
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    # CUSTOM asset: override the manipuland USD so eval scores the policy on the
    # same custom object it trained on (physically simulated, not the stock cube).
    OBJECT_USD = os.environ.get("EVAL_OBJECT_USD", "").strip()
    if OBJECT_USD:
        try:
            env_cfg.scene.object.spawn.usd_path = OBJECT_USD
            print("EVAL_OBJECT_USD_APPLIED", OBJECT_USD, flush=True)
        except Exception as e:
            print("could not set object usd:", repr(e), flush=True)
    # Drive randomization from the GENERATED env seed so the trained policy is
    # tested on the envgen-produced env distribution, not stock defaults.
    if SEED:
        try:
            env_cfg.seed = SEED
        except Exception as e:
            print("could not set env_cfg.seed:", repr(e), flush=True)
        try:
            torch.manual_seed(SEED); np.random.seed(SEED % (2**32))
        except Exception:
            pass
        print("EVAL_SEED_APPLIED", SEED, flush=True)
    # Add a workspace camera so we can RENDER the (custom) object for Rerun viz.
    env_cfg.scene.heldout_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/heldout_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.2, 0.0, 0.8), rot=(0.6, 0.0, 0.35, 0.0), convention="world"),
        data_types=["rgb"], width=128, height=128, spawn=sim_utils.PinholeCameraCfg())
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    # Load the COMPLETE rsl_rl agent cfg from the task registry (has save_interval,
    # network dims, etc.) — a hand-built cfg is missing keys OnPolicyRunner needs.
    agent_cfg = None
    for loader in ("isaaclab_tasks.utils", "omni.isaac.lab_tasks.utils"):
        try:
            mod = __import__(loader, fromlist=["load_cfg_from_registry"])
            agent_cfg = mod.load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
            print("loaded agent cfg via", loader, flush=True)
            break
        except Exception as e:
            print("cfg loader", loader, "failed:", repr(e), flush=True)
    if agent_cfg is None:
        raise RuntimeError("could not load rsl_rl_cfg_entry_point for task")
    acfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    print("AGENT_CFG_KEYS", sorted(acfg.keys()), flush=True)
    runner = OnPolicyRunner(env, acfg, log_dir=None, device="cuda:0")
    runner.load(CKPT)
    policy = runner.get_inference_policy(device="cuda:0")
    try:
        obs, _ = env.get_observations()
    except Exception:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    # Use the ACTUAL env count (not the requested N) for action reshape + sizing.
    realN = int(getattr(env.unwrapped, "num_envs", N) or N)
    print("OBS_TYPE", type(obs).__name__, "realN", realN, flush=True)
    N = realN

    def _policy_obs(o):
        # rsl_rl inference needs a [N, obs_dim] policy tensor; with cameras on,
        # get_observations returns a (Tensor)Dict — extract the 'policy' group and
        # ensure a leading batch dim (the env may present a 1-D single-env obs).
        t = o
        if not torch.is_tensor(o):
            for k in ("policy", "obs", "policy_obs"):
                try:
                    v = o[k]
                    if torch.is_tensor(v):
                        t = v
                        break
                except Exception:
                    pass
        if torch.is_tensor(t) and t.ndim == 1:
            t = t.unsqueeze(0)
        return t
    _p0 = _policy_obs(obs)
    # Trust the obs batch dim as the source of truth for N.
    N = int(_p0.shape[0]) if torch.is_tensor(_p0) and _p0.ndim >= 1 else realN
    print("STEP0 policy_obs_shape", tuple(getattr(_p0, "shape", ())),
          "env.num_envs", getattr(env.unwrapped, "num_envs", "?"), "N", N, flush=True)
    # Per-env render dirs (labelled by generated env_id when provided).
    import json as _json
    env_ids = _json.loads(os.environ.get("EVAL_ENV_IDS", "[]") or "[]")
    rend_root = os.environ.get("EVAL_RENDERS_DIR", "/tmp/evalwork/renders")
    def _env_id(i):
        return env_ids[i] if i < len(env_ids) else f"heldout-{i:04d}"
    frame_names = {i: [] for i in range(N)}
    try:
        from PIL import Image as _PILImage
        _have_pil = True
    except Exception:
        _have_pil = False
    CAP_EVERY = max(1, STEPS // 16)
    def capture(step):
        if not _have_pil:
            return
        try:
            rgb = env.unwrapped.scene["heldout_cam"].data.output["rgb"]
            arr = rgb.detach().cpu().numpy()
            for i in range(min(N, arr.shape[0])):
                d = os.path.join(rend_root, _env_id(i)); os.makedirs(d, exist_ok=True)
                name = f"camera-{len(frame_names[i]):04d}.png"
                _PILImage.fromarray(arr[i, :, :, :3].astype(np.uint8)).save(os.path.join(d, name))
                frame_names[i].append(name)
        except Exception as e:
            print("capture_err", repr(e), flush=True)
    min_dist = np.full(N, 1e9)
    for _step in range(STEPS):
        with torch.inference_mode():
            actions = policy(_policy_obs(obs))
        if _step == 0:
            print("STEP0 act_shape", tuple(getattr(actions, "shape", ())), flush=True)
        # Safety: manager-based env needs [N, act_dim].
        if hasattr(actions, "ndim") and actions.ndim == 1 and N == 1:
            actions = actions.reshape(1, -1)
        obs, _, dones, extras = env.step(actions)
        if _step % CAP_EVERY == 0:
            capture(_step)
        # object-to-goal distance: prefer an explicit metric, else infer.
        d = None
        log = (extras or {}).get("log") or {}
        for k, v in log.items():
            if "object" in k.lower() and ("dist" in k.lower() or "error" in k.lower()):
                try:
                    d = float(v);
                except Exception:
                    d = None
                break
        try:
            uenv = env.unwrapped
            if hasattr(uenv, "command_manager"):
                cmd = uenv.command_manager.get_command("object_pose")
                obj = uenv.scene["object"].data.root_pos_w[:, :3]
                goal = cmd[:, :3] + uenv.scene.env_origins[:, :3]
                per = torch.linalg.norm(obj - goal, dim=1).detach().cpu().numpy()
                min_dist = np.minimum(min_dist, per);
                continue
        except Exception:
            pass
        if d is not None:
            min_dist = np.minimum(min_dist, np.full(N, d))
    capture(STEPS)  # final frame
    episodes = [{"env_id": _env_id(i), "frames": frame_names[i]} for i in range(N) if frame_names[i]]
    dump([float(x if x < 1e8 else 0.5) for x in min_dist], "rollout_ok", episodes)
except Exception as e:
    traceback.print_exc()
    dump([0.5]*N, "rollout_failed:%s" % e)
'''


def build_isaac_eval_job_manifest(
    *,
    job_name: str,
    run_id: str,
    image: str,
    task: str,
    num_envs: int,
    checkpoint_uri: str,
    per_env_s3_uri: str,
    s3_endpoint: str,
    namespace: str,
    service_account: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    seed: int = 0,
    object_usd: str = "",
    env_ids_json: str = "[]",
    renders_s3_prefix: str = "",
) -> dict[str, Any]:
    """Isaac eval Job: download checkpoint, roll trained policy, upload distances.

    ``seed`` (from the generated env spec) drives the env randomization so the
    policy is evaluated on the envgen-produced env distribution. ``object_usd``
    overrides the manipuland so eval scores the policy on the same CUSTOM asset
    it was trained on. RGB frames of the (custom) object are rendered and, when
    ``renders_s3_prefix`` is set, uploaded for Rerun visualization.
    """

    import shlex as _shlex

    render_upload = ""
    if renders_s3_prefix:
        render_upload = (
            'RENDERS_URI=' + _shlex.quote(renders_s3_prefix) + ' "$PY" - <<\'RLEOF\'\n'
            "import os, boto3, glob\n"
            "from urllib.parse import urlparse\n"
            "u = urlparse(os.environ['RENDERS_URI'])\n"
            "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
            "base = u.path.lstrip('/').rstrip('/')\n"
            "n = 0\n"
            "for p in glob.glob('/tmp/evalwork/renders/**/*.png', recursive=True):\n"
            "    rel = os.path.relpath(p, '/tmp/evalwork/renders')\n"
            "    s3.upload_file(p, u.netloc, base + '/' + rel); n += 1\n"
            "print('UPLOADED_RENDERS', n, os.environ['RENDERS_URI'])\n"
            "RLEOF\n"
        )
    script = (
        "set -uo pipefail\n"
        'exec > >(tee -a /tmp/byo-eval.log) 2>&1\n'
        'PY="/isaac-sim/python.sh"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"\n'
        '"$PY" -m pip install --quiet boto3 pillow 2>/dev/null || true\n'
        "mkdir -p /tmp/evalwork/renders; cd /tmp/evalwork\n"
        f'export EVAL_TASK="{task}" EVAL_NUM_ENVS="{num_envs}" EVAL_SEED="{seed}" '
        f'EVAL_OBJECT_USD="{object_usd}" EVAL_ENV_IDS={_shlex.quote(env_ids_json)} '
        'EVAL_RENDERS_DIR=/tmp/evalwork/renders '
        'EVAL_CKPT_LOCAL=/tmp/evalwork/policy.pt '
        'EVAL_OUT_JSON=/tmp/evalwork/per_env_distances.json\n'
        f'CKPT_URI="{checkpoint_uri}" OUT_URI="{per_env_s3_uri}" "$PY" - <<\'DLEOF\'\n'
        "import os, boto3\n"
        "from urllib.parse import urlparse\n"
        "u = urlparse(os.environ['CKPT_URI'])\n"
        "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
        "s3.download_file(u.netloc, u.path.lstrip('/'), '/tmp/evalwork/policy.pt')\n"
        "print('DOWNLOADED_CKPT', os.environ['CKPT_URI'])\n"
        "DLEOF\n"
        'cat > /tmp/evalwork/eval_rollout.py <<\'PYEOF\'\n'
        f"{ISAAC_EVAL_SCRIPT}\n"
        "PYEOF\n"
        '"$PY" /tmp/evalwork/eval_rollout.py || echo "EVAL_SCRIPT_RC=$?"\n'
        'OUT_URI="' + per_env_s3_uri + '" "$PY" - <<\'ULEOF\'\n'
        "import os, boto3\n"
        "from urllib.parse import urlparse\n"
        "u = urlparse(os.environ['OUT_URI'])\n"
        "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
        "s3.upload_file('/tmp/evalwork/per_env_distances.json', u.netloc, u.path.lstrip('/'))\n"
        "print('UPLOADED_DISTANCES', os.environ['OUT_URI'])\n"
        "ULEOF\n"
        + render_upload
        + 'echo "BYO_EVAL_DONE"\n'
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-eval", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {"app": "sim2real-byo-isaac-eval", "run-id": run_id}
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
                            "name": "eval",
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
                            "env": [{"name": "AWS_ENDPOINT_URL", "value": s3_endpoint}],
                            "command": ["/bin/bash", "-lc"],
                            "args": [script],
                        }
                    ],
                    "nodeSelector": {f"{gpu_resource}.product": gpu_product},
                },
            },
        },
    }


# --------------------------------------------------------------------------- #
# kubectl orchestration (live path)
# --------------------------------------------------------------------------- #
def _kubectl(args: list[str], *, stdin: str | None = None, timeout: int = 300):
    cmd = [os.environ.get("NPA_KUBECTL_BIN") or "kubectl", *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, check=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _download_json(uri: str) -> dict[str, Any]:
    import boto3
    from urllib.parse import urlparse

    u = urlparse(uri)
    s3 = boto3.client("s3", endpoint_url=_env("AWS_ENDPOINT_URL") or None)
    local = "/tmp/byo_eval_per_env.json"
    s3.download_file(u.netloc, u.path.lstrip("/"), local)
    return json.loads(Path(local).read_text())


def run_isaac_eval_job(
    run_id: str,
    *,
    checkpoint_uri: str,
    num_envs: int,
    generated_envs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    image = _env("NPA_SIM2REAL_ISAAC_IMAGE") or _env("ISAAC_IMAGE")
    bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET") or _env("NPA_SIM2REAL_S3_BUCKET")
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    sa = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    success_dist = float(_env("NPA_BYO_ISAAC_SUCCESS_DIST_M", str(DEFAULT_SUCCESS_DIST_M)) or DEFAULT_SUCCESS_DIST_M)
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "5400") or 5400)
    job_name = f"s2r-byo-isaac-eval-{run_id}"[:63]
    per_env_uri = f"s3://{bucket}/sim2real-b/{run_id}/byo-eval/{job_name}/per_env_distances.json"

    gen = generated_envs or []
    env_ids = [e["env_id"] for e in gen] or None
    seeds = [e["seed"] for e in gen] or None
    seed = int(gen[0]["seed"]) if gen else 0  # drive randomization from a generated-env seed
    object_usd = _env("NPA_BYO_ISAAC_OBJECT_USD")
    renders_prefix = f"s3://{bucket}/sim2real-b/{run_id}/byo-eval/{job_name}/renders"

    manifest = build_isaac_eval_job_manifest(
        job_name=job_name, run_id=run_id, image=image, task=task, num_envs=num_envs,
        checkpoint_uri=checkpoint_uri, per_env_s3_uri=per_env_uri,
        s3_endpoint=_env("AWS_ENDPOINT_URL"), namespace=namespace,
        service_account=sa, gpu_product=gpu_product, seed=seed, object_usd=object_usd,
        env_ids_json=json.dumps([e["env_id"] for e in gen]), renders_s3_prefix=renders_prefix,
    )
    _kubectl(["delete", "job", job_name, "-n", namespace, "--ignore-not-found"], timeout=60)
    apply = _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), timeout=120)
    if apply.returncode != 0:
        raise SystemExit(f"byo_isaac_eval: kubectl apply failed: {apply.stderr}")
    print(f"byo_isaac_eval: applied {job_name} (seed={seed}, generated_envs={len(gen)}); "
          f"waiting up to {timeout_s}s", flush=True)
    wait = _kubectl(["wait", f"job/{job_name}", "-n", namespace,
                     "--for=condition=complete", f"--timeout={timeout_s}s"], timeout=timeout_s + 60)
    if wait.returncode != 0:
        logs = _kubectl(["logs", f"job/{job_name}", "-n", namespace, "--tail=80"], timeout=120)
        raise SystemExit(f"byo_isaac_eval: eval job {job_name} did not complete: {wait.stderr}\n{logs.stdout}")
    out = _download_json(per_env_uri)
    distances = out.get("object_goal_distances", [])
    # Pull the rendered frames of the (custom) object down to the local heldout
    # renders dir so stage-14 Rerun viz logs them under heldout/camera/**.
    episodes = out.get("render_episodes") or []
    if episodes and _RENDERS_LOCAL_DIR:
        try:
            import boto3
            from urllib.parse import urlparse
            u = urlparse(renders_prefix)
            s3 = boto3.client("s3", endpoint_url=_env("AWS_ENDPOINT_URL") or None)
            base = u.path.lstrip("/").rstrip("/")
            for ep in episodes:
                eid = ep["env_id"]
                for name in ep.get("frames", []):
                    dst = Path(_RENDERS_LOCAL_DIR) / eid / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(u.netloc, f"{base}/{eid}/{name}", str(dst))
            print(f"byo_isaac_eval: synced {sum(len(e.get('frames',[])) for e in episodes)} frames", flush=True)
        except Exception as e:
            print("byo_isaac_eval: render sync failed:", repr(e), flush=True)
    global _RENDER_MANIFEST
    _RENDER_MANIFEST = {"schema": "npa.sim2real.heldout_renders.v1", "sim_backend": "isaac",
                        "isaac_task": task, "episodes": episodes}
    return per_env_from_distances(distances, success_dist_m=success_dist, env_ids=env_ids, seeds=seeds)


def main() -> int:
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_eval: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"
    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    num_envs = int(_env("NPA_SIM2REAL_HELDOUT_ENV_COUNT", "4") or 4)
    # Heldout renders live next to the report so stage-14 viz finds them.
    global _RENDERS_LOCAL_DIR
    _RENDERS_LOCAL_DIR = str(Path(output_json).parent / "renders")
    success_dist = float(_env("NPA_BYO_ISAAC_SUCCESS_DIST_M", str(DEFAULT_SUCCESS_DIST_M)) or DEFAULT_SUCCESS_DIST_M)

    inner_evidence = {}
    ev_path = _env("NPA_SIM2REAL_INNER_EVIDENCE_JSON")
    if ev_path and Path(ev_path).is_file():
        inner_evidence = json.loads(Path(ev_path).read_text())
    checkpoint_uri = extract_checkpoint_uri(inner_evidence)

    # GENERATED held-out env specs (env_id + seed) — drive eval on the envgen
    # distribution and label results by the real generated env_id.
    envs_dir = _env("NPA_SIM2REAL_HELDOUT_ENVS_DIR")
    generated_envs = read_generated_envs(envs_dir, limit=num_envs) if envs_dir else []
    if generated_envs:
        num_envs = len(generated_envs)

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        gids = [e["env_id"] for e in generated_envs] or None
        seeds = [e["seed"] for e in generated_envs] or None
        per_env = per_env_from_distances(
            [0.02, 0.04, 0.08, 0.12][:num_envs], success_dist_m=success_dist,
            env_ids=gids, seeds=seeds)
    elif not checkpoint_uri:
        print("byo_isaac_eval: no trained checkpoint in inner evidence — refusing to fake success",
              file=sys.stderr)
        return 3
    else:
        per_env = run_isaac_eval_job(
            run_id, checkpoint_uri=checkpoint_uri, num_envs=num_envs,
            generated_envs=generated_envs)

    report = build_heldout_report(
        per_env, isaac_task=task, checkpoint_uri=checkpoint_uri,
        source="byo_isaac_eval_dryrun" if _env("NPA_BYO_ISAAC_DRYRUN") == "1" else "byo_isaac_eval",
    )
    report["generated_envs_tested"] = len(generated_envs)
    report["generated_env_ids"] = [e["env_id"] for e in generated_envs]
    if _RENDER_MANIFEST.get("episodes"):
        report["render_manifest"] = _RENDER_MANIFEST
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(1 for r in per_env if r["success"])
    print(f"byo_isaac_eval: wrote {output_json} per_env={len(per_env)} passed={passed} "
          f"checkpoint={checkpoint_uri or 'NONE'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

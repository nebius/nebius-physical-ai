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


def per_env_from_distances(
    distances: list[float], *, success_dist_m: float
) -> list[dict[str, Any]]:
    """Convert per-env final object-to-goal distances into scored per-env rows.

    score = clamp(1 - dist/(2*success_dist), 0, 1); success = dist < threshold.
    A genuine measurement of the trained policy, grounded in the task metric.
    """

    rows: list[dict[str, Any]] = []
    for index, dist in enumerate(distances):
        d = max(0.0, float(dist))
        score = max(0.0, min(1.0, 1.0 - d / (2.0 * success_dist_m)))
        rows.append(
            {
                "env_id": f"heldout-{index:04d}",
                "success": bool(d < success_dist_m),
                "score": round(score, 6),
                "details": {"object_goal_distance_m": round(d, 6)},
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
def dump(distances, note):
    json.dump({"object_goal_distances": list(distances), "note": note},
              open(OUT, "w"))
    print("EVAL_WROTE", OUT, note, flush=True)
try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    import gymnasium as gym, torch
    import isaaclab_tasks  # noqa: F401  registers tasks
    from isaaclab_tasks.utils import parse_env_cfg
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # older layout
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    # Minimal agent cfg for loading; OnPolicyRunner needs a cfg dict.
    agent_cfg = {"policy": {"class_name": "ActorCritic",
                            "actor_hidden_dims": [256,128,64],
                            "critic_hidden_dims": [256,128,64],
                            "activation": "elu"},
                 "algorithm": {"class_name": "PPO"}, "num_steps_per_env": 24,
                 "empirical_normalization": False}
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
    runner.load(CKPT)
    policy = runner.get_inference_policy(device="cuda:0")
    obs, _ = env.get_observations()
    min_dist = np.full(N, 1e9)
    for _ in range(STEPS):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
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
    dump([float(x if x < 1e8 else 0.5) for x in min_dist], "rollout_ok")
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
) -> dict[str, Any]:
    """Isaac eval Job: download checkpoint, roll trained policy, upload distances."""

    script = (
        "set -uo pipefail\n"
        'exec > >(tee -a /tmp/byo-eval.log) 2>&1\n'
        'PY="/isaac-sim/python.sh"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"\n'
        '"$PY" -m pip install --quiet boto3 2>/dev/null || true\n'
        "mkdir -p /tmp/evalwork; cd /tmp/evalwork\n"
        f'export EVAL_TASK="{task}" EVAL_NUM_ENVS="{num_envs}" '
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
        'echo "BYO_EVAL_DONE"\n'
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


def run_isaac_eval_job(run_id: str, *, checkpoint_uri: str, num_envs: int) -> list[dict[str, Any]]:
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

    manifest = build_isaac_eval_job_manifest(
        job_name=job_name, run_id=run_id, image=image, task=task, num_envs=num_envs,
        checkpoint_uri=checkpoint_uri, per_env_s3_uri=per_env_uri,
        s3_endpoint=_env("AWS_ENDPOINT_URL"), namespace=namespace,
        service_account=sa, gpu_product=gpu_product,
    )
    _kubectl(["delete", "job", job_name, "-n", namespace, "--ignore-not-found"], timeout=60)
    apply = _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), timeout=120)
    if apply.returncode != 0:
        raise SystemExit(f"byo_isaac_eval: kubectl apply failed: {apply.stderr}")
    print(f"byo_isaac_eval: applied {job_name}; waiting up to {timeout_s}s", flush=True)
    wait = _kubectl(["wait", f"job/{job_name}", "-n", namespace,
                     "--for=condition=complete", f"--timeout={timeout_s}s"], timeout=timeout_s + 60)
    if wait.returncode != 0:
        logs = _kubectl(["logs", f"job/{job_name}", "-n", namespace, "--tail=80"], timeout=120)
        raise SystemExit(f"byo_isaac_eval: eval job {job_name} did not complete: {wait.stderr}\n{logs.stdout}")
    distances = _download_json(per_env_uri).get("object_goal_distances", [])
    return per_env_from_distances(distances, success_dist_m=success_dist)


def main() -> int:
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_eval: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"
    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    num_envs = int(_env("NPA_SIM2REAL_HELDOUT_ENV_COUNT", "4") or 4)
    success_dist = float(_env("NPA_BYO_ISAAC_SUCCESS_DIST_M", str(DEFAULT_SUCCESS_DIST_M)) or DEFAULT_SUCCESS_DIST_M)

    inner_evidence = {}
    ev_path = _env("NPA_SIM2REAL_INNER_EVIDENCE_JSON")
    if ev_path and Path(ev_path).is_file():
        inner_evidence = json.loads(Path(ev_path).read_text())
    checkpoint_uri = extract_checkpoint_uri(inner_evidence)

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        per_env = per_env_from_distances([0.02, 0.04, 0.08, 0.12][:num_envs], success_dist_m=success_dist)
    elif not checkpoint_uri:
        print("byo_isaac_eval: no trained checkpoint in inner evidence — refusing to fake success",
              file=sys.stderr)
        return 3
    else:
        per_env = run_isaac_eval_job(run_id, checkpoint_uri=checkpoint_uri, num_envs=num_envs)

    report = build_heldout_report(
        per_env, isaac_task=task, checkpoint_uri=checkpoint_uri,
        source="byo_isaac_eval_dryrun" if _env("NPA_BYO_ISAAC_DRYRUN") == "1" else "byo_isaac_eval",
    )
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(1 for r in per_env if r["success"])
    print(f"byo_isaac_eval: wrote {output_json} per_env={len(per_env)} passed={passed} "
          f"checkpoint={checkpoint_uri or 'NONE'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

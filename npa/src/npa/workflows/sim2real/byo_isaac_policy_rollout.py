"""BYO policy rollout: roll the CURRENT trained policy in Isaac for the VLM.

Wired in via ``sim2real run --byo-policy-command 'python3 -m
npa.workflows.sim2real.byo_isaac_policy_rollout'``. This closes the sim2real
loop: instead of the synthetic ``generate_action_rollouts`` fallback (random
actions + procedural PPM frames), the inner loop rolls the **current policy** in
Isaac on ``Isaac-Lift-Cube-Franka-v0`` and captures the policy's *actual*
behavior as RGB frames + actions. The Cosmos-Reason VLM then critiques those
real frames, and that critique shapes the next training step's reward — a
genuine closed loop rather than a critique of synthetic rollouts.

Contract (``run_policy_rollout_component`` → ``_run_policy_rollouts_via_command``):
read ``NPA_SIM2REAL_OUTPUT_DIR`` (where rollout dirs go) and
``NPA_SIM2REAL_ROLLOUT_COUNT`` / ``NPA_SIM2REAL_STEPS_PER_ROLLOUT``; write each
rollout as ``<output_dir>/rollout-NNNN/`` with ``camera-NNN.png`` frames and a
``manifest.json`` (schema ``npa.sim2real.action_rollout.v1``); write
``NPA_SIM2REAL_OUTPUT_JSON`` with ``{"rollout_dirs": [...]}``. The engine uses
those dirs (else falls back to synthetic).

**Which policy?** The current policy = the most-recent ``model_latest.pt`` the
BYO trainer has uploaded for this run (``s3://<bucket>/sim2real-b/<run_id>/
byo-trainer/.../model_latest.pt``). On the very first inner iteration none
exists yet, so an **untrained** rsl_rl policy is rolled — that is the correct RL
loop (critique the initial policy → shape training → re-roll the improved one).

Runs in the orchestrator pod (no Isaac), so it submits an Isaac sibling Job that
rolls the policy, captures per-env frames + actions, and uploads them to S3;
this process downloads them into the local rollout dirs.

``NPA_BYO_ISAAC_DRYRUN=1`` skips kubectl/S3 and emits deterministic rollout dirs
(procedural frames) for unit tests / wiring checks without a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
ROLLOUT_SCHEMA = "npa.sim2real.action_rollout.v1"
DEFAULT_TASK_DESCRIPTION = (
    "Move the manipulation object to the target while maintaining stable contact."
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a cluster)
# --------------------------------------------------------------------------- #
def build_rollout_manifest(
    *,
    rollout_id: str,
    frames: list[str],
    actions: list[dict[str, Any]],
    checkpoint_uri: str,
    is_trained: bool,
    task_description: str = DEFAULT_TASK_DESCRIPTION,
) -> dict[str, Any]:
    """Build an ``npa.sim2real.action_rollout.v1`` manifest for one rollout.

    Matches the schema the engine's synthetic ``generate_action_rollouts`` emits
    (so the VLM evaluator consumes it unchanged) and adds provenance fields
    making clear this is a REAL Isaac policy rollout, not synthetic.
    """

    return {
        "schema": ROLLOUT_SCHEMA,
        "rollout_id": rollout_id,
        "task_description": task_description,
        "steps": len(actions),
        "camera_observations": list(frames),
        "actions": list(actions),
        # Provenance: distinguishes a real policy rollout from the synthetic stub.
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "policy_checkpoint": checkpoint_uri,
        "policy_trained": bool(is_trained),
    }


def latest_checkpoint_uri(bucket: str, run_id: str, *, s3_endpoint: str = "") -> str:
    """Return the most-recent BYO-trainer ``model_latest.pt`` for this run.

    Scans ``s3://<bucket>/sim2real-b/<run_id>/byo-trainer/`` and returns the
    newest ``model_latest.pt`` URI, or ``""`` when none exists yet (first inner
    iteration → roll an untrained policy). Best-effort: any S3 error → "".
    """

    if not bucket or not run_id:
        return ""
    try:
        import boto3

        s3 = boto3.client("s3", endpoint_url=s3_endpoint or None)
        prefix = f"sim2real-b/{run_id}/byo-trainer/"
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
        print(f"byo_isaac_policy_rollout: checkpoint scan failed: {exc!r}", flush=True)
        return ""


def _write_ppm(path: Path, *, red: int, green: int, blue: int, size: int = 16) -> None:
    """Tiny solid-colour PPM frame (DRYRUN only — never used in the live path)."""

    header = f"P6\n{size} {size}\n255\n".encode("ascii")
    body = bytes([red & 255, green & 255, blue & 255]) * (size * size)
    path.write_bytes(header + body)


def write_dryrun_rollouts(
    output_dir: Path,
    *,
    count: int,
    steps_per_rollout: int,
    checkpoint_uri: str,
) -> list[str]:
    """Emit deterministic rollout dirs (procedural frames) for wiring tests.

    Honest: these are NOT real Isaac frames — DRYRUN only validates the contract
    (dir layout, manifest schema, rollout_dirs JSON). The live path rolls the
    real policy in Isaac.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    dirs: list[str] = []
    is_trained = bool(checkpoint_uri)
    for index in range(count):
        rollout_id = f"rollout-{index:04d}"
        rdir = output_dir / rollout_id
        rdir.mkdir(parents=True, exist_ok=True)
        frames: list[str] = []
        actions: list[dict[str, Any]] = []
        for step in range(steps_per_rollout):
            name = f"camera-{step:03d}.ppm"
            _write_ppm(rdir / name, red=60 + index * 10, green=40 + step * 8, blue=90)
            frames.append(name)
            actions.append({"step": step, "action": [0.01 * step, -0.01 * index, 0.0]})
        manifest = build_rollout_manifest(
            rollout_id=rollout_id, frames=frames, actions=actions,
            checkpoint_uri=checkpoint_uri, is_trained=is_trained,
        )
        (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        dirs.append(str(rdir))
    return dirs


# In-Isaac rollout script (runs in the sibling Job). Rolls the policy (trained
# checkpoint if present, else an untrained rsl_rl net), captures per-env RGB
# frames + the policy's actions, and uploads them to S3. Mirrors the proven
# byo_isaac_eval rollout (reset-first batched obs; whole obs (Tensor)Dict to the
# policy; per-env [realN,...] sizing) but records actions instead of distances.
ISAAC_ROLLOUT_SCRIPT = r'''
import json, os, sys, traceback
import numpy as np
N = int(os.environ.get("ROLLOUT_COUNT", "4"))
STEPS = int(os.environ.get("ROLLOUT_STEPS", "8"))
TASK = os.environ["ROLLOUT_TASK"]
CKPT = os.environ.get("ROLLOUT_CKPT_LOCAL", "").strip()
OUT_S3 = os.environ["ROLLOUT_OUT_S3"]            # s3 prefix for frames+actions
FRAMES_DIR = os.environ.get("ROLLOUT_FRAMES_DIR", "/tmp/rollwork/frames")
def upload_and_exit(rollouts, note):
    # rollouts: list of {rollout_id, frames:[names], actions:[{step,action}]}
    meta = {"rollouts": rollouts, "note": note, "policy_trained": bool(CKPT)}
    json.dump(meta, open("/tmp/rollwork/rollouts.json", "w"))
    print("ROLLOUT_WROTE", note, "rollouts", len(rollouts), flush=True)
    try:
        import boto3, glob
        from urllib.parse import urlparse
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
        u = urlparse(OUT_S3); base = u.path.lstrip("/").rstrip("/")
        s3.upload_file("/tmp/rollwork/rollouts.json", u.netloc, base + "/rollouts.json")
        n = 0
        for p in glob.glob(FRAMES_DIR + "/**/*.png", recursive=True):
            rel = os.path.relpath(p, FRAMES_DIR)
            s3.upload_file(p, u.netloc, base + "/" + rel); n += 1
        print("ROLLOUT_UPLOADED", n, OUT_S3, flush=True)
        print("BYO_ROLLOUT_DONE", flush=True)
    except Exception as e:
        print("rollout_upload_err", repr(e), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    import gymnasium as gym, torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
    OBJECT_USD = os.environ.get("ROLLOUT_OBJECT_USD", "").strip()
    if OBJECT_USD:
        try:
            env_cfg.scene.object.spawn.usd_path = OBJECT_USD
            print("ROLLOUT_OBJECT_USD_APPLIED", OBJECT_USD, flush=True)
        except Exception as e:
            print("could not set object usd:", repr(e), flush=True)
    env_cfg.scene.rollout_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/rollout_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.2, 0.0, 0.8), rot=(0.6, 0.0, 0.35, 0.0), convention="world"),
        data_types=["rgb"], width=128, height=128, spawn=sim_utils.PinholeCameraCfg())
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    agent_cfg = None
    for loader in ("isaaclab_tasks.utils", "omni.isaac.lab_tasks.utils"):
        try:
            mod = __import__(loader, fromlist=["load_cfg_from_registry"])
            agent_cfg = mod.load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
            break
        except Exception as e:
            print("cfg loader", loader, "failed:", repr(e), flush=True)
    if agent_cfg is None:
        raise RuntimeError("could not load rsl_rl_cfg_entry_point for task")
    acfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    runner = OnPolicyRunner(env, acfg, log_dir=None, device="cuda:0")
    trained = False
    if CKPT and os.path.isfile(CKPT):
        try:
            runner.load(CKPT); trained = True
            print("ROLLOUT_CKPT_LOADED", CKPT, flush=True)
        except Exception as e:
            print("ckpt_load_failed:", repr(e), "-> untrained policy", flush=True)
    else:
        print("ROLLOUT_UNTRAINED_POLICY (no checkpoint yet)", flush=True)
    policy = runner.get_inference_policy(device="cuda:0")
    realN = int(getattr(env.unwrapped, "num_envs", N) or N)
    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception:
        obs, _ = env.get_observations()
    N = realN
    def _to_batched(v):
        if not torch.is_tensor(v) or v.ndim != 1:
            return v
        n = int(v.shape[0])
        if realN > 1 and n % realN == 0:
            return v.reshape(realN, n // realN)
        return v.unsqueeze(0)
    def _batched_obs(o):
        if torch.is_tensor(o):
            return _to_batched(o)
        try:
            for k in list(o.keys()):
                o[k] = _to_batched(o[k])
        except Exception:
            pass
        return o
    obs = _batched_obs(obs)
    print("ROLLOUT realN", realN, "STEPS", STEPS, flush=True)
    try:
        from PIL import Image as _PILImage
        _have_pil = True
    except Exception:
        _have_pil = False
    rollout_ids = [f"rollout-{i:04d}" for i in range(N)]
    frame_names = {i: [] for i in range(N)}
    actions_log = {i: [] for i in range(N)}
    def capture(step):
        if not _have_pil:
            return
        try:
            rgb = env.unwrapped.scene["rollout_cam"].data.output["rgb"]
            arr = rgb.detach().cpu().numpy()
            for i in range(min(N, arr.shape[0])):
                d = os.path.join(FRAMES_DIR, rollout_ids[i]); os.makedirs(d, exist_ok=True)
                name = "camera-%03d.png" % len(frame_names[i])
                _PILImage.fromarray(arr[i, :, :, :3].astype(np.uint8)).save(os.path.join(d, name))
                frame_names[i].append(name)
        except Exception as e:
            print("capture_err", repr(e), flush=True)
    for _step in range(STEPS):
        with torch.inference_mode():
            actions = policy(_batched_obs(obs))
        if _step == 0:
            print("STEP0 act_shape", tuple(getattr(actions, "shape", ())), flush=True)
        if hasattr(actions, "ndim") and actions.ndim == 1:
            actions = actions.reshape(N, -1)
        capture(_step)
        a_np = actions.detach().cpu().numpy()
        for i in range(min(N, a_np.shape[0])):
            actions_log[i].append({"step": _step, "action": [round(float(x), 5) for x in a_np[i].tolist()]})
        obs, _, dones, extras = env.step(actions)
    capture(STEPS)
    rollouts = [{"rollout_id": rollout_ids[i], "frames": frame_names[i], "actions": actions_log[i]}
                for i in range(N)]
    upload_and_exit(rollouts, "rollout_ok" if trained else "rollout_ok_untrained")
except Exception as e:
    traceback.print_exc()
    upload_and_exit([], "rollout_failed:%s" % e)
'''


def build_isaac_rollout_job_manifest(
    *,
    job_name: str,
    run_id: str,
    image: str,
    task: str,
    rollout_count: int,
    steps_per_rollout: int,
    checkpoint_uri: str,
    out_s3_prefix: str,
    s3_endpoint: str,
    namespace: str,
    service_account: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    object_usd: str = "",
) -> dict[str, Any]:
    """Isaac policy-rollout Job: roll the policy, capture frames+actions, upload.

    When ``checkpoint_uri`` is set, downloads + loads it (trained policy); else
    rolls an untrained net. ``object_usd`` overrides the manipuland so the VLM
    critiques the policy on the same CUSTOM asset it trains on.
    """

    import shlex as _shlex

    download = ""
    if checkpoint_uri:
        download = (
            f'CKPT_URI={_shlex.quote(checkpoint_uri)} "$PY" - <<\'DLEOF\'\n'
            "import os, boto3\n"
            "from urllib.parse import urlparse\n"
            "u = urlparse(os.environ['CKPT_URI'])\n"
            "s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)\n"
            "s3.download_file(u.netloc, u.path.lstrip('/'), '/tmp/rollwork/policy.pt')\n"
            "print('DOWNLOADED_CKPT', os.environ['CKPT_URI'])\n"
            "DLEOF\n"
        )
        ckpt_local = "/tmp/rollwork/policy.pt"
    else:
        ckpt_local = ""

    script = (
        "set -uo pipefail\n"
        'exec > >(tee -a /tmp/byo-rollout.log) 2>&1\n'
        'PY="/isaac-sim/python.sh"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"\n'
        '"$PY" -m pip install --quiet boto3 pillow 2>/dev/null || true\n'
        "mkdir -p /tmp/rollwork/frames; cd /tmp/rollwork\n"
        f'export ROLLOUT_TASK="{task}" ROLLOUT_COUNT="{rollout_count}" '
        f'ROLLOUT_STEPS="{steps_per_rollout}" ROLLOUT_OBJECT_USD="{object_usd}" '
        f'ROLLOUT_CKPT_LOCAL="{ckpt_local}" '
        f'ROLLOUT_OUT_S3={_shlex.quote(out_s3_prefix)} '
        'ROLLOUT_FRAMES_DIR=/tmp/rollwork/frames\n'
        + download
        + 'cat > /tmp/rollwork/rollout.py <<\'PYEOF\'\n'
        f"{ISAAC_ROLLOUT_SCRIPT}\n"
        "PYEOF\n"
        '"$PY" /tmp/rollwork/rollout.py || echo "ROLLOUT_SCRIPT_RC=$?"\n'
        'echo "BYO_ROLLOUT_EXIT"\n'
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-rollout", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {"app": "sim2real-byo-isaac-rollout", "run-id": run_id}
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
                            "name": "rollout",
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


def materialize_rollout_dirs(
    output_dir: Path,
    meta: dict[str, Any],
    out_s3_prefix: str,
    *,
    checkpoint_uri: str,
    s3_endpoint: str,
) -> list[str]:
    """Download per-env frames from S3 and write local action_rollout.v1 dirs."""

    import boto3
    from urllib.parse import urlparse

    s3 = boto3.client("s3", endpoint_url=s3_endpoint or None)
    u = urlparse(out_s3_prefix)
    base = u.path.lstrip("/").rstrip("/")
    is_trained = bool(meta.get("policy_trained"))
    dirs: list[str] = []
    for roll in meta.get("rollouts", []) or []:
        rid = roll["rollout_id"]
        rdir = output_dir / rid
        rdir.mkdir(parents=True, exist_ok=True)
        for name in roll.get("frames", []):
            try:
                s3.download_file(u.netloc, f"{base}/{rid}/{name}", str(rdir / name))
            except Exception as exc:  # pragma: no cover - network
                print(f"byo_isaac_policy_rollout: frame download failed {rid}/{name}: {exc!r}", flush=True)
        manifest = build_rollout_manifest(
            rollout_id=rid, frames=roll.get("frames", []), actions=roll.get("actions", []),
            checkpoint_uri=checkpoint_uri, is_trained=is_trained,
        )
        (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        dirs.append(str(rdir))
    return dirs


def run_isaac_rollout_job(
    output_dir: Path,
    *,
    run_id: str,
    rollout_count: int,
    steps_per_rollout: int,
) -> list[str]:
    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    image = _env("NPA_SIM2REAL_ISAAC_IMAGE") or _env("ISAAC_IMAGE")
    bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET") or _env("NPA_SIM2REAL_S3_BUCKET")
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    sa = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    endpoint = _env("AWS_ENDPOINT_URL")
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "5400") or 5400)
    object_usd = _env("NPA_BYO_ISAAC_OBJECT_USD")

    checkpoint_uri = latest_checkpoint_uri(bucket, run_id, s3_endpoint=endpoint)
    # Unique per (run, outer, inner) — the engine sets a distinct OUTPUT_DIR name.
    job_suffix = output_dir.name or "iter"
    job_name = f"s2r-byo-isaac-roll-{run_id}-{job_suffix}"[:63]
    out_s3 = f"s3://{bucket}/sim2real-b/{run_id}/byo-rollouts/{job_suffix}"

    manifest = build_isaac_rollout_job_manifest(
        job_name=job_name, run_id=run_id, image=image, task=task,
        rollout_count=rollout_count, steps_per_rollout=steps_per_rollout,
        checkpoint_uri=checkpoint_uri, out_s3_prefix=out_s3, s3_endpoint=endpoint,
        namespace=namespace, service_account=sa, gpu_product=gpu_product,
        object_usd=object_usd,
    )
    _kubectl(["delete", "job", job_name, "-n", namespace, "--ignore-not-found"], timeout=60)
    apply = _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), timeout=120)
    if apply.returncode != 0:
        raise SystemExit(f"byo_isaac_policy_rollout: kubectl apply failed: {apply.stderr}")
    print(f"byo_isaac_policy_rollout: applied {job_name} "
          f"(checkpoint={checkpoint_uri or 'UNTRAINED'}); waiting up to {timeout_s}s", flush=True)
    wait = _kubectl(["wait", f"job/{job_name}", "-n", namespace,
                     "--for=condition=complete", f"--timeout={timeout_s}s"], timeout=timeout_s + 60)
    if wait.returncode != 0:
        logs = _kubectl(["logs", f"job/{job_name}", "-n", namespace, "--tail=80"], timeout=120)
        raise SystemExit(
            f"byo_isaac_policy_rollout: rollout job {job_name} did not complete: "
            f"{wait.stderr}\n{logs.stdout}")

    # Pull the rollouts manifest, then materialize local rollout dirs.
    import boto3
    from urllib.parse import urlparse

    u = urlparse(out_s3)
    s3 = boto3.client("s3", endpoint_url=endpoint or None)
    local_meta = output_dir / "rollouts.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    s3.download_file(u.netloc, f"{u.path.lstrip('/').rstrip('/')}/rollouts.json", str(local_meta))
    meta = json.loads(local_meta.read_text())
    return materialize_rollout_dirs(
        output_dir, meta, out_s3, checkpoint_uri=checkpoint_uri, s3_endpoint=endpoint)


def main() -> int:
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print("byo_isaac_policy_rollout: NPA_SIM2REAL_OUTPUT_JSON not set", file=sys.stderr)
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"
    output_dir = Path(_env("NPA_SIM2REAL_OUTPUT_DIR") or str(Path(output_json).parent))
    rollout_count = int(_env("NPA_SIM2REAL_ROLLOUT_COUNT", "4") or 4)
    steps_per_rollout = int(_env("NPA_SIM2REAL_STEPS_PER_ROLLOUT", "8") or 8)

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET")
        checkpoint_uri = latest_checkpoint_uri(bucket, run_id, s3_endpoint=_env("AWS_ENDPOINT_URL"))
        rollout_dirs = write_dryrun_rollouts(
            output_dir, count=rollout_count, steps_per_rollout=steps_per_rollout,
            checkpoint_uri=checkpoint_uri)
    else:
        rollout_dirs = run_isaac_rollout_job(
            output_dir, run_id=run_id, rollout_count=rollout_count,
            steps_per_rollout=steps_per_rollout)

    payload = {
        "schema": "npa.sim2real.policy_rollouts.v1",
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "rollout_dirs": rollout_dirs,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"byo_isaac_policy_rollout: wrote {output_json} rollout_dirs={len(rollout_dirs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

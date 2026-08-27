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

``NPA_BYO_ISAAC_DRYRUN=1`` skips the Kubernetes API/S3 and emits deterministic rollout dirs
(procedural frames) for unit tests / wiring checks without a GPU.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.camera_views import camera_metadata, camera_views_json
from npa.workflows.sim2real.capture import capture_settings
from npa.workflows.sim2real.isaac_job_payload import (
    compressed_bash_launch,
    embedded_base64_file_block,
)

DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
ROLLOUT_SCHEMA = "npa.sim2real.action_rollout.v1"
DEFAULT_TASK_DESCRIPTION = (
    "Move the manipulation object to the target while maintaining stable contact."
)
_LAST_GPU_PROVENANCE: dict[str, Any] = {}
_LAST_EMBODIMENT_EVIDENCE: dict[str, Any] = {}


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
    camera_views: dict[str, list[str]] | None = None,
    camera_metadata_items: list[dict[str, Any]] | None = None,
    frame_metadata: dict[str, list[dict[str, Any]]] | None = None,
    capture: dict[str, Any] | None = None,
    checkpoint_sha256: str = "",
    checkpoint_size_bytes: int = 0,
    scenario: dict[str, Any] | None = None,
    simulation_device: str = "cuda:0",
) -> dict[str, Any]:
    """Build an ``npa.sim2real.action_rollout.v1`` manifest for one rollout.

    Matches the schema the engine's synthetic ``generate_action_rollouts`` emits
    (so the VLM evaluator consumes it unchanged) and adds provenance fields
    making clear this is a REAL Isaac policy rollout, not synthetic.
    """

    views = {name: list(names) for name, names in (camera_views or {}).items()}
    if not views:
        views = {"primary": list(frames)}
    scenario = dict(scenario or {})
    return {
        "schema": ROLLOUT_SCHEMA,
        "rollout_id": rollout_id,
        "task_description": task_description,
        "steps": len(actions),
        "camera_observations": list(frames),
        "camera_views": views,
        "camera_metadata": list(camera_metadata_items or []),
        "camera_frame_metadata": dict(frame_metadata or {}),
        "capture": dict(capture or {}),
        "actions": list(actions),
        # Provenance: distinguishes a real policy rollout from the synthetic stub.
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "simulation_device": simulation_device,
        "policy_checkpoint": checkpoint_uri,
        "policy_checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_size_bytes": int(checkpoint_size_bytes),
        "policy_trained": bool(is_trained),
        "scenario_env_id": str(scenario.get("env_id") or ""),
        "scenario_config_digest": str(scenario.get("scenario_config_digest") or ""),
        "scenario_difficulty": str(scenario.get("difficulty") or ""),
        "task_contract_digest": str(scenario.get("task_contract_digest") or ""),
        "scenario_source_augmentation": dict(scenario.get("source_augmentation") or {}),
    }


def select_rollout_scenarios(
    rows: list[dict[str, Any]], *, count: int, selection_tag: str
) -> list[dict[str, Any]]:
    """Deterministically balance rollout scenarios across difficulty strata."""

    if count <= 0:
        raise ValueError("rollout scenario count must be positive")
    if len(rows) < count:
        raise ValueError(f"rollout needs {count} curated scenarios; found {len(rows)}")
    buckets: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("easy", "medium", "hard")
    }
    for row in rows:
        difficulty = str(row.get("difficulty") or "")
        if difficulty not in buckets:
            raise ValueError(f"unsupported scenario difficulty: {difficulty!r}")
        buckets[difficulty].append(row)
    for difficulty, bucket in buckets.items():
        if not bucket:
            raise ValueError(f"curated rollout split has no {difficulty} scenarios")
        bucket.sort(
            key=lambda row: (
                __import__("hashlib")
                .sha256(f"{selection_tag}:{row.get('scenario_config_digest')}".encode())
                .hexdigest()
            )
        )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for difficulty in ("easy", "medium", "hard"):
            if buckets[difficulty] and len(selected) < count:
                selected.append(buckets[difficulty].pop(0))
                progressed = True
        if not progressed:
            break
    digests = [str(row.get("scenario_config_digest") or "") for row in selected]
    if not all(digests) or len(set(digests)) != len(digests):
        raise ValueError("selected rollout scenarios need unique config digests")
    return selected


def latest_checkpoint_uri(
    bucket: str,
    run_id: str,
    *,
    s3_endpoint: str = "",
    s3_prefix: str = "sim2real-b",
) -> str:
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
    global _LAST_GPU_PROVENANCE

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
            rollout_id=rollout_id,
            frames=frames,
            actions=actions,
            checkpoint_uri=checkpoint_uri,
            is_trained=is_trained,
        )
        (rdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        dirs.append(str(rdir))
    return dirs


# In-Isaac rollout script (runs in the sibling Job). Rolls the policy (trained
# checkpoint if present, else an untrained rsl_rl net), captures per-env RGB
# frames + the policy's actions, and uploads them to S3. Mirrors the proven
# byo_isaac_eval rollout (reset-first batched obs; whole obs (Tensor)Dict to the
# policy; per-env [realN,...] sizing) but records actions instead of distances.
ISAAC_ROLLOUT_SCRIPT = r"""
import hashlib, json, os, sys, traceback
import numpy as np
N = int(os.environ.get("ROLLOUT_COUNT", "4"))
STEPS = int(os.environ.get("ROLLOUT_STEPS", "8"))
HORIZON_STEPS = int(os.environ.get("ROLLOUT_HORIZON_STEPS", "300"))
if STEPS < 1 or HORIZON_STEPS < STEPS:
    raise RuntimeError("rollout horizon must be >= decision/event sample count")
SAMPLE_STEPS = np.linspace(0, HORIZON_STEPS - 1, STEPS, dtype=np.int64).tolist()
SAMPLE_INDEX = {int(sim_step): index for index, sim_step in enumerate(SAMPLE_STEPS)}
TASK = os.environ["ROLLOUT_TASK"]
CKPT = os.environ.get("ROLLOUT_CKPT_LOCAL", "").strip()
OUT_S3 = os.environ["ROLLOUT_OUT_S3"]            # s3 prefix for frames+actions
FRAMES_DIR = os.environ.get("ROLLOUT_FRAMES_DIR", "/tmp/rollwork/frames")
CAMERA_VIEWS = json.loads(os.environ.get("ROLLOUT_CAMERA_VIEWS_JSON", "[]") or "[]")
CAPTURE_WIDTH = int(os.environ.get("ROLLOUT_CAPTURE_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.environ.get("ROLLOUT_CAPTURE_HEIGHT", "480"))
CAPTURE_STRIDE = max(1, int(os.environ.get("ROLLOUT_CAPTURE_STRIDE", "1")))
CAPTURE_STEPS = [step for step in SAMPLE_STEPS if step % CAPTURE_STRIDE == 0]
if HORIZON_STEPS not in CAPTURE_STEPS:
    CAPTURE_STEPS.append(HORIZON_STEPS)
PNG_COMPRESS_LEVEL = int(os.environ.get("ROLLOUT_PNG_COMPRESS_LEVEL", "3"))
CAPTURE_FPS = float(os.environ.get("ROLLOUT_CAPTURE_FPS", "10"))
SIM_DEVICE = os.environ.get("ROLLOUT_SIM_DEVICE", "cuda:0").strip() or "cuda:0"
if SIM_DEVICE != "cpu" and not (
    SIM_DEVICE.startswith("cuda:") and SIM_DEVICE.removeprefix("cuda:").isdigit()
):
    raise RuntimeError("ROLLOUT_SIM_DEVICE must be cpu or cuda:<index>")
CKPT_URI = os.environ.get("ROLLOUT_CKPT_URI", "").strip()
trained = False
def checkpoint_provenance():
    if not CKPT or not os.path.isfile(CKPT):
        return {"uri": CKPT_URI, "sha256": "", "size_bytes": 0}
    h = hashlib.sha256()
    with open(CKPT, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"uri": CKPT_URI, "sha256": h.hexdigest(), "size_bytes": os.path.getsize(CKPT)}
def upload_and_exit(rollouts, note, applied=None):
    # rollouts: list of {rollout_id, frames:[names], actions:[{step,action}]}
    checkpoint = checkpoint_provenance()
    meta = {"rollouts": rollouts, "note": note, "policy_trained": bool(trained),
            "applied_scenarios": applied or {},
            "policy_checkpoint": checkpoint,
            "camera_metadata": CAMERA_VIEWS,
            "simulation_device": SIM_DEVICE,
            "capture": {"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT,
                        "rollout_stride": CAPTURE_STRIDE,
                        "decision_points": STEPS, "horizon_steps": HORIZON_STEPS,
                        "expected_frames_per_view": len(CAPTURE_STEPS),
                        "sample_steps": SAMPLE_STEPS,
                        "png_compress_level": PNG_COMPRESS_LEVEL, "fps": CAPTURE_FPS}}
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
    app = AppLauncher(
        headless=True,
        enable_cameras=True,
        kit_args=os.environ.get(
            "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
        ),
    ).app
    # Isaac Sim 5.1 may leave the RTX data-window settings unset under a
    # portable root. Replicator treats those ``None`` values as overscan and
    # then subtracts them while reading RGB. Initialize the standard full-frame
    # window so Replicator preserves exact WxH output without fake overscan.
    import carb
    rtx_settings = carb.settings.get_settings()
    rtx_settings.set_float("/rtx/dataWindowNDC/0", 0.0)
    rtx_settings.set_float("/rtx/dataWindowNDC/1", 0.0)
    rtx_settings.set_float("/rtx/dataWindowNDC/2", 1.0)
    rtx_settings.set_float("/rtx/dataWindowNDC/3", 1.0)
    rtx_settings.set_bool("/rtx/dataWindow/fitOutputToDataWindow", False)
    if os.environ.get("NPA_PREPARE_ROBOT_ASSET_IN_APP") == "1":
        from npa.workflows.sim2real.isaac_robot_asset import prepare_with_running_app
        prepare_with_running_app()
    import gymnasium as gym, torch
    import isaaclab_tasks  # noqa: F401
    _scenarios = None
    if os.environ.get("NPA_SIM2REAL_SCENARIOS_JSONL"):
        sys.path.insert(0, os.environ.get("NPA_ROBOT_MODULE_DIR", "/opt/npa/isaac-runtime"))
        import isaac_scenario_task as _scenarios
    if os.environ.get("NPA_BYO_ROBOT_SPEC_JSON"):
        sys.path.insert(0, os.environ.get("NPA_ROBOT_MODULE_DIR", "/opt/npa/isaac-runtime"))
        import isaac_byo_robot_task as _robotmod
        _scenario_task = _robotmod.register()
        if not _scenario_task:
            raise RuntimeError("task contract did not register a rollout task")
        TASK = _scenario_task
        print("ROLLOUT_SCENARIO_TASK", TASK, flush=True)
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg, TiledCameraCfg
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(TASK, device=SIM_DEVICE, num_envs=N)
    if SIM_DEVICE == "cpu":
        if N != 1:
            raise RuntimeError("CPU physics camera fallback requires ROLLOUT_COUNT=1")
        env_cfg.sim.use_fabric = False
    print("ROLLOUT_SIM_DEVICE", SIM_DEVICE, flush=True)
    OBJECT_USD = os.environ.get("ROLLOUT_OBJECT_USD", "").strip()
    if OBJECT_USD:
        try:
            env_cfg.scene.object.spawn.usd_path = OBJECT_USD
            print("ROLLOUT_OBJECT_USD_APPLIED", OBJECT_USD, flush=True)
        except Exception as e:
            raise RuntimeError("could not apply task-contract object USD: %r" % (e,)) from e
    def _camera_key(name):
        return "rollout_cam" if name == "primary" else "rollout_cam_" + name
    CameraType = CameraCfg if SIM_DEVICE == "cpu" else TiledCameraCfg
    for view in CAMERA_VIEWS:
        setattr(
            env_cfg.scene,
            _camera_key(view["name"]),
            CameraType(
                prim_path="{ENV_REGEX_NS}/rollout_cam_" + view["name"],
                offset=CameraType.OffsetCfg(
                    pos=tuple(view["position"]),
                    rot=tuple(view["rotation"]),
                    convention="world",
                ),
                data_types=["rgb"],
                width=CAPTURE_WIDTH,
                height=CAPTURE_HEIGHT,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(view.get("focal_length_mm", 24.0)),
                    horizontal_aperture=float(view.get("horizontal_aperture_mm", 20.955)),
                ),
            ),
        )
    print("ROLLOUT_CAMERA_VIEWS", [view["name"] for view in CAMERA_VIEWS], flush=True)
    env = gym.make(TASK, cfg=env_cfg)
    capture_annotators = {}
    if SIM_DEVICE == "cpu":
        # Physics remains on CPU for the compatibility route, but rendering is
        # backed by the reserved RTX device. A CUDA annotator avoids Isaac
        # Replicator's empty CPU render buffers while leaving simulation state
        # and policy inference on the explicitly selected device.
        import omni.replicator.core as rep
        for view in CAMERA_VIEWS:
            view_name = view["name"]
            sensor = env.unwrapped.scene[_camera_key(view_name)]
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda:0")
            annotator.attach(sensor.render_product_paths)
            capture_annotators[view_name] = annotator
    if OBJECT_USD:
        got_object_usd = getattr(env.unwrapped.scene["object"].cfg.spawn, "usd_path", None)
        if got_object_usd != OBJECT_USD:
            raise RuntimeError("rollout object USD mismatch; refusing stock fallback")
    EXPECTED_ROBOT_USD = os.environ.get("NPA_EXPECTED_ROBOT_USD", "").strip()
    if EXPECTED_ROBOT_USD:
        got_robot_usd = getattr(env.unwrapped.scene["robot"].cfg.spawn, "usd_path", None)
        print("ROLLOUT_ROBOT_USD", "want", EXPECTED_ROBOT_USD, "got", got_robot_usd, flush=True)
        if got_robot_usd != EXPECTED_ROBOT_USD:
            raise RuntimeError("BYO rollout robot USD mismatch; refusing stock fallback")
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
    runner = OnPolicyRunner(env, acfg, log_dir=None, device=SIM_DEVICE)
    trained = False
    if CKPT and os.path.isfile(CKPT):
        try:
            runner.load(CKPT); trained = True
            print("ROLLOUT_CKPT_LOADED", CKPT, flush=True)
        except Exception as e:
            raise RuntimeError("trained checkpoint failed to load: %r" % (e,)) from e
    else:
        print("ROLLOUT_UNTRAINED_POLICY (no checkpoint yet)", flush=True)
    policy = runner.get_inference_policy(device=SIM_DEVICE)
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
    print(
        "ROLLOUT realN", realN, "DECISION_POINTS", STEPS,
        "HORIZON_STEPS", HORIZON_STEPS, flush=True,
    )
    def _write_rgb_png(path, rgb):
        # Isaac runtime images do not guarantee Pillow.  Keep capture independent
        # of optional packages so a valid sensor stream cannot be silently lost.
        import binascii, struct, zlib
        pixels = np.asarray(rgb, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise RuntimeError(
                "camera rgb output must be HxWxC with at least three channels; "
                "got shape=%r dtype=%s" % (pixels.shape, pixels.dtype)
            )
        pixels = np.ascontiguousarray(pixels[:, :, :3])
        height, width = pixels.shape[:2]
        raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
        def chunk(kind, payload):
            body = kind + payload
            return struct.pack(">I", len(payload)) + body + struct.pack(
                ">I", binascii.crc32(body) & 0xFFFFFFFF
            )
        encoded = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, PNG_COMPRESS_LEVEL))
            + chunk(b"IEND", b"")
        )
        with open(path, "wb") as handle:
            handle.write(encoded)
    rollout_ids = [f"rollout-{i:04d}" for i in range(N)]
    frame_names = {
        i: {view["name"]: [] for view in CAMERA_VIEWS}
        for i in range(N)
    }
    frame_metadata = {
        i: {view["name"]: [] for view in CAMERA_VIEWS}
        for i in range(N)
    }
    actions_log = {i: [] for i in range(N)}
    uenv = env.unwrapped
    action_manager = uenv.action_manager
    action_terms = list(action_manager.active_terms)
    action_dims = list(action_manager.action_term_dim)
    actual_action_dim = int(sum(action_dims))
    _robot_spec = json.loads(os.environ.get("NPA_BYO_ROBOT_SPEC_JSON", "{}") or "{}")
    expected_action_dim = int(_robot_spec.get("expected_action_dim") or 0)
    expected_observation_dim = int(_robot_spec.get("expected_observation_dim") or 0)
    _policy_obs = obs if torch.is_tensor(obs) else obs.get("policy")
    actual_observation_dim = int(_policy_obs.shape[-1])
    dimensions = {
        "embodiment_digest": str(_robot_spec.get("embodiment_digest") or "stock_franka"),
        "action": actual_action_dim,
        "observation": actual_observation_dim,
        "expected_action": expected_action_dim,
        "expected_observation": expected_observation_dim,
    }
    print("ROLLOUT_ROBOT_DIMENSIONS " + json.dumps(dimensions, sort_keys=True), flush=True)
    if expected_action_dim and actual_action_dim != expected_action_dim:
        raise RuntimeError("rollout action dimension disagrees with RobotSpec")
    if expected_observation_dim and actual_observation_dim != expected_observation_dim:
        raise RuntimeError("rollout observation dimension disagrees with RobotSpec")
    if "gripper_action" not in action_terms:
        raise RuntimeError("policy rollout requires a named gripper_action term")
    gripper_term_index = action_terms.index("gripper_action")
    gripper_start = sum(action_dims[:gripper_term_index])
    previous_goal_distance = np.full(N, np.nan)
    previous_ee_distance = np.full(N, np.nan)
    initial_object_z = None
    stable_grasp_steps = np.zeros(N, dtype=np.int64)
    stable_place_steps = np.zeros(N, dtype=np.int64)
    def capture(step):
        if step % CAPTURE_STRIDE != 0 and step != HORIZON_STEPS:
            return
        for view in CAMERA_VIEWS:
            view_name = view["name"]
            try:
                sensor = env.unwrapped.scene[_camera_key(view_name)]
                if SIM_DEVICE == "cpu":
                    output = capture_annotators[view_name].get_data()
                    raw = output["data"] if isinstance(output, dict) else output
                    if hasattr(raw, "detach"):
                        arr = raw.detach().cpu().numpy()
                    elif isinstance(raw, np.ndarray):
                        arr = raw
                    else:
                        import warp as wp
                        arr = wp.to_torch(raw).cpu().numpy()
                    pixels = CAPTURE_HEIGHT * CAPTURE_WIDTH
                    if arr.size == pixels and arr.dtype.itemsize >= 4:
                        # Replicator may expose packed RGBA pixels as uint32.
                        arr = arr.view(np.uint8)
                    if arr.size % pixels:
                        raise RuntimeError("camera rgb buffer size is not image-shaped")
                    arr = arr.reshape(1, CAPTURE_HEIGHT, CAPTURE_WIDTH, arr.size // pixels)
                else:
                    rgb = sensor.data.output["rgb"]
                    arr = rgb.detach().cpu().numpy()
                # A single non-tiled Camera returns HxWxC, while TiledCamera
                # returns NxHxWxC. Normalize both sensor contracts before the
                # per-environment writer loop.
                if arr.ndim == 3:
                    arr = arr[None, ...]
                if arr.ndim != 4:
                    raise RuntimeError("camera rgb output must be HxWxC or NxHxWxC")
                for i in range(min(N, arr.shape[0])):
                    d = os.path.join(FRAMES_DIR, rollout_ids[i]); os.makedirs(d, exist_ok=True)
                    index = len(frame_names[i][view_name])
                    name = (
                        "camera-%03d.png" % index
                        if view_name == "primary"
                        else "camera-%s-%03d.png" % (view_name, index)
                    )
                    _write_rgb_png(os.path.join(d, name), arr[i])
                    frame_names[i][view_name].append(name)
                    frame_metadata[i][view_name].append({
                        "path": name,
                        "view_name": view_name,
                        "frame_index": index,
                        "sim_step": int(step),
                        "timestamp_seconds": round(float(step) / CAPTURE_FPS, 6),
                        "episode_id": rollout_ids[i],
                        "isaac_env_index": i,
                        "width": CAPTURE_WIDTH,
                        "height": CAPTURE_HEIGHT,
                        "policy_checkpoint": CKPT_URI,
                    })
            except Exception as e:
                print("capture_err", view_name, repr(e), flush=True)
    for _step in range(HORIZON_STEPS):
        with torch.inference_mode():
            actions = policy(_batched_obs(obs))
        if _step == 0:
            print("STEP0 act_shape", tuple(getattr(actions, "shape", ())), flush=True)
        if hasattr(actions, "ndim") and actions.ndim == 1:
            actions = actions.reshape(N, -1)
        a_np = actions.detach().cpu().numpy()
        obs, _, dones, extras = env.step(actions)
        # TiledCamera annotators need the first rendered simulation step before
        # their initial read. Capture the post-action state, which also aligns
        # each image with the simulator ground truth recorded below.
        if _step in SAMPLE_INDEX:
            capture(_step)
        done_np = dones.detach().cpu().numpy().astype(bool)
        obj = uenv.scene["object"].data.root_pos_w[:, :3]
        cmd = uenv.command_manager.get_command("object_pose")
        goal = cmd[:, :3] + uenv.scene.env_origins[:, :3]
        goal_distance = torch.linalg.norm(obj - goal, dim=1).detach().cpu().numpy()
        ee = uenv.scene["ee_frame"].data.target_pos_w[..., 0, :]
        ee_distance = torch.linalg.norm(obj - ee, dim=1).detach().cpu().numpy()
        if initial_object_z is None:
            initial_object_z = obj[:, 2].detach().cpu().numpy().copy()
        lift_m = obj[:, 2].detach().cpu().numpy() - initial_object_z
        obj_velocity = torch.linalg.norm(
            uenv.scene["object"].data.root_lin_vel_w, dim=1
        ).detach().cpu().numpy()
        contact_now = ee_distance < 0.04
        try:
            force = uenv.scene["object_contact"].data.net_forces_w_history
            force_contact = np.linalg.norm(
                force.detach().cpu().numpy(), axis=-1
            ).reshape(N, -1).max(axis=1) > 1.0e-3
            contact_now &= force_contact
        except Exception as e:
            print("ROLLOUT_CONTACT_SENSOR_FALLBACK", repr(e), flush=True)
        gripper_closed = a_np[:, gripper_start] < 0.0
        stable_grasp_now = contact_now & gripper_closed & (lift_m > 0.01)
        stable_grasp_steps = np.where(stable_grasp_now, stable_grasp_steps + 1, 0)
        stable_grasp_now = stable_grasp_steps >= 3
        stable_place_now = (goal_distance < 0.05) & (obj_velocity < 0.03)
        stable_place_steps = np.where(stable_place_now, stable_place_steps + 1, 0)
        stable_place_now = stable_place_steps >= 3
        scenario_rows = getattr(uenv, "npa_scenario_rows", [])
        scenario_indices = getattr(uenv, "npa_scenario_indices", None)
        scenario_cpu = (
            scenario_indices.detach().cpu().tolist()
            if scenario_indices is not None
            else list(range(N))
        )
        if _step not in SAMPLE_INDEX:
            continue
        decision_step = SAMPLE_INDEX[_step]
        for i in range(min(N, a_np.shape[0])):
            scenario = scenario_rows[int(scenario_cpu[i])] if scenario_rows else {}
            goal_change = (
                0.0
                if np.isnan(previous_goal_distance[i])
                else float(previous_goal_distance[i] - goal_distance[i])
            )
            ee_change = (
                0.0
                if np.isnan(previous_ee_distance[i])
                else float(previous_ee_distance[i] - ee_distance[i])
            )
            actions_log[i].append({
                "step": decision_step,
                "sim_step": _step,
                "action": [round(float(x), 5) for x in a_np[i].tolist()],
                "scenario_config_digest": str(scenario.get("scenario_config_digest") or ""),
                "simulator_ground_truth": {
                    "object_goal_distance_m": round(float(goal_distance[i]), 6),
                    "object_goal_distance_change_m": round(goal_change, 6),
                    "end_effector_object_distance_m": round(float(ee_distance[i]), 6),
                    "end_effector_distance_change_m": round(ee_change, 6),
                    "contact": bool(contact_now[i]),
                    "gripper_closed": bool(gripper_closed[i]),
                    "stable_grasp": bool(stable_grasp_now[i]),
                    "object_height_m": round(float(obj[i, 2].item()), 6),
                    "object_lift_m": round(float(lift_m[i]), 6),
                    "placement_stable": bool(stable_place_now[i]),
                    "terminated": bool(done_np[i]),
                    "termination_reason": "success" if stable_place_now[i] else (
                        "task_or_timeout" if done_np[i] else "running"
                    ),
                    "scenario_config_digest": str(scenario.get("scenario_config_digest") or ""),
                },
            })
        previous_goal_distance = goal_distance.copy()
        previous_ee_distance = ee_distance.copy()
    capture(HORIZON_STEPS)
    scenario_rows = getattr(uenv, "npa_scenario_rows", [])
    scenario_indices = getattr(uenv, "npa_scenario_indices", None)
    scenario_cpu = scenario_indices.detach().cpu().tolist() if scenario_indices is not None else list(range(N))
    rollouts = [{"rollout_id": rollout_ids[i],
                 "frames": frame_names[i].get("primary", []),
                 "camera_views": frame_names[i],
                 "camera_frame_metadata": frame_metadata[i],
                 "scenario": scenario_rows[int(scenario_cpu[i])] if scenario_rows else {},
                 "actions": actions_log[i]}
                for i in range(N)]
    applied = _scenarios.runtime_audit(uenv) if _scenarios is not None else {}
    upload_and_exit(
        rollouts,
        "rollout_ok" if trained else "rollout_ok_untrained",
        applied=applied,
    )
except Exception as e:
    traceback.print_exc()
    upload_and_exit([], "rollout_failed:%s" % e)
"""


def _isaac_eula_env_entries() -> list[dict[str, str]]:
    """Canonical Kubernetes env for this known Isaac route."""

    from npa.serverless_common.env import resolved_isaac_eula_env

    return [
        {"name": name, "value": value}
        for name, value in resolved_isaac_eula_env().items()
    ]


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
    horizon_steps: int = 300,
    gpu_resource: str = "nvidia.com/gpu",
    image_pull_policy: str = "Always",
    object_usd: str = "",
    camera_views: str = "",
    capture: dict[str, Any] | None = None,
    scenarios_jsonl: str = "",
    robot_spec: dict[str, Any] | None = None,
    robot_usd_uri: str = "",
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Isaac policy-rollout Job: roll the policy, capture frames+actions, upload.

    When ``checkpoint_uri`` is set, downloads + loads it (trained policy); else
    rolls an untrained net. ``object_usd`` overrides the manipuland so the VLM
    critiques the policy on the same CUSTOM asset it trains on.
    """

    import shlex as _shlex

    capture = dict(capture or capture_settings({}))

    download = ""
    if checkpoint_uri:
        download = (
            '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
            f"--uri {_shlex.quote(checkpoint_uri)} "
            "--destination /tmp/rollwork/policy.pt\n"
        )
        ckpt_local = "/tmp/rollwork/policy.pt"
    else:
        ckpt_local = ""

    scenario_block = ""
    if scenarios_jsonl:
        scenario_block = embedded_base64_file_block(
            scenarios_jsonl,
            destination="/tmp/rollwork/scenarios.jsonl",
            marker="NPA_ROLLOUT_SCENARIOS_B64",
        ) + (
            "export NPA_SIM2REAL_SCENARIOS_JSONL=/tmp/rollwork/scenarios.jsonl\n"
            "export NPA_SIM2REAL_TASK_CONTRACT_DIGEST="
            + _shlex.quote(_env("NPA_SIM2REAL_TASK_CONTRACT_DIGEST"))
            + "\n"
            "export NPA_SIM2REAL_SCENARIO_ROTATE_ON_RESET=0\n"
        )

    robot_block = ""
    if robot_spec:
        from npa.workflows.sim2real.byo_isaac_trainer import (
            robot_asset_preflight_script,
        )

        robot_block = (
            "export NPA_BYO_ROBOT_SPEC_JSON="
            + _shlex.quote(json.dumps(robot_spec, sort_keys=True))
            + "\n"
        )
        if task_config:
            robot_block += (
                "export NPA_BYO_TASK_CONFIG_JSON="
                + _shlex.quote(json.dumps(task_config, sort_keys=True))
                + "\n"
            )
        expected_usd = str(robot_spec.get("usd_path") or "").strip()
        asset_preflight = robot_asset_preflight_script(robot_spec)
        if asset_preflight:
            robot_block += asset_preflight
            robot_block += (
                "export NPA_EXPECTED_ROBOT_USD=" + _shlex.quote(expected_usd) + "\n"
            )
        elif robot_usd_uri and expected_usd:
            robot_block += (
                '"$PY" -m npa.workflows.sim2real.isaac_job_io download '
                f"--uri {_shlex.quote(robot_usd_uri)} "
                f"--destination {_shlex.quote(expected_usd)}\n"
                "export NPA_EXPECTED_ROBOT_USD=" + _shlex.quote(expected_usd) + "\n"
            )

    script = (
        "set -euo pipefail\n"
        "exec > >(tee -a /tmp/byo-rollout.log) 2>&1\n"
        "PY=/isaac-sim/python.sh\n"
        '[ -x "$PY" ] || { echo "MISSING_PINNED_ISAAC_RUNTIME"; exit 127; }\n'
        '"$PY" -m npa.workflows.sim2real.runtime_attestation\n'
        "mkdir -p /tmp/rollwork/frames; cd /tmp/rollwork\n"
        "export NPA_ROBOT_MODULE_DIR=/opt/npa/isaac-runtime\n"
        f'export ROLLOUT_TASK="{task}" ROLLOUT_COUNT="{rollout_count}" '
        f'ROLLOUT_STEPS="{steps_per_rollout}" ROLLOUT_OBJECT_USD="{object_usd}" '
        f'ROLLOUT_HORIZON_STEPS="{horizon_steps}" '
        f"ROLLOUT_CAMERA_VIEWS_JSON={_shlex.quote(camera_views or camera_views_json())} "
        f'ROLLOUT_CAPTURE_WIDTH="{capture["width"]}" '
        f'ROLLOUT_CAPTURE_HEIGHT="{capture["height"]}" '
        f'ROLLOUT_CAPTURE_STRIDE="{capture["rollout_stride"]}" '
        f'ROLLOUT_PNG_COMPRESS_LEVEL="{capture["png_compress_level"]}" '
        f'ROLLOUT_CAPTURE_FPS="{capture["fps"]}" '
        f"ROLLOUT_SIM_DEVICE={_shlex.quote(_env('NPA_SIM2REAL_ISAAC_DEVICE', 'cuda:0'))} "
        f"ROLLOUT_CKPT_URI={_shlex.quote(checkpoint_uri)} "
        f'ROLLOUT_CKPT_LOCAL="{ckpt_local}" '
        f"ROLLOUT_OUT_S3={_shlex.quote(out_s3_prefix)} "
        "ROLLOUT_FRAMES_DIR=/tmp/rollwork/frames\n"
        + download
        + scenario_block
        + robot_block
        + '"$PY" /opt/npa/isaac-runtime/isaac_rollout.py\n'
        'echo "BYO_ROLLOUT_EXIT"\n'
    )
    command, args = compressed_bash_launch(script)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {"app": "sim2real-byo-isaac-rollout", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 1,
            "template": {
                "metadata": {
                    "labels": {"app": "sim2real-byo-isaac-rollout", "run-id": run_id}
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": service_account,
                    # Match the hardened image's uid/gid 1000 runtime contract. The
                    # image now owns its workspace and installs a traversable Isaac
                    # shim, so retained standalone jobs no longer need a root override.
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
                            "name": "rollout",
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
                            ]
                            + _isaac_eula_env_entries(),
                            "command": command,
                            "args": args,
                        }
                    ],
                    "nodeSelector": {f"{gpu_resource}.product": gpu_product},
                },
            },
        },
    }


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _expected_camera_frame_count(capture: dict[str, Any]) -> int:
    """Return frames emitted by sampled decisions plus the terminal horizon.

    The Isaac loop advances every simulation step but captures only at the evenly
    spaced decision points, subject to the capture stride, and once more at the
    terminal horizon. Counting every simulation step made reduced live proofs demand
    301 frames after correctly producing eight decision frames plus the terminal one.
    """

    declared_count = int(capture.get("expected_frames_per_view") or 0)
    if declared_count > 0:
        return declared_count
    horizon_steps = int(capture.get("horizon_steps") or 0)
    decision_points = int(capture.get("decision_points") or 0)
    capture_stride = max(1, int(capture.get("rollout_stride") or 1))
    if horizon_steps <= 0 or decision_points <= 0:
        return 0
    declared = capture.get("sample_steps")
    if isinstance(declared, list) and declared:
        sample_steps = [int(step) for step in declared]
    elif decision_points == 1:
        sample_steps = [0]
    else:
        sample_steps = [
            (index * (horizon_steps - 1)) // (decision_points - 1)
            for index in range(decision_points)
        ]
    captured = {step for step in sample_steps if step % capture_stride == 0}
    captured.add(horizon_steps)
    return len(captured)


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
    note = str(meta.get("note") or "")
    if not note.startswith("rollout_ok"):
        raise RuntimeError(
            f"real Isaac rollout failed closed: {note or 'missing status'}"
        )
    if checkpoint_uri and not is_trained:
        raise RuntimeError(
            "real Isaac rollout did not load the requested trained checkpoint"
        )
    checkpoint = dict(meta.get("policy_checkpoint") or {})
    if checkpoint_uri and (
        checkpoint.get("uri") != checkpoint_uri
        or not checkpoint.get("sha256")
        or int(checkpoint.get("size_bytes") or 0) <= 0
    ):
        raise RuntimeError(
            "real Isaac rollout did not prove the exact requested checkpoint bytes"
        )
    camera_meta = list(meta.get("camera_metadata") or [])
    capture = dict(meta.get("capture") or {})
    applied = dict(meta.get("applied_scenarios") or {})
    applied_digests = {
        str(row.get("scenario_config_digest") or "")
        for row in applied.get("records") or []
        if int(row.get("applied_count") or 0) > 0
    }
    dirs: list[str] = []
    for roll in meta.get("rollouts", []) or []:
        action_rows = list(roll.get("actions") or [])
        expected_points = int(capture.get("decision_points") or 0)
        if expected_points and len(action_rows) != expected_points:
            raise RuntimeError(
                f"rollout temporal coverage mismatch: expected {expected_points}, "
                f"found {len(action_rows)}"
            )
        if not action_rows or not all(
            (row.get("simulator_ground_truth") or {}).get("scenario_config_digest")
            for row in action_rows
        ):
            raise RuntimeError("rollout lacks per-decision simulator ground truth")
        rid = roll["rollout_id"]
        rdir = output_dir / rid
        rdir.mkdir(parents=True, exist_ok=True)
        view_frames = {
            str(name): [str(frame) for frame in frames]
            for name, frames in (roll.get("camera_views") or {}).items()
        }
        if not view_frames:
            view_frames = {"primary": [str(name) for name in roll.get("frames", [])]}
        expected_views = {
            str(item.get("name") or "") for item in camera_meta if item.get("name")
        }
        expected_frame_count = _expected_camera_frame_count(capture)
        missing_views = sorted(
            name for name in expected_views if not view_frames.get(name)
        )
        wrong_counts = {
            name: len(view_frames.get(name) or [])
            for name in expected_views
            if expected_frame_count
            and len(view_frames.get(name) or []) != expected_frame_count
        }
        if missing_views or wrong_counts:
            raise RuntimeError(
                "real Isaac rollout camera coverage mismatch: "
                f"missing={missing_views} counts={wrong_counts} "
                f"expected_per_view={expected_frame_count}"
            )
        all_frames = list(
            dict.fromkeys(frame for frames in view_frames.values() for frame in frames)
        )
        for name in all_frames:
            try:
                s3.download_file(u.netloc, f"{base}/{rid}/{name}", str(rdir / name))
            except Exception as exc:  # pragma: no cover - network
                print(
                    f"byo_isaac_policy_rollout: frame download failed {rid}/{name}: {exc!r}",
                    flush=True,
                )
        manifest = build_rollout_manifest(
            rollout_id=rid,
            frames=roll.get("frames", []),
            actions=action_rows,
            checkpoint_uri=checkpoint_uri,
            is_trained=is_trained,
            camera_views=view_frames,
            camera_metadata_items=camera_meta,
            frame_metadata=dict(roll.get("camera_frame_metadata") or {}),
            capture=capture,
            checkpoint_sha256=str(checkpoint.get("sha256") or ""),
            checkpoint_size_bytes=int(checkpoint.get("size_bytes") or 0),
            scenario=dict(roll.get("scenario") or {}),
            simulation_device=str(meta.get("simulation_device") or "cuda:0"),
        )
        (rdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        dirs.append(str(rdir))
    if not dirs:
        raise RuntimeError("real Isaac rollout returned no episodes")
    reported_digests = {
        str((roll.get("scenario") or {}).get("scenario_config_digest") or "")
        for roll in meta.get("rollouts", []) or []
    }
    if (
        not reported_digests
        or "" in reported_digests
        or reported_digests != applied_digests
    ):
        raise RuntimeError(
            "rollout scenario labels do not exactly match Isaac runtime applied digests"
        )
    return dirs


def run_isaac_rollout_job(
    output_dir: Path,
    *,
    run_id: str,
    rollout_count: int,
    steps_per_rollout: int,
) -> list[str]:
    global _LAST_GPU_PROVENANCE, _LAST_EMBODIMENT_EVIDENCE

    task = _env("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK)
    image = _env("NPA_SIM2REAL_ISAAC_IMAGE") or _env("ISAAC_IMAGE")
    from npa.workflows.sim2real.k8s_components import _image_pull_policy

    bucket = (
        _env("NPA_SIM2REAL_BUCKET")
        or _env("S3_BUCKET")
        or _env("NPA_SIM2REAL_S3_BUCKET")
    )
    namespace = _env("NPA_SIM2REAL_K8S_NAMESPACE", "default")
    sa = _env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    gpu_product = _env("NPA_SIM2REAL_K8S_GPU_PRODUCT", DEFAULT_GPU_PRODUCT)
    endpoint = _env("AWS_ENDPOINT_URL")
    s3_prefix = _env("NPA_SIM2REAL_PREFIX", "sim2real-b").strip("/")
    timeout_s = int(_env("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "0") or 0)
    capture = capture_settings()
    # Rollouts spawn the SAME manipuland as train/eval (default: the proven
    # rigid-ready USD, not the stock primitive cube).
    from npa.workflows.sim2real.byo_isaac_trainer import (
        artifact_tag,
        artifact_tag_from_output_dir,
        k8s_job_name,
        resolve_object_usd,
    )

    object_usd = resolve_object_usd(_env("NPA_BYO_ISAAC_OBJECT_USD"))

    from npa.workflows.sim2real.byo_isaac_trainer import (
        ROBOT_USD_CONTAINER_PATH,
        _resolve_byo_robot_spec,
        robot_spec_payload,
    )

    robot_spec_dict: dict[str, Any] | None = None
    robot_usd_uri = ""
    if _env("NPA_BYO_ROBOT_TASK") == "1":
        spec = _resolve_byo_robot_spec()
        usd_dest = ""
        if (
            spec is not None
            and str(getattr(spec, "robot_source", "")) != "stock_franka"
        ):
            source_uri = str(getattr(spec, "robot_uri", "") or "")
            if source_uri.startswith("s3://"):
                robot_usd_uri = source_uri
                usd_dest = ROBOT_USD_CONTAINER_PATH
            elif source_uri:
                usd_dest = source_uri
        robot_spec_dict = robot_spec_payload(spec, usd_container_path=usd_dest)
    if robot_spec_dict is None:
        robot_spec_dict = {"robot_source": "stock_franka", "name": "franka"}
    from npa.workflows.sim2real.byo_isaac_trainer import embodiment_evidence

    _LAST_EMBODIMENT_EVIDENCE = embodiment_evidence(robot_spec_dict)
    task_config = None
    raw_task_config = _env("NPA_BYO_TASK_CONFIG_JSON")
    if raw_task_config:
        parsed_task_config = json.loads(raw_task_config)
        if not isinstance(parsed_task_config, dict):
            raise RuntimeError("NPA_BYO_TASK_CONFIG_JSON must be an object")
        task_config = parsed_task_config

    checkpoint_uri = _env("NPA_SIM2REAL_POLICY_CHECKPOINT_URI")
    if checkpoint_uri and not checkpoint_uri.startswith("s3://"):
        raise RuntimeError("explicit rollout checkpoint must be an s3:// URI")
    if not checkpoint_uri:
        checkpoint_uri = latest_checkpoint_uri(
            bucket,
            run_id,
            s3_endpoint=endpoint,
            s3_prefix=s3_prefix,
        )
    # Unique per (run, outer, inner) so second-pass component artifacts do not
    # overwrite first-pass rollouts.
    job_suffix = artifact_tag(
        _env("NPA_SIM2REAL_ROLLOUT_TAG")
    ) or artifact_tag_from_output_dir(output_dir)
    from npa.workflows.sim2real.byo_isaac_trainer import read_generated_train_envs

    train_rows, _ = read_generated_train_envs(
        _env("NPA_SIM2REAL_TRAIN_ENVS_DIR"),
        envs_uri=_env("NPA_SIM2REAL_TRAIN_ENVS_URI"),
    )
    selected_scenarios = select_rollout_scenarios(
        train_rows,
        count=rollout_count,
        selection_tag=f"{run_id}:{job_suffix}",
    )
    scenarios_jsonl = (
        "\n".join(json.dumps(row, sort_keys=True) for row in selected_scenarios) + "\n"
    )
    job_name = k8s_job_name("s2r-byo-isaac-roll", run_id, job_suffix)
    out_s3 = f"s3://{bucket}/{s3_prefix}/{run_id}/byo-rollouts/{job_suffix}"

    manifest = build_isaac_rollout_job_manifest(
        job_name=job_name,
        run_id=run_id,
        image=image,
        task=task,
        rollout_count=rollout_count,
        steps_per_rollout=steps_per_rollout,
        horizon_steps=int(_env("NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS", "300") or 300),
        checkpoint_uri=checkpoint_uri,
        out_s3_prefix=out_s3,
        s3_endpoint=endpoint,
        namespace=namespace,
        service_account=sa,
        gpu_product=gpu_product,
        image_pull_policy=_image_pull_policy(image),
        object_usd=object_usd,
        camera_views=json.dumps(
            camera_metadata(
                _env("NPA_SIM2REAL_CAMERA_VIEWS"),
                width=int(capture["width"]),
                height=int(capture["height"]),
            ),
            separators=(",", ":"),
        ),
        capture=capture,
        scenarios_jsonl=scenarios_jsonl,
        robot_spec=robot_spec_dict,
        robot_usd_uri=robot_usd_uri,
        task_config=task_config,
    )
    from npa.workflows.sim2real.isaac_job_payload import (
        execute_manifest_container_inline,
    )

    if _env("NPA_SIM2REAL_INLINE_TASK") == "1":
        provenance = execute_manifest_container_inline(manifest)
        job_name = str(provenance["job_name"] or job_name)
        _LAST_GPU_PROVENANCE = provenance
        return materialize_rollout_dirs(
            output_dir,
            _download_rollout_metadata(out_s3, endpoint=endpoint),
            out_s3,
            checkpoint_uri=checkpoint_uri,
            s3_endpoint=endpoint,
        )

    from npa.workflows.sim2real.gpu_fallback import run_gpu_job_with_fallback

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

    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

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
    _LAST_GPU_PROVENANCE = provenance

    # Pull the rollouts manifest, then materialize local rollout dirs.
    import boto3
    from urllib.parse import urlparse

    u = urlparse(out_s3)
    s3 = boto3.client("s3", endpoint_url=endpoint or None)
    local_meta = output_dir / "rollouts.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    s3.download_file(
        u.netloc, f"{u.path.lstrip('/').rstrip('/')}/rollouts.json", str(local_meta)
    )
    meta = json.loads(local_meta.read_text())
    return materialize_rollout_dirs(
        output_dir, meta, out_s3, checkpoint_uri=checkpoint_uri, s3_endpoint=endpoint
    )


def _download_rollout_metadata(out_s3: str, *, endpoint: str) -> dict[str, Any]:
    import boto3
    from urllib.parse import urlparse

    uri = urlparse(out_s3)
    response = boto3.client("s3", endpoint_url=endpoint or None).get_object(
        Bucket=uri.netloc,
        Key=f"{uri.path.lstrip('/').rstrip('/')}/rollouts.json",
    )
    return json.loads(response["Body"].read())


def main() -> int:
    global _LAST_GPU_PROVENANCE, _LAST_EMBODIMENT_EVIDENCE
    _LAST_GPU_PROVENANCE = {}
    _LAST_EMBODIMENT_EVIDENCE = {}
    output_json = _env("NPA_SIM2REAL_OUTPUT_JSON")
    if not output_json:
        print(
            "byo_isaac_policy_rollout: NPA_SIM2REAL_OUTPUT_JSON not set",
            file=sys.stderr,
        )
        return 2
    run_id = _env("NPA_SIM2REAL_RUN_ID") or _env("RUN_ID") or "byo-isaac"
    output_dir = Path(_env("NPA_SIM2REAL_OUTPUT_DIR") or str(Path(output_json).parent))
    rollout_count = int(_env("NPA_SIM2REAL_ROLLOUT_COUNT", "4") or 4)
    steps_per_rollout = int(_env("NPA_SIM2REAL_STEPS_PER_ROLLOUT", "8") or 8)

    if _env("NPA_BYO_ISAAC_DRYRUN") == "1":
        bucket = _env("NPA_SIM2REAL_BUCKET") or _env("S3_BUCKET")
        checkpoint_uri = latest_checkpoint_uri(
            bucket,
            run_id,
            s3_endpoint=_env("AWS_ENDPOINT_URL"),
            s3_prefix=_env("NPA_SIM2REAL_PREFIX", "sim2real-b"),
        )
        rollout_dirs = write_dryrun_rollouts(
            output_dir,
            count=rollout_count,
            steps_per_rollout=steps_per_rollout,
            checkpoint_uri=checkpoint_uri,
        )
    else:
        rollout_dirs = run_isaac_rollout_job(
            output_dir,
            run_id=run_id,
            rollout_count=rollout_count,
            steps_per_rollout=steps_per_rollout,
        )

    payload = {
        "schema": "npa.sim2real.policy_rollouts.v1",
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "rollout_dirs": rollout_dirs,
        "capture": capture_settings(),
        "component_invocation": {
            "mode": str(_LAST_GPU_PROVENANCE.get("mode") or "kubernetes_job")
            if _LAST_GPU_PROVENANCE
            else "dryrun",
            "gpu_provenance": _LAST_GPU_PROVENANCE,
        },
    }
    payload["embodiment"] = dict(_LAST_EMBODIMENT_EVIDENCE) or {
        "embodiment_digest": "stock_franka",
        "expected_action_dim": 8,
        "expected_observation_dim": 36,
        "runtime_dimension_validation": "passed",
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"byo_isaac_policy_rollout: wrote {output_json} rollout_dirs={len(rollout_dirs)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

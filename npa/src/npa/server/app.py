"""npa-lerobot-server: FastAPI server with in-process LeRobot policy inference."""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("npa-lerobot-server")

# ── Configuration (all from env) ──────────────────────────────────────────

SERVER_HOST = os.environ.get("NPA_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("NPA_SERVER_PORT", "8080"))
CHECKPOINT_DIR = os.environ.get("NPA_CHECKPOINT_DIR", "/opt/lerobot/checkpoints")
CHECKPOINT_BUCKET = os.environ.get("NPA_CHECKPOINT_BUCKET", "")
JOB_STATUS_DIR = os.environ.get("NPA_JOB_STATUS_DIR", "/opt/lerobot/job_status")
LOG_DIR = os.environ.get("NPA_LOG_DIR", "/var/log/npa-lerobot")

# S3 credentials (for checkpoint pulls)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", os.environ.get("NEBIUS_S3_ENDPOINT", ""))


# ── In-process policy state ───────────────────────────────────────────────

class PolicyState:
    """Holds the loaded LeRobot policy, preprocessor, and postprocessor in-process."""

    def __init__(self) -> None:
        self.policy = None  # PreTrainedPolicy
        self.preprocessor = None  # PolicyProcessorPipeline
        self.postprocessor = None  # PolicyProcessorPipeline
        self.device: torch.device | None = None
        self.checkpoint: str = ""
        self.loaded_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.policy is not None

    def load(
        self,
        checkpoint_path: str,
        env_type: str | None = None,
        env_task: str | None = None,
    ) -> None:
        """Load a policy from a checkpoint, following lerobot_record.py pattern."""
        with self._lock:
            self.unload_unlocked()

            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.utils.device_utils import get_safe_torch_device

            logger.info("Loading policy from: %s", checkpoint_path)

            # Load config from pretrained path
            policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
            policy_cfg.pretrained_path = checkpoint_path

            # Build env_cfg if env info was provided (needed by make_policy for shapes)
            env_cfg = None
            if env_type:
                from lerobot.envs.configs import EnvConfig
                # EnvConfig is abstract with registered subclasses (e.g. "aloha" → AlohaEnv)
                env_cls = EnvConfig.get_choice_class(env_type)
                kwargs = {}
                if env_task:
                    kwargs["task"] = env_task
                env_cfg = env_cls(**kwargs)

            # Create policy (loads weights, moves to device, sets eval mode)
            policy = make_policy(policy_cfg, env_cfg=env_cfg)

            device = get_safe_torch_device(policy_cfg.device)

            # Create preprocessor and postprocessor from checkpoint
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=checkpoint_path,
            )

            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.device = device
            self.checkpoint = checkpoint_path
            self.loaded_at = time.time()

            logger.info(
                "Policy loaded: %s on %s (params: %s)",
                type(policy).__name__,
                device,
                sum(p.numel() for p in policy.parameters()),
            )

    def unload(self) -> None:
        with self._lock:
            self.unload_unlocked()

    def unload_unlocked(self) -> None:
        if self.policy is not None:
            logger.info("Unloading policy: %s", self.checkpoint)
            del self.policy
            del self.preprocessor
            del self.postprocessor
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            self.device = None
            self.checkpoint = ""
            self.loaded_at = 0.0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(self, observation: dict[str, np.ndarray]) -> list[float]:
        """Run a forward pass, following predict_action() from control_utils.py."""
        with self._lock:
            if self.policy is None:
                raise RuntimeError("No policy loaded")

            from lerobot.policies.utils import prepare_observation_for_inference

            # Convert numpy observation to tensors on device
            obs_tensors = prepare_observation_for_inference(
                observation, self.device
            )

            # Run preprocessor pipeline (normalization, etc.)
            obs_tensors = self.preprocessor(obs_tensors)

            # Forward pass
            with torch.inference_mode():
                if self.device.type == "cuda" and self.policy.config.use_amp:
                    with torch.autocast(device_type="cuda"):
                        action = self.policy.select_action(obs_tensors)
                else:
                    action = self.policy.select_action(obs_tensors)

            # Run postprocessor pipeline (unnormalization, etc.)
            action = self.postprocessor(action)

            # Convert tensor to list
            if isinstance(action, torch.Tensor):
                return action.squeeze().cpu().tolist()
            return action


policy_state = PolicyState()


# ── Checkpoint resolution ─────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint reference to a local path.

    Resolution order:
    1. S3 URI (s3://...) → download to CHECKPOINT_DIR cache
    2. Local absolute path on VM
    3. Relative path under CHECKPOINT_DIR
    4. HF Hub repo ID → let lerobot handle it natively
    """
    if checkpoint.startswith("s3://"):
        return _pull_from_s3(checkpoint)

    if checkpoint.startswith("/") and Path(checkpoint).exists():
        return checkpoint

    candidate = Path(CHECKPOINT_DIR) / checkpoint
    if candidate.exists():
        return str(candidate)

    # Assume HF Hub repo — lerobot will resolve it
    return checkpoint


def _pull_from_s3(uri: str) -> str:
    """Download checkpoint from S3 to local cache if not present."""
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    # Use bucket + full path as cache key to avoid collisions between URIs
    # that share the same basename (e.g. .../job-a/pretrained_model vs
    # .../job-b/pretrained_model).
    cache_key = f"{bucket}_{prefix.replace('/', '_')}"
    local_dir = Path(CHECKPOINT_DIR) / "s3_cache" / cache_key

    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info("Using cached checkpoint: %s", local_dir)
        return str(local_dir)

    logger.info("Pulling checkpoint from %s to %s", uri, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL or None,
        aws_access_key_id=AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
    )
    paginator = s3.get_paginator("list_objects_v2")
    prefix_with_slash = prefix + "/" if not prefix.endswith("/") else prefix

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix_with_slash):]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))

    return str(local_dir)


# ── Observation parsing ───────────────────────────────────────────────────

def _parse_observation(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert a JSON observation payload to a dict of numpy arrays.

    Supports two formats for image values:
    - Base64-encoded string: decoded to uint8 array, then reshaped to (H, W, C)
    - Nested list (array): converted directly to numpy
    State/scalar values are converted to float32 arrays.
    """
    observation: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            # Base64-encoded image
            decoded = base64.b64decode(value)
            arr = np.frombuffer(decoded, dtype=np.uint8)
            observation[key] = arr
        elif isinstance(value, list):
            arr = np.array(value)
            if arr.ndim >= 3:
                # Image array (H, W, C) — keep as uint8 if values are 0-255
                if arr.max() > 1.0:
                    observation[key] = arr.astype(np.uint8)
                else:
                    observation[key] = arr.astype(np.float32)
            else:
                observation[key] = arr.astype(np.float32)
        elif isinstance(value, (int, float)):
            observation[key] = np.array([value], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported observation type for key '{key}': {type(value)}")
    return observation


# ── Job status helpers ────────────────────────────────────────────────────

def _read_jobs() -> list[dict[str, Any]]:
    status_dir = Path(JOB_STATUS_DIR)
    if not status_dir.exists():
        return []
    jobs = []
    for f in sorted(status_dir.glob("*.json")):
        try:
            jobs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return jobs


# ── FastAPI app ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(JOB_STATUS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("npa-lerobot-server starting on %s:%s", SERVER_HOST, SERVER_PORT)
    yield
    policy_state.unload()
    logger.info("npa-lerobot-server shutting down")


app = FastAPI(title="npa-lerobot-server", lifespan=lifespan)


class ServeRequest(BaseModel):
    checkpoint: str
    env_type: str | None = None
    env_task: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def get_status():
    return {
        "policy_server": {
            "running": policy_state.loaded,
            "checkpoint": policy_state.checkpoint,
            "uptime_seconds": round(time.time() - policy_state.loaded_at, 1) if policy_state.loaded else 0,
            "policy_class": type(policy_state.policy).__name__ if policy_state.loaded else None,
            "device": str(policy_state.device) if policy_state.device else None,
        },
        "jobs": _read_jobs(),
        "checkpoint_dir": CHECKPOINT_DIR,
        "log_dir": LOG_DIR,
    }


@app.post("/serve")
async def start_serve(req: ServeRequest):
    try:
        local_path = _resolve_checkpoint(req.checkpoint)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Checkpoint resolution failed: {exc}")

    try:
        policy_state.load(local_path, env_type=req.env_type, env_task=req.env_task)
    except Exception as exc:
        logger.exception("Failed to load policy from %s", local_path)
        raise HTTPException(status_code=500, detail=f"Failed to load policy: {exc}")

    return {
        "status": "serving",
        "checkpoint": local_path,
        "original_checkpoint": req.checkpoint,
        "policy_class": type(policy_state.policy).__name__,
        "device": str(policy_state.device),
    }


@app.delete("/serve")
async def stop_serve():
    policy_state.unload()
    return {"status": "stopped"}


@app.post("/infer")
async def run_infer(observation: dict[str, Any]):
    if not policy_state.loaded:
        raise HTTPException(
            status_code=409,
            detail="No policy loaded. POST /serve with a checkpoint first.",
        )

    try:
        obs_arrays = _parse_observation(observation)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid observation payload: {exc}")

    try:
        start = time.time()
        actions = policy_state.predict(obs_arrays)
        inference_ms = round((time.time() - start) * 1000, 2)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return {
        "actions": actions,
        "inference_ms": inference_ms,
        "checkpoint": policy_state.checkpoint,
    }


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(
        "npa.server.app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()

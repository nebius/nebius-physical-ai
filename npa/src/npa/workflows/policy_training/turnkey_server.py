"""Serve the promoted SmolVLA checkpoint to an authenticated benchmark session."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Annotated

from pydantic import BaseModel, Field

from .public_vla_data import write_json
from .public_vla_export import file_sha256


class Observation(BaseModel):
    """A single episode's ordered camera and state observation."""

    episode: int = Field(ge=0, strict=True)
    step: int = Field(ge=0, strict=True)
    task: str = Field(min_length=1, max_length=4096)
    state: list[Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(
        min_length=8, max_length=8
    )
    image: str = Field(max_length=400_000)
    wrist: str = Field(max_length=400_000)


def create_app(bundle: Path, output: Path, expected: str, token: str, server):
    """Load an exact CUDA policy and expose a private, single-session HTTP API.

    Args:
        bundle: Portable exported model directory.
        output: Private request and runtime artifact directory.
        expected: Promoted weight digest.
        token: Run-scoped bearer credential, never included in reports.
        server: Uvicorn server used for orderly client completion.
    Returns:
        FastAPI application with authenticated health, reset, infer and finish.
    Raises:
        ValueError: Model identity differs from the promoted checkpoint.
    """
    from fastapi import FastAPI

    state = _load(bundle, output, expected)
    app = FastAPI()
    _routes(app, state, output, token, server)
    return app


def _load(bundle, output, expected):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if file_sha256(bundle / "policy/model.safetensors") != expected:
        raise ValueError("served model differs from the promoted checkpoint")
    os.chdir(bundle)
    torch.manual_seed(42)
    policy = SmolVLAPolicy.from_pretrained("policy").to("cuda").eval()
    counter = _count_chunks(policy)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path="policy",
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    runtime = _runtime(policy, expected, torch)
    write_json(output / "runtime.json", runtime)
    return {
        "policy": policy,
        "pre": pre,
        "post": post,
        "runtime": runtime,
        "episode": None,
        "next_step": 0,
        "lock": threading.Lock(),
        "finished": False,
        "chunk_counter": counter,
    }


def _runtime(policy, expected, torch):
    return {
        "policy": "continued-trained SmolVLA",
        "checkpoint_sha256": expected,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "scheduler": "kubernetes",
        "n_action_steps": policy.config.n_action_steps,
        "physical_robot_tested": False,
        "session_mode": "single sequential benchmark client",
    }


def _count_chunks(policy):
    # LeRobot 0.6.0's native chunk call; covered by the release compatibility audit.
    counter = {"calls": 0}
    original = policy._get_action_chunk

    def measured(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    policy._get_action_chunk = measured
    return counter


def _routes(app, state, output, token, server):
    from fastapi import Header, HTTPException

    def authorize(authorization):
        import secrets

        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "benchmark session authentication required")

    @app.get("/health")
    def health(authorization: str = Header(default="")):
        authorize(authorization)
        return state["runtime"] | {"ready": True}

    @app.post("/reset/{episode}")
    def reset(episode: int, authorization: str = Header(default="")):
        authorize(authorization)
        with state["lock"]:
            for component in (state["policy"], state["pre"], state["post"]):
                component.reset()
            state.update(episode=episode, next_step=0)
        return {"episode": episode, "reset": True}

    _inference_route(app, state, output, authorize, Header)
    _finish_route(app, state, server, authorize, Header)


def _finish_route(app, state, server, authorize, Header):
    @app.post("/finish")
    def finish(completed: bool, authorization: str = Header(default="")):
        authorize(authorization)
        state["finished"] = completed
        server.should_exit = True
        return {"completed": completed}


def _inference_route(app, state, output, authorize, Header):
    @app.post("/infer")
    def infer(request: Observation, authorization: str = Header(default="")):
        from fastapi import HTTPException

        authorize(authorization)
        with state["lock"]:
            if (
                request.episode != state["episode"]
                or request.step != state["next_step"]
            ):
                raise HTTPException(409, "episode and step are out of sequence")
            batch = _batch(request)
            row = _action(state, batch, request)
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            state["next_step"] += 1
            return row


def _batch(request):
    import numpy as np
    import torch
    from fastapi import HTTPException
    from PIL import Image

    values = np.asarray(request.state, dtype=np.float32)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise HTTPException(422, "finite eight-dimensional state required")
    batch = {"observation.state": torch.from_numpy(values), "task": request.task}
    for encoded, name in ((request.image, "image"), (request.wrist, "image2")):
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
        if image.size != (256, 256):
            raise HTTPException(422, "256x256 RGB observation required")
        raw = np.asarray(image.convert("RGB")).copy()
        if raw.shape != (256, 256, 3):
            raise HTTPException(422, "256x256 RGB observation required")
        batch["observation.images." + name] = (
            torch.from_numpy(raw).permute(2, 0, 1).float() / 255
        )
    return batch


def _action(state, batch, request):
    import numpy as np
    import torch

    started = time.perf_counter()
    prior_chunks = state["chunk_counter"]["calls"]
    with torch.inference_mode():
        action = state["post"](state["policy"].select_action(state["pre"](batch)))
    torch.cuda.synchronize()
    values = action.cpu().numpy().reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError("native policy produced invalid actions")
    return {
        "episode": request.episode,
        "step": request.step,
        "action": values.tolist(),
        "server_ms": (time.perf_counter() - started) * 1000,
        "new_action_chunk": state["chunk_counter"]["calls"] > prior_chunks,
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 2**20,
    }

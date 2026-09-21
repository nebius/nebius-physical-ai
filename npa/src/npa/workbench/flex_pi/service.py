"""Authenticated flex-pi inference over an operator-owned input and output root."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request

from npa.workbench.flex_pi.runtime import (
    ARTIFACT_SCHEMA,
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_CHECKPOINT_REVISION,
    DEFAULT_INPUT_MANIFEST,
    FlexPiError,
    FlexPiRequest,
    run_inference,
)
from npa.workbench.flex_pi.schemas import InferenceBody, TrainingBody


def _output_root(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme == "s3":
        if (
            not parsed.netloc
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("service output root requires an S3 bucket and prefix")
        return value.rstrip("/")
    if parsed.scheme or not value:
        raise ValueError("configure a local directory or S3 output prefix")
    root = Path(value).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("local output root must be owner-only mode 0700")
    return str(root.resolve())


def create_app(
    *,
    token: str | None = None,
    output_root: str | None = None,
    input_manifest: str | None = None,
) -> FastAPI:
    """Create an authenticated service with startup-snapshotted trusted input.

    Args:
        token: Optional bearer token override for embedding/tests.
        output_root: Operator-owned local directory or S3 prefix.
        input_manifest: Trusted local public-observation manifest.
    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(service: FastAPI):
        secret = token if token is not None else os.environ.get("NPA_FLEX_PI_TOKEN", "")
        if not secret:
            raise ValueError("NPA_FLEX_PI_TOKEN is required")
        root = _output_root(
            output_root or os.environ.get("NPA_FLEX_PI_OUTPUT_ROOT", "")
        )
        source = Path(
            input_manifest
            or os.environ.get("NPA_FLEX_PI_INPUT", DEFAULT_INPUT_MANIFEST)
        )
        contents = source.read_bytes()
        with tempfile.TemporaryDirectory(prefix="npa-flex-pi-service-") as scratch:
            snapshot = Path(scratch) / "input.json"
            snapshot.write_bytes(contents)
            snapshot.chmod(0o400)
            service.state.configuration = (secret.encode(), root, snapshot)
            try:
                yield
            finally:
                service.state.configuration = None

    service = FastAPI(title="NPA flex-pi", version="1.0", lifespan=lifespan)
    service.state.configuration = None

    def authorize(request: Request) -> tuple[bytes, str, Path]:
        config = request.app.state.configuration
        if config is None:
            raise HTTPException(
                status_code=503, detail="Service configuration is not ready"
            )
        supplied = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(supplied, b"Bearer " + config[0]):
            raise HTTPException(
                status_code=401, detail="Bearer authentication required"
            )
        return config

    @service.get("/health", dependencies=[Depends(authorize)])
    def health() -> dict[str, str]:
        return {"status": "ok", "schema": ARTIFACT_SCHEMA}

    @service.get("/system-info", dependencies=[Depends(authorize)])
    def system_info() -> dict[str, Any]:
        return {
            "checkpoint_id": DEFAULT_CHECKPOINT_ID,
            "checkpoint_revision": DEFAULT_CHECKPOINT_REVISION,
            "weights_baked": False,
        }

    @service.post("/run")
    def run(body: InferenceBody, config=Depends(authorize)) -> dict[str, Any]:
        _, root, snapshot = config
        destination = (
            root + "/" + body.output_path + "-" + uuid4().hex
            if root.startswith("s3://")
            else tempfile.mkdtemp(prefix=body.output_path + "-", dir=root)
        )
        try:
            return run_inference(
                FlexPiRequest(
                    input_path=str(snapshot),
                    output_path=destination,
                    **body.model_dump(exclude={"output_path"}),
                )
            )
        except FlexPiError as exc:
            raise HTTPException(
                status_code=422, detail="flex-pi inference failed"
            ) from exc

    @service.get("/status", dependencies=[Depends(authorize)])
    def status() -> dict[str, str]:
        return {"status": "ready"}

    @service.post("/train")
    def train(body: TrainingBody, config=Depends(authorize)) -> dict[str, Any]:
        from npa.workbench.flex_pi.training import TrainingRequest, run_training

        _, root, _ = config
        if not root.startswith("s3://"):
            raise HTTPException(status_code=422, detail="training requires an operator-owned S3 output root")
        try:
            return run_training(TrainingRequest(
                output_path=root + "/" + body.output_path + "-" + uuid4().hex,
                **body.model_dump(exclude={"output_path"}),
            ))
        except FlexPiError as exc:
            raise HTTPException(status_code=422, detail="flex-pi training failed") from exc

    @service.get("/list", dependencies=[Depends(authorize)])
    def list_capabilities() -> list[dict[str, str]]:
        return [{"name": "action-only-policy-inference", "schema": ARTIFACT_SCHEMA}]

    return service


app = create_app()

"""Authenticated inference with operator-owned model, manifest, and outputs."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request

from npa.workbench.alpamayo2_super.runtime import (
    ARTIFACT_SCHEMA,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    run_inference,
)
from npa.workbench.alpamayo2_super.schemas import InferenceBody


def _output_root(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
            raise ValueError("The service output root requires an S3 bucket and prefix")
        return value.rstrip("/")
    if parsed.scheme or not value:
        raise ValueError("Configure a local directory or S3 prefix for service outputs")
    root = Path(value).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("The local service output root must be owned by this user with mode 0700")
    return str(root.resolve())


def create_app(
    *, token: str | None = None, output_root: str | None = None,
    manifest: str | None = None,
) -> FastAPI:
    """Snapshot the operator's trusted manifest and configure the HTTP boundary.

    HTTP clients may select a sample and inference controls. Only the repository
    model and dataset revisions are available through HTTP; trusted CLI/SDK
    custom models remain separate from this boundary.
    """

    @asynccontextmanager
    async def lifespan(service: FastAPI):
        secret = token if token is not None else os.environ.get("NPA_ALPAMAYO2_SUPER_TOKEN", "")
        if not secret:
            raise ValueError("NPA_ALPAMAYO2_SUPER_TOKEN is required")
        root = _output_root(
            output_root if output_root is not None
            else os.environ.get("NPA_ALPAMAYO2_SUPER_OUTPUT_ROOT", "")
        )
        source = Path(manifest if manifest is not None else os.environ.get(
            "NPA_ALPAMAYO2_SUPER_MANIFEST", DEFAULT_MANIFEST,
        ))
        contents = source.read_bytes()
        samples = json.loads(contents)["samples"]
        if not isinstance(samples, list) or not samples:
            raise ValueError("The configured manifest requires a nonempty samples list")
        with tempfile.TemporaryDirectory(prefix="npa-alpamayo-service-") as scratch:
            snapshot = Path(scratch) / "manifest.json"
            snapshot.write_bytes(contents)
            snapshot.chmod(0o400)
            service.state.configuration = (secret.encode(), root, snapshot, len(samples))
            try:
                yield
            finally:
                service.state.configuration = None

    service = FastAPI(title="NPA Alpamayo 2 Super", version="1.0", lifespan=lifespan)
    service.state.configuration = None

    def authorize(request: Request) -> tuple[bytes, str, Path, int]:
        config = request.app.state.configuration
        if config is None:
            raise HTTPException(status_code=503, detail="Service configuration is not ready")
        supplied = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(supplied, b"Bearer " + config[0]):
            raise HTTPException(
                status_code=401, detail="Bearer authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return config

    @service.get("/health", dependencies=[Depends(authorize)])
    def health() -> dict[str, str]:
        return {"status": "ok", "schema": ARTIFACT_SCHEMA}

    @service.get("/system-info", dependencies=[Depends(authorize)])
    def system_info() -> dict[str, Any]:
        return {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION, "weights_baked": False,
        }

    @service.post("/run")
    def run(body: InferenceBody, config=Depends(authorize)) -> dict[str, Any]:
        _, root, snapshot, sample_count = config
        if body.sample_index >= sample_count:
            raise HTTPException(status_code=422, detail="sample_index exceeds the configured manifest")
        # Fresh outputs prevent clients from overwriting other runs or using the
        # service's storage identity outside its configured root.
        if root.startswith("s3://"):
            destination = root + "/" + body.output_path + "-" + uuid4().hex
        else:
            destination = tempfile.mkdtemp(prefix=body.output_path + "-", dir=root)
        try:
            values = body.model_dump(exclude={"output_path"})
            return run_inference(Alpamayo2SuperRequest(
                **values, output_path=destination, manifest=str(snapshot),
            ))
        except Alpamayo2SuperError as exc:
            # Upstream diagnostics can contain credential-bearing download URLs.
            raise HTTPException(status_code=422, detail="Alpamayo inference failed") from exc
        finally:
            if not root.startswith("s3://"):
                try:
                    Path(destination).rmdir()
                except OSError:
                    pass  # Successful artifacts remain under the configured root.

    @service.get("/status", dependencies=[Depends(authorize)])
    def status() -> dict[str, str]:
        return {"status": "ready"}

    @service.get("/list", dependencies=[Depends(authorize)])
    def list_capabilities() -> list[dict[str, Any]]:
        return [{"name": "trajectory-inference", "schema": ARTIFACT_SCHEMA}]

    return service


app = create_app()

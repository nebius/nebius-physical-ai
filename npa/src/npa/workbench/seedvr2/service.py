"""Expose the shared SeedVR2 GPU-stage implementation through an authenticated API."""

from __future__ import annotations

from contextlib import contextmanager
import hmac
import os
import threading
from typing import Iterator, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException

from npa.workbench.storage_scope import StorageScope, use_storage_scope

from . import artifacts, runtime
from .schemas import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_REVISION,
    RestoreRequest,
    VideoArtifactRequest,
)


def create_app(
    *,
    token: str | None = None,
    allowed_s3_roots: Sequence[str] | None = None,
) -> FastAPI:
    """Create a request-scoped, serialized SeedVR2 service."""

    secret = token if token is not None else os.environ.get("SEEDVR2_TOKEN", "")
    scope = (
        StorageScope.from_env("SEEDVR2")
        if allowed_s3_roots is None
        else StorageScope.from_config(s3_roots=allowed_s3_roots)
    )
    app = FastAPI(title="NPA SeedVR2", version="1.0")
    lock = threading.Lock()

    def authenticate(authorization: str = Header(default="")) -> None:
        if not secret:
            raise HTTPException(503, "SEEDVR2_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {secret}"):
            raise HTTPException(401, "invalid token")

    @contextmanager
    def operation() -> Iterator[None]:
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "another SeedVR2 operation is active")
        try:
            with use_storage_scope(scope):
                yield
        except (runtime.SeedVR2Error, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "seedvr2"}

    @app.get("/status", dependencies=[Depends(authenticate)])
    def status() -> dict[str, bool]:
        return {"busy": lock.locked()}

    @app.get("/system-info", dependencies=[Depends(authenticate)])
    def system_info() -> dict[str, object]:
        return {
            "source_revision": SOURCE_REVISION,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "weights_baked": False,
        }

    @app.get("/list", dependencies=[Depends(authenticate)])
    def list_capabilities() -> dict[str, object]:
        return {
            "capabilities": ["probe", "restore", "verify", "review"],
            "input_output": "request-scoped S3 artifacts",
        }

    @app.post("/probe", dependencies=[Depends(authenticate)])
    def probe(request: VideoArtifactRequest) -> dict:
        with operation():
            return artifacts.probe(request)

    @app.post("/run", dependencies=[Depends(authenticate)])
    @app.post("/restore", dependencies=[Depends(authenticate)])
    def restore(request: RestoreRequest) -> dict:
        with operation():
            return runtime.restore(request)

    @app.post("/verify", dependencies=[Depends(authenticate)])
    def verify(request: VideoArtifactRequest) -> dict:
        with operation():
            return artifacts.verify(request)

    @app.post("/review", dependencies=[Depends(authenticate)])
    def review(request: VideoArtifactRequest) -> dict:
        with operation():
            return artifacts.review(request)

    return app


__all__ = ["create_app"]

"""Authenticated API for the shared GPU-stage implementation.

This synchronous service serializes GPU requests with an explicit busy response;
it never launches nested workers or owns cloud infrastructure.
"""

from __future__ import annotations

import hmac
import os
import threading
from contextlib import contextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from npa.workbench.storage_scope import StorageScope, use_storage_scope

from . import runtime
from .schemas import PrepareRequest, RunRequest, SOURCE_REVISION


def create_app(*, token: str | None = None, allowed_s3_roots=None):
    secret = token if token is not None else os.environ.get("CUROBO_TOKEN", "")
    scope = (
        StorageScope.from_env("CUROBO")
        if allowed_s3_roots is None
        else StorageScope.from_config(s3_roots=allowed_s3_roots)
    )
    app = FastAPI(title="NPA cuRobo V2")
    lock = threading.Lock()

    def authenticate(authorization: str = Header(default="")):
        if not secret:
            raise HTTPException(503, "CUROBO_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {secret}"):
            raise HTTPException(401, "invalid token")

    @contextmanager
    def operation():
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "another cuRobo operation is active")
        try:
            with use_storage_scope(scope):
                yield
        except (runtime.CuroboError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "curobo"}

    @app.get("/status", dependencies=[Depends(authenticate)])
    def status():
        return {"busy": lock.locked()}

    @app.get("/system-info", dependencies=[Depends(authenticate)])
    def system_info():
        return {"source_revision": SOURCE_REVISION, "engine": "nvidia-curobo-v2"}

    @app.get("/list", dependencies=[Depends(authenticate)])
    def list_operations():
        return {
            "capabilities": ["prepare", "plan", "benchmark", "validate", "visualize"],
            "artifacts": "request-scoped S3 output; no global run inventory",
        }

    @app.post("/prepare", dependencies=[Depends(authenticate)])
    def prepare(request: PrepareRequest):
        with operation():
            return runtime.prepare(request)

    @app.post("/plan", dependencies=[Depends(authenticate)])
    def plan(request: RunRequest):
        with operation():
            return runtime.plan(request)

    @app.post("/benchmark", dependencies=[Depends(authenticate)])
    @app.post("/run", dependencies=[Depends(authenticate)])
    def benchmark(request: RunRequest):
        with operation():
            return runtime.benchmark(request)

    @app.post("/validate", dependencies=[Depends(authenticate)])
    def validate(request: RunRequest):
        with operation():
            return runtime.validate(request)

    @app.post("/visualize", dependencies=[Depends(authenticate)])
    def visualize(request: RunRequest):
        with operation():
            return runtime.visualize(request)

    return app

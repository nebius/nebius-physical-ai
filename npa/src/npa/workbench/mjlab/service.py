"""Expose authenticated, serialized MJLab operations with scoped S3 access."""

from __future__ import annotations

import hmac
import os
import threading

from fastapi import Depends, FastAPI, Header, HTTPException

from npa.workbench.storage_scope import StorageScope, use_storage_scope

from . import runtime
from .schemas import EvalRequest, ExportRequest, MjlabError, TrainRequest


class _Service:
    def __init__(self, token, scope):
        self.token = token
        self.scope = scope
        self.lock = threading.Lock()

    def authenticate(self, authorization: str = Header(default="")):
        if not self.token:
            raise HTTPException(503, "MJLAB_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {self.token}"):
            raise HTTPException(401, "Invalid token")

    def call(self, operation, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Another MJLab operation is active")
        try:
            with use_storage_scope(self.scope):
                return operation(request)
        except (ValueError, MjlabError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            self.lock.release()


def create_app(*, token: str | None = None, allowed_s3_roots=None) -> FastAPI:
    """Create the MJLab HTTP service; no GPU or model imports occur at startup.

    Args:
        token: Bearer token; defaults to MJLAB_TOKEN.
        allowed_s3_roots: Authorized prefixes; defaults to MJLAB_ALLOWED_S3_ROOTS.
    Returns:
        A service with health, status, system-info, list, train, eval and export.
    Raises:
        ValueError: Invalid storage-root configuration.
    """
    scope = (
        StorageScope.from_env("MJLAB")
        if allowed_s3_roots is None
        else StorageScope.from_config(s3_roots=allowed_s3_roots)
    )
    service = _Service(
        token if token is not None else os.environ.get("MJLAB_TOKEN", ""), scope
    )
    app = FastAPI(title="NPA MJLab")
    app.state.mjlab = service
    app.add_api_route("/health", lambda: {"status": "ok", "service": "mjlab"})
    auth = [Depends(service.authenticate)]
    app.add_api_route(
        "/status",
        lambda: {"busy": service.lock.locked(), **runtime.system_info()},
        dependencies=auth,
    )
    app.add_api_route("/system-info", runtime.system_info, dependencies=auth)
    app.add_api_route("/list", _list_tasks, dependencies=auth)
    _capability_routes(app, service, auth)
    return app


def _list_tasks():
    try:
        return runtime.list_tasks()
    except MjlabError as exc:
        raise HTTPException(503, str(exc)) from exc


def _capability_routes(app, service, auth):
    @app.post("/train", dependencies=auth)
    def train(request: TrainRequest):
        return service.call(runtime.train, request)

    @app.post("/eval", dependencies=auth)
    @app.post("/run", dependencies=auth)
    def evaluate(request: EvalRequest):
        return service.call(runtime.evaluate, request)

    @app.post("/export", dependencies=auth)
    def export(request: ExportRequest):
        return service.call(runtime.export, request)

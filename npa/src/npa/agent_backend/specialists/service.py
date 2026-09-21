"""Serve authenticated specialist controls and a local browser monitor."""

from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
import os
from pathlib import Path
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .config import load_config
from .team import SpecialistTeam
from .worker import supervise


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Submit(_Request):
    goal: str = Field(min_length=1)
    specialist: str = "auto"
    task_id: str = ""
    parent_id: str = ""


class _Pause(_Request):
    paused: bool


class _Reconcile(_Request):
    call_id: str = ""
    result: dict | None = None
    retry: bool = False


def create_app(config_path: str, *, with_workers: bool = True, team=None):
    """Build an authenticated service using the same durable coordinator as the SDK.

    Args: config_path: Operator JSON. with_workers: Supervise local workers when true.
        team: Optional injected team for tests.
    Returns: FastAPI application with monitor and task/control endpoints.
    Raises: ValueError: The service bearer credential is missing or too short.
    """
    active = team or SpecialistTeam(load_config(config_path))
    token = os.environ.get(active.config.token_env, "")
    if len(token) < 24:
        raise ValueError("set a service bearer credential of at least 24 characters")

    @asynccontextmanager
    async def lifespan(app):
        with supervise(config_path) if with_workers else nullcontext():
            yield

    app = FastAPI(title="Workbench specialists", lifespan=lifespan)
    auth = _authentication(token)
    _static_routes(app)
    _task_routes(app, active, auth)
    _control_routes(app, active, auth)
    _error_routes(app)
    return app


def _authentication(token):
    bearer = HTTPBearer(auto_error=False)

    def authenticate(credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credential is None or not secrets.compare_digest(
            credential.credentials, token
        ):
            raise HTTPException(
                401, "Authentication required", headers={"WWW-Authenticate": "Bearer"}
            )

    return authenticate


def _static_routes(app):
    directory = Path(__file__).with_name("ui")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(
            directory / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/app.js", include_in_schema=False)
    def script():
        return FileResponse(directory / "app.js", media_type="text/javascript")

    @app.get("/health")
    def health():
        return {"status": "ok"}


def _task_routes(app, team, auth):
    @app.get("/api/status", dependencies=[Depends(auth)])
    def status():
        return team.status()

    @app.post("/api/tasks", dependencies=[Depends(auth)])
    def submit(request: _Submit):
        return team.submit(**request.model_dump())

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(auth)])
    def task(task_id: str):
        return team.status(task_id)

    @app.get("/api/tasks/{task_id}/patch", dependencies=[Depends(auth)])
    def patch(task_id: str):
        return PlainTextResponse(
            team.patch(task_id), headers={"Cache-Control": "no-store"}
        )


def _control_routes(app, team, auth):
    @app.post("/api/tasks/{task_id}/cancel", dependencies=[Depends(auth)])
    def cancel(task_id: str):
        return team.cancel(task_id)

    @app.post("/api/tasks/{task_id}/pause", dependencies=[Depends(auth)])
    def pause(task_id: str, request: _Pause):
        return team.pause(task_id=task_id, paused=request.paused)

    @app.post("/api/profiles/{specialist}/pause", dependencies=[Depends(auth)])
    def pause_profile(specialist: str, request: _Pause):
        return team.pause(specialist=specialist, paused=request.paused)

    @app.post("/api/tasks/{task_id}/reconcile", dependencies=[Depends(auth)])
    def reconcile(task_id: str, request: _Reconcile):
        return team.reconcile(task_id, **request.model_dump())


def _error_routes(app):
    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "Unknown task or specialist"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

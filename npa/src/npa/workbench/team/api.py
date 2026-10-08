"""Expose authenticated team operations without exposing the private scheduler API."""

from __future__ import annotations

import fcntl
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .authentication import TokenVerifier
from .access_profile import access_profile
from .browser import create_browser, install_browser, secure_response
from .enrollment import verify_enrollment
from .errors import TeamError
from .models import SubmitRequest, load_config
from .service import TeamService


def create_app(config_path: Path, *, service=None, verifier=None):
    """Build the authenticated API with a single durable supervisor process.

    Args:
        config_path: Administrator-owned policy, reloaded on every operation.
        service, verifier: Injectable boundaries for offline acceptance tests.
    Returns:
        FastAPI application serving only team operations.
    Raises:
        TeamError: Installation configuration is invalid.
    """
    service = service or TeamService(
        lambda: load_config(config_path), enrollment_check=verify_enrollment
    )
    verifier = verifier or TokenVerifier(service.config().identity)
    browser = create_browser(service.initial, verifier)

    app = FastAPI(
        title="Workbench team API",
        lifespan=_lifespan(service, browser),
        docs_url=None,
        redoc_url=None,
    )
    actor = _authentication(service, verifier, browser)
    _errors(app)
    _run_routes(app, service, actor)
    _artifact_routes(app, service, actor)
    _access_routes(app, service, actor, browser)
    return app


def _lifespan(service, browser):
    @asynccontextmanager
    async def lifespan(app):
        with (service.initial.state_dir / "server.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            service.ledger.recover()
            try:
                yield
            finally:
                if browser:
                    browser.provider.close()

    return lifespan


def _authentication(service, verifier, browser):
    def actor(request: Request):
        service.config()
        authorization = request.headers.get("authorization", "")
        if not authorization and browser:
            authorization = browser.authorization(request)
        return verifier.verify(authorization)

    return actor


def _access_routes(app, service, actor, browser):
    @app.middleware("http")
    async def privacy(request, call_next):
        return secure_response(await call_next(request))

    @app.get("/v1/me")
    def me(request: Request, identity=Depends(actor)):
        profile = access_profile(service.config(), identity)
        if browser and not request.headers.get("authorization"):
            profile.update(browser.details(request))
        return profile

    if browser:
        install_browser(app, browser, actor)


def _errors(app):
    @app.exception_handler(TeamError)
    async def team_error(request, error):
        return JSONResponse({"error": str(error)}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"error": "invalid team request"}, status_code=422)

    @app.exception_handler(Exception)
    async def unavailable(request, error):
        return JSONResponse(
            {"error": "operation unavailable; reconcile before retrying"},
            status_code=503,
        )


def _run_routes(app, service, actor):
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/runs", status_code=202)
    def submit(request: SubmitRequest, identity=Depends(actor)):
        return service.submit(identity, request)

    @app.get("/v1/runs")
    def runs(workspace: str, identity=Depends(actor)):
        return {"runs": service.list(identity, workspace)}

    @app.get("/v1/runs/{run_id}")
    def status(run_id: str, identity=Depends(actor)):
        return service.get(identity, run_id)

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, identity=Depends(actor)):
        return service.cancel(identity, run_id)

    @app.post("/v1/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str, identity=Depends(actor)):
        return service.resume(identity, run_id)

    @app.get("/v1/runs/{run_id}/logs")
    def logs(run_id: str, identity=Depends(actor)):
        return {"logs": service.logs(identity, run_id)}


def _artifact_routes(app, service, actor):
    @app.get("/v1/runs/{run_id}/artifacts")
    def artifacts(run_id: str, identity=Depends(actor)):
        return {"artifacts": service.artifacts(identity, run_id)}

    @app.get("/v1/runs/{run_id}/artifacts/{relative:path}")
    def artifact(run_id: str, relative: str, identity=Depends(actor)):
        body = service.artifacts(identity, run_id, relative)
        return StreamingResponse(_stream(body), media_type="application/octet-stream")


def _stream(body):
    try:
        yield from body.iter_chunks()
    finally:
        body.close()

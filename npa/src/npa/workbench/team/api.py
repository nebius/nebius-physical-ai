"""Expose authenticated team operations without exposing the private scheduler API."""

from __future__ import annotations

import fcntl
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .account_authentication import AccountAuthentication
from .access_profile import access_profile
from .authentication import TokenVerifier
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
    if service.initial.account_namespace:
        verifier = AccountAuthentication(service.initial, external=verifier)
    else:
        verifier = verifier or TokenVerifier(service.initial.identity)
    app = FastAPI(
        title="Workbench team API",
        lifespan=_lifespan(service),
        docs_url=None,
        redoc_url=None,
    )
    actor = _authentication(service, verifier)
    _errors(app)
    _run_routes(app, service, actor)
    _artifact_routes(app, service, actor)
    _access_routes(app, service, actor)
    return app


def _lifespan(service):
    @asynccontextmanager
    async def lifespan(app):
        with (service.initial.state_dir / "server.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            service.ledger.recover()
            yield

    return lifespan


def _authentication(service, verifier):
    def actor(request: Request):
        service.config()
        return verifier.verify(request.headers.get("authorization", ""))

    return actor


def _access_routes(app, service, actor):
    @app.middleware("http")
    async def privacy(request, call_next):
        return _secure_response(await call_next(request))

    @app.get("/v1/me")
    def me(identity=Depends(actor)):
        return access_profile(service.config(), identity)


def _secure_response(response):
    """Attach privacy and browser-hardening headers to every API response.

    Args:
        response: Completed HTTP response from the team API.
    Returns:
        Response with explicit cache and embedding protections.
    Raises:
        None.
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'none'"
    )
    return response


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

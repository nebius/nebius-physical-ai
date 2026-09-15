"""Authenticated detector service with persistent run and artifact identity."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import platform
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from npa.workbench.storage_scope import StorageAuthorizationError, StorageScope

from .evaluation import DetectionEvaluationError, evaluate_detector, evaluation_identity
from .run_store import RunStore, RunStoreError
from .schemas import DEFAULT_LANCE_URI, EvalRequest, EvalResponse, RunListResponse, StatusResponse, TrainRequest, TrainResponse
from .storage import read_bytes_uri
from .training import (
    checkpoint_uri_pattern,
    compute_manifest_sha256,
    make_run_id,
    metrics_uri,
    resolve_num_classes,
    train_detector,
)

LOGGER = logging.getLogger(__name__)


def _storage_policy(root: str) -> StorageScope:
    if root.startswith("s3://"):
        return StorageScope.from_config(s3_roots=[root])
    local_root = urlparse(root).path if root.startswith("file://") else root
    return StorageScope.from_config(local_roots=[local_root])


def create_app(*, auth_mode: str | None = None, token: str | None = None, state_dir: str | Path | None = None, output_scope: str | None = None) -> FastAPI:
    """Run one service worker per persistent volume; protected routes share its scope."""
    resolved_auth_mode = auth_mode or os.environ.get("DETECTION_TRAINING_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("DETECTION_TRAINING_TOKEN", "")
    scope = output_scope if output_scope is not None else os.environ.get("NPA_OUTPUT_PATH", "")
    if resolved_auth_mode not in {"none", "token"}:
        raise ValueError("unsupported detection-training authentication mode")
    store = RunStore(state_dir, scope=scope)
    output_policy = _storage_policy(scope) if scope else None
    input_root = os.environ.get("NPA_INPUT_PATH", "")
    input_policy = _storage_policy(input_root) if input_root else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if resolved_auth_mode == "token" and not resolved_token:
            raise RuntimeError("DETECTION_TRAINING_TOKEN is not configured")
        store.start()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            store.close()

    app = FastAPI(title="NPA Detection Training", lifespan=lifespan)
    app.state.store = store
    app.state.ready = False
    if resolved_auth_mode == "none":
        LOGGER.warning("detection-training service started with auth disabled")

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="service authentication is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    def require_output_scope(uri: str) -> None:
        # Never let authenticated clients use service credentials as a storage proxy
        # outside this deployment's configured destination.
        if output_policy is not None:
            try:
                output_policy.authorize(uri, operation="artifact")
            except StorageAuthorizationError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    def require_input_scope(uri: str) -> None:
        if input_policy is not None:
            try:
                input_policy.authorize(uri, operation="dataset")
            except StorageAuthorizationError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
        # This public endpoint reveals no workloads, credentials, paths, or hardware.
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")
        try:
            with store.connection():
                pass
        except (OSError, sqlite3.Error, RunStoreError) as exc:
            raise HTTPException(status_code=503, detail="run store is unavailable") from exc
        return {"status": "ok"}

    @app.get("/health")
    async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return {"status": "ok", "runs": len(store.list())}

    @app.get("/system-info")
    async def system_info(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return system_info_payload()

    @app.get("/list", response_model=RunListResponse)
    @app.get("/runs", response_model=RunListResponse)
    async def runs(request: Request, authorization: str = Header(default="")) -> RunListResponse:
        await require_auth(request, authorization)
        return RunListResponse(runs=store.list())

    @app.post("/train", response_model=TrainResponse)
    async def train(body: TrainRequest, background_tasks: BackgroundTasks, request: Request, authorization: str = Header(default="")) -> TrainResponse:
        await require_auth(request, authorization)
        effective_output = body.checkpoint_s3.uri or body.output_uri
        require_output_scope(effective_output)
        if any((body.checkpoint_s3.endpoint_url, body.checkpoint_s3.aws_access_key_id, body.checkpoint_s3.aws_secret_access_key)):
            raise HTTPException(status_code=400, detail="service runs use deployment storage credentials and endpoint")
        if body.lance_uri == DEFAULT_LANCE_URI and os.environ.get("NPA_INPUT_PATH"):
            body = body.model_copy(update={"lance_uri": os.environ["NPA_INPUT_PATH"]})
        require_input_scope(body.lance_uri)
        manifest = compute_manifest_sha256("train", body.model_dump(mode="json"))
        run_id = make_run_id("train", manifest)
        response = TrainResponse(
            run_id=run_id, status="queued",
            checkpoint_uri_pattern=checkpoint_uri_pattern(effective_output, run_id),
            metrics_uri=metrics_uri(effective_output, run_id),
            total_epochs=body.epochs, manifest_sha256=manifest,
        )
        store.create(StatusResponse(
            run_id=run_id, status="queued", total_epochs=body.epochs,
            checkpoint_uri_pattern=response.checkpoint_uri_pattern,
            metrics_uri=response.metrics_uri, manifest_sha256=manifest,
        ))
        background_tasks.add_task(_run_training, body, run_id, store)
        return response

    @app.post("/eval", response_model=EvalResponse)
    async def evaluate(body: EvalRequest, request: Request, authorization: str = Header(default="")) -> EvalResponse:
        await require_auth(request, authorization)
        require_output_scope(body.checkpoint_uri)
        require_output_scope(body.output_uri)
        if body.lance_uri == DEFAULT_LANCE_URI and os.environ.get("NPA_INPUT_PATH"):
            body = body.model_copy(update={"lance_uri": os.environ["NPA_INPUT_PATH"]})
        require_input_scope(body.lance_uri)
        eval_run_id, manifest = evaluation_identity(body)
        try:
            previous = store.get(eval_run_id)
        except KeyError:
            try:
                store.create(StatusResponse(run_id=eval_run_id, kind="eval", status="running", manifest_sha256=manifest))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(status_code=409, detail={"message": "evaluation already accepted", "run_id": eval_run_id}) from exc
        else:
            if previous.manifest_sha256 != manifest:
                raise HTTPException(status_code=409, detail="evaluation identity collision; use a new output prefix")
            if previous.status == "completed" and previous.evaluation is not None:
                return previous.evaluation
            raise HTTPException(status_code=409, detail={"message": "evaluation already accepted; inspect its durable status", "run_id": eval_run_id, "status": previous.status})
        try:
            # Keep GPU work off the event loop so readiness/status remain responsive.
            from starlette.concurrency import run_in_threadpool
            result = await run_in_threadpool(evaluate_detector, body)
            if result.eval_run_id != eval_run_id or result.manifest_sha256 != manifest:
                raise DetectionEvaluationError("evaluation result identity differs from the accepted run")
            store.update(eval_run_id, status="completed", evaluation=result, artifacts=result.artifacts, last_metrics={"mAP": result.mAP, "mAP_50": result.mAP_50, "mAP_75": result.mAP_75})
            return result
        except Exception as exc:
            store.update(eval_run_id, status="failed", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/status", response_model=StatusResponse)
    async def status(run_id: str, request: Request, authorization: str = Header(default="")) -> StatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id, store=store)

    @app.get("/artifacts")
    async def artifacts(run_id: str, request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        current = status_for_run(run_id, store=store)
        return {"run_id": run_id, "artifacts": [artifact.model_dump(mode="json") for artifact in current.artifacts]}

    @app.get("/artifacts/content")
    async def artifact_content(run_id: str, sha256: str, request: Request, authorization: str = Header(default="")) -> Response:
        await require_auth(request, authorization)
        current = status_for_run(run_id, store=store)
        artifact = next((item for item in current.artifacts if item.sha256 == sha256), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail="unknown artifact")
        require_output_scope(artifact.uri)
        try:
            from starlette.concurrency import run_in_threadpool
            data = await run_in_threadpool(read_bytes_uri, artifact.uri)
        except Exception as exc:
            raise HTTPException(status_code=409, detail="recorded artifact is no longer readable") from exc
        if hashlib.sha256(data).hexdigest() != artifact.sha256 or len(data) != artifact.size_bytes:
            raise HTTPException(status_code=409, detail="recorded artifact integrity check failed")
        return Response(data, media_type=artifact.media_type, headers={"ETag": f'"{artifact.sha256}"'})

    return app


def status_for_run(run_id: str, *, store: RunStore | None = None) -> StatusResponse:
    resolved = store or RunStore(scope=os.environ.get("NPA_OUTPUT_PATH", ""))
    try:
        return resolved.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown run_id") from exc


def system_info_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "ok", "python": platform.python_version(), "platform": platform.platform()}
    payload["runtime_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    storage_scope_path = Path(__file__).parent.parent / "storage_scope.py"
    payload["runtime_source_sha256"]["../storage_scope.py"] = hashlib.sha256(storage_scope_path.read_bytes()).hexdigest()
    try:
        import torch
        payload.update({"torch": getattr(torch, "__version__", ""), "cuda_available": bool(torch.cuda.is_available()), "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0})
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        payload["torch_error"] = str(exc)
    return payload


def _run_training(body: TrainRequest, run_id: str, store: RunStore) -> None:
    def update(status: str, epochs_completed: int, metrics: dict[str, Any], error: str | None) -> None:
        # Completion is committed only after the returned artifact manifest is verified.
        store.update(run_id, status="running" if status == "completed" else status, epochs_completed=epochs_completed, last_metrics=metrics, error=error)

    def publish_artifacts(artifacts: list[Any]) -> None:
        store.update(run_id, artifacts=artifacts)

    try:
        LOGGER.info("starting detection training run_id=%s num_classes=%s", run_id, resolve_num_classes(body))
        result = train_detector(body, run_id=run_id, status_callback=update, artifact_callback=publish_artifacts)
        store.update(run_id, status=result.status, epochs_completed=result.total_epochs, artifacts=result.artifacts, error=None)
    except Exception as exc:
        current = store.get(run_id)
        if current.status not in {"failed", "interrupted", "completed"}:
            # Persist the error for the authenticated owner; never serialize the request's credentials.
            store.update(run_id, status="failed", error=str(exc))


app = create_app()

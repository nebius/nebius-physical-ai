"""FastAPI service for the RoboCasa workbench."""

from __future__ import annotations

import hmac
import json
import logging
import multiprocessing
import os
import shutil
import signal
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from npa.workbench.robocasa.capabilities import (
    WORKER_ASSET_TEMP_ROOT_ENV,
    WORKER_TEMP_ROOT_ENV,
    RoboCasaError,
    _assets_root,
    compute_manifest_sha256,
    make_run_id,
    run_capability_with_output,
    system_info,
    verify_runtime_identity,
)
from npa.workbench.robocasa.schemas import (
    RoboCasaRunListResponse,
    RoboCasaRunRequest,
    RoboCasaRunResponse,
    RoboCasaStatusResponse,
    RoboCasaSystemInfo,
)

LOGGER = logging.getLogger(__name__)
_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_ACTIVE_STATUSES = frozenset({"queued", "running"})
_WORKER_MESSAGE_LIMIT = 16 * 1024 * 1024
_WORKER_TERMINATE_GRACE_SECONDS = 2.0


class RunCapacityError(RoboCasaError):
    """Raised when the bounded run registry has no reclaimable slot."""


@dataclass(frozen=True)
class _WorkerOutcome:
    result: dict[str, Any] | None = None
    error: str | None = None
    timed_out: bool = False
    stopped: bool = True


class GpuExecutionGate:
    """Serialize GPU work and permanently fail closed after an orphaned worker."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._busy = False
        self._poisoned = False

    @property
    def available(self) -> bool:
        with self._condition:
            return not self._poisoned

    def acquire(self) -> bool:
        with self._condition:
            while self._busy and not self._poisoned:
                self._condition.wait()
            if self._poisoned:
                return False
            self._busy = True
            return True

    def release(self) -> None:
        with self._condition:
            if not self._busy:
                raise RuntimeError("RoboCasa GPU execution gate is not acquired")
            self._busy = False
            self._condition.notify()

    def poison(self) -> None:
        with self._condition:
            self._poisoned = True
            self._condition.notify_all()


_GPU_EXECUTION_GATE = GpuExecutionGate()


class RunRegistry:
    """Concurrency-safe bounded run status registry.

    TTL and size eviction apply only to terminal runs. Active runs are never
    discarded; admission fails when every bounded slot is active.
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 24 * 60 * 60,
        clock: Any = time.monotonic,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, tuple[RoboCasaStatusResponse, float, int]] = {}
        self._sequence = 0

    def __len__(self) -> int:
        with self._lock:
            self._evict_locked()
            return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self._evict_locked()
            return iter(tuple(self._entries))

    def __setitem__(self, run_id: str, status: RoboCasaStatusResponse) -> None:
        with self._lock:
            self._reserve_new_slot_locked(run_id)
            self._sequence += 1
            self._entries[run_id] = (status, self._clock(), self._sequence)
            self._evict_locked()

    def get(self, run_id: str) -> RoboCasaStatusResponse | None:
        with self._lock:
            self._evict_locked()
            entry = self._entries.get(run_id)
            return entry[0] if entry else None

    def values(self) -> list[RoboCasaStatusResponse]:
        with self._lock:
            self._evict_locked()
            return [entry[0] for entry in self._entries.values()]

    def enqueue(
        self, run_id: str, status: RoboCasaStatusResponse
    ) -> tuple[bool, RoboCasaStatusResponse]:
        """Register work unless the same deterministic run is already active."""
        with self._lock:
            self._evict_locked()
            current = self._entries.get(run_id)
            if current is not None and current[0].status in _ACTIVE_STATUSES:
                if current[0].manifest_sha256 != status.manifest_sha256:
                    raise RoboCasaError(
                        f"active deterministic run id collision: {run_id}"
                    )
                return False, current[0]
            self._reserve_new_slot_locked(run_id)
            self._sequence += 1
            self._entries[run_id] = (status, self._clock(), self._sequence)
            self._evict_locked()
            return True, status

    def update(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self._lock:
            current = self._entries.get(run_id)
            if current is None:
                return
            self._sequence += 1
            updated = current[0].model_copy(
                update={"status": status, "result": result, "error": error}
            )
            self._entries[run_id] = (updated, self._clock(), self._sequence)
            self._evict_locked()

    def _evict_locked(self) -> None:
        now = self._clock()
        expired = [
            run_id
            for run_id, (record, updated_at, _sequence) in self._entries.items()
            if record.status in _TERMINAL_STATUSES
            and now - updated_at >= self.ttl_seconds
        ]
        for run_id in expired:
            self._entries.pop(run_id, None)

        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return
        terminal = sorted(
            (
                (updated_at, sequence, run_id)
                for run_id, (record, updated_at, sequence) in self._entries.items()
                if record.status in _TERMINAL_STATUSES
            )
        )
        for _updated_at, _sequence, run_id in terminal[:overflow]:
            self._entries.pop(run_id, None)

    def _reserve_new_slot_locked(self, run_id: str) -> None:
        self._evict_locked()
        if run_id in self._entries or len(self._entries) < self.max_entries:
            return
        terminal = min(
            (
                (updated_at, sequence, candidate)
                for candidate, (record, updated_at, sequence) in self._entries.items()
                if record.status in _TERMINAL_STATUSES
            ),
            default=None,
        )
        if terminal is not None:
            self._entries.pop(terminal[2], None)
        if len(self._entries) >= self.max_entries:
            raise RunCapacityError(
                f"RoboCasa run registry is full ({self.max_entries} active runs)"
            )


RUNS = RunRegistry(
    max_entries=int(os.environ.get("ROBOCASA_RUNS_MAX_ENTRIES", "256")),
    ttl_seconds=float(os.environ.get("ROBOCASA_RUNS_TTL_SECONDS", str(24 * 60 * 60))),
)


def create_app(
    *,
    auth_mode: str | None = None,
    token: str | None = None,
    runs: RunRegistry | None = None,
    execution_lock: Any = None,
    capability_executor: Any = None,
) -> FastAPI:
    """Create the RoboCasa FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("ROBOCASA_AUTH_MODE", "none")
    resolved_token = (
        token if token is not None else os.environ.get("ROBOCASA_TOKEN", "")
    )
    registry = runs if runs is not None else RUNS
    gpu_lock = execution_lock if execution_lock is not None else _GPU_EXECUTION_GATE
    app = FastAPI(title="NPA RoboCasa")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "robocasa service started with auth disabled; every endpoint is reachable "
            "without a token. Set ROBOCASA_AUTH_MODE=token and ROBOCASA_TOKEN."
        )

    async def require_auth(
        request: Request, authorization: str = Header(default="")
    ) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(
                status_code=500, detail="ROBOCASA_TOKEN is not configured"
            )
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        execution_available = _execution_available(gpu_lock)
        return {
            "status": "ok" if execution_available else "degraded",
            "runs": len(registry),
            "execution_available": execution_available,
        }

    @app.get("/system-info", response_model=RoboCasaSystemInfo)
    async def system_info_endpoint(
        request: Request, authorization: str = Header(default="")
    ) -> RoboCasaSystemInfo:
        await require_auth(request, authorization)
        # torch/CUDA and Gymnasium discovery can import native extensions and
        # probe drivers. Keep that blocking work off the ASGI event-loop thread.
        return await run_in_threadpool(system_info)

    @app.get("/runs", response_model=RoboCasaRunListResponse)
    async def runs(
        request: Request, authorization: str = Header(default="")
    ) -> RoboCasaRunListResponse:
        await require_auth(request, authorization)
        return RoboCasaRunListResponse(runs=registry.values())

    @app.post("/run", response_model=RoboCasaRunResponse)
    async def run(
        body: RoboCasaRunRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RoboCasaRunResponse:
        await require_auth(request, authorization)
        try:
            verify_runtime_identity(
                body.expected_image_source_sha,
                body.expected_image_manifest_digest,
            )
        except RoboCasaError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not _execution_available(gpu_lock):
            raise HTTPException(
                status_code=503,
                detail="RoboCasa GPU execution is unavailable after worker cleanup failed",
            )
        manifest = compute_manifest_sha256("run", body.model_dump(mode="json"))
        run_id = make_run_id(body.capability, manifest)
        queued = RoboCasaStatusResponse(
            run_id=run_id,
            status="queued",
            capability=body.capability,
            env_id=body.env_id,
            output_uri=body.output_uri,
            manifest_sha256=manifest,
        )
        try:
            accepted, current = registry.enqueue(run_id, queued)
        except RunCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RoboCasaError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if accepted:
            background_tasks.add_task(
                _run_capability,
                body,
                run_id,
                registry,
                gpu_lock,
                capability_executor,
            )
        return _run_response(current, manifest)

    @app.get("/status", response_model=RoboCasaStatusResponse)
    async def status(
        run_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RoboCasaStatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id, runs=registry)

    return app


def status_for_run(
    run_id: str, *, runs: RunRegistry | None = None
) -> RoboCasaStatusResponse:
    registry = runs if runs is not None else RUNS
    status = registry.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return status


def _run_response(
    status: RoboCasaStatusResponse, manifest_sha256: str
) -> RoboCasaRunResponse:
    return RoboCasaRunResponse(
        run_id=status.run_id,
        status=status.status,
        env_id=status.env_id,
        capability=status.capability,
        output_uri=status.output_uri,
        manifest_sha256=manifest_sha256,
    )


def _run_capability(
    body: RoboCasaRunRequest,
    run_id: str,
    runs: RunRegistry | None = None,
    execution_lock: Any = None,
    capability_executor: Any = None,
) -> None:
    registry = runs if runs is not None else RUNS
    gpu_lock = execution_lock if execution_lock is not None else _GPU_EXECUTION_GATE

    def update(status: str, result: dict[str, Any] | None, error: str | None) -> None:
        registry.update(run_id, status=status, result=result, error=error)

    release_lock = True
    if gpu_lock.acquire() is False:
        update(
            "failed",
            None,
            "RoboCasa GPU execution is unavailable after worker cleanup failed",
        )
        return
    try:
        update("running", None, None)
        LOGGER.info(
            "starting robocasa run_id=%s capability=%s env_id=%s",
            run_id,
            body.capability,
            body.env_id,
        )
        if capability_executor is None:
            outcome = _execute_capability_in_worker(body)
        else:
            with tempfile.TemporaryDirectory(prefix="robocasa_") as tmp:
                outcome = _WorkerOutcome(
                    result=capability_executor(body, output_dir=Path(tmp))
                )
        if not outcome.stopped:
            release_lock = False
            poison = getattr(gpu_lock, "poison", None)
            if callable(poison):
                poison()
            LOGGER.critical(
                "robocasa worker for run_id=%s did not stop; retaining the GPU "
                "execution lock to fail closed",
                run_id,
            )
        if outcome.error is not None:
            update("failed", None, outcome.error)
        elif outcome.result is None:
            update("failed", None, "RoboCasa worker returned no result")
        else:
            update("completed", outcome.result, None)
    except RoboCasaError as exc:
        update("failed", None, str(exc))
    except Exception as exc:  # pragma: no cover - defensive service boundary.
        update("failed", None, str(exc))
    finally:
        if release_lock:
            gpu_lock.release()


def _execution_available(execution_gate: Any) -> bool:
    available = getattr(execution_gate, "available", True)
    return bool(available() if callable(available) else available)


def _capability_worker_entry(sender: Any, request_payload: dict[str, Any]) -> None:
    """Execute one capability in its own killable process group."""
    try:
        os.setsid()
        _send_worker_message(
            sender,
            {"kind": "ready", "process_group": os.getpgrp()},
        )
        payload = dict(request_payload)
        output_dir = Path(str(payload.pop("_worker_output_dir")))
        asset_temp_root = str(payload.pop("_worker_asset_temp_root", "")).strip()
        worker_temp_root = output_dir / "scratch"
        worker_temp_root.mkdir(parents=True, exist_ok=True)
        os.environ[WORKER_TEMP_ROOT_ENV] = str(worker_temp_root)
        if asset_temp_root:
            os.environ[WORKER_ASSET_TEMP_ROOT_ENV] = asset_temp_root
        else:
            os.environ.pop(WORKER_ASSET_TEMP_ROOT_ENV, None)
        body = RoboCasaRunRequest.model_validate(payload)
        result = run_capability_with_output(body, output_dir=output_dir)
        message = {"kind": "result", "result": result}
    except RoboCasaError as exc:
        message = {"kind": "error", "error": str(exc)}
    except Exception as exc:  # pragma: no cover - child process boundary.
        message = {
            "kind": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        _send_worker_message(sender, message)
        # Keep the process-group leader alive until the parent has retained the
        # result and terminates the whole group. This closes the descendant leak
        # race that occurs when a simulator helper outlives a successful worker.
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _send_worker_message(sender: Any, payload: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        encoded = json.dumps(
            {
                "kind": "error",
                "error": f"RoboCasa worker result is not JSON serializable: {exc}",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if len(encoded) > _WORKER_MESSAGE_LIMIT:
        encoded = json.dumps(
            {
                "kind": "error",
                "error": "RoboCasa worker result exceeds the IPC size limit",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    sender.send_bytes(encoded)


def _execute_capability_in_worker(
    body: RoboCasaRunRequest,
    *,
    worker_target: Any = _capability_worker_entry,
    process_context: Any = None,
    output_root: Path | None = None,
) -> _WorkerOutcome:
    """Run one request with a hard deadline in an independently killable child."""
    context = process_context or multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=True)
    worker_output = Path(
        tempfile.mkdtemp(
            prefix="robocasa-worker-",
            dir=str(output_root) if output_root is not None else None,
        )
    )
    worker_id = uuid.uuid4().hex
    worker_asset_output: Path | None = None
    try:
        worker_asset_output = (
            _assets_root() / ".npa_asset_fetch" / "workers" / worker_id
        )
        worker_asset_output.mkdir(parents=True, exist_ok=False)
    except (OSError, RoboCasaError):
        worker_asset_output = None
    if body.download_assets and worker_asset_output is None:
        receiver.close()
        sender.close()
        shutil.rmtree(worker_output)
        raise RoboCasaError(
            "cannot create parent-owned RoboCasa asset staging directory"
        )
    request_payload = body.model_dump(mode="json")
    request_payload["_worker_output_dir"] = str(worker_output)
    request_payload["_worker_asset_temp_root"] = (
        str(worker_asset_output) if worker_asset_output is not None else ""
    )
    process = context.Process(
        target=worker_target,
        args=(sender, request_payload),
        name=f"robocasa-{body.capability}",
    )
    try:
        process.start()
    except Exception as exc:
        receiver.close()
        sender.close()
        shutil.rmtree(worker_output)
        if worker_asset_output is not None:
            shutil.rmtree(worker_asset_output, ignore_errors=True)
        raise RoboCasaError(f"failed to start RoboCasa worker: {exc}") from exc
    sender.close()
    message: dict[str, Any] | None = None
    timed_out = False
    cleanup_error: str | None = None
    protocol_error: str | None = None
    process_group: int | None = None
    deadline = time.monotonic() + body.timeout_seconds
    try:
        remaining = max(0.0, deadline - time.monotonic())
        if receiver.poll(remaining):
            ready = _receive_worker_message(receiver)
            reported_group = ready.get("process_group")
            if (
                ready.get("kind") == "ready"
                and isinstance(process.pid, int)
                and reported_group == process.pid
            ):
                process_group = reported_group
                remaining = max(0.0, deadline - time.monotonic())
                if receiver.poll(remaining):
                    message = _receive_worker_message(receiver)
                else:
                    timed_out = True
            else:
                protocol_error = (
                    "RoboCasa worker did not acknowledge its isolated process group"
                )
        else:
            timed_out = True
    except (EOFError, OSError, ValueError) as exc:
        protocol_error = f"RoboCasa worker IPC failed: {type(exc).__name__}: {exc}"
    finally:
        # The child waits after reporting its result so the parent can terminate
        # the complete process group, including simulator descendants, on every
        # path. Keep this endpoint open until the group has been signalled.
        try:
            stopped = _stop_worker(
                process,
                terminate=True,
                process_group=process_group,
            )
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary.
            stopped = False
            cleanup_error = (
                f"RoboCasa worker cleanup failed: {type(exc).__name__}: {exc}"
            )
            try:
                _signal_worker(process, signal.SIGKILL, process_group=process_group)
            except Exception:
                LOGGER.debug(
                    "RoboCasa fallback SIGKILL failed during worker cleanup",
                    exc_info=True,
                )
        if process_group is None:
            stopped = False
            protocol_error = protocol_error or (
                "RoboCasa worker isolation was not acknowledged"
            )
        try:
            receiver.close()
        except OSError:
            stopped = False
            cleanup_error = cleanup_error or "RoboCasa worker IPC cleanup failed"
    if not stopped:
        try:
            _reap_worker_async(process, worker_output, worker_asset_output)
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary.
            cleanup_error = cleanup_error or (
                f"RoboCasa worker reaper failed: {type(exc).__name__}: {exc}"
            )
    elif hasattr(process, "close"):
        process.close()
        try:
            shutil.rmtree(worker_output)
            if worker_asset_output is not None:
                shutil.rmtree(worker_asset_output)
        except OSError as exc:
            stopped = False
            cleanup_error = cleanup_error or (
                f"RoboCasa worker output cleanup failed: {type(exc).__name__}: {exc}"
            )

    if cleanup_error is not None:
        return _WorkerOutcome(error=cleanup_error, stopped=False)
    if protocol_error is not None:
        return _WorkerOutcome(
            error=protocol_error,
            timed_out=timed_out,
            stopped=False,
        )
    if timed_out:
        return _WorkerOutcome(
            error=(
                f"RoboCasa capability exceeded timeout_seconds={body.timeout_seconds}"
            ),
            timed_out=True,
            stopped=stopped,
        )
    if message is None:
        return _WorkerOutcome(
            error="RoboCasa worker exited without a result", stopped=stopped
        )
    if message.get("kind") != "result":
        return _WorkerOutcome(
            error=str(message.get("error") or "RoboCasa worker failed"),
            stopped=stopped,
        )
    result = message.get("result")
    if not isinstance(result, dict):
        return _WorkerOutcome(
            error="RoboCasa worker result must be an object", stopped=stopped
        )
    return _WorkerOutcome(result=result, stopped=stopped)


def _receive_worker_message(receiver: Any) -> dict[str, Any]:
    raw = receiver.recv_bytes(_WORKER_MESSAGE_LIMIT)
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("RoboCasa worker returned a non-object message")
    return parsed


def _stop_worker(
    process: Any,
    *,
    terminate: bool,
    process_group: int | None = None,
) -> bool:
    if not terminate:
        process.join(_WORKER_TERMINATE_GRACE_SECONDS)
        terminate = process.is_alive()
    group_alive = process_group is not None and _process_group_exists(process_group)
    if terminate and (process.is_alive() or group_alive):
        _signal_worker(process, signal.SIGTERM, process_group=process_group)
        process.join(_WORKER_TERMINATE_GRACE_SECONDS)
    leader_stopped = not process.is_alive()
    if leader_stopped and process_group is not None:
        _reap_exited_group_children(process_group)
    group_alive = process_group is not None and _process_group_exists(process_group)
    if not leader_stopped or group_alive:
        _signal_worker(process, signal.SIGKILL, process_group=process_group)
        process.join(_WORKER_TERMINATE_GRACE_SECONDS)
    leader_stopped = not process.is_alive()
    if not leader_stopped:
        return False
    if process_group is not None:
        _reap_exited_group_children(process_group)
    group_stopped = process_group is None or _wait_for_process_group_exit(
        process_group, _WORKER_TERMINATE_GRACE_SECONDS
    )
    return group_stopped


def _signal_worker(
    process: Any, signal_number: int, *, process_group: int | None
) -> None:
    if process_group is not None:
        try:
            os.killpg(process_group, signal_number)
            return
        except ProcessLookupError:
            pass
        except OSError:
            pass
    pid = process.pid
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.kill(pid, signal_number)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_exited_group_children(process_group: int) -> None:
    """Reap only exited descendants adopted from this worker process group."""

    while True:
        try:
            pid, _status = os.waitpid(-process_group, os.WNOHANG)
        except ChildProcessError:
            return
        except InterruptedError:
            continue
        if pid == 0:
            return


def _wait_for_process_group_exit(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        _reap_exited_group_children(process_group)
        if not _process_group_exists(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _reap_worker_async(
    process: Any,
    worker_output: Path,
    worker_asset_output: Path | None,
) -> None:
    def reap() -> None:
        process.join()
        shutil.rmtree(worker_output, ignore_errors=True)
        if worker_asset_output is not None:
            shutil.rmtree(worker_asset_output, ignore_errors=True)
        if hasattr(process, "close"):
            process.close()

    threading.Thread(
        target=reap,
        name=f"{getattr(process, 'name', 'robocasa-worker')}-reaper",
        daemon=True,
    ).start()


app = create_app()

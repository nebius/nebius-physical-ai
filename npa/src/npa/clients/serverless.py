"""Nebius Serverless AI endpoint client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import shutil
import shlex
import subprocess
import time
from typing import Any
from urllib.parse import urlparse


class ServerlessClientError(Exception):
    """Base exception for serverless client errors."""


class EndpointNotFoundError(ServerlessClientError):
    """Endpoint resource not found."""


class AuthError(ServerlessClientError):
    """Authentication or authorization failure. Not a NER condition."""


class NotEnoughResourcesError(ServerlessClientError):
    """Nebius project lacks capacity for the requested endpoint."""


class QuotaError(NotEnoughResourcesError):
    """Specific NER subtype for quota-limit failures."""


_NER_PATTERNS = [
    "quota exceeded",
    "quota limit",
    "limit reached",
    "insufficient capacity",
    "no capacity available",
    "scheduling failed",
    "no gpu available",
    "no resources available",
    "out of capacity",
    "resource not available",
]

_AUTH_PATTERNS = (
    "unauthorized",
    "permission denied",
    "401",
    "403",
    "forbidden",
)

_NOT_FOUND_PATTERNS = (
    "not found",
    "does not exist",
    "404",
)

_SECRET_KEY_PARTS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
)
_SENSITIVE_VALUE_FLAGS = {"--registry-password", "--token"}

logger = logging.getLogger(__name__)


class EndpointStatus(str, Enum):
    """Normalized endpoint lifecycle status."""

    UNKNOWN = "unknown"
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"

    @classmethod
    def from_value(cls, value: Any) -> "EndpointStatus":
        normalized = str(value or "").strip().lower()
        if normalized in {"running", "active", "ready"}:
            return cls.RUNNING
        if normalized in {"creating", "pending", "starting", "provisioning"}:
            return cls.CREATING
        if normalized in {"stopped", "stopping", "inactive"}:
            return cls.STOPPED
        if normalized in {"failed", "error", "crashed"}:
            return cls.FAILED
        if normalized in {"deleting", "terminating"}:
            return cls.DELETING
        if normalized in {"deleted"}:
            return cls.DELETED
        return cls.UNKNOWN


_JOB_STATUS_ALIASES = {
    "queued": {"queued", "pending", "created", "provisioning", "starting"},
    "running": {"running", "active"},
    "succeeded": {"succeeded", "success", "completed", "complete", "done"},
    "failed": {"failed", "error", "crashed"},
    "cancelling": {"cancelling", "canceling", "stopping"},
    "cancelled": {"cancelled", "canceled", "stopped"},
}
_JOB_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_JOB_QUERY_TIMEOUT = 60
_JOB_CREATE_TIMEOUT = 300
_JOB_CANCEL_TIMEOUT = 120


def _job_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    for status, aliases in _JOB_STATUS_ALIASES.items():
        if normalized in aliases:
            return status
    return "unknown"


@dataclass(frozen=True)
class EndpointSpec:
    """Create request for a Nebius Serverless AI endpoint."""

    name: str
    project_id: str
    image: str
    platform: str
    preset: str
    container_ports: list[int] = field(default_factory=lambda: [8080])
    public: bool = True
    auth: str = "none"
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    args: str = ""
    container_command: str = ""
    disk_size: str = ""
    shm_size: str = ""
    subnet_id: str = ""
    working_dir: str = ""
    preemptible: bool = False


@dataclass(frozen=True)
class EndpointInfo:
    """Endpoint details returned by Nebius."""

    id: str
    name: str
    project_id: str
    status: EndpointStatus = EndpointStatus.UNKNOWN
    url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobInfo:
    id: str
    name: str
    project_id: str
    status: str = "unknown"
    created_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    output_uris: tuple[str, ...] = ()
    log_tail: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def _classify_error(returncode: int, stderr: str) -> type[ServerlessClientError]:
    """Map subprocess error output to a typed exception class."""
    lower = stderr.lower()

    if any(pattern in lower for pattern in _AUTH_PATTERNS):
        return AuthError
    if "quota" in lower and any(
        marker in lower for marker in ("exceeded", "limit", "reached")
    ):
        return QuotaError
    if any(pattern in lower for pattern in _NER_PATTERNS):
        return NotEnoughResourcesError
    if any(pattern in lower for pattern in _NOT_FOUND_PATTERNS):
        return EndpointNotFoundError
    return ServerlessClientError


def _json_loads(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_exc:
        decoder = json.JSONDecoder()
        parsed: list[Any] = []
        last_exc: json.JSONDecodeError = first_exc
        for idx, char in enumerate(stripped):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[idx:])
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
            parsed.append(value)
        if parsed:
            return parsed[-1]
        raise last_exc


def _as_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "endpoints", "resources"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if data:
            return [data]
    return []


def _deep_get(data: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = data
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if current not in (None, ""):
            return current
    return ""


def _endpoint_name(data: Mapping[str, Any]) -> str:
    value = _deep_get(
        data,
        ("metadata", "name"),
        ("spec", "name"),
        ("name",),
    )
    return str(value or "")


def _endpoint_id(data: Mapping[str, Any]) -> str:
    value = _deep_get(
        data,
        ("metadata", "id"),
        ("resource_id",),
        ("id",),
    )
    return str(value or "")


def _endpoint_project_id(data: Mapping[str, Any], fallback: str = "") -> str:
    value = _deep_get(
        data,
        ("metadata", "parent_id"),
        ("metadata", "parentId"),
        ("parent_id",),
        ("parentId",),
        ("project_id",),
    )
    return str(value or fallback)


def _endpoint_status(data: Mapping[str, Any]) -> EndpointStatus:
    value = _deep_get(
        data,
        ("status", "state"),
        ("status", "status"),
        ("status",),
        ("state",),
        ("metadata", "status"),
    )
    return EndpointStatus.from_value(value)


def _endpoint_url(data: Mapping[str, Any]) -> str:
    def normalize_url(raw: Any) -> str:
        value = str(raw or "").strip()
        if not value:
            return ""
        if urlparse(value).scheme:
            return value
        return f"http://{value}"

    value = _deep_get(
        data,
        ("status", "url"),
        ("status", "endpoint_url"),
        ("status", "public_url"),
        ("endpoint_url",),
        ("public_url",),
        ("url",),
    )
    if value:
        return normalize_url(value)

    status = data.get("status")
    if isinstance(status, Mapping):
        for key in ("public_endpoints", "publicEndpoints"):
            public_endpoints = status.get(key)
            if isinstance(public_endpoints, list) and public_endpoints:
                return normalize_url(public_endpoints[0])
        endpoints = status.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            first = endpoints[0]
            if isinstance(first, Mapping):
                return normalize_url(first.get("url") or first.get("address") or "")
            return normalize_url(first)
    return ""


def _is_secret_env_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _is_sensitive_log_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("token", "key", "secret", "password", "passwd"))


def _redact_env_arg(value: str) -> str:
    key, sep, _raw_value = value.partition("=")
    if sep and _is_sensitive_log_key(key):
        return f"{key}=<redacted>"
    return value


def _redact_cli_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    redact_env_next = False
    for arg in args:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if redact_env_next:
            redacted.append(_redact_env_arg(arg))
            redact_env_next = False
            continue
        if arg.startswith("--env="):
            redacted.append("--env=" + _redact_env_arg(arg.removeprefix("--env=")))
            continue
        redacted.append(arg)
        if arg == "--env":
            redact_env_next = True
        elif arg in _SENSITIVE_VALUE_FLAGS:
            redact_next = True
    return redacted


class ServerlessClient:
    """Subprocess wrapper for ``nebius ai endpoint`` commands."""

    def __init__(
        self,
        *,
        nebius_bin: str | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        timeout: int = 900,
        poll_interval: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._nebius_bin = nebius_bin or shutil.which("nebius") or "nebius"
        self._runner = subprocess_runner or subprocess.run
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._sleep = sleep

    def create_endpoint(
        self,
        spec: EndpointSpec,
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> EndpointInfo:
        """Create an endpoint."""
        args = self._build_create_args(spec, extra_env=extra_env)
        result = self._run(args, timeout=self._timeout)
        if result.returncode != 0:
            self._raise_for_error(result, f"create_endpoint failed for {spec.name} in project {spec.project_id}")
        try:
            info = self._parse_endpoint_info(result.stdout, project_id=spec.project_id)
        except json.JSONDecodeError:
            return self.get_endpoint(spec.project_id, spec.name)
        if info.name:
            return info
        endpoint_ref = info.id or spec.name
        return self.get_endpoint(spec.project_id, endpoint_ref)

    def list_endpoints(self, project_id: str) -> list[EndpointInfo]:
        """List endpoints in a project."""
        result = self._run([
            "ai",
            "endpoint",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ])
        if result.returncode != 0:
            self._raise_for_error(result, f"list_endpoints failed for project {project_id}")
        data = _json_loads(result.stdout)
        return [
            self._info_from_dict(item, fallback_project_id=project_id)
            for item in _as_items(data)
        ]

    def get_endpoint(self, project_id: str, endpoint: str) -> EndpointInfo:
        """Return an endpoint by name or ID."""
        for info in self.list_endpoints(project_id):
            if endpoint in {info.id, info.name}:
                return info

        result = self._run([
            "ai",
            "endpoint",
            "get",
            "--id",
            endpoint,
            "--format",
            "json",
        ])
        if result.returncode != 0:
            self._raise_for_error(result, f"get_endpoint failed for {endpoint}")
        info = self._parse_endpoint_info(result.stdout, project_id=project_id)
        if info.id or info.name:
            return info
        raise EndpointNotFoundError(f"Endpoint {endpoint} not found in project {project_id}")

    def delete_endpoint(self, project_id: str, endpoint: str) -> None:
        """Delete an endpoint by name or ID. Missing endpoints are treated as deleted."""
        try:
            info = self.get_endpoint(project_id, endpoint)
        except EndpointNotFoundError:
            return
        result = self._run([
            "ai",
            "endpoint",
            "delete",
            "--id",
            info.id,
        ])
        if result.returncode != 0:
            error_class = _classify_error(result.returncode, result.stderr)
            if error_class is EndpointNotFoundError:
                return
            self._raise_for_error(result, f"delete_endpoint failed for {endpoint}")

    def stop_endpoint(self, project_id: str, endpoint: str) -> EndpointInfo:
        """Stop an endpoint by name or ID."""
        info = self.get_endpoint(project_id, endpoint)
        result = self._run([
            "ai",
            "endpoint",
            "stop",
            "--id",
            info.id,
            "--format",
            "json",
        ])
        if result.returncode != 0:
            self._raise_for_error(result, f"stop_endpoint failed for {endpoint}")
        return self._parse_endpoint_info(result.stdout, project_id=project_id)

    def start_endpoint(self, project_id: str, endpoint: str) -> EndpointInfo:
        """Start an endpoint by name or ID."""
        info = self.get_endpoint(project_id, endpoint)
        result = self._run([
            "ai",
            "endpoint",
            "start",
            "--id",
            info.id,
            "--format",
            "json",
        ])
        if result.returncode != 0:
            self._raise_for_error(result, f"start_endpoint failed for {endpoint}")
        return self._parse_endpoint_info(result.stdout, project_id=project_id)

    def get_endpoint_logs(
        self,
        project_id: str,
        endpoint: str,
        *,
        tail: int | None = None,
        since: str = "",
    ) -> str:
        """Return endpoint logs as text."""
        info = self.get_endpoint(project_id, endpoint)
        args = ["ai", "endpoint", "logs", info.id]
        if tail is not None:
            args.extend(["--tail", str(tail)])
        if since:
            args.extend(["--since", since])
        result = self._run(args)
        if result.returncode != 0:
            self._raise_for_error(result, f"get_endpoint_logs failed for {endpoint}")
        return result.stdout

    def wait_for_running(
        self,
        project_id: str,
        endpoint: str,
        *,
        timeout: int = 600,
        poll_interval: float | None = None,
    ) -> EndpointInfo:
        """Poll until the endpoint reaches RUNNING."""
        deadline = time.monotonic() + timeout
        interval = self._poll_interval if poll_interval is None else poll_interval
        last: EndpointInfo | None = None
        while time.monotonic() <= deadline:
            last = self.get_endpoint(project_id, endpoint)
            if last.status is EndpointStatus.RUNNING:
                return last
            if last.status in {EndpointStatus.FAILED, EndpointStatus.DELETED}:
                raise ServerlessClientError(
                    f"Endpoint {endpoint} reached terminal status {last.status.value}"
                )
            self._sleep(interval)
        status = last.status.value if last else "unknown"
        raise TimeoutError(
            f"Endpoint {endpoint} did not reach running within {timeout}s (last status: {status})"
        )

    def create_job(
        self, *, project_id: str, name: str, image: str, command: str, gpu_type: str,
        gpu_count: int, output_path: str, extra_env: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None, preset: str = "", timeout: str = "1h",
        subnet_id: str = "",
    ) -> JobInfo:
        for label, value in {
            "Job name": name,
            "Project ID": project_id,
            "Container image": image,
            "Job command": command,
        }.items():
            if not value:
                raise ValueError(f"{label} is required")
        if gpu_count < 1:
            raise ValueError("GPU count must be positive")
        args = ["ai", "job", "create", "--parent-id", project_id, "--name", name]
        args += ["--image", image, "--container-command", command, "--platform", gpu_type]
        args += ["--preset", preset or f"{gpu_count}gpu-16vcpu-200gb", "--env", f"NPA_OUTPUT_PATH={output_path}"]
        for key, value in (env or {}).items():
            if _is_secret_env_key(key):
                raise ValueError(f"Refusing to pass secret-like env var {key} on the command line")
            if not extra_env or key not in extra_env:
                args.extend(["--env", f"{key}={value}"])
        for key, value in (extra_env or {}).items():
            if value:
                args.extend(["--env", f"{key}={value}"])
        for flag, value in (("--timeout", timeout), ("--subnet-id", subnet_id)):
            if value:
                args.extend([flag, value])
        args.extend(["--format", "json"])
        result = self._run(args, timeout=_JOB_CREATE_TIMEOUT)
        if result.returncode != 0:
            self._raise_for_error(result, f"create_job failed for {name} in project {project_id}")
        try:
            info = self._parse_job_info(result.stdout, project_id=project_id)
        except json.JSONDecodeError:
            return self.get_job(name, project_id)
        return info if info.name else self.get_job(info.id or name, project_id)

    def list_jobs(self, project_id: str, name_prefix: str | None = None) -> list[JobInfo]:
        result = self._run(["ai", "job", "list", "--parent-id", project_id, "--format", "json"], timeout=_JOB_QUERY_TIMEOUT)
        if result.returncode != 0:
            self._raise_for_error(result, f"list_jobs failed for project {project_id}")
        jobs = [
            self._job_info_from_dict(item, fallback_project_id=project_id)
            for item in _as_items(_json_loads(result.stdout))
        ]
        return [job for job in jobs if not name_prefix or job.name.startswith(name_prefix)]

    def get_job(self, job_id_or_name: str, project_id: str) -> JobInfo:
        commands = (
            ["ai", "job", "get", "--id", job_id_or_name, "--format", "json"],
            ["ai", "job", "get-by-name", "--parent-id", project_id, "--name", job_id_or_name, "--format", "json"],
        )
        for index, args in enumerate(commands):
            result = self._run(args, timeout=_JOB_QUERY_TIMEOUT)
            if result.returncode == 0:
                info = self._parse_job_info(result.stdout, project_id=project_id)
                if info.id or info.name:
                    return info
            elif index == 0:
                continue
            elif _classify_error(result.returncode, result.stderr) is not EndpointNotFoundError:
                self._raise_for_error(result, f"get_job failed for {job_id_or_name}")
        raise EndpointNotFoundError(f"Job {job_id_or_name} not found in project {project_id}")

    def cancel_job(self, job_id_or_name: str, project_id: str) -> JobInfo:
        info = self.get_job(job_id_or_name, project_id)
        if info.status in _JOB_TERMINAL_STATUSES:
            return info
        result = self._run(["ai", "job", "cancel", "--id", info.id, "--format", "json"], timeout=_JOB_CANCEL_TIMEOUT)
        if result.returncode != 0:
            error_class = _classify_error(result.returncode, result.stderr)
            if error_class is EndpointNotFoundError:
                return info
            self._raise_for_error(result, f"cancel_job failed for {job_id_or_name}")
        try:
            parsed = self._parse_job_info(result.stdout, project_id=project_id)
        except json.JSONDecodeError:
            parsed = JobInfo(id="", name="", project_id=project_id)
        return parsed if parsed.id or parsed.name else self.get_job(info.id, project_id)

    def poll_job(
        self, job_id: str, project_id: str, *, interval_s: float = 30.0,
        ceiling_s: float = 2400.0, on_state_change: Callable[[JobInfo], None] | None = None,
    ) -> JobInfo:
        deadline = time.monotonic() + ceiling_s
        last: JobInfo | None = None
        last_status: str | None = None
        transient_failures = 0
        try:
            while time.monotonic() <= deadline:
                try:
                    current = self.get_job(job_id, project_id)
                except ServerlessClientError:
                    transient_failures += 1
                    if transient_failures > 1:
                        raise
                    self._sleep(interval_s)
                    continue
                transient_failures = 0
                last = current
                if current.status is not last_status and on_state_change is not None:
                    on_state_change(current)
                last_status = current.status
                if current.status in _JOB_TERMINAL_STATUSES:
                    return current
                self._sleep(interval_s)
        except KeyboardInterrupt:
            try:
                self.cancel_job(job_id, project_id)
            except ServerlessClientError as exc:
                logger.warning("Job cancellation after interrupt failed for %s: %s", job_id, exc)
            raise
        status = last.status if last else "unknown"
        raise TimeoutError(f"Job {job_id} did not finish within {ceiling_s}s (last status: {status})")

    def _run(
        self,
        args: list[str],
        *,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        full_args = [self._nebius_bin, *args]
        logger.debug("Running Nebius CLI: %s", shlex.join(_redact_cli_args(full_args)))
        effective_timeout = timeout or self._timeout
        try:
            return self._runner(
                full_args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=dict(env) if env is not None else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServerlessClientError(
                f"Nebius CLI timed out after {effective_timeout}s: {shlex.join(_redact_cli_args(full_args))}"
            ) from exc

    def _build_create_args(
        self,
        spec: EndpointSpec,
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> list[str]:
        if not spec.name:
            raise ValueError("Endpoint name is required")
        if not spec.project_id:
            raise ValueError("Project ID is required")
        if not spec.image:
            raise ValueError("Container image is required")

        args = [
            "ai",
            "endpoint",
            "create",
            "--parent-id",
            spec.project_id,
            "--name",
            spec.name,
            "--image",
            spec.image,
            "--auth",
            spec.auth,
        ]
        if spec.platform:
            args.extend(["--platform", spec.platform])
        if spec.preset:
            args.extend(["--preset", spec.preset])
        if spec.public:
            args.append("--public")
        for port in spec.container_ports:
            args.extend(["--container-port", str(port)])
        for key, value in spec.env.items():
            if _is_secret_env_key(key):
                raise ValueError(
                    f"Refusing to pass secret-like env var {key} on the command line"
                )
            if extra_env and key in extra_env:
                continue
            args.extend(["--env", f"{key}={value}"])
        for key, value in (extra_env or {}).items():
            if value:
                args.extend(["--env", f"{key}={value}"])
        for volume in spec.volumes:
            args.extend(["--volume", volume])
        if spec.args:
            args.extend(["--args", spec.args])
        if spec.container_command:
            args.extend(["--container-command", spec.container_command])
        if spec.disk_size:
            args.extend(["--disk-size", spec.disk_size])
        if spec.shm_size:
            args.extend(["--shm-size", spec.shm_size])
        if spec.subnet_id:
            args.extend(["--subnet-id", spec.subnet_id])
        if spec.working_dir:
            args.extend(["--working-dir", spec.working_dir])
        if spec.preemptible:
            args.append("--preemptible")
        args.extend(["--format", "json"])
        return args

    def _parse_endpoint_info(self, raw: str, *, project_id: str = "") -> EndpointInfo:
        data = _json_loads(raw)
        items = _as_items(data)
        if not items:
            return EndpointInfo(id="", name="", project_id=project_id, raw={})
        return self._info_from_dict(items[0], fallback_project_id=project_id)

    def _info_from_dict(
        self,
        data: dict[str, Any],
        *,
        fallback_project_id: str = "",
    ) -> EndpointInfo:
        return EndpointInfo(
            id=_endpoint_id(data),
            name=_endpoint_name(data),
            project_id=_endpoint_project_id(data, fallback_project_id),
            status=_endpoint_status(data),
            url=_endpoint_url(data),
            raw=data,
        )

    def _parse_job_info(self, raw: str, *, project_id: str = "") -> JobInfo:
        items = _as_items(_json_loads(raw))
        if not items:
            return JobInfo(id="", name="", project_id=project_id, raw={})
        return self._job_info_from_dict(items[0], fallback_project_id=project_id)

    def _job_info_from_dict(self, data: dict[str, Any], *, fallback_project_id: str = "") -> JobInfo:
        outputs = _deep_get(data, ("status", "output_uris"), ("status", "outputs"), ("output_uris",), ("output_path",), ("spec", "output_path"))
        if isinstance(outputs, str):
            output_uris = (outputs,) if outputs else ()
        elif isinstance(outputs, list):
            output_uris = tuple(str(value) for value in outputs if value)
        else:
            output_uris = ()
        return JobInfo(
            id=_endpoint_id(data),
            name=_endpoint_name(data),
            project_id=_endpoint_project_id(data, fallback_project_id),
            status=_job_status(_deep_get(data, ("status", "state"), ("status",), ("state",))),
            created_at=str(_deep_get(data, ("metadata", "created_at"), ("created_at",))),
            started_at=str(_deep_get(data, ("status", "started_at"), ("started_at",))),
            ended_at=str(_deep_get(data, ("status", "ended_at"), ("ended_at",))),
            output_uris=output_uris,
            log_tail=str(_deep_get(data, ("status", "message"), ("status", "log_tail"), ("log_tail",))),
            raw=data,
        )

    def _raise_for_error(
        self,
        result: subprocess.CompletedProcess[str],
        prefix: str,
    ) -> None:
        error_class = _classify_error(result.returncode, result.stderr)
        stderr = result.stderr.strip()
        raise error_class(f"{prefix}: {stderr}")

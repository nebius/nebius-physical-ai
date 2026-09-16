"""Structured subprocess client for the supported Antioch CLI surface.

This module deliberately does not call undocumented HTTP APIs.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from .redaction import redact_payload, redact_text


class AntiochCliError(RuntimeError):
    """A typed, redacted failure returned by the vendor CLI."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "antioch_cli_error",
        retryable: bool = False,
        http_status: int = 0,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.http_status = http_status
        self.exit_code = exit_code


@dataclass(frozen=True)
class CommandResult:
    payload: Any
    stderr: str = ""


def _error_from_process(result: subprocess.CompletedProcess[str]) -> AntiochCliError:
    raw = (result.stderr or result.stdout or "Antioch CLI command failed").strip()
    candidate: dict[str, Any] = {}
    for line in reversed(raw.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            candidate = parsed["error"]
            break
    status = int(candidate.get("http_status") or 0)
    retryable = bool(candidate.get("retryable")) or status == 429 or status >= 500
    return AntiochCliError(
        redact_text(str(candidate.get("message") or raw)),
        error_type=str(candidate.get("type") or "antioch_cli_error"),
        retryable=retryable,
        http_status=status,
        exit_code=int(candidate.get("exit_code") or result.returncode or 1),
    )


class AntiochCli:
    """Invoke one pinned Antioch executable and parse only its JSON contracts."""

    def __init__(self, executable: str | Path, *, config_dir: str = "") -> None:
        self.executable = str(executable)
        self.config_dir = config_dir.strip()

    def _run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        expect_json: bool = True,
        timeout_seconds: float | None = 60,
    ) -> CommandResult:
        env = dict(os.environ)
        env["NO_COLOR"] = "1"
        if self.config_dir:
            env["ANTIOCH_CONFIG_DIR"] = self.config_dir
        try:
            result = subprocess.run(
                [self.executable, *args],
                cwd=str(cwd) if cwd else None,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise AntiochCliError(
                "Antioch CLI command timed out",
                error_type="cli_timeout",
                retryable=True,
            ) from exc
        except FileNotFoundError as exc:
            raise AntiochCliError(
                "Antioch CLI is not installed in the configured runtime cache",
                error_type="cli_not_installed",
            ) from exc
        if result.returncode:
            raise _error_from_process(result)
        if not expect_json:
            return CommandResult(
                payload=(result.stdout or "").strip(),
                stderr=redact_text(result.stderr or ""),
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AntiochCliError(
                "Antioch CLI returned malformed structured output",
                error_type="malformed_cli_output",
                retryable=False,
            ) from exc
        return CommandResult(payload=payload, stderr=redact_text(result.stderr or ""))

    def version(self) -> str:
        raw = str(self._run(["--version"], expect_json=False).payload)
        return raw.rsplit(" ", 1)[-1].strip()

    def health(self) -> dict[str, Any]:
        version = self.version()
        identity = self._run(["auth", "whoami", "--json"]).payload
        if not isinstance(identity, dict):
            raise AntiochCliError(
                "Antioch identity response was not an object",
                error_type="malformed_cli_output",
            )
        projects = self._run(["project", "list", "--json"]).payload
        if not isinstance(projects, list):
            raise AntiochCliError(
                "Antioch project registry response was not an array",
                error_type="malformed_cli_output",
            )
        return {
            "authenticated": True,
            "cli_version": version,
        }

    def submit_suite(self, cwd: Path, suite: str) -> dict[str, Any]:
        payload = self._run(
            ["suite", "run", suite, "--detach", "--json"],
            cwd=cwd,
            timeout_seconds=None,
        ).payload
        if (
            not isinstance(payload, list)
            or not payload
            or not all(isinstance(item, dict) for item in payload)
        ):
            raise AntiochCliError(
                "detached suite response was not a non-empty array",
                error_type="malformed_cli_output",
            )
        suite_ids = {str(item.get("suite_run_id") or "") for item in payload}
        invocation_ids = {str(item.get("invocation_id") or "") for item in payload}
        if len(suite_ids) != 1 or "" in suite_ids or len(invocation_ids) != 1:
            raise AntiochCliError(
                "detached suite response did not identify one invocation",
                error_type="malformed_cli_output",
            )
        return {
            "suite_run_id": suite_ids.pop(),
            "invocation_id": invocation_ids.pop(),
            "suite": suite,
            "phase": "prepared",
        }

    def submit_scenario(
        self,
        cwd: Path,
        scenario: str,
        *,
        scenario_case: str = "",
        parameters: dict[str, str | int | float | bool] | None = None,
    ) -> dict[str, Any]:
        args = ["scenario", "run", "--scenario", scenario]
        if scenario_case:
            args.extend(["--case", scenario_case])
        for key, value in sorted((parameters or {}).items()):
            args.extend(["--set", f"{key}={json.dumps(value, separators=(',', ':'))}"])
        args.extend(["--detach", "--json"])
        payload = self._run(args, cwd=cwd, timeout_seconds=None).payload
        if (
            not isinstance(payload, list)
            or not payload
            or not isinstance(payload[0], dict)
        ):
            raise AntiochCliError(
                "detached scenario response was not a non-empty array",
                error_type="malformed_cli_output",
            )
        return payload[0]

    def list_for_project(
        self, cwd: Path, *, kind: str, project_id: str
    ) -> list[dict[str, Any]]:
        if kind not in {"scenario", "suite"}:
            raise AntiochCliError(
                "Antioch run kind must be scenario or suite",
                error_type="invalid_request",
            )
        manifest_path = cwd / "antioch.yaml"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise AntiochCliError(
                "Antioch project manifest is unavailable",
                error_type="project_identity_unavailable",
            )
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise AntiochCliError(
                "Antioch project manifest could not be read",
                error_type="project_identity_unavailable",
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("id") != project_id:
            raise AntiochCliError(
                "Antioch current project does not match the required identity",
                error_type="project_identity_mismatch",
            )
        args = [
            kind,
            "list",
            "--mine",
            "--limit",
            "200",
            "--json",
        ]
        payload = self._run(args, cwd=cwd).payload
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise AntiochCliError(
                "Antioch list response was malformed", error_type="malformed_cli_output"
            )
        return [item for item in payload["items"] if isinstance(item, dict)]

    def show(self, cwd: Path, *, kind: str, remote_id: str) -> dict[str, Any]:
        payload = self._run([kind, "show", remote_id, "--json"], cwd=cwd).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch status response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def cancel(self, cwd: Path, *, kind: str, remote_id: str) -> dict[str, Any]:
        payload = self._run([kind, "cancel", remote_id, "--json"], cwd=cwd).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch cancellation response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def rerun(self, cwd: Path, *, kind: str, remote_id: str) -> dict[str, Any]:
        payload = self._run([kind, "rerun", remote_id, "--json"], cwd=cwd).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch rerun response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def download(
        self, cwd: Path, *, scenario_run_id: str, output: Path
    ) -> dict[str, Any]:
        payload = self._run(
            [
                "scenario",
                "download",
                scenario_run_id,
                "--output",
                str(output),
                "--json",
            ],
            cwd=cwd,
        ).payload
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            raise AntiochCliError(
                "Antioch transfer manifest was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def logs(self, cwd: Path, *, scenario_run_id: str) -> dict[str, Any]:
        payload = self._run(
            ["scenario", "logs", scenario_run_id, "--json"], cwd=cwd
        ).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch log response was malformed", error_type="malformed_cli_output"
            )
        return redact_payload(payload)

    def project_build(self, cwd: Path, *, service: str = "sim") -> dict[str, Any]:
        if not service:
            raise AntiochCliError(
                "Antioch project build requires an exact service",
                error_type="invalid_request",
            )
        payload = self._run(
            ["project", "build", service, "--json"],
            cwd=cwd,
            timeout_seconds=None,
        ).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch project build response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def session_new(
        self, cwd: Path, *, revision: str = "", force: bool = False
    ) -> dict[str, Any]:
        args = ["session", "new"]
        if revision:
            args.extend(["--revision", revision])
        if force:
            args.append("--force")
        args.append("--json")
        payload = self._run(
            args,
            cwd=cwd,
            timeout_seconds=None,
        ).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch session startup response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def session_status(self, cwd: Path) -> dict[str, Any]:
        payload = self._run(["session", "status", "--json"], cwd=cwd).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch session status response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def session_release(self, cwd: Path, *, force: bool = False) -> dict[str, Any]:
        args = ["session", "release"]
        if force:
            args.append("--force")
        args.extend(["--yes", "--json"])
        payload = self._run(args, cwd=cwd, timeout_seconds=None).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch session release response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def service_exec(self, cwd: Path, service: str, command: Sequence[str]) -> str:
        if not service or not command:
            raise AntiochCliError(
                "Antioch service exec requires a service and command",
                error_type="invalid_request",
            )
        return str(
            self._run(
                [
                    "service",
                    "exec",
                    "--no-stream",
                    "--no-tty",
                    "--service",
                    service,
                    "--",
                    *command,
                ],
                cwd=cwd,
                expect_json=False,
                timeout_seconds=None,
            ).payload
        )

    def service_copy(self, cwd: Path, source: Path, destination: str) -> dict[str, Any]:
        payload = self._run(
            ["service", "cp", str(source), destination, "--json"],
            cwd=cwd,
            timeout_seconds=None,
        ).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch service copy response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def service_ps(self, cwd: Path, *, service: str = "") -> dict[str, Any]:
        args = ["service", "ps"]
        if service:
            args.append(service)
        args.append("--json")
        payload = self._run(args, cwd=cwd).payload
        if not isinstance(payload, dict):
            raise AntiochCliError(
                "Antioch service status response was malformed",
                error_type="malformed_cli_output",
            )
        return payload

    def service_ports_args(
        self, *, service: str, route: str, host: str, port: int
    ) -> list[str]:
        """Return the supported foreground route command for direct ownership."""

        if not service or not route or host not in {"127.0.0.1", "::1"}:
            raise AntiochCliError(
                "Antioch service route requires exact names and a loopback host",
                error_type="invalid_request",
            )
        if not 1 <= port <= 65_535:
            raise AntiochCliError(
                "Antioch service route port is outside 1..65535",
                error_type="invalid_request",
            )
        return [
            "service",
            "ports",
            "--bind",
            f"{service}.{route}={host}:{port}",
            "--serve",
            service,
        ]


def remote_id(payload: dict[str, Any], *, kind: str) -> str:
    keys = (
        ("suite_run_id", "id", "run_id")
        if kind == "suite"
        else ("scenario_run_id", "id", "run_id")
    )
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    raise AntiochCliError(
        "Antioch response did not contain a remote run id",
        error_type="malformed_cli_output",
    )


def invocation_id(payload: dict[str, Any]) -> str:
    return str(payload.get("invocation_id") or "").strip()


def public_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the status subset safe for manifests and ordinary logs."""

    allowed = {
        "id",
        "run_id",
        "suite_run_id",
        "scenario_run_id",
        "invocation_id",
        "scenario",
        "suite",
        "case",
        "phase",
        "outcome",
        "params",
        "results",
        "checks",
        "artifacts",
        "created_at",
        "started_at",
        "completed_at",
        "engine_version",
        "sdk_version",
        "scenario_runs",
    }
    return redact_payload(
        {key: value for key, value in payload.items() if key in allowed}
    )

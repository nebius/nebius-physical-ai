"""Run a real OpenPI pi0.5 policy container against an Antioch scenario.

This operator-side harness keeps the runtimes separate: OpenPI runs in a local
GPU container while Antioch dispatches the simulator. Only the upstream
websocket protocol crosses that boundary. The OpenPI terms decision is
inherited from the environment and is never accepted by a CLI flag, written to
disk, or included in the result artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from npa.workflows.byof.openpi import OPENPI_TERMS_ENV, require_openpi_terms
from npa.workflows.byof.openpi_pipeline import SOURCE_REF

EVIDENCE_SCHEMA = "npa.workbench.openpi.antioch-loop.v1"
MANAGED_CONTAINER_LABEL = "npa.openpi-antioch.managed=true"
MANAGED_CONTAINER_OWNER_LABEL = "npa.openpi-antioch.owner"
DEFAULT_CONTAINER_NAME = "npa-openpi-antioch-pi05"
REQUIRED_CHECKS = {
    "the jaw travels its stroke",
    "the server returns the documented chunk shape",
    "actions are finite",
    "the arm moved in response to the policy",
    "nothing diverged",
}

OPENPI_DOCKERFILE = """\
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates git git-lfs python3 python3-dev python3-pip python3-venv && \\
    rm -rf /var/lib/apt/lists/*
WORKDIR /opt/openpi
COPY . .
RUN python3 -m venv /opt/venv && \\
    /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip uv && \\
    GIT_LFS_SKIP_SMUDGE=1 /opt/venv/bin/uv pip install \\
      --python /opt/venv/bin/python --no-cache -e .
RUN printf '%s\\n' \\
    '#!/bin/sh' \\
    'if [ "${NPA_OPENPI_ACCEPT_GEMMA_TERMS:-}" != "YES" ]; then' \\
    '  echo NPA_OPENPI_TERMS_REFUSED >&2' \\
    '  exit 64' \\
    'fi' \\
    'exec /opt/venv/bin/python scripts/serve_policy.py "$@"' \\
    > /usr/local/bin/npa-openpi-entrypoint && chmod 0755 /usr/local/bin/npa-openpi-entrypoint
ENTRYPOINT ["/usr/local/bin/npa-openpi-entrypoint"]
"""


class OpenPIAntiochError(RuntimeError):
    """Raised when the connected OpenPI/Antioch validation is not proven."""


@dataclass(frozen=True)
class _ContainerIdentity:
    """Exact identity of one labelled local OpenPI policy container."""

    name: str
    container_id: str
    image: str
    managed: bool
    owner: str
    running: bool


@dataclass
class _LiveResources:
    """Track the exact resources that one live-loop invocation may clean up."""

    container: _ContainerIdentity | None = None
    scenario_run_id: str | None = None
    image_id: str | None = None
    evidence: dict[str, object] | None = None
    scenario_cleanup: str = "not_requested"
    scenario_phase: str | None = None
    scenario_outcome: str | None = None
    container_cleanup: str = "not_requested"
    container_absent: bool | None = None


@dataclass(frozen=True)
class LiveLoopConfig:
    """Operator-local inputs for one connected validation run."""

    project_dir: Path
    cache_dir: Path
    image: str
    policy_host: str
    host_port: int = 8000
    policy_port: int | None = None
    scenario: str = "pi05_droid_loop"
    chunks: int = 3
    policy_ready_timeout_s: float = 300.0
    scenario_timeout_s: float = 1800.0
    docker_bin: str = "docker"
    antioch_bin: str = "antioch"
    rerun_from: str | None = None
    machine: str | None = None
    script: str | None = None
    container_name: str = DEFAULT_CONTAINER_NAME
    cleanup_container: bool = False
    cleanup_scenario: bool = False
    resource_owner: str = ""
    private_receipt_path: Path | None = None


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2000:].strip()
        raise OpenPIAntiochError(
            f"{Path(argv[0]).name} exited {completed.returncode}: {detail}"
        )
    return completed


def _json_value(text: str, *, label: str) -> object:
    """Parse a complete JSON value or the final structured CLI line."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        for line in reversed(text.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        else:
            raise OpenPIAntiochError(f"{label} did not emit valid JSON") from exc
    return value


def _json_object(text: str, *, label: str) -> dict[str, Any]:
    """Parse one structured CLI object."""

    value = _json_value(text, label=label)
    if not isinstance(value, dict):
        raise OpenPIAntiochError(f"{label} JSON must be an object")
    return value


def build_local_image(
    *, openpi_dir: Path, image: str, docker_bin: str = "docker"
) -> dict[str, object]:
    """Build a local-only OpenPI image from the repository-pinned source."""

    require_openpi_terms()
    source_ref = _run(["git", "rev-parse", "HEAD"], cwd=openpi_dir).stdout.strip()
    if source_ref != SOURCE_REF:
        raise OpenPIAntiochError(
            f"OpenPI checkout must be pinned to {SOURCE_REF}, got {source_ref}"
        )
    source_status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=openpi_dir,
    ).stdout.strip()
    if source_status:
        raise OpenPIAntiochError("OpenPI checkout must be clean before image build")
    _run(
        [
            docker_bin,
            "build",
            "--label",
            "npa.validation=openpi-antioch",
            "--tag",
            image,
            "-f",
            "-",
            str(openpi_dir),
        ],
        input_text=OPENPI_DOCKERFILE,
    )
    inspected = _run(
        [docker_bin, "image", "inspect", image, "--format", "{{.Id}}"]
    ).stdout.strip()
    if not inspected.startswith("sha256:"):
        raise OpenPIAntiochError("built OpenPI image has no content-addressed ID")
    return {
        "source_ref": source_ref,
        "image_id": inspected,
        "redistribution": "local_only_not_published",
        "weights": "runtime_mount_not_baked",
    }


def _negative_terms_probe(config: LiveLoopConfig) -> None:
    """Prove the image entrypoint rejects a child without accepted terms."""

    child_env = dict(os.environ)
    child_env.pop(OPENPI_TERMS_ENV, None)
    completed = _run(
        [
            config.docker_bin,
            "run",
            "--rm",
            config.image,
        ],
        env=child_env,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 64 or "NPA_OPENPI_TERMS_REFUSED" not in output:
        raise OpenPIAntiochError(
            "OpenPI image entrypoint did not fail closed before model startup"
        )


def _accepted_container_argv(
    config: LiveLoopConfig, *, container_name: str
) -> list[str]:
    argv = [
        config.docker_bin,
        "run",
        "--detach",
        "--name",
        container_name,
        "--restart",
        "unless-stopped",
        "--label",
        MANAGED_CONTAINER_LABEL,
    ]
    if config.resource_owner:
        argv.extend(
            ["--label", f"{MANAGED_CONTAINER_OWNER_LABEL}={config.resource_owner}"]
        )
    return [
        *argv,
        "--gpus",
        "all",
        "--publish",
        f"{config.host_port}:8000",
        "--env",
        OPENPI_TERMS_ENV,
        "--mount",
        f"type=bind,src={config.cache_dir},dst=/root/.cache/openpi,readonly",
        "--health-cmd",
        (
            '/opt/venv/bin/python -c "import socket; '
            "s=socket.create_connection(('127.0.0.1',8000),2); s.close()\""
        ),
        "--health-interval",
        "10s",
        "--health-timeout",
        "3s",
        "--health-retries",
        "12",
        config.image,
        "--env",
        "DROID",
        "--port",
        "8000",
    ]


def _inspect_policy_container(
    config: LiveLoopConfig, container: str
) -> _ContainerIdentity | None:
    """Return the exact local identity for a container, when it still exists."""

    inspected = _run(
        [
            config.docker_bin,
            "inspect",
            container,
            "--format",
            (
                '{{index .Config.Labels "npa.openpi-antioch.managed"}}\t'
                '{{index .Config.Labels "npa.openpi-antioch.owner"}}\t'
                "{{.Config.Image}}\t{{.Id}}\t{{.State.Running}}"
            ),
        ],
        check=False,
    )
    if inspected.returncode != 0:
        return None
    fields = inspected.stdout.rstrip("\n").split("\t")
    if len(fields) != 5:
        raise OpenPIAntiochError("managed container inspection was malformed")
    managed, owner, image, container_id, running = fields
    if not container_id:
        raise OpenPIAntiochError("managed container inspection omitted its identity")
    return _ContainerIdentity(
        name=container,
        container_id=container_id,
        image=image,
        managed=managed == "true",
        owner=owner,
        running=running == "true",
    )


def _matches_requested_container(
    config: LiveLoopConfig, container: _ContainerIdentity
) -> bool:
    """Return whether a local container belongs to this exact request."""

    owner_matches = (
        not config.resource_owner or container.owner == config.resource_owner
    )
    return container.managed and container.image == config.image and owner_matches


def _ensure_policy_container(config: LiveLoopConfig) -> _ContainerIdentity:
    """Start or safely reuse the exact task-managed policy container.

    A name collision is accepted only when the management label, requested image,
    and optional task owner match. The returned ID is later used for cleanup.
    """

    existing = _inspect_policy_container(config, config.container_name)
    if existing is not None:
        if not _matches_requested_container(config, existing):
            raise OpenPIAntiochError(
                f"container name {config.container_name!r} is not the requested "
                "task-managed OpenPI container"
            )
        if not existing.running:
            _run([config.docker_bin, "start", config.container_name])
            existing = _inspect_policy_container(config, existing.container_id)
            if existing is None or not existing.running:
                raise OpenPIAntiochError("managed OpenPI container did not start")
        return _ContainerIdentity(
            name=config.container_name,
            container_id=existing.container_id,
            image=existing.image,
            managed=existing.managed,
            owner=existing.owner,
            running=existing.running,
        )
    _run(_accepted_container_argv(config, container_name=config.container_name))
    created = _inspect_policy_container(config, config.container_name)
    if created is None or not _matches_requested_container(config, created):
        raise OpenPIAntiochError(
            "created OpenPI container lacks the requested identity"
        )
    return created


def _remove_policy_container(
    config: LiveLoopConfig, container: _ContainerIdentity
) -> None:
    """Remove and verify absence of only the exact task-owned container."""

    current = _inspect_policy_container(config, container.container_id)
    if current is None:
        return
    if not _matches_requested_container(config, current):
        raise OpenPIAntiochError(
            f"refusing to remove container {container.name!r} with changed ownership"
        )
    _run([config.docker_bin, "rm", "--force", container.container_id])
    if _inspect_policy_container(config, container.container_id) is not None:
        raise OpenPIAntiochError("managed OpenPI container remains after cleanup")


def _deadline(timeout_s: float, *, operation: str) -> float:
    """Return a monotonic deadline for a bounded external operation."""

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise OpenPIAntiochError(
            f"{operation} timeout must be a positive finite number"
        )
    return time.monotonic() + timeout_s


def _wait_for_policy(config: LiveLoopConfig, *, container_name: str) -> None:
    """Wait for a local policy socket without leaving an unbounded GPU container."""

    deadline = _deadline(config.policy_ready_timeout_s, operation="policy readiness")
    while True:
        try:
            with socket.create_connection(("127.0.0.1", config.host_port), timeout=2):
                return
        except OSError:
            state = _run(
                [
                    config.docker_bin,
                    "inspect",
                    container_name,
                    "--format",
                    "{{.State.Running}} {{.State.ExitCode}}",
                ],
                check=False,
            )
            if state.returncode != 0 or not state.stdout.startswith("true "):
                logs = _run([config.docker_bin, "logs", container_name], check=False)
                tail = (logs.stdout + logs.stderr)[-2000:]
                raise OpenPIAntiochError(
                    f"OpenPI container exited before websocket readiness: {tail}"
                )
            if time.monotonic() >= deadline:
                raise OpenPIAntiochError(
                    "OpenPI policy did not become ready before timeout"
                )
            time.sleep(2)


def _scenario_run_id(payload: object) -> str:
    """Extract exactly one scenario ID from an authoritative submission response."""

    response: Mapping[str, Any]
    if isinstance(payload, Mapping):
        response = payload
    elif (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], Mapping)
    ):
        response = payload[0]
    else:
        raise OpenPIAntiochError(
            "Antioch submission did not identify exactly one scenario run"
        )
    for key in ("scenario_run_id", "id"):
        candidate = response.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    raise OpenPIAntiochError("Antioch submission contained no scenario run ID")


def _scenario_payload(config: LiveLoopConfig, scenario_run_id: str) -> dict[str, Any]:
    """Read one exact scenario record through the supported structured CLI."""

    shown = _run(
        [config.antioch_bin, "scenario", "show", scenario_run_id, "--json"],
        cwd=config.project_dir,
    )
    return _json_object(shown.stdout, label="Antioch scenario show")


def _scenario_is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether a scenario record proves no active remote execution remains."""

    phase = str(payload.get("phase") or "").lower()
    outcome = str(payload.get("outcome") or "").lower()
    return phase in {
        "completed",
        "cancelled",
        "canceled",
        "failed",
        "error",
    } or outcome in {
        "passed",
        "cancelled",
        "canceled",
        "failed",
        "error",
    }


def _wait_for_scenario(config: LiveLoopConfig, scenario_run_id: str) -> dict[str, Any]:
    """Wait for the selected scenario result until its explicit deadline."""

    deadline = _deadline(config.scenario_timeout_s, operation="scenario")
    while True:
        payload = _scenario_payload(config, scenario_run_id)
        if _scenario_is_terminal(payload):
            return payload
        if time.monotonic() >= deadline:
            raise OpenPIAntiochError("Antioch scenario did not complete before timeout")
        time.sleep(2)


def _cancel_active_scenario(
    config: LiveLoopConfig, scenario_run_id: str
) -> tuple[str, dict[str, Any]]:
    """Cancel one active submitted scenario and verify its terminal provider state."""

    payload = _scenario_payload(config, scenario_run_id)
    if _scenario_is_terminal(payload):
        return "already_terminal", payload
    _run(
        [config.antioch_bin, "scenario", "cancel", scenario_run_id, "--json"],
        cwd=config.project_dir,
    )
    deadline = _deadline(config.scenario_timeout_s, operation="scenario cleanup")
    while True:
        payload = _scenario_payload(config, scenario_run_id)
        if _scenario_is_terminal(payload):
            return "terminal_verified", payload
        if time.monotonic() >= deadline:
            raise OpenPIAntiochError(
                "Antioch scenario remained active after cancellation"
            )
        time.sleep(2)


def validate_scenario_evidence(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return sanitized evidence only when every live feedback gate passed."""

    if payload.get("outcome") != "passed":
        raise OpenPIAntiochError(
            f"Antioch scenario outcome was {payload.get('outcome')!r}, not 'passed'"
        )
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise OpenPIAntiochError("Antioch result contains no results object")
    checks = results.get("checks")
    if not isinstance(checks, list):
        raise OpenPIAntiochError("Antioch result contains no check evidence")
    passed = {
        str(check.get("criterion"))
        for check in checks
        if isinstance(check, Mapping) and check.get("passed") is True
    }
    missing = REQUIRED_CHECKS - passed
    if missing:
        raise OpenPIAntiochError(
            f"Antioch feedback result is missing passing checks: {sorted(missing)}"
        )
    chunks = results.get("chunks_run")
    inference_ms = results.get("mean_inference_ms")
    arm_travel = results.get("max_joint_travel_rad")
    jaw_travel = results.get("jaw_travel_mm")
    if not isinstance(chunks, int) or chunks < 1:
        raise OpenPIAntiochError("Antioch result proves no policy inference chunks")
    for name, value in (
        ("mean_inference_ms", inference_ms),
        ("max_joint_travel_rad", arm_travel),
        ("jaw_travel_mm", jaw_travel),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise OpenPIAntiochError(f"Antioch result has invalid {name}: {value!r}")
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "passed",
        "transport": "upstream_openpi_websocket",
        "simulation": "antioch_isaac",
        "policy": "openpi_pi05_droid",
        "containerized_policy": True,
        "chunks_run": chunks,
        "action_chunk_shape": [15, 8],
        "mean_inference_ms": inference_ms,
        "max_joint_travel_rad": arm_travel,
        "jaw_travel_mm": jaw_travel,
        "passing_checks": sorted(passed),
        "credentials_persisted": False,
        "acceptance_persisted": False,
    }


def _require_positive_chunks(chunks: int) -> None:
    """Reject a chunk count that cannot prove any policy action."""

    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise OpenPIAntiochError("chunks must be a positive integer")


def _positive_chunk_count(value: str) -> int:
    """Parse a CLI chunk count without permitting a no-op policy loop."""

    try:
        chunks = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("chunks must be an integer") from error
    try:
        _require_positive_chunks(chunks)
    except OpenPIAntiochError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return chunks


def _validate_live_config(config: LiveLoopConfig) -> None:
    """Reject unsafe cleanup or evidence settings before an external action."""

    _require_positive_chunks(config.chunks)
    cleanup_requested = config.cleanup_container or config.cleanup_scenario
    if cleanup_requested and not config.resource_owner:
        raise OpenPIAntiochError("task-owned cleanup requires a resource owner")
    if config.resource_owner and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", config.resource_owner
    ):
        raise OpenPIAntiochError("resource owner must be a safe task identifier")
    if config.cleanup_scenario and config.script:
        raise OpenPIAntiochError("scenario cleanup requires the recorded scenario path")
    if config.private_receipt_path is not None and not config.resource_owner:
        raise OpenPIAntiochError("a private receipt requires a resource owner")


def validate_run_output(output: str, *, expected_chunks: int) -> dict[str, object]:
    """Validate the measured output of a direct ``antioch run`` Pi loop."""

    _require_positive_chunks(expected_chunks)
    if "FAIL:" in output or "ALL GATES PASSED" not in output:
        raise OpenPIAntiochError(
            "direct Antioch run did not emit its all-gates verdict"
        )
    chunk_matches = re.findall(
        r"^chunk\s+(\d+):\s+([0-9.]+)\s+ms\s+shape=\(15,\s*8\)",
        output,
        flags=re.MULTILINE,
    )
    if len(chunk_matches) != expected_chunks:
        raise OpenPIAntiochError(
            f"direct Antioch run proved {len(chunk_matches)} of {expected_chunks} chunks"
        )
    jaw_match = re.search(r"^jaw travel:\s+([0-9.]+)\s+mm$", output, re.MULTILINE)
    arm_match = re.search(
        r"^max joint travel:\s+([0-9.]+)\s+rad$", output, re.MULTILINE
    )
    latency_match = re.search(
        r"^mean inference latency:\s+([0-9.]+)\s+ms over\s+(\d+)\s+chunks$",
        output,
        re.MULTILINE,
    )
    if not jaw_match or not arm_match or not latency_match:
        raise OpenPIAntiochError("direct Antioch run omitted measured loop evidence")
    jaw_travel = float(jaw_match.group(1))
    arm_travel = float(arm_match.group(1))
    inference_ms = float(latency_match.group(1))
    if (
        jaw_travel <= 0
        or arm_travel <= 0
        or inference_ms <= 0
        or int(latency_match.group(2)) != expected_chunks
    ):
        raise OpenPIAntiochError("direct Antioch run emitted invalid measured evidence")
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "passed",
        "transport": "upstream_openpi_websocket",
        "simulation": "antioch_isaac",
        "policy": "openpi_pi05_droid",
        "containerized_policy": True,
        "execution": "antioch_run",
        "chunks_run": expected_chunks,
        "action_chunk_shape": [15, 8],
        "mean_inference_ms": inference_ms,
        "max_joint_travel_rad": arm_travel,
        "jaw_travel_mm": jaw_travel,
        "passing_checks": sorted(REQUIRED_CHECKS),
        "credentials_persisted": False,
        "acceptance_persisted": False,
    }


def _run_direct_loop(config: LiveLoopConfig) -> dict[str, object]:
    """Run the documented direct harness when no scenario artifact is requested."""

    policy_port = config.policy_port or config.host_port
    run_argv = [config.antioch_bin, "run"]
    if config.machine:
        run_argv.extend(["--machine", config.machine])
    run_argv.extend(
        [
            "--no-stream",
            config.script or "",
            "--policy",
            "--host",
            config.policy_host,
            "--port",
            str(policy_port),
            "--chunks",
            str(config.chunks),
        ]
    )
    completed = _run(run_argv, cwd=config.project_dir)
    return validate_run_output(completed.stdout, expected_chunks=config.chunks)


def _submit_scenario(config: LiveLoopConfig) -> str:
    """Submit one scenario and bind all later work to its returned ID."""

    if config.rerun_from:
        queued = _run(
            [
                config.antioch_bin,
                "scenario",
                "rerun",
                config.rerun_from,
                "--json",
            ],
            cwd=config.project_dir,
        )
        return _scenario_run_id(
            _json_value(queued.stdout, label="Antioch scenario rerun")
        )

    policy_port = config.policy_port or config.host_port
    scenario_argv = [
        config.antioch_bin,
        "scenario",
        "run",
        "--scenario",
        config.scenario,
        "--set",
        f"host={config.policy_host}",
        "--set",
        f"port={policy_port}",
        "--set",
        f"chunks={config.chunks}",
    ]
    if config.machine:
        scenario_argv.extend(["--machine", config.machine])
    scenario_argv.extend(["--detach", "--json"])
    queued = _run(scenario_argv, cwd=config.project_dir)
    return _scenario_run_id(
        _json_value(queued.stdout, label="Antioch scenario submission")
    )


def _image_identity(config: LiveLoopConfig) -> str:
    """Return the exact content-addressed image ID used by this invocation."""

    image_id = _run(
        [config.docker_bin, "image", "inspect", config.image, "--format", "{{.Id}}"]
    ).stdout.strip()
    if not image_id.startswith("sha256:"):
        raise OpenPIAntiochError("OpenPI image has no content-addressed ID")
    return image_id


def _cleanup_live_resources(
    config: LiveLoopConfig, resources: _LiveResources
) -> list[Exception]:
    """Clean only exact resources, retaining all cleanup failures for disposition."""

    errors: list[Exception] = []
    if config.cleanup_scenario:
        if resources.scenario_run_id is None:
            resources.scenario_cleanup = "not_created"
        else:
            try:
                cleanup, payload = _cancel_active_scenario(
                    config, resources.scenario_run_id
                )
                resources.scenario_cleanup = cleanup
                resources.scenario_phase = str(payload.get("phase") or "") or None
                resources.scenario_outcome = str(payload.get("outcome") or "") or None
            except Exception as error:
                resources.scenario_cleanup = "failed"
                errors.append(error)
    if config.cleanup_container:
        if resources.container is None:
            resources.container_cleanup = "not_created"
        else:
            try:
                _remove_policy_container(config, resources.container)
                resources.container_cleanup = "absent_verified"
                resources.container_absent = True
            except Exception as error:
                resources.container_cleanup = "failed"
                resources.container_absent = False
                errors.append(error)
    return errors


def _harness_source_sha() -> str:
    """Return the exact repository source identity for private live evidence."""

    source_sha = _run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise OpenPIAntiochError("harness source has no immutable Git identity")
    return source_sha


def _write_private_receipt(
    config: LiveLoopConfig, resources: _LiveResources, *, status: str
) -> None:
    """Atomically retain private bindings and provider-side cleanup evidence."""

    receipt = config.private_receipt_path
    if receipt is None:
        return
    if receipt.parent.is_symlink() or not receipt.parent.is_dir():
        raise OpenPIAntiochError("private receipt parent must be an owned directory")
    payload = {
        "schema": "npa.workbench.openpi.antioch-live-receipt.v1",
        "status": status,
        "harness_source_sha": _harness_source_sha(),
        "openpi_source_ref": SOURCE_REF,
        "resource_owner": config.resource_owner,
        "image": config.image,
        "image_id": resources.image_id,
        "scenario": config.scenario,
        "scenario_run_id": resources.scenario_run_id,
        "container": (
            None
            if resources.container is None
            else {
                "name": resources.container.name,
                "id": resources.container.container_id,
                "image": resources.container.image,
                "owner": resources.container.owner,
            }
        ),
        "policy": "openpi_pi05_droid",
        "acceptance": (
            None
            if resources.evidence is None
            else {
                "action_chunk_shape": resources.evidence.get("action_chunk_shape"),
                "chunks_run": resources.evidence.get("chunks_run"),
                "mean_inference_ms": resources.evidence.get("mean_inference_ms"),
                "max_joint_travel_rad": resources.evidence.get("max_joint_travel_rad"),
                "jaw_travel_mm": resources.evidence.get("jaw_travel_mm"),
                "passing_checks": resources.evidence.get("passing_checks"),
            }
        ),
        "scenario_cleanup": {
            "status": resources.scenario_cleanup,
            "phase": resources.scenario_phase,
            "outcome": resources.scenario_outcome,
        },
        "container_cleanup": {
            "status": resources.container_cleanup,
            "absent_verified": resources.container_absent,
        },
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(receipt, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        try:
            os.unlink(receipt)
        except OSError:
            pass
        raise


def run_live_loop(config: LiveLoopConfig) -> dict[str, object]:
    """Run one owned loop and retain exact, private resource evidence."""

    _validate_live_config(config)
    require_openpi_terms()
    if not config.project_dir.is_dir() or not config.cache_dir.is_dir():
        raise OpenPIAntiochError("project and OpenPI cache directories must exist")
    _run([config.antioch_bin, "auth", "whoami"], cwd=config.project_dir)
    _run([config.antioch_bin, "project", "current"], cwd=config.project_dir)
    resources = _LiveResources(image_id=_image_identity(config))
    _negative_terms_probe(config)

    primary_error: BaseException | None = None
    result: dict[str, object] | None = None
    status = "failed"
    try:
        resources.container = _ensure_policy_container(config)
        _wait_for_policy(config, container_name=resources.container.name)
        if config.script:
            evidence = _run_direct_loop(config)
        else:
            resources.scenario_run_id = _submit_scenario(config)
            evidence = validate_scenario_evidence(
                _wait_for_scenario(config, resources.scenario_run_id)
            )
        resources.evidence = evidence
        result = {**evidence, "image_id": resources.image_id}
        status = "passed"
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_errors = _cleanup_live_resources(config, resources)
        try:
            _write_private_receipt(config, resources, status=status)
        except Exception as error:
            cleanup_errors.append(error)
        if primary_error is not None:
            raise primary_error
        if cleanup_errors:
            raise cleanup_errors[0]
    if result is None:
        raise OpenPIAntiochError("OpenPI/Antioch loop produced no evidence")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-image")
    build.add_argument("--openpi-dir", type=Path, required=True)
    build.add_argument("--image", required=True)
    build.add_argument("--docker-bin", default="docker")

    live = commands.add_parser("live-loop")
    live.add_argument("--project-dir", type=Path, required=True)
    live.add_argument("--cache-dir", type=Path, required=True)
    live.add_argument("--image", required=True)
    live.add_argument("--policy-host", required=True)
    live.add_argument("--host-port", type=int, default=8000)
    live.add_argument(
        "--policy-port",
        type=int,
        help="Simulator-facing port when it differs from the local published port",
    )
    live.add_argument("--scenario", default="pi05_droid_loop")
    live.add_argument(
        "--rerun-from",
        help="Completed run whose saved Antioch environment and inputs are reproduced",
    )
    live.add_argument("--machine", help="Exact assigned Antioch machine selector")
    live.add_argument(
        "--script",
        help="Project-relative Pi loop for direct `antioch run` execution",
    )
    live.add_argument("--chunks", type=_positive_chunk_count, default=3)
    live.add_argument(
        "--policy-ready-timeout-s",
        type=float,
        default=300.0,
        help="Maximum monotonic seconds to wait for the local policy socket",
    )
    live.add_argument(
        "--scenario-timeout-s",
        type=float,
        default=1800.0,
        help="Maximum monotonic seconds to wait for the selected Antioch result",
    )
    live.add_argument("--docker-bin", default="docker")
    live.add_argument("--antioch-bin", default="antioch")
    live.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    live.add_argument(
        "--cleanup-container",
        action="store_true",
        help="Remove only the exact labelled task-owned policy container after the run",
    )
    live.add_argument(
        "--cleanup-scenario",
        action="store_true",
        help="Cancel the exact submitted scenario and verify its terminal provider state",
    )
    live.add_argument(
        "--resource-owner",
        help="Unique task owner recorded on cleanup-eligible local resources",
    )
    live.add_argument(
        "--private-receipt",
        type=Path,
        help="Precreated private evidence directory's unique receipt path",
    )
    live.add_argument("--output", type=Path)
    return parser


def _failure_result(error: Exception) -> dict[str, str]:
    """Return a sanitized, machine-readable result for a local operation failure."""

    return {
        "status": "failed",
        "error_type": type(error).__name__,
        "message": "OpenPI/Antioch operation failed; inspect private local logs.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-image":
            result = build_local_image(
                openpi_dir=args.openpi_dir,
                image=args.image,
                docker_bin=args.docker_bin,
            )
        else:
            result = run_live_loop(
                LiveLoopConfig(
                    project_dir=args.project_dir,
                    cache_dir=args.cache_dir,
                    image=args.image,
                    policy_host=args.policy_host,
                    host_port=args.host_port,
                    policy_port=args.policy_port,
                    scenario=args.scenario,
                    chunks=args.chunks,
                    policy_ready_timeout_s=args.policy_ready_timeout_s,
                    scenario_timeout_s=args.scenario_timeout_s,
                    docker_bin=args.docker_bin,
                    antioch_bin=args.antioch_bin,
                    rerun_from=args.rerun_from,
                    machine=args.machine,
                    script=args.script,
                    container_name=args.container_name,
                    cleanup_container=args.cleanup_container,
                    cleanup_scenario=args.cleanup_scenario,
                    resource_owner=args.resource_owner or "",
                    private_receipt_path=args.private_receipt,
                )
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    except (OpenPIAntiochError, ValueError) as error:
        print(json.dumps(_failure_result(error), sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

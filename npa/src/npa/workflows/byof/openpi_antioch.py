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
import time
from typing import Any, Mapping, Sequence

from npa.workflows.byof.openpi import OPENPI_TERMS_ENV, require_openpi_terms
from npa.workflows.byof.openpi_pipeline import SOURCE_REF

EVIDENCE_SCHEMA = "npa.workbench.openpi.antioch-loop.v1"
MANAGED_CONTAINER_LABEL = "npa.openpi-antioch.managed=true"
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
ENTRYPOINT ["/opt/venv/bin/python", "scripts/serve_policy.py"]
"""


class OpenPIAntiochError(RuntimeError):
    """Raised when the connected OpenPI/Antioch validation is not proven."""


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
    docker_bin: str = "docker"
    antioch_bin: str = "antioch"
    rerun_from: str | None = None
    machine: str | None = None
    script: str | None = None
    container_name: str = DEFAULT_CONTAINER_NAME
    cleanup_container: bool = False


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


def _json_object(text: str, *, label: str) -> dict[str, Any]:
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
    """Prove an unaccepted child refuses before the OpenPI entrypoint starts."""

    child_env = dict(os.environ)
    child_env.pop(OPENPI_TERMS_ENV, None)
    completed = _run(
        [
            config.docker_bin,
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            config.image,
            "-c",
            f'test "${{{OPENPI_TERMS_ENV}:-}}" = YES || {{ '
            'echo "NPA_OPENPI_TERMS_REFUSED"; exit 64; }',
        ],
        env=child_env,
        check=False,
    )
    if completed.returncode != 64 or "NPA_OPENPI_TERMS_REFUSED" not in completed.stdout:
        raise OpenPIAntiochError(
            "negative OpenPI terms child did not fail closed before model startup"
        )


def _accepted_container_argv(
    config: LiveLoopConfig, *, container_name: str
) -> list[str]:
    return [
        config.docker_bin,
        "run",
        "--detach",
        "--name",
        container_name,
        "--restart",
        "unless-stopped",
        "--label",
        MANAGED_CONTAINER_LABEL,
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
            "/opt/venv/bin/python -c \"import socket; "
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


def _ensure_policy_container(config: LiveLoopConfig) -> bool:
    """Start or safely reuse the single task-managed policy container.

    Returns ``True`` when this call created the container. A name collision is
    accepted only when both the management label and requested image match.
    """

    inspected = _run(
        [
            config.docker_bin,
            "inspect",
            config.container_name,
            "--format",
            (
                '{{index .Config.Labels "npa.openpi-antioch.managed"}} '
                "{{.Config.Image}} {{.State.Running}}"
            ),
        ],
        check=False,
    )
    if inspected.returncode == 0:
        fields = inspected.stdout.strip().split()
        if len(fields) != 3 or fields[:2] != ["true", config.image]:
            raise OpenPIAntiochError(
                f"container name {config.container_name!r} is not the requested "
                "task-managed OpenPI container"
            )
        if fields[2] != "true":
            _run([config.docker_bin, "start", config.container_name])
        return False
    _run(_accepted_container_argv(config, container_name=config.container_name))
    return True


def _remove_policy_container(config: LiveLoopConfig) -> None:
    """Remove only the container carrying the task management label."""

    inspected = _run(
        [
            config.docker_bin,
            "inspect",
            config.container_name,
            "--format",
            '{{index .Config.Labels "npa.openpi-antioch.managed"}}',
        ],
        check=False,
    )
    if inspected.returncode != 0:
        return
    if inspected.stdout.strip() != "true":
        raise OpenPIAntiochError(
            f"refusing to remove unlabelled container {config.container_name!r}"
        )
    _run([config.docker_bin, "rm", "--force", config.container_name])


def _wait_for_policy(config: LiveLoopConfig, *, container_name: str) -> None:
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
            time.sleep(2)


def _scenario_run_id(payload: Mapping[str, Any]) -> str:
    candidates: list[object] = [payload.get("scenario_run_id"), payload.get("id")]
    for key in ("items", "runs", "scenario_runs"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], Mapping):
            candidates.extend(
                [value[0].get("scenario_run_id"), value[0].get("id")]
            )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    raise OpenPIAntiochError("Antioch queue response contained no scenario run ID")


def _latest_scenario(config: LiveLoopConfig) -> dict[str, Any] | None:
    listed = _run(
        [
            config.antioch_bin,
            "scenario",
            "list",
            "--scenario",
            config.scenario,
            "--mine",
            "--limit",
            "1",
            "--json",
        ],
        cwd=config.project_dir,
    )
    payload = _json_object(listed.stdout, label="Antioch scenario list")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    latest = items[0]
    if not isinstance(latest, dict):
        raise OpenPIAntiochError("Antioch scenario list item is not an object")
    return latest


def _wait_for_scenario(config: LiveLoopConfig, scenario_run_id: str) -> dict[str, Any]:
    while True:
        shown = _run(
            [
                config.antioch_bin,
                "scenario",
                "show",
                scenario_run_id,
                "--json",
            ],
            cwd=config.project_dir,
        )
        payload = _json_object(shown.stdout, label="Antioch scenario show")
        if payload.get("phase") == "completed" or payload.get("outcome"):
            return payload
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


def validate_run_output(output: str, *, expected_chunks: int) -> dict[str, object]:
    """Validate the measured output of a direct ``antioch run`` Pi loop."""

    if "FAIL:" in output or "ALL GATES PASSED" not in output:
        raise OpenPIAntiochError("direct Antioch run did not emit its all-gates verdict")
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


def run_live_loop(config: LiveLoopConfig) -> dict[str, object]:
    """Run the container, dispatch Antioch, and validate the returned feedback."""

    require_openpi_terms()
    if not config.project_dir.is_dir() or not config.cache_dir.is_dir():
        raise OpenPIAntiochError("project and OpenPI cache directories must exist")
    _run([config.antioch_bin, "auth", "whoami"], cwd=config.project_dir)
    _run([config.antioch_bin, "project", "current"], cwd=config.project_dir)
    _run([config.docker_bin, "image", "inspect", config.image])
    _negative_terms_probe(config)

    try:
        _ensure_policy_container(config)
        _wait_for_policy(config, container_name=config.container_name)
        if config.script:
            policy_port = config.policy_port or config.host_port
            run_argv = [config.antioch_bin, "run"]
            if config.machine:
                run_argv.extend(["--machine", config.machine])
            run_argv.extend(
                [
                    "--no-stream",
                    config.script,
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
            evidence = validate_run_output(
                completed.stdout, expected_chunks=config.chunks
            )
        elif config.rerun_from:
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
            run_id = _scenario_run_id(
                _json_object(queued.stdout, label="Antioch scenario rerun")
            )
        else:
            previous = _latest_scenario(config)
            previous_id = _scenario_run_id(previous) if previous else None
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
                "--no-stream",
                "--verbose",
            ]
            if config.machine:
                scenario_argv.extend(["--machine", config.machine])
            _run(scenario_argv, cwd=config.project_dir)
            latest = _latest_scenario(config)
            if latest is None:
                raise OpenPIAntiochError(
                    "Antioch completed without saving a scenario result"
                )
            run_id = _scenario_run_id(latest)
            if run_id == previous_id:
                raise OpenPIAntiochError(
                    "Antioch completed without creating a fresh scenario result"
                )
        if not config.script:
            result = _wait_for_scenario(config, run_id)
            evidence = validate_scenario_evidence(result)
        image_id = _run(
            [
                config.docker_bin,
                "image",
                "inspect",
                config.image,
                "--format",
                "{{.Id}}",
            ]
        ).stdout.strip()
        return {**evidence, "image_id": image_id}
    finally:
        if config.cleanup_container:
            _remove_policy_container(config)


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
    live.add_argument("--chunks", type=int, default=3)
    live.add_argument("--docker-bin", default="docker")
    live.add_argument("--antioch-bin", default="antioch")
    live.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    live.add_argument(
        "--cleanup-container",
        action="store_true",
        help="Remove the labelled policy container after the run instead of keeping it durable",
    )
    live.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
                docker_bin=args.docker_bin,
                antioch_bin=args.antioch_bin,
                rerun_from=args.rerun_from,
                machine=args.machine,
                script=args.script,
                container_name=args.container_name,
                cleanup_container=args.cleanup_container,
            )
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

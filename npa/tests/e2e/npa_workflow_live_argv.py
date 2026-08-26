"""Project-consistent argv builders for live npa.workflow lifecycle tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def _project_args(project: str | None) -> list[str]:
    return ["--project", project] if project else []


def _assume_args(assume_decision: str) -> list[str]:
    return (
        ["--assume-decision", assume_decision]
        if assume_decision.strip()
        else []
    )


def _preset_args(preset: str) -> list[str]:
    return ["--preset", preset] if preset.strip() else []


def plan_submit_args(
    path: Path,
    *,
    run_id: str,
    registry: str,
    project: str | None,
    assume_decision: str = "",
    preset: str = "",
    config_vars: Iterable[tuple[str, str]] = (),
    image_args: Sequence[str] = (),
    skypilot_config_args: Sequence[str] = (),
) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--plan-only",
        "--registry",
        registry,
        "--output-format",
        "json",
        *_project_args(project),
        *_assume_args(assume_decision),
        *_preset_args(preset),
        *image_args,
        *skypilot_config_args,
    ]
    for key, value in config_vars:
        args.extend(["--var", f"{key}={value}"])
    return args


def one_shot_submit_args(
    path: Path,
    *,
    run_id: str,
    registry: str,
    project: str | None,
    assume_decision: str = "",
    preset: str = "",
    config_vars: Iterable[tuple[str, str]] = (),
    image_args: Sequence[str] = (),
    secret_env_args: Sequence[str] = (),
    skypilot_config_args: Sequence[str] = (),
) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--registry",
        registry,
        "--submit-timeout",
        "1800",
        "--output-format",
        "json",
        *_project_args(project),
        *_assume_args(assume_decision),
        *_preset_args(preset),
        *image_args,
        *secret_env_args,
        *skypilot_config_args,
    ]
    for key, value in config_vars:
        args.extend(["--var", f"{key}={value}"])
    return args


def runtime_submit_args(
    path: Path,
    *,
    run_id: str,
    registry: str,
    project: str | None,
    poll_seconds: int,
    max_wait_seconds: int,
    cancel_on_timeout: bool,
    config_vars: Iterable[tuple[str, str]] = (),
    preset: str = "",
    image_args: Sequence[str] = (),
    secret_env_args: Sequence[str] = (),
    skypilot_config_args: Sequence[str] = (),
    resume: bool = False,
) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--runtime",
        "--registry",
        registry,
        "--poll-seconds",
        str(poll_seconds),
        "--max-wait-seconds",
        str(max_wait_seconds),
        "--submit-timeout",
        "1800",
        "--output-format",
        "json",
        *_project_args(project),
        *_preset_args(preset),
    ]
    if not cancel_on_timeout:
        args.append("--no-cancel-on-timeout")
    for key, value in config_vars:
        args.extend(["--var", f"{key}={value}"])
    args.extend([*image_args, *secret_env_args, *skypilot_config_args])
    if resume:
        args.append("--resume")
    return args


def status_args(
    run_id: str,
    *,
    project: str | None,
    workflow_s3_uri: str = "",
) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "status",
        run_id,
        "--json",
        *_project_args(project),
    ]
    if workflow_s3_uri:
        args.extend(["--workflow-s3-uri", workflow_s3_uri])
    return args

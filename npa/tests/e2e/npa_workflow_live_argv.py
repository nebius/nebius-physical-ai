"""Project-consistent argv builders for live npa.workflow lifecycle tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from pathlib import Path
import re
import stat


def _project_args(project: str | None) -> list[str]:
    return ["--project", project] if project else []


def _assume_args(assume_decision: str) -> list[str]:
    return ["--assume-decision", assume_decision] if assume_decision.strip() else []


def _preset_args(preset: str) -> list[str]:
    return ["--preset", preset] if preset.strip() else []


def _workflow_storage_args(prefix: str) -> list[str]:
    return ["--workflow-s3-prefix", prefix] if prefix.strip() else []


def _safe_relative_workflow_prefix(value: str) -> str:
    prefix = value.strip()
    valid = re.fullmatch(r"[A-Za-z0-9._/-]+", prefix or "") and all(
        part not in {"", ".", ".."} for part in prefix.split("/")
    )
    if not valid:
        raise ValueError("workflow storage prefix must be a safe relative key")
    return prefix


def _workflow_prefixes_disjoint(left: str, right: str) -> bool:
    return not (
        right == left or right.startswith(f"{left}/") or left.startswith(f"{right}/")
    )


def _owned_empty_isolation_root(value: str) -> Path:
    root = Path(value)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("isolation root must be an absolute canonical path")
    try:
        metadata = root.lstat()
    except FileNotFoundError as exc:
        raise ValueError("isolation root must already exist") from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ValueError("isolation root must be a real directory")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise ValueError("isolation root must be owner-only and owned by this user")
    if next(root.iterdir(), None) is not None:
        raise ValueError("isolation root must be fresh and empty")
    return root


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
    workflow_s3_prefix: str = "",
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
        *_workflow_storage_args(workflow_s3_prefix),
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
    workflow_s3_prefix: str = "",
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
        *_workflow_storage_args(workflow_s3_prefix),
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

"""Safe deferred removal of only the invoking repository-local NPA venv."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
from typing import Any, Callable

import typer


PLAN_SCHEMA = "npa.uninstall.plan.v1"
_RECEIPT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]{5,80}$")


@dataclass(frozen=True)
class EnvironmentInspection:
    target: Path
    repo_root: Path
    executable: Path
    safe: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": str(self.target),
            "repo_root": str(self.repo_root),
            "executable": str(self.executable),
            "safe": self.safe,
            "reasons": list(self.reasons),
        }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _receipt_root() -> Path:
    override = os.environ.get("NPA_UNINSTALL_RECEIPT_DIR", "").strip()
    return (
        Path(override).expanduser()
        if override
        else Path.home() / ".npa" / "uninstall-receipts"
    )


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _active_environment_processes(target: Path) -> list[str]:
    found: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        for label in ("exe", "cwd"):
            try:
                resolved = (item / label).resolve(strict=True)
            except OSError:
                continue
            if _path_inside(resolved, target):
                found.append(f"pid {item.name} {label}={resolved}")
        try:
            arguments = (item / "cmdline").read_bytes().split(b"\0")
        except OSError:
            arguments = []
        for raw_argument in arguments:
            argument = Path(os.fsdecode(raw_argument)) if raw_argument else Path()
            if argument.is_absolute() and _path_inside(argument, target):
                found.append(f"pid {item.name} argv={argument}")
                break
        try:
            environment = (item / "environ").read_bytes().split(b"\0")
        except OSError:
            environment = []
        virtual_env = next(
            (
                os.fsdecode(item_value).partition("=")[2]
                for item_value in environment
                if item_value.startswith(b"VIRTUAL_ENV=")
            ),
            "",
        )
        if virtual_env and Path(virtual_env).expanduser().absolute() == target:
            found.append(f"pid {item.name} VIRTUAL_ENV={target}")
    return found


def inspect_repository_environment(
    *,
    executable: Path | None = None,
    cwd: Path | None = None,
    current_prefix: Path | None = None,
    base_prefix: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    scan_processes: bool = True,
) -> EnvironmentInspection:
    """Resolve the target from the interpreter and prove all deletion invariants."""

    raw_executable = Path(executable or sys.executable).expanduser()
    if not raw_executable.is_absolute():
        raw_executable = (cwd or Path.cwd()) / raw_executable
    reasons: list[str] = []
    try:
        # A normal venv's bin/python is itself a symlink to the base interpreter.
        # Derive the environment from the lexical sys.executable path, while
        # separately resolving the target directory and interpreter for checks.
        lexical_executable = raw_executable.absolute()
        resolved_executable = lexical_executable.resolve(strict=True)
        lexical_target = lexical_executable.parent.parent
        target = lexical_target.resolve(strict=True)
    except OSError as exc:
        return EnvironmentInspection(
            raw_executable.parent.parent,
            raw_executable.parent.parent.parent.parent,
            raw_executable,
            False,
            (f"running interpreter cannot be resolved: {exc}",),
        )
    if not resolved_executable.is_file():
        reasons.append("the running interpreter does not resolve to a regular file")
    nested_repo = target.parent.parent
    root_repo = target.parent
    if (
        (nested_repo / ".git").exists()
        and (nested_repo / "npa" / "pyproject.toml").is_file()
        and target == nested_repo / "npa" / ".venv"
    ):
        repo_root = nested_repo
    elif (
        (root_repo / ".git").exists()
        and (root_repo / "npa" / "pyproject.toml").is_file()
        and target == root_repo / ".venv"
    ):
        # The public install guide has historically used repo-root .venv;
        # retain that one equally exact repository-local layout.
        repo_root = root_repo
    else:
        repo_root = nested_repo
    expected_targets = {repo_root / "npa" / ".venv", repo_root / ".venv"}
    if lexical_target.is_symlink() or lexical_target.absolute() != target:
        reasons.append("the environment path contains a symlink escape")
    if target not in expected_targets:
        reasons.append(
            "the running interpreter is not an exact supported repository-local "
            f"environment ({target}; expected one of "
            + ", ".join(str(item) for item in sorted(expected_targets))
            + ")"
        )
    if lexical_executable.parent.resolve() != target / "bin":
        reasons.append("the running interpreter is not directly inside target/bin")
    for protected, label in (
        (Path("/"), "filesystem root"),
        (Path.home().resolve(), "home directory"),
        (repo_root, "repository root"),
        (repo_root / "npa", "npa source directory"),
    ):
        try:
            if target.resolve() == protected.resolve():
                reasons.append(f"target resolves to the protected {label}")
        except OSError:
            reasons.append(f"could not resolve protected-path invariant for {label}")
    if target.is_symlink():
        reasons.append("the environment directory is a symlink")
    marker = target / "pyvenv.cfg"
    if marker.is_symlink() or not marker.is_file():
        reasons.append("pyvenv.cfg is missing or is a symlink")
    if (
        not (repo_root / ".git").exists()
        or not (repo_root / "npa" / "pyproject.toml").is_file()
    ):
        reasons.append(
            "repository identity markers (.git and npa/pyproject.toml) are missing"
        )
    if (target / "conda-meta").exists() or os.environ.get("CONDA_PREFIX"):
        reasons.append(
            "conda environments are shared/externally managed and are refused"
        )
    externally_managed = [
        target / "EXTERNALLY-MANAGED",
        *target.glob("lib/python*/EXTERNALLY-MANAGED"),
    ]
    if any(marker.is_file() for marker in externally_managed):
        reasons.append("externally managed Python environments are refused")
    if os.environ.get("PIPX_HOME") or "pipx" in str(target).lower():
        reasons.append("pipx/user-wide environments are refused")
    prefix = Path(current_prefix or sys.prefix).resolve()
    base = Path(base_prefix or sys.base_prefix).resolve()
    if executable is None and prefix != target:
        reasons.append(f"sys.prefix {prefix} does not match target {target}")
    if target == base:
        reasons.append("the target is the system/base Python prefix")
    try:
        if target.exists() and target.stat().st_uid != os.getuid():
            reasons.append("the environment is not owned by the invoking user")
    except OSError as exc:
        reasons.append(f"the environment cannot be stat'ed: {exc}")
    working = (cwd or Path.cwd()).resolve()
    if _path_inside(working, target):
        reasons.append("the current working directory is inside the target environment")
    if repo_root.exists():
        try:
            dirty = runner(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    str(target.relative_to(repo_root)),
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            reasons.append(f"could not verify dirty overlap: {exc}")
        else:
            if dirty.returncode != 0:
                reasons.append(
                    "could not verify dirty overlap: "
                    + str(dirty.stderr or f"git exit {dirty.returncode}").strip()
                )
            elif str(dirty.stdout or "").strip():
                reasons.append("tracked/dirty repository content overlaps the target")
    if scan_processes:
        active = _active_environment_processes(target)
        if active:
            reasons.append(
                "another process is using the environment: " + "; ".join(active[:5])
            )
    return EnvironmentInspection(
        target=target,
        repo_root=repo_root,
        executable=lexical_executable,
        safe=not reasons,
        reasons=tuple(reasons),
    )


def _pid_start(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except OSError:
        return ""
    return fields[21] if len(fields) > 21 else ""


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_receipt(receipt_id: str) -> tuple[Path, dict[str, Any]]:
    if not _RECEIPT_ID.fullmatch(receipt_id):
        raise typer.BadParameter("receipt id contains unsafe characters")
    path = _receipt_root() / f"{receipt_id}.json"
    if path.is_symlink() or not path.is_file():
        raise typer.BadParameter(f"uninstall receipt {receipt_id!r} was not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"uninstall receipt is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA:
        raise typer.BadParameter("uninstall receipt schema is invalid")
    return path, payload


def _public_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Hide one-time control material and process fingerprints from status output."""

    hidden = {"nonce", "parent_start", "pyvenv_sha256"}
    return {key: value for key, value in payload.items() if key not in hidden}


def _base_python(target: Path) -> Path:
    candidate = Path(getattr(sys, "_base_executable", "") or "")
    if not candidate.is_file():
        candidate = Path(sys.base_prefix) / "bin" / "python"
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"safe base Python for deferred uninstall is unavailable: {exc}"
        ) from exc
    if _path_inside(resolved, target):
        raise RuntimeError(
            "deferred helper interpreter resolves inside the target venv"
        )
    return resolved


def _new_plan(
    inspection: EnvironmentInspection, receipt_id: str = ""
) -> tuple[Path, dict[str, Any]]:
    target_stat = inspection.target.stat()
    marker = inspection.target / "pyvenv.cfg"
    identifier = receipt_id or f"uninstall-{secrets.token_hex(12)}"
    nonce = secrets.token_urlsafe(24)
    payload = {
        "schema_version": PLAN_SCHEMA,
        "receipt_id": identifier,
        "nonce": nonce,
        "state": "scheduled",
        "created_at": _utc_now(),
        "target": str(inspection.target),
        "target_device": target_stat.st_dev,
        "target_inode": target_stat.st_ino,
        "pyvenv_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        "repo_root": str(inspection.repo_root),
        "parent_pid": os.getpid(),
        "parent_start": _pid_start(os.getpid()),
        "helper": str(Path(__file__).with_name("uninstall_helper.py").resolve()),
        "recovery_command": (
            "npa uninstall --remove-environment --yes --retry " + identifier
        ),
        "error": "",
    }
    path = _receipt_root() / f"{identifier}.json"
    _write_atomic(path, payload)
    return path, payload


def launch_deferred_uninstall(
    plan_path: Path,
    payload: dict[str, Any],
    *,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> int:
    target = Path(str(payload["target"]))
    helper = Path(str(payload["helper"]))
    base_python = _base_python(target)
    process = popen(
        [base_python, helper, plan_path, str(payload["nonce"])],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return int(process.pid)


def uninstall_cmd(
    remove_environment: bool = typer.Option(
        False,
        "--remove-environment",
        help="Opt in to deferred removal of the exact invoking repository-local venv.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm the exact environment-removal plan.",
    ),
    status: str = typer.Option(
        "", "--status", help="Show one uninstall receipt by id and exit."
    ),
    retry: str = typer.Option(
        "", "--retry", help="Retry a failed deferred uninstall receipt by id."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable output."
    ),
) -> None:
    """Dry-run or safely remove only this repository's ``npa/.venv``.

    Ordinary ``npa cleanup`` never removes the invoking environment. Actual
    uninstall requires both --remove-environment and --yes and is performed by a
    one-time helper after this process exits.
    """

    if status:
        _path, payload = _load_receipt(status)
        if output_json:
            typer.echo(json.dumps(_public_receipt(payload), indent=2, sort_keys=True))
        else:
            typer.echo(f"receipt_id: {payload.get('receipt_id')}")
            typer.echo(f"state: {payload.get('state')}")
            typer.echo(f"target: {payload.get('target')}")
            if payload.get("error"):
                typer.echo(f"error: {payload.get('error')}", err=True)
            if payload.get("recovery_command") and payload.get("state") == "failed":
                typer.echo(f"retry: {payload.get('recovery_command')}")
        return
    if retry and (not remove_environment or not yes):
        raise typer.BadParameter("--retry requires both --remove-environment and --yes")
    if retry:
        old_path, old = _load_receipt(retry)
        if old.get("state") == "succeeded":
            typer.echo("The environment was already removed successfully.")
            return
        inspection = inspect_repository_environment(
            executable=Path(str(old.get("target") or "")) / "bin" / "python"
        )
        if not inspection.safe:
            message = "Retry refused: " + "; ".join(inspection.reasons)
            if output_json:
                typer.echo(
                    json.dumps({"outcome": "refused", "message": message}, indent=2)
                )
            else:
                typer.echo(message, err=True)
            raise typer.Exit(code=2)
        plan_path, payload = _new_plan(inspection, receipt_id=retry)
        if plan_path != old_path:
            raise RuntimeError("retry receipt path changed unexpectedly")
    else:
        inspection = inspect_repository_environment()
        if not inspection.safe:
            payload = {"outcome": "refused", **inspection.to_dict()}
            if output_json:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo("Uninstall refused:", err=True)
                for reason in inspection.reasons:
                    typer.echo(f"  - {reason}", err=True)
            raise typer.Exit(code=2)
        if not remove_environment or not yes:
            payload = {
                "outcome": "dry_run",
                "changed": False,
                **inspection.to_dict(),
                "required_flags": ["--remove-environment", "--yes"],
                "message": (
                    "No files were removed. Re-run `npa uninstall "
                    "--remove-environment --yes` to schedule exact venv removal."
                ),
            }
            if output_json:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo(str(payload["message"]))
                typer.echo(f"target: {inspection.target}")
                typer.echo("source/config/credentials/caches: preserved")
            return
        plan_path, payload = _new_plan(inspection)

    response = {
        "outcome": "scheduled",
        "changed": False,
        "receipt_id": payload["receipt_id"],
        "receipt_path": str(plan_path),
        "target": payload["target"],
        "status_command": f"npa uninstall --status {payload['receipt_id']}",
        "message": (
            "Deferred exact-path removal is scheduled after this npa process exits. "
            "Repository source, .git, user data, credentials, and unrelated caches "
            "are outside the plan."
        ),
    }
    if output_json:
        typer.echo(json.dumps(response, indent=2, sort_keys=True))
    else:
        typer.echo(str(response["message"]))
        typer.echo(f"target: {response['target']}")
        typer.echo(f"receipt: {response['receipt_path']}")
        typer.echo(f"verify: {response['status_command']}")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        helper_pid = launch_deferred_uninstall(plan_path, payload)
    except (OSError, RuntimeError) as exc:
        payload["state"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_atomic(plan_path, payload)
        typer.echo(
            f"Deferred uninstall could not start: {exc}. Retry with "
            f"`{payload['recovery_command']}`.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    payload["helper_pid"] = helper_pid
    # Do not rewrite after launch: the helper owns the receipt from this point.

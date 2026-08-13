"""Stdlib-only deferred remover for a validated repository-local NPA venv."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat as stat_module
import sys
import tempfile
import time
from typing import Any


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pid_start(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except OSError:
        return ""
    return fields[21] if len(fields) > 21 else ""


def _wait_for_parent(pid: int, start: str) -> None:
    while pid > 1 and _pid_start(pid) == start:
        time.sleep(0.1)


def _path_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _other_processes_using(target: Path) -> list[str]:
    users: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return users
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        for label in ("exe", "cwd"):
            try:
                resolved = (item / label).resolve(strict=True)
            except OSError:
                continue
            if _path_inside(resolved, target):
                users.append(f"pid {item.name} {label}={resolved}")
        try:
            arguments = (item / "cmdline").read_bytes().split(b"\0")
        except OSError:
            arguments = []
        for raw_argument in arguments:
            argument = Path(os.fsdecode(raw_argument)) if raw_argument else Path()
            if argument.is_absolute() and _path_inside(argument, target):
                users.append(f"pid {item.name} argv={argument}")
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
            users.append(f"pid {item.name} VIRTUAL_ENV={target}")
    return users


def _load_plan(path: Path, nonce: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("uninstall plan is missing or is a symlink")
    if path.stat().st_uid != os.getuid():
        raise RuntimeError("uninstall plan is not owned by the invoking user")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "npa.uninstall.plan.v1"
    ):
        raise RuntimeError("uninstall plan schema is invalid")
    if payload.get("nonce") != nonce:
        raise RuntimeError("uninstall plan nonce does not match")
    return payload


def _validate_target(payload: dict[str, Any]) -> Path:
    target = Path(str(payload.get("target") or ""))
    repo = Path(str(payload.get("repo_root") or ""))
    if target.is_symlink() or repo.is_symlink():
        raise RuntimeError("target or repository became a symlink")
    target = target.resolve(strict=True)
    repo = repo.resolve(strict=True)
    expected = {repo / "npa" / ".venv", repo / ".venv"}
    if target not in expected:
        raise RuntimeError(
            f"target changed: {target} is not one of "
            + ", ".join(str(item) for item in sorted(expected))
        )
    if target in {Path("/"), Path.home(), repo, repo / "npa"}:
        raise RuntimeError("target is a protected broad path")
    stat = target.stat()
    if stat.st_dev != int(payload.get("target_device") or -1) or stat.st_ino != int(
        payload.get("target_inode") or -1
    ):
        raise RuntimeError("target inode/device changed after scheduling")
    marker = target / "pyvenv.cfg"
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("pyvenv.cfg marker is missing or unsafe")
    if _digest(marker) != payload.get("pyvenv_sha256"):
        raise RuntimeError("pyvenv.cfg changed after scheduling")
    if not (repo / ".git").exists() or not (repo / "npa" / "pyproject.toml").is_file():
        raise RuntimeError("repository identity markers changed after scheduling")
    active = _other_processes_using(target)
    if active:
        raise RuntimeError(
            "another process is using the environment: " + "; ".join(active[:5])
        )
    return target


def _remove_exact_target(target: Path, payload: dict[str, Any]) -> None:
    """Remove the validated directory through its parent fd after one last check."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(target.parent, flags)
    try:
        stat = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.st_dev != int(payload.get("target_device") or -1) or stat.st_ino != int(
            payload.get("target_inode") or -1
        ):
            raise RuntimeError("target inode/device changed immediately before removal")
        _remove_tree_at(
            parent_fd,
            target.name,
            expected_device=int(payload.get("target_device") or -1),
            expected_inode=int(payload.get("target_inode") or -1),
        )
    finally:
        os.close(parent_fd)


def _remove_tree_at(
    parent_fd: int,
    name: str,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> None:
    """Descriptor-relative, no-symlink recursive deletion for one directory name."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(directory_fd)
        if (
            expected_device is not None
            and expected_inode is not None
            and (opened.st_dev != expected_device or opened.st_ino != expected_inode)
        ):
            raise RuntimeError(
                "target inode/device changed between final check and directory open"
            )
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                if stat_module.S_ISDIR(entry_stat.st_mode):
                    _remove_tree_at(directory_fd, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)


def run(plan_path: Path, nonce: str) -> int:
    payload: dict[str, Any] = {}
    try:
        payload = _load_plan(plan_path, nonce)
        _wait_for_parent(
            int(payload.get("parent_pid") or 0),
            str(payload.get("parent_start") or ""),
        )
        target = _validate_target(payload)
        payload["state"] = "deleting"
        payload["deletion_started_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _write_atomic(plan_path, payload)
        _remove_exact_target(target, payload)
        if target.exists() or target.is_symlink():
            raise RuntimeError("exact target still exists after removal")
        payload["state"] = "succeeded"
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["error"] = ""
        _write_atomic(plan_path, payload)
        return 0
    except Exception as exc:  # noqa: BLE001 - this helper must preserve evidence
        if payload:
            payload["state"] = "failed"
            payload["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["recovery_command"] = (
                "npa uninstall --remove-environment --yes --retry "
                + str(payload.get("receipt_id") or "")
            )
            try:
                _write_atomic(plan_path, payload)
            except OSError:
                pass
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2]))

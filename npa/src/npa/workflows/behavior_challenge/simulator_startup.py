"""Prepare and qualify a writable Isaac simulator startup before evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_SHA256_LENGTH = 64


@dataclass(frozen=True)
class FileIdentity:
    """Describe one expected regular file.

    Args:
        bytes: Expected byte count.
        sha256: Expected lowercase SHA-256 digest.
    """

    bytes: int
    sha256: str


@dataclass(frozen=True)
class IsaacAppsSpec:
    """Describe a source-bound writable Isaac application view.

    Args:
        isaac_root: Existing retained Isaac package root.
        owner_root: Existing writable run root that owns the new view.
        view_root: Fresh direct child of ``owner_root`` for writable files.
        version_file: Relative source file binding the Isaac installation.
        version: Expected identity of ``version_file``.
        applications: Exact filename-to-identity mapping for ``apps``.
        linked_directories: Source directories linked into the writable view.
        absent_directories: Directory names that must remain absent.
    """

    isaac_root: Path
    owner_root: Path
    view_root: Path
    version_file: str
    version: FileIdentity
    applications: Mapping[str, FileIdentity]
    linked_directories: tuple[str, ...]
    absent_directories: tuple[str, ...] = ()


@dataclass(frozen=True)
class OwnedTreeGuard:
    """Hold the identity of one fresh writable tree.

    Args:
        root: Direct child to remove after startup.
        owner_root: Parent run root that bounds removal.
        device: Device captured before child execution.
        inode: Inode captured before child execution.
        uid: Owner captured before child execution.
    """

    root: Path
    owner_root: Path
    device: int
    inode: int
    uid: int


@dataclass(frozen=True)
class SimulatorStartupSpec:
    """Describe one policy-free simulator startup child.

    Args:
        apps: Source-bound writable Isaac application view.
        appdata_root: Fresh writable app-data child of ``apps.owner_root``.
        command: Explicit child argv; shell execution is never used.
        marker_path: Fresh child startup-marker path.
        shutdown_request_path: Fresh child shutdown-request path.
        log_path: Fresh combined stdout/stderr path.
        environment: Additional child environment values.
        evaluation_context: Paths used by the immediately following evaluator.
        appdata_environment_variable: Variable receiving ``appdata_root``.
    """

    apps: IsaacAppsSpec
    appdata_root: Path
    command: Sequence[str]
    marker_path: Path
    shutdown_request_path: Path
    log_path: Path
    environment: Mapping[str, str]
    evaluation_context: Mapping[str, str]
    appdata_environment_variable: str = "OMNIGIBSON_APPDATA_PATH"


class OwnedTreeCleanupError(RuntimeError):
    """Report bounded cleanup failure evidence.

    Args:
        message: Human-readable failure description.
        evidence: Safe partial inventory and failure details.
    """

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _decode_identity(value: object) -> FileIdentity:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        raise ValueError("file identity fields differ")
    if type(value["bytes"]) is not int or not isinstance(value["sha256"], str):
        raise ValueError("file identity types differ")
    identity = FileIdentity(value["bytes"], value["sha256"])
    _expected(identity)
    return identity


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{label} keys and values must be strings")
    return dict(value)


def _decode_apps(value: object) -> IsaacAppsSpec:
    keys = {
        "isaac_root",
        "owner_root",
        "view_root",
        "version_file",
        "version",
        "applications",
        "linked_directories",
        "absent_directories",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Isaac apps specification fields differ")
    applications = value["applications"]
    if not isinstance(applications, dict):
        raise ValueError("Isaac application identities must be an object")
    return IsaacAppsSpec(
        isaac_root=Path(value["isaac_root"]),
        owner_root=Path(value["owner_root"]),
        view_root=Path(value["view_root"]),
        version_file=value["version_file"],
        version=_decode_identity(value["version"]),
        applications={
            key: _decode_identity(item) for key, item in applications.items()
        },
        linked_directories=tuple(value["linked_directories"]),
        absent_directories=tuple(value["absent_directories"]),
    )


def load_simulator_startup_spec(path: Path) -> SimulatorStartupSpec:
    """Load an exact JSON startup specification for the public CLI.

    Args:
        path: Nonsymlink regular JSON specification.
    Returns:
        Validated source, process, and evidence paths.
    Raises:
        OSError: The specification cannot be read.
        ValueError: Its schema, fields, or types differ.
    """
    _identity(path)
    value = json.loads(path.read_text())
    required = {
        "schema",
        "apps",
        "appdata_root",
        "command",
        "marker_path",
        "shutdown_request_path",
        "log_path",
        "environment",
        "evaluation_context",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("simulator startup specification fields differ")
    if value["schema"] != "npa.workbench.simulator-startup-spec.v1":
        raise ValueError("simulator startup specification schema differs")
    command = value["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("simulator startup command differs")
    return _decoded_spec(value, command)


def _decoded_spec(value: dict[str, object], command: list[str]) -> SimulatorStartupSpec:
    return SimulatorStartupSpec(
        apps=_decode_apps(value["apps"]),
        appdata_root=Path(value["appdata_root"]),
        command=tuple(command),
        marker_path=Path(value["marker_path"]),
        shutdown_request_path=Path(value["shutdown_request_path"]),
        log_path=Path(value["log_path"]),
        environment=_string_mapping(value["environment"], "startup environment"),
        evaluation_context=_string_mapping(
            value["evaluation_context"], "evaluation context"
        ),
    )


def build_evaluation_context(
    upstream_root: Path, evaluator_python: Path, data_root: Path, owner_root: Path
) -> dict[str, object]:
    """Bind startup evidence to the next evaluator's local inputs.

    Args:
        upstream_root: Pinned evaluator source root.
        evaluator_python: Simulator interpreter or executable.
        data_root: Authorized simulator data root.
        owner_root: Fresh worker workspace owning writable views.
    Returns:
        Resolved paths and exact evaluator executable identity.
    Raises:
        OSError: A required path cannot be resolved or read.
        ValueError: A required directory or executable differs.
    """
    directories = (upstream_root, data_root, owner_root)
    if any(path.is_symlink() or not path.is_dir() for path in directories):
        raise ValueError("simulator evaluation context directory differs")
    executable = evaluator_python.resolve(strict=True)
    declared = evaluator_python.absolute()
    return {
        "upstream_root": str(upstream_root.resolve(strict=True)),
        "upstream_revision": _source_revision(upstream_root),
        "evaluator_python": {
            "path": str(declared),
            "resolved_path": str(executable),
            **_identity(executable),
        },
        "startup_source_root": str(_public_source_root()),
        "startup_sources": _startup_source_rows(),
        "data_root": str(data_root.resolve(strict=True)),
        "owner_root": str(owner_root.resolve(strict=True)),
    }


def _startup_source_rows() -> list[dict[str, object]]:
    root = _public_source_root()
    return [
        {"path": str(path), **_identity(path)}
        for path in (
            root / "npa/__init__.py",
            root / "npa/workflows/__init__.py",
            root / "npa/workflows/behavior_challenge/__init__.py",
            root / "npa/workflows/behavior_challenge/__main__.py",
            root / "npa/workflows/behavior_challenge/simulator_startup.py",
        )
    ]


def _public_source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_revision(root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if changed.strip() or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("evaluator source identity differs")
    untracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            "OmniGibson/omnigibson",
        ],
        text=True,
    )
    if any(not _ignored_python_cache(row) for row in untracked.splitlines()):
        raise ValueError("untracked OmniGibson import source exists")
    return revision


def _ignored_python_cache(status_row: str) -> bool:
    path = status_row[3:]
    return status_row[:3] in {"?? ", "!! "} and (
        "/__pycache__/" in path or path.endswith((".pyc", ".pyo"))
    )


def _spec_evaluation_context(spec: SimulatorStartupSpec) -> dict[str, object]:
    required = {"upstream_root", "evaluator_python", "data_root"}
    if set(spec.evaluation_context) != required:
        raise ValueError("evaluation context fields differ")
    context = build_evaluation_context(
        Path(spec.evaluation_context["upstream_root"]),
        Path(spec.evaluation_context["evaluator_python"]),
        Path(spec.evaluation_context["data_root"]),
        spec.apps.owner_root,
    )
    if list(spec.command) != _startup_child_argv(spec, context):
        raise ValueError("startup child command differs from the built-in probe")
    return context


def _startup_child_argv(
    spec: SimulatorStartupSpec, context: dict[str, object]
) -> list[str]:
    return [
        str(context["evaluator_python"]["path"]),
        "-m",
        "npa.workflows.behavior_challenge",
        "simulator-startup-child",
        "--upstream-root",
        str(context["upstream_root"]),
        "--marker-path",
        str(spec.marker_path.resolve()),
        "--shutdown-request-path",
        str(spec.shutdown_request_path.resolve()),
    ]


def write_simulator_startup_receipt(
    spec_path: Path, output_path: Path
) -> dict[str, object]:
    """Run a startup specification and write its canonical local receipt.

    Args:
        spec_path: Exact JSON specification accepted by the public CLI.
        output_path: Fresh nonsymlink receipt destination.
    Returns:
        The same validated receipt written to ``output_path``.
    Raises:
        OSError: Startup or exclusive receipt creation fails.
        ValueError: Specification, startup, or output bounds differ.
    """
    spec = load_simulator_startup_spec(spec_path)
    _direct_fresh_child(output_path, spec.apps.owner_root, "startup receipt")
    try:
        receipt = run_simulator_startup(spec)
    except BaseException as error:
        _write_startup_failure(spec, output_path, error)
        raise
    receipt["spec_source"] = {"path": str(spec_path.resolve()), **_identity(spec_path)}
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    with output_path.open("xb") as stream:
        stream.write(raw)
    validate_simulator_startup_receipt(output_path, expected_spec_path=spec_path)
    return receipt


def _write_startup_failure(
    spec: SimulatorStartupSpec, output_path: Path, error: BaseException
) -> None:
    failure_path = output_path.with_name(f"{output_path.stem}-failure.json")
    evidence = {}
    for name, path in _startup_output_paths(spec).items():
        if path.is_file() and not path.is_symlink():
            evidence[name] = {"path": str(path.resolve()), **_identity(path)}
    value = {
        "schema": "npa.workbench.simulator-startup-failure.v1",
        "status": "startup_or_owned_cleanup_failed",
        "success_receipt_written": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "evidence": evidence,
    }
    _exclusive_json(failure_path, value)


def _identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file differs: {path}")
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _expected(identity: FileIdentity) -> dict[str, object]:
    if identity.bytes < 0 or not re.fullmatch(r"[0-9a-f]{64}", identity.sha256):
        raise ValueError("file identity is invalid")
    return {"bytes": identity.bytes, "sha256": identity.sha256}


def _direct_fresh_child(path: Path, owner_root: Path, label: str) -> Path:
    owner = owner_root.resolve(strict=True)
    if path.parent.resolve(strict=True) != owner:
        raise ValueError(f"{label} must be a direct child of its owner root")
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be fresh")
    return owner


def _component(name: str, label: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be one relative path component")
    return name


def _source_apps(spec: IsaacAppsSpec) -> list[dict[str, object]]:
    apps_root = spec.isaac_root / "apps"
    if apps_root.is_symlink() or not apps_root.is_dir():
        raise ValueError("Isaac source apps is not a regular directory")
    names = sorted(path.name for path in apps_root.iterdir())
    if names != sorted(spec.applications):
        raise ValueError("Isaac source application inventory differs")
    rows = []
    for name in names:
        _component(name, "Isaac application")
        observed = _identity(apps_root / name)
        if observed != _expected(spec.applications[name]):
            raise ValueError(f"Isaac source application differs: {name}")
        rows.append({"path": name, **observed})
    return rows


def _copy_apps(spec: IsaacAppsSpec, rows: list[dict[str, object]]) -> Path:
    spec.view_root.mkdir(mode=0o700, exist_ok=False)
    apps = spec.view_root / "apps"
    apps.mkdir(mode=0o700)
    for row in rows:
        source = spec.isaac_root / "apps" / str(row["path"])
        target = apps / str(row["path"])
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
        target.chmod(0o600)
        if _identity(target) != {key: row[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"writable Isaac application differs: {row['path']}")
    return apps


def _source_links(spec: IsaacAppsSpec) -> list[dict[str, str]]:
    links = []
    for name in spec.linked_directories:
        _component(name, "Isaac extension directory")
        source = spec.isaac_root / name
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"Isaac extension directory differs: {name}")
        links.append({"path": name, "target": str(source.resolve())})
    for name in spec.absent_directories:
        _component(name, "absent Isaac directory")
        if (spec.isaac_root / name).exists():
            raise ValueError(f"unexpected Isaac extension directory exists: {name}")
    return links


def _apps_claim(spec: IsaacAppsSpec) -> dict[str, object]:
    version_path = _contained_source_file(
        spec.isaac_root, spec.version_file, "Isaac version file"
    )
    version = _identity(version_path)
    if version != _expected(spec.version):
        raise ValueError("Isaac version identity differs")
    return {
        "schema": "npa.workbench.writable-isaac-apps.v1",
        "source_root": str(spec.isaac_root.resolve()),
        "version_file": spec.version_file,
        "source_version": version,
        "applications": _source_apps(spec),
        "linked_directories": _source_links(spec),
        "absent_directories": list(spec.absent_directories),
        "exp_path": str((spec.view_root / "apps").resolve()),
    }


def _link_extensions(spec: IsaacAppsSpec, links: list[dict[str, str]]) -> None:
    for row in links:
        (spec.view_root / row["path"]).symlink_to(
            row["target"], target_is_directory=True
        )
    for name in spec.absent_directories:
        if (spec.view_root / name).exists():
            raise ValueError(f"unexpected Isaac extension directory exists: {name}")


def prepare_writable_isaac_apps(spec: IsaacAppsSpec) -> dict[str, object]:
    """Create a source-bound writable Isaac apps view.

    Args:
        spec: Exact source identities, directories, and fresh writable paths.
    Returns:
        Receipt describing copied applications, links, and writable ``EXP_PATH``.
    Raises:
        OSError: A source or destination filesystem operation fails.
        ValueError: A path, source identity, inventory, or view differs.
    """
    _direct_fresh_child(spec.view_root, spec.owner_root, "Isaac apps view")
    claim = _apps_claim(spec)
    _copy_apps(spec, claim["applications"])
    _link_extensions(spec, claim["linked_directories"])
    return claim


def _contained_source_file(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must remain below the source root")
    current = root.resolve(strict=True)
    for index, component in enumerate(path.parts):
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink")
        if index < len(path.parts) - 1 and not current.is_dir():
            raise ValueError(f"{label} parent differs")
    return current


def hold_owned_tree(root: Path, owner_root: Path) -> OwnedTreeGuard:
    """Capture a fresh owned directory before an external child can modify it.

    Args:
        root: Existing direct child to guard.
        owner_root: Existing parent run root that bounds later removal.
    Returns:
        Immutable device, inode, and UID guard.
    Raises:
        OSError: Filesystem inspection fails.
        ValueError: The root is not an owned direct-child directory.
    """
    owner = owner_root.resolve(strict=True)
    if root.parent.resolve(strict=True) != owner or root.is_symlink():
        raise ValueError("owned tree must be a nonsymlink direct child")
    observed = root.lstat()
    if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.geteuid():
        raise ValueError("owned tree directory or UID differs")
    return OwnedTreeGuard(
        owner / root.name, owner, observed.st_dev, observed.st_ino, observed.st_uid
    )


def _kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _guard_dict(observed: os.stat_result) -> dict[str, int]:
    return {"device": observed.st_dev, "inode": observed.st_ino, "uid": observed.st_uid}


def _check_member(observed: os.stat_result, guard: OwnedTreeGuard, path: str) -> str:
    kind = _kind(observed.st_mode)
    if observed.st_uid != guard.uid:
        raise PermissionError(f"owned cleanup UID differs: {path}")
    if observed.st_dev != guard.device:
        raise ValueError(f"owned cleanup device differs: {path}")
    if kind == "special":
        raise ValueError(f"owned cleanup special member: {path}")
    return kind


def _repair_directory(
    parent_fd: int, name: str, observed: os.stat_result, repaired: int
) -> None:
    if os.chmod in os.supports_follow_symlinks:
        os.chmod(name, repaired, dir_fd=parent_fd, follow_symlinks=False)
        return
    if not hasattr(os, "O_PATH") or not Path("/proc/self/fd").is_dir():
        raise RuntimeError("no fd-safe directory chmod is available")
    flags = os.O_PATH | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    held_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        if _guard_dict(os.fstat(held_fd)) != _guard_dict(observed):
            raise ValueError("owned cleanup directory changed before chmod")
        os.chmod(f"/proc/self/fd/{held_fd}", repaired)
    finally:
        os.close(held_fd)


def _inventory_row(path: str, observed: os.stat_result, kind: str) -> dict[str, object]:
    return {
        "path": path,
        "type": kind,
        "uid": observed.st_uid,
        "mode": oct(stat.S_IMODE(observed.st_mode)),
        **_guard_dict(observed),
    }


def _scan_owned(
    parent_fd: int, name: str, relative: str, state: dict, held: bool = False
) -> None:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    kind = _check_member(observed, state["_held_guard"], relative)
    if held and (_guard_dict(observed) != state["guard"] or kind != "directory"):
        raise ValueError("owned cleanup root identity changed")
    state["inventory"].append(_inventory_row(relative, observed, kind))
    if kind != "directory":
        return
    needed = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    if stat.S_IMODE(observed.st_mode) & needed != needed:
        repaired = stat.S_IMODE(observed.st_mode) | needed
        _repair_directory(parent_fd, name, observed, repaired)
        state["repairs"].append(
            {
                "path": relative,
                "type": kind,
                "uid": observed.st_uid,
                "mode_before": oct(stat.S_IMODE(observed.st_mode)),
                "mode_after": oct(repaired),
                "method": "owner-user-rwx-only",
            }
        )
    child_fd = _open_directory(parent_fd, name, observed)
    try:
        for child in sorted(os.listdir(child_fd)):
            _scan_owned(child_fd, child, f"{relative}/{child}", state)
    finally:
        os.close(child_fd)


def _open_directory(parent_fd: int, name: str, observed: os.stat_result) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    if _guard_dict(os.fstat(child_fd)) != _guard_dict(observed):
        os.close(child_fd)
        raise ValueError("owned cleanup directory changed")
    return child_fd


def _direct_children(relative: str, inventory: list[dict[str, object]]) -> list[str]:
    return sorted(
        str(row["path"]).rsplit("/", 1)[-1]
        for row in inventory
        if row["path"] != relative and str(row["path"]).rsplit("/", 1)[0] == relative
    )


def _delete_owned(parent_fd: int, name: str, relative: str, state: dict) -> None:
    row = next(item for item in state["inventory"] if item["path"] == relative)
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if {**_guard_dict(observed), "type": _kind(observed.st_mode)} != {
        key: row[key] for key in ("device", "inode", "uid", "type")
    }:
        raise ValueError(f"owned cleanup member changed: {relative}")
    if row["type"] != "directory":
        os.unlink(name, dir_fd=parent_fd)
        return
    child_fd = _open_directory(parent_fd, name, observed)
    try:
        children = sorted(os.listdir(child_fd))
        if children != _direct_children(relative, state["inventory"]):
            raise ValueError(f"owned cleanup inventory changed: {relative}")
        for child in children:
            _delete_owned(child_fd, child, f"{relative}/{child}", state)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _inventory_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    keys = ("path", "type", "uid", "mode", "device", "inode")
    stable = [{key: row[key] for key in keys} for row in rows]
    raw = (json.dumps(stable, separators=(",", ":"), sort_keys=True) + "\n").encode()
    kinds = ("directory", "regular", "symlink")
    counts = {kind: sum(row["type"] == kind for row in rows) for kind in kinds}
    return {
        "members": stable,
        "counts": counts,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def remove_owned_tree(guard: OwnedTreeGuard) -> dict[str, object]:
    """Remove one held tree without following links or crossing ownership bounds.

    Args:
        guard: Identity captured by :func:`hold_owned_tree` before child execution.
    Returns:
        Derived pre-removal inventory, permission repairs, and removal status.
    Raises:
        OwnedTreeCleanupError: Validation or removal fails; evidence is attached.
    """
    evidence = _cleanup_evidence(guard)
    evidence["_held_guard"] = guard
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        owner_fd = os.open(guard.owner_root, flags)
        try:
            _scan_owned(owner_fd, guard.root.name, guard.root.name, evidence, True)
            _delete_owned(owner_fd, guard.root.name, guard.root.name, evidence)
        finally:
            os.close(owner_fd)
    except BaseException as error:
        evidence.pop("_held_guard")
        evidence["inventory"] = _inventory_summary(evidence["inventory"])
        evidence["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise OwnedTreeCleanupError(
            "bounded owned-tree cleanup failed", evidence
        ) from error
    evidence.pop("_held_guard")
    evidence["inventory"] = _inventory_summary(evidence.pop("inventory"))
    evidence["status"] = "owned_tree_removed_without_symlink_follow"
    return evidence


def _cleanup_evidence(guard: OwnedTreeGuard) -> dict[str, object]:
    return {
        "schema": "npa.workbench.owned-tree-cleanup.v1",
        "root": str(guard.root.absolute()),
        "guard": {"device": guard.device, "inode": guard.inode, "uid": guard.uid},
        "method": "fd-relative-lstat-no-follow-owned-postorder",
        "inventory": [],
        "repairs": [],
    }


def _marker(path: Path, pid: int) -> dict[str, object]:
    _identity(path)
    value = json.loads(path.read_text())
    expected = {
        "schema": "npa.workbench.simulator-startup-marker.v1",
        "status": "simulator_started_without_scene_policy_or_case",
        "pid": pid,
        "application_started": True,
        "simulator_started": True,
        "scene_count": 0,
        "model_loaded": False,
        "policy_started": False,
        "case_started": False,
    }
    if value != expected:
        raise ValueError("simulator startup marker differs")
    return value


def _shutdown_request(path: Path, pid: int, marker_path: Path) -> dict[str, object]:
    _identity(path)
    value = json.loads(path.read_text())
    expected = {
        "schema": "npa.workbench.simulator-shutdown-request.v1",
        "status": "shutdown_requested_after_verified_startup",
        "pid": pid,
        "startup_marker": _identity(marker_path),
        "model_loaded": False,
        "policy_started": False,
        "case_started": False,
    }
    if value != expected:
        raise ValueError("simulator shutdown request differs")
    return value


def _startup_environment(
    spec: SimulatorStartupSpec, apps: dict[str, object]
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(spec.environment)
    source = Path(spec.evaluation_context["upstream_root"]) / "OmniGibson"
    public_source = _public_source_root()
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source), str(public_source), inherited)
    )
    environment["EXP_PATH"] = str(apps["exp_path"])
    environment["ISAAC_PATH"] = str(spec.apps.isaac_root.resolve())
    environment["OMNIGIBSON_DATA_PATH"] = str(
        Path(spec.evaluation_context["data_root"]).resolve(strict=True)
    )
    environment["OMNIGIBSON_HEADLESS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[spec.appdata_environment_variable] = str(spec.appdata_root.resolve())
    return environment


def run_simulator_startup_child(
    upstream_root: Path, marker_path: Path, shutdown_request_path: Path
) -> None:
    """Launch pinned OmniGibson empty, mark success, then request shutdown.

    Args:
        upstream_root: Evaluator source root containing ``OmniGibson``.
        marker_path: Fresh startup marker written after real launch checks.
        shutdown_request_path: Fresh marker written immediately before shutdown.
    Returns:
        None. Pinned fast shutdown may terminate the interpreter directly.
    Raises:
        BaseException: Import, launch, validation, marker, or shutdown fails.
    """
    import omnigibson as og

    expected = upstream_root.resolve() / "OmniGibson/omnigibson/__init__.py"
    if Path(og.__file__).resolve() != expected:
        raise ValueError("startup child imported another OmniGibson source")
    og.launch()
    if og.app is None or og.sim is None or list(og.sim.scenes):
        raise ValueError("OmniGibson empty-scene startup differs")
    marker = _child_marker(os.getpid())
    _exclusive_json(marker_path, marker)
    request = _child_shutdown_request(os.getpid(), marker_path)
    _exclusive_json(shutdown_request_path, request)
    og.shutdown()


def _child_marker(pid: int) -> dict[str, object]:
    return {
        "schema": "npa.workbench.simulator-startup-marker.v1",
        "status": "simulator_started_without_scene_policy_or_case",
        "pid": pid,
        "application_started": True,
        "simulator_started": True,
        "scene_count": 0,
        "model_loaded": False,
        "policy_started": False,
        "case_started": False,
    }


def _child_shutdown_request(pid: int, marker_path: Path) -> dict[str, object]:
    return _child_shutdown_request_from_identity(pid, _identity(marker_path))


def _exclusive_json(path: Path, value: dict[str, object]) -> None:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)


def _fresh_startup_outputs(spec: SimulatorStartupSpec) -> None:
    for path in (spec.marker_path, spec.shutdown_request_path, spec.log_path):
        if path.parent.resolve(strict=True) != spec.apps.owner_root.resolve(
            strict=True
        ):
            raise ValueError("startup output must be a direct run-root child")
        if path.exists() or path.is_symlink():
            raise ValueError("startup output must be fresh")


def run_simulator_startup(spec: SimulatorStartupSpec) -> dict[str, object]:
    """Run one real empty-scene startup before any model, policy, or case.

    Args:
        spec: Source-bound paths, explicit child argv, marker paths, and environment.
    Returns:
        Receipt joining exact markers, child exit, derived inventories, and cleanup.
    Raises:
        OSError: Process or filesystem setup fails.
        subprocess.SubprocessError: The startup child cannot be launched.
        ValueError: Startup, marker, source, child-exit, or cleanup evidence differs.
    """
    _fresh_startup_outputs(spec)
    _spec_evaluation_context(spec)
    _direct_fresh_child(spec.apps.view_root, spec.apps.owner_root, "Isaac apps view")
    _direct_fresh_child(spec.appdata_root, spec.apps.owner_root, "simulator appdata")
    apps = prepare_writable_isaac_apps(spec.apps)
    spec.appdata_root.mkdir(mode=0o700, exist_ok=False)
    view_guard = hold_owned_tree(spec.apps.view_root, spec.apps.owner_root)
    appdata_guard = hold_owned_tree(spec.appdata_root, spec.apps.owner_root)
    return _run_and_cleanup(spec, apps, view_guard, appdata_guard)


def _run_and_cleanup(
    spec: SimulatorStartupSpec,
    apps: dict[str, object],
    view_guard: OwnedTreeGuard,
    appdata_guard: OwnedTreeGuard,
) -> dict[str, object]:
    with spec.log_path.open("xb") as stream:
        process = subprocess.Popen(
            list(spec.command),
            env=_startup_environment(spec, apps),
            cwd=_public_source_root(),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
    if returncode != 0 or process.poll() != returncode:
        raise ValueError("simulator startup child exit differs")
    marker = _marker(spec.marker_path, process.pid)
    request = _shutdown_request(
        spec.shutdown_request_path, process.pid, spec.marker_path
    )
    cleanup = {
        "appdata": remove_owned_tree(appdata_guard),
        "view": remove_owned_tree(view_guard),
    }
    return _startup_receipt(spec, apps, marker, request, cleanup)


def _startup_receipt(
    spec: SimulatorStartupSpec,
    apps: dict[str, object],
    marker: dict[str, object],
    request: dict[str, object],
    cleanup: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "npa.workbench.simulator-startup.v1",
        "status": "empty_scene_startup_and_owned_cleanup_verified",
        "command": list(spec.command),
        "apps": apps,
        "appdata_root": str(spec.appdata_root.resolve()),
        "environment": dict(spec.environment),
        "evaluation_context": _spec_evaluation_context(spec),
        "startup_marker": marker,
        "shutdown_request": request,
        "child": {"returncode": 0, "pid_gone": True},
        "evidence": _startup_evidence(spec),
        "cleanup": cleanup,
        "model_loaded": False,
        "policy_started": False,
        "case_started": False,
    }


def _startup_evidence(spec: SimulatorStartupSpec) -> dict[str, object]:
    return {
        name: {"path": str(path.resolve()), **_identity(path)}
        for name, path in _startup_output_paths(spec).items()
    }


def _startup_output_paths(spec: SimulatorStartupSpec) -> dict[str, Path]:
    return {
        "marker": spec.marker_path,
        "shutdown_request": spec.shutdown_request_path,
        "log": spec.log_path,
    }


def validate_simulator_startup_receipt(
    path: Path,
    expected_context: dict[str, object] | None = None,
    expected_spec_path: Path | None = None,
) -> dict[str, object]:
    """Validate a local startup receipt before policy or case preparation.
    Args:
        path: Nonsymlink regular JSON receipt produced by this module.
        expected_context: Exact next-evaluator binding when used for admission.
        expected_spec_path: Exact startup specification when resuming its output.
    Returns:
        Parsed receipt after checking startup and cleanup claims.
    Raises:
        OSError: The receipt cannot be read.
        ValueError: Receipt schema, scope, or cleanup evidence differs.
    """
    _identity(path)
    value = json.loads(path.read_text())
    _validate_receipt_header(value)
    _validate_bound_spec(value, expected_spec_path)
    _validate_stored_context(value.get("evaluation_context"))
    _validate_current_sources(value)
    _validate_cleanup_receipts(value)
    _validate_evidence_files(value)
    _validate_embedded_startup(value)
    if (
        expected_context is not None
        and value.get("evaluation_context") != expected_context
    ):
        raise ValueError("simulator startup evaluator context differs")
    return value


def _validate_receipt_header(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("simulator startup receipt differs")
    cleanup = value.get("cleanup")
    flags = ("model_loaded", "policy_started", "case_started")
    if (
        value.get("schema") != "npa.workbench.simulator-startup.v1"
        or value.get("status") != "empty_scene_startup_and_owned_cleanup_verified"
        or any(value.get(key) is not False for key in flags)
        or value.get("child") != {"returncode": 0, "pid_gone": True}
        or not isinstance(cleanup, dict)
        or set(cleanup) != {"appdata", "view"}
        or any(not isinstance(row, dict) for row in cleanup.values())
    ):
        raise ValueError("simulator startup receipt differs")


def _stored_spec_path(value: dict[str, object], expected_path: Path | None) -> Path:
    row = value.get("spec_source")
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError("simulator startup specification identity differs")
    path = Path(row["path"])
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ValueError("simulator startup specification identity differs")
    if row != {"path": str(path.resolve()), **_identity(path)}:
        raise ValueError("simulator startup specification identity differs")
    return path


def _validate_bound_spec(value: dict[str, object], expected_path: Path | None) -> None:
    spec = load_simulator_startup_spec(_stored_spec_path(value, expected_path))
    evidence = value.get("evidence", {})
    paths = _startup_output_paths(spec)
    expected_evidence_paths = {
        name: str(path.resolve()) for name, path in paths.items()
    }
    observed_evidence_paths = (
        {name: row.get("path") for name, row in evidence.items()}
        if isinstance(evidence, dict)
        else {}
    )
    if (
        value.get("apps") != _apps_claim(spec.apps)
        or value.get("evaluation_context") != _spec_evaluation_context(spec)
        or value.get("command") != list(spec.command)
        or value.get("environment") != dict(spec.environment)
        or value.get("appdata_root") != str(spec.appdata_root.resolve())
        or observed_evidence_paths != expected_evidence_paths
    ):
        raise ValueError("simulator startup receipt differs from its specification")


def _validate_current_sources(value: dict[str, object]) -> None:
    apps = value.get("apps")
    keys = {
        "schema",
        "source_root",
        "version_file",
        "source_version",
        "applications",
        "linked_directories",
        "absent_directories",
        "exp_path",
    }
    if not isinstance(apps, dict) or set(apps) != keys:
        raise ValueError("writable Isaac apps receipt differs")
    if apps["schema"] != "npa.workbench.writable-isaac-apps.v1":
        raise ValueError("writable Isaac apps receipt differs")
    owner = Path(value["evaluation_context"]["owner_root"])
    upstream = Path(value["evaluation_context"]["upstream_root"])
    if _source_revision(upstream) != value["evaluation_context"]["upstream_revision"]:
        raise ValueError("evaluator source changed after startup")
    exp_path = Path(apps["exp_path"])
    if exp_path.name != "apps" or exp_path.parent.parent != owner:
        raise ValueError("writable Isaac apps view binding differs")
    root = Path(apps["source_root"])
    version = _contained_source_file(root, apps["version_file"], "Isaac version file")
    if _identity(version) != apps["source_version"]:
        raise ValueError("Isaac source version changed after startup")
    rows = apps["applications"]
    names = sorted(path.name for path in (root / "apps").iterdir())
    if not isinstance(rows, list) or names != [row.get("path") for row in rows]:
        raise ValueError("Isaac source application inventory changed after startup")
    for row in rows:
        if _identity(root / "apps" / row["path"]) != _row_identity(row):
            raise ValueError("Isaac source application changed after startup")
    _validate_source_directories(root, apps)


def _validate_source_directories(root: Path, apps: dict[str, object]) -> None:
    links = apps["linked_directories"]
    if not isinstance(links, list):
        raise ValueError("Isaac linked-directory evidence differs")
    for row in links:
        source = root / row["path"]
        if not source.is_dir() or source.is_symlink():
            raise ValueError("Isaac linked directory changed after startup")
        if row != {"path": row["path"], "target": str(source.resolve())}:
            raise ValueError("Isaac linked-directory evidence differs")
    if any((root / name).exists() for name in apps["absent_directories"]):
        raise ValueError("absent Isaac directory appeared after startup")


def _row_identity(row: dict[str, object]) -> dict[str, object]:
    return {key: row[key] for key in ("bytes", "sha256")}


def _validate_evidence_files(value: dict[str, object]) -> None:
    evidence = value.get("evidence")
    names = {"marker", "shutdown_request", "log"}
    if not isinstance(evidence, dict) or set(evidence) != names:
        raise ValueError("simulator startup evidence differs")
    owner = Path(value["evaluation_context"]["owner_root"])
    for name, row in evidence.items():
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("simulator startup evidence differs")
        evidence_path = Path(row["path"])
        if evidence_path.parent != owner:
            raise ValueError("simulator startup evidence path differs")
        if _identity(evidence_path) != _row_identity(row):
            raise ValueError(f"simulator startup {name} changed")
    marker = json.loads(Path(evidence["marker"]["path"]).read_text())
    request = json.loads(Path(evidence["shutdown_request"]["path"]).read_text())
    if marker != value["startup_marker"] or request != value["shutdown_request"]:
        raise ValueError("simulator startup evidence content differs")


def _apps_from_receipt(
    receipt: dict[str, object], owner_root: Path, view_root: Path
) -> IsaacAppsSpec:
    value = receipt["apps"]
    applications = {
        row["path"]: FileIdentity(row["bytes"], row["sha256"])
        for row in value["applications"]
    }
    return IsaacAppsSpec(
        isaac_root=Path(value["source_root"]),
        owner_root=owner_root,
        view_root=view_root,
        version_file=value["version_file"],
        version=FileIdentity(**value["source_version"]),
        applications=applications,
        linked_directories=tuple(row["path"] for row in value["linked_directories"]),
        absent_directories=tuple(value["absent_directories"]),
    )


@contextmanager
def prepared_evaluator_environment(receipt: dict[str, object], owner_root: Path):
    """Keep a separate writable apps/app-data view for real evaluation.

    Args:
        receipt: Validated startup receipt bound to this worker.
        owner_root: Worker workspace owning the fresh evaluator trees.
    Yields:
        Environment values for the evaluator subprocess.
    Raises:
        OSError: Preparation, evidence writing, or cleanup fails.
        ValueError: Source identity or fresh path bounds differ.
        OwnedTreeCleanupError: Safe cleanup cannot prove its bounds.
    """
    owner = owner_root.resolve(strict=True)
    view, appdata, success, failure = _fresh_evaluator_attempt(owner)
    apps = prepare_writable_isaac_apps(_apps_from_receipt(receipt, owner, view))
    appdata.mkdir(mode=0o700, exist_ok=False)
    guards = hold_owned_tree(view, owner), hold_owned_tree(appdata, owner)
    environment = _evaluator_environment(receipt, apps, appdata)
    try:
        yield environment
    finally:
        _finish_evaluator_cleanup(success, failure, guards)


def _fresh_evaluator_attempt(owner: Path) -> tuple[Path, Path, Path, Path]:
    for index in itertools.count():
        suffix = f"{index:04d}"
        paths = (
            owner / f"simulator-evaluator-view-{suffix}",
            owner / f"simulator-appdata-{suffix}",
            owner / f"simulator-evaluator-cleanup-{suffix}.json",
            owner / f"simulator-evaluator-cleanup-failure-{suffix}.json",
        )
        if not any(path.exists() or path.is_symlink() for path in paths):
            return paths


def _evaluator_environment(
    receipt: dict[str, object], apps: dict[str, object], appdata: Path
) -> dict[str, str]:
    return {
        "EXP_PATH": str(apps["exp_path"]),
        "ISAAC_PATH": str(receipt["apps"]["source_root"]),
        "OMNIGIBSON_APPDATA_PATH": str(appdata.resolve()),
        "OMNIGIBSON_DATA_PATH": str(receipt["evaluation_context"]["data_root"]),
        "OMNIGIBSON_HEADLESS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _write_evaluator_cleanup(path: Path, cleanup: dict[str, object]) -> None:
    value = {
        "schema": "npa.workbench.simulator-evaluator-cleanup.v1",
        "status": "evaluator_owned_trees_removed",
        "cleanup": cleanup,
    }
    _exclusive_json(path, value)


def _finish_evaluator_cleanup(
    success: Path, failure: Path, guards: tuple[OwnedTreeGuard, OwnedTreeGuard]
) -> None:
    cleanup, failures = {}, {}
    for name, guard in (("appdata", guards[1]), ("view", guards[0])):
        try:
            cleanup[name] = remove_owned_tree(guard)
        except OwnedTreeCleanupError as error:
            failures[name] = error.evidence
    if failures:
        value = {
            "schema": "npa.workbench.simulator-evaluator-cleanup-failure.v1",
            "status": "evaluator_owned_tree_cleanup_failed",
            "cleanup": cleanup,
            "failures": failures,
        }
        _exclusive_json(failure, value)
        raise OwnedTreeCleanupError("evaluator cleanup failed", value)
    _write_evaluator_cleanup(success, cleanup)


def _validate_stored_context(value: object) -> None:
    keys = {
        "upstream_root",
        "upstream_revision",
        "evaluator_python",
        "startup_source_root",
        "startup_sources",
        "data_root",
        "owner_root",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("simulator startup evaluator context differs")
    for key in ("upstream_root", "startup_source_root", "data_root", "owner_root"):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError("simulator startup evaluator context differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value["upstream_revision"])):
        raise ValueError("simulator startup evaluator context differs")
    executable = value["evaluator_python"]
    executable_keys = {"path", "resolved_path", "bytes", "sha256"}
    if not isinstance(executable, dict) or set(executable) != executable_keys:
        raise ValueError("simulator startup evaluator context differs")
    for key in ("path", "resolved_path"):
        if (
            not isinstance(executable[key], str)
            or not Path(executable[key]).is_absolute()
        ):
            raise ValueError("simulator startup evaluator context differs")
    _decode_identity({key: executable[key] for key in ("bytes", "sha256")})
    sources = value["startup_sources"]
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("simulator startup evaluator context differs")
    if value["startup_source_root"] != str(_public_source_root()):
        raise ValueError("simulator startup source root changed")
    if sources != _startup_source_rows():
        raise ValueError("simulator startup source closure changed")


def _validate_cleanup_receipts(value: dict[str, object]) -> None:
    owner = Path(value["evaluation_context"]["owner_root"])
    view = Path(value["apps"]["exp_path"]).parent
    appdata = Path(value.get("appdata_root", ""))
    guards = [value["cleanup"][name].get("guard", {}) for name in ("view", "appdata")]
    if (
        view == appdata
        or any(root.parent != owner for root in (view, appdata))
        or any(owner.stat().st_dev != guard.get("device") for guard in guards)
    ):
        raise ValueError("simulator startup cleanup root differs")
    _validate_cleanup(value["cleanup"]["view"], view)
    _validate_cleanup(value["cleanup"]["appdata"], appdata)


def _validate_cleanup(value: dict[str, object], expected_root: Path) -> None:
    inventory = value.get("inventory")
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("members"), list
    ):
        raise ValueError("simulator startup cleanup inventory differs")
    expected = _inventory_summary(inventory["members"])
    if inventory != expected or not _cleanup_header_matches(value, expected_root):
        raise ValueError("simulator startup cleanup inventory differs")
    _validate_cleanup_members(value, expected_root)
    _validate_cleanup_repairs(value)


def _cleanup_header_matches(value: dict[str, object], root: Path) -> bool:
    guard = value.get("guard")
    return (
        value.get("schema") == "npa.workbench.owned-tree-cleanup.v1"
        and value.get("status") == "owned_tree_removed_without_symlink_follow"
        and value.get("method") == "fd-relative-lstat-no-follow-owned-postorder"
        and value.get("root") == str(root.absolute())
        and isinstance(guard, dict)
        and set(guard) == {"device", "inode", "uid"}
        and type(guard["device"]) is int
        and guard["device"] >= 0
        and type(guard["inode"]) is int
        and guard["inode"] > 0
        and guard["uid"] == os.geteuid()
    )


def _validate_cleanup_members(value: dict[str, object], root: Path) -> None:
    members = value["inventory"]["members"]
    guard = value["guard"]
    if not members or members[0].get("path") != root.name:
        raise ValueError("simulator startup cleanup root member differs")
    seen = {}
    for row in members:
        _validate_cleanup_member(row, root.name, guard, seen)
        seen[row["path"]] = row["type"]
    first = members[0]
    keys = ("device", "inode", "uid")
    if first.get("type") != "directory" or any(
        first[key] != guard[key] for key in keys
    ):
        raise ValueError("simulator startup cleanup guard differs")


def _validate_cleanup_member(row, root_name: str, guard: dict, seen: dict) -> None:
    keys = {"path", "type", "uid", "mode", "device", "inode"}
    path = row.get("path") if isinstance(row, dict) else None
    parts = Path(path).parts if isinstance(path, str) else ()
    parent = str(Path(path).parent) if isinstance(path, str) else ""
    if (
        not isinstance(row, dict)
        or set(row) != keys
        or not parts
        or parts[0] != root_name
        or any(part in {"", ".", ".."} for part in parts)
        or path != Path(path).as_posix()
        or path in seen
        or (len(parts) > 1 and seen.get(parent) != "directory")
        or row["type"] not in {"directory", "regular", "symlink"}
        or row["uid"] != guard["uid"]
        or row["device"] != guard["device"]
        or type(row["inode"]) is not int
        or row["inode"] <= 0
        or not _permission_mode(row["mode"])
    ):
        raise ValueError("simulator startup cleanup member differs")


def _permission_mode(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"0o[0-7]{1,4}", value) is not None


def _validate_cleanup_repairs(value: dict[str, object]) -> None:
    members = {row["path"]: row for row in value["inventory"]["members"]}
    repairs = value.get("repairs")
    if not isinstance(repairs, list):
        raise ValueError("simulator startup cleanup repairs differ")
    expected = {
        path
        for path, row in members.items()
        if row["type"] == "directory" and int(row["mode"], 8) & 0o700 != 0o700
    }
    observed = set()
    for row in repairs:
        _validate_cleanup_repair(row, members)
        observed.add(row["path"])
    if observed != expected or len(observed) != len(repairs):
        raise ValueError("simulator startup cleanup repairs differ")


def _validate_cleanup_repair(row: object, members: dict[str, dict]) -> None:
    keys = {"path", "type", "uid", "mode_before", "mode_after", "method"}
    if not isinstance(row, dict) or set(row) != keys or row.get("path") not in members:
        raise ValueError("simulator startup cleanup repair differs")
    member = members[row["path"]]
    if (
        row["type"] != "directory"
        or member["type"] != "directory"
        or row["uid"] != member["uid"]
        or row["mode_before"] != member["mode"]
        or row["method"] != "owner-user-rwx-only"
        or not _permission_mode(row["mode_after"])
        or int(row["mode_after"], 8) != int(row["mode_before"], 8) | 0o700
    ):
        raise ValueError("simulator startup cleanup repair differs")


def _validate_embedded_startup(value: dict[str, object]) -> None:
    marker = value.get("startup_marker", {})
    request = value.get("shutdown_request", {})
    pid = marker.get("pid") if isinstance(marker, dict) else None
    if type(pid) is not int or pid <= 0 or marker != _child_marker(pid):
        raise ValueError("simulator startup marker differs")
    evidence = value.get("evidence", {}).get("marker", {})
    expected = _child_shutdown_request_from_identity(pid, _row_identity(evidence))
    if request != expected:
        raise ValueError("simulator shutdown request differs")


def _child_shutdown_request_from_identity(
    pid: int, marker: dict[str, object]
) -> dict[str, object]:
    return {
        "schema": "npa.workbench.simulator-shutdown-request.v1",
        "status": "shutdown_requested_after_verified_startup",
        "pid": pid,
        "startup_marker": marker,
        "model_loaded": False,
        "policy_started": False,
        "case_started": False,
    }

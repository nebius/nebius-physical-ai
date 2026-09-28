"""Inspect retained Isaac application sources without creating a runtime view."""

from __future__ import annotations

from pathlib import Path
import stat

from . import simulator_startup

_EXTENSION_DIRECTORIES = (
    "exts",
    "extscache",
    "extsDeprecated",
    "extsPhysics",
    "extsUser",
)


def _mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.lstat().st_mode))


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _applications(
    root: Path,
) -> tuple[dict[str, simulator_startup.FileIdentity], list[dict]]:
    apps = root / "apps"
    _real_directory(apps, "Isaac apps source")
    identities = {}
    rows = []
    for path in sorted(apps.iterdir(), key=lambda item: item.name):
        simulator_startup._component(path.name, "Isaac application")
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise ValueError(f"Isaac application is not a regular file: {path.name}")
        identity = simulator_startup._identity(path)
        identities[path.name] = simulator_startup.FileIdentity(**identity)
        rows.append({"path": path.name, **identity, "mode": _mode(path)})
    if not rows:
        raise ValueError("Isaac application inventory is empty")
    return identities, rows


def _extension_directories(
    root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], list[dict]]:
    linked = []
    absent = []
    rows = []
    for name in _EXTENSION_DIRECTORIES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Isaac extension directory is a symlink: {name}")
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            absent.append(name)
            rows.append({"path": name, "type": "absent"})
            continue
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Isaac extension path is not a directory: {name}")
        linked.append(name)
        rows.append(
            {
                "path": name,
                "type": "directory",
                "mode": oct(stat.S_IMODE(mode)),
                "resolved_path": str(path.resolve(strict=True)),
            }
        )
    return tuple(linked), tuple(absent), rows


def _observed_spec(isaac_root, owner_root, view_root, version_file):
    root = _real_directory(isaac_root, "Isaac source root")
    owner = _real_directory(owner_root, "Isaac view owner")
    simulator_startup._direct_fresh_child(view_root, owner, "Isaac apps view")
    version_path = simulator_startup._contained_source_file(
        root, version_file, "Isaac version file"
    )
    version = simulator_startup._identity(version_path)
    applications, application_rows = _applications(root)
    linked, absent, extension_rows = _extension_directories(root)
    spec = simulator_startup.IsaacAppsSpec(
        isaac_root=root,
        owner_root=owner,
        view_root=view_root,
        version_file=version_file,
        version=simulator_startup.FileIdentity(**version),
        applications=applications,
        linked_directories=linked,
        absent_directories=absent,
    )
    observed = {
        "version": {"path": version_file, **version, "mode": _mode(version_path)},
        "applications": application_rows,
        "extension_directories": extension_rows,
    }
    return spec, observed


def _spec_document(spec: simulator_startup.IsaacAppsSpec) -> dict:
    return {
        "isaac_root": str(spec.isaac_root),
        "owner_root": str(spec.owner_root),
        "view_root": str(spec.view_root),
        "version_file": spec.version_file,
        "version": simulator_startup._expected(spec.version),
        "applications": {
            name: simulator_startup._expected(identity)
            for name, identity in sorted(spec.applications.items())
        },
        "linked_directories": list(spec.linked_directories),
        "absent_directories": list(spec.absent_directories),
    }


def inspect_isaac_source(
    isaac_root: Path,
    owner_root: Path,
    view_root: Path,
    *,
    version_file: str = "VERSION",
) -> dict:
    """Observe and validate one retained Isaac source without writing it.

    Args:
        isaac_root: Retained Isaac installation root.
        owner_root: Existing writable owner used by the future startup spec.
        view_root: Fresh direct child the future startup may create.
        version_file: Relative regular file identifying the installation.
    Returns:
        An exact startup ``apps`` object plus the production validator claim.
    Raises:
        OSError: Source metadata or bytes cannot be read.
        ValueError: A source member, extension path, or future view is unsafe.
    """
    spec, observed = _observed_spec(isaac_root, owner_root, view_root, version_file)
    return {
        "schema": "npa.workbench.isaac-source-inspection.v1",
        "status": "source_observed_and_production_apps_claim_validated",
        "apps_spec": _spec_document(spec),
        "observed": observed,
        "production_apps_claim": simulator_startup._apps_claim(spec),
        "effects": {
            "simulator_started": False,
            "policy_loaded": False,
            "writable_view_created": False,
        },
    }

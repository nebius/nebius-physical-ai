"""Owned scratch data and local inventory for cluster Terraform runs."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_SCRATCH_MARKER = ".npa-owned.json"
_SCRATCH_LOCK = ".npa-in-use.lock"
_INVENTORY_NAME = "terraform-lifecycle.json"
_LEGACY_CACHE_PARTS = ("deploy", "cluster", ".terraform")


class TerraformDataCleanupError(RuntimeError):
    """An exact NPA-owned Terraform scratch directory could not be removed."""


@dataclass(frozen=True)
class TerraformResidue:
    """One exact Terraform data directory relevant to NPA cleanup."""

    label: str
    path: Path
    removable: bool
    reason: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def terraform_scratch_root() -> Path:
    """Return the NPA-owned parent used for active Terraform data.

    The extra ``active`` level is an upgrade-safety boundary.  Older NPA cleanup
    versions scan only the immediate children of ``terraform-data/cluster``;
    they therefore preserve this unmarked parent instead of deleting a newer
    run whose advisory-lock contract they do not understand.
    """

    return _legacy_terraform_scratch_root() / "active"


def _legacy_terraform_scratch_root() -> Path:
    return Path(
        os.environ.get("NPA_CONFIG_DIR", "").strip() or (Path.home() / ".npa")
    ) / "terraform-data" / "cluster"


def terraform_inventory_file(context: str) -> Path:
    """Return the non-secret lifecycle inventory for one cluster context."""

    from npa.cluster.state import cluster_dir

    return cluster_dir(context) / _INVENTORY_NAME


def record_terraform_inventory(context: str, terraform_dir: Path) -> Path:
    """Record that Terraform apply may have created remotely managed resources."""

    path = terraform_inventory_file(context)
    payload = {
        "version": 1,
        "managed_by": "npa",
        "kind": "cluster-terraform-inventory",
        "context": context,
        "terraform_dir": str(terraform_dir.resolve()),
        "status": "apply_started",
        "updated_at": _now(),
    }
    _write_private_json(path, payload)
    return path


def has_terraform_inventory(context: str) -> bool:
    """Whether an earlier apply recorded potentially managed remote state."""

    return terraform_inventory_file(context).is_file()


def has_destroy_evidence(
    terraform_dir: Path,
    context: str,
    *,
    kubeconfig: Path | None = None,
) -> bool:
    """Return whether local evidence justifies provider/auth teardown work.

    A generated ``.terraform`` directory is deliberately not evidence: it is a
    provider/module cache, not resource state, and a failed no-cluster teardown
    can create it by itself. Interrupted applies are covered by the inventory
    written immediately before ``terraform apply``.
    """

    from npa.cluster.state import kubeconfig_file, state_file

    if has_terraform_inventory(context) or state_file(context).is_file():
        return True
    if kubeconfig is not None and kubeconfig.expanduser().is_file():
        return True
    if kubeconfig_file(context).is_file():
        return True
    if any(terraform_dir.glob("*.tfstate")):
        return True
    state_dir = terraform_dir / ".tfstate.d"
    return state_dir.is_dir() and any(state_dir.rglob("*.tfstate"))


@contextmanager
def isolated_terraform_data_dir(
    terraform_dir: Path,
    context: str,
) -> Iterator[Path]:
    """Yield a marked ``TF_DATA_DIR`` and remove it on every exit path."""

    root = terraform_scratch_root()
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        scratch = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        lock_fd = os.open(scratch / _SCRATCH_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_private_json(
            scratch / _SCRATCH_MARKER,
            {
                "version": 1,
                "managed_by": "npa",
                "kind": "cluster-terraform-data",
                "path": str(scratch.resolve()),
                "terraform_dir": str(terraform_dir.resolve()),
                "context": context,
                "created_at": _now(),
            },
        )
    except OSError as exc:
        if "lock_fd" in locals():
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if "scratch" in locals():
            try:
                shutil.rmtree(scratch)
            except OSError:
                pass
        try:
            root.rmdir()
            root.parent.rmdir()
        except OSError:
            pass
        raise RuntimeError(
            f"Could not create isolated Terraform data under {root}: {exc}"
        ) from exc

    primary_error: BaseException | None = None
    try:
        yield scratch
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error = _remove_owned_scratch(scratch, held_lock_fd=lock_fd)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        try:
            root.rmdir()
            root.parent.rmdir()
        except OSError:
            pass
        if cleanup_error:
            error = TerraformDataCleanupError(
                "Terraform scratch cleanup failed for the exact NPA-owned path "
                f"{scratch}: {cleanup_error}. Run `npa cleanup --full --yes` "
                "after fixing the reported filesystem problem."
            )
            if primary_error is not None:
                raise error from primary_error
            raise error


def collect_terraform_residue(start: Path | None = None) -> list[TerraformResidue]:
    """Find exact NPA Terraform scratch and legacy source-cache residue."""

    found: list[TerraformResidue] = []
    current_root = terraform_scratch_root()
    for root in (current_root, _legacy_terraform_scratch_root()):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            found.append(
                TerraformResidue(
                    "Unverified Terraform scratch root",
                    root,
                    False,
                    "scratch root is not an exact NPA-owned directory",
                )
            )
            continue
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError as exc:
            found.append(
                TerraformResidue(
                    "Unverified Terraform scratch root",
                    root,
                    False,
                    f"scratch root could not be inspected: {exc}",
                )
            )
        else:
            for child in children:
                if root != current_root and child == current_root:
                    continue
                if _owned_scratch_marker(child):
                    active, reason = _scratch_lock_status(child)
                    found.append(
                        TerraformResidue(
                            "Terraform runtime scratch",
                            child,
                            not active,
                            reason,
                        )
                    )
                else:
                    found.append(
                        TerraformResidue(
                            "Unverified Terraform scratch",
                            child,
                            False,
                            "ownership marker is missing, malformed, or mismatched",
                        )
                    )

    for repo_root in _candidate_repo_roots(start or Path.cwd()):
        cache = repo_root.joinpath(*_LEGACY_CACHE_PARTS)
        if _validated_legacy_source_cache(cache, repo_root):
            found.append(
                TerraformResidue(
                    "Legacy source-checkout Terraform cache", cache, True
                )
            )
        elif cache.exists() or cache.is_symlink():
            found.append(
                TerraformResidue(
                    "Unverified source-checkout Terraform path",
                    cache,
                    False,
                    "path failed exact legacy-cache ownership validation",
                )
            )
    return list(dict.fromkeys(found))


def remove_terraform_residue(item: TerraformResidue) -> str:
    """Remove one exact, revalidated NPA Terraform residue path."""

    if not item.removable:
        return item.reason or "ownership could not be validated"
    path = item.path
    if item.label == "Terraform runtime scratch":
        return _remove_owned_scratch(path)
    if item.label == "Legacy source-checkout Terraform cache":
        repo_root = path.parent.parent.parent
        if not _validated_legacy_source_cache(path, repo_root):
            return "legacy cache ownership validation changed before deletion"
        try:
            shutil.rmtree(path)
        except OSError as exc:
            return str(exc)
        return ""
    return "unsupported Terraform residue type"


def _remove_owned_scratch(path: Path, *, held_lock_fd: int | None = None) -> str:
    if not _owned_scratch_marker(path):
        return "ownership marker is missing, malformed, or mismatched"
    lock_fd = held_lock_fd
    if lock_fd is None:
        lock_fd, reason = _acquire_scratch_lock(path)
        if reason:
            return reason
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return str(exc)
    finally:
        if lock_fd is not None and held_lock_fd is None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
    root = path.parent
    try:
        root.rmdir()
        root.parent.rmdir()
    except OSError:
        pass
    return ""


def _acquire_scratch_lock(path: Path) -> tuple[int | None, str]:
    """Acquire an existing scratch lock or explain why removal is unsafe."""

    lock_path = path / _SCRATCH_LOCK
    try:
        descriptor = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        # Compatibility for marked scratch from versions before the lock existed.
        return None, ""
    except OSError as exc:
        return None, f"scratch ownership lock could not be verified: {exc}"
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None, "active Terraform run holds the scratch ownership lock"
    except OSError as exc:
        os.close(descriptor)
        return None, f"scratch ownership lock could not be verified: {exc}"
    return descriptor, ""


def _scratch_lock_status(path: Path) -> tuple[bool, str]:
    """Return whether an owned scratch directory is in use by another process.

    ``npa cleanup --full`` may run concurrently with a long Terraform apply from
    another checkout that shares ``~/.npa``.  The ownership marker proves that a
    directory belongs to NPA; it does not prove that deleting it is safe *now*.
    A non-blocking advisory lock closes that gap without relying on PIDs or
    wall-clock age.  Scratch created before this lock existed remains removable.
    """

    descriptor, reason = _acquire_scratch_lock(path)
    if reason:
        return True, reason
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return False, ""


def _owned_scratch_marker(path: Path) -> bool:
    roots = (terraform_scratch_root(), _legacy_terraform_scratch_root())
    try:
        if (
            not path.is_dir()
            or path.is_symlink()
            or path.parent.resolve() not in {root.resolve() for root in roots}
            or not path.name.startswith("run-")
        ):
            return False
        marker = json.loads((path / _SCRATCH_MARKER).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("managed_by") == "npa"
        and marker.get("kind") == "cluster-terraform-data"
        and marker.get("path") == str(path.resolve())
    )


def _candidate_repo_roots(start: Path) -> list[Path]:
    candidates: list[Path] = []
    for seed in (start.resolve(), Path(__file__).resolve()):
        for current in (seed, *seed.parents):
            if (current / ".git").exists():
                candidates.append(current)
                break
    return list(dict.fromkeys(candidates))


def _validated_legacy_source_cache(path: Path, repo_root: Path) -> bool:
    expected = repo_root / "deploy" / "cluster" / ".terraform"
    try:
        deploy_dir = repo_root / "deploy"
        terraform_dir = deploy_dir / "cluster"
        if path != expected or any(
            component.is_symlink()
            for component in (repo_root, deploy_dir, terraform_dir, path)
        ):
            return False
        if not path.is_dir() or path.resolve() != expected.absolute():
            return False
        if not all(
            (terraform_dir / name).is_file()
            for name in ("main.tf", "versions.tf", ".terraform.lock.hcl")
        ):
            return False
        return any(
            (path / name).exists()
            for name in ("environment", "modules", "providers", "terraform.tfstate")
        )
    except OSError:
        return False


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        path.chmod(0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

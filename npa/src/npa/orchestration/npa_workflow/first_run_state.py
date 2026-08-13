"""Project/workflow-scoped first-run identities.

The historical ``~/.npa/paidf-first-run-id`` file had no project identity and
could silently resume another project's days-old run.  This store is locked,
atomic, non-secret, and keyed by stable project ID plus workflow identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any, Iterator

STATE_SCHEMA = "npa.workflow.first-run.v1"
_NPA_CONFIG_DIR = Path(
    os.environ.get("NPA_CONFIG_DIR", "").strip() or (Path.home() / ".npa")
)
DEFAULT_ROOT = _NPA_CONFIG_DIR / "workflow-runs"
LEGACY_PATH = _NPA_CONFIG_DIR / "paidf-first-run-id"
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-._")[:48] or "unknown"


def _scope_part(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(value)}-{digest}"


def resolve_project_identity(project: str) -> tuple[str, str, str]:
    """Return stable identity, display alias, and identity provenance."""

    alias = str(project or "").strip()
    try:
        from npa.clients.config import resolve_environment

        environment = resolve_environment(alias or None)
        project_id = str(getattr(environment, "project_id", "") or "").strip()
        resolved_alias = str(getattr(environment, "alias", "") or alias or "default")
        if project_id:
            return project_id, resolved_alias, "configured_project_id"
    except Exception:  # noqa: BLE001 - an explicit alias still isolates state
        logger.debug(
            "Could not resolve stable project identity; isolating by explicit alias",
            exc_info=True,
        )
    if alias.startswith("project-"):
        return alias, alias, "explicit_project_id"
    return f"alias:{alias or 'default'}", alias or "default", "project_alias_fallback"


def terminal_run_evidence(
    *, project: str, run_id: str, state_root: Path | None = None
) -> dict[str, Any]:
    """Read a unique verified terminal project/run observation without mutation."""

    stable_project, _alias, _source = resolve_project_identity(project)
    root = (state_root or DEFAULT_ROOT) / _scope_part(stable_project)
    if root.is_symlink():
        return {}
    try:
        paths = list(root.glob("*.json"))
    except OSError:
        return {}
    matches: list[dict[str, Any]] = []
    terminal = {"SUCCEEDED", "FAILED", "FAILED_STARTUP", "CANCELLED", "BLOCKED"}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        state = str(payload.get("last_known_state") or "").upper()
        if (
            payload.get("schema_version") == STATE_SCHEMA
            and str(payload.get("project_identity") or "") == stable_project
            and str(payload.get("run_id") or "") == run_id
            and str(payload.get("last_verification_status") or "").upper() == "VERIFIED"
            and state in terminal
        ):
            matches.append(dict(payload))
    return matches[0] if len(matches) == 1 else {}


def state_path(
    *,
    project_identity: str,
    workflow_identity: str,
    state_root: Path | None = None,
) -> Path:
    root = state_root or DEFAULT_ROOT
    return (
        root / _scope_part(project_identity) / f"{_scope_part(workflow_identity)}.json"
    )


@dataclass(frozen=True)
class RunPreparation:
    run_id: str
    generated_new: bool
    state_path: str
    warning: str = ""
    previous_run: dict[str, Any] | None = None


def _age_seconds(value: object, *, now: datetime) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        observed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return max(0, int((now - observed.astimezone(timezone.utc)).total_seconds()))


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(fd, "r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
    finally:
        # fd is owned/closed by fdopen on normal and exceptional exits.
        pass


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
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
        tmp.unlink(missing_ok=True)
        raise


def prepare_run(
    *,
    project: str,
    workflow_identity: str,
    resume_run: str = "",
    new_run_id: str = "",
    state_root: Path | None = None,
    legacy_path: Path | None = None,
    persist: bool = True,
) -> RunPreparation:
    """Generate a fresh ID, optionally recording the requested identity.

    ``persist=False`` is the plan/preflight path.  It performs no mkdir, lock-file
    creation or state write, so a rejected submit cannot look like a submitted
    run to later status/cancel commands.
    """

    stable_project, alias, identity_source = resolve_project_identity(project)
    path = state_path(
        project_identity=stable_project,
        workflow_identity=workflow_identity,
        state_root=state_root,
    )
    legacy = legacy_path or LEGACY_PATH
    warning = ""
    if legacy.exists():
        warning = (
            f"legacy run-id state exists at {legacy}; it is not project/workflow scoped "
            "and was not reused or deleted"
        )

    def _prepare(
        existing: dict[str, Any],
    ) -> tuple[str, bool, dict[str, Any] | None, dict[str, Any]]:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat().replace("+00:00", "Z")
        previous_run = None
        if existing.get("run_id"):
            previous_run = {
                "run_id": str(existing["run_id"]),
                "age_seconds": _age_seconds(
                    existing.get("last_used_at") or existing.get("created_at"),
                    now=now_dt,
                ),
                "created_at": str(existing.get("created_at") or ""),
                "last_used_at": str(existing.get("last_used_at") or ""),
                "last_known_state": str(
                    existing.get("last_known_state") or "UNKNOWN"
                ).upper(),
                "last_verification_status": str(
                    existing.get("last_verification_status") or "UNVERIFIED"
                ).upper(),
            }
        if resume_run.strip() and new_run_id.strip():
            raise ValueError("resume_run and new_run_id are mutually exclusive")
        if resume_run.strip():
            run_id = resume_run.strip()
            source = "explicit_resume"
            generated = False
        elif new_run_id.strip():
            run_id = new_run_id.strip()
            source = "explicit_new"
            generated = True
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"{_slug(workflow_identity)}-{stamp}-{secrets.token_hex(3)}"
            source = "generated_new"
            generated = True
        created_at = (
            now
            if generated or existing.get("run_id") != run_id
            else str(existing.get("created_at") or now)
        )
        payload = {
            "schema_version": STATE_SCHEMA,
            "run_id": run_id,
            "project_identity": stable_project,
            "project_alias": alias,
            "project_identity_source": identity_source,
            "workflow_identity": workflow_identity,
            "created_at": created_at,
            "last_used_at": now,
            "source": source,
            "last_known_state": (
                existing.get("last_known_state", "UNKNOWN")
                if existing.get("run_id") == run_id
                else "UNKNOWN"
            ),
            "last_verification_status": (
                existing.get("last_verification_status", "UNVERIFIED")
                if existing.get("run_id") == run_id
                else "UNVERIFIED"
            ),
        }
        return run_id, generated, previous_run, payload

    def _read_existing() -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"scoped run state is unreadable: {path}: {exc}"
            ) from exc
        return loaded if isinstance(loaded, dict) else {}

    if not persist:
        run_id, generated, previous_run, _payload = _prepare(_read_existing())
    else:
        with _locked(path):
            run_id, generated, previous_run, payload = _prepare(_read_existing())
            _atomic_write(path, payload)
    return RunPreparation(
        run_id=run_id,
        generated_new=generated,
        state_path=str(path),
        warning=warning,
        previous_run=previous_run,
    )


def update_run_observation(
    *,
    project: str,
    workflow_identity: str,
    run_id: str,
    last_known_state: str,
    verification_status: str,
    state_root: Path | None = None,
) -> bool:
    """Update only a matching scoped record; never cross-link an arbitrary run."""

    stable_project, _, _ = resolve_project_identity(project)
    path = state_path(
        project_identity=stable_project,
        workflow_identity=workflow_identity,
        state_root=state_root,
    )
    with _locked(path):
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or str(payload.get("run_id") or "") != run_id:
            return False
        payload["last_used_at"] = _now()
        payload["last_known_state"] = str(last_known_state or "UNKNOWN").upper()
        payload["last_verification_status"] = str(
            verification_status or "UNVERIFIED"
        ).upper()
        _atomic_write(path, payload)
    return True

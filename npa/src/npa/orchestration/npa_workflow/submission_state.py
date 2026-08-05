"""Owner-only local state for restart-safe workflow submission side effects.

The authoritative workflow status remains the existing S3 run manifest.  This
small ledger only records client-side operations that happen before or after the
managed job exists: source staging, launch identity, and agent artifact loading.
It deliberately never stores credentials.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping


SCHEMA_VERSION = "npa.workflow.submission.v1"
_SECRET_KEY = re.compile(r"(secret|password|credential|token|access_key)", re.IGNORECASE)


@dataclass(frozen=True)
class SubmissionStateRead:
    """Result of inspecting a receipt without conflating absence and I/O failure."""

    outcome: Literal["found", "absent", "unavailable"]
    payload: dict[str, Any]
    error: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-.")
    return cleaned[:96] or fallback


def submission_state_path(project: str, run_id: str) -> Path:
    """Return the per-project/run ledger path below the current user's config."""

    return (
        Path.home()
        / ".npa"
        / "workflow-submissions"
        / _safe_component(project, "default")
        / f"{_safe_component(run_id, 'workflow')}.json"
    )


def _contains_secret(value: object, *, parent: str = "") -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if _SECRET_KEY.search(name) and name not in {"secret_safe"}:
                return True
            if _contains_secret(child, parent=name):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret(item, parent=parent) for item in value)
    return False


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def load_submission_state(project: str, run_id: str) -> dict[str, Any]:
    return _read(submission_state_path(project, run_id))


def inspect_submission_state(project: str, run_id: str) -> SubmissionStateRead:
    """Read one exact receipt and retain why it could not be read.

    ``load_submission_state`` intentionally remains the forgiving API used by
    restart-safe submit operations. Run resolution needs a stricter distinction:
    an unreadable/corrupt owner receipt is verification-unavailable, not proof
    that the run never launched.
    """

    path = submission_state_path(project, run_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return SubmissionStateRead("absent", {})
    except OSError as exc:
        return SubmissionStateRead("unavailable", {}, f"could not read {path}: {exc}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return SubmissionStateRead("unavailable", {}, f"invalid receipt JSON at {path}: {exc}")
    if not isinstance(payload, dict):
        return SubmissionStateRead("unavailable", {}, f"invalid receipt object at {path}")
    if str(payload.get("run_id") or "") != run_id:
        return SubmissionStateRead("unavailable", {}, f"receipt run id does not match {run_id!r}")
    expected_project = project or "default"
    if str(payload.get("project") or "") != expected_project:
        return SubmissionStateRead(
            "unavailable", {}, f"receipt project does not match {expected_project!r}"
        )
    return SubmissionStateRead("found", dict(payload))


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        path.chmod(0o600)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def submission_lock(project: str, run_id: str) -> Iterator[Path]:
    """Serialize source/launch/load transitions for one local run."""

    path = submission_state_path(project, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_submission_state(
    project: str,
    run_id: str,
    updates: Mapping[str, Any],
    *,
    locked: bool = False,
) -> dict[str, Any]:
    """Atomically merge non-secret submission metadata into the local ledger."""

    if _contains_secret(updates):
        raise ValueError("workflow submission state must not contain credentials or secrets")

    def _update(path: Path) -> dict[str, Any]:
        payload = _read(path)
        payload.update(dict(updates))
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "project": project or "default",
                "run_id": run_id,
                "updated_at": _utc_now(),
            }
        )
        _write_atomic(path, payload)
        return payload

    if locked:
        return _update(submission_state_path(project, run_id))
    with submission_lock(project, run_id) as path:
        return _update(path)

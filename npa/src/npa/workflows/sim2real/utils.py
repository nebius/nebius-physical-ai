"""Shared helpers for the Sim2Real workflow package."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError


def _split_csv(value: Any) -> list[str]:
    """Split scalar or already-parsed values without leaking container reprs."""

    values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    parts: list[str] = []
    for item in values:
        parts.extend(str(item or "").replace(";", ",").split(","))
    return [part.strip() for part in parts if part.strip()]


def _serviceaccount_namespace() -> str:
    path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _artifact_root_uri(config: Sim2RealLoopConfig) -> str:
    parts = [part for part in (config.s3_prefix.strip("/"), config.run_id) if part]
    return f"s3://{config.s3_bucket}/{'/'.join(parts)}"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise Sim2RealLoopError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically replace a JSON artifact after flushing its complete bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"path": str(path), "payload": payload}

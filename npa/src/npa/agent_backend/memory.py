"""Persistent per-agent run/experiment memory for the NPA agent backend (Gap 5).

Lets the agent answer cross-session questions like "why did run B regress vs run
A" from *stored run metadata*, not model recall. Memory is keyed by run id and
holds the numeric signals + provenance the agent already computes, so answers
stay grounded.

Storage is injected via a tiny ``store`` protocol (``read`` / ``write`` /
``list_keys``) so the module is backend-agnostic and unit-tests with an
in-memory fake. The VM backend wires a JSON-file store under the agent data dir;
no bucket name, project id, or secret is hardcoded here.

Phase G: this module is *shipped* to the agent VM as an importable file (see
``npa/src/npa/agent_backend/__init__.py``) rather than string-substituted; the
backend imports it via ``from agent_backend.memory import ...``. The
``npa/src/npa/cli/agent_memory.py`` shim re-exports it for existing callers/tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

MEMORY_KEY_PREFIX = "runs/"
INDEX_KEY = "index.json"

# Run ids come from requests; constrain them to a safe token so a crafted id
# (e.g. "../../session_state") can never escape the memory store directory.
_SAFE_RUN_ID_RE = re.compile(r"[^A-Za-z0-9._:-]")


def _safe_run_id(run_id: str) -> str:
    token = _SAFE_RUN_ID_RE.sub("_", str(run_id or "").strip())
    # Collapse any dot runs so "." / ".." cannot form a traversal segment.
    token = re.sub(r"\.{2,}", "_", token).strip("._")
    return token


class InMemoryStore:
    """Dict-backed store for tests and ephemeral use."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        return self._data.get(str(key))

    def write(self, key: str, value: str) -> None:
        self._data[str(key)] = str(value)

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(str(prefix)))


class JsonFileStore:
    """Filesystem-backed store rooted at a base directory (no bucket/secret).

    Keys map to files under ``base_dir``; nested keys create subdirectories.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)

    def _path(self, key: str) -> Path:
        return self._base / str(key)

    def read(self, key: str) -> str | None:
        try:
            return self._path(key).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    def write(self, key: str, value: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value), encoding="utf-8")

    def list_keys(self, prefix: str = "") -> list[str]:
        root = self._base
        if not root.exists():
            return []
        keys: list[str] = []
        for path in root.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(root))
                if rel.startswith(str(prefix)):
                    keys.append(rel)
        return sorted(keys)


def _fallback_compare(run_a: Any, run_b: Any) -> dict[str, Any]:
    """Minimal success_rate delta when no richer comparator is injected."""
    def _sr(entry: Any) -> float | None:
        if isinstance(entry, dict):
            value = entry.get("success_rate")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    sr_a = _sr(run_a if isinstance(run_a, dict) else {})
    sr_b = _sr(run_b if isinstance(run_b, dict) else {})
    delta = None
    regressed = False
    if sr_a is not None and sr_b is not None:
        delta = round(sr_b - sr_a, 6)
        regressed = delta < 0
    return {
        "delta_success_rate": delta,
        "regressed": regressed,
        "improved": bool(delta is not None and delta > 0),
        "verdict": "regression" if regressed else ("improvement" if (delta or 0) > 0 else "no_change"),
        "notes": [],
    }


class RunMemory:
    """Persistent, grounded run/experiment memory over an injected store."""

    def __init__(
        self,
        store: Any,
        *,
        comparator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._comparator = comparator or _fallback_compare

    # ── persistence ─────────────────────────────────────────────────────────
    def _run_key(self, run_id: str) -> str:
        return f"{MEMORY_KEY_PREFIX}{_safe_run_id(run_id)}.json"

    def _read_index(self) -> list[str]:
        raw = self._store.read(INDEX_KEY)
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return [str(x) for x in data] if isinstance(data, list) else []

    def _write_index(self, run_ids: list[str]) -> None:
        # De-dupe preserving most-recent-first ordering.
        seen: list[str] = []
        for run_id in run_ids:
            if run_id and run_id not in seen:
                seen.append(run_id)
        self._store.write(INDEX_KEY, json.dumps(seen))

    def record_run(
        self, run_id: str, metadata: dict[str, Any], *, source: str = "api"
    ) -> dict[str, Any]:
        """Persist a run's metadata/metrics; returns the stored record.

        ``source`` records provenance ("drive" for agent-driven runs, "api" for
        operator-supplied metadata) so downstream comparisons can distinguish
        authoritative run data from hand-entered records.
        """
        run_id = _safe_run_id(run_id)
        if not run_id:
            raise ValueError("run_id is required")
        record = dict(metadata) if isinstance(metadata, dict) else {"value": metadata}
        record["run_id"] = run_id
        record.setdefault("source", source)
        self._store.write(self._run_key(run_id), json.dumps(record, sort_keys=True))
        index = self._read_index()
        self._write_index([run_id, *index])
        return record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        raw = self._store.read(self._run_key(run_id))
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def list_runs(self, *, limit: int = 20) -> list[str]:
        index = self._read_index()
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        return index[: max(0, limit)]

    # ── grounded analysis ────────────────────────────────────────────────────
    def compare_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        rec_a = self.get_run(run_a)
        rec_b = self.get_run(run_b)
        if rec_a is None or rec_b is None:
            missing = [rid for rid, rec in ((run_a, rec_a), (run_b, rec_b)) if rec is None]
            return {"ok": False, "error": f"missing run metadata for: {', '.join(missing)}"}
        comparison = self._comparator(rec_a, rec_b)
        comparison["ok"] = True
        comparison["run_a"] = str(run_a)
        comparison["run_b"] = str(run_b)
        return comparison

    def explain_regression(self, run_b: str, baseline: str) -> str:
        """Grounded explanation of run_b vs a baseline (stored metadata only)."""
        result = self.compare_runs(baseline, run_b)
        if not result.get("ok"):
            return (
                f"**Cannot compare** — {result.get('error', 'missing run metadata')}. "
                "Record both runs first via run memory."
            )
        delta = result.get("delta_success_rate")
        lines = [
            f"**{run_b} vs {baseline}** (grounded on stored run metadata):",
            f"- **verdict**: `{result.get('verdict')}`",
        ]
        if delta is not None:
            lines.append(f"- **delta_success_rate**: `{delta}` (run_b − baseline)")
        for note in result.get("notes", []):
            lines.append(f"- {note}")
        if result.get("regressed"):
            lines.append("- This is a **regression** — inspect config/diagnosis deltas between the runs.")
        return "\n".join(lines)

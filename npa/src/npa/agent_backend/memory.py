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
from typing import Any, Callable, Mapping

MEMORY_KEY_PREFIX = "runs/"
INDEX_KEY = "index.json"
# Cap the most-recent-first index so it cannot grow without bound across many
# drives. Per-run records stay on the store; only the index list is trimmed.
MAX_INDEX_ENTRIES = 500

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


_OUTCOME_HIGHER = frozenset(
    {"accuracy", "f1", "precision", "recall", "reward", "return", "score", "success_rate"}
)
_OUTCOME_LOWER = frozenset(
    {
        "collision_rate",
        "corruption_rate",
        "error_rate",
        "failed_check_count",
        "failure_rate",
        "loss",
    }
)
_EFFICIENCY_LOWER = frozenset(
    {
        "cost_usd",
        "duration",
        "duration_s",
        "duration_seconds",
        "elapsed_s",
        "latency",
        "latency_ms",
        "latency_s",
        "mean_steps_to_success",
        "total_cost_usd",
        "wall_clock_s",
    }
)
_EFFICIENCY_HIGHER = frozenset({"throughput"})
_COUNTER_TOKENS = frozenset(
    {"epoch", "epochs", "sample", "samples", "step", "steps", "timestamp", "tokens"}
)
_METRIC_ALIASES = {
    "created_at": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "updated_at": "timestamp",
}


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten only observed numeric fields; booleans are not measurements."""
    flattened: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.update(_flatten_numeric(item, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and prefix:
        flattened[prefix] = value
    return flattened


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalars(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.update(_flatten_scalars(item, path))
    elif prefix and (value is None or isinstance(value, (str, int, float, bool))):
        flattened[prefix] = value
    return flattened


def _normalized_metric_name(field: str) -> str:
    leaf = re.split(r"[.\[\]]", str(field or ""))[-1]
    normalized = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
    return _METRIC_ALIASES.get(normalized, normalized)


def _metric_taxonomy(field: str) -> tuple[str, str]:
    """Return ``(category, direction)`` from an explicit normalized taxonomy."""
    name = _normalized_metric_name(field)
    if name in _OUTCOME_HIGHER or name.endswith(
        ("_accuracy", "_precision", "_recall", "_reward", "_return", "_score", "_success_rate")
    ):
        return "outcome", "higher_is_better"
    if name in _OUTCOME_LOWER or name.endswith(("_loss", "_error_rate", "_failure_rate")):
        return "outcome", "lower_is_better"
    if name in _EFFICIENCY_HIGHER or name.endswith(
        ("_throughput", "_per_second", "_per_sec", "_per_s")
    ):
        return "efficiency", "higher_is_better"
    if name in _EFFICIENCY_LOWER or name.endswith(
        ("_cost_usd", "_duration", "_duration_s", "_duration_seconds", "_latency", "_latency_ms")
    ):
        return "efficiency", "lower_is_better"
    if name in _METRIC_ALIASES.values() or any(
        token in _COUNTER_TOKENS for token in name.split("_")
    ):
        return "counter", "neutral"
    return "context", "observed_change"


def _metric_direction(field: str) -> str:
    return _metric_taxonomy(field)[1]


def _metric_assessment(delta: float, direction: str, category: str = "outcome") -> str:
    if delta == 0:
        return "unchanged"
    if category == "counter":
        return "neutral"
    if category == "context":
        return "changed"
    if category == "efficiency":
        improved = (delta > 0) == (direction == "higher_is_better")
        return "more_efficient" if improved else "less_efficient"
    if direction == "higher_is_better":
        return "regressed" if delta < 0 else "improved"
    if direction == "lower_is_better":
        return "regressed" if delta > 0 else "improved"
    return "changed"


def _metric_field_preference(field: str) -> tuple[int, str]:
    normalized = str(field or "").lower()
    if normalized.startswith("metrics."):
        return 0, normalized
    if normalized.startswith(("evaluation.", "eval.", "success_summary.")):
        return 1, normalized
    if "." not in normalized:
        return 2, normalized
    if normalized.startswith(("metadata.", "context.")):
        return 4, normalized
    return 3, normalized


def _normalized_numeric_metrics(record: Mapping[str, Any]) -> dict[str, tuple[str, float | int]]:
    """Deduplicate aliases while preferring authoritative metric containers."""
    selected: dict[str, tuple[str, float | int]] = {}
    for field, value in _flatten_numeric(record).items():
        if field.startswith(("config.", "parameters.", "resources.")):
            continue
        identity = _normalized_metric_name(field)
        current = selected.get(identity)
        if current is None or _metric_field_preference(field) < _metric_field_preference(
            current[0]
        ):
            selected[identity] = (field, value)
    return selected


def _display_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


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
        # De-dupe preserving most-recent-first ordering, then cap the index size
        # so it cannot grow unbounded over many drives.
        seen: list[str] = []
        for run_id in run_ids:
            if run_id and run_id not in seen:
                seen.append(run_id)
            if len(seen) >= MAX_INDEX_ENTRIES:
                break
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

    def explain_regression_data(self, run_b: str, baseline: str) -> dict[str, Any]:
        """Return planner-ready regression evidence from two stored records.

        Metric direction is inferred only from the stored field name. Config
        changes are reported as coincident evidence, never asserted as causes.
        """
        comparison = self.compare_runs(baseline, run_b)
        if not comparison.get("ok"):
            return comparison
        baseline_record = self.get_run(baseline) or {}
        candidate_record = self.get_run(run_b) or {}

        baseline_metrics = _normalized_numeric_metrics(baseline_record)
        candidate_metrics = _normalized_numeric_metrics(candidate_record)
        metric_evidence: list[dict[str, Any]] = []
        for identity in sorted(set(baseline_metrics) & set(candidate_metrics)):
            baseline_field, baseline_value = baseline_metrics[identity]
            candidate_field, candidate_value = candidate_metrics[identity]
            delta = round(float(candidate_value) - float(baseline_value), 12)
            if delta == 0:
                continue
            field = min(
                (baseline_field, candidate_field),
                key=_metric_field_preference,
            )
            category, direction = _metric_taxonomy(field)
            metric_evidence.append(
                {
                    "field": field,
                    "metric": identity,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "delta": delta,
                    "category": category,
                    "direction": direction,
                    "assessment": _metric_assessment(delta, direction, category),
                }
            )
        metric_evidence.sort(
            key=lambda item: (
                item["category"] != "outcome",
                item["assessment"] != "regressed",
                item["field"],
            )
        )

        config_changes: list[dict[str, Any]] = []
        for section in ("config", "parameters", "resources"):
            baseline_values = _flatten_scalars(baseline_record.get(section), section)
            candidate_values = _flatten_scalars(candidate_record.get(section), section)
            for field in sorted(set(baseline_values) & set(candidate_values)):
                if baseline_values[field] == candidate_values[field]:
                    continue
                config_changes.append(
                    {
                        "field": field,
                        "baseline": baseline_values[field],
                        "candidate": candidate_values[field],
                    }
                )

        if any(
            item["category"] == "outcome" and item["assessment"] == "regressed"
            for item in metric_evidence
        ):
            verdict = "regression"
        elif any(
            item["category"] == "outcome" and item["assessment"] == "improved"
            for item in metric_evidence
        ):
            verdict = "improvement"
        elif metric_evidence:
            verdict = "no_quality_change"
        else:
            verdict = "no_change"

        lines = [
            f"**{run_b} vs {baseline}** (grounded on stored run metadata):",
            f"- **verdict**: `{verdict}`",
        ]
        delta_success_rate = comparison.get("delta_success_rate")
        if delta_success_rate is not None:
            lines.append(
                f"- **delta_success_rate**: `{delta_success_rate}` (run_b − baseline)"
            )
        for evidence in metric_evidence:
            lines.append(
                "- **{field}**: baseline `{baseline}`, candidate `{candidate}`, "
                "delta `{delta}` — {assessment} ({category}; {direction}).".format(
                    field=evidence["field"],
                    baseline=_display_value(evidence["baseline"]),
                    candidate=_display_value(evidence["candidate"]),
                    delta=evidence["delta"],
                    assessment=evidence["assessment"],
                    category=evidence["category"],
                    direction=evidence["direction"],
                )
            )
        if not metric_evidence:
            lines.append("- No changed numeric field is shared by both stored records.")
        for change in config_changes:
            lines.append(
                "- Coincident stored config change **{field}**: `{baseline}` → "
                "`{candidate}` (correlation only).".format(
                    field=change["field"],
                    baseline=_display_value(change["baseline"]),
                    candidate=_display_value(change["candidate"]),
                )
            )

        return {
            "ok": True,
            "baseline_run": str(baseline),
            "candidate_run": str(run_b),
            "verdict": verdict,
            "delta_success_rate": delta_success_rate,
            "metric_evidence": metric_evidence,
            "config_changes": config_changes,
            "sources": {
                "baseline": baseline_record.get("source"),
                "candidate": candidate_record.get("source"),
            },
            "explanation": "\n".join(lines),
        }

    def explain_regression(self, run_b: str, baseline: str) -> str:
        """Grounded explanation of run_b vs a baseline (stored metadata only)."""
        result = self.explain_regression_data(run_b, baseline)
        if not result.get("ok"):
            return (
                f"**Cannot compare** — {result.get('error', 'missing run metadata')}. "
                "Record both runs first via run memory."
            )
        return str(result.get("explanation") or "")

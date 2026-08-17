"""Process-wide locked JSON state store for the agent backend.

Embedded into the agent-VM backend the same way as ``agent_rrd_proxy``.
Provides atomic read-modify-write so concurrent Starlette threadpool handlers
cannot clobber confirm tokens, chat history, or sim-viz selection.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class StateStore:
    """Thread-safe JSON file state with optional after-save hook (e.g. S3 mirror)."""

    def __init__(
        self,
        path: Path | str,
        *,
        default_factory: Callable[[], dict[str, Any]] | None = None,
        after_save: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._default_factory = default_factory or (lambda: {})
        self._after_save = after_save
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def load(self) -> dict[str, Any]:
        """Load state from disk. Caller must hold ``lock`` for RMW safety."""
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:  # noqa: BLE001
                logger.debug(
                    "failed to load agent state from %s", self.path, exc_info=True
                )
        return self._default_factory()

    def save(self, state: dict[str, Any]) -> None:
        """Persist state to disk. Caller must hold ``lock`` for RMW safety."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self._after_save is not None:
            try:
                self._after_save(state)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "agent state after_save hook failed for %s",
                    self.path,
                    exc_info=True,
                )

    def mutate(self, fn: Callable[[dict[str, Any]], T]) -> T:
        """Atomically load → mutate → save under the process-wide lock."""
        with self._lock:
            state = self.load()
            if not isinstance(state, dict):
                state = self._default_factory()
            result = fn(state)
            self.save(state)
            return result

    def read(self) -> dict[str, Any]:
        """Load under lock (snapshot); mutations need ``mutate`` or explicit lock."""
        with self._lock:
            data = self.load()
            return dict(data) if isinstance(data, dict) else self._default_factory()


def preserve_latest_namespaces(
    candidate: dict[str, Any],
    latest: dict[str, Any],
    namespaces: tuple[str, ...],
) -> dict[str, Any]:
    """Keep atomic namespaces when a legacy load-then-save caller is stale.

    The agent still has older request handlers that call ``load`` and ``save``
    separately.  A slow handler must not overwrite namespaces whose writers use
    ``StateStore.mutate`` for an atomic transaction.
    """

    merged = dict(candidate)
    for namespace in namespaces:
        current = latest.get(namespace)
        if isinstance(current, dict):
            merged[namespace] = dict(current)
    return merged

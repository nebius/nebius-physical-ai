from __future__ import annotations

import json
import threading
from pathlib import Path


class FallbackChain:
    """Run-wide project fallback chain for serverless e2e NER recovery."""

    _instance: "FallbackChain | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        chain_file = Path("/tmp/npa-serverless-fallback-chain.txt")
        if not chain_file.exists():
            raise RuntimeError("Fallback chain file not found. Was Phase 0 run?")
        self._chain = [
            line.strip()
            for line in chain_file.read_text().splitlines()
            if line.strip()
        ]
        self._exhausted: set[str] = set()
        self._current_idx = 0
        self._selection = self._load_selection()

    @classmethod
    def instance(cls) -> "FallbackChain":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def current_project(self) -> str | None:
        while self._current_idx < len(self._chain):
            project_id = self._chain[self._current_idx]
            if project_id not in self._exhausted:
                return project_id
            self._current_idx += 1
        return None

    def mark_ner(self, project_id: str) -> str | None:
        self._exhausted.add(project_id)
        print(f"!!! NER on project {project_id}; rotating to next in fallback chain", flush=True)
        return self.current_project()

    def all_projects(self) -> list[str]:
        return list(self._chain)

    def exhausted_projects(self) -> set[str]:
        return set(self._exhausted)

    def project_key(self, project_id: str) -> str:
        mapping = self._selection.get("project_id_to_key") or {}
        return str(mapping.get(project_id) or project_id)

    def project_region(self, project_id: str) -> str:
        mapping = self._selection.get("project_id_to_region") or {}
        return str(mapping.get(project_id) or "")

    def _load_selection(self) -> dict:
        path = Path("/tmp/npa-serverless-project-selection.json")
        if not path.exists():
            return {}
        return json.loads(path.read_text())


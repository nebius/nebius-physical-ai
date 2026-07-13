from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.clients.config import list_projects, resolve_environment

CHAIN_PATH = Path("/tmp/npa-serverless-fallback-chain.txt")
SELECTION_PATH = Path("/tmp/npa-serverless-project-selection.json")
PREFER_REGION = "eu-north1"


@dataclass(frozen=True)
class ServerlessProjectSelection:
    """Resolved primary project plus NER fallback chain for serverless e2e."""

    primary_project_id: str
    chain: list[str]
    project_id_to_key: dict[str, str]
    project_id_to_region: dict[str, str]

    def as_selection_dict(self) -> dict[str, Any]:
        return {
            "primary_project_id": self.primary_project_id,
            "chain": list(self.chain),
            "project_id_to_key": dict(self.project_id_to_key),
            "project_id_to_region": dict(self.project_id_to_region),
        }


def _is_non_production(alias: str, project_id: str) -> bool:
    blob = f"{alias} {project_id}".lower()
    return "prod" not in blob and "production" not in blob


def _config_project_id(alias: str, cfg: dict[str, Any]) -> str:
    return str(cfg.get("project_id") or alias).strip()


def _config_region(cfg: dict[str, Any]) -> str:
    return str(cfg.get("region") or cfg.get("location") or cfg.get("zone") or "").strip()


def discover_serverless_projects(
    *,
    primary_project_id: str | None = None,
    prefer_region: str = PREFER_REGION,
) -> ServerlessProjectSelection:
    """Build a NER fallback chain from env + ``~/.npa/config.yaml``.

    Primary resolution order:
      1. explicit ``primary_project_id``
      2. ``NPA_E2E_SERVERLESS_PROJECT``
      3. first non-prod config entry whose region matches ``prefer_region``
      4. first non-prod config entry in declaration order
    """

    projects = list_projects()
    id_to_key: dict[str, str] = {}
    id_to_region: dict[str, str] = {}
    ordered_ids: list[str] = []
    prefer_ids: list[str] = []

    for alias, pdata in projects.items():
        if not isinstance(pdata, dict):
            continue
        project_id = _config_project_id(alias, pdata)
        if not project_id or not _is_non_production(alias, project_id):
            continue
        region = _config_region(pdata)
        if project_id not in id_to_key:
            ordered_ids.append(project_id)
            id_to_key[project_id] = alias
            id_to_region[project_id] = region
            if prefer_region and region == prefer_region:
                prefer_ids.append(project_id)

    env_primary = (
        str(primary_project_id or "").strip()
        or os.environ.get("NPA_E2E_SERVERLESS_PROJECT", "").strip()
    )
    if env_primary:
        primary = env_primary
        if primary not in id_to_key:
            # Env-only primary: map key to itself; try resolve_environment for region.
            id_to_key[primary] = primary
            region = ""
            try:
                env = resolve_environment(None, project_id=primary)
            except Exception:
                env = None
            if env is not None:
                region = str(getattr(env, "region", "") or "")
            id_to_region[primary] = region
    elif prefer_ids:
        primary = prefer_ids[0]
    elif ordered_ids:
        primary = ordered_ids[0]
    else:
        raise RuntimeError(
            "No serverless e2e project available. Set NPA_E2E_SERVERLESS_PROJECT "
            "or configure a non-production project in ~/.npa/config.yaml."
        )

    chain = [primary] + [pid for pid in ordered_ids if pid != primary]
    return ServerlessProjectSelection(
        primary_project_id=primary,
        chain=chain,
        project_id_to_key=id_to_key,
        project_id_to_region=id_to_region,
    )


def ensure_serverless_phase0(
    *,
    chain_path: Path = CHAIN_PATH,
    selection_path: Path = SELECTION_PATH,
    write_files: bool = True,
    force: bool = False,
) -> ServerlessProjectSelection:
    """Idempotent Phase 0 for serverless e2e.

    Reuses existing Phase 0 files when present (preserving run-wide NER
    exhaustion across processes). Otherwise discovers from env + config and
    optionally writes the cache files.
    """

    force = force or os.environ.get("NPA_E2E_SERVERLESS_RESET_PHASE0", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if chain_path.exists() and not force:
        chain = [
            line.strip()
            for line in chain_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selection: dict[str, Any] = {}
        if selection_path.exists():
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                selection = {}
        primary = str(selection.get("primary_project_id") or (chain[0] if chain else "")).strip()
        if not primary or not chain:
            # Corrupt cache — rediscover.
            pass
        else:
            return ServerlessProjectSelection(
                primary_project_id=primary,
                chain=chain,
                project_id_to_key={
                    str(k): str(v)
                    for k, v in (selection.get("project_id_to_key") or {}).items()
                },
                project_id_to_region={
                    str(k): str(v)
                    for k, v in (selection.get("project_id_to_region") or {}).items()
                },
            )

    discovered = discover_serverless_projects()
    if write_files:
        chain_path.write_text("\n".join(discovered.chain) + "\n", encoding="utf-8")
        selection_path.write_text(
            json.dumps(discovered.as_selection_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT", "").strip():
        os.environ["NPA_E2E_SERVERLESS_PROJECT"] = discovered.primary_project_id
    return discovered


class FallbackChain:
    """Run-wide project fallback chain for serverless e2e NER recovery."""

    _instance: "FallbackChain | None" = None
    _lock = threading.Lock()

    def __init__(self, selection: ServerlessProjectSelection | None = None) -> None:
        resolved = selection or ensure_serverless_phase0()
        if not resolved.chain:
            raise RuntimeError(
                "Serverless fallback chain is empty after Phase 0 discovery. "
                "Set NPA_E2E_SERVERLESS_PROJECT or configure ~/.npa/config.yaml."
            )
        self._chain = list(resolved.chain)
        self._exhausted: set[str] = set()
        self._current_idx = 0
        self._selection = resolved.as_selection_dict()

    @classmethod
    def instance(cls) -> "FallbackChain":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(ensure_serverless_phase0())
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Clear the singleton (unit tests only)."""

        with cls._lock:
            cls._instance = None

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

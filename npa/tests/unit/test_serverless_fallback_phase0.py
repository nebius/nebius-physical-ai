"""Unit tests for serverless e2e Phase 0 / FallbackChain auto-init."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HELPER = Path(__file__).resolve().parents[1] / "e2e" / "_serverless_fallback.py"
_SPEC = importlib.util.spec_from_file_location("npa_e2e_serverless_fallback", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

FallbackChain = _mod.FallbackChain
discover_serverless_projects = _mod.discover_serverless_projects
ensure_serverless_phase0 = _mod.ensure_serverless_phase0


@pytest.fixture(autouse=True)
def _reset_fallback_singleton() -> None:
    FallbackChain.reset_for_tests()
    yield
    FallbackChain.reset_for_tests()


def test_discover_prefers_env_primary_and_orders_chain(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_SERVERLESS_PROJECT", "project-primary")
    monkeypatch.setattr(
        _mod,
        "list_projects",
        lambda: {
            "eu-north1": {
                "project_id": "project-eu",
                "region": "eu-north1",
            },
            "us-central1": {
                "project_id": "project-us",
                "region": "us-central1",
            },
            "prod-keepout": {
                "project_id": "project-prod",
                "region": "eu-north1",
            },
        },
    )
    selection = discover_serverless_projects()
    assert selection.primary_project_id == "project-primary"
    assert selection.chain[0] == "project-primary"
    assert "project-eu" in selection.chain
    assert "project-us" in selection.chain
    assert "project-prod" not in selection.chain


def test_discover_prefers_eu_north1_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_SERVERLESS_PROJECT", raising=False)
    monkeypatch.setattr(
        _mod,
        "list_projects",
        lambda: {
            "us-central1": {"project_id": "project-us", "region": "us-central1"},
            "eu-north1": {"project_id": "project-eu", "region": "eu-north1"},
        },
    )
    selection = discover_serverless_projects()
    assert selection.primary_project_id == "project-eu"
    assert selection.chain == ["project-eu", "project-us"]
    assert selection.project_id_to_key["project-eu"] == "eu-north1"


def test_ensure_phase0_writes_and_reuses_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_E2E_SERVERLESS_PROJECT", "project-primary")
    monkeypatch.delenv("NPA_E2E_SERVERLESS_RESET_PHASE0", raising=False)
    monkeypatch.setattr(
        _mod,
        "list_projects",
        lambda: {"sandbox": {"project_id": "project-secondary", "region": "eu-north1"}},
    )
    chain_path = tmp_path / "chain.txt"
    selection_path = tmp_path / "selection.json"

    first = ensure_serverless_phase0(chain_path=chain_path, selection_path=selection_path)
    assert chain_path.exists()
    assert selection_path.exists()
    assert first.chain[0] == "project-primary"

    # Corrupt discovery source; existing files should still win.
    monkeypatch.setattr(_mod, "list_projects", lambda: {})
    second = ensure_serverless_phase0(chain_path=chain_path, selection_path=selection_path)
    assert second.chain == first.chain
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert payload["primary_project_id"] == "project-primary"


def test_fallback_chain_instance_auto_inits_without_phase0_files(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_E2E_SERVERLESS_PROJECT", "project-a")
    monkeypatch.setattr(
        _mod,
        "list_projects",
        lambda: {"a": {"project_id": "project-a", "region": "eu-north1"}},
    )
    original = _mod.ensure_serverless_phase0

    def _ensure(**kwargs):
        kwargs.setdefault("chain_path", tmp_path / "chain.txt")
        kwargs.setdefault("selection_path", tmp_path / "selection.json")
        return original(**kwargs)

    monkeypatch.setattr(_mod, "ensure_serverless_phase0", _ensure)
    chain = FallbackChain.instance()
    assert chain.current_project() == "project-a"
    assert chain.project_key("project-a") in {"a", "project-a"}

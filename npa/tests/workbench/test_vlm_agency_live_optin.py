"""Require explicit live opt-in before the agency test can resolve a provider."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("explicit", [False, True])
def test_agency_live_marker_uses_actual_collection_optin(monkeypatch, explicit):
    tests = Path(__file__).resolve().parents[1]
    live = _load(tests / "e2e/test_vlm_agency_calibration_live.py", "agency_live_optin")
    hooks = _load(tests / "e2e/conftest.py", "agency_live_collection")
    marks = {mark.name: mark for mark in live.pytestmark}
    assert {"e2e", "token_factory_e2e"} <= marks.keys()
    added = []
    item = SimpleNamespace(
        get_closest_marker=marks.get,
        add_marker=added.append,
    )
    if explicit:
        monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
        monkeypatch.setattr(hooks, "_hf_token_configured", lambda: True)
    else:
        monkeypatch.delenv("NPA_INTEGRATION_E2E", raising=False)
        monkeypatch.setattr(
            hooks,
            "_hf_token_configured",
            lambda: pytest.fail("default collection must not resolve credentials"),
        )
    hooks.pytest_collection_modifyitems(SimpleNamespace(), [item])
    assert [mark.name for mark in added] == ([] if explicit else ["skip"])
    if not explicit:
        assert added[0].kwargs["reason"] == "e2e tests require NPA_INTEGRATION_E2E=1"

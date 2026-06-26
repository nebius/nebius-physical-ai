from __future__ import annotations

import importlib.util
from pathlib import Path

_E2E_CONFTEST_PATH = Path(__file__).resolve().parent / "e2e" / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("npa_tests_e2e_conftest", _E2E_CONFTEST_PATH)
assert _SPEC and _SPEC.loader
e2e_conftest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(e2e_conftest)


def test_default_e2e_project_prefers_writable_default(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_PROJECT", raising=False)
    monkeypatch.setattr(e2e_conftest, "_storage_is_writable", lambda project: project is None)
    monkeypatch.setattr(e2e_conftest, "list_projects", lambda: {"eu-north1": {}, "other": {}})

    assert e2e_conftest._default_e2e_project() is None


def test_default_e2e_project_uses_writable_named_project(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_PROJECT", raising=False)

    def writable(project: str | None) -> bool:
        return project == "eu-north1"

    monkeypatch.setattr(e2e_conftest, "_storage_is_writable", writable)
    monkeypatch.setattr(e2e_conftest, "list_projects", lambda: {"other": {}, "eu-north1": {}})

    assert e2e_conftest._default_e2e_project() == "eu-north1"

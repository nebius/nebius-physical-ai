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


def test_pytest_configure_maps_aws_env_onto_npa_e2e_s3(monkeypatch) -> None:
    monkeypatch.delenv("NPA_E2E_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("NPA_E2E_S3_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("NPA_E2E_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("NPA_E2E_S3_BUCKET", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setenv("S3_BUCKET", "bucket-from-aws")

    class _Config:
        pass

    e2e_conftest.pytest_configure(_Config())
    assert e2e_conftest.os.environ["NPA_E2E_S3_ACCESS_KEY_ID"] == "AKIATEST"
    assert e2e_conftest.os.environ["NPA_E2E_S3_SECRET_ACCESS_KEY"] == "SECRET"
    assert e2e_conftest.os.environ["NPA_E2E_S3_ENDPOINT"] == "https://storage.example"
    assert e2e_conftest.os.environ["NPA_E2E_S3_BUCKET"] == "bucket-from-aws"

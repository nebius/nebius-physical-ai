"""Verify public demo selection, project isolation, and exact report retrieval."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.clients.config import ConfigError, StorageConfig
from npa.orchestration.npa_workflow import demos


@pytest.fixture
def configured_project(monkeypatch, tmp_path):
    storage = StorageConfig(
        "s3://example-bucket/operator-prefix",
        "https://example.invalid",
        "example-key",
        "example-secret",
    )
    monkeypatch.setattr(
        demos,
        "resolve_environment",
        lambda **kw: SimpleNamespace(project_id="fixture-project"),
    )
    monkeypatch.setattr(demos, "default_project_name", lambda: "configured")

    def selected_storage(**kwargs):
        assert kwargs == {
            "project": "configured",
            "include_shared_credentials": False,
            "include_environment": False,
        }
        return storage

    monkeypatch.setattr(demos, "resolve_project_storage", selected_storage)
    monkeypatch.setattr(
        demos, "resolve_npa_workflow_spec", lambda name: tmp_path / name
    )
    return storage


@pytest.mark.parametrize("name", [demo.name for demo in demos.DEMOS])
def test_full_demo_uses_project_storage_and_run_scoped_outputs(
    configured_project, name
):
    result = demos.prepare_demo(name, run_id="test-run")
    assert result["project"] == "configured"
    assert result["s3_prefix"] == f"operator-prefix/demos/{name}/test-run"
    assert (
        result["report_uri"]
        == f"s3://example-bucket/operator-prefix/demos/{name}/test-run/reports/index.html"
    )
    assert "source_overlay=true" in result["var"]
    assert "example-secret" not in repr(result)


def test_fresh_invocations_have_distinct_identities(configured_project):
    first = demos.prepare_demo("synthetic-data")
    second = demos.prepare_demo("synthetic-data")
    assert first["run_id"] != second["run_id"]
    assert first["s3_prefix"] != second["s3_prefix"]


@pytest.mark.parametrize(
    "run_id", ["../other", "foo/bar", "", "x?secret=1", "A", "a" * 64]
)
def test_unsafe_report_identity_fails_before_storage_resolution(monkeypatch, run_id):
    monkeypatch.setattr(
        demos, "_storage", lambda *a: pytest.fail("must not resolve credentials")
    )
    with pytest.raises(ValueError):
        demos.download_demo_report("nurec", run_id)


def test_unknown_project_cannot_use_global_storage(configured_project, monkeypatch):
    monkeypatch.setattr(demos, "resolve_environment", lambda **kw: None)
    with pytest.raises(ConfigError, match="not configured"):
        demos.prepare_demo("nurec", project="missing")


def test_missing_spec_does_not_silently_run_another_demo(
    configured_project, monkeypatch
):
    monkeypatch.setattr(demos, "resolve_npa_workflow_spec", lambda name: None)
    with pytest.raises(ValueError, match="complete demo revision"):
        demos.prepare_demo("real-to-sim")


def test_report_download_is_exact_and_uses_project_credential_pair(
    configured_project, monkeypatch, tmp_path
):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path))
    calls = []

    class Storage:
        def __init__(self, **kwargs):
            assert kwargs["aws_access_key_id"] == "example-key"
            assert kwargs["aws_secret_access_key"] == "example-secret"
            assert kwargs["endpoint_url"] == "https://example.invalid"

        def download_file(self, uri, path):
            calls.append(uri)
            Path(path).write_text("<!doctype html><p>Measured sample</p>")

    monkeypatch.setattr(demos, "StorageClient", Storage)
    path = demos.download_demo_report("nurec", "test-run")
    assert calls == [
        "s3://example-bucket/operator-prefix/demos/nurec/test-run/reports/index.html"
    ]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_partial_project_credentials_never_adopt_ambient_credentials(
    configured_project, monkeypatch
):
    configured_project.aws_secret_access_key = ""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unrelated-secret")
    monkeypatch.setattr(
        demos, "StorageClient", lambda **kw: pytest.fail("no client for a partial pair")
    )
    with pytest.raises(ConfigError, match="complete S3 credential pair"):
        demos.download_demo_report("nurec", "test-run")

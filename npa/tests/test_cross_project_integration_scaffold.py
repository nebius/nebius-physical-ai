from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.cli.demo import stage_artifacts
from npa.errors import ScopedCredentialError
from fakes import _access_denied, _fake_s3_factory, _manifest


def test_mock_cross_project_creds_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch, host_mode="default")

    result = stage_artifacts(
        target_bucket="target",
        manifest_path=_manifest(tmp_path / "manifest.yaml"),
        source_project="project-source",
        target_project="project-target",
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert clients["src-key"].get_calls == [("source", "path/file.bin")]
    assert clients["tgt-key"].put_calls == [("target", "staged/file.bin")]


def test_mock_cross_project_creds_target_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch, host_mode="default")
    clients["tgt-key"].fail_put = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-target"):
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )


def test_mock_cross_project_creds_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch, host_mode="default")
    clients["src-key"].fail_get = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-source"):
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )


def test_mock_cross_project_creds_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
    caplog,
) -> None:
    clients = _fake_s3_factory(monkeypatch, host_mode="default")
    clients["tgt-key"].fail_put = _access_denied()

    with caplog.at_level("WARNING"):
        result = stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
            allow_host_creds=True,
        )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert "falling back to host credentials" in caplog.text


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_CROSS_PROJECT") != "1",
    reason="set NPA_INTEGRATION_CROSS_PROJECT=1 and provide real test projects",
)
def test_live_cross_project_demo_stage_placeholder() -> None:
    pytest.skip(
        "Live cross-project wiring requires two Nebius test projects, distinct "
        "scoped principals, and one bucket in each project."
    )

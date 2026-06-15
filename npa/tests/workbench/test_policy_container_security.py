from __future__ import annotations

from pathlib import Path

import pytest

from npa.workbench.lerobot.policy_container import (
    PolicyContainerError,
    jail_output_dir,
)


def test_jail_output_dir_defaults_under_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path))
    resolved = jail_output_dir(None, default_name="feedback")
    assert resolved == (tmp_path / "feedback").resolve()


def test_jail_output_dir_allows_relative_subdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path))
    resolved = jail_output_dir("run-1/adapters", default_name="feedback")
    assert resolved == (tmp_path / "run-1" / "adapters").resolve()
    assert tmp_path.resolve() in resolved.parents


def test_jail_output_dir_rejects_parent_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    with pytest.raises(PolicyContainerError):
        jail_output_dir("../../etc/cron.d", default_name="feedback")


def test_jail_output_dir_rejects_absolute_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    with pytest.raises(PolicyContainerError):
        jail_output_dir("/etc/passwd", default_name="feedback")


def test_feedback_endpoint_rejects_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    monkeypatch.delenv("NPA_POLICY_CHECKPOINT", raising=False)

    from npa.workbench.lerobot.policy_container import create_app

    client = fastapi_testclient.TestClient(create_app())
    response = client.post(
        "/feedback/train-step",
        json={"feedback": [], "output_dir": "/etc/cron.d/evil"},
    )
    assert response.status_code == 400
    assert "output_dir" in response.json()["detail"]

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.lerobot.policy_container import (
    PolicyContainerError,
    jail_output_dir,
)


def test_jail_output_dir_defaults_under_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path))
    resolved = jail_output_dir(None, default_name="feedback")
    assert resolved == (tmp_path / "feedback").resolve()


def test_jail_output_dir_allows_relative_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path))
    resolved = jail_output_dir("run-1/adapters", default_name="feedback")
    assert resolved == (tmp_path / "run-1" / "adapters").resolve()
    assert tmp_path.resolve() in resolved.parents


def test_jail_output_dir_rejects_parent_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    with pytest.raises(PolicyContainerError):
        jail_output_dir("../../etc/cron.d", default_name="feedback")


def test_jail_output_dir_rejects_absolute_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    with pytest.raises(PolicyContainerError):
        jail_output_dir("/etc/passwd", default_name="feedback")


def test_feedback_endpoint_rejects_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


@pytest.mark.parametrize("control", ["false", "true", 0, 1, [], {}, None])
def test_vlm_signal_endpoint_rejects_non_boolean_control_before_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    output_root = tmp_path / "jail"
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(output_root))
    monkeypatch.delenv("NPA_POLICY_CHECKPOINT", raising=False)

    import npa.workbench.lerobot.policy_container as policy_container

    def unexpected_update(*_args, **_kwargs):
        pytest.fail("malformed control reached the policy update")

    monkeypatch.setattr(
        policy_container, "run_vlm_signal_training_step", unexpected_update
    )
    client = fastapi_testclient.TestClient(policy_container.create_app())
    response = client.post(
        "/feedback/train-step",
        json={"signals": [{}], "control": control},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "control must be a boolean"}
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("request_fields", "expected_control"),
    [({}, False), ({"control": False}, False), ({"control": True}, True)],
)
def test_vlm_signal_endpoint_preserves_boolean_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_fields: dict,
    expected_control: bool,
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(tmp_path / "jail"))
    monkeypatch.delenv("NPA_POLICY_CHECKPOINT", raising=False)

    import npa.workbench.lerobot.policy_container as policy_container

    observed = []

    def record_update(*_args, **kwargs):
        observed.append(kwargs["control"])
        return SimpleNamespace(to_dict=lambda: {"control": kwargs["control"]})

    monkeypatch.setattr(policy_container, "parse_vlm_signal_batch", lambda _payload: [])
    monkeypatch.setattr(policy_container, "run_vlm_signal_training_step", record_update)
    client = fastapi_testclient.TestClient(policy_container.create_app())
    response = client.post(
        "/feedback/train-step",
        json={"signals": [{}], **request_fields},
    )

    assert response.status_code == 200
    assert response.json() == {"control": expected_control}
    assert observed == [expected_control]

"""Regression coverage for strict LeRobot feedback parsing."""

from pathlib import Path
from typing import Any

import pytest

from npa.workbench.lerobot.policy_container import (
    PolicyContainerError,
    _boolean_field,
    _feedback_reward,
    parse_feedback_batch,
)


@pytest.mark.parametrize(
    ("success", "expected_reward"),
    [(False, 0.5), (True, 0.9)],
)
def test_feedback_boolean_success_preserves_reward_semantics(
    success: bool, expected_reward: float
) -> None:
    feedback = parse_feedback_batch(
        {"success": success, "score": 0.9, "rationale": "reviewed"}
    )[0]

    assert feedback.success is success
    assert _feedback_reward(feedback) == expected_reward


@pytest.mark.parametrize(
    "success",
    ["false", "False", "no", 0, 1, 0.0, None, [False], [], {}],
)
def test_feedback_rejects_non_boolean_success(success: Any) -> None:
    with pytest.raises(PolicyContainerError, match=r"^success must be a boolean$"):
        parse_feedback_batch(
            {"success": success, "score": 0.9, "rationale": "malformed"}
        )


def test_feedback_requires_success_despite_optional_boolean_defaults() -> None:
    with pytest.raises(PolicyContainerError, match="missing required keys: success"):
        parse_feedback_batch({"score": 0.9, "rationale": "missing verdict"})


@pytest.mark.parametrize("default", [False, True])
def test_boolean_field_preserves_omitted_default_and_rejects_null(
    default: bool,
) -> None:
    assert _boolean_field({}, "control", default=default) is default
    with pytest.raises(PolicyContainerError, match="control must be a boolean"):
        _boolean_field({"control": None}, "control", default=default)


@pytest.mark.parametrize("success", ["false", [False]])
@pytest.mark.parametrize("valid_prefix", [False, True])
def test_feedback_endpoint_rejects_non_boolean_success_before_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, success: Any, valid_prefix: bool
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    output_root = tmp_path / "jail"
    monkeypatch.setenv("NPA_POLICY_OUTPUT_ROOT", str(output_root))
    monkeypatch.delenv("NPA_POLICY_CHECKPOINT", raising=False)

    import npa.workbench.lerobot.policy_container as policy_container

    def unexpected_update(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("malformed success reached the policy update")

    monkeypatch.setattr(
        policy_container, "run_feedback_training_step", unexpected_update
    )
    client = fastapi_testclient.TestClient(policy_container.create_app())
    malformed = {"success": success, "score": 0.9, "rationale": "malformed"}
    payload = (
        {"feedback": [{"success": True, "score": 0.9, "rationale": "valid"}, malformed]}
        if valid_prefix
        else malformed
    )
    response = client.post(
        "/feedback/train-step",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "success must be a boolean"}
    assert not output_root.exists()

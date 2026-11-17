"""Check matched-trial arithmetic and binding to the actual training receipt."""

import json

import pytest

from npa.workbench.groot.offline_evaluation import compare_rollouts


@pytest.fixture
def comparison_inputs(tmp_path):
    receipt = {
        "schema_version": "npa.groot-offline-training/v1",
        "base_checkpoint_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
    }
    rollouts = {
        "schema_version": "npa.groot-offline-rollouts/v1",
        "baseline_checkpoint_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "trials": [
            {
                "trial_id": "trial-0",
                "baseline_success": False,
                "candidate_success": True,
            },
            {
                "trial_id": "trial-1",
                "baseline_success": True,
                "candidate_success": True,
            },
            {
                "trial_id": "trial-2",
                "baseline_success": True,
                "candidate_success": False,
            },
        ],
    }
    return tmp_path, receipt, rollouts


def _compare(inputs):
    directory, receipt, rollouts = inputs
    receipt_path = directory / "training.json"
    receipt_path.write_text(json.dumps(receipt))
    rollouts_path = directory / "rollouts.json"
    rollouts_path.write_text(json.dumps(rollouts))
    return compare_rollouts(rollouts_path, receipt_path, directory / "comparison.json")


def test_matched_rates_intervals_and_operator_decision(comparison_inputs):
    report = _compare(comparison_inputs)
    assert report["trials"] == 3
    assert report["baseline"]["successes"] == report["candidate"]["successes"] == 2
    assert report["candidate"]["rate"] == pytest.approx(2 / 3)
    assert report["success_rate_delta"] == 0
    assert report["paired_wins"] == report["paired_losses"] == 1
    low, high = report["candidate"]["wilson_95"]
    assert low == pytest.approx(0.2076596008)
    assert high == pytest.approx(0.9385080553)
    assert report["deployment_decision"] == "operator-review-required"
    assert "robot_evaluation_performed" not in report


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update(candidate_checkpoint_sha256="d" * 64),
        lambda payload: payload.update(baseline_checkpoint_sha256="b" * 64),
        lambda payload: payload.update(trials=[]),
        lambda payload: payload["trials"].append(payload["trials"][0]),
        lambda payload: payload["trials"][0].update(candidate_success="false"),
        lambda payload: payload["trials"][0].update(candidate_success=1),
        lambda payload: payload.update(protocol_sha256="unbound"),
    ],
)
def test_invalid_or_mismatched_evaluation_refuses(comparison_inputs, change):
    change(comparison_inputs[2])
    with pytest.raises(ValueError):
        _compare(comparison_inputs)
    assert not (comparison_inputs[0] / "comparison.json").exists()


def test_comparison_does_not_overwrite_report(comparison_inputs):
    _compare(comparison_inputs)
    with pytest.raises(FileExistsError):
        _compare(comparison_inputs)


def test_training_receipt_must_be_an_object(comparison_inputs):
    directory, _, rollouts = comparison_inputs
    with pytest.raises(ValueError, match="training receipt"):
        _compare((directory, [], rollouts))

"""Verify reference learning, checkpoint integrity, and private-to-public report boundaries."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.dataset.storage import write_json_uri
from npa.workflows.policy_training import batch as batch_module
from npa.workflows.policy_training.demo import _LocalReferenceExecutor
from npa.workflows.policy_training.demo_report import _history
from npa.workflows.policy_training.reference_model import (
    evaluate,
    load_weights,
    rollout,
    train,
)


def _training_records():
    generator = np.random.default_rng(8)
    states = generator.uniform(-0.8, 0.8, (24, 4))
    actions = 0.28 * (states[:, 2:] - states[:, :2])
    return {0: (states, actions)}


def test_updates_reduce_loss_resume_weights_and_improve_actual_rollout(tmp_path):
    weights = load_weights(None)
    before = rollout(weights, [-0.6, 0.4], [0.6, 0.8], 0.85)
    request = {"stage": "pretrain", "partition": "train"}
    identities = []
    for attempt in range(4):
        request["result_uri"] = str(tmp_path / str(attempt) / "result.json")
        identity, losses = train(request, _training_records(), weights)
        assert losses[1] < losses[0]
        identities.append(identity["sha256"])
        weights = load_weights(identity)
    after = rollout(weights, [-0.6, 0.4], [0.6, 0.8], 0.85)
    assert not before["success"] and after["success"]
    assert after["distance"] < before["distance"]
    assert len(set(identities)) == 4
    assert len(after["positions"]) == 17


def test_reference_rejects_holdout_training_and_training_evaluation(tmp_path):
    request = {
        "stage": "pretrain",
        "partition": "holdout_1",
        "result_uri": str(tmp_path / "result.json"),
    }
    with pytest.raises(ValueError, match="cannot consume holdouts"):
        train(request, _training_records(), load_weights(None))
    with pytest.raises(ValueError, match="requires a holdout"):
        evaluate(
            {"stage": "evaluate-pretrain", "partition": "train"}, {}, load_weights(None)
        )


def test_reference_reports_counts_from_trajectories():
    request = {"stage": "evaluate-pretrain", "partition": "holdout_1"}
    systems, trajectories = evaluate(request, _training_records(), load_weights(None))
    for name, trials in trajectories.items():
        assert systems[name]["successes"] == sum(t["distance"] <= 0.04 for t in trials)
        assert systems[name]["trials"] == len(trials) == 4


@pytest.mark.parametrize("invalid", ["digest", "shape", "nonfinite"])
def test_checkpoint_rejects_modified_bytes_and_invalid_arrays(tmp_path, invalid):
    weights = np.zeros((2, 2) if invalid == "shape" else (3, 2))
    if invalid == "nonfinite":
        weights[0, 0] = np.nan
    buffer = io.BytesIO()
    np.save(buffer, weights, allow_pickle=False)
    path = tmp_path / "checkpoint.npy"
    path.write_bytes(buffer.getvalue())
    identity = {
        "uri": str(path),
        "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
    }
    if invalid == "digest":
        path.write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError):
        load_weights(identity)


def test_converged_optimizer_fails_instead_of_inventing_a_gate_pass(tmp_path):
    weights = np.array([[0.28, 0], [0, 0.28], [0, 0]])
    request = {
        "stage": "pretrain",
        "partition": "train",
        "result_uri": str(tmp_path / "result.json"),
    }
    with pytest.raises(ValueError, match="converged without a passing gate"):
        train(request, _training_records(), weights)


def test_production_rejects_reference_transport():
    with pytest.raises(ValueError, match="production transport"):
        batch_module._submit(
            {"transport": "reference-local"}, "pretrain", "/private/request.json"
        )


@pytest.mark.parametrize(
    "uri",
    [
        "s3://other-bucket/key",
        "s3://example-bucket/policy-training/reference-demo/../../secret",
    ],
)
def test_reference_executor_rejects_external_storage_and_path_escape(tmp_path, uri):
    executor = _LocalReferenceExecutor(tmp_path)
    with pytest.raises(ValueError):
        executor.local(uri)


def test_report_history_drops_worker_paths_and_arbitrary_private_fields(tmp_path):
    directory = tmp_path / "pretrain/1"
    directory.mkdir(parents=True)
    private = "PRIVATE_SENTINEL_DO_NOT_EXPORT"
    common = {"checkpoint": {"uri": private, "sha256": "a" * 64}, "private": private}
    write_json_uri(
        str(directory / "decision.json"),
        {
            **common,
            "decision": "promote_checkpoint",
            "success_rates": {"nominal": 1},
            "thresholds": {"nominal": 0.9},
        },
    )
    write_json_uri(
        str(directory / "candidate.json"), {**common, "training_loss": [1, 0.1]}
    )
    write_json_uri(
        str(directory / "rollouts.json"),
        {"systems": {"nominal": []}, "private": private},
    )
    write_json_uri(
        str(directory / "evaluation.json"),
        {
            **common,
            "rollouts_uri": str(directory / "rollouts.json"),
            "systems": {"nominal": {"successes": 1, "trials": 1}},
        },
    )
    payload = json.dumps(_history(tmp_path, "pretrain"))
    assert private not in payload
    assert str(tmp_path) not in payload


def test_report_template_is_offline_and_blocks_external_connections():
    template = (
        Path(__file__).resolve().parents[2]
        / "src/npa/workflows/policy_training/demo_report.html"
    )
    content = template.read_text()
    assert "connect-src 'none'" in content
    assert 'src="http' not in content
    assert 'href="http' not in content

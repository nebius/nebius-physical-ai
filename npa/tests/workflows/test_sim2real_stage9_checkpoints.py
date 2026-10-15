"""Keep stronger periodic actors and their exact identities across canonical retries."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import workflow_stage
from npa.workflows.sim2real.checkpoint_selection import (
    checkpoint_candidate_has_complete_identity,
    require_complete_checkpoint_candidate_pool,
    resolve_selected_checkpoint,
    select_best_checkpoint,
    stage9_checkpoint_candidate,
)
from npa.workflows.sim2real.stage9_checkpoints import (
    validate_replay_checkpoint_pool,
    validation_checkpoints,
)
from npa.workflows.sim2real.stage9_replay import existing_replay


def _update():
    return {
        "ppo": {"iterations": 2000},
        "ppo_telemetry": {"observed_iterations": 2000},
        "checkpoint_path": "s3://unit/train/model_latest.pt",
        "periodic_checkpoints": [
            {
                "training_iteration": index,
                "checkpoint_completed_updates": index + 1,
                "checkpoint_iteration_semantics": "last_completed_zero_based",
                "checkpoint_uri": f"s3://unit/train/checkpoints/model_{index}.pt",
            }
            for index in (100, 1999)
        ],
    }


class _ValidationEvaluator:
    def __init__(self):
        self.calls = []

    def __call__(self, _args, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls)
        return {
            "success_rate": 0.75 if index == 1 else 0.0,
            "policy_checkpoint_sha256": str(index) * 64,
            "policy_checkpoint_size_bytes": 100 + index,
            "policy_inference_provenance": {
                "generator_policy_sha256": str(index) * 64,
            },
            "per_env": [
                {"env_id": f"validation-{env}", "success": index == 1 and env < 48}
                for env in range(64)
            ],
        }


def test_canonical_sweep_keeps_stronger_periodic_actor_after_full_pass(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        root_uri="s3://unit/run",
        run_id="run",
        ppo_iterations=2000,
        outer_iteration=1,
        inner_iteration=1,
        validation_count=64,
    )
    writes = []

    evaluator = _ValidationEvaluator()
    monkeypatch.setattr(workflow_stage, "_run_eval", evaluator)
    monkeypatch.setattr(workflow_stage, "_assert_embodiment_evidence", lambda **_: {})
    monkeypatch.setattr(
        workflow_stage, "write_json", lambda uri, *_a, **_k: writes.append(uri)
    )
    candidates = workflow_stage._stage9_validation_candidates(
        args, tmp_path, _update(), {}
    )
    selected = select_best_checkpoint(candidates)
    assert args.ppo_iterations == 2000
    assert selected["training_iteration"] == 100
    assert selected["checkpoint_completed_updates"] == 101
    assert selected["checkpoint_uri"].endswith("model_100.pt")
    assert selected["checkpoint_size_bytes"] == 101
    assert selected["generator_policy_sha256"] == "1" * 64
    selection, candidate = resolve_selected_checkpoint(
        {
            "selected_checkpoint_uri": selected["checkpoint_uri"],
            "final_checkpoint_uri": selected["checkpoint_uri"],
            "checkpoint_selection": selected,
            "checkpoint_candidates": candidates,
        }
    )
    assert selection["checkpoint_size_bytes"] == candidate["checkpoint_size_bytes"]
    assert selection["generator_policy_sha256"] == candidate["generator_policy_sha256"]
    assert len(writes) == len(set(writes)) == 2
    assert all(
        call["split"] == "validation" and call["env_count"] == 64
        for call in evaluator.calls
    )
    assert len({call["tag"] for call in evaluator.calls}) == 2


def test_stage9_candidate_rejects_incomplete_validation_identity() -> None:
    args = SimpleNamespace(outer_iteration=1, inner_iteration=1)
    checkpoint = {
        "checkpoint_uri": "s3://sim2real-test/checkpoints/model_000001.pt",
        "training_iteration": 1,
    }
    validation = {
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 1,
        "policy_inference_provenance": {},
    }
    with pytest.raises(RuntimeError, match="identity is incomplete"):
        stage9_checkpoint_candidate(
            checkpoint,
            validation,
            "s3://sim2real-test/eval/validation/report.json",
            {"robot": "Franka"},
            outer_iteration=args.outer_iteration,
            inner_iteration=args.inner_iteration,
        )


def test_checkpoint_candidate_pool_rejects_incomplete_byte_identity() -> None:
    candidate = {
        "checkpoint_uri": "s3://sim2real-test/checkpoints/model_000001.pt",
        "checkpoint_sha256": "a" * 64,
        "generator_policy_sha256": "a" * 64,
        "checkpoint_size_bytes": 1,
    }
    del candidate["checkpoint_size_bytes"]

    with pytest.raises(ValueError, match="lacks complete checkpoint byte identity"):
        require_complete_checkpoint_candidate_pool([candidate])


def test_stage9_commit_rejects_incomplete_retained_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    incomplete = {
        "checkpoint_uri": "s3://sim2real-test/checkpoints/model_000001.pt",
        "checkpoint_sha256": "a" * 64,
        "generator_policy_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        workflow_stage, "_stage9_validation_candidates", lambda *_args: []
    )

    with pytest.raises(RuntimeError, match="without complete byte identity"):
        workflow_stage._stage9_commit(
            SimpleNamespace(),
            "s3://sim2real-test/run",
            tmp_path,
            {},
            {"checkpoint_candidates": [incomplete]},
            {},
            {},
            "s3://sim2real-test/run/evidence.json",
        )


def test_checkpoint_identity_accepts_a_zero_based_initial_checkpoint() -> None:
    candidate = {
        "checkpoint_uri": "s3://sim2real-test/checkpoints/model_000000.pt",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 1,
        "generator_policy_sha256": "a" * 64,
        "training_iteration": 0,
        "checkpoint_completed_updates": 1,
        "checkpoint_iteration_semantics": "last_completed_zero_based",
    }
    assert checkpoint_candidate_has_complete_identity(candidate)
    evidence = {
        "selected_checkpoint_uri": candidate["checkpoint_uri"],
        "final_checkpoint_uri": candidate["checkpoint_uri"],
        "checkpoint_selection": candidate,
        "checkpoint_candidates": [copy.deepcopy(candidate)],
    }
    selection, resolved_candidate = resolve_selected_checkpoint(evidence)
    assert (
        selection["training_iteration"] == resolved_candidate["training_iteration"] == 0
    )


@pytest.mark.parametrize(
    ("record_name", "field", "value"),
    [
        ("checkpoint_selection", "checkpoint_completed_updates", 2),
        ("checkpoint_candidates", "checkpoint_completed_updates", True),
        ("checkpoint_selection", "checkpoint_iteration_semantics", "completed_updates"),
        ("checkpoint_candidates", "checkpoint_iteration_semantics", None),
    ],
)
def test_checkpoint_identity_rejects_unproven_zero_based_checkpoint(
    record_name, field, value
) -> None:
    candidate = {
        "checkpoint_uri": "s3://sim2real-test/checkpoints/model_000000.pt",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 1,
        "generator_policy_sha256": "a" * 64,
        "training_iteration": 0,
        "checkpoint_completed_updates": 1,
        "checkpoint_iteration_semantics": "last_completed_zero_based",
    }
    evidence = {
        "selected_checkpoint_uri": candidate["checkpoint_uri"],
        "final_checkpoint_uri": candidate["checkpoint_uri"],
        "checkpoint_selection": candidate,
        "checkpoint_candidates": [copy.deepcopy(candidate)],
    }
    record = evidence[record_name]
    if record_name == "checkpoint_candidates":
        record = record[0]
    record[field] = value
    with pytest.raises(ValueError, match="training_iteration sources disagree"):
        resolve_selected_checkpoint(evidence)


@pytest.mark.parametrize(
    "corruption",
    ["empty", "duplicate", "bool", "completed", "semantics", "partial_pass"],
)
def test_checkpoint_pool_refuses_inexact_or_ambiguous_update_accounting(corruption):
    update = _update()
    rows = update["periodic_checkpoints"]
    if corruption == "empty":
        rows.clear()
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "bool":
        rows[0]["training_iteration"] = True
    elif corruption == "completed":
        rows[0]["checkpoint_completed_updates"] = 2000
    elif corruption == "semantics":
        rows[0]["checkpoint_iteration_semantics"] = "completed_updates"
    else:
        update["ppo_telemetry"]["observed_iterations"] = 101
    with pytest.raises(RuntimeError):
        validation_checkpoints(update, configured_updates=2000)


def test_complete_periodic_validation_pool_can_replay():
    update = _update()
    candidates = validation_checkpoints(update, configured_updates=2000)
    validate_replay_checkpoint_pool(update, candidates)
    for changed in (candidates[:1], candidates + candidates[:1]):
        with pytest.raises(RuntimeError, match="partial or inconsistent"):
            validate_replay_checkpoint_pool(update, changed)


def _committed_replay():
    update = _update()
    evaluator = _ValidationEvaluator()
    candidates = [
        {
            **checkpoint,
            "evaluation_split": "validation",
            "outer_iteration": 1,
            "inner_iteration": 1,
            "checkpoint_sha256": str(index + 1) * 64,
            "checkpoint_size_bytes": 100 + index + 1,
            "generator_policy_sha256": str(index + 1) * 64,
            "validation_report_uri": f"s3://unit/validation/{index}.json",
            "validation_report": evaluator(None),
        }
        for index, checkpoint in enumerate(update["periodic_checkpoints"])
    ]
    selection = select_best_checkpoint(candidates)
    inputs = {
        "iteration": 1,
        "actions_uri": "s3://unit/actions/",
        "vlm_eval_uri": "s3://unit/evaluations/",
        "signal_uri": "s3://unit/signals/",
        "sample_vlm_eval": {"rollout_id": "rollout-1"},
        "sample_signal": {"rollout_id": "rollout-1"},
    }
    prior = {
        "outer_iteration": 1,
        "iterations": [{**inputs, "update": update}],
        "checkpoint_candidates": candidates,
        "selected_checkpoint_uri": selection["checkpoint_uri"],
        "final_checkpoint_uri": selection["checkpoint_uri"],
        "checkpoint_selection": selection,
        "selected_validation_report": selection["validation_report"],
    }
    return prior, inputs


def test_same_iteration_retry_adopts_earlier_checkpoint_and_full_ppo_evidence():
    prior, inputs = _committed_replay()
    candidate, selection, update = existing_replay(
        prior=prior,
        outer_iteration=1,
        inner_iteration=1,
        actions_uri=inputs["actions_uri"],
        evaluation_uri=inputs["vlm_eval_uri"],
        signal_uri=inputs["signal_uri"],
        sample_vlm_eval=inputs["sample_vlm_eval"],
        sample_signal=inputs["sample_signal"],
    )
    assert candidate["checkpoint_uri"] == selection["checkpoint_uri"]
    assert candidate["checkpoint_completed_updates"] == 101
    assert update["ppo_telemetry"]["observed_iterations"] == 2000
    prior["checkpoint_candidates"].pop()
    with pytest.raises(RuntimeError, match="partial or inconsistent"):
        existing_replay(
            prior=prior,
            outer_iteration=1,
            inner_iteration=1,
            actions_uri=inputs["actions_uri"],
            evaluation_uri=inputs["vlm_eval_uri"],
            signal_uri=inputs["signal_uri"],
            sample_vlm_eval=inputs["sample_vlm_eval"],
            sample_signal=inputs["sample_signal"],
        )


def test_same_iteration_retry_rejects_incomplete_checkpoint_identity():
    prior, inputs = _committed_replay()
    del prior["checkpoint_candidates"][0]["checkpoint_size_bytes"]

    with pytest.raises(RuntimeError, match="lacks complete byte identity"):
        existing_replay(
            prior=prior,
            outer_iteration=1,
            inner_iteration=1,
            actions_uri=inputs["actions_uri"],
            evaluation_uri=inputs["vlm_eval_uri"],
            signal_uri=inputs["signal_uri"],
            sample_vlm_eval=inputs["sample_vlm_eval"],
            sample_signal=inputs["sample_signal"],
        )

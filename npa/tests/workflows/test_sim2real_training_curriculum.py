"""Verify that resume exploration uses the selected simulator validation only."""

from __future__ import annotations

import copy
import json
from argparse import Namespace

import pytest

from npa.workflows.sim2real.training_curriculum import resume_environment


def _candidate(reliable_lifts: int, iteration: int = 1) -> dict:
    checkpoint = f"s3://example-bucket/sim2real/run/checkpoints/model-{iteration}.pt"
    checksum = str(iteration) * 64
    return {
        "evaluation_split": "validation",
        "outer_iteration": 1,
        "inner_iteration": iteration,
        "checkpoint_uri": checkpoint,
        "checkpoint_sha256": checksum,
        "validation_report_uri": f"s3://example-bucket/sim2real/run/validation/{iteration}.json",
        "validation_report": {
            "evaluation_split": "validation",
            "policy_checkpoint": checkpoint,
            "policy_checkpoint_sha256": checksum,
            "success_rate": 0.0,
            "decomposed_metrics": {"lift": {"rate": reliable_lifts / 64}},
            "per_env": [
                {
                    "details": {
                        "stable_grasp": index < reliable_lifts,
                        "lift": index < reliable_lifts,
                        "closest_object_goal_distance_m": 0.06
                        if index < reliable_lifts
                        else 0.4,
                    }
                }
                for index in range(64)
            ],
        },
    }


@pytest.mark.parametrize(
    "reliable_lifts,phase",
    [(0, "exploration"), (31, "exploration"), (32, "convergence"), (64, "convergence")],
)
def test_resume_phase_follows_actual_validation_skill(reliable_lifts, phase):
    environment = resume_environment(
        {"checkpoint_candidates": [_candidate(reliable_lifts)]}
    )
    assert environment["NPA_BYO_ISAAC_RESUME_PHASE"] == phase
    audit = json.loads(environment["NPA_SIM2REAL_RESUME_CURRICULUM_JSON"])
    assert audit["stable_grasp_and_lift_episodes"] == reliable_lifts
    assert audit["validation_episodes"] == 64
    assert audit["decision_source"] == "simulator_validation_only"


def test_resume_uses_better_validation_checkpoint_instead_of_latest():
    best, newer = _candidate(40), _candidate(2, iteration=2)
    environment = resume_environment({"checkpoint_candidates": [newer, best]})
    assert environment["NPA_SIM2REAL_RESUME_CHECKPOINT_URI"] == best["checkpoint_uri"]
    assert environment["NPA_SIM2REAL_POLICY_CHECKPOINT_URI"] == best["checkpoint_uri"]
    assert (
        environment["NPA_SIM2REAL_RESUME_CHECKPOINT_SHA256"]
        == best["checkpoint_sha256"]
    )
    assert environment["NPA_BYO_ISAAC_AUTO_RESUME"] == "0"


@pytest.mark.parametrize("near_goal_episodes", [0, 31, 32, 64])
def test_resume_retains_trainable_transport_until_paired_goal_approach(
    near_goal_episodes,
):
    candidate = _candidate(64)
    for index, row in enumerate(candidate["validation_report"]["per_env"]):
        row["details"]["closest_object_goal_distance_m"] = (
            0.06 if index < near_goal_episodes else 0.2
        )
    environment = resume_environment({"checkpoint_candidates": [candidate]})
    assert environment["NPA_BYO_ISAAC_RESUME_PHASE"] == (
        "convergence" if near_goal_episodes >= 32 else "transport"
    )
    audit = json.loads(environment["NPA_SIM2REAL_RESUME_CURRICULUM_JSON"])
    assert audit["goal_approach_episodes"] == near_goal_episodes
    assert audit["goal_approach_distance_m"] == 0.08


def test_missing_goal_distance_never_freezes_an_otherwise_reliable_checkpoint():
    candidate = _candidate(64)
    for row in candidate["validation_report"]["per_env"]:
        del row["details"]["closest_object_goal_distance_m"]
    environment = resume_environment({"checkpoint_candidates": [candidate]})
    assert environment["NPA_BYO_ISAAC_RESUME_PHASE"] == "transport"


@pytest.mark.parametrize("distance", [True, -0.1, float("nan"), float("inf"), "0.01"])
def test_resume_rejects_invalid_goal_distance(distance):
    candidate = _candidate(64)
    candidate["validation_report"]["per_env"][0]["details"][
        "closest_object_goal_distance_m"
    ] = distance
    with pytest.raises(ValueError, match="finite nonnegative distance"):
        resume_environment({"checkpoint_candidates": [candidate]})


@pytest.mark.parametrize(
    "mutation", ["gold", "checkpoint", "sha", "checksum", "truth", "missing"]
)
def test_resume_rejects_unbound_or_invalid_validation(mutation):
    candidate = copy.deepcopy(_candidate(40))
    report = candidate["validation_report"]
    if mutation == "gold":
        report["evaluation_split"] = "gold_heldout"
    elif mutation == "checkpoint":
        report["policy_checkpoint"] = "s3://example-bucket/other.pt"
    elif mutation == "sha":
        report["policy_checkpoint_sha256"] = "a" * 64
    elif mutation == "checksum":
        candidate["checkpoint_sha256"] = ""
    elif mutation == "truth":
        report["per_env"][0]["details"]["stable_grasp"] = 1
    else:
        del report["per_env"][0]["details"]["lift"]
    with pytest.raises(ValueError):
        resume_environment({"checkpoint_candidates": [candidate]})


def test_first_pass_disables_unvalidated_automatic_resume():
    assert resume_environment({}) == {"NPA_BYO_ISAAC_AUTO_RESUME": "0"}


def test_grasp_and_lift_must_succeed_in_the_same_validation_episode():
    candidate = _candidate(32)
    for index, row in enumerate(candidate["validation_report"]["per_env"]):
        row["details"]["lift"] = index >= 32
    environment = resume_environment({"checkpoint_candidates": [candidate]})
    assert environment["NPA_BYO_ISAAC_RESUME_PHASE"] == "exploration"
    audit = json.loads(environment["NPA_SIM2REAL_RESUME_CURRICULUM_JSON"])
    assert audit["stable_grasp_and_lift_episodes"] == 0


@pytest.mark.parametrize(
    "outer,inner,previous_outer", [(1, 2, 1), (2, 1, 1), (2, 2, 2)]
)
def test_resume_reads_the_preceding_completed_training_evidence(
    monkeypatch, tmp_path, outer, inner, previous_outer
):
    from npa.workflows.sim2real import workflow_stage

    reads = []
    evidence = {"checkpoint_candidates": [_candidate(40)]}

    def read(uri, *, directory):
        reads.append(uri)
        return evidence

    monkeypatch.setattr(workflow_stage, "read_json", read)
    args = Namespace(
        root_uri="s3://example-bucket/sim2real/run",
        outer_iteration=outer,
        inner_iteration=inner,
    )
    result = workflow_stage._prior_training_evidence(args, tmp_path)
    assert result == evidence
    assert reads == [
        f"s3://example-bucket/sim2real/run/inner_loop/outer-{previous_outer:02d}/evidence.json"
    ]


def test_initial_pass_does_not_read_validation_or_gold(monkeypatch, tmp_path):
    from npa.workflows.sim2real import workflow_stage

    def forbidden_read(*args, **kwargs):
        pytest.fail("An initial pass must not read checkpoint or gold evidence")

    monkeypatch.setattr(workflow_stage, "read_json", forbidden_read)
    args = Namespace(
        root_uri="s3://example-bucket/sim2real/run",
        outer_iteration=1,
        inner_iteration=1,
    )
    assert workflow_stage._prior_training_evidence(args, tmp_path) == {}

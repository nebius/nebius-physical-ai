"""Contract tests for reusable BEHAVIOR campaign panels and comparisons."""

from __future__ import annotations

import copy

import pytest

from npa.workflows.behavior_challenge import campaign


def _registry() -> list[str]:
    return [f"task_{index:03d}" for index in range(100)]


def _policy(name: str, marker: str) -> dict:
    return campaign.freeze_policy_identity(
        name,
        {
            "checkpoint": {"sha256": marker * 64, "bytes": 100},
            "serving": {"sha256": marker * 64, "bytes": 200},
        },
    )


def _declaration(
    *, tasks: list[str] | None = None, workers: int = 3, candidate_marker: str = "b"
) -> dict:
    return campaign.declare_campaign(
        "focused-control",
        _registry(),
        tasks if tasks is not None else ["task_000", "task_001"],
        _policy("stock", "a"),
        _policy("candidate", candidate_marker),
        worker_count=workers,
        minimum_paired_cases=10,
    )


def _rollout(case: dict, score: float, *, success: bool | None = None) -> dict:
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    value = {key: case[key] for key in ("task", "index", "instance_id", "rollout_id")}
    value.update(
        {
            "q_score": score,
            "video_frames": 12,
            "files": {
                f"json/{stem}.json": "c" * 64,
                f"videos/{stem}.mp4": "d" * 64,
            },
        }
    )
    if success is not None:
        value.update(success=success, steps=120)
    return value


def _aggregate(panel: dict, offset: float, *, with_success: bool = True) -> dict:
    receipts = []
    for ordinal, case in enumerate(panel["cases"]):
        score = min(1.0, offset + ordinal / 100)
        success = score >= 0.5 if with_success else None
        rollout = _rollout(case, score, success=success)
        receipts.append(campaign.bind_inspected_rollout(panel, rollout))
    return campaign.aggregate_panel(panel, receipts)


def test_policy_identity_requires_immutable_checkpoint_and_serving_artifacts():
    identity = _policy("stock", "a")
    assert campaign.validate_policy_identity(identity) == identity
    assert identity["identity_sha256"] == campaign.canonical_digest(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    with pytest.raises(ValueError, match="checkpoint and serving"):
        campaign.freeze_policy_identity(
            "stock", {"checkpoint": {"sha256": "a" * 64, "bytes": 1}}
        )
    changed = copy.deepcopy(identity)
    changed["artifacts"]["checkpoint"]["bytes"] += 1
    with pytest.raises(ValueError, match="digest"):
        campaign.validate_policy_identity(changed)


def test_panel_identity_is_reusable_across_campaigns_and_worker_layouts():
    first = _declaration(workers=2)
    second = campaign.declare_campaign(
        "another-comparison",
        _registry(),
        ["task_001", "task_000"],
        _policy("stock", "a"),
        _policy("new-candidate", "e"),
        worker_count=5,
        minimum_paired_cases=20,
    )
    first_panel = first["panels"]["development"]["baseline"]
    second_panel = second["panels"]["development"]["baseline"]
    assert first_panel["panel_id"] == second_panel["panel_id"]
    assert first_panel == second_panel
    assert first["campaign_sha256"] != second["campaign_sha256"]
    assert (
        first["partitions"]["development"]["baseline"]
        != second["partitions"]["development"]["baseline"]
    )


def test_policy_alias_cannot_disguise_identical_baseline_candidate_bytes():
    artifacts = _policy("stock", "a")["artifacts"]
    with pytest.raises(ValueError, match="must differ"):
        campaign.declare_campaign(
            "aliased-policies",
            _registry(),
            ["task_000"],
            campaign.freeze_policy_identity("stock", artifacts),
            campaign.freeze_policy_identity("renamed-stock", artifacts),
            worker_count=1,
        )


def test_development_and_reporting_cases_are_fixed_disjoint_protocol_splits():
    declaration = _declaration(tasks=["task_002"])
    development = declaration["panels"]["development"]["baseline"]
    reporting = declaration["panels"]["reporting"]["baseline"]
    assert [case["index"] for case in development["cases"]] == list(range(10, 20))
    assert [case["instance_id"] for case in development["cases"]] == list(
        range(311, 321)
    )
    assert [case["index"] for case in reporting["cases"]] == list(range(10))
    assert [case["instance_id"] for case in reporting["cases"]] == list(range(301, 311))
    assert not {case["case_id"] for case in development["cases"]} & {
        case["case_id"] for case in reporting["cases"]
    }


def test_any_nonempty_official_task_subset_and_full_registry_are_supported():
    three = _declaration(tasks=_registry()[:3])
    assert three["task_count"] == 3
    assert three["panels"]["reporting"]["baseline"]["case_count"] == 30
    full = _declaration(tasks=_registry(), workers=32)
    assert full["task_count"] == 100
    assert full["panels"]["reporting"]["baseline"]["case_count"] == 1000
    assert full["full_challenge_competitive_claim_allowed"] is False
    with pytest.raises(ValueError, match="distinct official tasks"):
        _declaration(tasks=[])


def test_partition_is_deterministic_balanced_and_complete():
    panel = _declaration()["panels"]["development"]["baseline"]
    partition = campaign.partition_panel(panel, 3)
    assert campaign.validate_partition(partition, panel) == partition
    assigned = [case for worker in partition["workers"] for case in worker["case_ids"]]
    expected = [case["case_id"] for case in panel["cases"]]
    assert len(assigned) == len(set(assigned)) == len(expected)
    assert set(assigned) == set(expected)
    assert [len(worker["case_ids"]) for worker in partition["workers"]] == [7, 7, 6]
    assert partition == campaign.partition_panel(panel, 3)
    altered = copy.deepcopy(partition)
    altered["workers"][0]["case_ids"].reverse()
    with pytest.raises(ValueError, match="deterministic"):
        campaign.validate_partition(altered, panel)


def test_complete_aggregate_binds_inspected_artifacts_without_claiming_byte_access():
    panel = _declaration()["panels"]["development"]["baseline"]
    aggregate = _aggregate(panel, 0.4)
    assert campaign.validate_panel_aggregate(aggregate, panel) == aggregate
    assert aggregate["complete"] is True and aggregate["case_count"] == 20
    assert aggregate["success_count"] == 10
    assert aggregate["artifact_validation_contract"] == (
        "npa.behavior.inspect-rollout.v1"
    )
    assert aggregate["artifact_bytes_verified_by_aggregator"] is False
    assert aggregate["full_challenge_competitive_claim_allowed"] is False
    assert all(
        receipt["policy_identity_sha256"] == panel["policy"]["identity_sha256"]
        for receipt in aggregate["case_receipts"]
    )


def test_aggregate_rejects_missing_duplicate_and_changed_case_evidence():
    panel = _declaration()["panels"]["development"]["baseline"]
    receipts = [
        campaign.bind_inspected_rollout(panel, _rollout(case, 0.25))
        for case in panel["cases"]
    ]
    with pytest.raises(ValueError, match="every prescribed case"):
        campaign.aggregate_panel(panel, receipts[:-1])
    with pytest.raises(ValueError, match="every prescribed case"):
        campaign.aggregate_panel(panel, receipts[:-1] + [receipts[0]])
    altered = copy.deepcopy(receipts)
    altered[0]["files"][next(iter(altered[0]["files"]))] = "e" * 64
    with pytest.raises(ValueError, match="canonical panel binding"):
        campaign.aggregate_panel(panel, altered)


def test_success_and_steps_must_be_present_for_every_case_or_none():
    panel = _declaration()["panels"]["reporting"]["baseline"]
    receipts = [
        campaign.bind_inspected_rollout(panel, _rollout(case, 0.25))
        for case in panel["cases"]
    ]
    aggregate = campaign.aggregate_panel(panel, receipts)
    assert aggregate["success_count"] is None and aggregate["mean_steps"] is None
    mixed = receipts[:]
    mixed[0] = campaign.bind_inspected_rollout(
        panel, _rollout(panel["cases"][0], 0.25, success=False)
    )
    with pytest.raises(ValueError, match="every panel case or none"):
        campaign.aggregate_panel(panel, mixed)


def test_comparison_uses_exact_paired_cases_q_success_and_reviewed_threshold():
    declaration = _declaration()
    baseline_panel = declaration["panels"]["reporting"]["baseline"]
    candidate_panel = declaration["panels"]["reporting"]["candidate"]
    baseline = _aggregate(baseline_panel, 0.35)
    candidate = _aggregate(candidate_panel, 0.45)
    result = campaign.compare_panels(declaration, "reporting", baseline, candidate)
    assert result["paired_case_count"] == 20
    assert result["paired_case_threshold_met"] is True
    assert result["mean_q_gain"] == pytest.approx(0.1)
    assert result["success_rate_gain"] == pytest.approx(0.5)
    assert result["q_outcome"] == "candidate"
    assert result["claim_scope"] == "focused_task_panel"
    assert result["complete_panels_only"] is True
    assert result["no_case_selection_after_observation"] is True
    assert result["full_challenge_competitive_claim_allowed"] is False


def test_comparison_rejects_wrong_panel_partial_or_changed_campaign():
    declaration = _declaration()
    baseline_panel = declaration["panels"]["development"]["baseline"]
    candidate_panel = declaration["panels"]["development"]["candidate"]
    baseline = _aggregate(baseline_panel, 0.1)
    candidate = _aggregate(candidate_panel, 0.2)
    with pytest.raises(ValueError, match="referenced panel"):
        campaign.compare_panels(declaration, "reporting", baseline, candidate)
    partial = copy.deepcopy(candidate)
    partial["case_receipts"].pop()
    with pytest.raises(ValueError, match="every prescribed case"):
        campaign.compare_panels(declaration, "development", baseline, partial)
    changed = copy.deepcopy(declaration)
    changed["minimum_paired_cases"] = 1
    with pytest.raises(ValueError, match="Campaign digest"):
        campaign.compare_panels(changed, "development", baseline, candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("q_score", float("nan")),
        ("q_score", 1.1),
        ("video_frames", 0),
        ("success", True),
    ],
)
def test_case_receipt_rejects_invalid_measurements(field, value):
    panel = _declaration()["panels"]["development"]["baseline"]
    rollout = _rollout(panel["cases"][0], 0.2)
    rollout[field] = value
    with pytest.raises(ValueError):
        campaign.bind_inspected_rollout(panel, rollout)

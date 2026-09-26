"""Test generic non-reporting TRAIN declarations, evidence, and ranking."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import av
import numpy as np
import pytest

from npa.workflows.behavior_challenge import artifacts
from npa.workflows.behavior_challenge import nonreporting_train as train
from npa.workflows.behavior_challenge.campaign import (
    canonical_digest,
    freeze_policy_identity,
)


def _artifact(marker: str, size: int = 10) -> dict:
    return {"sha256": marker * 64, "bytes": size}


def _mapping(task: str, marker: str = "a") -> dict:
    return {
        "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
        "split": "train",
        "task": task,
        "data_namespace": "fixture_dataset",
        "data_task_id": f"row-{task}",
        "source_manifest": _artifact("9"),
        "split_manifest": _artifact("0"),
        "mapping_artifact": _artifact(marker),
    }


def _lineage() -> dict:
    return {
        "behavior_upstream_commit": "a" * 40,
        "task_registry": _artifact("b"),
        "dataset": _artifact("c"),
        "dataset_view": _artifact("d"),
        "normalization": _artifact("e"),
        "tokenizer": _artifact("f"),
        "action_semantics": _artifact("1"),
    }


def _evaluator() -> dict:
    return {
        "argv_contract": _artifact("9"),
        "evaluator_source": _artifact("2"),
        "controller_source": _artifact("3"),
        "robot_config": _artifact("4"),
        "rng_contract": _artifact("5"),
        "wrapper": "fixture.Wrapper",
        "mode": "train",
        "num_envs": 1,
        "num_rollouts": 1,
        "write_video": True,
        "max_steps_argument": None,
        "model_prediction_horizon": 32,
        "executed_prefix": 32,
        "fresh_policy_process_per_case": True,
        "qualification_process_discarded": True,
    }


def _protocol(task: str = "fixture_task", cases: list[dict] | None = None) -> dict:
    return train.declare_train_protocol(
        task,
        _mapping(task),
        cases
        if cases is not None
        else [
            {"instance_id": 0, "rollout_id": 0},
            {"instance_id": 7, "rollout_id": 0},
        ],
        _lineage(),
        _evaluator(),
    )


def _policy(marker: str) -> dict:
    return freeze_policy_identity(
        f"fixture-{marker}",
        {
            "checkpoint": _artifact(marker, 100),
            "serving": _artifact(marker, 200),
        },
    )


def _panel(marker: str = "6", **kwargs) -> dict:
    return train.declare_train_panel(_protocol(**kwargs), _policy(marker))


def _inspected(case: dict, score: float, success: bool) -> dict:
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    return {
        **case,
        "q_score": score,
        "success": success,
        "steps": 120,
        "video_frames": 120,
        "files": {
            f"json/{stem}.json": "7" * 64,
            f"videos/{stem}.mp4": "8" * 64,
        },
    }


def _aggregate(panel: dict, scores: tuple[float, ...], root) -> dict:
    outputs = {}
    for case, score in zip(panel["cases"], scores, strict=True):
        output = root / case["case_id"]
        output.mkdir(parents=True)
        _write_real_rollout(output, case, score, score >= 0.5)
        outputs[case["case_id"]] = output
    return train.aggregate_train_outputs(panel, outputs)


def _study_row(key: str, aggregate: dict, primary: float, secondary: float) -> dict:
    return {
        "policy_key": key,
        "policy_binding_sha256": aggregate["policy_binding_sha256"],
        "panel_id": aggregate["panel_id"],
        "protocol_sha256": aggregate["protocol_sha256"],
        "case_count": aggregate["case_count"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "primary_loss": primary,
        "secondary_loss": secondary,
    }


def test_protocol_is_generic_canonical_and_train_only():
    first = _protocol()
    second = _protocol(
        task="another_task",
        cases=[
            {"instance_id": 42, "rollout_id": 0},
            {"instance_id": 99, "rollout_id": 0},
        ],
    )

    assert train.validate_train_protocol(first) == first
    assert train.validate_train_protocol(second) == second
    assert second["task_mapping"]["task"] == "another_task"
    assert second["case_count"] == 2
    assert second["development_or_report_allowed"] is False
    assert second["reporting_claim_allowed"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(split="development"), "canonical"),
        (lambda value: value.update(mode="report"), "canonical"),
        (lambda value: value.update(reporting_claim_allowed=True), "canonical"),
        (lambda value: value["task_mapping"].update(task="other"), "mapping"),
        (
            lambda value: value["task_mapping"].update(split="development"),
            "TRAIN source split",
        ),
        (
            lambda value: value["evaluator_contract"].update(mode="public_test"),
            "TRAIN cases",
        ),
        (lambda value: value["science_lineage"].pop("dataset"), "exactly"),
        (
            lambda value: value["evaluator_contract"].update(max_steps_argument=10),
            "max step",
        ),
        (
            lambda value: value["evaluator_contract"].update(
                fresh_policy_process_per_case=False
            ),
            "fresh policy process",
        ),
        (
            lambda value: value["evaluator_contract"].update(
                qualification_process_discarded=False
            ),
            "discarded process",
        ),
    ],
)
def test_protocol_rejects_cross_split_missing_lineage_and_rehashed_mutations(
    mutation, match
):
    changed = copy.deepcopy(_protocol())
    mutation(changed)
    changed["protocol_sha256"] = canonical_digest(
        {key: value for key, value in changed.items() if key != "protocol_sha256"}
    )
    with pytest.raises(ValueError, match=match):
        train.validate_train_protocol(changed)


def test_protocol_rejects_duplicate_cases_and_wrong_identity():
    cases = [{"instance_id": 1, "rollout_id": 0}] * 2
    with pytest.raises(ValueError, match="distinct"):
        _protocol(cases=cases)
    changed = copy.deepcopy(_protocol())
    changed["protocol_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="canonical"):
        train.validate_train_protocol(changed)
    with pytest.raises(ValueError, match="CaseStore-compatible"):
        _protocol(task="../escaped")
    with pytest.raises(ValueError, match="CaseStore-compatible"):
        _protocol(task="not-case-store-compatible")
    with pytest.raises(ValueError, match="requires rollout_id 0"):
        _protocol(cases=[{"instance_id": 0, "rollout_id": 1}])
    evaluator = _evaluator()
    evaluator["wrapper"] = "../not-an-import"
    with pytest.raises(ValueError, match="dotted import path"):
        train.declare_train_protocol(
            "fixture_task",
            _mapping("fixture_task"),
            [{"instance_id": 0, "rollout_id": 0}],
            _lineage(),
            evaluator,
        )


def test_train_evaluator_argv_is_one_exact_train_case_without_step_override():
    lineage = _lineage()
    lineage["behavior_upstream_commit"] = "6cbf70b075816096e9be53958780769f3264d25d"
    protocol = train.declare_train_protocol(
        "fixture_task",
        _mapping("fixture_task"),
        [{"instance_id": 7, "rollout_id": 0}],
        lineage,
        _evaluator(),
    )
    panel = train.declare_train_panel(protocol, _policy("6"))

    argv = train.train_evaluator_argv(
        protocol,
        panel["cases"][0],
        root=Path("/verified/upstream"),
        python="/locked/python",
        host="127.0.0.1",
        port=8000,
        output=Path("/case/output"),
    )

    assert argv[:3] == ["/locked/python", "-m", "omnigibson.eval.eval"]
    assert argv[argv.index("--mode") + 1] == "train"
    assert argv[argv.index("--instance-indices") + 1] == "7"
    assert argv[argv.index("--num-rollouts") + 1] == "1"
    assert argv[argv.index("--num-envs") + 1] == "1"
    assert argv[-2:] == ["--write-video", "--headless"]
    assert "--max-steps" not in argv
    with pytest.raises(ValueError, match="not prescribed"):
        train.train_evaluator_argv(
            protocol,
            {**panel["cases"][0], "instance_id": 8},
            root=Path("/verified/upstream"),
            python="/locked/python",
            host="127.0.0.1",
            port=8000,
            output=Path("/case/output"),
        )


def test_panel_and_partition_bind_complete_policy_and_case_order():
    panel = _panel()
    partition = train.partition_train_panel(panel, 2)

    assert train.validate_train_panel(panel) == panel
    assert train.validate_train_partition(partition, panel) == partition
    assert [worker["case_ids"] for worker in partition["workers"]] == [
        [panel["cases"][0]["case_id"]],
        [panel["cases"][1]["case_id"]],
    ]
    changed = copy.deepcopy(panel)
    changed["policy_binding"]["artifacts"]["checkpoint"]["bytes"] += 1
    changed["panel_id"] = canonical_digest(
        {key: value for key, value in changed.items() if key != "panel_id"}
    )
    with pytest.raises(ValueError, match="Policy identity"):
        train.validate_train_panel(changed)
    with pytest.raises(ValueError, match="policy identity"):
        train.declare_train_panel(_protocol(), {"schema": "incomplete"})
    arbitrary = {
        "schema": "fixture.self-hashed.v1",
        "name": "no-checkpoint-or-serving-artifacts",
    }
    arbitrary["policy_identity_sha256"] = canonical_digest(arbitrary)
    with pytest.raises(ValueError, match="policy identity"):
        train.declare_train_panel(_protocol(), arbitrary)


def test_aggregate_requires_every_inspection_shaped_case_exactly_once():
    panel = _panel(cases=[{"instance_id": 0, "rollout_id": 0}] * 1)
    record = _inspected(panel["cases"][0], 0.75, True)
    aggregate = train.aggregate_train_panel(panel, [record])

    assert train.validate_train_panel_aggregate(aggregate, panel) == aggregate
    assert aggregate["mean_q"] == 0.75
    assert aggregate["success_count"] == 1
    assert aggregate["artifact_validation_contract"] == (
        "npa.behavior.caller-supplied-rollout-record.v1"
    )
    assert aggregate["artifact_bytes_verified_by_aggregator"] is False
    with pytest.raises(ValueError, match="every prescribed case"):
        train.aggregate_train_panel(panel, [])
    with pytest.raises(ValueError, match="every prescribed case"):
        train.aggregate_train_panel(panel, [record, record])
    changed = copy.deepcopy(record)
    changed["split"] = "report"
    with pytest.raises(ValueError, match="not a prescribed TRAIN case"):
        train.aggregate_train_panel(panel, [changed])


def _write_real_rollout(output, case, score=0.625, success=True):
    stem = f"{case['task']}_{case['instance_id']}_0"
    metrics = {
        "task": case["task"],
        "instance_id": case["instance_id"],
        "rollout_id": case["rollout_id"],
        "steps": 2,
        "success": success,
        "q_score": {"final": score},
        "agent_distance": {name: 1.0 for name in ("base", "left", "right")},
        "normalized_agent_distance": {name: 1.0 for name in ("base", "left", "right")},
        "time": {
            "simulator_steps": 2,
            "simulator_time": 2 / 30,
            "normalized_time": 10.0,
        },
    }
    (output / "json").mkdir()
    (output / "videos").mkdir()
    (output / f"json/{stem}.json").write_text(json.dumps(metrics) + "\n")
    with av.open(str(output / f"videos/{stem}.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 128):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_real_decoded_rollout_composes_into_train_aggregate(tmp_path):
    panel = _panel(cases=[{"instance_id": 5, "rollout_id": 0}])
    case = panel["cases"][0]
    _write_real_rollout(tmp_path, case)

    inspected = artifacts.inspect_rollout(tmp_path, case)
    unverified = train.aggregate_train_panel(panel, [inspected])
    aggregate = train.aggregate_train_outputs(panel, {case["case_id"]: tmp_path})

    assert aggregate["mean_q"] == 0.625
    assert aggregate["case_receipts"][0]["video_frames"] == 2
    assert aggregate["artifact_validation_contract"] == (
        "npa.behavior.inspect-rollout.v1"
    )
    assert aggregate["artifact_bytes_verified_by_aggregator"] is True
    assert unverified["artifact_bytes_verified_by_aggregator"] is False
    assert train.validate_train_panel_aggregate(aggregate, panel) == aggregate

    study_panel = _panel("5", cases=[{"instance_id": 5, "rollout_id": 0}])
    unverified_row = _study_row("unverified", unverified, 0.1, 0.2)
    verified_output = tmp_path / "verified-study"
    verified_output.mkdir()
    _write_real_rollout(verified_output, study_panel["cases"][0])
    verified = train.aggregate_train_outputs(
        study_panel, {study_panel["cases"][0]["case_id"]: verified_output}
    )
    verified_row = _study_row("verified", verified, 0.2, 0.3)
    study = train.declare_train_study([unverified_row, verified_row])
    with pytest.raises(ValueError, match="evidence boundary"):
        train.rank_train_study(study, [unverified, verified])


def test_file_backed_aggregate_requires_exact_output_mapping(tmp_path):
    panel = _panel(cases=[{"instance_id": 5, "rollout_id": 0}])
    case = panel["cases"][0]
    _write_real_rollout(tmp_path, case)

    with pytest.raises(ValueError, match="every prescribed case"):
        train.aggregate_train_outputs(panel, {})
    with pytest.raises(ValueError, match="case IDs and Path"):
        train.aggregate_train_outputs(panel, {case["case_id"]: str(tmp_path)})

    metrics = next((tmp_path / "json").iterdir())
    original = tmp_path / "original.json"
    metrics.replace(original)
    metrics.symlink_to(original)
    with pytest.raises(ValueError, match="non-symlink"):
        train.aggregate_train_outputs(panel, {case["case_id"]: tmp_path})


def test_file_backed_aggregate_rejects_step_frame_mismatch(tmp_path):
    panel = _panel(cases=[{"instance_id": 5, "rollout_id": 0}])
    case = panel["cases"][0]
    _write_real_rollout(tmp_path, case)
    metrics = next((tmp_path / "json").iterdir())
    payload = json.loads(metrics.read_text())
    payload["steps"] = 3
    metrics.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="frame count must equal evaluator steps"):
        train.aggregate_train_outputs(panel, {case["case_id"]: tmp_path})


def _rank_fixture(specs, root):
    aggregates = []
    rows = []
    for index, (scores, primary, secondary) in enumerate(specs):
        panel = _panel(str(index + 1), cases=[{"instance_id": 0, "rollout_id": 0}])
        aggregate = _aggregate(panel, scores, root / str(index))
        aggregates.append(aggregate)
        rows.append(_study_row(f"policy-{index}", aggregate, primary, secondary))
    return train.declare_train_study(rows), aggregates


def test_rank_agreement_emits_ties_pair_counts_and_pareto_relations(tmp_path):
    study, aggregates = _rank_fixture(
        [
            ((0.9,), 0.1, 0.2),
            ((0.9,), 0.1, 0.2),
            ((0.4,), 0.2, 0.1),
            ((0.2,), 0.3, 0.1),
        ],
        tmp_path,
    )
    result = train.rank_train_study(study, aggregates)

    assert result["policy_count"] == 4
    assert result["calibration_midrank_vector"] == [1.5, 1.5, 3.0, 4.0]
    assert result["closed_loop_midrank_vector"] == [1.5, 1.5, 3.0, 4.0]
    assert result["kendall_tau_b"]["pair_count"] == 6
    assert (
        sum(
            result["kendall_tau_b"][name]
            for name in (
                "P_concordant",
                "Q_discordant",
                "T_c_calibration_only_ties",
                "T_r_rollout_only_ties",
                "T_joint_ties",
            )
        )
        == 6
    )
    assert result["kendall_tau_b"]["value"] == pytest.approx(1.0)
    assert result["spearman_midrank"]["value"] == pytest.approx(1.0)
    assert train.pareto_dominates(result["rows"][0], result["rows"][2])
    assert result["rows"][0]["dominates"] == ["policy-2", "policy-3"]


def test_rank_agreement_reports_zero_variance_without_selection_claim(tmp_path):
    study, aggregates = _rank_fixture(
        [
            ((0.9,), 0.1, 0.2),
            ((0.6,), 0.1, 0.2),
            ((0.2,), 0.1, 0.2),
        ],
        tmp_path,
    )
    result = train.rank_train_study(study, aggregates)

    assert result["kendall_tau_b"]["value"] is None
    assert result["kendall_tau_b"]["undefined_reason"] == "zero_denominator"
    assert result["spearman_midrank"]["value"] is None
    assert result["spearman_midrank"]["undefined_reason"] == "zero_variance"
    assert result["claim"] == "descriptive_only_no_selection_or_significance"
    assert not any("selected" in row for row in result["rows"])


def test_rank_rejects_incomplete_wrong_and_consistently_rehashed_aggregate(tmp_path):
    study, aggregates = _rank_fixture(
        [((0.9,), 0.1, 0.2), ((0.2,), 0.2, 0.3)], tmp_path
    )
    with pytest.raises(ValueError, match="every declared aggregate"):
        train.rank_train_study(study, aggregates[:1])
    changed = copy.deepcopy(aggregates)
    changed[0]["case_receipts"][0]["q_score"] = 0.1
    changed[0]["case_receipts"][0]["case_receipt_sha256"] = canonical_digest(
        {
            key: value
            for key, value in changed[0]["case_receipts"][0].items()
            if key != "case_receipt_sha256"
        }
    )
    changed[0]["aggregate_sha256"] = canonical_digest(
        {key: value for key, value in changed[0].items() if key != "aggregate_sha256"}
    )
    mutated_rows = copy.deepcopy(study["rows"])
    mutated_rows[0]["aggregate_sha256"] = changed[0]["aggregate_sha256"]
    mutated_study = train.declare_train_study(mutated_rows)
    with pytest.raises(ValueError, match="summaries differ"):
        train.rank_train_study(mutated_study, changed)


def test_rank_rejects_cross_protocol_and_study_row_identity_substitution(tmp_path):
    study, aggregates = _rank_fixture(
        [((0.9,), 0.1, 0.2), ((0.2,), 0.2, 0.3)], tmp_path
    )
    changed_rows = copy.deepcopy(study["rows"])
    changed_rows[0]["panel_id"] = "f" * 64
    changed_study = train.declare_train_study(changed_rows)
    with pytest.raises(ValueError, match="study row bindings"):
        train.rank_train_study(changed_study, aggregates)

    other_panel = _panel(
        "f",
        task="other_task",
        cases=[{"instance_id": 0, "rollout_id": 0}],
    )
    other = _aggregate(other_panel, (0.4,), tmp_path / "other")
    mixed_rows = [study["rows"][0], _study_row("other", other, 0.2, 0.3)]
    with pytest.raises(ValueError, match="share one protocol"):
        train.declare_train_study(mixed_rows)


def test_rank_rejects_consistently_rehashed_forged_case_identity(tmp_path):
    study, aggregates = _rank_fixture(
        [((0.9,), 0.1, 0.2), ((0.2,), 0.2, 0.3)], tmp_path
    )
    changed = copy.deepcopy(aggregates)
    receipt = changed[0]["case_receipts"][0]
    receipt["case"]["case_id"] = "e" * 64
    receipt["case_receipt_sha256"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "case_receipt_sha256"}
    )
    changed[0]["aggregate_sha256"] = canonical_digest(
        {key: value for key, value in changed[0].items() if key != "aggregate_sha256"}
    )
    rows = copy.deepcopy(study["rows"])
    rows[0]["aggregate_sha256"] = changed[0]["aggregate_sha256"]
    changed_study = train.declare_train_study(rows)
    with pytest.raises(ValueError, match="case identity differs"):
        train.rank_train_study(changed_study, changed)

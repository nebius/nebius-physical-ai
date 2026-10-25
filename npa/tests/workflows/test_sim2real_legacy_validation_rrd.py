"""Retained candidate populations reach the real CPU RRD writer and reader."""

import copy
import json

import pytest

from npa.viz.recordings import load_recording
from npa.workflows import sim2real_viz as viz
from npa.workflows.sim2real.checkpoint_selection import (
    resolve_selected_checkpoint,
    select_best_checkpoint,
    validation_checkpoint_candidate,
)


def _candidate(outer, training, rate):
    uri = f"s3://unit/run/outer-{outer}/model_{training}.pt"
    report = {
        "report_uri": f"s3://unit/run/outer-{outer}/validation-{training}.json",
        "policy_checkpoint": uri,
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 128,
        "policy_inference_provenance": {"generator_policy_sha256": "a" * 64},
        "success_rate": rate,
        "per_env": [{"env_id": "validation-1", "success": True}],
        "evaluation_split": "validation",
    }
    return validation_checkpoint_candidate(
        report,
        checkpoint_uri=uri,
        outer_iteration=outer,
        inner_iteration=1,
        training_iteration=training,
    )


def _evidence(prior_outer):
    outer = 2 if prior_outer else 1
    candidates = [_candidate(1, 100, 0.75), _candidate(outer, 200, 0.25)]
    selection = select_best_checkpoint(candidates)
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "trainer_source": "byo_command",
        "outer_iteration": outer,
        "checkpoint_candidates": candidates,
        "checkpoint_selection": selection,
        "selected_checkpoint_uri": selection["checkpoint_uri"],
        "final_checkpoint_uri": selection["checkpoint_uri"],
        "iterations": [
            {
                "iteration": 1,
                "update": {"checkpoint_path": "s3://unit/run/model_latest.pt"},
                "validation_report": selection["validation_report"],
                "periodic_validation_reports": [candidates[-1]["validation_report"]],
            }
        ],
    }
    resolve_selected_checkpoint(evidence)
    return evidence


@pytest.mark.parametrize("prior_outer", [False, True])
def test_legacy_periodic_selection_roundtrips_real_rrd(tmp_path, prior_outer):
    evidence = _evidence(prior_outer)
    before = copy.deepcopy(evidence)
    output = tmp_path / "reports/sim2real.rrd"
    result = viz.emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=evidence,
        heldout_report=None,
        output_rrd=output,
        run_metadata={"run_id": "legacy-fixture"},
        stage_components=[{"name": "stage_09_train", "tier": "UNKNOWN"}],
        allow_progress_only=True,
    )
    assert result.status == "written"
    assert output.stat().st_size > 0
    metrics, labels = [], []
    for chunk in load_recording(output).chunks():
        batch = chunk.to_record_batch()
        if chunk.entity_path == "/evaluation/validation_strict_success":
            metrics.extend(
                row[0] for row in batch.column("Scalars:scalars").to_pylist()
            )
        if chunk.entity_path.startswith("/training/checkpoints/"):
            labels.extend(
                row[0] for row in batch.column("TextDocument:text").to_pylist()
            )
    assert metrics == pytest.approx([0.75])
    assert len(labels) == 1
    assert evidence["selected_checkpoint_uri"] in labels[0]
    assert "model_latest.pt" not in labels[0]
    assert evidence == before


@pytest.mark.parametrize(
    "corruption", ["duplicate", "wrong_digest", "future", "missing"]
)
def test_legacy_validation_projection_rejects_unproven_claims(tmp_path, corruption):
    evidence = _evidence(True)
    if corruption == "duplicate":
        evidence["checkpoint_candidates"].append(
            copy.deepcopy(evidence["checkpoint_candidates"][0])
        )
    elif corruption == "wrong_digest":
        evidence["iterations"][0]["validation_report"] = dict(
            evidence["iterations"][0]["validation_report"],
            policy_checkpoint_sha256="b" * 64,
        )
    elif corruption == "future":
        evidence["checkpoint_candidates"][0]["outer_iteration"] = 3
    else:
        evidence["iterations"][0].pop("validation_report")
    with pytest.raises(viz.Sim2RealVizError, match="validation|Validation"):
        viz._all_inner_iteration_records(tmp_path, evidence)


def test_canonical_candidate_contract_is_not_relaxed(tmp_path):
    evidence = _evidence(False)
    evidence["iterations"][0]["vlm_eval_uri"] = "s3://unit/run/evaluation.json"
    with pytest.raises(viz.Sim2RealVizError, match="ambiguous"):
        viz._all_inner_iteration_records(tmp_path, evidence)


def test_legacy_projection_keeps_persisted_candidate_population(tmp_path):
    evidence = _evidence(True)
    path = tmp_path / "inner_loop/outer-02/evidence.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(evidence, sort_keys=True)
    path.write_text(original)
    records = viz._all_inner_iteration_records(tmp_path, evidence)
    assert len(records) == 1
    assert (
        records[0][1]["validation_report"]["policy_checkpoint"]
        == evidence["selected_checkpoint_uri"]
    )
    assert path.read_text() == original

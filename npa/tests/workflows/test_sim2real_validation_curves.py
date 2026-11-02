"""Regress the complete native checkpoint pools that previously broke Stage 14."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from npa.workflows import sim2real_viz as viz
from npa.workflows.sim2real.validation_curves import native_pass_validation_candidates


def _native_evidence():
    payload = {"outer_iteration": 1, "iterations": [], "checkpoint_candidates": []}
    for inner in (1, 2, 3):
        indices = (
            [0, *range(100, 2000, 100), 1999]
            if inner == 1
            else list(range(200, 2101, 100))
        )
        pool = []
        for index in indices:
            uri = f"s3://unit/run/pass-{inner}/checkpoints/model_{index}.pt"
            checkpoint = {
                "checkpoint_uri": uri,
                "training_iteration": index,
                "checkpoint_completed_updates": index + 1,
                "checkpoint_iteration_semantics": "last_completed_zero_based",
            }
            pool.append(checkpoint)
            digest = f"{inner * 10000 + index:064x}"
            report = {
                "evaluation_split": "validation",
                "policy_checkpoint": uri,
                "policy_checkpoint_sha256": digest,
                "success_rate": 0.78125 if inner == 1 and index == 100 else 0.0,
            }
            payload["checkpoint_candidates"].append(
                {
                    **checkpoint,
                    "outer_iteration": 1,
                    "inner_iteration": inner,
                    "evaluation_split": "validation",
                    "checkpoint_sha256": digest,
                    "validation_report": report,
                }
            )
        final = payload["checkpoint_candidates"][-1]
        payload["iterations"].append(
            {
                "iteration": inner,
                "update": {
                    "checkpoint_path": f"s3://unit/run/pass-{inner}/model_latest.pt",
                    "checkpoint_sha256": final["checkpoint_sha256"],
                    "final_checkpoint_uri": final["checkpoint_uri"],
                    "final_checkpoint_sha256": final["checkpoint_sha256"],
                    "final_checkpoint_alias_verified": True,
                    "ppo": {"iterations": 2000},
                    "ppo_telemetry": {"observed_iterations": 2000},
                    "periodic_checkpoints": pool,
                },
            }
        )
    payload["selected_validation_report"] = payload["checkpoint_candidates"][1][
        "validation_report"
    ]
    return payload


def test_native_sweep_keeps_all_61_candidates_and_each_pass_final_score(tmp_path: Path):
    evidence = _native_evidence()
    original = json.dumps(evidence, sort_keys=True)
    path = tmp_path / "inner_loop/outer-01/evidence.json"
    path.parent.mkdir(parents=True)
    path.write_text(original)
    records = viz._all_inner_iteration_records(tmp_path, evidence)
    assert len(evidence["checkpoint_candidates"]) == 61
    assert [record["validation_report"]["success_rate"] for _, record in records] == [
        0.0
    ] * 3
    assert [
        record["validation_report"]["policy_checkpoint"] for _, record in records
    ] == [record["update"]["final_checkpoint_uri"] for record in evidence["iterations"]]
    assert evidence["selected_validation_report"]["success_rate"] == 0.78125
    assert json.dumps(evidence, sort_keys=True) == path.read_text() == original


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing",
        "wrong_outer",
        "alias_digest",
        "unverified_alias",
        "final_report_digest",
    ],
)
def test_native_sweep_refuses_incomplete_or_conflicting_identity(
    tmp_path: Path, corruption
):
    evidence = _native_evidence()
    if corruption == "duplicate":
        evidence["checkpoint_candidates"].append(
            copy.deepcopy(evidence["checkpoint_candidates"][0])
        )
    elif corruption == "missing":
        evidence["checkpoint_candidates"].pop(0)
    elif corruption == "wrong_outer":
        evidence["checkpoint_candidates"][0]["outer_iteration"] = 2
    elif corruption == "alias_digest":
        evidence["iterations"][0]["update"]["final_checkpoint_sha256"] = "f" * 64
    elif corruption == "unverified_alias":
        evidence["iterations"][0]["update"]["final_checkpoint_alias_verified"] = False
    else:
        evidence["checkpoint_candidates"][-1]["validation_report"][
            "policy_checkpoint_sha256"
        ] = "f" * 64
    with pytest.raises(viz.Sim2RealVizError, match="Validation"):
        viz._all_inner_iteration_records(tmp_path, evidence)


def test_native_sweep_maps_overlapping_resumed_indices_to_distinct_inner_passes():
    selected = native_pass_validation_candidates(_native_evidence(), 1)
    assert [row["training_iteration"] for row in selected.values()] == [
        1999,
        2100,
        2100,
    ]
    assert len({row["checkpoint_uri"] for row in selected.values()}) == 3

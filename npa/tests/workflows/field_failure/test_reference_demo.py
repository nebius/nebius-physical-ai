"""Test replay geometry, cohort isolation and fail-closed development selection."""

import copy
import json
from types import SimpleNamespace

import pytest

from npa.workflows.field_failure.reference_demo_inputs import (
    capture_recipe,
    cohort_manifest,
    prepare_warehouse,
)
from npa.workflows.field_failure.reference_demo_evaluate import development_decision
from npa.workflows.field_failure.reference_demo_cohorts import freeze_balanced_cohorts
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.reference_replay import (
    apply_replay,
    compose_scene,
    translated_cases,
)
from npa.workflows.navigation.reference_scene import cases, warehouse

IMAGE = "registry.example/native@sha256:" + "a" * 64


def test_frozen_cohorts_have_distinct_physical_resets_not_just_new_ids(tmp_path):
    output = tmp_path / "baseline"
    manifest = prepare_warehouse(output, image=IMAGE, count=20)
    recipe = json.loads((output / "recipe.json").read_text())
    final = json.loads((output / "final-cases.json").read_text())
    assert manifest["cohorts"]["final"]["count"] == 20
    assert not manifest["final_used_for_selection"]
    duplicate = copy.deepcopy(recipe["train_cases"][0])
    duplicate.update(id="different-name", seed=99999999)
    with pytest.raises(ValueError, match="overlap"):
        cohort_manifest({"train": recipe["train_cases"], "final": [duplicate]})
    assert not {r["seed"] for r in final} & {r["seed"] for r in recipe["eval_cases"]}


def test_replay_preserves_exact_collision_geometry_and_case_relative_routes(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    original = tmp_path / "original.usdz"
    warehouse(original)
    combined = tmp_path / "combined.usdz"
    evidence = compose_scene(original, original, combined)
    assert evidence["office_sha256"] == file_sha256(original)
    assert not evidence["geometry_rescaled"] and not evidence["added_support_geometry"]
    source = Usd.Stage.Open(str(original))
    output = Usd.Stage.Open(str(combined))
    originals = [
        UsdGeom.Mesh(p) for p in source.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    merged = [
        UsdGeom.Mesh(p) for p in output.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(merged) == 2 * len(originals)
    for mesh in merged:
        assert list(mesh.GetPointsAttr().Get()) == list(
            originals[0].GetPointsAttr().Get()
        )
        assert list(mesh.GetFaceVertexIndicesAttr().Get()) == list(
            originals[0].GetFaceVertexIndicesAttr().Get()
        )
    original_case = cases(2)["train_cases"][0]
    moved = translated_cases([original_case], (50.0, 0.0, 0.0))[0]
    assert moved["position_m"][0] == original_case["position_m"][0] + 50
    assert moved["goal_m"][0] == original_case["goal_m"][0] + 50
    assert moved["heading_rad"] == original_case["heading_rad"]


def test_scene_composition_rejects_regions_with_overlapping_sensor_horizons(tmp_path):
    scene = tmp_path / "scene.usdz"
    warehouse(scene)
    with pytest.raises(ValueError, match="ray horizon"):
        compose_scene(scene, scene, tmp_path / "overlap.usdz", translation=(25, 0, 0))


def test_reconstruction_replay_retains_every_baseline_and_scan_training_case(tmp_path):
    baseline = tmp_path / "baseline"
    prepare_warehouse(baseline, image=IMAGE, count=4)
    capture = tmp_path / "capture"
    capture.mkdir()
    scan = cases(4)
    frozen = freeze_balanced_cohorts(baseline, scan, baseline / "scene.usdz")
    capture_recipe(capture, scan, baseline)
    recipe = json.loads((capture / "recipe.json").read_text())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    warehouse(prepared / "scene.usdz")
    evidence = apply_replay(capture, prepared, recipe)
    baseline_recipe = json.loads((baseline / "recipe.json").read_text())
    assert evidence["warehouse_routes"] == evidence["office_routes"] == 4
    assert recipe["train_cases"][1::2] == baseline_recipe["train_cases"]
    assert recipe["train_cases"][::2] == translated_cases(
        scan["train_cases"], [50, 0, 0]
    )
    assert len(recipe["eval_cases"]) == 4
    assert recipe["eval_cases"][0]["id"] == scan["eval_cases"][0]["id"]
    assert evidence["frozen_geometry_matched"]
    assert evidence["collision_identity"] == frozen["geometry"]["collision_identity"]
    assert recipe["scene_sha256"] == file_sha256(prepared / "scene.usdz")


def _report(checkpoint, rate, contacts=0):
    successes = round(rate * 200)
    return {
        "checkpoint_sha256": checkpoint * 64,
        "evaluation_inputs_sha256": "c" * 64,
        "success_rate": successes / 200,
        "episodes": [
            {
                "seed": i + 1,
                "case_id": "case-" + str(i + 1),
                "success": i < successes,
                "collision_steps": contacts if i == 199 else 0,
                "physical_failure_steps": 0,
                "goal_distance_m": 0.0 if i < successes else 2.0,
            }
            for i in range(200)
        ],
    }


def _regions():
    return {
        name: {
            "count": 100,
            "seeds": list(range(start, 201, 2)),
            "case_ids": ["case-" + str(i) for i in range(start, 201, 2)],
        }
        for name, start in (("office", 1), ("warehouse", 2))
    }


@pytest.mark.parametrize("rate,contacts", [(0.79, 0), (0.705, 0), (0.9, 1)])
def test_development_selection_never_treats_runtime_completion_as_quality(
    rate, contacts
):
    result = development_decision(
        _report("a", 0.7), _report("b", rate, contacts), _regions()
    )
    assert result["runtime_completed"] is True
    assert result["eligible"] is False
    assert result["final_cohort_consumed"] is False
    assert result["selected_checkpoint_sha256"] is None


def test_development_pass_requires_matching_cohort_and_changed_checkpoint():
    before, after = _report("a", 0.7), _report("b", 0.85)
    assert development_decision(before, after, _regions())["eligible"]
    after["evaluation_inputs_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different evaluation"):
        development_decision(before, after, _regions())


def test_diagnostic_replay_runs_all_batches_without_losing_cases(tmp_path, monkeypatch):
    from npa.workflows.field_failure import native_policy

    recipe = {"num_envs": 2, "train_cases": [{"id": str(i)} for i in range(4)]}
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    visited = []

    def replay(*args):
        selected = args[-1]
        visited.extend(selected)
        return {
            "episodes": selected,
            "success_rate": 0.5,
            "checkpoint_sha256": "a" * 64,
        }

    monkeypatch.setattr(native_policy, "_replay_batch", replay)
    result = native_policy._diagnostic_replay(
        {}, tmp_path, {}, None, tmp_path, "before"
    )
    assert visited == result["episodes"] == recipe["train_cases"]
    assert len(result["batches"]) == 2
    assert not result["used_for_promotion"]


def test_failed_development_publishes_html_before_refusing_final(tmp_path, monkeypatch):
    from npa.workflows.field_failure import reference_demo_evaluate as evaluate
    from npa.workflows.field_failure import reference_demo_report as report

    published = []
    monkeypatch.setattr(
        evaluate,
        "_read",
        lambda uri: (
            {"cohorts": {"regions": {"development": _regions()}}}
            if uri.endswith("reference-plan.json")
            else _report("a" if "baseline" in uri else "b", 0.1),
            "d",
        ),
    )
    monkeypatch.setattr(
        evaluate, "_publish", lambda uri, value: published.append(value)
    )
    monkeypatch.setattr(
        report, "publish_report", lambda args, selection: published.append("html")
    )
    with pytest.raises(RuntimeError, match="final cohort remains untouched"):
        evaluate.select_candidate(SimpleNamespace(output_root="s3://example/demo"))
    assert published[-1] == "html"
    assert published[0]["eligible"] is False


def test_offline_report_keeps_quality_and_promotion_separate():
    from npa.workflows.field_failure.reference_demo_report import (
        render_html,
        result_summary,
    )

    plan = {
        "cohorts": {"sha256": "a" * 64},
        "num_envs": 4000,
        "baseline_iterations": 1500,
        "candidate_iterations": 1500,
    }
    selection = development_decision(_report("a", 0.7), _report("b", 0.85), _regions())
    final = {
        "promote_checkpoint": True,
        "recommendation": "promote",
        "episodes_per_policy": 4000,
        "regressions": [],
        "metrics": {
            "success": {"baseline_mean": 0.6, "candidate_mean": 0.7, "improvement": 0.1}
        },
    }
    report = result_summary(plan, selection, final, selection["regional"])
    assert not report["quality_passed"]
    assert report["recommendation"] == "retain_baseline"
    from PIL import Image
    from npa.workflows.preview_html import image_preview

    groups = [
        {
            "title": "Synthetic unit fixture",
            "frames": [
                {
                    "label": "Unit frame",
                    "images": [
                        {
                            "label": "Fixture",
                            "data": image_preview(Image.new("RGB", (2, 2))),
                        }
                    ],
                }
            ],
        }
    ]
    rendered = render_html(report, groups)
    assert "Measured results and frozen cohort identities" in rendered
    assert "npa.field-failure.reference-result.v1" in rendered
    assert "Baseline retained" in rendered
    assert "warehouse retention" in rendered

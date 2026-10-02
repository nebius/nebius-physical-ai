"""Prove balanced route isolation and prevent office gains masking warehouse loss."""

import copy
import json

import pytest

from npa.workflows.field_failure.reference_demo_cohorts import freeze_balanced_cohorts
from npa.workflows.field_failure.reference_demo_inputs import (
    capture_recipe,
    evaluation_bundle,
    prepare_warehouse,
    reference_metrics,
)
from npa.workflows.field_failure.reference_demo_evaluate import development_decision
from npa.workflows.field_failure.reference_demo_regions import region_comparison
from npa.workflows.navigation.reference_identity import collision_identity
from npa.workflows.navigation.reference_replay import apply_replay, compose_scene
from npa.workflows.navigation.reference_scene import cases, warehouse

IMAGE = "registry.example/native@sha256:" + "a" * 64


def _balanced(tmp_path, count=20):
    baseline = tmp_path / "baseline"
    prepare_warehouse(baseline, image=IMAGE, count=count)
    scan = cases(count)
    manifest = freeze_balanced_cohorts(baseline, scan, baseline / "scene.usdz")
    return baseline, scan, manifest


def test_balanced_split_keeps_all_training_and_seals_disjoint_final_routes(tmp_path):
    baseline, scan, manifest = _balanced(tmp_path)
    assert manifest["cohorts"]["training"]["count"] == 40
    for cohort in ("development", "final"):
        regions = manifest["regions"][cohort]
        assert regions["office"]["count"] == regions["warehouse"]["count"] == 10
        selected = json.loads((baseline / (cohort + "-cases.json")).read_text())
        assert selected[0]["id"] == regions["office"]["case_ids"][0]
    capture = tmp_path / "capture"
    capture.mkdir()
    capture_recipe(capture, scan, baseline)
    training = json.loads((capture / "recipe.json").read_text())
    learner_seeds = {
        row["seed"] for key in ("train_cases", "eval_cases") for row in training[key]
    }
    final_seeds = {
        seed
        for region in manifest["regions"]["final"].values()
        for seed in region["seeds"]
    }
    assert not final_seeds & learner_seeds
    for final in (False, True):
        output = tmp_path / str(final)
        recipe = evaluation_bundle(baseline, output, final=final)
        assert recipe["scene_sha256"] == manifest["geometry"]["scene_sha256"]
        assert len(recipe["eval_cases"]) == 20
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    warehouse(prepared / "scene.usdz")
    apply_replay(capture, prepared, training)
    from npa.workflows.field_failure.reference_demo_inputs import cohort_manifest

    actual = cohort_manifest({"training": training["train_cases"]})
    assert actual["cohorts"]["training"] == manifest["cohorts"]["training"]


def test_reconstruction_cannot_change_the_frozen_collision_surface(tmp_path):
    baseline, scan, _ = _balanced(tmp_path, count=4)
    capture = tmp_path / "capture"
    capture.mkdir()
    capture_recipe(capture, scan, baseline)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    compose_scene(
        baseline / "scene.usdz", baseline / "scene.usdz", prepared / "scene.usdz"
    )
    recipe = json.loads((capture / "recipe.json").read_text())
    with pytest.raises(ValueError, match="pre-learning freeze"):
        apply_replay(capture, prepared, recipe)


def test_failure_observation_keeps_frozen_training_cases_and_never_changes_cohorts(
    tmp_path,
):
    from npa.workflows.field_failure.reference_demo_observation import (
        observation_bundle,
    )
    from npa.workflows.navigation.artifacts import _files
    from npa.workflows.navigation.reference_replay import translated_cases

    baseline, scan, frozen = _balanced(tmp_path, count=4)
    capture = tmp_path / "capture"
    capture.mkdir()
    capture_recipe(capture, scan, baseline)
    original = _files(baseline), _files(capture)
    recipe = observation_bundle(baseline, capture, tmp_path / "observation")
    assert recipe["eval_cases"] == translated_cases(
        scan["train_cases"], [50.0, 0.0, 0.0]
    )
    actual = {case["seed"] for case in recipe["eval_cases"]}
    assert len(actual) == 4
    for cohort in ("development", "final"):
        for region in frozen["regions"][cohort].values():
            assert not actual.intersection(region["seeds"])
    assert (_files(baseline), _files(capture)) == original
    assert recipe["iterations"] == 1500 and recipe["episode_steps"] == 300


def test_geometry_identity_ignores_triangle_order_but_keeps_winding_and_position(
    tmp_path,
):
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "source.usdz"
    warehouse(source)
    expected = collision_identity(source)
    stage = Usd.Stage.Open(str(source))
    stage.Export(str(tmp_path / "reordered.usda"))
    changed = Usd.Stage.Open(str(tmp_path / "reordered.usda"))
    mesh = next(
        UsdGeom.Mesh(p) for p in changed.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
    )
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    faces = [indices[i : i + 3] for i in range(0, len(indices), 3)]
    mesh.GetFaceVertexIndicesAttr().Set([i for face in reversed(faces) for i in face])
    changed.GetRootLayer().Save()
    assert collision_identity(tmp_path / "reordered.usda") == expected
    mesh.GetFaceVertexIndicesAttr().Set([i for face in faces for i in reversed(face)])
    changed.GetRootLayer().Save()
    assert collision_identity(tmp_path / "reordered.usda") != expected


def _evaluations():
    regions = {
        name: {
            "count": 100,
            "seeds": list(range(start, start + 100)),
            "case_ids": [str(i) for i in range(start, start + 100)],
        }
        for name, start in (("office", 0), ("warehouse", 100))
    }
    before, after = [], []
    for seed in range(200):
        baseline_success = seed >= 100 or seed < 60
        candidate_success = seed < 190
        for rows, success in ((before, baseline_success), (after, candidate_success)):
            rows.append(
                {
                    "seed": seed,
                    "case_id": str(seed),
                    "success": success,
                    "collision_steps": 0,
                    "physical_failure_steps": 0,
                    "goal_distance_m": 0.0 if success else 1.0,
                }
            )
    return before, after, regions


def test_improved_overall_success_cannot_hide_warehouse_regression():
    before, after, regions = _evaluations()
    reports = [
        {
            "episodes": rows,
            "success_rate": sum(row["success"] for row in rows) / 200,
            "evaluation_inputs_sha256": "c" * 64,
            "checkpoint_sha256": identity * 64,
        }
        for rows, identity in ((before, "a"), (after, "b"))
    ]
    decision = development_decision(*reports, regions, reference_metrics(300))
    assert decision["success_rate_gain"] > 0.01
    assert decision["candidate_success_rate"] >= 0.8
    assert not decision["eligible"]
    assert not decision["regional"]["warehouse_retention_passed"]
    assert "warehouse regressed on success" in decision["reasons"]
    assert not decision["final_cohort_consumed"]


def test_region_checks_reject_missing_duplicate_or_reassigned_episodes():
    before, after, regions = _evaluations()
    for rows in (after[:-1], after + [after[0]]):
        with pytest.raises(ValueError, match="frozen regional cohort"):
            region_comparison(before, rows, regions)
    changed = copy.deepcopy(after)
    changed[0]["case_id"] = "another-case"
    with pytest.raises(ValueError, match="identity differs"):
        region_comparison(before, changed, regions)


def test_final_html_summary_cannot_promote_when_warehouse_retention_fails():
    from npa.workflows.field_failure.reference_demo_report import result_summary
    from npa.workflows.field_failure.reference_demo_paired import (
        paired_development_regressions,
    )

    before, after, regions = _evaluations()
    regional = region_comparison(before, after, regions)
    selection = {
        "eligible": True,
        "baseline_success_rate": 0.8,
        "candidate_success_rate": 0.95,
        "success_rate_gain": 0.15,
        "reasons": [],
        "final_cohort_consumed": False,
        "regional": regional,
        "paired": paired_development_regressions(
            before, after, regions, reference_metrics(300)
        ),
    }
    final = {
        "promote_checkpoint": True,
        "recommendation": "promote",
        "episodes_per_policy": 200,
        "regressions": [],
        "metrics": {
            "success": {
                "baseline_mean": 0.8,
                "candidate_mean": 0.95,
                "improvement": 0.15,
            }
        },
    }
    plan = {
        "cohorts": {},
        "num_envs": 200,
        "baseline_iterations": 1500,
        "candidate_iterations": 1500,
    }
    summary = result_summary(plan, selection, final, regional)
    assert not summary["quality_passed"]
    assert summary["recommendation"] == "retain_baseline"
    assert (
        summary["final"]["regional"]["regions"]["office"]["candidate_success_rate"]
        == 1.0
    )


def test_replay_routes_must_match_pre_learning_case_bytes(tmp_path):
    baseline, scan, _ = _balanced(tmp_path, count=4)
    capture = tmp_path / "capture"
    capture.mkdir()
    capture_recipe(capture, scan, baseline)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    warehouse(prepared / "scene.usdz")
    recipe = json.loads((capture / "recipe.json").read_text())
    recipe["train_cases"][0]["goal_m"][0] += 0.1
    with pytest.raises(ValueError, match="routes differ from the pre-learning freeze"):
        apply_replay(capture, prepared, recipe)


def test_per_region_contact_increase_cannot_hide_behind_other_region_decrease():
    before, _, regions = _evaluations()
    after = copy.deepcopy(before)
    before[99]["collision_steps"] = 2
    after[199]["collision_steps"] = 2
    result = region_comparison(before, after, regions)
    assert not result["passed"]
    assert "warehouse regressed on collision_steps" in result["reasons"]

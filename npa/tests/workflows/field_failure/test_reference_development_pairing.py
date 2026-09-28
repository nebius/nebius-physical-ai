"""Keep development eligibility aligned with final paired rules without reading final data."""

import copy
import math
from types import SimpleNamespace

import pytest

from npa.workflows.field_failure.comparison import _paired_metrics
from npa.workflows.field_failure.reference_demo_evaluate import development_decision
from npa.workflows.field_failure.reference_demo_inputs import reference_metrics
from npa.workflows.field_failure.reference_demo_paired import (
    paired_development_regressions,
)


def _experiment():
    regions = {
        name: {
            "count": 2000,
            "seeds": list(range(start, start + 2000)),
            "case_ids": [f"unit-case-{seed}" for seed in range(start, start + 2000)],
        }
        for name, start in (("office", 0), ("warehouse", 2000))
    }
    arms = []
    for checkpoint, successes in (("a", 1620), ("b", 1640)):
        rows = [
            {
                "case_id": f"unit-case-{seed}",
                "seed": seed,
                "success": seed % 2000 < successes,
                "collision_steps": 0,
                "physical_failure_steps": 0,
                "goal_distance_m": 0.0 if seed % 2000 < successes else 1.0,
            }
            for seed in range(4000)
        ]
        arms.append(
            {
                "episodes": rows,
                "checkpoint_sha256": checkpoint * 64,
                "evaluation_inputs_sha256": "c" * 64,
                "success_rate": successes / 2000,
            }
        )
    return *arms, regions, reference_metrics(300)


def _final_pairing(before, after, metrics):
    bundle = SimpleNamespace(
        metrics=[SimpleNamespace(**m) for m in metrics], primary_metric="success"
    )
    arms = [
        SimpleNamespace(
            episodes=[
                SimpleNamespace(
                    scenario_id="unit-development",
                    seed=row["seed"],
                    metrics={m["name"]: float(row[m["name"]]) for m in metrics},
                )
                for row in report["episodes"]
            ]
        )
        for report in (before, after)
    ]
    return _paired_metrics(bundle, *arms)


def _assert_regression_parity(decision, before, after, metrics):
    _, final_rows, _ = _final_pairing(before, after, metrics)
    assert {(row["metric"], row["seed"], row["improvement"]) for row in final_rows} == {
        (row["metric"], row["seed"], row["improvement"])
        for row in decision["paired"]["violations"]
    }


def _masked_success_regression():
    before, after, regions, metrics = _experiment()
    after["episodes"][0].update(success=False, goal_distance_m=1.0)
    after["episodes"][1640].update(success=True, goal_distance_m=0.0)
    return before, after, regions, metrics


def test_exact_one_percentage_point_uses_identical_paired_final_arithmetic():
    before, after, regions, metrics = _experiment()
    assert after["success_rate"] - before["success_rate"] < 0.01
    decision = development_decision(before, after, regions, metrics)
    final_metrics, regressions, promote = _final_pairing(before, after, metrics)
    assert decision["eligible"] and promote and not regressions
    assert (
        decision["success_rate_gain"] == final_metrics["success"]["improvement"] == 0.01
    )
    assert decision["paired"]["paired_cases"] == 4000
    assert not decision["paired"]["violation_count"]


def test_success_loss_cannot_hide_behind_identical_improved_region_means():
    before, after, regions, metrics = _masked_success_regression()
    decision = development_decision(before, after, regions, metrics)
    assert decision["regional"]["passed"] and decision["success_rate_gain"] == 0.01
    assert not decision["eligible"] and decision["selected_checkpoint_sha256"] is None
    assert decision["paired"]["regressed_cases"] == 1
    assert decision["paired"]["violation_count"] == 2
    assert decision["paired"]["violations_by_metric"]["success"] == 1
    assert decision["paired"]["violations_by_region"] == {"office": 2, "warehouse": 0}
    _assert_regression_parity(decision, before, after, metrics)


@pytest.mark.parametrize("metric", ["collision_steps", "physical_failure_steps"])
def test_paired_safety_loss_cannot_hide_behind_another_cases_recovery(metric):
    before, after, regions, metrics = _experiment()
    before["episodes"][1900][metric] = 2
    after["episodes"][1901][metric] = 2
    decision = development_decision(before, after, regions, metrics)
    assert decision["regional"]["passed"]
    assert not decision["eligible"]
    assert decision["paired"]["violation_count"] == 1
    assert decision["paired"]["violations"][0]["seed"] == 1901
    _assert_regression_parity(decision, before, after, metrics)


@pytest.mark.parametrize(
    "distance,blocked", [(0.25, False), (math.nextafter(0.25, math.inf), True)]
)
def test_goal_distance_strict_boundary_matches_final_without_epsilon(distance, blocked):
    before, after, regions, metrics = _experiment()
    for report in (before, after):
        report["episodes"][1900].update(goal_distance_m=0.0, physical_failure_steps=1)
    after["episodes"][1900]["goal_distance_m"] = distance
    decision = development_decision(before, after, regions, metrics)
    assert decision["regional"]["passed"]
    assert decision["eligible"] is not blocked
    assert bool(decision["paired"]["violation_count"]) is blocked
    _assert_regression_parity(decision, before, after, metrics)


def test_guard_reads_frozen_metric_limits_instead_of_substituting_defaults():
    before, after, regions, metrics = _experiment()
    metrics[-1]["maximum_regression"] = 0.125
    after["episodes"][1900]["goal_distance_m"] = 1.2
    decision = development_decision(before, after, regions, metrics)
    assert not decision["eligible"]
    assert decision["paired"]["violations"][0]["maximum_regression"] == 0.125
    assert decision["paired"]["metric_limits"] == metrics
    _assert_regression_parity(decision, before, after, metrics)


def test_pairing_is_independent_of_episode_order():
    before, after, regions, metrics = _masked_success_regression()
    expected = development_decision(before, after, regions, metrics)
    before["episodes"] = before["episodes"][::2] + before["episodes"][1::2]
    after["episodes"].reverse()
    assert development_decision(before, after, regions, metrics) == expected


@pytest.mark.parametrize("change", ["missing_id", "wrong_seed", "duplicate", "missing"])
def test_pairing_requires_exact_frozen_case_ids_and_seeds(change):
    before, after, regions, metrics = _experiment()
    if change == "missing_id":
        after["episodes"][0].pop("case_id")
    elif change == "wrong_seed":
        after["episodes"][0]["seed"] = 100000
    elif change == "duplicate":
        after["episodes"][1] = copy.deepcopy(after["episodes"][0])
    else:
        after["episodes"].pop()
    with pytest.raises(ValueError, match="case/seed"):
        paired_development_regressions(
            before["episodes"], after["episodes"], regions, metrics
        )


def _fixture_preview():
    from PIL import Image
    from npa.workflows.preview_html import image_preview

    return [
        {
            "title": "Synthetic test fixture",
            "note": "Unit rendering fixture; not runtime evidence.",
            "frames": [
                {
                    "label": "Fixture",
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


def _selection_records():
    before, after, regions, metrics = _masked_success_regression()
    plan = {
        "cohorts": {"regions": {"development": regions}},
        "metrics": metrics,
        "num_envs": 4000,
        "baseline_iterations": 1500,
        "candidate_iterations": 1500,
    }
    args = SimpleNamespace(output_root="s3://example/demo", run_id="unit-development")
    return args, {
        args.output_root + "/development-baseline.json": before,
        args.output_root + "/development-candidate.json": after,
        args.output_root + "/reference-plan.json": plan,
    }


def _capture_selection(monkeypatch, records):
    from npa.workflows.field_failure import reference_demo_evaluate as evaluate
    from npa.workflows.field_failure import reference_demo_report as report
    from npa.workflows.field_failure import reference_demo_media as media
    from npa.workflows.field_failure import reference_demo_attribution as attribution

    # This test isolates cohort reads; sealed attribution has dedicated tests.
    monkeypatch.setattr(attribution, "sample_credit", lambda *_: None)

    observed, published = [], {}

    def read(uri, *_):
        observed.append(uri)
        assert uri in records, "selection must never read final artifacts"
        return records[uri], "d" * 64

    def preview(_args, final):
        assert final is None
        return _fixture_preview()

    for module in (evaluate, report):
        monkeypatch.setattr(module, "_read", read)
        monkeypatch.setattr(
            module, "_publish", lambda uri, value: published.update({uri: value})
        )
    monkeypatch.setattr(media, "preview_groups", preview)
    storage = SimpleNamespace(
        put_bytes_conditional=lambda data, uri, **_: published.update(
            {uri: data.decode()}
        )
    )
    monkeypatch.setattr(report, "_storage", lambda: storage)
    return observed, published


def test_paired_failure_publishes_readable_html_without_reading_any_final_artifact(
    monkeypatch,
):
    from npa.workflows.field_failure import reference_demo_evaluate as evaluate

    args, records = _selection_records()
    observed, published = _capture_selection(monkeypatch, records)
    with pytest.raises(RuntimeError, match="final cohort remains untouched"):
        evaluate.select_candidate(args)
    selected = published[args.output_root + "/selection.json"]
    html = published[args.output_root + "/reports/index.html"]
    assert not selected["eligible"] and not selected["final_cohort_consumed"]
    assert selected["paired"]["violation_count"] == 2
    assert "Development selection blocked" in html and "unit-case-0" in html
    assert "View all 2 paired metric violations" in html and "Allowed" in html
    assert set(observed) == set(records)

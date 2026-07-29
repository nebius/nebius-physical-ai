"""Guards for daily comprehensive-workflow image coverage.

Protects the invariant the daily dev-VM ``e2e-daily`` tier relies on: every
required workflow-reachable image is exercised by at least one >= 4-step
(comprehensive) workflow, and the daily plan-set actually covers them.
"""

from __future__ import annotations

from npa.orchestration.npa_workflow import daily_coverage as dc


def test_some_comprehensive_specs_exist() -> None:
    comp = dc.comprehensive_specs()
    assert comp, "expected at least one >= 4-step workflow spec"
    for summary in comp:
        assert summary.exec_steps >= dc.MIN_COMPREHENSIVE_STEPS


def test_required_images_are_covered_by_comprehensive_workflows() -> None:
    # Regression guard: a required image dropping out of every >= 4-step
    # workflow must fail here (extend a workflow or justify EXEMPT_IMAGE_TOOLS).
    dc.assert_coverage()


def test_covered_and_exempt_do_not_overlap() -> None:
    report = dc.image_coverage()
    assert not (report.covered & dc.EXEMPT_IMAGE_TOOLS), (
        "an image is both covered and exempt; drop it from EXEMPT_IMAGE_TOOLS"
    )


def test_exempt_images_are_real_workflow_image_tools() -> None:
    # Exemptions must reference images the renderer actually knows about, so the
    # gap stays meaningful and typos are caught.
    assert dc.EXEMPT_IMAGE_TOOLS <= dc.WORKFLOW_IMAGE_TOOLS


def test_daily_plan_set_covers_every_covered_image() -> None:
    summaries = dc.spec_step_summary()
    report = dc.image_coverage(summaries)
    for day in (0, 1, 100, 200, 366):
        plan = dc.daily_plan_set(day, summaries)
        assert plan, "daily plan-set must not be empty"
        planned_images: set[str] = set()
        for summary in plan:
            assert summary.is_comprehensive
            planned_images |= summary.image_tools
        assert report.covered <= planned_images, (
            f"day {day}: plan-set misses {sorted(report.covered - planned_images)}"
        )


def test_rotating_spec_changes_across_days() -> None:
    summaries = dc.spec_step_summary()
    picks = {dc.rotating_spec(day, summaries).name for day in range(len(dc.comprehensive_specs(summaries)))}
    assert len(picks) > 1, "rotating spec should vary across days"

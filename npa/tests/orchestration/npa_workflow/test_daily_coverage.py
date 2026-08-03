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
    comp = dc.comprehensive_specs(summaries)
    assert comp, "expected comprehensive specs to rotate over"
    picks = {dc.rotating_spec(day, summaries).name for day in range(len(comp))}
    assert len(picks) > 1, "rotating spec should vary across days"


def test_gpu_submit_rotation_covers_all_twins_and_excludes_plan_only() -> None:
    from npa.orchestration.npa_workflow.submit_matrix import (
        gpu_submit_cases,
        rotating_gpu_submit_case,
    )

    cases = gpu_submit_cases()
    assert cases, "expected at least one real-GPU-launching workflow twin"
    # Never rotate onto a plan-only stub (those never launch a GPU) or a
    # rotation_skip twin that cannot pass standalone today (each carries a
    # skip_reason, e.g. sonic-eval consumes an ONNX a previous export wrote).
    assert all(not c.plan_only for c in cases)
    assert all(not c.rotation_skip for c in cases)
    assert all(c.tier in {"gpu", "multi"} for c in cases)
    rotation = {c.spec for c in cases}
    # Verified-passing on real GPU (RTXPRO-6000) stay in the rotation.
    for good in (
        "mjlab-eval.yaml",
        "cosmos3-reason.yaml",
        "tokenfactory-rollout-judge.yaml",
        # SONIC twins are self-contained now: the in-job train runtime writes a
        # checkpoint each downstream stage reads back from S3.
        "sonic-train.yaml",
        "sonic-export.yaml",
        "sonic-export-eval.yaml",
        "sonic-locomotion-finetuning.yaml",
        "tokenfactory-cosmos-gate.yaml",
        # Self-hosted vLLM, bounded by serving a 2B VLM and pre-fetching weights.
        "vlm-eval-single.yaml",
    ):
        assert good in rotation, f"{good} should be in the rotation"
    # Twins that can't pass as a standalone submit today are excluded.
    for bad in (
        "sonic-eval.yaml",
        "bdd100k-pipeline.yaml",
    ):
        assert bad not in rotation, f"{bad} should be excluded from the rotation"
    # Every rotation_skip twin must document why (so the gap stays visible).
    from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

    assert all(c.skip_reason for c in SUBMIT_LIVE_MATRIX if c.rotation_skip)
    # Over one full cycle the rotation visits every GPU twin.
    seen = {rotating_gpu_submit_case(day).spec for day in range(len(cases))}
    assert seen == {c.spec for c in cases}


def test_declared_case_budget_survives_the_daily_runner_cap(monkeypatch) -> None:
    """A case's declared ``max_wait_seconds`` must not be truncated by the daily cap.

    ``scripts/dev-vm-daily-tests.sh`` exports
    ``NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS=2400`` for every rotating GPU
    submit. The submit e2e used to compute two different deadlines from that: the
    CLI got ``case.max_wait_seconds`` while the polling loop got the env value, so
    a case declaring a longer budget was cancelled at 40 minutes by
    ``CANCEL_ON_TIMEOUT`` even though the run was healthy. Any workflow slower
    than the cap (cold multi-GB image pull plus a long train) would go red on
    whichever day the rotation reached it.
    """
    import importlib
    from pathlib import Path

    # The documented full-suite invocation runs from the repository root and
    # points pytest at ``npa/tests``. In that layout the tests package's parent
    # is not necessarily on sys.path, so make the package import deterministic.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    mod = importlib.import_module("tests.e2e.test_npa_workflow_submit_live_e2e")
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS", "2400")

    from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

    declared = [c for c in SUBMIT_LIVE_MATRIX if c.max_wait_seconds]
    assert declared, "expected at least one case to declare its own budget"
    for case in declared:
        assert mod._case_max_wait(case) >= case.max_wait_seconds, (
            f"{case.spec} would be cancelled at the daily cap despite declaring "
            f"{case.max_wait_seconds}s"
        )
    # And a case that declares nothing still follows the operator's env value.
    plain = next((c for c in SUBMIT_LIVE_MATRIX if not c.max_wait_seconds), None)
    if plain is not None:
        assert mod._case_max_wait(plain) == 2400

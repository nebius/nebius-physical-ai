"""Select candidates on development evidence without consulting final evaluation cases."""

from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _publish, _read
from npa.workflows.field_failure.native_artifacts import _bundle, _download
from npa.workflows.field_failure.native_policy import _initialize
from npa.workflows.navigation.stages import prepare, run_stage


def evaluate_development(args, arm):
    """Evaluate one exact checkpoint on the frozen development cohort only.

    Args:
        args: Run artifact location and immutable execution settings.
        arm: Baseline or candidate policy selector.
    Returns:
        Verified native development evaluation report.
    Raises:
        ValueError: Sealed scene, cohort, image or checkpoint identity differs.
        RuntimeError: Native physical controls or evaluation fail.
        OSError: Required artifacts are unavailable.
    """
    plan, _ = _read(args.output_root + "/reference-plan.json")
    policy = _policy(args, arm)
    protocol, _ = _read(plan["protocol"]["uri"], plan["protocol"]["sha256"])
    with tempfile.TemporaryDirectory(prefix="npa-development-") as temporary:
        root = Path(temporary)
        source = _bundle(plan["development"], root, "input")
        checkpoint = root / "baseline.pt"
        _download(policy["checkpoint"], checkpoint)
        _initialize(source, protocol, checkpoint)
        prepared = root / "prepared"
        prepare(str(source), str(prepared), protocol["navigation_image"])
        report = run_stage(
            "evaluate-checkpoint",
            str(prepared),
            args.output_root + "/development/" + arm,
        )
    _verify_development(report, plan, policy)
    _publish(args.output_root + "/development-" + arm + ".json", report)
    return report


def _policy(args, arm):
    if arm == "baseline":
        artifact, _ = _read(args.output_root + "/bundle-reference.json")
        bundle, _ = _read(artifact["uri"], artifact["sha256"])
        return bundle["baseline"]
    if arm != "candidate":
        raise ValueError("development arm must be baseline or candidate")
    training, _ = _read(args.output_root + "/loop/" + args.run_id + "/training.json")
    return training["candidate"]


def _verify_development(report, plan, policy):
    if report["checkpoint_sha256"] != policy["checkpoint"]["sha256"]:
        raise ValueError("development evaluation loaded a different checkpoint")
    seeds = [row["seed"] for row in report["episodes"]]
    if sorted(seeds) != sorted(plan["development_seeds"]):
        raise ValueError("development evaluation differs from its frozen cohort")
    if report["runtime"]["scene_sha256"] != plan["cohorts"]["geometry"]["scene_sha256"]:
        raise ValueError("development scene differs from the pre-learning freeze")
    if (
        report["evaluation_inputs_sha256"]
        != plan["cohorts"]["cohorts"]["development"]["sha256"]
    ):
        raise ValueError("development case bytes differ from the pre-learning freeze")


def development_decision(baseline, candidate, regions, metrics):
    """Separate development eligibility from final policy promotion and runtime success.

    Args:
        baseline: Native baseline development report.
        candidate: Native candidate report on the identical cohort.
        regions: Precommitted balanced region identities from the reference plan.
        metrics: Frozen reference-plan metric contracts shared with final comparison.
    Returns:
        Eligibility, actual quality measurements and failure explanations.
    Raises:
        ValueError: Cohorts differ or checkpoint identities were reused.
    """
    if baseline["evaluation_inputs_sha256"] != candidate["evaluation_inputs_sha256"]:
        raise ValueError("development arms used different evaluation cases")
    if baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]:
        raise ValueError("development candidate is the unchanged baseline")
    from npa.workflows.field_failure.reference_demo_regions import region_comparison
    from npa.workflows.field_failure.reference_demo_paired import (
        paired_development_regressions,
    )

    regional = region_comparison(baseline["episodes"], candidate["episodes"], regions)
    paired = paired_development_regressions(
        baseline["episodes"], candidate["episodes"], regions, metrics
    )
    _verify_success_rates(baseline, candidate)
    gain = paired["success_rate_gain"]
    reasons = _selection_reasons(baseline, candidate, regional, paired, gain)
    return {
        "schema": "npa.field-failure.development-selection.v1",
        "eligible": not reasons,
        "runtime_completed": True,
        "baseline_success_rate": baseline["success_rate"],
        "candidate_success_rate": candidate["success_rate"],
        "success_rate_gain": gain,
        "regional": regional,
        "paired": paired,
        "reasons": reasons,
        "selected_checkpoint_sha256": candidate["checkpoint_sha256"]
        if not reasons
        else None,
        "final_cohort_consumed": False,
        "deployment_authorized": False,
    }


def _selection_reasons(baseline, candidate, regional, paired, gain):
    reasons = list(regional["reasons"])
    if candidate["success_rate"] < 0.8:
        reasons.append("candidate development success is below 80%")
    if gain < 0.01:
        reasons.append("candidate development gain is below one percentage point")
    for metric in ("collision_steps", "physical_failure_steps"):
        before = sum(row[metric] for row in baseline["episodes"])
        after = sum(row[metric] for row in candidate["episodes"])
        if after > before:
            reasons.append("candidate increased development " + metric)
    if not paired["passed"]:
        reasons.append(
            f"candidate has {paired['violation_count']} paired development metric violations "
            f"across {paired['regressed_cases']} cases; final cohort remains untouched"
        )
    return reasons


def _verify_success_rates(*reports):
    for report in reports:
        actual = sum(row["success"] for row in report["episodes"]) / len(
            report["episodes"]
        )
        if actual != report["success_rate"]:
            raise ValueError(
                "development summary differs from actual episode successes"
            )


def select_candidate(args):
    """Freeze development selection and withhold final evaluation on failed quality.

    Args:
        args: Run-scoped artifact location.
    Returns:
        Published development selection record.
    Raises:
        RuntimeError: Candidate is ineligible; an HTML quality report is published first.
        ValueError: Evaluation identities or metrics are inconsistent.
        OSError: Required evidence cannot be read or published.
    """
    before, _ = _read(args.output_root + "/development-baseline.json")
    after, _ = _read(args.output_root + "/development-candidate.json")
    plan, _ = _read(args.output_root + "/reference-plan.json")
    decision = development_decision(
        before, after, plan["cohorts"]["regions"]["development"], plan["metrics"]
    )
    from npa.workflows.field_failure.reference_demo_publication import publish_record

    publish_record(args.output_root + "/selection.json", decision)
    if not decision["eligible"]:
        from npa.workflows.field_failure.reference_demo_report import publish_report

        publish_report(args, selection=decision)
        raise RuntimeError(
            "development quality gate failed; final cohort remains untouched; see reports/index.html"
        )
    return decision

"""Publish an offline HTML report separating completed runtime from policy quality."""

import html
import json
from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _read
from npa.workflows.field_failure.reference_demo_publication import (
    publish_html,
    publish_record,
)


def publish_report(args, *, selection=None):
    """Publish factual development or final results without live infrastructure details.

    Args:
        args: Run-scoped artifact root and run identifier.
        selection: Failed development selection, or None for completed final comparison.
    Returns:
        Sanitized result summary with a separate quality outcome.
    Raises:
        ValueError: Expected measured comparison fields are missing or invalid.
        OSError: Evidence retrieval or publication fails.
    """
    plan, _ = _read(args.output_root + "/reference-plan.json")
    final = None
    if selection is None:
        selection, _ = _read(args.output_root + "/selection.json")
        final, _ = _read(args.output_root + "/loop/" + args.run_id + "/decision.json")
    from npa.workflows.field_failure.reference_demo_regions import final_regions

    regional = None if final is None else final_regions(args, plan, final)
    report = result_summary(plan, selection, final, regional)
    from npa.workflows.field_failure.reference_demo_attribution import sample_credit

    report["public_sample"] = sample_credit(args.output_root, plan)
    from npa.workflows.field_failure.reference_demo_admission_report import (
        bound_admission_summary,
    )

    report["failure_admission"] = bound_admission_summary(args)
    from npa.workflows.field_failure.reference_demo_media import preview_groups

    groups = preview_groups(args, final)
    publish_record(args.output_root + "/reports/result.json", report)
    publish_html(
        args.output_root + "/reports/index.html",
        render_html(report, groups),
    )
    return report


def result_summary(plan, selection, final, regional=None):
    """Derive readiness only from actual development selection and final comparison.

    Args:
        plan: Frozen public reference cohort manifest and recipe schedule.
        selection: Measured development eligibility.
        final: Actual final comparison or None when development blocked it.
        regional: Hash-bound final regional metrics, or None before final evaluation.
    Returns:
        Public summary with no credentials, storage locations or infrastructure names.
    Raises:
        KeyError: Required measured evidence is missing.
    """
    scores = None if final is None else final["metrics"]["success"]
    quality = bool(
        final
        and final["promote_checkpoint"]
        and scores["candidate_mean"] >= 0.8
        and regional
        and regional["passed"]
    )
    return {
        "schema": "npa.field-failure.reference-result.v1",
        "runtime_completed": True,
        "quality_passed": quality,
        "final_evaluated": final is not None,
        "deployment_authorized": False,
        "recommendation": "promote" if quality else "retain_baseline",
        "development": _development_summary(selection),
        "final": _final_summary(final, scores, regional),
        "cohorts": plan["cohorts"],
        "robots": plan["num_envs"],
        "baseline_iterations": plan["baseline_iterations"],
        "candidate_iterations": plan["candidate_iterations"],
        "scope": "Balanced office adaptation and warehouse retention: new routes in both known public layouts. The focal rendered episode is an office route; aggregate scores cover both regions.",
        "limitations": [
            "Actual baseline simulation failures admit the public capture for reconstruction and continuation. Initial scene preparation supports observation; physical field-log ingestion and private robot integration remain operator-supplied.",
            "Range observations; camera-conditioned navigation is not qualified.",
            "New-site transfer and private robot integration are not established.",
            "The earlier scan-to-unseen-warehouse failure and original observation gap remain historical evidence.",
        ],
    }


def _development_summary(selection):
    return {
        key: selection[key]
        for key in (
            "eligible",
            "baseline_success_rate",
            "candidate_success_rate",
            "success_rate_gain",
            "reasons",
            "final_cohort_consumed",
            "regional",
            "paired",
        )
    }


def _final_summary(final, scores, regional):
    if scores is None:
        return None
    return {
        "baseline_success_rate": scores["baseline_mean"],
        "candidate_success_rate": scores["candidate_mean"],
        "success_rate_gain": scores["improvement"],
        "episodes_per_policy": final["episodes_per_policy"],
        "regressions": len(final["regressions"]),
        "comparison_recommendation": final["recommendation"],
        "regional": regional,
    }


def render_html(report, groups):
    """Render a standalone accessible result page with explicit scope and quality status.

    Args:
        report: Sanitized result summary derived from actual evidence.
        groups: Actual scored-frame groups with embedded image data.
    Returns:
        Complete HTML with embedded styles and no remote assets.
    Raises:
        KeyError: Required summary fields are absent.
    """
    from npa.workflows.preview_html import write_preview

    metrics = _display_metrics(report)
    summary = " ".join(
        [
            report["scope"],
            *report["limitations"],
            "Runtime completion and policy quality are separate. No policy is deployed.",
        ]
    )
    data = html.escape(json.dumps(report, indent=2, allow_nan=False))
    with tempfile.TemporaryDirectory(prefix="npa-report-html-") as temporary:
        path = Path(temporary) / "index.html"
        write_preview(
            path,
            title="Public RL improvement demo",
            summary=summary,
            metrics=metrics,
            groups=groups,
            details={"public_sample": report["public_sample"]}
            if report.get("public_sample")
            else None,
        )
        contents = path.read_text()
    details = f"<details><summary>Measured results and frozen cohort identities</summary><pre>{data}</pre></details>"
    diagnostics = _development_diagnostics(report["development"])
    return contents.replace("</body>", diagnostics + details + "</body>")


def _display_metrics(report):
    status = (
        "Quality gates passed"
        if report["quality_passed"]
        else "Baseline retained: quality gates not passed"
    )
    metrics = {
        "Quality": status,
        "Parallel robots": report["robots"],
        "Baseline PPO iterations": report["baseline_iterations"],
        "Continuation PPO iterations": report["candidate_iterations"],
        "Development paired checks": _paired_status(report["development"]["paired"]),
    }
    if "failure_admission" in report:
        admission = report["failure_admission"]
        metrics["Observed baseline simulation failures"] = (
            f"{admission['observed_failures']} / {admission['episodes']} office training routes"
        )
        metrics["Public capture admitted from measured failures"] = admission[
            "admitted"
        ]
    for name in ("development", "final"):
        values = report[name]
        if values is None:
            metrics[name.title()] = "Not evaluated; final cohort remains untouched"
            continue
        before, after = (
            values["baseline_success_rate"],
            values["candidate_success_rate"],
        )
        metrics[name.title() + " success"] = (
            f"Baseline {before:.2%} → candidate {after:.2%} ({after - before:+.2%})"
        )
        _display_regions(metrics, name, values["regional"])
    return metrics


def _paired_status(paired):
    if paired["passed"]:
        return f"Passed · {paired['paired_cases']:,} paired cases · no per-case regressions"
    return (
        f"Blocked · {paired['violation_count']:,} metric violations "
        f"across {paired['regressed_cases']:,} cases"
    )


def _development_diagnostics(development):
    if development["eligible"]:
        return ""
    reasons = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in development["reasons"]
    )
    paired = development["paired"]
    contents = (
        "<section><h2>Development selection blocked</h2>"
        "<p>The candidate was not selected. Final evaluation remains untouched.</p>"
        f"<ul>{reasons}</ul>"
    )
    if paired["violations"]:
        rows = "".join(_violation_row(row) for row in paired["violations"])
        contents += (
            f"<details><summary>View all {paired['violation_count']:,} paired metric violations</summary>"
            '<div style="overflow-x:auto"><table><caption>Development cases only; limits come from the frozen plan.</caption>'
            "<thead><tr><th>Region</th><th>Case</th><th>Seed</th><th>Metric</th>"
            "<th>Baseline</th><th>Candidate</th><th>Regression</th><th>Allowed</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div></details>"
        )
    return contents + "</section>"


def _violation_row(row):
    values = [row["region"], row["case_id"], row["seed"], row["metric"]]
    values.extend(
        str(value)
        for value in (
            row["baseline"],
            row["candidate"],
            -row["improvement"],
            row["maximum_regression"],
        )
    )
    cells = "".join(
        f'<td style="padding:8px">{html.escape(str(value))}</td>' for value in values
    )
    return f"<tr>{cells}</tr>"


def _display_regions(metrics, cohort, regional):
    if regional is None:
        metrics[cohort.title() + " region checks"] = "Missing; quality cannot pass"
        return
    metrics[cohort.title() + " warehouse retention"] = (
        "Passed" if regional["warehouse_retention_passed"] else "Failed"
    )
    for name, scores in regional["regions"].items():
        before, after = (
            scores["baseline_success_rate"],
            scores["candidate_success_rate"],
        )
        count = scores["episodes_per_policy"]
        metrics[cohort.title() + " · " + name + " success"] = (
            f"Baseline {before:.2%} → candidate {after:.2%} ({count:,} routes each)"
        )

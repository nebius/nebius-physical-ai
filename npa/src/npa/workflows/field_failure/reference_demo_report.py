"""Publish an offline HTML report separating completed runtime from policy quality."""

import html
import json
from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _publish, _read, _storage


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
    report = result_summary(plan, selection, final)
    from npa.workflows.field_failure.reference_demo_media import preview_groups

    groups = preview_groups(args, final)
    _publish(args.output_root + "/reports/result.json", report)
    _storage().put_bytes_conditional(
        render_html(report, groups).encode(),
        args.output_root + "/reports/index.html",
        if_none_match=True,
        content_type="text/html; charset=utf-8",
    )
    return report


def result_summary(plan, selection, final):
    """Derive readiness only from actual development selection and final comparison.

    Args:
        plan: Frozen public reference cohort manifest and recipe schedule.
        selection: Measured development eligibility.
        final: Actual final comparison or None when development blocked it.
    Returns:
        Public summary with no credentials, storage locations or infrastructure names.
    Raises:
        KeyError: Required measured evidence is missing.
    """
    scores = None if final is None else final["metrics"]["success"]
    quality = bool(
        final and final["promote_checkpoint"] and scores["candidate_mean"] >= 0.8
    )
    return {
        "schema": "npa.field-failure.reference-result.v1",
        "runtime_completed": True,
        "quality_passed": quality,
        "final_evaluated": final is not None,
        "deployment_authorized": False,
        "recommendation": "promote" if quality else "retain_baseline",
        "development": {
            key: selection[key]
            for key in (
                "eligible",
                "baseline_success_rate",
                "candidate_success_rate",
                "success_rate_gain",
                "reasons",
                "final_cohort_consumed",
            )
        },
        "final": _final_summary(final, scores),
        "cohorts": plan["cohorts"],
        "robots": plan["num_envs"],
        "baseline_iterations": plan["baseline_iterations"],
        "candidate_iterations": plan["candidate_iterations"],
        "scope": "Public scan reconstruction with baseline warehouse replay; new warehouse routes in a known layout.",
        "limitations": [
            "Range observations; camera-conditioned navigation is not qualified.",
            "New-site transfer and private robot integration are not established.",
            "The earlier scan-to-unseen-warehouse failure and original observation gap remain historical evidence.",
        ],
    }


def _final_summary(final, scores):
    if scores is None:
        return None
    return {
        "baseline_success_rate": scores["baseline_mean"],
        "candidate_success_rate": scores["candidate_mean"],
        "success_rate_gain": scores["improvement"],
        "episodes_per_policy": final["episodes_per_policy"],
        "regressions": len(final["regressions"]),
        "comparison_recommendation": final["recommendation"],
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
        )
        contents = path.read_text()
    details = f"<details><summary>Measured results and frozen cohort identities</summary><pre>{data}</pre></details>"
    return contents.replace("</body>", details + "</body>")


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
    }
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
    return metrics

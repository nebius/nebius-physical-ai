"""JSON + Markdown report writers for the evaluation harness."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from npa.workflows.byof.artifact_io import (
    ArtifactPathError,
    join_output_path,
    write_bytes,
)
from npa.workbench.eval_harness.tasks import EvalHarnessError


def _write_report(
    output_path: str,
    report: Mapping[str, Any],
    *,
    json_name: str,
    markdown: str,
) -> dict[str, str]:
    payload = dict(report)
    payload["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        json_path = join_output_path(output_path, json_name)
        md_path = join_output_path(output_path, json_name.replace(".json", ".md"))
        write_bytes(
            json_path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            content_type="application/json",
        )
        write_bytes(
            md_path,
            (markdown + "\n").encode("utf-8"),
            content_type="text/markdown; charset=utf-8",
        )
    except ArtifactPathError as exc:
        raise EvalHarnessError(str(exc)) from exc
    return {"json": json_path, "markdown": md_path}


def write_run_report(output_path: str, report: Mapping[str, Any]) -> dict[str, str]:
    """Write ``report.json`` + ``report.md`` below *output_path*."""
    artifacts = _write_report(
        output_path,
        report,
        json_name="report.json",
        markdown=render_run_markdown(report),
    )
    return {
        "report_json": artifacts["json"],
        "report_markdown": artifacts["markdown"],
    }


def write_compare_report(output_path: str, report: Mapping[str, Any]) -> dict[str, str]:
    """Write ``compare.json`` + ``compare.md`` below *output_path*."""
    artifacts = _write_report(
        output_path,
        report,
        json_name="compare.json",
        markdown=render_compare_markdown(report),
    )
    return {
        "compare_json": artifacts["json"],
        "compare_markdown": artifacts["markdown"],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_run_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    wilson = summary.get("wilson_95", {})
    lines = [
        f"# eval_harness run: {report.get('task')}",
        "",
        f"- policy: `{report.get('policy')}`",
        f"- episodes: {report.get('episodes')} (seed {report.get('seed')})",
        f"- judge: `{report.get('judge')}`",
        "",
        "## Summary",
        "",
        f"- success rate: {_fmt(summary.get('success_rate'))} "
        f"({summary.get('successes')}/{summary.get('episodes')})",
        f"- Wilson 95% CI: [{_fmt(wilson.get('lo'))}, {_fmt(wilson.get('hi'))}]",
        f"- mean steps/episode: {_fmt(summary.get('mean_steps'))}",
        f"- median time-to-success: {_fmt(summary.get('median_time_to_success'))}",
        f"- mean total reward: {_fmt(summary.get('mean_total_reward'))}",
    ]
    return "\n".join(lines)


def render_compare_markdown(report: Mapping[str, Any]) -> str:
    ci = report.get("bootstrap_95", {})
    lines = [
        f"# eval_harness compare: {report.get('task')}",
        "",
        f"- policy A: `{report.get('policy_a')}` "
        f"(success rate {_fmt(report.get('summary_a', {}).get('success_rate'))})",
        f"- policy B: `{report.get('policy_b')}` "
        f"(success rate {_fmt(report.get('summary_b', {}).get('success_rate'))})",
        f"- episodes: {report.get('episodes')} (paired seeds from "
        f"{report.get('seed')})",
        f"- judge: `{report.get('judge')}`",
        "",
        "## Result",
        "",
        f"- success-rate difference (A − B): {_fmt(report.get('success_rate_diff'))}",
        f"- paired bootstrap 95% CI: [{_fmt(ci.get('lo'))}, {_fmt(ci.get('hi'))}]",
    ]
    return "\n".join(lines)


__all__ = [
    "render_compare_markdown",
    "render_run_markdown",
    "write_compare_report",
    "write_run_report",
]

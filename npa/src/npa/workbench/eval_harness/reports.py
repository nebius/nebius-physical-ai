"""JSON + Markdown report writers for the evaluation harness.

``output_uri`` is a directory: ``file://`` URIs and plain local paths are
supported (mirroring the pipeline URI conventions); anything else raises an
informative error instead of silently writing somewhere unexpected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from npa.workbench.eval_harness.tasks import EvalHarnessError


def _output_dir(output_uri: str) -> Path:
    if not output_uri:
        raise EvalHarnessError("output_uri is required")
    parsed = urlparse(output_uri)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme == "file" else output_uri)
    else:
        raise EvalHarnessError(
            f"unsupported output_uri scheme {parsed.scheme!r} in {output_uri!r}; "
            "eval_harness writes to file:// URIs and local paths"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def write_run_report(
    output_uri: str, report: Mapping[str, Any]
) -> dict[str, str]:
    """Write ``report.json`` + ``report.md`` under *output_uri*."""
    directory = _output_dir(output_uri)
    payload = dict(report)
    payload["written_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    json_path = directory / "report.json"
    md_path = directory / "report.md"
    _write_text(
        json_path, json.dumps(payload, indent=2, sort_keys=True)
    )
    _write_text(md_path, render_run_markdown(payload))
    return {"report_json": str(json_path), "report_markdown": str(md_path)}


def write_compare_report(
    output_uri: str, report: Mapping[str, Any]
) -> dict[str, str]:
    """Write ``compare.json`` + ``compare.md`` under *output_uri*."""
    directory = _output_dir(output_uri)
    payload = dict(report)
    payload["written_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    json_path = directory / "compare.json"
    md_path = directory / "compare.md"
    _write_text(
        json_path, json.dumps(payload, indent=2, sort_keys=True)
    )
    _write_text(md_path, render_compare_markdown(payload))
    return {"compare_json": str(json_path), "compare_markdown": str(md_path)}


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
        f"- success-rate difference (A − B): "
        f"{_fmt(report.get('success_rate_diff'))}",
        f"- paired bootstrap 95% CI: [{_fmt(ci.get('lo'))}, {_fmt(ci.get('hi'))}]",
    ]
    return "\n".join(lines)


__all__ = [
    "render_compare_markdown",
    "render_run_markdown",
    "write_compare_report",
    "write_run_report",
]

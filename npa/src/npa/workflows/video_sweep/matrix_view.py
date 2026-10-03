"""Render the configured parameter fanout with planned or recorded outcomes."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from npa.workflows.video_sweep import prompt_view

_STYLE = """
<style>
.matrix-panel {margin:28px 0;padding:24px;background:#11212c;border:1px solid #30424e;border-radius:12px;color:#e7eef1;font:14px/1.6 system-ui,sans-serif;min-width:0}
.matrix-panel h2 {margin:0 0 8px;font-size:22px}.matrix-panel p {color:#b9c7cf}
.matrix-axes {display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.matrix-axis {border:1px solid #506270;border-radius:8px;padding:12px}.matrix-axis b {display:block;color:#cdf66e}
.matrix-table {overflow:auto;max-height:460px}.matrix-panel table {border-collapse:collapse;width:100%;white-space:nowrap;text-align:left}.matrix-panel th,.matrix-panel td {padding:10px 14px;border-bottom:1px solid #30424e}
.matrix-panel button {cursor:pointer;background:#cdf66e;color:#101914;border:0;border-radius:4px;padding:6px 10px}.matrix-fixed {overflow-wrap:anywhere}
.prompt-flow {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(230px,100%),1fr));gap:12px;margin:16px 0}.prompt-stage {padding:14px;border:1px solid #506270;border-radius:8px;min-width:0}.prompt-stage b {color:#cdf66e}.prompt-stage p {margin:6px 0 0}.prompt-groups {padding-left:20px;overflow-wrap:anywhere}
</style>
"""


def render(summary: dict | None, candidates: list[dict] | None = None) -> str:
    """Render a public-safe fanout table with optional links to actual clips.

    Args:
        summary: Sanitized matrix description, or None for historical runs.
        candidates: Ordered reviewed clips; absent for a planning-only preview.
    Returns:
        Escaped HTML fragment with every combination and worker assignment.
    Raises:
        ValueError: Recorded clips do not cover the planned matrix.
    """
    if summary is None:
        return ""
    if candidates is not None and len(candidates) != len(summary["jobs"]):
        raise ValueError("Recorded clips do not cover the parameter matrix")
    axes = summary["axes"]
    factors = " × ".join(f"{len(values)} {escape(key)}" for key, values in axes.items())
    equation = f"{summary['sources']} source(s) × ({factors}) = {len(summary['jobs'])} candidates"
    chips = _axis_chips(axes)
    heads = ["Candidate", "Source", "Prompt", *axes, "GPU worker", "Result"]
    header = "".join(f"<th>{escape(key)}</th>" for key in heads)
    rows = "".join(_row(job, candidates) for job in summary["jobs"])
    status = (
        "Recorded generation results"
        if candidates is not None
        else "Plan only · no compute submitted · no generated clips"
    )
    return (
        _STYLE
        + f"""<section class="matrix-panel" id="matrix-panel" aria-label="Parameter matrix">
<h2>Configuration → parameter fanout</h2><p>{status}</p>
{prompt_view.render(summary)}
<strong id="matrix-equation">{equation}</strong><div class="matrix-axes">{chips}</div>
<p class="matrix-fixed">Fixed controls: {escape(json.dumps(summary["base"], sort_keys=True))}. Prompt text and source locations are private.</p>
<p>{summary["workers"]} GPU workers process disjoint candidate partitions; each worker processes its assigned candidates sequentially.</p>
<div class="matrix-table"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div></section>"""
    )


def _row(job, candidates):
    values = [
        job["candidate"],
        job["source"],
        job.get("prompt_group", "—"),
        *job["parameters"].values(),
        job["worker"],
    ]
    columns = "".join(f"<td>{escape(str(value))}</td>" for value in values)
    outcome = "Planned"
    if candidates is not None:
        index = job["candidate"] - 1
        row = candidates[index]
        decision = "Accepted" if row["accepted"] else "Held out"
        outcome = f'<button data-matrix-candidate="{index}">{decision} · {float(row["score"]):.2f} · Play</button>'
    return f"<tr>{columns}<td>{outcome}</td></tr>"


def _axis_chips(axes):
    return "".join(
        f'<div class="matrix-axis"><b>{escape(key)}</b>{escape(json.dumps(values))}</div>'
        for key, values in axes.items()
    )


def write_preview(output: Path, summary: dict) -> None:
    """Write a standalone matrix preview without credentials or provider calls.

    Args:
        output: New private output directory.
        summary: Sanitized matrix description.
    Returns:
        None.
    Raises:
        OSError: The directory exists or writing fails.
    """
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cosmos3 parameter fanout</title><body style="background:#09121a;margin:24px">'
    (output / "index.html").write_text(page + render(summary) + "</body></html>")
    (output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in output.iterdir():
        path.chmod(0o600)

"""Rerun visualization of an adversarial scenario set.

Emits a ``.rrd`` recording next to the JSON manifest so the mined scenarios can
be inspected visually in the Rerun viewer (``npa rerun host <uri>`` prints an
app.rerun.io URL; the NPA agent classifies ``.rrd`` as a rerun artifact).

The recording contains, over a ``rank`` timeline (scenarios ordered by
severity):

* time-series of ``severity`` / ``diversity`` / ``failure_score`` per rank,
* a severity bar chart, a severity-vs-diversity scatter, and a
  scenario x perturbation-axis heatmap tensor,
* a markdown summary table.

``rerun-sdk`` is imported lazily; when it is unavailable the caller degrades
gracefully (no ``.rrd``, empty ``viz_uri``) instead of failing generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .schemas import ADVERSARIAL_SET_SCHEMA, ScenarioRecord
from .storage import uri_join, write_bytes_uri

APPLICATION_ID = "npa_scenario_gen"


def viz_uri(output_uri: str) -> str:
    return uri_join(output_uri, "scenarios.rrd")


def _import_rerun() -> Any:
    import rerun as rr  # lazy: optional heavy viewer dependency

    return rr


def _set_rank(rr: Any, recording: Any, index: int) -> None:
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("rank", sequence=index, recording=recording)
            return
        except TypeError:
            pass
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("rank", index, recording=recording)


def render_adversarial_rrd(
    records: list[ScenarioRecord],
    *,
    output_uri: str,
    task_name: str,
    run_id: str,
) -> str:
    """Write an adversarial-set ``.rrd`` to ``{output_uri}/scenarios.rrd``.

    Returns the artifact URI, or ``""`` when ``rerun-sdk`` is not installed.
    """
    try:
        rr = _import_rerun()
    except ImportError:
        return ""
    if not records:
        return ""

    import numpy as np

    target = viz_uri(output_uri)
    with tempfile.TemporaryDirectory(prefix="npa-scenario-gen-rrd-") as tmp:
        local = Path(tmp) / "scenarios.rrd"
        recording = rr.RecordingStream(APPLICATION_ID, recording_id=run_id or "scenario-gen")
        recording.save(str(local))

        axes = sorted({axis for record in records for axis in record.perturbation})

        # Summary markdown table (static).
        header = "| rank | scenario | severity | diversity | failure |\n| --- | --- | --- | --- | --- |\n"
        rows = "".join(
            f"| {i} | {r.scenario_id} | {r.severity:.4f} | {r.diversity:.4f} | {r.failure_score:.4f} |\n"
            for i, r in enumerate(records, start=1)
        )
        summary = (
            f"# Adversarial scenario set\n\n"
            f"- task: `{task_name}`\n- run: `{run_id}`\n"
            f"- schema: `{ADVERSARIAL_SET_SCHEMA}`\n- scenarios: {len(records)}\n\n"
            f"{header}{rows}"
        )
        if hasattr(rr, "TextDocument"):
            rr.log("summary/README", rr.TextDocument(summary, media_type="text/markdown"), static=True, recording=recording)

        # Severity bar chart + severity-vs-diversity scatter (static).
        severities = np.array([r.severity for r in records], dtype=np.float32)
        if hasattr(rr, "BarChart"):
            rr.log("summary/severity_ranked", rr.BarChart(severities), static=True, recording=recording)
        if hasattr(rr, "Points2D"):
            positions = np.array([[r.diversity, r.severity] for r in records], dtype=np.float32)
            rr.log(
                "summary/severity_vs_diversity",
                rr.Points2D(
                    positions,
                    radii=0.01,
                    labels=[r.scenario_id for r in records],
                ),
                static=True,
                recording=recording,
            )
        # Scenario x axis perturbation heatmap.
        if axes and hasattr(rr, "Tensor"):
            matrix = np.array(
                [[float(r.perturbation.get(axis, 0.0)) for axis in axes] for r in records],
                dtype=np.float32,
            )
            tensor_kwargs: dict[str, Any] = {}
            try:
                rr.log(
                    "summary/perturbations",
                    rr.Tensor(matrix, dim_names=["scenario", "axis"]),
                    static=True,
                    recording=recording,
                )
            except TypeError:
                rr.log("summary/perturbations", rr.Tensor(matrix, **tensor_kwargs), static=True, recording=recording)

        # Per-rank time series.
        for index, record in enumerate(records, start=1):
            _set_rank(rr, recording, index)
            if hasattr(rr, "Scalars"):
                rr.log("metrics/severity", rr.Scalars(float(record.severity)), recording=recording)
                rr.log("metrics/diversity", rr.Scalars(float(record.diversity)), recording=recording)
                rr.log("metrics/failure_score", rr.Scalars(float(record.failure_score)), recording=recording)

        flush = getattr(recording, "flush", None)
        if callable(flush):
            try:
                flush(blocking=True)
            except TypeError:
                flush()
        data = local.read_bytes()

    write_bytes_uri(target, data)
    return target


__all__ = ["APPLICATION_ID", "render_adversarial_rrd", "viz_uri"]

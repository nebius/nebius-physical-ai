"""Contract: Runs & artifacts are consolidated and sorted latest-first."""

from __future__ import annotations

from pathlib import Path

from npa.cli.agent import AGENT_UI_VERSION

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"


def _embedded_ui_html(source: str = "") -> str:
    """Return rendered agent UI HTML (sourced from agent_ui.html)."""
    from npa.cli.agent import rendered_agent_ui_html

    return rendered_agent_ui_html()



def test_ui_consolidates_active_run_and_artifacts() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    assert 'id="runsArtifactsPanel"' in ui
    assert "Runs &amp; artifacts" in ui
    assert "latest first" in ui
    assert "mergeRunsLatestFirst" in ui
    assert "applyMergedRunSelectors" in ui
    # Old split subsections / duplicate run picker must be gone.
    assert "<h4>Active run</h4>" not in ui
    assert "<h4>Artifacts</h4>" not in ui
    assert 'id="artifactRunSelect"' not in ui
    assert ui.count('id="runIdSelect"') == 1


def test_available_run_ids_use_latest_first_helper() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    status = source.split('@app.get("/sim-viz/status")')[1].split('@app.get("/sim-viz/runs")')[0]
    assert "available_run_ids" in status
    assert "available_runs" in status
    assert "_sim_viz_runs(state)" in status
    # Alphabetical sort of run keys is the old bug.
    assert "sorted(str(key) for key in runs.keys()" not in status
    load_fn = source.split("def _sim_viz_load_response")[1].split("@app.post")[0]
    assert "_sim_viz_runs(state)" in load_fn
    assert "sorted(str(key) for key in runs.keys()" not in load_fn

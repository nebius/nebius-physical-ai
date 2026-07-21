"""Stages panel must let operators pick and load a run without leaving Chat."""

from __future__ import annotations

from pathlib import Path

from npa.cli.agent import AGENT_STAGES_RUN_PICKER_CONTRACT, AGENT_UI_VERSION

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"


def _embedded_ui_html(source: str = "") -> str:
    """Return rendered agent UI HTML (sourced from agent_ui.html)."""
    from npa.cli.agent import rendered_agent_ui_html

    return rendered_agent_ui_html()



def test_stages_panel_has_run_picker_and_load() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_STAGES_RUN_PICKER_CONTRACT:
        assert marker in ui, marker
    # Picker lives in the Stages panel (Chat layout), not only the Rerun rail.
    stages = ui.split('id="stagesPanel"')[1].split('id="panelRerun"')[0]
    assert 'id="stagesRunSelect"' in stages
    assert 'id="stagesLoadRun"' in stages
    assert "Search or paste run ID" in stages
    assert "filterStagesRunSelect" in ui
    assert "resolveStagesRunChoice" in ui
    assert "mergedRunsCache" in ui
    assert "loadSelectedRun" in ui
    assert "updateRunSelector" in ui
    assert "fillRunSelectOptionsRich(document.getElementById(\"stagesRunSelect\")" in ui
    assert "mergeRunsLatestFirst" in ui
    assert "applyMergedRunSelectors" in ui


def test_stages_and_rerun_selectors_share_load_path() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert "loadSelectedRun(chosen)" in ui
    # Selecting either dropdown loads the run (not input-only sync).
    assert 'getElementById("stagesRunSelect")' in ui
    assert 'getElementById("runIdSelect")' in ui
    load_fn = ui.split("async function loadSelectedRun")[1].split("function normalizeStageStatus")[0]
    assert "loadRunData()" in load_fn
    assert "syncRunChooserFields" in load_fn


def test_artifact_backed_stages_skip_unrelated_draft_overlay() -> None:
    """Historical capture runs must not inherit an unrelated workflow draft as pending."""
    source = AGENT_MODULE.read_text(encoding="utf-8")
    stages_mod = (AGENT_MODULE.parent / "agent_stages.py").read_text(encoding="utf-8")
    assert "def run_owns_workflow_stage_overlay" in stages_mod
    assert "def build_artifact_backed_stages" in stages_mod
    assert "_AGENT_STAGES_EMBED" in source
    assert "overlay_unmatched = run_owns_workflow_stage_overlay(state, run_id)" in source
    assert "build_artifact_backed_stages(" in source
    assert "Historical capture runs must not inherit" in stages_mod

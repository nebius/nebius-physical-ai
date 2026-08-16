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
    assert "Search NPA workflow/artifact runs" in stages
    assert "Codex maintenance job IDs" in stages
    assert 'id="stagesRunSearchResult"' in stages
    assert "filterStagesRunSelect" in ui
    assert "resolveStagesRunChoice" in ui
    assert "mergedRunsCache" in ui
    assert "loadSelectedRun" in ui
    assert "updateRunSelector" in ui
    assert 'fillRunSelectOptionsRich(document.getElementById("stagesRunSelect")' in ui
    assert "mergeRunsLatestFirst" in ui
    assert "applyMergedRunSelectors" in ui


def test_stages_and_rerun_selectors_share_load_path() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert "loadSelectedRun(chosen)" in ui
    # Selecting either dropdown loads the run (not input-only sync).
    assert 'getElementById("stagesRunSelect")' in ui
    assert 'getElementById("runIdSelect")' in ui
    load_fn = ui.split("async function loadSelectedRun")[1].split(
        "function normalizeStageStatus"
    )[0]
    assert "syncRunChooserFields" in load_fn
    # The shared chooser keeps provenance and dispatches each source to its own
    # compatible load path instead of blindly POSTing every run to load-run.
    assert 'entry.source_type === "local_demo"' in load_fn
    assert 'entry.source_type === "artifact_storage"' in load_fn
    assert (
        "loadArtifactsForSelectedRun(chosen, null, entry, { pendingSelection: true })"
        in load_fn
    )
    assert "loadWorkflowHistoryRun(chosen, activeArtifactRunRef)" in load_fn


def test_failed_exact_search_is_separate_from_currently_loaded_run() -> None:
    ui = _embedded_ui_html()

    assert "currently loaded run <strong>" in ui
    assert "Currently loaded run remains" in ui
    assert "until lookup succeeds" in ui
    assert "Exact NPA run lookup failed" in ui
    assert 'clearVisibleRunState("Search changed.' not in ui


def test_selected_run_capability_is_installed_before_rerun_mount() -> None:
    """A newly selected recording must not mount through the Basic-Auth blob fallback."""
    ui = _embedded_ui_html()
    assert "function syncRerunRecordingCapability(simViz)" in ui

    load_run = ui.split("async function loadWorkflowHistoryRun")[1].split(
        "async function selectCamera"
    )[0]
    assert load_run.index(
        "syncRerunRecordingCapability(data && data.sim_viz)"
    ) < load_run.index("bestEffortMountRerun")

    load_artifact = ui.split("async function loadArtifact(payload)")[1].split(
        "async function loadVoxelDataset"
    )[0]
    assert load_artifact.index(
        "syncRerunRecordingCapability(simViz)"
    ) < load_artifact.index("swapRerunRecordingInPlace")


def test_artifact_run_load_is_independent_from_rerun_preview() -> None:
    """Loading an artifact-backed training run must not require a Rerun recording."""
    ui = _embedded_ui_html()
    assert 'id="artifactRoleFilter"' in ui

    load_fn = ui.split("async function loadRunData")[1].split(
        "async function selectCamera"
    )[0]
    assert "await loadArtifactsForSelectedRun(runRef || runId" in load_fn
    assert "deferPreferredViewer: true" in load_fn
    assert "No RRD/MCAP recording; use the artifacts below" in load_fn


def test_artifact_backed_stages_skip_unrelated_draft_overlay() -> None:
    """Historical capture runs must not inherit an unrelated workflow draft as pending."""
    source = AGENT_MODULE.read_text(encoding="utf-8")
    stages_mod = (AGENT_MODULE.parent / "agent_stages.py").read_text(encoding="utf-8")
    runtime_mod = (AGENT_MODULE.parent / "agent_stage_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "def run_owns_workflow_stage_overlay" in stages_mod
    assert "def build_artifact_backed_stages" in stages_mod
    assert "_AGENT_STAGES_EMBED" in source
    assert (
        "overlay_unmatched=run_owns_workflow_stage_overlay(state, run_id)"
        in runtime_mod
    )
    assert "build_artifact_backed_stages(" in runtime_mod
    assert "Historical capture runs must not inherit" in stages_mod

"""Focused contracts for optional LeIsaac UI and smooth run hydration."""

from __future__ import annotations

from npa.cli.agent import rendered_agent_ui_html


def _ui() -> str:
    return rendered_agent_ui_html()


def test_leisaac_ui_uses_only_server_rendered_configuration() -> None:
    ui = _ui()
    start = ui.split("function startApp()", 1)[1].split(
        'if (document.readyState === "loading")', 1
    )[0]
    boot = ui.split("async function bootPage()", 1)[1].split(
        "function startPeriodicRefresh", 1
    )[0]

    assert 'id="enableLeIsaac"' not in ui
    assert "const LEISAAC_UI_ENABLED = false;" in ui
    assert "LEISAAC_UI_STORAGE_KEY" not in ui
    assert "if (leisaacUiEnabled())" in start
    assert "ensureLeIsaacTab(leisaacCapability);" in start
    assert "if (leisaacUiEnabled()) refreshLeIsaacCapability()" in boot
    assert 'id = "disableLeIsaac"' not in ui


def test_boot_reuses_session_and_status_without_blocking_on_run_details() -> None:
    ui = _ui()
    boot = ui.split("async function bootPage()", 1)[1].split(
        "function startPeriodicRefresh", 1
    )[0]
    refresh = ui.split("async function refresh(options)", 1)[1].split(
        "function selectedCamera", 1
    )[0]

    assert "restoredSession = await restoreSession()" in boot
    assert "refresh({ session: restoredSession })" in boot
    assert "ensureFrankaRerunLoaded(lastSimVizStatus)" in boot
    assert "opts.session ? Promise.resolve(opts.session)" in refresh
    assert "opts.simViz ? Promise.resolve(opts.simViz)" in refresh
    assert "void loadRunDetails(activeRunId).catch" in refresh
    assert "await loadRunDetails(activeRunId)" not in refresh


def test_artifact_filters_reuse_any_source_qualified_inventory() -> None:
    ui = _ui()
    wiring = ui.split(
        'for (const id of ["artifactStageFilter", "artifactTypeFilter", "artifactRoleFilter", "artifactSort"])',
        1,
    )[1].split('// "Find run (name or ID)" box', 1)[0]
    loader = ui.split("async function loadArtifactsForSelectedRun", 1)[1].split(
        "async function loadExactArtifactSource", 1
    )[0]

    assert "reuseInventory: true" in wiring
    assert "const cachedInventoryMatches = activeArtifactInventoryPage" in loader
    assert 'String(activeArtifactInventoryPage.run_ref || "") === runRef' in loader
    assert "const cachedInventory = cachedInventoryMatches" in loader
    assert "context.reuseInventory || context.completeInventory" in loader
    assert "if (!data)" in loader
    assert "activeArtifactInventoryPage = data" in loader
    assert "activeArtifactInventoryComplete = data.pagination_complete === true" in loader


def test_default_inventory_is_lazy_and_list_artifacts_resumes_cached_cursor() -> None:
    ui = _ui()
    wiring = ui.split(
        'bindClick("artifactLoadRunArtifacts"', 1
    )[1].split('bindClick("loadRerunViewer"', 1)[0]
    selector = ui.split("async function _loadSelectedRun", 1)[1].split(
        "function normalizeStageStatus", 1
    )[0]
    loader = ui.split("async function loadArtifactsForSelectedRun", 1)[1].split(
        "async function loadExactArtifactSource", 1
    )[0]

    assert "completeSelectedArtifactInventory" in wiring
    complete_inventory = ui.split("function completeSelectedArtifactInventory", 1)[1].split(
        "async function loadArtifactsForSelectedRun", 1
    )[0]
    assert "completeInventory: true" in complete_inventory
    assert "deferInventoryCompletion: true" in selector
    assert "seededPage = cachedInventory" in loader
    assert 'params.set("cursor", String(seededPage.next_cursor))' in loader
    assert "no_recording: inventoryComplete && !hasRecording" in loader
    assert "has_recording: inventoryComplete ? hasRecording : null" in loader
    assert "if (summary.has_recording === false)" in ui


def test_superseded_direct_inventory_does_not_fall_back_and_clear_selection() -> None:
    ui = _ui()
    loader = ui.split("async function loadRunData", 1)[1].split(
        "async function selectCamera", 1
    )[0]

    assert "if (loaded === false)" in loader
    assert "artifacts_loaded: false, superseded: true" in loader
    assert loader.index("if (loaded === false)") < loader.index("const dataPromise = loadSelectedRun")


def test_newer_run_selection_supersedes_stale_responses() -> None:
    ui = _ui()
    selector = ui.split("let selectedRunLoadGeneration", 1)[1].split(
        "function normalizeStageStatus", 1
    )[0]
    history = ui.split("async function loadWorkflowHistoryRun", 1)[1].split(
        "async function loadRunData", 1
    )[0]

    assert "const generation = ++selectedRunLoadGeneration" in selector
    assert "activeRunSelectionGeneration += 1" in selector
    assert "generation === selectedRunLoadGeneration" in selector
    assert "isCurrent," in selector
    assert "if (!isCurrent()) return null" in history
    assert "if (leisaacUiEnabled()) await refreshLeIsaacCapability()" in history


def test_superseded_local_demo_cannot_reach_final_refresh() -> None:
    ui = _ui()
    selector = ui.split("async function _loadSelectedRun", 1)[1].split(
        "function normalizeStageStatus", 1
    )[0]
    local_demo = selector.split('if (entry.source_type === "local_demo")', 1)[1].split(
        "return loadWorkflowHistoryRun", 1
    )[0]

    final_guard = local_demo.rindex("if (!isCurrent()) return null;")
    details_guard = local_demo.index("if (!isCurrent()) return null;", local_demo.index("await loadRunDetails"))
    assert details_guard < local_demo.index("await bestEffortMountRerun")
    assert final_guard > local_demo.index("await bestEffortMountRerun")
    assert final_guard < local_demo.index("await refresh();")


def test_foxglove_popup_uses_stable_run_boundary_not_mutable_run_ref() -> None:
    ui = _ui()
    handler = ui.split("async function openFoxgloveWeb", 1)[1].split(
        "async function captureFoxgloveContext", 1
    )[0]
    stale_guard = handler.split("const selectedResult", 1)[1].split(
        "const info = data && data.export", 1
    )[0]

    assert 'String(activeRunId || "").trim() !== runId' in stale_guard
    assert "activeArtifactRunRef" not in stale_guard
    assert "foxgloveArtifactIdentityMatches(selectedResult, selected)" in handler

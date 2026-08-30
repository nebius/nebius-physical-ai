"""Contract: Runs & artifacts are consolidated and sorted latest-first."""

from __future__ import annotations

from pathlib import Path

from npa.cli.agent import AGENT_UI_VERSION

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
AGENT_STAGES_MODULE = AGENT_MODULE.with_name("agent_stages.py")


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


def test_available_runs_expose_viewer_activity_not_as_recency() -> None:
    """Regression: opening a days-old run in the viewer must not relabel it as recent.

    The per-run viewer-load time (``rrd_updated_at``) is exposed as ``activity_at``
    and must NOT be surfaced as ``last_modified`` — otherwise ``mergeRunsLatestFirst``
    (client) would take the max and float a run that was merely opened today to the
    top, labeled with today's date even though it ran days ago. The run's start
    time is exposed separately as ``started_at``.
    """
    source = AGENT_STAGES_MODULE.read_text(encoding="utf-8")
    available = source.split("def build_available_sim_viz_runs")[1].split(
        "def local_demo_run_details"
    )[0]
    # Viewer-load time is exposed under activity_at, start under started_at,
    # and last_modified is blank so S3 discovery owns the displayed recency.
    assert '"activity_at": str(' in available
    assert '"started_at": str(item.get("submitted_at")' in available
    assert '"last_modified": ""' in available
    # The old bug: rrd_updated_at collapsed into last_modified.
    assert '"last_modified": str(' not in available


def test_client_merge_dates_runs_by_start_not_recency_or_activity() -> None:
    """The merged run list must date/sort by run start, then recency, then activity."""
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert "function effectiveRunTs(" in ui
    # Effective timestamp prefers run start, then artifact recency, then activity.
    assert "run.started_at || run.last_modified || run.activity_at" in ui
    merge_fn = ui.split("function mergeRunsLatestFirst")[1].split("function fillRunSelectOptionsRich")[0]
    # Start is kept as the earliest across sources; activity/recency stay separate.
    assert "prev.started_at" in merge_fn
    assert "startTs < prev.started_at" in merge_fn
    assert "prev.activity_at" in merge_fn
    assert "prev.last_modified = artifactTs" in merge_fn
    assert "effectiveRunTs(b).localeCompare(effectiveRunTs(a))" in merge_fn
    # Known/available runs carry started_at + activity_at through to the merge.
    assert "started_at: String((item && (item.started_at || item.submitted_at))" in ui
    assert "activity_at: String((item && (item.activity_at || item.rrd_updated_at))" in ui


def test_client_consumes_incomplete_viewability_without_fabricating_false() -> None:
    ui = _embedded_ui_html()
    merge_fn = ui.split("function mergeRunsLatestFirst")[1].split(
        "function fillRunSelectOptionsRich"
    )[0]
    picker_fn = ui.split("function fillRunSelectOptionsRich")[1].split(
        "function runEntryFromSelect"
    )[0]

    assert "has_viewable: null" in merge_fn
    assert "summary_complete: null" in merge_fn
    assert "run.has_viewable === true" in merge_fn
    assert "run.summary_complete === false" in merge_fn
    assert 'bits.push("viewable")' in picker_fn
    assert 'bits.push("viewability unknown")' in picker_fn


def test_client_omits_stale_stock_demo_alias_for_artifact_run() -> None:
    """A stock viewer state relabeled with a real run id is not another source."""
    ui = _embedded_ui_html()
    merge_fn = ui.split("function mergeRunsLatestFirst")[1].split(
        "function fillRunSelectOptionsRich"
    )[0]

    assert 'const staleDemoAlias = runId !== "franka-demo"' in merge_fn
    assert 'String((run && run.stage) || "").trim().toLowerCase() === "demo"' in merge_fn
    assert '!String((run && run.run_ref) || "").trim()' in merge_fn
    assert "if (staleDemoAlias) continue;" in merge_fn


def test_client_reuses_discovered_identity_for_bound_viewer_history() -> None:
    """A history row bound to one of two S3 sources must not become a third row."""
    ui = _embedded_ui_html()
    merge_fn = ui.split("function mergeRunsLatestFirst")[1].split(
        "function fillRunSelectOptionsRich"
    )[0]

    assert "const knownRunRef" in merge_fn
    assert "const matchingRunRef = knownRunRef && matches.some" in merge_fn
    assert "const matchedSource = matchingRunRef && matches.find" in merge_fn
    assert "const key = matchedSource" in merge_fn
    assert "? String(matchedSource.run_ref || runId)" in merge_fn


def test_client_consolidates_aliases_only_within_project_bucket_run_boundary() -> None:
    ui = _embedded_ui_html()
    consolidate = ui.split("function consolidateArtifactRunSources", 1)[1].split(
        "function mergeRunsLatestFirst", 1
    )[0]
    group_identity = ui.split("function artifactSourceGroupIdentity", 1)[1].split(
        "function artifactSourceStableIdentity", 1
    )[0]

    assert "run.project_id" in group_identity
    assert "run.bucket" in group_identity
    assert "run.run_id" in group_identity
    assert "resolved_prefix" not in group_identity
    assert "const key = artifactSourceGroupIdentity(run)" in consolidate
    assert "primary.alternate_sources" in consolidate
    assert "source_selected: true" in consolidate


def test_client_primary_source_preference_is_deterministic_and_active_first() -> None:
    ui = _embedded_ui_html()
    preference = ui.split("function compareArtifactSourcePreference", 1)[1].split(
        "function consolidateArtifactRunSources", 1
    )[0]

    ordered_markers = [
        "activeScore(b) - activeScore(a)",
        'booleanScore(b, "has_viewable")',
        'booleanScore(b, "summary_complete")',
        'numericScore(b, "artifact_count")',
        "b && b.last_modified",
        "artifactSourceStableIdentity(a).localeCompare",
    ]
    positions = [preference.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert "activeArtifactRunRef" in preference
    assert "activeRunId" in preference


def test_pasted_basename_resolves_to_consolidated_exact_primary() -> None:
    ui = _embedded_ui_html()
    resolver = ui.split("function resolveStagesRunChoice", 1)[1].split(
        "function resolveRunSelection", 1
    )[0]
    load_run = ui.split("async function loadRunData", 1)[1].split(
        "async function selectCamera", 1
    )[0]

    assert "exact.length === 1" in resolver
    assert "exact[0].run_ref || exact[0].run_id" in resolver
    assert "preferredRunEntry(runRef || runId)" in load_run
    assert "loadArtifactsForSelectedRun(runRef || runId, null, exactEntry" in load_run

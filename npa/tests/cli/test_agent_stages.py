"""Behavioral unit tests for artifact-backed Stages overlay helpers."""

from __future__ import annotations

from npa.cli.agent_stages import (
    artifact_stage_key,
    artifact_stage_label,
    build_artifact_backed_stages,
    run_owns_workflow_stage_overlay,
)


def test_run_owns_overlay_false_for_unrelated_capture_run() -> None:
    state = {
        "latest_submit": {"run_id": "agent-run-active"},
        "workflow_draft": {
            "name": "sim2real-vlm-rl",
            "plan": {"run_id": "draft-vlm-rl-loop", "steps": [{"state": "augment"}]},
        },
        "sim2real_runs": {},
    }
    assert run_owns_workflow_stage_overlay(state, "franka-topdown-sim-20260709t031107z") is False


def test_run_owns_overlay_true_for_latest_submit() -> None:
    state = {"latest_submit": {"run_id": "agent-run-active"}, "workflow_draft": {}}
    assert run_owns_workflow_stage_overlay(state, "agent-run-active") is True


def test_run_owns_overlay_true_for_tracked_sim2real_submit() -> None:
    state = {
        "latest_submit": {},
        "sim2real_runs": {
            "agent-run-tracked": {"submitted_at": "2026-07-19T00:00:00Z", "status": "submitted"}
        },
        "workflow_draft": {},
    }
    assert run_owns_workflow_stage_overlay(state, "agent-run-tracked") is True


def test_run_owns_overlay_true_for_draft_plan_run_id() -> None:
    state = {
        "latest_submit": {},
        "workflow_draft": {"name": "sim2real-vlm-rl", "plan": {"run_id": "draft-vlm-rl-loop"}},
    }
    assert run_owns_workflow_stage_overlay(state, "draft-vlm-rl-loop") is True


def test_build_stages_skips_unmatched_draft_when_not_owned() -> None:
    keys = [
        "checkpoints/sim2real-b/franka-topdown/isaac-capture/frame_000.png",
        "checkpoints/sim2real-b/franka-topdown/reports/demo.mp4",
    ]
    workflow_defs = [
        ("augment", "augment", ["augment"]),
        ("envgen", "envgen", ["envgen", "envs"]),
        ("rollouts", "rollouts", ["rollouts", "actions"]),
    ]
    stages = build_artifact_backed_stages(
        keys,
        run_id="franka-topdown",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=workflow_defs,
        overlay_unmatched=False,
    )
    ids = [s["id"] for s in stages]
    assert "augment" not in ids
    assert "envgen" not in ids
    assert "rollouts" not in ids
    assert all(s["status"] == "succeeded" for s in stages)
    assert "isaac-capture" in ids
    assert "reports" in ids


def test_build_stages_keeps_pending_draft_when_owned() -> None:
    keys = [
        "checkpoints/sim2real-b/agent-run-active/isaac-capture/frame_000.png",
    ]
    workflow_defs = [
        ("augment", "augment", ["augment"]),
        ("envgen", "envgen", ["envgen"]),
    ]
    stages = build_artifact_backed_stages(
        keys,
        run_id="agent-run-active",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=workflow_defs,
        overlay_unmatched=True,
    )
    by_id = {s["id"]: s for s in stages}
    assert by_id["augment"]["status"] == "pending"
    assert by_id["envgen"]["status"] == "pending"
    assert by_id["isaac-capture"]["status"] == "succeeded"


def test_build_stages_marks_matched_draft_state_succeeded() -> None:
    keys = [
        "checkpoints/sim2real-b/run-1/augment/manifest.json",
        "checkpoints/sim2real-b/run-1/envs/raw/shard.json",
    ]
    workflow_defs = [
        ("augment", "augment", ["augment"]),
        ("envgen", "envgen", ["envgen", "envs"]),
        ("rollouts", "rollouts", ["rollouts", "actions"]),
    ]
    stages = build_artifact_backed_stages(
        keys,
        run_id="run-1",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=workflow_defs,
        overlay_unmatched=True,
    )
    by_id = {s["id"]: s for s in stages}
    assert by_id["augment"]["status"] == "succeeded"
    assert by_id["envgen"]["status"] == "succeeded"
    assert by_id["rollouts"]["status"] == "pending"


def test_build_stages_emit_stage_key_for_clickable_timeline() -> None:
    # Artifact-grouped stages (paidf-style, no draft) must carry stage_key so the
    # agent timeline rows are clickable and scope the artifact browser.
    keys = [
        "checkpoints/physical-ai-data-factory/run-1/cosmos_augmented/aug-run-1/frame-00000.png",
        "checkpoints/physical-ai-data-factory/run-1/curation/report.json",
    ]
    stages = build_artifact_backed_stages(
        keys,
        run_id="run-1",
        prefix="checkpoints/physical-ai-data-factory",
        workflow_stage_defs=[],
        overlay_unmatched=False,
    )
    by_key = {s["stage_key"]: s for s in stages}
    assert "cosmos_augmented" in by_key and "curation" in by_key
    assert all(s.get("stage_key") for s in stages)


def test_artifact_stage_key_and_label() -> None:
    key = "checkpoints/sim2real-b/run-1/isaac-capture/frame_001.png"
    assert artifact_stage_key(key, "run-1", "checkpoints/sim2real-b") == "isaac-capture"
    assert artifact_stage_label("isaac-capture") == "Isaac capture"
    assert artifact_stage_label("reports") == "Reports / visualization"


# Runs nested as <run>/<workflow-name>/<stage>/... must expose their real stages,
# not collapse into the single workflow-name wrapper row.
_NESTED_KEYS = [
    "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/plan/scene_reasoning.json",
    "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/scene/frame_000.png",
    "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/augment/frame-00000.png",
    "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/gate/decision.json",
    "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/scores/vlm_eval_stub.json",
]


def test_run_stage_wrapper_detects_workflow_name_nesting() -> None:
    from npa.cli.agent_stages import run_stage_wrapper

    assert run_stage_wrapper(_NESTED_KEYS, "run-1", "npa-workflow-e2e") == "tokenfactory-cosmos-gate"


def test_nested_run_exposes_real_pipeline_stages_not_wrapper() -> None:
    stages = build_artifact_backed_stages(
        _NESTED_KEYS,
        run_id="run-1",
        prefix="npa-workflow-e2e",
        workflow_stage_defs=[],
        overlay_unmatched=False,
    )
    stage_keys = {s["stage_key"] for s in stages}
    assert stage_keys == {"plan", "scene", "augment", "gate", "scores"}
    assert "tokenfactory-cosmos-gate" not in stage_keys


def test_artifact_stage_key_strips_wrapper() -> None:
    key = "npa-workflow-e2e/run-1/tokenfactory-cosmos-gate/augment/frame-00000.png"
    assert artifact_stage_key(key, "run-1", "npa-workflow-e2e", "tokenfactory-cosmos-gate") == "augment"


def test_run_stage_wrapper_leaves_flat_layouts_untouched() -> None:
    from npa.cli.agent_stages import run_stage_wrapper

    # Multi-stage flat layout: no common wrapper.
    flat = [
        "checkpoints/paidf/run-1/configs/manifest.json",
        "checkpoints/paidf/run-1/cosmos_augmented/manifest.json",
    ]
    assert run_stage_wrapper(flat, "run-1", "checkpoints/paidf") == ""
    # A single real stage whose children are files must NOT be stripped.
    reports_only = [
        "checkpoints/paidf/run-1/reports/final.json",
        "checkpoints/paidf/run-1/reports/sim2real.rrd",
    ]
    assert run_stage_wrapper(reports_only, "run-1", "checkpoints/paidf") == ""


# ── custom npa.workflow runs: per-state subprefix → named succeeded stages ────
# Regression for the "almost all stages show not run / no artifacts" report: a
# custom workflow (not the 14-stage sim2real engine) that persists each state's
# outputs under <run_id>/<state>/... must render one succeeded stage per state,
# and a run with no persisted artifacts must render no succeeded stages.

_REDTEAM_KEYS = [
    "checkpoints/sim2real-b/redteam-1/hypothesize-failures/tasks.txt",
    "checkpoints/sim2real-b/redteam-1/hypothesize-failures/hypotheses.jsonl",
    "checkpoints/sim2real-b/redteam-1/derive-mitigation-prompts/mitigation_prompts.txt",
    "checkpoints/sim2real-b/redteam-1/synthesize-mitigations/mitigations.jsonl",
    "checkpoints/sim2real-b/redteam-1/assemble-eval-contract/eval_contract.jsonl",
    "checkpoints/sim2real-b/redteam-1/reports/summary.json",
    "checkpoints/sim2real-b/redteam-1/spec/sim2real-redteam-mitigation.yaml",
    "checkpoints/sim2real-b/redteam-1/state/workflow_state.json",
]


def test_per_state_subprefix_yields_one_named_succeeded_stage_each() -> None:
    stages = build_artifact_backed_stages(
        _REDTEAM_KEYS,
        run_id="redteam-1",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=[],  # custom workflow: no engine stage-defs overlay
        overlay_unmatched=False,
    )
    by_id = {s["id"]: s for s in stages}
    # Each workflow state that persisted artifacts becomes its own succeeded row.
    for state_id in (
        "hypothesize-failures",
        "derive-mitigation-prompts",
        "synthesize-mitigations",
        "assemble-eval-contract",
        "reports",
    ):
        assert state_id in by_id, (state_id, list(by_id))
        assert by_id[state_id]["status"] == "succeeded"
    # The two files under hypothesize-failures collapse into one stage row.
    assert by_id["hypothesize-failures"]["summary"].startswith("2 artifacts")
    # Custom state names get a readable title-cased label (not left blank).
    assert by_id["hypothesize-failures"]["label"] == "Hypothesize failures"
    # Known key keeps its curated label.
    assert by_id["reports"]["label"] == "Reports / visualization"


def test_artifact_stage_key_strips_per_state_prefix() -> None:
    key = "checkpoints/sim2real-b/redteam-1/assemble-eval-contract/eval_contract.jsonl"
    assert artifact_stage_key(key, "redteam-1", "checkpoints/sim2real-b") == "assemble-eval-contract"


def test_no_artifacts_yields_no_succeeded_stages() -> None:
    # The reported symptom's root cause: a run that persisted nothing to storage
    # has no artifact-backed stages to show (the UI then renders them as not-run).
    assert (
        build_artifact_backed_stages(
            [],
            run_id="redteam-local-only",
            prefix="checkpoints/sim2real-b",
            workflow_stage_defs=[],
            overlay_unmatched=False,
        )
        == []
    )


def test_owned_run_without_artifacts_shows_all_stages_pending() -> None:
    # Same symptom via the overlay path: an owned run whose states have no
    # artifacts yet renders every workflow state as pending ("not run").
    workflow_defs = [
        ("hypothesize-failures", "Hypothesize failures", ["hypothesize-failures"]),
        ("synthesize-mitigations", "Synthesize mitigations", ["synthesize-mitigations"]),
    ]
    stages = build_artifact_backed_stages(
        [],
        run_id="redteam-owned",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=workflow_defs,
        overlay_unmatched=True,
    )
    assert stages, "owned run should still surface its declared stages"
    assert all(s["status"] == "pending" for s in stages)

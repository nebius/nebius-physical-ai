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

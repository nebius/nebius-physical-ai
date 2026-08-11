"""Behavioral unit tests for artifact-backed Stages overlay helpers."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from types import SimpleNamespace

from npa.cli import agent_stages
from npa.cli.agent_stages import (
    artifact_stage_key,
    artifact_stage_label,
    build_artifact_backed_stages,
    build_available_sim_viz_runs,
    parse_stage_evidence_documents,
    run_owns_workflow_stage_overlay,
    summarize_stage_evidence,
)


def test_available_run_projection_preserves_selected_artifact_source() -> None:
    available = build_available_sim_viz_runs(
        [
            {
                "run_id": "duplicate-run",
                "source_type": "artifact_storage",
                "source_label": "S3 artifacts",
                "bucket": "bucket-a",
                "project_id": "project-a",
                "resolved_prefix": "preferred-source",
            }
        ]
    )

    assert available[0]["run_id"] == "duplicate-run"
    assert available[0]["bucket"] == "bucket-a"
    assert available[0]["project_id"] == "project-a"
    assert available[0]["resolved_prefix"] == "preferred-source"


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
    assert all(s["status"] == "observed_output" for s in stages)
    assert all(s["authority"] == "observed" for s in stages)
    assert "isaac-capture" in ids
    assert "reports" in ids


def test_build_stages_keeps_unknown_draft_when_owned() -> None:
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
    assert by_id["augment"]["status"] == "status_unavailable"
    assert by_id["envgen"]["status"] == "status_unavailable"
    assert by_id["isaac-capture"]["status"] == "observed_output"
    assert not {stage["status"] for stage in stages} & {"succeeded", "failed", "not_run"}


def test_build_stages_marks_matched_draft_state_observed_only() -> None:
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
    assert by_id["augment"]["status"] == "observed_output"
    assert by_id["envgen"]["status"] == "observed_output"
    assert by_id["rollouts"]["status"] == "status_unavailable"


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
# outputs under <run_id>/<state>/... must render one observed-output stage per
# state, and a run with no persisted artifacts must render no stages.

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


def test_per_state_subprefix_yields_one_named_observed_stage_each() -> None:
    stages = build_artifact_backed_stages(
        _REDTEAM_KEYS,
        run_id="redteam-1",
        prefix="checkpoints/sim2real-b",
        workflow_stage_defs=[],  # custom workflow: no engine stage-defs overlay
        overlay_unmatched=False,
    )
    by_id = {s["id"]: s for s in stages}
    # Each workflow state that persisted artifacts becomes an observed row; an
    # output object is not proof the producing command succeeded.
    for state_id in (
        "hypothesize-failures",
        "derive-mitigation-prompts",
        "synthesize-mitigations",
        "assemble-eval-contract",
        "reports",
    ):
        assert state_id in by_id, (state_id, list(by_id))
        assert by_id[state_id]["status"] == "observed_output"
        assert by_id[state_id]["evidence_type"] == "artifact_observation"
    # The two files under hypothesize-failures collapse into one stage row.
    assert by_id["hypothesize-failures"]["summary"].startswith("Observed 2 artifacts")
    # Custom state names get a readable title-cased label (not left blank).
    assert by_id["hypothesize-failures"]["label"] == "Hypothesize failures"
    # Known key keeps its curated label.
    assert by_id["reports"]["label"] == "Reports / visualization"


def test_artifact_stage_key_strips_per_state_prefix() -> None:
    key = "checkpoints/sim2real-b/redteam-1/assemble-eval-contract/eval_contract.jsonl"
    assert artifact_stage_key(key, "redteam-1", "checkpoints/sim2real-b") == "assemble-eval-contract"


def test_no_artifacts_yields_no_fabricated_stages() -> None:
    # The reported symptom's root cause: a run that persisted nothing to storage
    # has no artifact-backed stages to show and must remain evidence-free.
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


def test_owned_run_without_artifacts_shows_status_unavailable() -> None:
    # A plan declares graph membership, not an execution attempt or outcome.
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
    assert all(s["status"] == "status_unavailable" for s in stages)
    assert not {stage["status"] for stage in stages} & {"succeeded", "failed", "not_run"}


def test_artifact_only_run_has_only_observed_groups_and_grounded_summary() -> None:
    keys = [
        "runs/foreign-workflow/run-7/capture/frame.png",
        "runs/foreign-workflow/run-7/capture/metadata.json",
        "runs/foreign-workflow/run-7/train/checkpoint.bin",
        "runs/foreign-workflow/run-7/evaluation/metrics.json",
    ]
    stages = build_artifact_backed_stages(
        keys,
        run_id="run-7",
        prefix="runs/foreign-workflow",
        workflow_stage_defs=[],
        overlay_unmatched=False,
    )
    assert [stage["stage_key"] for stage in stages] == ["capture", "evaluation", "train"]
    assert {stage["status"] for stage in stages} == {"observed_output"}
    assert all(stage["artifact_count"] > 0 for stage in stages)
    summary = summarize_stage_evidence(stages)
    assert summary["text"] == "3 observed groups · execution status unavailable"
    assert summary["succeeded_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["not_run_count"] == 0


def test_authoritative_manifest_preserves_all_explicit_statuses_with_provenance() -> None:
    documents = [
        {
            "key": "runs/run-8/npa-workflow/manifest.json",
            "payload": {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "foreign-workflow",
                "run_id": "run-8",
                "status": "running",
                "updated_at": "2026-08-07T01:02:03Z",
                "steps": [
                    {"state": "prepare", "status": "ok", "returncode": 0},
                    {"state": "train", "status": "failed", "returncode": 17},
                    {"state": "evaluate", "status": "running"},
                    {"state": "optional", "status": "skipped"},
                    {"state": "disabled", "status": "not_run"},
                    {"state": "publish", "status": "submitted"},
                ],
            },
        }
    ]
    parsed = parse_stage_evidence_documents(documents)
    assert parsed["workflow_name"] == "foreign-workflow"
    assert parsed["run_status"] == "running"
    by_id = {stage["id"]: stage for stage in parsed["stages"]}
    assert {key: by_id[key]["status"] for key in by_id} == {
        "prepare": "succeeded",
        "train": "failed",
        "evaluate": "running",
        "optional": "skipped",
        "disabled": "not_run",
        "publish": "pending",
    }
    assert by_id["publish"]["status_label"] == "Submitted"
    assert all(stage["authority"] == "authoritative" for stage in by_id.values())
    assert all(stage["evidence_source"].endswith("/npa-workflow/manifest.json") for stage in by_id.values())


def test_authoritative_graph_is_workflow_specific_and_artifacts_do_not_turn_it_green() -> None:
    parsed = parse_stage_evidence_documents(
        [
            {
                "key": "runs/run-9/npa-workflow/manifest.json",
                "payload": {
                    "schema_version": "npa.workflow.run.v1",
                    "workflow": "two-stage-inspection",
                    "status": "submitted",
                    "steps": [
                        {"state": "inspect", "status": "submitted"},
                        {"state": "publish", "status": "submitted"},
                    ],
                },
            }
        ]
    )
    stages = build_artifact_backed_stages(
        ["runs/run-9/inspect/result.json"],
        run_id="run-9",
        prefix="runs",
        workflow_stage_defs=[],
        overlay_unmatched=False,
        authoritative_stages=parsed["stages"],
    )
    assert [stage["id"] for stage in stages] == ["inspect", "publish"]
    assert [stage["status"] for stage in stages] == ["pending", "pending"]
    assert stages[0]["artifact_count"] == 1
    assert stages[1]["artifact_count"] == 0
    assert all(not stage["id"].startswith("stage-0") for stage in stages)


def test_status_document_overrides_manifest_snapshot_and_preserves_timestamps() -> None:
    parsed = parse_stage_evidence_documents(
        [
            {
                "key": "workflow/run/manifest.json",
                "payload": {
                    "workflow_name": "durable-workflow",
                    "run_prefix_uri": "s3://bucket/workflow/run",
                    "stages": {"train": {"name": "Train"}},
                },
            },
            {
                "key": "workflow/run/logs/train/status.json",
                "payload": {
                    "stage": "train",
                    "state": "FAILED",
                    "start_time": "2026-08-07T00:00:00Z",
                    "end_time": "2026-08-07T00:03:00Z",
                    "error_summary": "exit 17",
                },
            },
        ]
    )
    assert len(parsed["stages"]) == 1
    stage = parsed["stages"][0]
    assert stage["status"] == "failed"
    assert stage["started_at"] == "2026-08-07T00:00:00Z"
    assert stage["finished_at"] == "2026-08-07T00:03:00Z"
    assert stage["evidence_type"] == "workflow_status"


def test_report_promotes_only_explicit_stage_outcomes_and_drops_secrets() -> None:
    marker = "secret-value-must-not-escape"
    parsed = parse_stage_evidence_documents(
        [
            {
                "key": "runs/run-10/reports/run-report.json",
                "payload": {
                    "token": marker,
                    "stages": [
                        {"id": "scored", "status": "success", "api_key": marker},
                        {"id": "metrics-only", "score": 0.9, "api_key": marker},
                    ],
                    "stage_outcomes": {"reviewed": {"status": "skipped", "password": marker}},
                },
            }
        ]
    )
    by_id = {stage["id"]: stage for stage in parsed["stages"]}
    assert set(by_id) == {"scored", "reviewed"}
    assert by_id["scored"]["status"] == "succeeded"
    assert by_id["reviewed"]["status"] == "skipped"
    assert marker not in str(parsed)


def test_unrecognized_manifest_remains_an_observed_artifact_group() -> None:
    key = "runs/run-11/configs/manifest.json"
    parsed = parse_stage_evidence_documents(
        [{"key": key, "payload": {"dataset": "example", "samples": 3}}]
    )
    assert parsed["stages"] == []
    assert parsed["consumed_sources"] == []
    stages = build_artifact_backed_stages(
        [key],
        run_id="run-11",
        prefix="runs",
        workflow_stage_defs=[],
        overlay_unmatched=False,
        authoritative_stages=parsed["stages"],
        evidence_keys=parsed["consumed_sources"],
    )
    assert len(stages) == 1
    assert stages[0]["stage_key"] == "configs"
    assert stages[0]["status"] == "observed_output"


def test_public_workflow_command_redacts_separator_and_inline_canaries() -> None:
    from npa.cli.agent_stage_runtime import _public_workflow_command

    canaries = [secrets.token_urlsafe(32) for _index in range(11)]
    canaries[2] = "--" + canaries[2]
    command = _public_workflow_command(
        [
            "tool",
            f"Authorization: Bearer {canaries[0]}",
            f"--secret-key:{canaries[1]}",
            "--password",
            canaries[2],
            f"--password={canaries[3]}",
            f"subcommand --password {canaries[4]} --output kept.json",
            "Bearer",
            canaries[5],
            "Authorization:",
            "Bearer",
            canaries[6],
            "--secret-key:",
            canaries[7],
            f"AWS_SECRET_ACCESS_KEY={canaries[8]}",
            f"--access-key-id:{canaries[9]}",
            "AWS_SESSION_TOKEN",
            canaries[10],
            "--password=",
            "--unrelated",
            "kept",
        ]
    )

    assert all(canary not in command for canary in canaries)
    assert "Authorization: Bearer <redacted>" in command
    assert "--secret-key:<redacted>" in command
    assert "--password <redacted>" in command
    assert "--password=<redacted>" in command
    assert "AWS_SECRET_ACCESS_KEY=<redacted>" in command
    assert "--access-key-id:<redacted>" in command
    assert "--output kept.json" in command
    assert "--unrelated kept" in command


class _EvidenceBody:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int):
        self.read_sizes.append(size)
        return self.payload[:size]

    def close(self):
        self.closed = True


class _EvidenceS3:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[str] = []
        self.bodies: list[_EvidenceBody] = []

    def get_object(self, *, Bucket, Key):  # noqa: N803
        del Bucket
        self.calls.append(Key)
        body = _EvidenceBody(self.payloads[Key])
        self.bodies.append(body)
        return {"Body": body}


def test_stage_evidence_reads_are_count_bounded_and_deterministic() -> None:
    from npa.cli.agent_stage_runtime import (
        _MAX_STAGE_EVIDENCE_DOCUMENTS,
        _stage_evidence_documents,
    )

    artifacts = [
        SimpleNamespace(key=f"run/logs/stage-{index:02d}/status.json", size=32)
        for index in range(12)
    ]
    payloads = {
        artifact.key: json.dumps({"stage": artifact.key, "status": "ok"}).encode()
        for artifact in artifacts
    }
    s3 = _EvidenceS3(payloads)

    documents = _stage_evidence_documents(s3, "bucket", list(reversed(artifacts)))

    expected = sorted(payloads)[:_MAX_STAGE_EVIDENCE_DOCUMENTS]
    assert s3.calls == expected
    assert [item["key"] for item in documents] == expected
    assert all(body.closed for body in s3.bodies)


def test_stage_evidence_reads_enforce_byte_cap_and_close_every_body() -> None:
    from npa.cli.agent_stage_runtime import (
        _MAX_STAGE_EVIDENCE_BYTES,
        _stage_evidence_documents,
    )

    oversized_metadata = SimpleNamespace(
        key="run/reports/metadata-report.json",
        size=_MAX_STAGE_EVIDENCE_BYTES + 1,
    )
    oversized_stream = SimpleNamespace(key="run/reports/stream-report.json", size=0)
    malformed = SimpleNamespace(key="run/reports/malformed-report.json", size=8)
    valid = SimpleNamespace(key="run/npa-workflow/manifest.json", size=32)
    s3 = _EvidenceS3(
        {
            oversized_stream.key: b"{" + b"x" * _MAX_STAGE_EVIDENCE_BYTES,
            malformed.key: b"not-json",
            valid.key: b'{"schema_version":"npa.workflow.run.v1"}',
        }
    )

    documents = _stage_evidence_documents(
        s3,
        "bucket",
        [oversized_metadata, oversized_stream, malformed, valid],
    )

    assert oversized_metadata.key not in s3.calls
    assert [item["key"] for item in documents] == [valid.key]
    assert all(body.read_sizes == [_MAX_STAGE_EVIDENCE_BYTES + 1] for body in s3.bodies)
    assert all(body.closed for body in s3.bodies)


def test_report_summary_uses_bounded_object_read_without_persisting_download() -> None:
    runtime_source = (
        Path(agent_stages.__file__)
        .with_name("agent_stage_runtime.py")
        .read_text(encoding="utf-8")
    )

    report_block = runtime_source.split("if report_artifact:", 1)[1].split(
        "stage_summary =", 1
    )[0]
    assert "_read_bounded_json_object(s3, run_bucket, report_artifact.key)" in report_block
    assert "download_s3_uri" not in report_block
    assert "RECORDINGS_DIR" not in report_block


def test_stage_evidence_document_order_preserves_status_precedence() -> None:
    from npa.cli.agent_stage_runtime import _stage_evidence_documents

    artifacts = [
        SimpleNamespace(key="run/logs/train/status.json", size=32),
        SimpleNamespace(key="run/npa-workflow/manifest.json", size=32),
        SimpleNamespace(key="run/reports/final-report.json", size=32),
    ]
    s3 = _EvidenceS3(
        {
            artifacts[0].key: b'{"stage":"train","status":"failed"}',
            artifacts[1].key: b'{"steps":[{"state":"train","status":"running"}]}',
            artifacts[2].key: b'{"stages":{"train":{"status":"succeeded"}}}',
        }
    )

    documents = _stage_evidence_documents(s3, "bucket", artifacts)
    assert [item["key"] for item in documents] == [
        "run/reports/final-report.json",
        "run/npa-workflow/manifest.json",
        "run/logs/train/status.json",
    ]
    parsed = parse_stage_evidence_documents(documents)
    assert parsed["stages"][0]["status"] == "failed"
    assert parsed["stages"][0]["evidence_source"] == "run/logs/train/status.json"

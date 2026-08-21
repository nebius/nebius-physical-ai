"""Regression contracts for recording-independent artifact viewing."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.artifacts import (
    GROOT_ARTIFACT_PATHS,
    INLINE_TEXT_MAX_BYTES,
    Artifact,
    ArtifactDiscoveryError,
    artifact_category_for_relative_key,
    authorize_artifact_inventory_key,
    build_run_summary,
    build_text_preview,
    parse_http_byte_range,
    render_hint_for_object,
    safe_content_disposition,
)


RUN_ID = "groot17-8gpu-20260806T024557Z-3dfb0270"
AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
ARTIFACT_CONTENT_MODULE = AGENT_MODULE.with_name("agent_artifact_content.py")
AGENT_UI = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"


def _artifact(
    relative_key: str,
    *,
    size: int = 10,
    role: str = "output",
    namespace: str = "",
) -> Artifact:
    key = f"{namespace + '/' if namespace else ''}{RUN_ID}/{relative_key}"
    render = render_hint_for_object(key=key)
    return Artifact(
        run_id=RUN_ID,
        key=key,
        s3_uri=f"s3://bucket/{key}",
        size=size,
        last_modified="2026-08-06T03:00:00+00:00",
        render=render,
        inline=render != "download",
        role=role,
        namespace=namespace,
        relative_key=relative_key,
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("manifest.json", "json"),
        ("workflow.yaml", "text"),
        ("evidence/training.log", "text"),
        ("training-summary.png", "image"),
        ("data/episode.mp4", "video"),
        ("recording.rrd", "rerun"),
        ("recording.mcap", "mcap"),
        ("checkpoints/model.pt", "download"),
        ("unknown.payload", "download"),
    ],
)
def test_artifact_classification_preserves_download_fallback(key: str, expected: str) -> None:
    assert render_hint_for_object(key=key) == expected


def test_artifact_roles_categories_and_binary_download_only_metadata() -> None:
    checkpoint = _artifact("checkpoints/model.pt", size=9_335_640_879)
    staged = _artifact("data/episode.mp4", role="input", namespace="groot-1-7-finetune")

    assert checkpoint.to_dict()["category"] == "checkpoint"
    assert checkpoint.to_dict()["download_only"] is True
    assert checkpoint.to_dict()["content_type"] == "application/octet-stream"
    assert checkpoint.to_dict()["is_output"] is True
    assert staged.to_dict()["category"] == "input"
    assert staged.to_dict()["is_output"] is False
    assert artifact_category_for_relative_key("manifest.json") == "manifest"
    assert artifact_category_for_relative_key("workflow.yaml") == "config"


def test_run_key_authorization_is_exact_and_rejects_traversal_or_other_runs() -> None:
    allowed = [f"{RUN_ID}/manifest.json", f"staged/{RUN_ID}/data/episode.mp4"]

    assert authorize_artifact_inventory_key(RUN_ID, allowed[0], allowed) == allowed[0]
    with pytest.raises(ArtifactDiscoveryError, match="traversal"):
        authorize_artifact_inventory_key(RUN_ID, f"{RUN_ID}/../other/secret", allowed)
    with pytest.raises(ArtifactDiscoveryError, match="validated run"):
        authorize_artifact_inventory_key(RUN_ID, "other-run/manifest.json", allowed)
    with pytest.raises(Exception, match="run-id"):
        authorize_artifact_inventory_key("../other-run", allowed[0], allowed)


def test_text_preview_caps_utf8_formats_json_and_redacts_secrets() -> None:
    payload = (
        b'{"run_id":"safe","password":"do-not-show","nested":'
        b'{"authorization":"Bearer abcdefghijklmnop"},"message":"ok"}'
    )
    preview = build_text_preview(payload, total_bytes=len(payload), render="json")

    assert preview["text"].startswith("{\n")
    assert "do-not-show" not in preview["text"]
    assert "abcdefghijklmnop" not in preview["text"]
    assert preview["text"].count("[REDACTED]") == 2
    assert preview["redacted"] is True
    assert preview["truncated"] is False

    capped = build_text_preview(
        b"token=supersecret\n" + b"x" * 20 + b"\xff",
        total_bytes=100,
        render="text",
        max_bytes=24,
    )
    assert capped["bytes_read"] == 24
    assert capped["max_bytes"] == 24
    assert capped["truncated"] is True
    assert "supersecret" not in capped["text"]
    assert INLINE_TEXT_MAX_BYTES == 256 * 1024


@pytest.mark.parametrize(
    ("value", "total", "expected"),
    [
        ("", 100, None),
        ("bytes=0-99", 1000, (0, 99)),
        ("bytes=900-", 1000, (900, 999)),
        ("bytes=-100", 1000, (900, 999)),
        ("bytes=0-9999", 1000, (0, 999)),
    ],
)
def test_single_http_byte_ranges(value: str, total: int, expected: tuple[int, int] | None) -> None:
    assert parse_http_byte_range(value, total) == expected


@pytest.mark.parametrize("value", ["items=0-1", "bytes=0-1,4-5", "bytes=1000-", "bytes=4-2"])
def test_invalid_or_multiple_byte_ranges_are_rejected(value: str) -> None:
    with pytest.raises(ArtifactDiscoveryError):
        parse_http_byte_range(value, 1000)


def test_safe_content_disposition_drops_path_and_header_characters() -> None:
    header = safe_content_disposition('../bad\r\nname".pt', attachment=True)

    assert header == 'attachment; filename="bad_name_.pt"'
    assert "\r" not in header and "\n" not in header and "/" not in header


def test_run_summary_parses_safe_completed_training_fields_without_recording() -> None:
    artifacts = [
        _artifact("manifest.json", size=100),
        _artifact("workflow.yaml", size=200),
        _artifact("evidence/training.log", size=300),
        _artifact("training-summary.png", size=400),
        _artifact("checkpoints/model.pt", size=500),
        _artifact(
            "data/episode.mp4",
            size=600,
            role="input",
            namespace="groot-1-7-finetune",
        ),
    ]
    documents = {
        "manifest.json": {"workflow_name": "groot-1-7-finetune", "status": "completed"},
        "npa-workflow/manifest.json": {
            "workflow": "groot-1-7-finetune",
            "steps": [{"tool_ref": "workbench.groot.finetune"}],
        },
        "evidence/training.json": {
            "status": "completed",
            "gpu_count": 8,
            "accelerator": "RTX PRO 6000 Blackwell Server Edition",
            "world_size": 8,
            "optimizer_steps": 1,
            "train_loss": 1.03125,
            "loss_finite": True,
        },
    }

    summary = build_run_summary(RUN_ID, artifacts, documents)

    assert summary == {
        "run_id": RUN_ID,
        "completion_status": "completed",
        "workflow": "groot-1-7-finetune",
        "tool": "workbench.groot.finetune",
        "accelerator_count": 8,
        "accelerator_type": "RTX PRO 6000 Blackwell Server Edition",
        "world_size": 8,
        "training_steps": 1,
        "loss": 1.03125,
        "finite_loss": True,
        "loss_history": [],
        "loss_point_count": 0,
        "loss_source": "evidence/training.json",
        "loss_validation_error": "",
        "artifact_count": 6,
        "output_artifact_count": 5,
        "input_artifact_count": 1,
        "metadata_artifact_count": 0,
        "total_bytes": 2100,
        "has_recording": False,
        "recording_state": "No RRD/MCAP recording; use the artifacts below",
    }


def test_unshipped_task_performance_report_does_not_override_run_summary() -> None:
    report = {
        "schema": "npa.groot.task_performance.v1",
        "status": "passed",
        "task": {"name": "PushT", "goal": "Push the T-shaped block onto the target."},
        "platform": {
            "label": "Simulated",
            "physical_robot": False,
            "simulation": True,
            "environment": {"id": "gym_pusht/PushT-v0", "version": "0.1.6"},
            "embodiment": {"name": "2D simulated circular pusher"},
        },
        "paired_evaluation": {
            "episode_count": 24,
            "same_initial_conditions": True,
            "baseline_checkpoint": {"uri": "s3://bucket/baseline/"},
            "trained_checkpoint": {"uri": "s3://bucket/trained/"},
            "episodes": [{"seed": 7, "outcome_category": "trained_win"}],
        },
        "performance": {
            "primary_metric": "maximum_goal_coverage",
            "baseline_success_rate": 0.1,
            "trained_success_rate": 0.4,
            "success_rate_delta": 0.3,
            "baseline_task_score": 0.2,
            "trained_task_score": 0.5,
            "task_score_delta": 0.3,
            "primary_evidence": {
                "confidence_level": 0.95,
                "ci_low": 0.1,
                "ci_high": 0.5,
                "paired_test": "paired sign randomization",
                "p_value": 0.001,
            },
            "improvement_gate_passed": True,
            "conclusion": "PASS",
        },
        "action": {
            "semantics": ["absolute target pusher x", "absolute target pusher y"],
            "units": ["workspace pixels", "workspace pixels"],
            "range": [[0, 512], [0, 512]],
        },
        "success_definition": {"predicate": "coverage > 0.95"},
    }
    artifacts = [
        _artifact("reports/task-performance-report.json"),
        _artifact("reports/task-performance.rrd"),
    ]
    summary = build_run_summary(
        RUN_ID, artifacts, {"reports/task-performance-report.json": report}
    )
    assert "task_performance" not in summary
    assert summary["completion_status"] == "unknown"


def test_run_summary_recognizes_operational_two_gpu_report() -> None:
    report = {
        "schema": "npa.groot.learning.v1",
        "status": "completed",
        "pipeline_status": "succeeded",
        "learning_outcome": "not_improved",
        "candidate_promoted": False,
        "evaluation_kind": "offline held-out policy evaluation",
        "closed_loop": False,
        "dataset": {
            "embodiment": "NEW_EMBODIMENT",
            "camera_names": ["front"],
            "source_resolution": "640x480",
            "train_episodes": 2,
            "heldout_episodes": 1,
            "heldout_samples": 201,
            "split_hash": "split-sha256",
            "leakage_free": True,
        },
        "provenance": {"primary_camera": "front"},
        "training": {
            "gpu_count": 2,
            "optimizer_steps": 4,
            "training_examples": 8,
            "epoch_equivalent": 0.0199,
            "checkpoint_uri": "s3://bucket/run/checkpoints/candidate/checkpoint-4/",
            "final_loss": 1.3203,
            "loss_history": [
                {"optimizer_step": 1, "loss": 1.2812},
                {"optimizer_step": 2, "loss": 1.3516},
                {"optimizer_step": 3, "loss": 1.3281},
                {"optimizer_step": 4, "loss": 1.3203},
            ],
        },
        "evaluation": {
            "metric_name": "action_mse",
            "baseline_value": 0.1,
            "posttrain_value": 0.2,
        },
    }
    summary = build_run_summary(
        RUN_ID,
        [_artifact("reports/two-gpu-pipeline-report.json")],
        {"reports/two-gpu-pipeline-report.json": report},
    )

    assert summary["completion_status"] == "completed"
    assert summary["accelerator_count"] == 2
    assert summary["training_steps"] == 4
    assert summary["learning"]["pipeline_status"] == "succeeded"
    assert summary["learning"]["learning_outcome"] == "not_improved"
    assert summary["learning"]["candidate_promoted"] is False
    assert summary["finite_loss"] is True
    assert summary["loss_point_count"] == 4
    assert summary["loss_source"] == "reports/two-gpu-pipeline-report.json"
    assert summary["learning"]["artifact_contract"]["authoritative"] is True
    assert summary["learning"]["artifact_contract"]["primary_camera"] == "front"
    artifact_content_source = ARTIFACT_CONTENT_MODULE.read_text(encoding="utf-8")
    assert "GROOT_ARTIFACT_PATHS[\"report\"]" in artifact_content_source
    assert summary["loss_source"] == GROOT_ARTIFACT_PATHS["report"][0]


def test_real_workflow_layout_serializes_one_authoritative_path_contract() -> None:
    report = {
        "schema": "npa.groot.learning.v1",
        "evaluation_kind": "offline held-out policy evaluation",
        "closed_loop": False,
        "dataset": {"camera_names": ["wrist_rgb"]},
        "provenance": {"primary_camera": "wrist_rgb"},
        "training": {
            "final_loss": 0.75,
            "loss_history": [
                {"optimizer_step": 1, "loss": 1.0},
                {"optimizer_step": 2, "loss": 0.75},
            ],
        },
    }
    paths = [
        "reports/two-gpu-pipeline-report.json",
        "reports/groot-offline-evaluation.rrd",
        "reports/groot-offline-evaluation.mcap",
        "offline/baseline/evaluation.json",
        "offline/trained/evaluation.json",
        "checkpoints/candidate/npa_groot_finetune_manifest.json",
        "reports/trained-checkpoint.json",
    ]
    summary = build_run_summary(
        RUN_ID,
        [_artifact(path) for path in paths],
        {"reports/two-gpu-pipeline-report.json": report},
    )
    contract = summary["learning"]["artifact_contract"]
    assert contract["authoritative"] is True
    assert contract["primary_camera"] == "wrist_rgb"
    assert contract["matches"]["rrd"] == ["reports/groot-offline-evaluation.rrd"]
    assert contract["matches"]["mcap"] == ["reports/groot-offline-evaluation.mcap"]
    assert contract["matches"]["baseline_evaluation"] == [
        "offline/baseline/evaluation.json"
    ]
    assert contract["matches"]["trained_evaluation"] == [
        "offline/trained/evaluation.json"
    ]
    assert summary["loss_history"] == [
        {"optimizer_step": 1, "loss": 1.0},
        {"optimizer_step": 2, "loss": 0.75},
    ]


@pytest.mark.parametrize(
    "loss_history,final_loss",
    [
        ([{"optimizer_step": 1, "loss": float("nan")}], float("nan")),
        ([{"optimizer_step": 2, "loss": 1.0}, {"optimizer_step": 1, "loss": 0.5}], 0.5),
        ([{"optimizer_step": 1, "loss": 1.0}], 0.5),
    ],
)
def test_gr00t_loss_evidence_fails_closed_when_malformed_or_nonfinite(
    loss_history: list[dict], final_loss: float
) -> None:
    report = {
        "schema": "npa.groot.learning.v1",
        "evaluation_kind": "offline held-out policy evaluation",
        "closed_loop": False,
        "dataset": {"camera_names": ["front"]},
        "provenance": {"primary_camera": "front"},
        "training": {"loss_history": loss_history, "final_loss": final_loss},
    }
    summary = build_run_summary(
        RUN_ID,
        [_artifact("reports/two-gpu-pipeline-report.json")],
        {"reports/two-gpu-pipeline-report.json": report},
    )
    assert summary["finite_loss"] is False
    assert summary["loss"] is None
    assert summary["loss_history"] == []
    assert summary["loss_validation_error"] == "malformed_or_nonfinite_loss_evidence"


def test_candidate_training_manifest_is_a_real_loss_source_without_report_history() -> None:
    report = {
        "schema": "npa.groot.learning.v1",
        "evaluation_kind": "offline held-out policy evaluation",
        "closed_loop": False,
        "dataset": {"camera_names": ["front"]},
        "provenance": {"primary_camera": "front"},
        "training": {},
    }
    manifest = {
        "loss_history": [
            {"optimizer_step": 1, "loss": 1.2},
            {"optimizer_step": 4, "loss": 1.1},
        ],
        "final_step_loss": 1.1,
    }
    summary = build_run_summary(
        RUN_ID,
        [
            _artifact("reports/two-gpu-pipeline-report.json"),
            _artifact("checkpoints/candidate/npa_groot_finetune_manifest.json"),
        ],
        {
            "reports/two-gpu-pipeline-report.json": report,
            "checkpoints/candidate/npa_groot_finetune_manifest.json": manifest,
        },
    )
    assert summary["finite_loss"] is True
    assert summary["loss"] == 1.1
    assert summary["loss_source"].startswith("checkpoints/candidate/")


def test_ui_makes_operational_offline_learning_primary() -> None:
    source = AGENT_UI.read_text(encoding="utf-8")
    assert "Policy learning summary" in source
    assert "Open GR00T offline RRD" in source
    assert "Open GR00T offline MCAP" in source
    assert "Training loss timeline" in source
    assert "artifact_contract" in source


def test_secure_content_endpoint_contract_is_s3_only_and_range_aware() -> None:
    source = ARTIFACT_CONTENT_MODULE.read_text(encoding="utf-8")

    assert '@app.api_route("/artifacts/content", methods=["GET", "HEAD"])' in source
    assert "authorize_artifact_inventory_key(" in source
    assert '"X-Content-Type-Options": "nosniff"' in source
    assert '"Content-Range"' in source
    assert '"Accept-Ranges": "bytes"' in source
    assert 'not request.headers.get("range")' in source
    assert "INLINE_TEXT_MAX_BYTES" in source
    assert 'render in {"json", "text"} and not download' in source
    assert "def _exact_artifact_source(" in source
    assert '"code": "exact_artifact_source_required"' in source
    assert "_authorize_exact_run_ref_source(" in source
    assert "source_authorized=True" in source
    assert '"X-NPA-Source-Selected": "true"' in source
    guard = source.split("def _resolved_artifact_for_content", 1)[1].split(
        "def _artifact_stream", 1
    )[0]
    assert "runs_root" not in guard


def test_artifact_failures_are_logged_and_never_echo_raw_exception_text() -> None:
    source = ARTIFACT_CONTENT_MODULE.read_text(encoding="utf-8")
    assert '_artifact_content_logger.exception("Artifact content storage request failed")' in source
    assert '"error": "artifact storage request failed"' in source
    assert '"error_code": "artifact_storage_error"' in source
    route = source.split("def artifacts_content", 1)[1].split("def artifact_file", 1)[0]
    assert '"error": str(exc)' not in route
    download = source.split("def artifacts_download", 1)[1]
    assert '"error": str(exc)' not in download


def test_load_artifact_s3_uri_requires_inventory_run_id_with_stable_error() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    block = source.split("def sim_viz_load_artifact", 1)[1].split(
        "def _foxglove_convert_run", 1
    )[0]
    assert '"code": "run_id_required_for_s3_uri"' in block
    assert '"contract_version": "npa.agent.load-artifact.v2"' in block
    assert '"migration"' in block
    assert '"required_fields": ["run_id", "s3_uri"]' in block
    assert "_resolved_artifact_for_content(" in block
    assert '"error": "artifact storage request failed"' in block
    assert '"error": str(exc)' not in block


def test_ui_keeps_artifact_list_after_preview_errors_and_never_requires_recording() -> None:
    source = AGENT_UI.read_text(encoding="utf-8")

    assert "No RRD/MCAP recording; use the artifacts below" in source
    assert "previewArtifact" in source
    assert "host.replaceChildren" in source
    assert "artifactList.replaceChildren" not in source[source.index("async function previewArtifact"):]
    assert "textContent" in source[source.index("async function previewArtifact"):]
    assert "video.controls = true" in source
    video_block = source.split('} else if (render === "video") {', 1)[1].split(
        '} else if (render === "json"', 1
    )[0]
    assert "await validateVideoPreviewResponse(contentUrl);" in video_block
    assert video_block.index("await validateVideoPreviewResponse(contentUrl);") < video_block.index(
        "video.src = contentUrl;"
    )
    assert "Video preview response is not browser media" in source
    assert "Video decode/playback failed after the media route passed HTTP validation" in source
    assert "downloadArtifact" in source
    assert '(download ? "/api/artifacts/download?" : "/api/artifacts/content?")' in source

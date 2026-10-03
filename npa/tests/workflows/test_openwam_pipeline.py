"""Contract tests for the pinned OpenWAM-alpha to LIBERO workflow."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workflows import openwam_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows" / "testing" / "openwam-libero-four-stage.yaml"
DIGEST_IMAGE = "registry.example.invalid/operator/openwam@sha256:" + "0" * 64


def test_openwam_sources_are_pinned_to_the_documented_architecture() -> None:
    assert pipeline.OPENWAM_SOURCE_REF == "48bd67b89d489b14d03b8d92bc66e65d306df32e"
    assert pipeline.FOUNDATION_REVISION == "52df4e66c82c5c8b480adcc8d01f4db7415dfb56"
    assert pipeline.WAN_REVISION == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    assert pipeline.LIBERO_SOURCE_REF == "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    assert pipeline.LIBERO_DATASET_REVISION == "bcb2eaf1121ae4cbd324f8862807abce282500e8"


def test_workflow_has_five_connected_substantive_stages() -> None:
    spec = load_spec(SPEC)
    validate_spec(spec)
    plan = build_plan(spec, run_id="openwam-contract")

    assert [step.state for step in plan.steps] == [
        "prepare-assets",
        "fine-tune-openwam",
        "deployed-policy-rollout",
        "evaluate-heldout-episode",
        "emit-factual-rrd",
    ]
    assert len(plan.steps) == 5
    assert spec.states["fine-tune-openwam"].inputs[1].uri == spec.states["prepare-assets"].outputs[1].uri
    assert spec.states["deployed-policy-rollout"].inputs[2].uri == spec.states["fine-tune-openwam"].outputs[1].uri
    assert spec.states["evaluate-heldout-episode"].inputs[3].uri == spec.states["deployed-policy-rollout"].outputs[0].uri
    assert spec.states["emit-factual-rrd"].inputs[2].uri == spec.states["evaluate-heldout-episode"].outputs[0].uri
    assert spec.states["emit-factual-rrd"].outputs[0].schema == "application/vnd.rerun.rrd"


def test_openwam_toolrefs_call_real_pipeline_stages() -> None:
    expected = {
        "workflow.openwam.prepare": "prepare",
        "workflow.openwam.fine_tune": "fine-tune",
        "workflow.openwam.rollout": "rollout",
        "workflow.openwam.evaluate": "evaluate",
        "workflow.openwam.visualize": "visualize",
    }
    for tool_ref, stage in expected.items():
        argv = TOOL_CATALOG[tool_ref].argv_template
        assert argv[:3] == ["python3", "-m", "npa.workflows.openwam_pipeline"]
        assert argv[3] == stage
        assert "--runtime-image" in argv


def test_fine_tune_invokes_the_upstream_libero_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "prepared-assets"
    assets.mkdir()
    workspace = tmp_path / "work"
    commands: list[list[str]] = []
    persisted: dict[str, object] = {}

    def fake_copy_repository(_source: Path, destination: Path) -> Path:
        (destination / "scripts").mkdir(parents=True)
        (destination / "scripts" / "train.sh").write_text("#!/usr/bin/env bash\n")
        (destination / "scripts" / "deploy.py").write_text("\n")
        return destination

    monkeypatch.setattr(pipeline, "_runtime_image_provenance", lambda image: {"declared": image})
    monkeypatch.setattr(
        pipeline,
        "_read_json",
        lambda _uri: {"archive_sha256": "a" * 64},
    )
    monkeypatch.setattr(pipeline, "_restore_archive", lambda _manifest, _target: assets)
    monkeypatch.setattr(pipeline, "_copy_repository", fake_copy_repository)
    monkeypatch.setattr(
        pipeline,
        "_checkpoint_record",
        lambda _root: {"sha256": "b" * 64, "size_bytes": 7, "relative_checkpoint": "checkpoint_step_20.safetensors"},
    )
    monkeypatch.setattr(pipeline, "_command", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(pipeline, "_archive_tree", lambda _root, _archive: {"sha256": "c" * 64, "size_bytes": 9})
    monkeypatch.setattr(pipeline, "_upload_file", lambda _uri, _source: None)
    monkeypatch.setattr(pipeline, "_write_json", lambda uri, payload: persisted.update({uri: payload}))

    report = pipeline.fine_tune(
        argparse.Namespace(
            run_id="test",
            runtime_image=DIGEST_IMAGE,
            openwam_root="/opt/openwam",
            work_dir=str(workspace),
            prepared_assets_uri="prepared.json",
            checkpoint_archive_uri="trained.tar.gz",
            output_uri="training.json",
            gpu_count="1",
        )
    )

    assert commands == [
        [
            "bash",
            "scripts/train.sh",
            "dataloader=libero",
            f"training.finetune_ckpt_path={workspace / 'repo' / 'assets' / 'openwam_ckpt' / 'openwam_alpha' / 'OpenWAM-Alpha-Pretrain-Foundation-Model'}",
            "training.debug=true",
            f"training.output_path={workspace / 'training-output'}",
        ]
    ]
    assert report["mode"] == "upstream training.debug=true (20-step operational smoke)"
    assert persisted["training.json"] == report


def test_rollout_uses_the_separate_upstream_libero_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source = repo / "assets" / "libero_source"
    source.mkdir(parents=True)
    (repo / "benchmarks" / "libero").mkdir(parents=True)
    (repo / "benchmarks" / "libero" / "policy_config.yml").write_text("action_mode: eef\n")
    client = tmp_path / "libero-client-python"
    client.write_text("#!/bin/sh\n")
    client.chmod(0o755)
    result = tmp_path / "result"
    result.mkdir()
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setenv("LIBERO_PYTHON", str(client))
    monkeypatch.setattr(pipeline, "_apply_libero_patch", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_start_server", lambda *_args: object())
    monkeypatch.setattr(pipeline, "_wait_for_port", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_stop_server", lambda _process: "server stopped")

    def fake_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        del cwd
        calls.append((command, env))
        (result / "results.json").write_text(
            '{"trial_start": 1, "num_trials": 1, "success_rate": 1.0, "trials": [{"policy_steps": 12}]}'
        )

    monkeypatch.setattr(pipeline, "_command", fake_command)
    observed = pipeline._run_libero_trial(repo, tmp_path / "checkpoint", result, trial_start=1, port=8848)

    assert calls[0][0][0] == str(client)
    assert calls[0][0][1:3] == ["benchmarks/libero/single_eval.py", "--config"]
    assert calls[0][1]["LIBERO_PATH"] == str(source)
    assert str(source) in calls[0][1]["PYTHONPATH"]
    assert observed["success_rate"] == 1.0
    assert (result / "openwam-server.log").read_text() == "server stopped"


def test_visualize_writes_and_decodes_an_rrd_from_local_test_artifacts(tmp_path: Path) -> None:
    """Exercise the real Rerun writer with test-only numerical fixtures."""

    training_uri = str(tmp_path / "training.json")
    rollout_uri = str(tmp_path / "rollout.json")
    evaluation_uri = str(tmp_path / "evaluation.json")
    rrd_uri = str(tmp_path / "openwam-libero.rrd")
    output_uri = str(tmp_path / "visualization.json")
    Path(training_uri).write_text(json.dumps({"checkpoint": {"sha256": "a" * 64, "size_bytes": 123}}))
    Path(rollout_uri).write_text(
        json.dumps({"result": {"success_rate": 0.0, "trials": [{"policy_steps": 7}]}})
    )
    Path(evaluation_uri).write_text(
        json.dumps({"result": {"success_rate": 1.0, "trials": [{"policy_steps": 9}]}})
    )

    report = pipeline.visualize(
        argparse.Namespace(
            run_id="openwam-rrd-test",
            work_dir=str(tmp_path / "work"),
            training_uri=training_uri,
            rollout_uri=rollout_uri,
            evaluation_uri=evaluation_uri,
            rrd_uri=rrd_uri,
            output_uri=output_uri,
        )
    )

    assert Path(rrd_uri).is_file()
    assert report["verification"]["rerun_rrd_verify"] == "passed"
    assert report["heldout_success_rate"] == 1.0


def test_archive_extraction_rejects_escaping_members_and_links(tmp_path: Path) -> None:
    for name, link_name in (("../../escape", None), ("linked", "../../escape")):
        archive_path = tmp_path / f"{name.replace('/', '_')}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo(name)
            if link_name:
                info.type = tarfile.SYMTYPE
                info.linkname = link_name
                archive.addfile(info)
            else:
                payload = b"unsafe"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        with tarfile.open(archive_path, "r:gz") as archive:
            with pytest.raises(pipeline.OpenWAMPipelineError):
                pipeline._safe_extract(archive, tmp_path / "extract")

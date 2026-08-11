"""Public import compatibility for behavior extracted from the Sim2Real engine."""

from __future__ import annotations

from pathlib import Path

from npa.workflows.sim2real.models import Sim2RealLoopConfig


def test_engine_reexports_extracted_decision_and_state_io() -> None:
    from npa.workflows.sim2real import (
        artifact_upload,
        decision,
        engine,
        workflow_state_io,
    )

    assert engine.threshold_decision is decision.threshold_decision
    assert engine._workflow_state_path is workflow_state_io._workflow_state_path
    assert engine._read_workflow_state is workflow_state_io._read_workflow_state
    assert engine._write_workflow_state is workflow_state_io._write_workflow_state
    assert (
        engine.sync_workflow_state_to_s3 is workflow_state_io.sync_workflow_state_to_s3
    )
    assert (
        engine.emit_active_progress_rerun
        is workflow_state_io.emit_active_progress_rerun
    )
    assert engine._upload_final_report is artifact_upload._upload_final_report
    assert engine.upload_run_artifacts is artifact_upload.upload_run_artifacts


def test_extracted_state_and_upload_skip_contracts(tmp_path: Path) -> None:
    from npa.workflows.sim2real import artifact_upload, workflow_state_io

    config = Sim2RealLoopConfig(run_id="unit-extracted", output_dir=tmp_path)
    payload = {"run_id": config.run_id, "components": []}

    assert workflow_state_io._write_workflow_state(tmp_path, payload) == payload
    assert workflow_state_io._read_workflow_state(tmp_path) == payload
    assert workflow_state_io.sync_workflow_state_to_s3(config, tmp_path) is None
    assert artifact_upload.upload_run_artifacts(config, tmp_path)["status"] == "skipped"
    assert (
        artifact_upload._upload_final_report(
            config, tmp_path / "reports" / "sim2real-report.json"
        )["status"]
        == "skipped"
    )

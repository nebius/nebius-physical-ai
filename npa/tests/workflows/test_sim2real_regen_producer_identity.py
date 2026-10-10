"""Post-authority replay controls; fixture media is not a real pipeline run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import npa.workflows.sim2real_rerun_regen as regen
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.workflow_io import build_component_record
from npa.workflows.sim2real.component_authority import stage14_report_authority_sha256
from npa.workflows.sim2real.regeneration_authority import authority_sha256
from npa.workflows.sim2real_viz import Sim2RealVizResult

pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")

REPLAY_IMAGE = "ghcr.io/example/npa@sha256:" + "f" * 64
REPLAY_JOB = "fixture-replay-job"
SOURCE_SHA = "b" * 40
ROOT = "s3://demo-bucket/sim2real-b/run-a"
ORIGINAL_REPORT = f"{ROOT}/reports/generations/{'c' * 32}/sim2real-report.json"


def _config() -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id="run-a", s3_bucket="demo-bucket", s3_prefix="sim2real-b"
    )


def _original_component() -> dict[str, Any]:
    return build_component_record(
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence="Original fixture producer",
        artifacts={
            "rrd": ORIGINAL_REPORT.replace("-report.json", ".rrd"),
            "mcap": ORIGINAL_REPORT.replace("-report.json", ".mcap"),
            "report": ORIGINAL_REPORT,
        },
        execution_provenance={
            "image": "ghcr.io/example/npa@sha256:" + "c" * 64,
            "image_digest": "sha256:" + "c" * 64,
            "source_sha": SOURCE_SHA,
            "execution_mode": "standard_npa_workflow_skypilot",
            "workflow_job": "original-fixture-job",
            "gpu_rows": ["NVIDIA H100, GPU-original"],
            "gpu_products": ["NVIDIA H100"],
        },
    )


def _replay_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_TASK_IMAGE", REPLAY_IMAGE)
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("SKYPILOT_TASK_ID", REPLAY_JOB)
    monkeypatch.delenv("SKYPILOT_CLUSTER_NAME", raising=False)


def _original_report():
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "source_sha": SOURCE_SHA,
        "component_records": [_original_component()],
        "report_uri": ORIGINAL_REPORT,
    }
    component = report["component_records"][0]
    report["rrd_uri"] = component["artifacts"]["rrd"]
    report["mcap_uri"] = component["artifacts"]["mcap"]
    component["artifacts"]["report_authority_sha256"] = stage14_report_authority_sha256(
        report
    )
    component.pop("content_sha256")
    component["content_sha256"] = authority_sha256(component)
    return report


def _stub_recording_emission(monkeypatch, calls):
    def emit(_config: Any, _work: Path, output: Path, *_args: Any) -> tuple:
        calls.append("emit")
        output.write_bytes(b"explicit-unit-control-not-an-rrd")
        return (
            Sim2RealVizResult(
                status="written",
                output_rrd_path=str(output),
                rollout_count=1,
                frame_count=1,
                heldout_frame_count=1,
            ),
            0.1,
        )

    monkeypatch.setattr(regen, "_emit_regen_rrd", emit)
    monkeypatch.setattr(
        regen, "_finalize_regen_result", lambda *_a, **_kw: calls.append("finalize")
    )


def _post_authority_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[regen._RegenState, list[str], bytes]:
    """Isolate producer binding after the separately tested authority reader."""
    report = _original_report()
    report_path = tmp_path / "reports" / "sim2real-report.json"
    report_path.parent.mkdir()
    original = (json.dumps(report, sort_keys=True) + "\n").encode()
    report_path.write_bytes(original)
    state = regen._RegenState({}, {}, report_path, report, {})
    calls: list[str] = []
    monkeypatch.setattr(regen, "_load_regen_state", lambda *_a, **_kw: state)
    monkeypatch.setattr(regen, "_verify_regeneration_inputs", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        regen, "_capture_regen_publication_snapshots", lambda *_a, **_kw: {}
    )

    _stub_recording_emission(monkeypatch, calls)
    return state, calls, original


@pytest.mark.parametrize("missing", ["image", "job", "digest", "source_attestation"])
def test_canonical_upload_rejects_unattested_replay_before_emission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _replay_environment(monkeypatch)
    state, calls, original = _post_authority_state(tmp_path, monkeypatch)
    if missing == "digest":
        monkeypatch.setenv("NPA_TASK_IMAGE", "ghcr.io/example/npa@sha256:invalid")
    elif missing == "source_attestation":
        monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "e" * 40)
    else:
        monkeypatch.delenv(
            "NPA_TASK_IMAGE" if missing == "image" else "SKYPILOT_TASK_ID"
        )
    with pytest.raises(regen.Sim2RealRerunRegenError, match="replay producer"):
        regen.regen_sim2real_rrd(
            _config(),
            local_dir=tmp_path,
            sync_inputs=False,
            upload=True,
            client=object(),
        )
    assert calls == []
    assert state.report_path.read_bytes() == original
    assert not (tmp_path / "reports" / "sim2real.rrd").exists()


def test_canonical_upload_preserves_historical_input_with_a_new_writer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _replay_environment(monkeypatch)
    state, calls, original = _post_authority_state(tmp_path, monkeypatch)
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "e" * 40)
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "e" * 40)
    regen.regen_sim2real_rrd(
        _config(), local_dir=tmp_path, sync_inputs=False, upload=True, client=object()
    )
    assert calls == ["emit", "finalize"]
    assert state.report["source_sha"] == SOURCE_SHA
    authority = state.report["regeneration"]
    assert authority["input"]["source_sha"] == SOURCE_SHA
    assert authority["writer"]["source_sha"] == "e" * 40
    assert (
        authority["input"]["stage14_record"]
        == json.loads(original)["component_records"][0]
    )


def test_attested_replay_binds_current_image_job_and_original_record_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _replay_environment(monkeypatch)
    state, calls, _original = _post_authority_state(tmp_path, monkeypatch)
    original_component = copy.deepcopy(state.report["component_records"][0])
    regen.regen_sim2real_rrd(
        _config(), local_dir=tmp_path, sync_inputs=False, upload=True, client=object()
    )
    assert calls == ["emit", "finalize"]
    component = state.report["component_records"][0]
    assert component["artifacts"]["workflow_job"] == REPLAY_JOB
    assert component["artifacts"]["image"] == REPLAY_IMAGE
    assert component["artifacts"]["source_sha"] == SOURCE_SHA
    assert "gpu_rows" not in component["artifacts"]
    assert "gpu_products" not in component["artifacts"]
    assert (
        component["artifacts"]["regenerated_from_component_sha256"]
        == (original_component["content_sha256"])
    )
    assert component["content_sha256"] != original_component["content_sha256"]


def test_local_preview_preserves_canonical_report_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, calls, original = _post_authority_state(tmp_path, monkeypatch)
    original_report = copy.deepcopy(state.report)
    monkeypatch.delenv("NPA_TASK_IMAGE", raising=False)
    regen.regen_sim2real_rrd(
        _config(), local_dir=tmp_path, sync_inputs=False, upload=False, client=object()
    )
    assert calls == ["emit", "finalize"]
    assert state.report == original_report
    assert state.report_path.read_bytes() == original

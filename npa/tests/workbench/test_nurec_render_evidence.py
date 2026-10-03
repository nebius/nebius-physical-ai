"""Reject mismatched digital-twin media and keep portable GPU evidence private."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from npa.workbench.nurec import render_evidence as evidence
from npa.workbench.nurec.nurec import NurecConfig, NurecError, render_novel_views
from npa.workbench.nurec.preview import write_nurec_preview


def _run_tree(root: Path) -> tuple[Path, Path]:
    artifact = root / "reconstruction" / "last.usdz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"scene fixture")
    for name, color in (("input", "blue"), ("novel_views", "green")):
        directory = root / name
        directory.mkdir()
        Image.new("RGB", (48, 32), color).save(directory / "frame.png")
    return artifact, root / "novel_views"


def _record(root: Path) -> dict:
    artifact, output = _run_tree(root)
    telemetry = evidence.RenderTelemetry()
    telemetry.samples = [
        {
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "driver": "580.173.02",
            "utilization_percent": 78,
            "memory_mib": 4200,
        }
    ]
    telemetry.elapsed_seconds = 2.4
    return evidence.write_render_evidence(
        output, artifact, telemetry, renderer="default", novel_view=True
    )


def test_html_binds_real_media_and_drops_untrusted_metadata(tmp_path):
    record = _record(tmp_path)
    record["private_endpoint"] = "https://private.example.invalid"
    record["telemetry"]["hostname"] = "private-hostname"
    (tmp_path / "novel_views" / evidence.EVIDENCE_FILENAME).write_text(
        json.dumps(record)
    )
    output = tmp_path / "twin.html"
    write_nurec_preview(tmp_path, output)
    html = output.read_text()
    assert "Media hashes verified; GPU activity observed" in html
    assert "RTX PRO 6000" in html
    assert "scene SHA-256" in html
    assert "private.example.invalid" not in html
    assert "private-hostname" not in html
    assert "connect-src 'none'" in html


@pytest.mark.parametrize("changed", ["scene", "frame", "extra-frame"])
def test_changed_media_cannot_reuse_gpu_evidence(tmp_path, changed):
    _record(tmp_path)
    if changed == "scene":
        (tmp_path / "reconstruction" / "last.usdz").write_bytes(b"different scene")
    else:
        filename = "frame.png" if changed == "frame" else "other.png"
        Image.new("RGB", (48, 32), "red").save(tmp_path / "novel_views" / filename)
    with pytest.raises(ValueError, match="mismatch"):
        write_nurec_preview(tmp_path, tmp_path / "twin.html")
    assert not (tmp_path / "twin.html").exists()


def test_missing_telemetry_is_unverified_not_a_gpu_claim(tmp_path):
    artifact, output = _run_tree(tmp_path)
    evidence.write_render_evidence(
        output,
        artifact,
        evidence.RenderTelemetry(),
        renderer="default",
        novel_view=True,
    )
    summary = evidence.verified_render_summary(tmp_path)
    assert summary["render GPU"] == "Unverified"
    assert "GPU activity unverified" in summary["render evidence"]


def test_failed_renderer_cannot_publish_success_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "_gpu_snapshot", lambda: [])
    artifact = tmp_path / "scene.usdz"
    artifact.write_bytes(b"scene fixture")
    result = render_novel_views(
        NurecConfig.from_env(environ={}),
        artifact_path=str(artifact),
        output_dir=str(tmp_path / "renders"),
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "failed"
        ),
    )
    assert not result.ok
    assert not (tmp_path / "renders" / evidence.EVIDENCE_FILENAME).exists()


def test_stale_frames_refuse_a_new_render_before_execution(tmp_path):
    artifact, output = _run_tree(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("renderer executed over pre-existing media")

    with pytest.raises(NurecError, match="already contains frames"):
        render_novel_views(
            NurecConfig.from_env(environ={}),
            artifact_path=str(artifact),
            output_dir=str(output),
            runner=unexpected,
        )


def test_driver_probe_failure_does_not_fabricate_samples(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(evidence.subprocess, "run", missing)
    with evidence.RenderTelemetry() as telemetry:
        pass
    assert telemetry.summary()["sample_count"] == 0


def test_non_rt_gpu_telemetry_does_not_claim_rt_rendering(tmp_path):
    record = _record(tmp_path)
    record["telemetry"]["gpu_models"] = ["NVIDIA B200"]
    (tmp_path / "novel_views" / evidence.EVIDENCE_FILENAME).write_text(
        json.dumps(record)
    )
    assert evidence.verified_render_summary(tmp_path)["render GPU"] == "Unverified"


def test_training_camera_replay_is_not_labeled_a_novel_view(tmp_path):
    record = _record(tmp_path)
    record["novel_view"] = False
    (tmp_path / "novel_views" / evidence.EVIDENCE_FILENAME).write_text(
        json.dumps(record)
    )
    output = tmp_path / "twin.html"
    write_nurec_preview(tmp_path, output)
    html = output.read_text()
    assert "Training views" in html
    assert "nonzero rig offset" not in html

"""Reject mismatched digital-twin media and keep portable GPU evidence private."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from npa.workbench.nurec import render_evidence as evidence
from npa.workbench.nurec.nurec import NurecConfig, render_novel_views
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
    generation = Path(result.output_dir)
    assert not (generation / evidence.EVIDENCE_FILENAME).exists()
    receipt = json.loads((generation / "nre-render.json").read_text())
    assert receipt["status"] == "failed"


@pytest.mark.parametrize("suffix", ["png", "webp"])
def test_native_render_retains_decoded_receipt_and_portable_media_evidence(
    tmp_path, monkeypatch, suffix
):
    import imageio_ffmpeg

    monkeypatch.setattr(evidence, "_gpu_snapshot", lambda: [])
    artifact = tmp_path / "scene.usdz"
    artifact.write_bytes(b"scene fixture")

    def native(command, **kwargs):
        generation = Path(command[command.index("--output-dir") + 1])
        frames = []
        for index in range(2):
            frame = Image.new("RGB", (16, 16))
            frame.putdata(
                [(x * 16, y * 16, index * 100) for y in range(16) for x in range(16)]
            )
            frame.save(generation / f"{index:06d}.{suffix}")
            frames.append(frame.tobytes())
        writer = imageio_ffmpeg.write_frames(
            str(generation / "novel.mp4"), (16, 16), fps=2, codec="libx264"
        )
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        return subprocess.CompletedProcess(command, 0, "", "")

    result = render_novel_views(
        NurecConfig.from_env(environ={}),
        artifact_path=str(artifact),
        output_dir=str(tmp_path / "renders"),
        image_format=suffix,
        runner=native,
    )
    assert result.ok
    generation = Path(result.output_dir)
    receipt = json.loads(Path(result.evidence_path).read_text())
    portable = json.loads((generation / evidence.EVIDENCE_FILENAME).read_text())
    assert receipt["output"]["all_frames_decoded"] is True
    assert receipt["output"]["decoded_video_frames"] == 2
    assert portable["frame_count"] == result.frame_count == 2
    assert (
        portable["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert portable["telemetry"]["sample_count"] == 0
    assert result.gpu_names == ()


def test_portable_evidence_write_failure_preserves_decoded_receipt(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evidence, "_gpu_snapshot", lambda: [])
    artifact, output = _run_tree(tmp_path)

    def native(command, **kwargs):
        generation = Path(command[command.index("--output-dir") + 1])
        Image.new("RGB", (48, 32), "red").save(generation / "new.png")
        return subprocess.CompletedProcess(command, 0, "", "")

    def failed_write(root, *args, **kwargs):
        receipt = json.loads((root / "nre-render.json").read_text())
        assert receipt["output"]["all_frames_decoded"] is True
        raise OSError("evidence write refused")

    monkeypatch.setattr(evidence, "write_render_evidence", failed_write)
    with pytest.raises(OSError, match="evidence write refused"):
        render_novel_views(
            NurecConfig.from_env(environ={}),
            artifact_path=str(artifact),
            output_dir=str(output),
            runner=native,
        )


def test_new_generation_evidence_excludes_and_preserves_stale_frames(tmp_path):
    artifact, output = _run_tree(tmp_path)
    previous = (output / "frame.png").read_bytes()

    def native(command, **kwargs):
        generation = Path(command[command.index("--output-dir") + 1])
        assert generation != output
        Image.new("RGB", (48, 32), "red").save(generation / "new.png")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = render_novel_views(
        NurecConfig.from_env(environ={}),
        artifact_path=str(artifact),
        output_dir=str(output),
        runner=native,
    )
    assert result.ok
    generation = Path(result.output_dir)
    record = json.loads((generation / evidence.EVIDENCE_FILENAME).read_text())
    assert record["frame_count"] == 1
    assert record["frames_sha256"] == evidence._media_digest([generation / "new.png"])
    assert (output / "frame.png").read_bytes() == previous
    assert not (output / evidence.EVIDENCE_FILENAME).exists()


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


@pytest.mark.parametrize(
    "model, expected",
    [
        ("NVIDIA RTX A6000", "RT-capable GPU"),
        ("NVIDIA Quadro RTX 6000", "RT-capable GPU"),
        ("NVIDIA RTX 6000 Ada Generation", "RT-capable GPU"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000"),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000"),
        ("NVIDIA RTX PRO 60000", "RT-capable GPU"),
    ],
)
def test_gpu_family_label_matches_observed_model(tmp_path, model, expected):
    record = _record(tmp_path)
    record["telemetry"]["gpu_models"] = [model]
    (tmp_path / "novel_views" / evidence.EVIDENCE_FILENAME).write_text(
        json.dumps(record)
    )
    assert evidence.verified_render_summary(tmp_path)["render GPU"] == expected
    output = tmp_path / "twin.html"
    write_nurec_preview(tmp_path, output)
    html = output.read_text()
    assert expected in html
    if expected != "RTX PRO 6000":
        assert "RTX PRO 6000" not in html


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

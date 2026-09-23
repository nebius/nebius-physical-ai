"""Reject claimed NuRec success unless native receipts and decoded outputs agree."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml


# torch.save({'state_dict': {'positions': tensor([0., 1., 2.])}}), protocol 2.
# The verifier inspects these opcodes without executing the pickle.
_TENSOR_METADATA = (
    b"\x80\x02}q\x00X\n\x00\x00\x00state_dictq\x01}q\x02X\t\x00\x00\x00positionsq\x03"
    b"ctorch._utils\n_rebuild_tensor_v2\nq\x04((X\x07\x00\x00\x00storageq\x05"
    b"ctorch\nFloatStorage\nq\x06X\x01\x00\x00\x000q\x07X\x03\x00\x00\x00cpuq\x08"
    b"K\x03tq\tQK\x00K\x03\x85q\nK\x01\x85q\x0b\x89ccollections\nOrderedDict\n"
    b"q\x0c)Rq\rtq\x0eRq\x0fss."
)


@pytest.fixture
def verifier():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/specialists/workflows/nurec_verify.py"
    )
    spec = importlib.util.spec_from_file_location("nurec_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal(directory):
    raw = directory / "raw-status.json"
    _json(raw, {"run_id": "test-run", "status": "SUCCEEDED"})
    path = directory / "terminal.json"
    _json(
        path,
        {
            "source": "workbench-live-status",
            "terminal": True,
            "status": "succeeded",
            "run_id": "test-run",
            "observed_at": "2026-09-22T12:00:00+00:00",
            "submitted_spec_sha256": "a" * 64,
            "raw_status_path": raw.name,
            "raw_status_sha256": _hash(raw),
            "stages": [
                {"name": name, "status": "succeeded"}
                for name in (
                    "check",
                    "fetch",
                    "reconstruct",
                    "render",
                    "visualize",
                    "finalize",
                )
            ],
        },
    )
    return path


def _package(root, directory):
    Usd = pytest.importorskip("pxr.Usd")
    Sdf = pytest.importorskip("pxr.Sdf")
    layer = directory / "scene.usda"
    stage = Usd.Stage.CreateNew(str(layer))
    stage.DefinePrim("/Scene", "Xform")
    stage.GetRootLayer().Save()
    checkpoint = directory / "checkpoint.ckpt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", _TENSOR_METADATA)
        archive.writestr("archive/data/0", struct.pack("fff", 0.0, 1.0, 2.0))
        archive.writestr("archive/version", "3\n")
    target = root / "reconstruction/last.usdz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with Sdf.ZipFileWriter.CreateNew(str(target)) as archive:
        archive.AddFile(str(layer), "scene.usda")
        archive.AddFile(str(checkpoint), "checkpoint.ckpt")


def _media(root):
    image = pytest.importorskip("PIL.Image")
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    path = root / "novel_views/camera/000000.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.new("RGB", (16, 16), (45, 120, 200)).save(path)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(path),
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(root / "novel_views/camera.mp4"),
        ],
        check=True,
        capture_output=True,
    )


def _recording(root, *, strings_only=False, wrong_frame=False, legacy=False):
    rr = pytest.importorskip("rerun")
    pytest.importorskip("rerun.chunk")
    image = pytest.importorskip("PIL.Image")
    recording = rr.RecordingStream("neural-reconstruction", recording_id="test-run")
    recording.save(str(root / "reports/sim2real.rrd"))
    settings = {
        "schema": "npa.nurec.rrd-review.v1",
        "max_frames_per_entity": 24,
        "max_frame_dim": 512,
        "jpeg_quality": 75,
    }
    metrics = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    for entity, payload in (
        ("provenance/rrd_review", settings),
        ("gaussians/summary", metrics),
    ):
        if legacy and entity == "provenance/rrd_review":
            continue
        recording.log(
            entity, rr.TextDocument(f"```json\n{json.dumps(payload)}\n```"), static=True
        )
    recording.set_time("frame", sequence=0)
    if strings_only:
        recording.log("novel_view/camera", rr.TextDocument("render succeeded"))
    else:
        with image.open(root / "novel_views/camera/000000.png") as source:
            rgb = source.convert("RGB")
        if wrong_frame:
            rgb = image.new("RGB", (16, 16), (0, 0, 0))
        encoded = io.BytesIO()
        rgb.save(encoded, format="JPEG", quality=75)
        recording.log(
            "novel_view/camera",
            rr.EncodedImage(contents=encoded.getvalue(), media_type="image/jpeg"),
        )
    recording.flush()


def _receipt():
    return {
        "status": "ok",
        "novel_view": True,
        "frame_count": 1,
        "video_count": 1,
        "command": [
            "/app/run",
            "render",
            "--no-replicate-training-views",
            "--rig-translation-offset",
            "0",
            "0.25",
            "0",
            "--rig-rotation-offset",
            "0",
            "0",
            "0",
        ],
    }


def _render_evidence(root, directory):
    receipt = _receipt()
    raw = directory / "raw-render.json"
    _json(raw, receipt)
    path = directory / "render.json"
    files = [item for item in (root / "novel_views").rglob("*") if item.is_file()]
    _json(
        path,
        {
            "source": "workbench-render-receipt",
            "run_id": "test-run",
            "submitted_spec_sha256": "a" * 64,
            "receipt": receipt,
            "raw_receipt_path": raw.name,
            "raw_receipt_sha256": _hash(raw),
            "artifact_sha256": _hash(root / "reconstruction/last.usdz"),
            "media_sha256": {
                str(item.relative_to(root)): _hash(item) for item in files
            },
        },
    )
    return path


def _manifest(root):
    _json(
        root / "ncore/manifest.json",
        {
            "status": "ok",
            "scene": "scene-a",
            "variant": "auto",
            "observed_scene": "scene-a",
            "observed_variant": "auto",
            "dataset_id": "fixture/capture",
            "shard_count": 2,
            "camera_ids": ["camera"],
        },
    )


@pytest.fixture
def run_tree(tmp_path):
    root = tmp_path / "download"
    _package(root, tmp_path)
    _media(root)
    _manifest(root)
    (root / "reconstruction/metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "aggregated_metrics": {
                    "test/psnr": {"value": 25.0},
                    "test/ssim": {"value": 0.85},
                    "test/lpips": {"value": 0.15},
                }
            }
        )
    )
    _json(
        root / "reports/final.json",
        {
            "status": "ok",
            "run_id": "test-run",
            "capability": "neural-reconstruction",
            "has_usdz": True,
            "has_rrd": True,
            "has_novel_views": True,
            "errors": [],
        },
    )
    _recording(root)
    return root, _terminal(tmp_path), _render_evidence(root, tmp_path)


def _verify(verifier, run_tree, **changes):
    root, terminal, render = run_tree
    options = {
        "scene": "scene-a",
        "variant": "auto",
        "novel_offsets": (0.0, 0.25, 0.0),
        "terminal_evidence": terminal,
        "render_evidence": render,
    }
    options.update(changes)
    return verifier.verify_run(root, **options)


def test_real_decoders_validate_fixture_and_return_only_sanitized_outcome(
    verifier, run_tree
):
    result = _verify(verifier, run_tree)
    assert result["passed"], result["errors"]
    assert result["offsets_verified"]
    assert result["checks"]["media"]["decoded_video_frames"] == 2
    assert result["checks"]["rrd"]["verified_image_rows"] == 1
    assert result["checks"]["usdz"]["tensor_storage_count"] == 1
    assert "test-run" not in json.dumps(result)
    assert str(run_tree[0].parent) not in json.dumps(result)


@pytest.mark.parametrize("state", ["running", "failed", "cancelled"])
def test_nonterminal_or_failed_workflow_cannot_pass_even_with_final_report(
    verifier, tmp_path, state
):
    terminal = _terminal(tmp_path)
    payload = json.loads(terminal.read_text())
    payload["status"] = state
    payload["terminal"] = state != "running"
    _json(terminal, payload)
    _json(tmp_path / "reports/final.json", {"status": "ok", "has_usdz": True})
    result = verifier.verify_run(
        tmp_path,
        scene="scene-a",
        variant="auto",
        novel_offsets=(0, 0.25, 0),
        terminal_evidence=terminal,
    )
    assert not result["passed"]
    assert result["errors"][0]["check"] == "workflow"


@pytest.mark.parametrize("defect", ["stage", "duplicate", "source", "raw_bytes"])
def test_terminal_requires_all_native_stages_and_unchanged_raw_evidence(
    verifier, tmp_path, defect
):
    terminal = _terminal(tmp_path)
    payload = json.loads(terminal.read_text())
    if defect == "stage":
        payload["stages"][2]["status"] = "failed"
    elif defect == "duplicate":
        payload["stages"][2] = payload["stages"][0]
    elif defect == "source":
        payload["source"] = "agent-claims-success"
    else:
        (tmp_path / "raw-status.json").write_text("changed")
    _json(terminal, payload)
    with pytest.raises(ValueError):
        verifier._workflow(terminal)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("psnr", None),
        ("ssim", float("nan")),
        ("lpips", float("inf")),
        ("ssim", True),
        ("lpips", -1),
    ],
)
def test_missing_nonfinite_or_invalid_metrics_are_rejected(
    verifier, tmp_path, metric, value
):
    metrics = {"psnr": 25.0, "ssim": 0.85, "lpips": 0.15}
    metrics[metric] = value
    path = tmp_path / "reconstruction/metrics.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump({"test": metrics}))
    with pytest.raises(ValueError):
        verifier._metrics(tmp_path)


@pytest.mark.parametrize(
    "artifact",
    [
        "reconstruction/last.usdz",
        "novel_views/camera/000000.png",
        "novel_views/camera.mp4",
        "reports/sim2real.rrd",
    ],
)
def test_fabricated_or_corrupt_artifacts_cannot_pass(verifier, run_tree, artifact):
    (run_tree[0] / artifact).write_bytes(
        b"success=true; this is not the promised artifact"
    )
    result = _verify(verifier, run_tree)
    assert not result["passed"]
    assert result["errors"]


@pytest.mark.parametrize("defect", ["strings", "other_run_image"])
def test_rrd_requires_actual_matching_images_not_success_text(
    verifier, run_tree, defect
):
    _recording(
        run_tree[0],
        strings_only=defect == "strings",
        wrong_frame=defect == "other_run_image",
    )
    result = _verify(verifier, run_tree)
    assert not result["passed"]
    assert any(error["check"] == "rrd" for error in result["errors"])


def test_novelty_is_unverified_without_render_receipt(verifier, run_tree):
    result = _verify(verifier, run_tree, render_evidence=None)
    assert not result["passed"] and not result["offsets_verified"]
    assert {"check": "render", "reason": "missing_render_receipt"} in result["errors"]


def test_wrong_requested_offset_is_rejected(verifier, run_tree):
    result = _verify(verifier, run_tree, novel_offsets=(0, -0.25, 0))
    assert not result["passed"] and not result["offsets_verified"]
    assert {"check": "render", "reason": "wrong_novel_offset"} in result["errors"]


def test_wrong_capture_and_final_claims_are_rejected(verifier, run_tree):
    final = run_tree[0] / "reports/final.json"
    payload = json.loads(final.read_text())
    payload["has_rrd"] = False
    _json(final, payload)
    result = _verify(verifier, run_tree, scene="other-scene")
    assert not result["passed"]
    assert {error["check"] for error in result["errors"]} == {"provenance", "final"}


def test_decoder_exceptions_do_not_expose_private_paths(
    verifier, run_tree, monkeypatch
):
    def fail(_root):
        raise RuntimeError(
            "provider error at s3://private-fixture-bucket/private-prefix"
        )

    monkeypatch.setattr(verifier, "_usdz", fail)
    result = _verify(verifier, run_tree)
    assert not result["passed"]
    assert "private-fixture" not in json.dumps(result)


@pytest.mark.parametrize("payload", [b"not pickle", b"\x80\x02}", b"\x80\x02}."])
def test_checkpoint_requires_parseable_tensor_metadata(verifier, payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        verifier._checkpoint_metadata(archive, archive.namelist())


def test_edited_normalized_receipt_cannot_borrow_unchanged_raw_receipt(
    verifier, run_tree
):
    payload = json.loads(run_tree[2].read_text())
    payload["receipt"]["command"][5] = "-0.25"
    _json(run_tree[2], payload)
    result = _verify(verifier, run_tree, novel_offsets=(0, -0.25, 0))
    assert not result["passed"]
    assert {"check": "render", "reason": "render_receipt_differs_from_raw"} in result[
        "errors"
    ]


@pytest.mark.parametrize("sizes", [(12, 0, 0), (0, 0), ()])
def test_checkpoint_allows_optional_empty_storage_but_requires_model_data(
    verifier, sizes
):
    checkpoint = io.BytesIO()
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", _TENSOR_METADATA)
        archive.writestr("archive/version", "3\n")
        for index, size in enumerate(sizes):
            archive.writestr(f"archive/data/{index}", bytes(size))
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("checkpoint.ckpt", checkpoint.getvalue())
    package.seek(0)
    with zipfile.ZipFile(package) as archive:
        if not any(sizes):
            with pytest.raises(ValueError, match="no_checkpoint_tensors"):
                verifier._checkpoint(archive, archive.infolist())
            return
        result = verifier._checkpoint(archive, archive.infolist())
    assert result["tensor_storage_count"] == 3
    assert result["empty_tensor_storage_count"] == 2


def _legacy_review(run_tree):
    root, terminal, _ = run_tree
    producer = {
        "source": "verified-viewer-producer",
        "producer_image": "example.invalid/viewer@sha256:" + "b" * 64,
        "producer_source_sha256": "c" * 64,
        "settings": {
            "schema": "npa.nurec.rrd-review.v1",
            "max_frames_per_entity": 24,
            "max_frame_dim": 512,
            "jpeg_quality": 75,
        },
    }
    raw = terminal.parent / "raw-producer.json"
    _json(raw, producer)
    evidence = terminal.parent / "legacy-review.json"
    _json(
        evidence,
        {
            **producer,
            "source": "workbench-pinned-viewer",
            "run_id": "test-run",
            "submitted_spec_sha256": "a" * 64,
            "rrd_sha256": _hash(root / "reports/sim2real.rrd"),
            "raw_producer_path": raw.name,
            "raw_producer_sha256": _hash(raw),
        },
    )
    return evidence


def test_legacy_rrd_requires_explicit_producer_evidence(verifier, run_tree):
    _recording(run_tree[0], legacy=True)
    rejected = _verify(verifier, run_tree)
    assert {"check": "rrd", "reason": "missing_or_duplicate_rrd_document"} in rejected[
        "errors"
    ]
    accepted = _verify(
        verifier, run_tree, legacy_rrd_review_evidence=_legacy_review(run_tree)
    )
    assert accepted["passed"], accepted["errors"]
    assert accepted["checks"]["rrd"]["verified_image_rows"] == 1
    assert (
        accepted["checks"]["rrd"]["review_settings_source"]
        == "independent_pinned_producer"
    )


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "submitted_spec_sha256",
        "rrd_sha256",
        "producer_image",
        "producer_source_sha256",
        "raw_producer_sha256",
        "settings",
    ],
)
def test_legacy_rrd_evidence_cannot_be_rebound_or_edited(verifier, run_tree, field):
    _recording(run_tree[0], legacy=True)
    evidence = _legacy_review(run_tree)
    payload = json.loads(evidence.read_text())
    payload[field] = {} if field == "settings" else "changed"
    _json(evidence, payload)
    result = _verify(verifier, run_tree, legacy_rrd_review_evidence=evidence)
    assert not result["passed"]
    assert any(error["check"] == "rrd" for error in result["errors"])


@pytest.mark.parametrize("defect", ["wrong_image", "producer_bytes", "wrong_encoding"])
def test_legacy_evidence_preserves_image_and_raw_producer_checks(
    verifier, run_tree, defect
):
    _recording(run_tree[0], legacy=True, wrong_frame=defect == "wrong_image")
    evidence = _legacy_review(run_tree)
    payload = json.loads(evidence.read_text())
    raw = evidence.parent / payload["raw_producer_path"]
    if defect == "producer_bytes":
        raw.write_text("changed")
    elif defect == "wrong_encoding":
        producer = json.loads(raw.read_text())
        producer["settings"]["jpeg_quality"] = 10
        _json(raw, producer)
        payload.update(settings=producer["settings"], raw_producer_sha256=_hash(raw))
        _json(evidence, payload)
    result = _verify(verifier, run_tree, legacy_rrd_review_evidence=evidence)
    assert not result["passed"]
    assert any(error["check"] == "rrd" for error in result["errors"])


def test_external_settings_cannot_override_current_recording(verifier, run_tree):
    result = _verify(
        verifier, run_tree, legacy_rrd_review_evidence=_legacy_review(run_tree)
    )
    assert not result["passed"]
    assert {
        "check": "rrd",
        "reason": "legacy_evidence_for_current_recording",
    } in result["errors"]

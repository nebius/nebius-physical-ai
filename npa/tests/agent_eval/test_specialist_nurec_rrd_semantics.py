"""Check RRD source semantics independently of the producing viewer helpers."""

import hashlib
import importlib.util
import io
import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def verifier():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/specialists/workflows/nurec_verify.py"
    )
    spec = importlib.util.spec_from_file_location("rrd_semantic_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_images(root):
    image = pytest.importorskip("PIL.Image")
    sources = {}
    for modality_index, modality in enumerate(
        ("pred_rgb", "pred_distance", "pred_opacity")
    ):
        for camera, frames in (
            ("front", (7, 13, 29, 41, 67)),
            ("rear", (107, 113, 131, 173, 191, 211)),
        ):
            for index, frame in enumerate(frames):
                path = (
                    root / "reconstruction/val" / modality / camera / f"{frame:06d}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                image.new(
                    "RGB", (16, 16), (17 + modality_index * 61, frame, 29 + index * 23)
                ).save(path)
                sources[f"/reconstruction/val/{modality}/{camera}", frame] = path
    for frame in (5, 11, 25, 50, 99):
        path = root / "novel_views/lens" / f"{frame:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.new("RGB", (16, 16), (frame, 61, 197)).save(path)
        sources["/novel_view/lens", frame] = path
    (root / "reconstruction/metrics.yaml").write_text(
        yaml.safe_dump({"test": {"psnr": 23.0, "ssim": 0.8, "lpips": 0.2}})
    )
    return sources


def _selected(frames, cap, pattern):
    if cap <= 0 or len(frames) <= cap:
        return frames
    if pattern == "legacy":
        return [frames[int(index * len(frames) / cap)] for index in range(cap)]
    if cap == 1:
        return frames[:1]
    if pattern == "late":
        return frames[:1] + frames[-(cap - 1) :]
    return frames[: cap - 1] + frames[-1:]


def _encoded(path, wrong=False):
    image = pytest.importorskip("PIL.Image")
    with image.open(path) as source:
        rgb = source.convert("RGB")
    if wrong:
        rgb = image.new("RGB", (16, 16), (0, 0, 0))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()


def _log_images(recording, sources, cap, pattern, defect):
    rr = pytest.importorskip("rerun")
    groups = defaultdict(list)
    for entity, frame in sources:
        groups[entity].append(frame)
    for entity, frames in groups.items():
        chosen = _selected(sorted(frames), cap, pattern)
        if defect == "missing_last":
            chosen = sorted(frames)[:cap]
        if defect == "missing_first":
            chosen = sorted(frames)[-cap:]
        if defect == "missing_modality" and "/pred_distance/" in entity:
            continue
        target = (
            "/reconstruction/" + entity.rsplit("/", 1)[-1]
            if defect == "merged" and entity.startswith("/reconstruction/")
            else entity
        )
        for ordinal, frame in enumerate(chosen):
            reset = defect == "reset_ids" and entity.startswith("/reconstruction/")
            recording.set_time("frame", sequence=ordinal if reset else frame)
            contents = _encoded(
                sources[entity, frame],
                defect == "wrong_reconstruction_image"
                and entity.startswith("/reconstruction/"),
            )
            recording.log(
                target, rr.EncodedImage(contents=contents, media_type="image/jpeg")
            )
            if defect == "duplicate" and entity.startswith("/reconstruction/"):
                recording.log(
                    target, rr.EncodedImage(contents=contents, media_type="image/jpeg")
                )


def _recording(root, sources, cap, *, pattern="early", defect="", legacy=False):
    rr = pytest.importorskip("rerun")
    pytest.importorskip("rerun.chunk")
    path = root / "reports/sim2real.rrd"
    path.parent.mkdir()
    recording = rr.RecordingStream("neural-reconstruction", recording_id="rrd-contract")
    recording.save(str(path))
    settings = {
        "schema": "npa.nurec.rrd-review.v1",
        "max_frames_per_entity": cap,
        "max_frame_dim": 0,
        "jpeg_quality": 75,
    }
    metrics = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    for entity, document in (
        ("provenance/rrd_review", settings),
        ("gaussians/summary", metrics),
    ):
        if legacy and entity == "provenance/rrd_review":
            continue
        recording.log(
            entity,
            rr.TextDocument(f"```json\n{json.dumps(document)}\n```"),
            static=True,
        )
    _log_images(recording, sources, cap, pattern, defect)
    recording.flush()
    recording.disconnect()
    return settings


def _legacy_evidence(root, settings):
    raw = root / "producer.json"
    producer = {
        "source": "verified-viewer-producer",
        "producer_image": "example.invalid/viewer@sha256:" + "a" * 64,
        "producer_source_sha256": "b" * 64,
        "settings": settings,
    }
    _write(raw, producer)
    evidence = root / "legacy-review.json"
    _write(
        evidence,
        {
            **producer,
            "source": "workbench-pinned-viewer",
            "run_id": "rrd-contract",
            "submitted_spec_sha256": "c" * 64,
            "rrd_sha256": _sha(root / "reports/sim2real.rrd"),
            "raw_producer_path": raw.name,
            "raw_producer_sha256": _sha(raw),
        },
    )
    return evidence


def _terminal():
    return {"run_id": "rrd-contract", "submitted_spec_sha256": "c" * 64}


@pytest.mark.parametrize("cap", [0, -1, 1, 2, 3, 24])
@pytest.mark.parametrize("pattern", ["early", "late"])
def test_corrected_rrd_accepts_endpoint_preserving_interior_choices(
    verifier, tmp_path, cap, pattern
):
    sources = _source_images(tmp_path)
    _recording(tmp_path, sources, cap, pattern=pattern)
    result = verifier._rrd(tmp_path, _terminal())
    assert result["image_verification_scope"] == "novel_views_and_reconstruction"
    expected = (
        len(sources)
        if cap <= 0
        else sum(min(count, cap) for count in (5, 6, 5, 6, 5, 6, 5))
    )
    assert result["verified_image_rows"] == expected
    assert result["verified_reconstruction_image_rows"] > 0
    assert result["review_settings_source"] == "embedded"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_last",
        "missing_first",
        "merged",
        "reset_ids",
        "wrong_reconstruction_image",
        "missing_modality",
        "duplicate",
    ],
)
def test_corrected_rrd_rejects_lost_source_semantics(verifier, tmp_path, defect):
    sources = _source_images(tmp_path)
    cap = 3 if defect in {"missing_last", "missing_first"} else 0
    _recording(tmp_path, sources, cap, defect=defect)
    with pytest.raises(ValueError):
        verifier._rrd(tmp_path, _terminal())


def test_embedded_recording_cannot_use_old_endpoint_omission(verifier, tmp_path):
    sources = _source_images(tmp_path)
    _recording(tmp_path, sources, 3, pattern="legacy")
    with pytest.raises(ValueError, match="rrd_missing_last_frame"):
        verifier._rrd(tmp_path, _terminal())


def test_producer_bound_legacy_retains_explicit_novel_only_scope(verifier, tmp_path):
    sources = _source_images(tmp_path)
    settings = _recording(
        tmp_path, sources, 3, pattern="legacy", defect="merged", legacy=True
    )
    with pytest.raises(ValueError, match="missing_or_duplicate_rrd_document"):
        verifier._rrd(tmp_path, _terminal())
    result = verifier._rrd(tmp_path, _terminal(), _legacy_evidence(tmp_path, settings))
    assert result["image_verification_scope"] == "legacy_novel_views_only"
    assert result["verified_image_rows"] == result["verified_novel_image_rows"] == 3
    assert result["verified_reconstruction_image_rows"] == 0


def test_legacy_checks_do_not_relax_historical_image_selection(verifier, tmp_path):
    sources = _source_images(tmp_path)
    settings = _recording(tmp_path, sources, 3, legacy=True)
    with pytest.raises(ValueError, match="rrd_image_differs_from_run"):
        verifier._rrd(tmp_path, _terminal(), _legacy_evidence(tmp_path, settings))


def test_producer_evidence_cannot_downgrade_embedded_recording(verifier, tmp_path):
    sources = _source_images(tmp_path)
    settings = _recording(tmp_path, sources, 3, pattern="legacy", defect="merged")
    with pytest.raises(ValueError, match="legacy_evidence_for_current_recording"):
        verifier._rrd(tmp_path, _terminal(), _legacy_evidence(tmp_path, settings))

"""Synthetic unit fixtures prove conversion/readback, not PAIDF live acceptance."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import stat
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import pytest
from rerun.recording import load_recording
import yaml

from npa.workflows.paidf_evidence_viz import (
    APPLICATION_ID,
    EVIDENCE_SCHEMA,
    IMAGE_NAMES,
    PaidfEvidenceError,
    build_image_evidence_rrd,
)


def _evidence(media: Path, media_type: str = "image") -> dict:
    return {
        "schema": EVIDENCE_SCHEMA,
        "image_name": "npa-paidf-image-edit-sky",
        "image_digest": "sha256:" + "a" * 64,
        "run_id": "synthetic-unit-run",
        "source_revisions": ["b" * 40],
        "image_build_source_revision": "b" * 40,
        "runtime_source_revisions": ["b" * 40],
        "upstream_sources": [{
            "repository": "https://github.com/NVIDIA-AI-Blueprints/physical-ai-data-factory",
            "revision": "c" * 40, "license": "Apache-2.0",
            "adaptation": "NPA workflow adapters and factual Rerun conversion",
        }],
        "validation": {"status": "passed", "checks": [{"name": "output_decode", "status": "passed"}]},
        "stages": [
            {"state": "generate", "source_revision": "b" * 40, "status": "completed",
             "duration_seconds": 2.5, "metrics": {"output_count": 1}},
            {"state": "evaluate", "source_revision": "b" * 40, "status": "completed",
             "duration_seconds": 0.25, "metrics": {"score": 0.75}},
        ],
        "source_artifacts": [
            {"role": "visual_output", "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
             "size_bytes": media.stat().st_size, "media_type": media_type},
            {"role": "run_report", "sha256": "d" * 64, "size_bytes": 200},
        ],
        "gpu": {"model": "NVIDIA B200", "count": 1},
        "limitations": ["Synthetic unit fixture; this does not establish live acceptance"],
    }


def test_image_inventory_matches_restricted_packaging_contract() -> None:
    contract_path = (
        Path(__file__).resolve().parents[3]
        / "npa"
        / "docker"
        / "workbench"
        / "packaging-contract.yaml"
    )
    images = yaml.safe_load(contract_path.read_text(encoding="utf-8"))["images"]
    paidf_images = {
        f"npa-{name}": details
        for name, details in images.items()
        if name.startswith("paidf-")
    }
    assert set(paidf_images) == IMAGE_NAMES
    assert {details["redistribution"] for details in paidf_images.values()} == {"restricted"}


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    source = Image.new("RGB", (32, 24), (12, 34, 56))
    exif = Image.Exif()
    exif[270] = "unit fixture private source metadata"
    source.save(path, exif=exif)
    return path


def _rows(path: Path, entity: str, column: str, timeline: str | None = None) -> list:
    result = []
    for chunk in load_recording(path).chunks():
        batch = chunk.to_record_batch()
        if str(chunk.entity_path) != entity or column not in batch.schema.names:
            continue
        values = batch.column(column).to_pylist()
        indices = batch.column(timeline).to_pylist() if timeline else [None] * len(values)
        result.extend((index, value[0]) for index, value in zip(indices, values) if value)
    return sorted(result, key=lambda row: -1 if row[0] is None else row[0])


def _merge_recordings(paths: list[Path], destination: Path):
    from rerun.recording import load_archive

    merged = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "merge",
         "--output", str(destination), *map(str, paths)],
        capture_output=True, text=True, check=False,
    )
    assert merged.returncode == 0, merged.stderr
    return load_archive(destination).all_recordings()


def test_same_run_and_application_merge_with_original_identity(
    tmp_path: Path, media: Path,
) -> None:
    """Use actual Rerun bytes to demonstrate the existing identity collision."""
    import rerun as rr

    paths = []
    for index, image in enumerate(("npa-paidf-image-edit-sky", "npa-paidf-event-video-sky")):
        evidence = _evidence(media)
        evidence["image_name"] = image
        evidence["image_digest"] = "sha256:" + "a" * 63 + str(index)
        path = tmp_path / f"original-{index}.rrd"
        recording = rr.RecordingStream(APPLICATION_ID, recording_id=evidence["run_id"])
        recording.save(path)
        recording.log("provenance/run", rr.TextDocument(json.dumps(evidence)), static=True)
        recording.flush()
        recording.disconnect()
        path.chmod(0o600)
        paths.append(path)
    recordings = _merge_recordings(paths, tmp_path / "original-merged.rrd")
    assert len(recordings) == 1
    assert recordings[0].recording_id() == "synthetic-unit-run"


def test_distinct_full_image_digests_remain_separate_after_real_rrd_merge(
    tmp_path: Path, media: Path,
) -> None:
    """Exercise Rerun's archive ingestion, preserving each image's actual pixels."""
    paths = []
    expected = {}
    for index, image in enumerate(("npa-paidf-image-edit-sky", "npa-paidf-event-video-sky")):
        pixels = tmp_path / f"pixels-{index}.png"
        color = (12 + index * 80, 34, 56)
        Image.new("RGB", (32, 24), color).save(pixels)
        evidence = _evidence(pixels)
        evidence["image_name"] = image
        # Distinct image names and source pixels exercise image isolation.
        evidence["image_digest"] = "sha256:" + "a" * 63 + str(index)
        path = tmp_path / f"candidate-{index}.rrd"
        build_image_evidence_rrd(evidence, path, {"visual_output": pixels})
        paths.append(path)
        expected[image] = (evidence, color)
    recordings = _merge_recordings(paths, tmp_path / "candidate-merged.rrd")
    assert len(recordings) == 2
    assert len({recording.recording_id() for recording in recordings}) == 2
    assert {recording.application_id() for recording in recordings} == {APPLICATION_ID}
    observed = set()
    for recording in recordings:
        provenance = []
        frames = []
        for chunk in recording.chunks():
            batch = chunk.to_record_batch()
            if str(chunk.entity_path) == "/provenance/run":
                provenance.extend(json.loads(row[0]) for row in batch.column("TextDocument:text").to_pylist())
            if str(chunk.entity_path) == "/media/visual_output":
                frames.extend(row[0] for row in batch.column("EncodedImage:blob").to_pylist())
        assert len(provenance) == len(frames) == 1
        evidence = provenance[0]
        original, color = expected[evidence["image_name"]]
        assert evidence == original
        assert evidence["run_id"] == "synthetic-unit-run"
        with Image.open(BytesIO(bytes(frames[0]))) as decoded:
            assert decoded.getpixel((4, 5)) == color
        observed.add(evidence["image_name"])
    assert observed == set(expected)


def test_same_image_media_with_distinct_full_digests_remain_separate(
    tmp_path: Path, media: Path,
) -> None:
    """Only the final digest digit differs, isolating full-digest identity."""
    paths = []
    expected = {}
    for index in range(2):
        evidence = _evidence(media)
        evidence["image_digest"] = "sha256:" + "a" * 63 + str(index)
        path = tmp_path / f"digest-{index}.rrd"
        build_image_evidence_rrd(evidence, path, {"visual_output": media})
        paths.append(path)
        expected[evidence["image_digest"]] = evidence
    variants = list(expected.values())
    assert {key for key in variants[0] if variants[0][key] != variants[1][key]} == {"image_digest"}
    recordings = _merge_recordings(paths, tmp_path / "digests-merged.rrd")
    assert len(recordings) == 2
    assert len({recording.recording_id() for recording in recordings}) == 2
    observed = set()
    for recording in recordings:
        provenance = []
        for chunk in recording.chunks():
            if str(chunk.entity_path) == "/provenance/run":
                provenance.extend(json.loads(row[0]) for row in chunk.to_record_batch().column("TextDocument:text").to_pylist())
        assert len(provenance) == 1
        evidence = provenance[0]
        assert evidence == expected[evidence["image_digest"]]
        observed.add(evidence["image_digest"])
    assert observed == set(expected)


def test_same_image_updated_evidence_remains_separate_in_real_archive(
    tmp_path: Path, media: Path,
) -> None:
    paths = []
    for index in range(2):
        evidence = _evidence(media)
        if index:
            evidence["validation"]["checks"].append({"name": "additional_review", "status": "passed"})
        path = tmp_path / f"revision-{index}.rrd"
        build_image_evidence_rrd(evidence, path, {"visual_output": media})
        paths.append(path)
    recordings = _merge_recordings(paths, tmp_path / "revisions-merged.rrd")
    assert len(recordings) == 2
    evidence_rows = []
    for recording in recordings:
        for chunk in recording.chunks():
            if str(chunk.entity_path) == "/provenance/run":
                evidence_rows.extend(json.loads(row[0]) for row in chunk.to_record_batch().column("TextDocument:text").to_pylist())
    assert len({row["run_id"] for row in evidence_rows}) == 1
    assert len({row["image_digest"] for row in evidence_rows}) == 1
    assert sorted(len(row["validation"]["checks"]) for row in evidence_rows) == [1, 2]


def test_normalized_mapping_order_does_not_split_same_evidence(
    tmp_path: Path, media: Path,
) -> None:
    evidence = _evidence(media)
    paths = []
    for index, ordered in enumerate((evidence, dict(reversed(list(evidence.items()))))):
        path = tmp_path / f"order-{index}.rrd"
        build_image_evidence_rrd(ordered, path, {"visual_output": media})
        paths.append(path)
    assert load_recording(paths[0]).recording_id() == load_recording(paths[1]).recording_id()
    assert len(_merge_recordings(paths, tmp_path / "same-evidence-merged.rrd")) == 1


@pytest.mark.parametrize("image_name", sorted(IMAGE_NAMES))
def test_rrd_contains_bound_provenance_real_pixels_and_exact_stage_rows(
    tmp_path: Path, media: Path, image_name: str,
) -> None:
    evidence = _evidence(media)
    evidence["image_name"] = image_name
    out = tmp_path / "reports" / "image.rrd"
    result = build_image_evidence_rrd(evidence, out, {"visual_output": media})
    recording = load_recording(out)
    assert recording.recording_id() == result["decoded"]["recording_id"]
    assert evidence["run_id"] in recording.recording_id()
    assert recording.application_id() == APPLICATION_ID
    assert result["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert result["size_bytes"] == out.stat().st_size > 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert stat.S_IMODE(out.parent.stat().st_mode) == 0o700
    assert result["decoded"]["timelines"] == ["source_frame", "stage_index"]
    assert json.loads(_rows(out, "/provenance/run", "TextDocument:text")[0][1]) == evidence
    events = _rows(out, "/stages/events", "TextLog:text", "stage_index")
    assert [(index, json.loads(text)["state"]) for index, text in events] == [(0, "generate"), (1, "evaluate")]
    assert _rows(out, "/stages/generate/metrics/duration_seconds", "Scalars:scalars", "stage_index") == [(0, 2.5)]
    assert _rows(out, "/stages/evaluate/metrics/score", "Scalars:scalars", "stage_index") == [(1, 0.75)]
    blobs = _rows(out, "/media/visual_output", "EncodedImage:blob", "source_frame")
    assert len(blobs) == 1 and blobs[0][0] == 0
    with Image.open(BytesIO(bytes(blobs[0][1]))) as decoded:
        assert decoded.size == (32, 24)
        assert decoded.getpixel((4, 5)) == (12, 34, 56)
        assert not decoded.getexif()
        assert not decoded.info
    if image_name == "npa-paidf-image-edit-sky":
        checked = subprocess.run(
            [str(Path(sys.executable).with_name("rerun")), "rrd", "verify", str(out)],
            capture_output=True, text=True, check=False,
        )
        assert checked.returncode == 0, checked.stderr


def test_video_preserves_every_decoded_source_frame(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for color in ((12, 34, 56), (180, 20, 40), (40, 170, 50)):
            pixels = np.full((24, 32, 3), color, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video)) as container:
        expected = [frame.to_image().convert("RGB").tobytes() for frame in container.decode(video=0)]
    evidence = _evidence(video, "video")
    evidence["image_name"] = "npa-paidf-event-video-sky"
    output = tmp_path / "video.rrd"
    result = build_image_evidence_rrd(evidence, output, {"visual_output": video})
    rows = _rows(output, "/media/visual_output", "EncodedImage:blob", "source_frame")
    assert [row[0] for row in rows] == [0, 1, 2]
    assert result["decoded"]["media"]["visual_output"]["frame_count"] == 3
    for (_, blob), pixels in zip(rows, expected):
        with Image.open(BytesIO(bytes(blob))) as image:
            assert image.convert("RGB").tobytes() == pixels


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(stages=[]),
    lambda e: e["stages"][0].update(duration_seconds=-1),
    lambda e: e["stages"][0]["metrics"].update(score=float("nan")),
    lambda e: e["stages"][0]["metrics"].update(duration_seconds=99),
    lambda e: e["stages"][0].update(source_revision="e" * 40),
    lambda e: e.update(image_digest="mutable-tag"),
    lambda e: e.update(raw_log="unreviewed raw output"),
    lambda e: e.update(source_revisions=["short"]),
    lambda e: e.update(image_build_source_revision="short"),
    lambda e: e.update(runtime_source_revisions=[]),
    lambda e: e.update(image_build_source_revision="f" * 40),
    lambda e: e.update(source_revisions=["b" * 40, "f" * 40]),
    lambda e: e.update(runtime_source_revisions=["b" * 40, "b" * 40]),
    lambda e: e["validation"]["checks"][0].update(status="failed"),
    lambda e: e.update(source_artifacts=[]),
    lambda e: e["upstream_sources"][0].update(repository="https://github.com/other/fixture"),
])
def test_rejects_incomplete_or_contradictory_evidence(tmp_path: Path, media: Path, mutation) -> None:
    evidence = deepcopy(_evidence(media))
    mutation(evidence)
    output = tmp_path / "rejected.rrd"
    with pytest.raises(PaidfEvidenceError):
        build_image_evidence_rrd(evidence, output, {"visual_output": media})
    assert not output.exists()


@pytest.mark.parametrize("private", [
    "https://private.invalid/path?signature=synthetic",
    "s3://synthetic-private-bucket/output",
    "registry.example.invalid/image",
    "192.0.2.1",
    "2001:db8::1",
    "Observed at 2001:db8:1:2:3:4:5:6 during execution",
    "Bearer synthetic-unit-secret",
    "Authorization: bEaReR synthetic-unit-secret",
    "project_id=synthetic-private-project",
    "token=synthetic-private-value",
    "hf_" + "synthetic" * 5,
    "e00" + "synthetic" * 3,
])
def test_rejects_private_provenance(tmp_path: Path, media: Path, private: str) -> None:
    evidence = _evidence(media)
    evidence["limitations"] = [private]
    with pytest.raises(PaidfEvidenceError, match="private or secret-shaped"):
        build_image_evidence_rrd(evidence, tmp_path / "private.rrd", {"visual_output": media})


def test_rejects_media_mismatch_absence_and_invalid_bytes(tmp_path: Path, media: Path) -> None:
    evidence = _evidence(media)
    for paths in ({}, {"wrong_role": media}):
        with pytest.raises(PaidfEvidenceError, match="declared media roles"):
            build_image_evidence_rrd(evidence, tmp_path / "missing.rrd", paths)
    media.write_bytes(b"changed source")
    with pytest.raises(PaidfEvidenceError, match="media hash or size"):
        build_image_evidence_rrd(evidence, tmp_path / "mismatch.rrd", {"visual_output": media})
    with pytest.raises(PaidfEvidenceError, match="could not be decoded"):
        build_image_evidence_rrd(_evidence(media), tmp_path / "invalid.rrd", {"visual_output": media})
    assert not list(tmp_path.glob("*.rrd"))
    assert not list(tmp_path.glob(".paidf-evidence-*"))


def test_does_not_overwrite_existing_evidence(tmp_path: Path, media: Path) -> None:
    path = tmp_path / "existing.rrd"
    path.write_bytes(b"existing evidence")
    with pytest.raises(PaidfEvidenceError, match="cannot be overwritten"):
        build_image_evidence_rrd(_evidence(media), path, {"visual_output": media})
    assert path.read_bytes() == b"existing evidence"


def test_distinguishes_build_from_runtime_and_checks_complete_scalar_sequence(
    tmp_path: Path, media: Path,
) -> None:
    evidence = _evidence(media)
    evidence["image_build_source_revision"] = "f" * 40
    evidence["source_revisions"].append("f" * 40)
    evidence["stages"].append({
        "state": "generate", "source_revision": "b" * 40, "status": "completed",
        "duration_seconds": 3.5, "metrics": {"output_count": 2},
    })
    output = tmp_path / "revisions.rrd"
    result = build_image_evidence_rrd(evidence, output, {"visual_output": media})
    assert result["image_build_source_revision"] == "f" * 40
    assert result["runtime_source_revisions"] == ["b" * 40]
    assert _rows(output, "/stages/generate/metrics/output_count", "Scalars:scalars", "stage_index") == [(0, 1.0), (2, 2.0)]
    evidence["stages"][0]["source_revision"] = "f" * 40
    with pytest.raises(PaidfEvidenceError, match="absent from runtime_source_revisions"):
        build_image_evidence_rrd(evidence, tmp_path / "wrong-runtime.rrd", {"visual_output": media})

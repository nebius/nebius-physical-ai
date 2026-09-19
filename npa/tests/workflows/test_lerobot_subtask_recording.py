"""Decode the recorded LeRobot proof and verify its before/after artifact bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

REPOSITORY = Path(__file__).parents[3]
SCRIPT = REPOSITORY / "npa/scripts/record_lerobot_subtask_proof.py"
EVIDENCE = REPOSITORY / "docs/workbench/evidence/lerobot-subtasks"


@pytest.fixture(scope="module")
def recorder():
    spec = importlib.util.spec_from_file_location(
        "record_lerobot_subtask_proof", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, recorder):
    root = tmp_path_factory.mktemp("recorded-subtasks") / "proof"
    manifest = recorder.record_proof(root, "test-recorded-subtasks")
    return root, manifest


def test_recorded_yaml_proves_grasp_and_preserves_source(recorded) -> None:
    root, manifest = recorded
    original = pq.read_table(root / "input/data/chunk-000/file-000.parquet")
    reviewed = pq.read_table(root / "reviewed/data/chunk-000/file-000.parquet")
    assert "subtask_index" not in original.column_names
    assert original.equals(reviewed.select(original.column_names), check_metadata=True)
    assert reviewed.column("subtask_index").to_pylist() == [1, 1, 2, 2, 0, 3, 3, 3]
    assert manifest["input_unchanged"] is True
    assert manifest["original_frame_columns_unchanged"] is True
    proof = json.loads((root / "subtask-proof.json").read_text())
    assert proof["proof"]["subtask"] == "grasp"
    assert proof["proof"]["frame_index"] == 2
    assert proof["summary"]["labeled_frame_count"] == 8
    workflow = json.loads((root / "workflow-report.json").read_text())
    assert workflow["status"] == "completed"
    assert workflow["steps"][0]["returncode"] == 0


def test_rerun_records_all_input_and_labeled_rows(recorded) -> None:
    root, manifest = recorded
    recording = load_recording(root / "lerobot-subtasks.rrd")
    assert recording.application_id() == "npa-lerobot-subtask-proof"
    assert recording.recording_id() == "test-recorded-subtasks"
    labels = []
    for chunk in recording.chunks():
        if chunk.entity_path != "/reviewed/frame":
            continue
        batch = chunk.to_record_batch()
        for ordinal, text in zip(
            batch.column("dataset_row").to_pylist(),
            batch.column("TextDocument:text").to_pylist(),
            strict=True,
        ):
            labels.append((ordinal, json.loads(text[0])["resolved_subtask"]))
    assert sorted(labels) == list(
        enumerate(
            [
                "approach",
                "approach",
                "grasp",
                "grasp",
                "align",
                "place",
                "place",
                "place",
            ]
        )
    )
    assert manifest["recording"]["samples_per_entity"] == {
        "/input/action_0": 8,
        "/input/frame": 8,
        "/reviewed/frame": 8,
        "/subtasks/index": 8,
    }
    assert manifest["recording"]["rrd_verify"] == "1 file verified without error."


def test_bundle_inventory_matches_every_saved_byte(recorded) -> None:
    root, manifest = recorded
    with zipfile.ZipFile(root / "proof-bundle.zip") as archive:
        assert set(archive.namelist()) == {*manifest["artifacts"], "manifest.json"}
        assert json.loads(archive.read("manifest.json")) == manifest
        for relative, digest in manifest["artifacts"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest
            assert archive.read(relative) == (root / relative).read_bytes()
    assert str(REPOSITORY) not in json.dumps(manifest)
    assert str(root) not in (root / "workflow-report.json").read_text()
    assert "not captured robot data" in manifest["provenance"]["limitations"]


def test_recording_destination_cannot_be_overwritten(recorded, recorder) -> None:
    root, _manifest = recorded
    before = (root / "proof-bundle.zip").read_bytes()
    with pytest.raises(FileExistsError):
        recorder.record_proof(root, "another-run")
    assert (root / "proof-bundle.zip").read_bytes() == before


def test_recording_inspection_rejects_wrong_run_id(recorded, recorder) -> None:
    root, _manifest = recorded
    rows = json.loads((root / "before-after.json").read_text())
    with pytest.raises(ValueError, match="identity"):
        recorder._inspect_recording(
            root / "lerobot-subtasks.rrd", rows, "wrong-recording"
        )


def test_recording_inspection_rejects_wrong_label(recorded, recorder) -> None:
    root, _manifest = recorded
    rows = json.loads((root / "before-after.json").read_text())
    rows[2]["subtask"] = "fabricated-label"
    with pytest.raises(ValueError, match="every source frame and subtask"):
        recorder._inspect_recording(
            root / "lerobot-subtasks.rrd", rows, "test-recorded-subtasks"
        )


def test_recording_rejects_modified_original_column(tmp_path, recorder) -> None:
    recorder._write_input(tmp_path / "input")
    recorder._label_copy(tmp_path)
    path = tmp_path / "reviewed/data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.column_names.index("action"), "action", pa.array([[99.0]] * 8)
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="original frame column"):
        recorder._before_after(tmp_path)


def test_committed_bundle_proves_the_saved_parquet_row() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    with zipfile.ZipFile(EVIDENCE / "proof-bundle.zip") as archive:
        for relative, digest in manifest["artifacts"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest
        proof = json.loads(archive.read("subtask-proof.json"))["proof"]
        data = archive.read("reviewed/" + proof["source_data_file"])
        assert hashlib.sha256(data).hexdigest() == proof["source_parquet_sha256"]
        row = pq.read_table(pa.BufferReader(data)).to_pylist()[2]
        catalog = pq.read_table(
            pa.BufferReader(archive.read("reviewed/meta/subtasks.parquet"))
        ).to_pylist()
        labels = {item["subtask_index"]: item["subtask"] for item in catalog}
        assert labels[row["subtask_index"]] == proof["subtask"] == "grasp"
        assert (
            archive.read("lerobot-subtasks.rrd")
            == (EVIDENCE / "lerobot-subtasks.rrd").read_bytes()
        )
        assert json.loads(archive.read("manifest.json")) == manifest
    for relative, digest in manifest["provenance"]["recipe_sha256"].items():
        recipe = REPOSITORY / relative
        if relative == "npa/src/npa/fiftyone_lerobot_subtasks.py":
            recipe = EVIDENCE.parent / "lerobot-subtask-recipes" / f"{digest}.py"
        assert hashlib.sha256(recipe.read_bytes()).hexdigest() == digest


def test_agent_ui_mp4_decodes_and_matches_its_capture_receipt() -> None:
    import av

    receipt = json.loads((EVIDENCE / "agent-ui-capture.json").read_text())
    video = receipt["video"]
    path = EVIDENCE / video["path"]
    assert path.stat().st_size == video["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"]
    source = receipt["source"]
    assert (
        hashlib.sha256((EVIDENCE / source["path"]).read_bytes()).hexdigest()
        == source["sha256"]
    )
    sampled_pixels = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (video["width"], video["height"])
        assert float(stream.average_rate) == 25.0
        assert float(stream.duration * stream.time_base) == pytest.approx(
            video["duration_seconds"]
        )
        timestamps = []
        for index, frame in enumerate(container.decode(video=0)):
            timestamps.append(frame.pts)
            if index in (0, 225, 600):
                sampled_pixels.add(
                    hashlib.sha256(frame.to_ndarray(format="rgb24")).hexdigest()
                )
    assert len(timestamps) == video["decoded_frame_count"]
    assert all(left < right for left, right in zip(timestamps, timestamps[1:]))
    assert len(sampled_pixels) == 3

"""Verify actual RRD bytes, archive isolation, legacy content, and decoder failures."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import rerun as rr
import rerun.blueprint as rrb

from npa.viz.recordings import load_recording, load_recordings


def _write(path, recording_id):
    recording = rr.RecordingStream("npa-reader-contract", recording_id=recording_id)
    recording.save(path)
    recording.send_blueprint(rrb.TimeSeriesView(origin="metrics"))
    recording.log("provenance/run", rr.TextDocument(recording_id), static=True)
    for step, value in ((0, 0.25), (3, 0.75)):
        recording.set_time("optimizer_step", sequence=step)
        recording.log("metrics/loss", rr.Scalars(value))
    recording.flush()
    recording.disconnect()


def _metric_rows(recording):
    rows = []
    for chunk in recording.chunks():
        if str(chunk.entity_path) == "/metrics/loss":
            batch = chunk.to_record_batch()
            rows.extend(
                zip(
                    batch["optimizer_step"].to_pylist(),
                    batch["Scalars:scalars"].to_pylist(),
                    strict=True,
                )
            )
    return sorted(rows)


def test_round_trip_preserves_identity_timelines_values_and_blueprint_separation(
    tmp_path,
):
    path = tmp_path / "recording.rrd"
    _write(path, "reader-round-trip")
    recording = load_recording(path)
    assert recording.application_id() == "npa-reader-contract"
    assert recording.recording_id() == "reader-round-trip"
    assert _metric_rows(recording) == [(0, [0.25]), (3, [0.75])]
    provenance = [
        chunk.to_record_batch()["TextDocument:text"].to_pylist()
        for chunk in recording.chunks()
        if str(chunk.entity_path) == "/provenance/run"
    ]
    assert provenance == [[["reader-round-trip"]]]


def test_multiple_recordings_are_rejected_until_selected_explicitly(tmp_path):
    inputs = [tmp_path / f"{index}.rrd" for index in range(2)]
    for index, path in enumerate(inputs):
        _write(path, f"run-{index}")
    archive = tmp_path / "archive.rrd"
    subprocess.run(
        [
            str(Path(sys.executable).with_name("rerun")),
            "rrd",
            "merge",
            "--output",
            str(archive),
            *map(str, inputs),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="exactly one recording; found 2"):
        load_recording(archive)
    recordings = load_recordings(archive)
    assert {recording.recording_id() for recording in recordings} == {"run-0", "run-1"}
    assert all(
        _metric_rows(recording) == [(0, [0.25]), (3, [0.75])]
        for recording in recordings
    )


def test_existing_031_recording_retains_its_original_bytes_and_rows():
    path = (
        Path(__file__).parents[2]
        / "docs/workbench/evidence/lerobot-subtasks/lerobot-subtasks.rrd"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recording = load_recording(path)
    assert recording.application_id() == "npa-lerobot-subtask-proof"
    assert recording.recording_id() == "lerobot-subtask-recorded-proof"
    rows = [
        row
        for chunk in recording.chunks()
        if str(chunk.entity_path) == "/subtasks/index"
        for row in chunk.to_record_batch()["dataset_row"].to_pylist()
    ]
    assert sorted(rows) == list(range(8))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_invalid_rrd_fails_decoding(tmp_path):
    path = tmp_path / "invalid.rrd"
    path.write_bytes(b"not a Rerun recording")
    with pytest.raises(RuntimeError, match="Not an RRD file"):
        load_recording(path)

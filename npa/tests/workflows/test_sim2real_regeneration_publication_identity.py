"""Actual format/persist/seal controls with synthetic inputs and upload capture."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import rerun as rr
from mcap.reader import make_reader
from mcap.writer import Writer
from rerun.chunk import RrdReader

from npa.workflows import sim2real_rerun_regen as regen
from npa.workflows.data_factory_viz import _bounded_rrd_batches
from npa.workflows.sim2real.component_authority import (
    validate_stage14_component_record,
    validate_remote_regeneration_authority,
)
from npa.workflows.sim2real_viz import Sim2RealVizResult, _log_summary_documents

from test_sim2real_regen_producer_identity import (
    ROOT,
    SOURCE_SHA,
    _config,
    _post_authority_state,
    _replay_environment,
)

pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")
_VERIFY_INPUTS = regen._verify_regeneration_inputs
_FINALIZE_RESULT = regen._finalize_regen_result


def _real_diagnostic_recordings(directory: Path, authority: dict):
    writer = authority["writer"]
    event = json.dumps(
        {"synthetic_diagnostic": True, "current_writer": writer}, sort_keys=True
    )
    rrd, mcap = directory / "sim2real.rrd", directory / "sim2real.mcap"
    stream = rr.RecordingStream("writer-control", recording_id="frozen-control")
    stream.save(rrd)
    stream.log("diagnostic/writer", rr.TextDocument(event), static=True)
    _log_summary_documents(
        rr,
        stream,
        local_dir=directory,
        inner_evidence={},
        heldout_report=None,
        stage_components=[],
        run_metadata={"regeneration": authority},
        critique_panel_rows=[],
        counts={},
    )
    stream.flush()
    stream.disconnect()
    _write_diagnostic_mcap(mcap, event)
    _validate_diagnostic_recordings(rrd, mcap, event, authority)
    return rrd, mcap


def _write_diagnostic_mcap(mcap: Path, event: str):
    with mcap.open("wb") as output:
        encoder = Writer(output)
        encoder.start()
        channel = encoder.register_channel("diagnostic/writer", "json", schema_id=0)
        encoder.add_message(channel, log_time=0, publish_time=0, data=event.encode())
        encoder.finish()


def _rrd_texts(rrd: Path, entity_path: str):
    reader = RrdReader(rrd)
    return [
        value[0]
        for entity, batch in _bounded_rrd_batches(
            chunk
            for entry in reader.recordings()
            for chunk in reader.stream(store=entry)
        )
        if entity == entity_path
        for value in batch.column("TextDocument:text").to_pylist()
    ]


def _validate_diagnostic_recordings(rrd, mcap, event, authority):
    writer = authority["writer"]
    assert _rrd_texts(rrd, "/diagnostic/writer") == [event]
    summary = _rrd_texts(rrd, "/summary/regeneration")
    assert len(summary) == 1
    assert authority["input"]["source_sha"] in summary[0]
    assert writer["source_sha"] in summary[0]
    assert writer["workflow_job"] not in summary[0]
    assert writer["image"] not in summary[0]
    assert authority["input"]["report_uri"] not in summary[0]
    with mcap.open("rb") as encoded:
        assert [
            message.data.decode()
            for _, _, message in make_reader(encoded).iter_messages()
        ] == [event]


def _publication(tmp_path, rrd, mcap, authority):
    generation = regen._recording_publication_id(rrd, mcap, regeneration=authority)
    prefix = f"{ROOT}/reports/generations/{generation}/"
    return regen._RegenPublication(
        prefix=f"{ROOT}/",
        local_dir=tmp_path,
        report_path=tmp_path / "unused-heldout.json",
        renders_dir=tmp_path / "unused-renders",
        publish_renders=False,
        rrd_path=rrd,
        publication_id=generation,
        generation_prefix=prefix,
        immutable_rrd_uri=f"{prefix}sim2real.rrd",
        visual_index_path=tmp_path / "unused-index.json",
        final_report_path=tmp_path / "reports/sim2real-report.json",
        candidate_path=tmp_path / "unused-candidate.json",
    )


def _prepared_replay(tmp_path, monkeypatch, *, cross_source=False):
    _replay_environment(monkeypatch)
    if cross_source:
        monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "e" * 40)
        monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "e" * 40)
    state, _, original = _post_authority_state(tmp_path, monkeypatch)
    monkeypatch.setattr(regen, "_verify_regeneration_inputs", _VERIFY_INPUTS)
    (state.report_path.parent / "original-input.json").write_bytes(original)
    old_report = copy.deepcopy(state.report)
    writer = regen._regen_producer_provenance(state, upload=True)
    authority = regen._regen_input_authority(state, writer)
    state = replace(state, regeneration_authority=authority)
    rrd, mcap = _real_diagnostic_recordings(state.report_path.parent, authority)
    result = Sim2RealVizResult(
        status="written", output_rrd_path=str(rrd), heldout_frame_count=1
    )
    regen._persist_regen_report(
        _config(),
        tmp_path,
        rrd,
        state,
        result,
        [],
        "diagnostic",
        0.125,
        producer_provenance=writer,
    )
    regen._capture_regen_mcap_identity(state.report_path, mcap)
    return state, old_report, original, _publication(tmp_path, rrd, mcap, authority)


def _capture_publication(monkeypatch, publication):
    uploads = []

    def upload(_storage, path, uri):
        uploads.append((uri, path.read_bytes()))
        return uri

    monkeypatch.setattr(regen, "upload_immutable_file", upload)
    record_path = regen._publish_regen_final_report(
        _ReplayInputStore(publication),
        publication,
        f"{publication.generation_prefix}sim2real.mcap",
    )
    return (
        json.loads(publication.final_report_path.read_text()),
        json.loads(record_path.read_text()),
        uploads,
    )


class _ReplayInputStore:
    """Memory transport; actual production input download/verification remains active."""

    def __init__(self, publication):
        report = json.loads(publication.final_report_path.read_text())
        retained = report["regeneration"]["input"]
        self.objects = {
            retained["report_uri"]: (
                publication.final_report_path.parent / "original-input.json"
            ).read_bytes(),
            retained["stage14_history_uri"]: json.dumps(
                retained["stage14_record"]
            ).encode(),
        }

    def download_path(self, uri, destination):
        Path(destination).write_bytes(self.objects[uri])


@pytest.mark.parametrize("cross_source", [False, True])
def test_real_format_publication_keeps_input_authority_and_names_new_writer(
    tmp_path, monkeypatch, cross_source
):
    state, old, raw, publication = _prepared_replay(
        tmp_path, monkeypatch, cross_source=cross_source
    )
    report, record, uploads = _capture_publication(monkeypatch, publication)
    validate_stage14_component_record(record, report, expected_source_sha=SOURCE_SHA)
    authority = report["regeneration"]
    assert report["source_sha"] == old["source_sha"] == authority["input"]["source_sha"]
    assert authority["input"]["report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert authority["input"]["stage14_record"] == old["component_records"][-1]
    for key, value in authority["writer"].items():
        assert record["artifacts"][key] == value
    assert (
        authority["writer"]["workflow_job"]
        != old["component_records"][-1]["artifacts"]["workflow_job"]
    )
    assert record["artifacts"]["source_sha"] == (
        "e" * 40 if cross_source else SOURCE_SHA
    )
    assert len(uploads) == 2
    assert json.loads(uploads[-1][1]) == record
    assert record["content_sha256"] != old["component_records"][-1]["content_sha256"]
    assert (
        authority["output"]["rrd"]["sha256"]
        == hashlib.sha256(publication.rrd_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("tamper", ["input_hash", "writer_job", "absent_writer"])
def test_false_writer_or_input_cannot_upload_new_authority(
    tmp_path, monkeypatch, tamper
):
    state, _, _, publication = _prepared_replay(tmp_path, monkeypatch)
    report = json.loads(state.report_path.read_text())
    if tamper == "input_hash":
        report["regeneration"]["input"]["report_sha256"] = "0" * 64
    elif tamper == "writer_job":
        report["regeneration"]["writer"]["workflow_job"] = "forged-original-job"
    else:
        monkeypatch.delenv("SKYPILOT_TASK_ID")
    state.report_path.write_text(json.dumps(report))
    uploads = []
    monkeypatch.setattr(regen, "upload_immutable_file", lambda *_: uploads.append(True))
    with pytest.raises(regen.Sim2RealRerunRegenError):
        regen._publish_regen_final_report(
            object(), publication, f"{publication.generation_prefix}sim2real.mcap"
        )
    assert uploads == []


def test_unresealed_current_writer_tamper_is_rejected(tmp_path, monkeypatch):
    _, _, _, publication = _prepared_replay(tmp_path, monkeypatch)
    report, record, _ = _capture_publication(monkeypatch, publication)
    record["artifacts"]["workflow_job"] = "forged-job"
    with pytest.raises(ValueError):
        validate_stage14_component_record(
            record, report, expected_source_sha=SOURCE_SHA
        )


@pytest.mark.parametrize("name", ["rrd", "mcap"])
def test_post_encoding_output_tamper_cannot_be_resealed(tmp_path, monkeypatch, name):
    _, _, _, publication = _prepared_replay(tmp_path, monkeypatch)
    path = publication.rrd_path if name == "rrd" else tmp_path / "reports/sim2real.mcap"
    path.write_bytes(b"explicit rejected post-encoding replacement")
    uploads = []
    monkeypatch.setattr(regen, "upload_immutable_file", lambda *_: uploads.append(True))
    with pytest.raises(
        regen.Sim2RealRerunRegenError, match="bytes changed after encoding"
    ):
        regen._publish_regen_final_report(
            _ReplayInputStore(publication),
            publication,
            f"{publication.generation_prefix}sim2real.mcap",
        )
    assert uploads == []


def test_preview_of_an_existing_regeneration_keeps_its_sealed_authority(
    tmp_path, monkeypatch
):
    state, _, _, publication = _prepared_replay(tmp_path, monkeypatch)
    report, _, _ = _capture_publication(monkeypatch, publication)
    state = replace(state, report=report, regeneration_authority=None)
    before = state.report_path.read_bytes()
    (tmp_path / "reports/sim2real.mcap").write_bytes(
        b"explicit preview transport fixture"
    )
    monkeypatch.setattr(regen, "_emit_regen_mcap", lambda *_: {"status": "written"})
    _FINALIZE_RESULT(
        _config(),
        tmp_path,
        object(),
        state,
        Sim2RealVizResult(
            status="written",
            output_rrd_path=str(publication.rrd_path),
            heldout_frame_count=1,
        ),
        upload=False,
    )
    assert state.report_path.read_bytes() == before


@pytest.mark.parametrize("tamper", [None, "report", "history"])
def test_referenced_input_report_and_history_bytes_are_verified(
    tmp_path, monkeypatch, tamper
):
    _, old, raw, publication = _prepared_replay(
        tmp_path, monkeypatch, cross_source=True
    )
    report, _, _ = _capture_publication(monkeypatch, publication)
    retained = report["regeneration"]["input"]
    objects = {
        retained["report_uri"]: raw,
        retained["stage14_history_uri"]: json.dumps(
            old["component_records"][-1]
        ).encode(),
    }
    before = copy.deepcopy(objects)
    if tamper == "report":
        objects[retained["report_uri"]] = raw.replace(
            b'"source_sha": "' + SOURCE_SHA.encode(), b'"source_sha": "' + b"e" * 40
        )
    elif tamper == "history":
        objects[retained["stage14_history_uri"]] = b"{}"
    if tamper:
        with pytest.raises(ValueError, match="input"):
            validate_remote_regeneration_authority(report, objects.__getitem__)
    else:
        validate_remote_regeneration_authority(report, objects.__getitem__)
        assert objects == before

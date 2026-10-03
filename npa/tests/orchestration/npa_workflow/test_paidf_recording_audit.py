"""Exercise the shared PAIDF live audit against decoded local RRD bytes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


BUCKET = "synthetic-private-bucket"
PREFIX = "paidf-cosmos3/synthetic-run/"
CURATOR = "curation/cosmos_curator.json"
CURATION = "curation/report.json"


@pytest.fixture
def live_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    return importlib.import_module("tests.e2e.test_npa_workflow_submit_live_e2e")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("paidf-audit-video") / "candidate-a.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=5",
            "-frames:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _source_reports():
    curator = {
        "schema": "npa.cosmos_curate.curation.v1",
        "status": "completed",
        "engine": "cosmos-curator-stages",
        "variant_count": 1,
        "clip_count": 1,
        "motion_filter": "score-only",
        "encoder": "libx264",
    }
    curation = {
        "schema": "npa.fiftyone.curation.v1",
        "status": "curated",
        "curation_engine": "fiftyone-brain",
        "augmented_clips": 1,
        "clip_ids": ["candidate-a"],
        "video_count": 1,
        "frame_count": 2,
        "curated_kept": 1,
        "curated_dropped": 0,
        "multiply": {"mode": "single-variant", "variant_count": 1},
        "fiftyone": {"fiftyone_version": "1.0.0", "dedup_threshold": 0.99},
    }
    selected = {CURATOR: curator, CURATION: curation}
    originals = copy.deepcopy(selected)
    originals[CURATOR].update(
        {
            "augment_uri": f"s3://{BUCKET}/{PREFIX}cosmos_augmented/",
            "source": "/private/operator/curator-checkout",
            "runtime": {"hostname": "private-synthetic-worker"},
        }
    )
    originals[CURATION]["input_source"] = {
        "staged_video_uri": f"s3://{BUCKET}/{PREFIX}input/source.mp4"
    }
    bodies = {
        key: (json.dumps(value, indent=3) + "\n").encode()
        for key, value in originals.items()
    }
    for key, report in selected.items():
        report["source_report"] = {
            "artifact": key,
            "sha256": hashlib.sha256(bodies[key]).hexdigest(),
        }
    return bodies, selected


def _write_recording(path, video, selected, *, defect=""):
    import rerun as rr

    rec = rr.RecordingStream("neural-reconstruction", recording_id="synthetic-run")
    rec.save(str(path))
    for entity, key in (
        ("pipeline/4_cosmos_curator", CURATOR),
        ("pipeline/4_curation", CURATION),
    ):
        text = "## Producer report\n\n```json\n" + json.dumps(selected[key]) + "\n```"
        rr.log(entity, rr.TextDocument(text), static=True, recording=rec)
    caption = (
        "candidate-a/frame-00000.png: blue cloth"
        if defect != "caption"
        else "blue cloth"
    )
    rr.log(
        "captions/labeled_augmented",
        rr.TextDocument(caption),
        static=True,
        recording=rec,
    )
    disposition = "REJECTED" if defect == "disposition" else "ACCEPTED"
    rr.log(
        "augmented/candidate-a/disposition",
        rr.TextDocument(disposition),
        static=True,
        recording=rec,
    )
    asset = rr.AssetVideo(path=video)
    if defect != "video":
        rr.log("augmented/candidate-a/video", asset, static=True, recording=rec)
    timestamps = asset.read_frame_timestamps_nanos()
    if defect == "frame-count":
        timestamps = timestamps[:1]
    if defect == "timestamp":
        timestamps = timestamps + 1_000_000
    rr.send_columns(
        "augmented/candidate-a/video",
        indexes=[rr.TimeColumn("video_time", duration=timestamps * 1e-9)],
        columns=rr.VideoFrameReference.columns_nanos(timestamps),
        recording=rec,
    )
    rec.flush()
    rec.disconnect()


@pytest.fixture
def recording_case(tmp_path, video):
    from npa.workflows.paidf_cosmos3_media import probe_video

    bodies, selected = _source_reports()
    local_video = tmp_path / "candidate-a.mp4"
    shutil.copyfile(video, local_video)
    recording = tmp_path / "uploaded.rrd"
    reads = []

    def get_object(*, Bucket, Key):
        assert Bucket == BUCKET and Key.startswith(PREFIX)
        reads.append(Key)
        return {"Body": io.BytesIO(bodies[Key[len(PREFIX) :]])}

    def download_file(bucket, key, destination):
        assert bucket == BUCKET and key == PREFIX + "reports/sim2real.rrd"
        shutil.copyfile(recording, destination)

    return SimpleNamespace(
        bodies=bodies,
        selected=selected,
        video=local_video,
        folder=tmp_path,
        path=recording,
        reads=reads,
        client=SimpleNamespace(get_object=get_object, download_file=download_file),
        variants=[
            {
                "clip": "candidate-a",
                "temporal_alignment": {
                    "generated_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "decoded_frames": probe_video(video)["decoded_frames"],
                },
            }
        ],
    )


def _audit(live_audit, case):
    live_audit._assert_transfer_recording(
        case.client,
        BUCKET,
        PREFIX,
        case.folder,
        case.variants,
        lambda relative: json.loads(case.bodies[relative]),
    )


def test_live_audit_accepts_sanitized_recording_and_hashes_original_bytes(
    live_audit, recording_case
):
    case = recording_case
    _write_recording(case.path, case.video, case.selected)

    _audit(live_audit, case)

    assert set(case.reads) == {PREFIX + CURATOR, PREFIX + CURATION}
    for relative, body in case.bodies.items():
        canonical = json.dumps(json.loads(body), indent=2, sort_keys=True).encode()
        assert hashlib.sha256(body).digest() != hashlib.sha256(canonical).digest()


def _stage_producer_run(case):
    run = case.folder / "producer-run"
    for relative, raw in case.bodies.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    candidate = run / "cosmos_augmented/candidate-a"
    candidate.mkdir(parents=True)
    shutil.copyfile(case.video, candidate / "augmented_video.mp4")
    source = run / "input/source.mp4"
    source.parent.mkdir(parents=True)
    shutil.copyfile(case.video, source)
    payloads = {
        "cosmos_augmented/manifest.json": {
            "schema": "npa.cosmos2.transfer.v1",
            "mode": "cosmos_transfer2.5_gpu",
            "status": "executed",
            "node_count": 1,
            "variant_count": 1,
            "variants": [
                {
                    "clip": "candidate-a",
                    "variant_index": 0,
                    "control_uris": {},
                    "augmented_video_uri": f"s3://{BUCKET}/{PREFIX}cosmos_augmented/candidate-a/augmented_video.mp4",
                }
            ],
        },
        "grade/quality_disposition.json": {"quality_status": "accepted", "score": 0.82},
        "labeled_augmented/captions.json": {
            "captions": [
                {"image": "candidate-a/frame-00000.png", "caption": "blue cloth"}
            ]
        },
    }
    for relative, payload in payloads.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return run


def test_live_audit_accepts_current_recording_producer_output(
    live_audit, recording_case
):
    from npa.workflows.data_factory_viz import build_run_rrd

    case = recording_case
    run = _stage_producer_run(case)
    before = {path: path.read_bytes() for path in run.rglob("*.json")}
    result = build_run_rrd(str(run), str(case.path))
    assert result["augmented_video_components"] == 1
    assert result["source_video_components"] == 1

    _audit(live_audit, case)

    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_live_audit_keeps_grounded_public_provenance_links(live_audit, recording_case):
    case = recording_case
    case.selected[CURATOR]["upstream"] = {
        "repo": "https://github.com/nvidia-cosmos/cosmos-curate",
        "license": "Apache-2.0",
    }
    _write_recording(case.path, case.video, case.selected)

    _audit(live_audit, case)


@pytest.mark.parametrize(
    ("relative", "field", "incorrect"),
    [
        (CURATOR, "engine", "unavailable"),
        (CURATOR, "clip_count", 2),
        (CURATOR, "variant_count", 2),
        (CURATOR, "status", "skipped"),
        (CURATOR, "schema", "unrelated.schema"),
        (CURATOR, "encoder", "wrong-encoder"),
        (CURATION, "curation_engine", "report-only"),
        (CURATION, "augmented_clips", 2),
        (CURATION, "clip_ids", ["wrong-candidate"]),
        (CURATION, "video_count", 2),
        (CURATION, "frame_count", 3),
        (CURATION, "curated_kept", 2),
        (CURATION, "curated_dropped", 1),
        (CURATION, "augmented_clips", True),
    ],
)
def test_live_audit_rejects_changed_curation_fact(
    live_audit, recording_case, relative, field, incorrect
):
    case = recording_case
    case.selected[relative][field] = incorrect
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="Recorded curation fact differs"):
        _audit(live_audit, case)


@pytest.mark.parametrize(
    ("section", "field", "incorrect"),
    [
        ("multiply", "mode", "multi-variant"),
        ("multiply", "variant_count", 2),
        ("fiftyone", "fiftyone_version", "wrong-version"),
        ("fiftyone", "dedup_threshold", 0.5),
    ],
)
def test_live_audit_rejects_changed_nested_curation_fact(
    live_audit, recording_case, section, field, incorrect
):
    case = recording_case
    case.selected[CURATION][section][field] = incorrect
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="Recorded curation fact differs"):
        _audit(live_audit, case)


@pytest.mark.parametrize(
    "reference",
    [
        "curation/report.json",
        "../curation/cosmos_curator.json",
        "/curation/cosmos_curator.json",
        "curation/cosmos_curator.json?download=1",
        "curation/cosmos_curator.json#source",
        "curation/cosmos_curator.json-extra",
    ],
)
def test_live_audit_requires_exact_relative_source_report(
    live_audit, recording_case, reference
):
    case = recording_case
    case.selected[CURATOR]["source_report"]["artifact"] = reference
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="exact run-relative report"):
        _audit(live_audit, case)


@pytest.mark.parametrize("alter_uploaded_bytes", [False, True])
def test_live_audit_requires_hash_of_exact_uploaded_bytes(
    live_audit, recording_case, alter_uploaded_bytes
):
    case = recording_case
    if alter_uploaded_bytes:
        case.bodies[CURATOR] += b"\n"
    else:
        case.selected[CURATOR]["source_report"]["sha256"] = "0" * 64
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="source-report bytes differ"):
        _audit(live_audit, case)


@pytest.mark.parametrize(
    "private",
    [
        {"note": BUCKET},
        {"note": "s3://external-synthetic-bucket/private/report.json"},
        {"runtime": {"hostname": "synthetic-private-worker"}},
        {"endpoint_url": "https://private-endpoint.invalid"},
        {"note": "https://private-endpoint.invalid/report.json"},
        {"note": "https://private-endpoint.invalid/report?X-Amz-Signature=synthetic"},
        {"note": "file:///private/operator/report.json"},
        {"note": "/private/operator/report.json"},
    ],
)
def test_live_audit_rejects_private_recording_metadata(
    live_audit, recording_case, private
):
    case = recording_case
    case.selected[CURATION].update(private)
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="Recording text"):
        _audit(live_audit, case)


@pytest.mark.parametrize(
    "field",
    [
        "aws_access_key_id",
        "aws_session_token",
        "authorization",
        "secret_key",
        "Authorization",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "credential",
        "credentials",
        "cookie",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
        "access_key",
        "api_key",
        "client_secret",
        "access_key_id",
        "secret_access_key",
        "session_token",
        "iam_token",
        "hf_token",
        "ngc_api_key",
        "nebius_iam_token",
    ],
)
def test_live_audit_rejects_credential_metadata(live_audit, recording_case, field):
    case = recording_case
    value = (
        "Bearer synthetic-private-token"
        if field.lower() == "authorization"
        else "synthetic-private-credential"
    )
    case.selected[CURATION]["metadata"] = {field: value}
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(
        AssertionError, match="Recording text contains private metadata"
    ):
        _audit(live_audit, case)


def test_live_audit_does_not_treat_semantic_identifier_values_as_credential_fields(
    live_audit, recording_case
):
    case = recording_case
    case.selected[CURATOR]["model"] = "example/aws_access_key_id"
    _write_recording(case.path, case.video, case.selected)

    _audit(live_audit, case)


def test_live_audit_rejects_matching_nonreal_engine_in_source_and_recording(
    live_audit, recording_case
):
    case = recording_case
    source = json.loads(case.bodies[CURATOR])
    source["engine"] = case.selected[CURATOR]["engine"] = "report-only"
    case.bodies[CURATOR] = json.dumps(source).encode()
    case.selected[CURATOR]["source_report"]["sha256"] = hashlib.sha256(
        case.bodies[CURATOR]
    ).hexdigest()
    _write_recording(case.path, case.video, case.selected)

    with pytest.raises(AssertionError, match="real engine"):
        _audit(live_audit, case)


@pytest.mark.parametrize(
    "defect",
    ["video", "frame-count", "timestamp", "caption", "disposition", "video-hash"],
)
def test_live_audit_preserves_video_timeline_caption_and_disposition_checks(
    live_audit, recording_case, defect
):
    case = recording_case
    _write_recording(case.path, case.video, case.selected, defect=defect)
    if defect == "video-hash":
        case.variants[0]["temporal_alignment"]["generated_sha256"] = "0" * 64

    with pytest.raises(AssertionError):
        _audit(live_audit, case)

"""Decode COLMAP capture lineage from the existing NuRec Rerun adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from npa.workflows import data_factory_viz as viz


COLMAP_DOCUMENTS = {
    "source/attribution.json": {
        "dataset": "synthetic-colmap-fixture",
        "revision": "a" * 40,
        "sha256": "b" * 64,
        "creator": "Synthetic test fixture",
        "license": "CC-BY-4.0",
        "selected_capture": "fixture",
        "source_counts": {"images": 4, "cameras": 2, "points": 12},
    },
    "ncore/sequence/conversion.json": {
        "schema_version": 1,
        "status": "ok",
        "engine": "nvidia-ncore-colmap",
        "converter": {"revision": "c" * 40, "license": "Apache-2.0"},
        "source": {
            "archive_sha256": "b" * 64,
            "origin_points_filtered": 2,
            "counts": {"images": 4, "poses": 4, "cameras": 2, "points": 10},
        },
        "counts": {"images": 4, "poses": 4, "cameras": 2, "points": 10},
        "poses_component_group": "npa_rig",
        "time_mapping": "upstream assigns per-camera image order timestamps at 1 FPS",
        "point_filter": "upstream excludes float32 SfM points whose norm is <= 1e-6",
    },
    "ncore/sequence/npa-rig.json": {
        "status": "ok",
        "reference_camera": "camera2",
        "pose_count": 2,
        "cameras": ["camera1", "camera2"],
        "already_present": False,
        "poses_component_group": "npa_rig",
    },
}


@pytest.fixture
def colmap_run(tmp_path):
    root = tmp_path / "colmap-run"
    for relative, payload in COLMAP_DOCUMENTS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    frame = root / "novel_views/camera2/000001.png"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 50, 60)).save(frame)
    return root


def test_new_artifact_layout_supplies_capture_and_rig_lineage(colmap_run):
    stage_log = []
    docs = viz._load_nurec_docs(colmap_run, stage_log)
    assert "synthetic-colmap-fixture" in docs["pipeline/1_ncore"]
    assert "CC-BY-4.0" in docs["provenance/source"]
    assert "1 FPS" in docs["provenance/conversion"]
    assert "SfM" in docs["provenance/conversion"]
    assert "camera2" in docs["provenance/rig"]
    assert "npa_rig" in docs["provenance/rig"]
    assert any("ncore:" in line for line in stage_log)


def test_conversion_takes_precedence_over_a_stale_fetch_manifest(colmap_run):
    (colmap_run / "ncore/manifest.json").write_text('{"scene": "stale-scene"}')
    docs = viz._load_nurec_docs(colmap_run, [])
    assert "synthetic-colmap-fixture" in docs["pipeline/1_ncore"]
    assert "stale-scene" not in docs["pipeline/1_ncore"]


def test_legacy_fetch_with_rig_sidecar_retains_its_capture_summary(colmap_run):
    (colmap_run / "source/attribution.json").unlink()
    (colmap_run / "ncore/sequence/conversion.json").unlink()
    (colmap_run / "ncore/manifest.json").write_text('{"scene": "legacy-capture"}')
    assert "legacy-capture" in viz._load_nurec_docs(colmap_run, [])["pipeline/1_ncore"]


def test_rrd_decodes_actual_lineage_documents_and_image_rows(colmap_run, tmp_path):
    from rerun.recording import load_recording

    out = tmp_path / "sim2real.rrd"
    viz.build_run_rrd(str(colmap_run), str(out), app_id="neural-reconstruction")
    verified = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "verify", str(out)],
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    recording = load_recording(out)
    assert recording.application_id() == "neural-reconstruction"
    assert recording.recording_id() == colmap_run.name
    chunks = list(recording.chunks())
    for entity, relative in (
        ("source", "source/attribution.json"),
        ("conversion", "ncore/sequence/conversion.json"),
        ("rig", "ncore/sequence/npa-rig.json"),
    ):
        batches = [
            c.to_record_batch()
            for c in chunks
            if str(c.entity_path) == f"/provenance/{entity}"
        ]
        texts = [
            row[0] for b in batches for row in b.column("TextDocument:text").to_pylist()
        ]
        assert len(texts) == 1
        assert (
            hashlib.sha256((colmap_run / relative).read_bytes()).hexdigest() in texts[0]
        )
    images = [
        c.to_record_batch()
        for c in chunks
        if str(c.entity_path) == "/novel_view/camera2"
    ]
    assert sum(len(b.column("EncodedImage:blob")) for b in images) == 1
    assert images[0].column("frame").to_pylist() == [1]


def test_lineage_omits_private_locations_and_unrelated_source_members(colmap_run):
    for relative in (
        "source/attribution.json",
        "ncore/sequence/conversion.json",
        "ncore/sequence/npa-rig.json",
    ):
        path = colmap_run / relative
        payload = json.loads(path.read_text())
        payload.update(
            source_meta="/private/operator/source.json",
            output_meta="/private/operator/output.json",
            input_uri="s3://synthetic-private-bucket/capture",
            source_url="https://example.com/?token=synthetic-secret",
        )
        if "source" in payload:
            payload["source"]["members"] = [{"path": "private-customer-frame.jpg"}]
        path.write_text(json.dumps(payload))
    text = "\n".join(viz._load_nurec_docs(colmap_run, []).values())
    for private in (
        "/private/",
        "synthetic-private-bucket",
        "synthetic-secret",
        "private-customer",
    ):
        assert private not in text


@pytest.mark.parametrize(
    "relative",
    [
        "source/attribution.json",
        "ncore/sequence/conversion.json",
        "ncore/sequence/npa-rig.json",
    ],
)
def test_present_but_corrupt_lineage_fails_instead_of_silently_disappearing(
    colmap_run, relative
):
    (colmap_run / relative).write_text("{truncated")
    with pytest.raises(viz.DataFactoryVizError, match="lineage"):
        viz._load_nurec_docs(colmap_run, [])


def test_source_materialization_fetches_only_attribution(tmp_path):
    downloads = []

    class Storage:
        def download_path(self, uri, destination):
            downloads.append(uri)

        def download_file(self, uri, destination):
            downloads.append(uri)

    viz._materialize_run("s3://unit-bucket/run", tmp_path, storage_client=Storage())
    assert "s3://unit-bucket/run/source/attribution.json" in downloads
    assert "s3://unit-bucket/run/source/" not in downloads

"""Offline unit tests only: synthetic data, fake S3, and no live submissions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec

BUCKET = "unit-bucket"
RUN_ID = "unit-run"
COLMAP = "nurec-colmap-reconstruct.yaml"
EXPLICIT_ROOT = "unit-authorized/Exact_root.v1"


class FakeS3:
    """In-memory unit fixture; never constructs an SDK client."""

    def __init__(self):
        self.objects = {}
        self.writes = []
        self.reads = []

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        assert Bucket == BUCKET
        self.objects[Key] = Body
        self.writes.append(Key)

    def upload_file(self, filename, bucket, key):
        self.put_object(Bucket=bucket, Key=key, Body=Path(filename).read_bytes())

    def get_object(self, *, Bucket, Key):
        assert Bucket == BUCKET
        self.reads.append(Key)
        return {"Body": BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket, Key):
        assert Bucket == BUCKET
        self.reads.append(Key)
        return {"ContentLength": len(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == BUCKET
        yield {
            "Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]
        }

    def download_file(self, bucket, key, filename):
        with self.get_object(Bucket=bucket, Key=key)["Body"] as body:
            Path(filename).write_bytes(body.read())


@pytest.fixture
def helpers(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "e2e/npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("unit_live_prefix_helpers", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for variable in (
        "NPA_E2E_S3_PREFIX",
        "NPA_E2E_NUREC_COLMAP_ARCHIVE",
        "NPA_E2E_FORCE_ACCELERATORS",
        "NPA_E2E_ACCELERATOR_REMAP",
        "NPA_WORKFLOW_GPU_ACCELERATOR",
        "NPA_E2E_CLOUD_REMAP",
        "NPA_E2E_RELAX_CPU_MEM",
        "NPA_E2E_BDD100K_SYNTHETIC_ROWS",
        "NPA_E2E_BDD100K_EPOCHS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        module,
        "_download_nurec_colmap_archive",
        Mock(side_effect=AssertionError("unit tests must not download data")),
    )
    return module


@pytest.fixture(autouse=True)
def storage(monkeypatch):
    client = FakeS3()
    factory = Mock(return_value=client)
    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", factory
    )
    return client, factory


@pytest.fixture(params=[None, EXPLICIT_ROOT], ids=["legacy", "explicit"])
def root(request, helpers, monkeypatch):
    if request.param is not None:
        monkeypatch.setenv("NPA_E2E_S3_PREFIX", request.param)
        return request.param
    return f"npa-workflow-e2e/{RUN_ID}"


@pytest.fixture
def source_archive(helpers, monkeypatch, tmp_path):
    # Synthetic unit archive, not NVIDIA data or evidence of a valid conversion.
    archive = tmp_path / "unit-source.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("unit-fixture.txt", "synthetic unit data")
    monkeypatch.setenv("NPA_E2E_NUREC_COLMAP_ARCHIVE", str(archive))
    monkeypatch.setattr(
        helpers, "NUREC_COLMAP_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    return archive


def test_colmap_materialization_scopes_actual_plan_paths(helpers, root, tmp_path):
    path = helpers.materialize_live_spec(tmp_path, COLMAP, bucket=BUCKET, run_id=RUN_ID)
    spec = load_spec(path)
    expected = f"{root}/nurec-colmap-reconstruct"
    assert spec.config["prefix"] == expected
    assert spec.config["bucket"] == BUCKET
    plan = build_plan(spec, run_id=RUN_ID)
    steps = {step.state: step for step in plan.steps}
    uri = f"s3://{BUCKET}/{expected}/"
    assert steps["convert"].inputs == [
        {"uri": uri + "source/struktur28_colmap.zip", "schema": ""}
    ]
    assert steps["convert"].outputs == [
        {"uri": uri + "ncore/sequence/sequence.json", "schema": ""},
        {"uri": uri + "ncore/sequence/conversion.json", "schema": ""},
    ]
    assert steps["reconstruct"].inputs == [
        {"uri": uri + "ncore/sequence/", "schema": ""}
    ]
    assert steps["visualize"].outputs[0]["uri"] == uri + "reports/sim2real.rrd"
    assert steps["finalize"].outputs[0]["uri"] == uri + "reports/final.json"
    for step in plan.steps:
        for item in [*step.inputs, *step.outputs]:
            assert item["uri"].startswith(uri)
        for argument in step.argv:
            if argument.startswith("s3://"):
                assert argument.startswith(uri)


def test_colmap_source_uploads_share_materialized_root(
    helpers, root, source_archive, storage, tmp_path
):
    helpers.seed_live_workflow_inputs(spec_name=COLMAP, bucket=BUCKET, run_id=RUN_ID)
    client, _ = storage
    source = f"{root}/nurec-colmap-reconstruct/source/"
    assert client.writes == [
        source + "struktur28_colmap.zip",
        source + "attribution.json",
    ]
    assert (
        client.objects[source + "struktur28_colmap.zip"] == source_archive.read_bytes()
    )
    attribution = json.loads(client.objects[source + "attribution.json"])
    assert attribution["sha256"] == helpers.NUREC_COLMAP_SHA256
    assert attribution["revision"] == helpers.NUREC_COLMAP_REVISION
    assert attribution["license"] == "CC-BY-4.0"
    path = helpers.materialize_live_spec(tmp_path, COLMAP, bucket=BUCKET, run_id=RUN_ID)
    plan = build_plan(load_spec(path), run_id=RUN_ID)
    assert plan.steps[0].inputs[0]["uri"] == f"s3://{BUCKET}/{client.writes[0]}"


def test_colmap_bad_source_hash_still_fails_before_upload(
    helpers, root, source_archive, storage
):
    source_archive.write_bytes(b"tampered synthetic archive")
    with pytest.raises(
        pytest.fail.Exception, match="differs from the pinned complete dataset"
    ):
        helpers.seed_live_workflow_inputs(
            spec_name=COLMAP, bucket=BUCKET, run_id=RUN_ID
        )
    assert storage[0].writes == []


def _unit_colmap_outputs(helpers, root, client):
    """Invented provenance for readback unit tests; not a real NCore artifact."""
    prefix = f"{root}/nurec-colmap-reconstruct/"
    sequence = prefix + "ncore/sequence/"
    members = {
        "sequence.json": json.dumps(
            {"version": "v4", "component_stores": [{"path": "unit-store.bin"}]}
        ).encode(),
        "npa-rig.json": b'{"unit_fixture": true}',
        "unit-store.bin": b"synthetic component bytes",
    }
    counts = {"images": 518, "poses": 518, "cameras": 3, "points": 163453}
    report = {
        "status": "ok",
        "engine": "nvidia-ncore-colmap",
        "converter": {"revision": "59c698d206da92b406a4f72619fce3b3a2c64bfd"},
        "source": {
            "archive_sha256": helpers.NUREC_COLMAP_SHA256,
            "counts": counts,
            "origin_points_filtered": 0,
        },
        "options": {"dataset_root": "struktur28"},
        "poses_component_group": "npa_rig",
        "counts": counts,
        "members": [
            {
                "path": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            for name, body in members.items()
        ],
    }
    client.objects.update({sequence + name: body for name, body in members.items()})
    client.objects[sequence + "conversion.json"] = json.dumps(report).encode()
    client.objects[prefix + "source/attribution.json"] = json.dumps(
        {"revision": helpers.NUREC_COLMAP_REVISION, "license": "CC-BY-4.0"}
    ).encode()
    client.objects[prefix + "reports/final.json"] = json.dumps(
        {"has_usdz": True, "has_novel_views": True, "has_rrd": True}
    ).encode()
    client.objects[prefix + "reconstruction/metrics.yaml"] = b"unit: true"
    client.objects[prefix + "reconstruction/parsed.yaml"] = b"unit: true"
    client.objects[prefix + "reconstruction/last.usdz"] = b"unit fixture, not a USDZ"
    client.objects[prefix + "novel_views/camera1/000000.png"] = b"unit frame fixture"
    client.objects[prefix + "reports/sim2real.rrd"] = b"unit fixture, not a real RRD"
    return sequence


def test_colmap_readback_uses_same_root_for_every_artifact(
    helpers, root, storage, monkeypatch
):
    client, _ = storage
    _unit_colmap_outputs(helpers, root, client)
    # Actual format readback is exercised in test_nurec_colmap_workflow; this
    # fixture isolates legacy/explicit object-root routing across every download.
    decode = Mock()
    monkeypatch.setattr(helpers, "_assert_nurec_downstream_proof", decode)
    helpers.assert_nurec_colmap_live_outputs(bucket=BUCKET, run_id=RUN_ID)
    decode.assert_called_once()
    assert decode.call_args.kwargs == {"recording_id": "nurec-colmap-reconstruct"}
    assert set(client.reads) == set(client.objects)
    assert client.writes == []


def test_colmap_readback_still_rejects_member_hash_mismatch(helpers, root, storage):
    client, _ = storage
    sequence = _unit_colmap_outputs(helpers, root, client)
    key = sequence + "unit-store.bin"
    client.objects[key] = b"X" * len(client.objects[key])
    with pytest.raises(AssertionError, match="published sequence member hash differs"):
        helpers.assert_nurec_colmap_live_outputs(bucket=BUCKET, run_id=RUN_ID)
    assert key in client.reads


@pytest.mark.parametrize(
    "name",
    [
        "token-factory-generate.yaml",
        "paidf-cosmos3.yaml",
        "token-factory-gate-loop.yaml",
    ],
)
def test_other_marker_seeds_match_materialization(
    helpers, root, storage, tmp_path, name
):
    path = helpers.materialize_live_spec(tmp_path, name, bucket=BUCKET, run_id=RUN_ID)
    expected = f"{root}/{Path(name).stem}"
    assert load_spec(path).config["prefix"] == expected
    helpers.seed_live_workflow_inputs(spec_name=name, bucket=BUCKET, run_id=RUN_ID)
    assert storage[0].writes
    assert all(key.startswith(expected + "/") for key in storage[0].writes)


def test_delayed_trigger_captures_materialized_root(
    helpers, root, storage, monkeypatch, tmp_path
):
    # A mock Timer captures the callback; no thread or background process starts.
    timer_factory = Mock()
    monkeypatch.setattr("threading.Timer", timer_factory)
    name = "token-factory-trigger-watch.yaml"
    path = helpers.materialize_live_spec(tmp_path, name, bucket=BUCKET, run_id=RUN_ID)
    helpers.seed_trigger_inbox_later(bucket=BUCKET, run_id=RUN_ID, spec_name=name)
    timer_factory.return_value.start.assert_called_once_with()
    assert storage[0].writes == []
    monkeypatch.setenv("NPA_E2E_S3_PREFIX", "unit-unrelated/later-root")
    timer_factory.call_args.args[1]()
    prefix = load_spec(path).config["prefix"]
    assert prefix == f"{root}/{Path(name).stem}"
    assert storage[0].writes == [f"{prefix}/inbox/frame_{i:03d}.png" for i in range(2)]


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        " ",
        " unit-root",
        "unit-root ",
        "unit root",
        "unit\troot",
        "unit\nroot",
        "/",
        "/unit-root",
        "unit-root/",
        "unit//root",
        ".",
        "..",
        "unit/./root",
        "unit/../root",
        "s3://unit-bucket/unit-root",
        "https://unit.invalid/root",
        "unit\\root",
        "unit?query",
        "unit#fragment",
        "unit/%2e%2e/root",
        'unit/"root',
        "unit/{{run.id}}",
        "unit/$ROOT",
        "unit/\x7froot",
    ],
)
@pytest.mark.parametrize("operation", ["materialize", "seed", "readback", "delayed"])
def test_invalid_explicit_prefix_fails_before_any_write_or_client(
    helpers, source_archive, storage, monkeypatch, tmp_path, invalid, operation
):
    monkeypatch.setenv("NPA_E2E_S3_PREFIX", invalid)
    timer_factory = Mock()
    monkeypatch.setattr("threading.Timer", timer_factory)
    with pytest.raises(ValueError, match="NPA_E2E_S3_PREFIX"):
        if operation == "materialize":
            helpers.materialize_live_spec(
                tmp_path, COLMAP, bucket=BUCKET, run_id=RUN_ID
            )
        elif operation == "seed":
            helpers.seed_live_workflow_inputs(
                spec_name=COLMAP, bucket=BUCKET, run_id=RUN_ID
            )
        elif operation == "readback":
            helpers.assert_nurec_colmap_live_outputs(bucket=BUCKET, run_id=RUN_ID)
        else:
            helpers.seed_trigger_inbox_later(
                bucket=BUCKET, run_id=RUN_ID, spec_name=COLMAP
            )
    client, factory = storage
    factory.assert_not_called()
    timer_factory.assert_not_called()
    assert not (tmp_path / COLMAP).exists()
    assert client.writes == client.reads == []


def test_shared_sonic_compatibility_constant_is_not_rebased(helpers, root):
    assert "SONIC_MOTION_FIXTURE_PREFIX" in helpers.__all__
    assert helpers.SONIC_MOTION_FIXTURE_PREFIX == (
        "npa-workflow-e2e/fixtures/sonic-motion-soma-g1/"
    )
    assert not helpers.SONIC_MOTION_FIXTURE_PREFIX.startswith(root + "/")


@pytest.mark.parametrize(
    "name,shared_prefix",
    [
        ("sim2real-two-step.yaml", f"sim2real-triggers/{RUN_ID}/lerobot-pusht/"),
        ("insights-smoke.yaml", "insights-fixtures/run/"),
        ("insights-aggregate.yaml", f"runs/{RUN_ID}/"),
    ],
)
def test_legacy_fixture_roots_outside_config_prefix_remain_explicit_exceptions(
    helpers, root, storage, tmp_path, name, shared_prefix
):
    path = helpers.materialize_live_spec(tmp_path, name, bucket=BUCKET, run_id=RUN_ID)
    plan = build_plan(load_spec(path), run_id=RUN_ID)
    assert any(
        argument.startswith(f"s3://{BUCKET}/{shared_prefix}")
        for step in plan.steps
        for argument in step.argv
    )
    helpers.seed_live_workflow_inputs(spec_name=name, bucket=BUCKET, run_id=RUN_ID)
    assert storage[0].writes
    assert all(key.startswith(shared_prefix) for key in storage[0].writes)
    assert all(not key.startswith(root + "/") for key in storage[0].writes)

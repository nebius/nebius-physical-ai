"""Exercise PAIDF recovery against the production storage client and S3 protocol."""

import hashlib
import io
from urllib.parse import urlsplit

from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
import pytest

from npa.clients.storage import StorageClient
from npa.workflows import paidf_variant_recovery as recovery

OUTPUT = "s3://example-bucket/unit-run/cosmos_augmented/"
IMAGE = "ghcr.io/example/npa-cosmos3@sha256:" + "a" * 64
MODEL = {"revision": "b" * 40, "files": {"fixture": {"sha256": "c" * 64}}}


@pytest.fixture
def storage():
    client = StorageClient(
        endpoint_url="https://storage.example.invalid",
        aws_access_key_id="unit-key",
        aws_secret_access_key="unit-secret",
    )
    with Stubber(client.s3) as stubber:
        yield client, stubber
        stubber.assert_no_pending_responses()


def _object_params(uri):
    parsed = urlsplit(uri)
    return {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}


def _expect_read(stubber, uri, payload):
    params = _object_params(uri)
    if payload is None:
        stubber.add_client_error(
            "get_object",
            service_error_code="NoSuchKey",
            http_status_code=404,
            expected_params=params,
        )
        return
    stubber.add_response(
        "get_object",
        {
            "Body": StreamingBody(io.BytesIO(payload), len(payload)),
            "ETag": '"unit-etag"',
        },
        params,
    )


def _expect_create(stubber, uri, payload, conflict=False, content_type=None):
    params = {
        **_object_params(uri),
        "Body": payload,
        "IfNoneMatch": "*",
        "ContentType": content_type or "application/octet-stream",
    }
    if conflict:
        stubber.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params=params,
        )
        return
    stubber.add_response("put_object", {"ETag": '"unit-etag"'}, params)


@pytest.fixture
def native_batch(monkeypatch):
    monkeypatch.setattr(recovery, "_source_binding", lambda: {"fixture.py": "d" * 64})
    monkeypatch.setattr(recovery, "_hardware_binding", lambda _env: ["fixture-gpu,1,1"])
    identity = {"input_sha256": "e" * 64, "input_provenance": {"sha256": "e" * 64}}
    bound = {
        **identity,
        "runtime_image": IMAGE,
        "npa_sources": {"fixture.py": "d" * 64},
        "gpu_runtime": ["fixture-gpu,1,1"],
    }
    return identity, {
        "schema": recovery._BATCH_SCHEMA,
        "identity": bound,
        "model_binding": MODEL,
    }


def test_batch_uses_real_client_and_preserves_pinned_revision(storage, native_batch):
    client, stubber = storage
    identity, batch = native_batch
    uri = OUTPUT + "_generation-recovery/attempt-00/batch.json"
    _expect_read(stubber, uri, None)
    _expect_create(stubber, uri, ANY, content_type="application/json")
    _expect_read(stubber, uri, recovery._json_bytes(batch))
    _expect_read(stubber, uri, recovery._json_bytes(batch))
    calls = []

    def snapshot(mode, revision, _environ):
        calls.append((mode, revision))
        return (
            {"revision": MODEL["revision"]}
            if mode == "resolve"
            else {"model_binding": MODEL}
        )

    kwargs = {
        "storage": client,
        "output_uri": OUTPUT,
        "attempt": 0,
        "identity": identity,
        "environ": {"NPA_TASK_IMAGE": IMAGE},
        "snapshot_command": snapshot,
    }
    first = recovery.VariantRecovery(**kwargs)
    second = recovery.VariantRecovery(**kwargs)
    assert first.batch == second.batch == batch
    assert calls == [
        ("resolve", ""),
        ("materialize", MODEL["revision"]),
        ("materialize", MODEL["revision"]),
    ]


@pytest.mark.parametrize("conflict", [False, True])
@pytest.mark.parametrize("readback", [b"actual native object", b"changed bytes", None])
def test_publication_uses_real_conditional_create_and_readback(
    storage,
    tmp_path,
    conflict,
    readback,
):
    client, stubber = storage
    source = tmp_path / "native-object.bin"
    source.write_bytes(b"actual native object")
    uri = OUTPUT + "immutable/native-object.bin"
    _expect_create(stubber, uri, source.read_bytes(), conflict=conflict)
    _expect_read(stubber, uri, readback)
    publication = recovery.ImmutableVariantPublication(client, OUTPUT + "immutable/")
    if readback != source.read_bytes():
        with pytest.raises(recovery.VariantRecoveryError, match="byte readback"):
            publication.upload_file(str(source), uri)
        assert publication.objects == {}
        return
    assert publication.upload_file(str(source), uri) == uri
    assert publication.objects["native-object.bin"]["bytes"] == len(readback)


@pytest.mark.parametrize("changed", [False, True])
def test_recovered_inventory_reads_real_client_objects(storage, changed):
    client, stubber = storage
    names = [
        "augmented_video.mp4",
        "metadata.json",
        "transfer.json",
        "source_edges.mkv",
        "native_execution.json",
        "source_rgb.mkv",
        "frame-00001.png",
    ]
    payloads = {name: ("explicit unit fixture " + name).encode() for name in names}
    receipt = {
        "variant": {"frame_count": 1},
        "objects": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    selected = names[:1] if changed else names
    for name in selected:
        _expect_read(stubber, OUTPUT + name, b"changed" if changed else payloads[name])
    coordinator = recovery.VariantRecovery.__new__(recovery.VariantRecovery)
    coordinator.storage = client
    if changed:
        with pytest.raises(recovery.VariantRecoveryError, match="object bytes changed"):
            coordinator._verify_objects(receipt, OUTPUT, {"rgb_weight": 0.25})
        return
    assert (
        coordinator._verify_objects(receipt, OUTPUT, {"rgb_weight": 0.25}) == payloads
    )

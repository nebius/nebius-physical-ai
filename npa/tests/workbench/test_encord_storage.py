from __future__ import annotations

import hashlib
import sys
import base64
from pathlib import Path

import pytest

from npa.workbench.encord.schemas import EncordToolError
from npa.workbench.encord.storage import (
    ArtifactConflict,
    ArtifactInvalid,
    ArtifactVersion,
    ConditionalArtifactStore,
    S3ObjectStorageGateway,
    TransferDigest,
    checksum_from_head,
    head_object,
)

sys.path.insert(0, str(Path(__file__).parents[1]))

from encord_fakes import FakeStorageClient, StreamingBody  # noqa: E402


class ConditionalClient:
    def __init__(self) -> None:
        self.calls = []
        self.values = {}
        self.tokens = {}
        self.s3 = self

    def put_bytes_conditional(self, payload, uri, **kwargs):
        self.calls.append((uri, kwargs))
        if kwargs.get("if_none_match") and uri in self.values:
            from npa.clients.storage import StoragePreconditionFailed

            raise StoragePreconditionFailed("exists")
        if kwargs.get("if_match") and kwargs["if_match"] != self.tokens.get(uri):
            from npa.clients.storage import StoragePreconditionFailed

            raise StoragePreconditionFailed("stale")
        token = f'"etag-{len(self.calls)}"'
        self.values[uri] = payload
        self.tokens[uri] = token
        return token

    def read_bytes_with_etag(self, uri):
        if uri not in self.values:
            return None
        return self.values[uri], self.tokens[uri]

    def head_object(self, *, Bucket, Key):
        payload = self.values[f"s3://{Bucket}/{Key}"]
        return {
            "ContentLength": len(payload),
            "ContentType": "application/json",
            "ETag": self.tokens[f"s3://{Bucket}/{Key}"],
        }


def test_s3_create_and_replace_use_exact_etag_guards() -> None:
    client = ConditionalClient()
    store = ConditionalArtifactStore(client)
    uri = "s3://result-bucket/run/receipt.json"
    first = store.create_json(uri, {"revision": 0})
    second = store.replace_json(uri, {"revision": 1}, first)
    assert first == ArtifactVersion(kind="s3_etag", token='"etag-1"')
    assert second == ArtifactVersion(kind="s3_etag", token='"etag-2"')
    assert client.calls[0][1]["if_none_match"] is True
    assert client.calls[1][1]["if_match"] == '"etag-1"'


def test_s3_stale_replace_is_an_artifact_conflict() -> None:
    client = ConditionalClient()
    store = ConditionalArtifactStore(client)
    uri = "s3://result-bucket/run/receipt.json"
    store.create_json(uri, {"revision": 0})
    with pytest.raises(ArtifactConflict):
        store.replace_json(uri, {"revision": 1}, ArtifactVersion(kind="s3_etag", token='"stale"'))


def test_local_create_does_not_overwrite_and_stale_replace_fails(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    store = ConditionalArtifactStore(ConditionalClient())
    version = store.create_json(str(target), {"revision": 0})
    with pytest.raises(ArtifactConflict):
        store.create_json(str(target), {"revision": 99})
    store.replace_json(str(target), {"revision": 1}, version)
    with pytest.raises(ArtifactConflict):
        store.replace_json(str(target), {"revision": 2}, version)
    assert '"revision": 1' in target.read_text()


def test_invalid_json_and_top_level_arrays_are_rejected(tmp_path: Path) -> None:
    store = ConditionalArtifactStore(ConditionalClient())
    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"not-json")
    with pytest.raises(ArtifactInvalid):
        store.read_json(str(malformed))
    malformed.write_text("[]")
    with pytest.raises(ArtifactInvalid):
        store.read_json(str(malformed))


@pytest.mark.parametrize(
    "uri",
    ["s3://bucket/a/../receipt.json", "s3://bucket/receipt.json?token=x", "s3:///missing.json"],
)
def test_unsafe_artifact_uri_fails_before_storage_call(uri: str) -> None:
    client = ConditionalClient()
    with pytest.raises((EncordToolError, ValueError)):
        ConditionalArtifactStore(client).create_json(uri, {"revision": 0})
    assert client.calls == []


def test_opaque_etag_is_never_labeled_as_a_content_hash() -> None:
    checksum, kind = checksum_from_head({"ETag": '"d41d8cd98f00b204e9800998ecf8427e"'})
    assert checksum == "d41d8cd98f00b204e9800998ecf8427e"
    assert kind == "etag_opaque"


def test_head_preserves_opaque_etag_separately() -> None:
    client = ConditionalClient()
    uri = "s3://result-bucket/run/receipt.json"
    version = ConditionalArtifactStore(client).create_json(uri, {"revision": 0})
    metadata = head_object(client, uri)
    assert metadata.etag_kind == "etag_opaque"
    assert metadata.etag == version.token.strip('"')


def test_object_gateway_lists_every_page_and_heads_each_object() -> None:
    client = FakeStorageClient()
    client.s3.objects[("source-bucket", "incoming/a.mp4")] = b"a"
    client.s3.objects[("source-bucket", "incoming/b.mp4")] = b"bb"
    rows = list(S3ObjectStorageGateway(client).list_objects("s3://source-bucket/incoming/"))
    assert [row.size for row in rows] == [1, 2]
    assert sum(event[0] == "head" for event in client.s3.events) == 2


def test_object_gateway_preserves_literal_percent_encoded_key() -> None:
    client = FakeStorageClient()
    literal_key = "incoming/a%2Fb.mp4"
    client.s3.objects[("source-bucket", literal_key)] = b"video"
    rows = list(
        S3ObjectStorageGateway(client).list_objects(
            "s3://source-bucket/incoming/"
        )
    )
    assert rows[0].uri == "s3://source-bucket/incoming/a%252Fb.mp4"
    assert ("head", "source-bucket", literal_key) in client.s3.events


def test_streamed_download_hashes_and_leaves_no_completed_file_on_failure(
    tmp_path: Path,
) -> None:
    client = FakeStorageClient()
    client.s3.objects[("source-bucket", "incoming/clip.mp4")] = b"video"
    destination = tmp_path / "clip.mp4"
    digest = S3ObjectStorageGateway(client).download_to_file(
        "s3://source-bucket/incoming/clip.mp4", destination
    )
    assert digest.size == 5
    assert digest.sha256 == hashlib.sha256(b"video").hexdigest()

    def interrupted(*, Bucket, Key):
        del Bucket, Key
        return {"Body": StreamingBody(b"partial", fail=True)}

    client.s3.get_object = interrupted  # type: ignore[method-assign]
    failed_destination = tmp_path / "failed.mp4"
    with pytest.raises(OSError):
        S3ObjectStorageGateway(client).download_to_file(
            "s3://source-bucket/incoming/clip.mp4", failed_destination
        )
    assert not failed_destination.exists()


def test_upload_requires_matching_destination_head(tmp_path: Path) -> None:
    client = FakeStorageClient()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    identity = ("result-bucket", "run/clip.mp4")
    client.s3.wrong_size_after_write.add(identity)
    with pytest.raises(EncordToolError, match="size did not verify"):
        S3ObjectStorageGateway(client).upload_file(
            source, "s3://result-bucket/run/clip.mp4"
        )


def test_copy_rejects_compatible_full_checksum_mismatch() -> None:
    client = FakeStorageClient()
    source = ("source-bucket", "incoming/clip.mp4")
    destination = ("result-bucket", "run/clip.mp4")
    client.s3.objects[source] = b"same"
    client.s3.head_overrides[source] = {
        "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"same").digest()).decode(),
        "ChecksumType": "FULL_OBJECT",
    }
    client.s3.head_overrides[destination] = {
        "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"diff").digest()).decode(),
        "ChecksumType": "FULL_OBJECT",
    }
    with pytest.raises(EncordToolError, match="checksum did not verify"):
        S3ObjectStorageGateway(client).copy(
            "s3://source-bucket/incoming/clip.mp4",
            "s3://result-bucket/run/clip.mp4",
        )


def test_upload_rejects_compatible_destination_checksum_mismatch(
    tmp_path: Path,
) -> None:
    client = FakeStorageClient()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    identity = ("result-bucket", "run/clip.mp4")
    client.s3.head_overrides[identity] = {
        "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"wrong").digest()).decode(),
        "ChecksumType": "FULL_OBJECT",
    }
    with pytest.raises(EncordToolError, match="checksum did not verify"):
        S3ObjectStorageGateway(client).upload_file(
            source,
            "s3://result-bucket/run/clip.mp4",
            digest=TransferDigest(
                size=5, sha256=hashlib.sha256(b"video").hexdigest()
            ),
        )

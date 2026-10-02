from __future__ import annotations

import hashlib
from io import BytesIO

from botocore.exceptions import ClientError
import pytest

from npa.workflows.s3_input_preflight import (
    S3InputPreflightError,
    preflight_s3_inputs,
)


class TrackedBody:
    def __init__(self, payload: bytes) -> None:
        self.stream = BytesIO(payload)
        self.closed = False
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        assert size > 0
        self.read_sizes.append(size)
        return self.stream.read(size)

    def close(self) -> None:
        self.closed = True
        self.stream.close()


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[TrackedBody] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self.calls.append((Bucket, Key))
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from error
        body = TrackedBody(payload)
        self.bodies.append(body)
        return {
            "Body": body,
            "ContentLength": len(payload),
            "ETag": '"etag"',
        }

    def head_object(self, **_kwargs) -> None:
        raise AssertionError("HEAD size is not exact-byte verification")


class FakeStorage:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.s3 = FakeS3(objects)


def _declaration(uri: str, payload: bytes) -> dict[str, object]:
    return {
        "uri": uri,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_streams_exact_bytes_with_selected_client_and_closes_body() -> None:
    payload = b"verified input"
    storage = FakeStorage({("bucket", "inputs/model.bin"): payload})

    rows = preflight_s3_inputs(
        [_declaration("s3://bucket/inputs/model.bin", payload)], storage=storage
    )

    assert rows[0]["verified"] is True
    assert rows[0]["actual"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert storage.s3.calls == [("bucket", "inputs/model.bin")]
    assert storage.s3.bodies[0].closed is True
    assert max(storage.s3.bodies[0].read_sizes) == 8 * 1024 * 1024


def test_reports_every_missing_and_identity_mismatch() -> None:
    right = b"right"
    wrong_size = b"too long"
    wrong_hash = b"different"
    storage = FakeStorage(
        {
            ("bucket", "wrong-size"): wrong_size,
            ("bucket", "wrong-hash"): wrong_hash,
        }
    )
    inputs = [
        _declaration("s3://bucket/missing", b"missing"),
        _declaration("s3://bucket/wrong-size", right),
        _declaration("s3://bucket/wrong-hash", b"same-size"),
    ]

    with pytest.raises(S3InputPreflightError) as raised:
        preflight_s3_inputs(inputs, storage=storage)

    assert [row["status"] for row in raised.value.rows] == [
        "missing",
        "mismatch",
        "mismatch",
    ]
    assert raised.value.rows[1]["mismatches"] == ["bytes", "sha256"]
    assert raised.value.rows[2]["mismatches"] == ["sha256"]
    assert storage.s3.calls == [
        ("bucket", "missing"),
        ("bucket", "wrong-size"),
        ("bucket", "wrong-hash"),
    ]
    assert all(body.closed for body in storage.s3.bodies)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uri", "s3://bucket/prefix/"),
        ("uri", "s3://bucket/a/../b"),
        ("uri", "s3://bucket/key?version=1"),
        ("sha256", "ABC"),
        ("bytes", True),
        ("bytes", -1),
    ],
)
def test_invalid_declarations_fail_before_network(field: str, value: object) -> None:
    payload = b"input"
    declaration = _declaration("s3://bucket/input", payload)
    declaration[field] = value
    storage = FakeStorage({("bucket", "input"): payload})

    with pytest.raises(S3InputPreflightError) as raised:
        preflight_s3_inputs([declaration], storage=storage)

    assert raised.value.rows[0]["status"] == "invalid"
    assert storage.s3.calls == []


def test_all_invalid_rows_are_reported_without_network() -> None:
    storage = FakeStorage({})

    with pytest.raises(S3InputPreflightError) as raised:
        preflight_s3_inputs(
            [
                {"uri": "not-s3", "sha256": "x", "bytes": 1},
                {"uri": "s3://bucket/key", "sha256": "0" * 64, "bytes": "1"},
            ],
            storage=storage,
        )

    assert len(raised.value.rows) == 2
    assert storage.s3.calls == []


def test_empty_declarations_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        preflight_s3_inputs([], storage=FakeStorage({}))


def test_provider_declared_length_must_match_streamed_bytes() -> None:
    payload = b"input"
    storage = FakeStorage({("bucket", "input"): payload})
    original = storage.s3.get_object

    def get_object(**kwargs) -> dict:
        response = original(**kwargs)
        response["ContentLength"] = len(payload) + 1
        return response

    storage.s3.get_object = get_object

    with pytest.raises(S3InputPreflightError) as raised:
        preflight_s3_inputs(
            [_declaration("s3://bucket/input", payload)], storage=storage
        )

    assert raised.value.rows[0]["mismatches"] == ["provider_content_length"]

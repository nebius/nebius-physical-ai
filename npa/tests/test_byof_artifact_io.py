"""Tests for create-only BYOF stage artifact writes."""

from __future__ import annotations

import pytest

from npa.workflows.byof import artifact_io
from npa.clients.storage import StoragePreconditionFailed


class _FakeStorage:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def put_bytes_conditional(self, payload, bucket_uri, **kwargs):
        self.calls.append({"payload": payload, "bucket_uri": bucket_uri, **kwargs})
        return '"etag"'


def test_write_bytes_to_s3_is_create_only() -> None:
    storage = _FakeStorage()

    written = artifact_io.write_bytes(
        "s3://evidence-bucket/runs/841/report.json",
        b'{"status":"completed"}\n',
        content_type="application/json",
        storage=storage,  # type: ignore[arg-type]
    )

    assert written == "s3://evidence-bucket/runs/841/report.json"
    assert storage.calls == [
        {
            "payload": b'{"status":"completed"}\n',
            "bucket_uri": "s3://evidence-bucket/runs/841/report.json",
            "if_none_match": True,
            "content_type": "application/json",
        }
    ]


def test_write_bytes_resumes_only_an_exact_s3_replay() -> None:
    class ExistingStorage(_FakeStorage):
        def __init__(self, existing: bytes) -> None:
            super().__init__()
            self.existing = existing

        def put_bytes_conditional(self, *args, **kwargs):
            raise StoragePreconditionFailed("already exists")

        def read_bytes_with_etag(self, _output_path):
            return self.existing, '"etag"'

    same = ExistingStorage(b"{}\n")
    assert (
        artifact_io.write_bytes(
            "s3://evidence-bucket/runs/841/report.json",
            b"{}\n",
            content_type="application/json",
            storage=same,  # type: ignore[arg-type]
        )
        == "s3://evidence-bucket/runs/841/report.json"
    )

    with pytest.raises(artifact_io.StorageError, match="non-identical artifact"):
        artifact_io.write_bytes(
            "s3://evidence-bucket/runs/841/report.json",
            b'{"different":true}\n',
            content_type="application/json",
            storage=ExistingStorage(b"{}\n"),  # type: ignore[arg-type]
        )


def test_write_bytes_local_is_for_hermetic_stage_tests(tmp_path) -> None:
    output = tmp_path / "artifact.json"

    assert artifact_io.write_bytes(
        str(output), b"{}\n", content_type="application/json"
    ) == str(output)
    assert output.read_bytes() == b"{}\n"


@pytest.mark.parametrize(
    "value",
    ("s3://bucket/", "s3:///missing-bucket/report.json", "https://host/report.json"),
)
def test_write_bytes_rejects_non_object_remote_paths(value) -> None:
    with pytest.raises(artifact_io.ArtifactPathError, match="exact local file or s3"):
        artifact_io.write_bytes(value, b"{}", content_type="application/json")


def test_join_output_path_preserves_s3_prefix() -> None:
    assert (
        artifact_io.join_output_path(
            "s3://evidence-bucket/runs/841/compare", "compare.json"
        )
        == "s3://evidence-bucket/runs/841/compare/compare.json"
    )

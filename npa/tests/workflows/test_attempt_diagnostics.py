from __future__ import annotations

from pathlib import Path

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.attempt_diagnostics import publish_failed_attempt


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []
        self.corrupt_readback_for = ""

    def put_bytes_conditional(
        self, payload: bytes, uri: str, *, if_none_match: bool
    ) -> str:
        assert if_none_match is True
        if uri in self.objects:
            raise StoragePreconditionFailed("object already exists")
        self.objects[uri] = payload
        self.writes.append(uri)
        return f"etag-{len(self.writes)}"

    def read_bytes_with_etag(self, uri: str):
        payload = self.objects.get(uri)
        if payload is None:
            return None
        if uri == self.corrupt_readback_for:
            payload += b"changed"
        return payload, "etag"


def test_publishes_exact_files_before_final_receipt(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    partial = tmp_path / "partial.json"
    log.write_bytes(b"raw worker output\n")
    partial.write_bytes(b'{"partial":true}\n')
    storage = MemoryStorage()

    receipt = publish_failed_attempt(
        "s3://bucket/run/diagnostics",
        run_id="run-1",
        stage="policy-smoke",
        attempt_id="attempt-2",
        exit_code=7,
        files={"worker.log": log, "partial/output.json": partial},
        storage=storage,
    )

    receipt_uri = receipt["receipt_uri"]
    assert storage.writes[-1] == receipt_uri
    assert receipt["status"] == "failed"
    assert receipt["classification"] == ("diagnostic_only_not_success_or_qualification")
    assert receipt["files"]["worker.log"]["sha256"] == (
        "a7895671b100f16aca7eca6cdeebd5f589549bb4b39cacdf1cf2efacccf142ea"
    )
    assert (
        storage.objects[
            "s3://bucket/run/diagnostics/failed-attempts/"
            "run-1/policy-smoke/attempt-2/originals/partial/output.json"
        ]
        == partial.read_bytes()
    )


@pytest.mark.parametrize(
    "name",
    [".", "../escape", "nested/../../escape", "/absolute", "a\\b", "a//b", "bad?name"],
)
def test_rejects_noncanonical_destination_names(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    source.write_text("diagnostic")
    storage = MemoryStorage()

    with pytest.raises(ValueError, match="canonical relative path"):
        publish_failed_attempt(
            "s3://bucket/diagnostics",
            run_id="run-1",
            stage="stage-1",
            attempt_id="attempt-1",
            exit_code=1,
            files={"valid-first.log": source, name: source},
            storage=storage,
        )
    assert storage.writes == []


def test_rejects_symlinks_and_success_exit_codes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("diagnostic")
    link = tmp_path / "link"
    link.symlink_to(source)
    arguments = {
        "run_id": "run-1",
        "stage": "stage-1",
        "attempt_id": "attempt-1",
        "files": {"worker.log": link},
        "storage": MemoryStorage(),
    }

    with pytest.raises(ValueError, match="nonzero"):
        publish_failed_attempt("s3://bucket/diagnostics", exit_code=0, **arguments)
    with pytest.raises(ValueError, match="regular file"):
        publish_failed_attempt("s3://bucket/diagnostics", exit_code=1, **arguments)


def test_readback_mismatch_never_publishes_receipt(tmp_path: Path) -> None:
    source = tmp_path / "worker.log"
    source.write_text("diagnostic")
    storage = MemoryStorage()
    file_uri = (
        "s3://bucket/diagnostics/failed-attempts/run-1/stage-1/attempt-1/"
        "originals/worker.log"
    )
    storage.corrupt_readback_for = file_uri

    with pytest.raises(ValueError, match="readback differs"):
        publish_failed_attempt(
            "s3://bucket/diagnostics",
            run_id="run-1",
            stage="stage-1",
            attempt_id="attempt-1",
            exit_code=1,
            files={"worker.log": source},
            storage=storage,
        )

    assert not any(uri.endswith("failure-diagnostic.json") for uri in storage.objects)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({}, "canonical relative path"),
        ({"run_id": "../run"}, "run_id"),
        ({"stage": ""}, "stage"),
        ({"attempt_id": "attempt/1"}, "attempt_id"),
        ({"exit_code": True}, "nonzero exit code"),
        ({"exit_code": "1"}, "nonzero exit code"),
        ({"diagnostic_root_uri": "s3:///missing"}, "canonical s3"),
        ({"diagnostic_root_uri": "s3://bucket//root"}, "canonical s3"),
        ({"diagnostic_root_uri": "s3://bucket/."}, "canonical s3"),
        ({"diagnostic_root_uri": "s3://bucket/a/../b"}, "canonical s3"),
        ({"diagnostic_root_uri": "s3://bucket/root?q=1"}, "canonical s3"),
    ],
)
def test_invalid_metadata_or_name_writes_nothing(
    tmp_path: Path, override: dict[str, object], match: str
) -> None:
    source = tmp_path / "worker.log"
    source.write_text("diagnostic")
    storage = MemoryStorage()
    arguments = {
        "diagnostic_root_uri": "s3://bucket/diagnostics",
        "run_id": "run-1",
        "stage": "stage-1",
        "attempt_id": "attempt-1",
        "exit_code": 1,
        "files": {"worker.log": source, ".": source},
        "storage": storage,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=match):
        publish_failed_attempt(**arguments)

    assert storage.writes == []


def test_run_id_separates_identical_stage_attempt_names(tmp_path: Path) -> None:
    source = tmp_path / "worker.log"
    source.write_text("diagnostic")
    storage = MemoryStorage()

    for run_id in ("run-1", "run-2"):
        publish_failed_attempt(
            "s3://bucket/diagnostics",
            run_id=run_id,
            stage="stage-1",
            attempt_id="attempt-1",
            exit_code=1,
            files={"worker.log": source},
            storage=storage,
        )

    receipt_uris = [
        uri for uri in storage.objects if uri.endswith("failure-diagnostic.json")
    ]
    assert len(receipt_uris) == 2
    assert any("/run-1/stage-1/attempt-1/" in uri for uri in receipt_uris)
    assert any("/run-2/stage-1/attempt-1/" in uri for uri in receipt_uris)


def test_partial_retry_accepts_identical_existing_original(tmp_path: Path) -> None:
    source = tmp_path / "worker.log"
    source.write_bytes(b"original bytes")
    storage = MemoryStorage()
    original_uri = (
        "s3://bucket/diagnostics/failed-attempts/run-1/stage-1/attempt-1/"
        "originals/worker.log"
    )
    storage.objects[original_uri] = source.read_bytes()

    receipt = publish_failed_attempt(
        "s3://bucket/diagnostics",
        run_id="run-1",
        stage="stage-1",
        attempt_id="attempt-1",
        exit_code=1,
        files={"worker.log": source},
        storage=storage,
    )

    assert receipt["status"] == "failed"
    assert storage.objects[original_uri] == b"original bytes"


def test_partial_retry_rejects_different_existing_original(tmp_path: Path) -> None:
    source = tmp_path / "worker.log"
    source.write_bytes(b"expected bytes")
    storage = MemoryStorage()
    original_uri = (
        "s3://bucket/diagnostics/failed-attempts/run-1/stage-1/attempt-1/"
        "originals/worker.log"
    )
    storage.objects[original_uri] = b"different bytes"

    with pytest.raises(ValueError, match="readback differs"):
        publish_failed_attempt(
            "s3://bucket/diagnostics",
            run_id="run-1",
            stage="stage-1",
            attempt_id="attempt-1",
            exit_code=1,
            files={"worker.log": source},
            storage=storage,
        )

    assert not any(uri.endswith("failure-diagnostic.json") for uri in storage.objects)

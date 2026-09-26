from __future__ import annotations

from pathlib import Path
import os

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


def _arguments(tmp_path: Path, storage: MemoryStorage) -> dict:
    source = tmp_path / "worker.log"
    source.write_bytes(b"original bytes")
    return dict(
        diagnostic_root_uri="s3://bucket/diagnostics",
        run_id="run-1",
        stage="stage-1",
        attempt_id="attempt-1",
        exit_code=1,
        files={"worker.log": source},
        storage=storage,
    )


@pytest.mark.parametrize("suffix", ["\n", "\t", "\x00", "\x7f", " ", "?", "#"])
def test_rejects_uri_parser_normalization(tmp_path: Path, suffix: str) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    arguments["diagnostic_root_uri"] += suffix
    with pytest.raises(ValueError, match="canonical s3"):
        publish_failed_attempt(**arguments)
    assert storage.writes == []


@pytest.mark.parametrize(
    "bucket", ["bad..bucket", "bad.-bucket", "bad-.bucket", "127.0.0.1"]
)
def test_rejects_invalid_bucket_labels(tmp_path: Path, bucket: str) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    arguments["diagnostic_root_uri"] = f"s3://{bucket}/diagnostics"
    with pytest.raises(ValueError, match="canonical s3"):
        publish_failed_attempt(**arguments)
    assert storage.writes == []


@pytest.mark.parametrize("changed_field", ["files", "exit_code"])
def test_completed_attempt_conflict_writes_nothing(
    tmp_path: Path, changed_field: str
) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    receipt = publish_failed_attempt(**arguments)
    original_objects = dict(storage.objects)
    original_writes = list(storage.writes)
    assert publish_failed_attempt(**arguments) == receipt
    if changed_field == "files":
        arguments["files"]["extra.log"] = arguments["files"]["worker.log"]
    else:
        arguments["exit_code"] = 2
    with pytest.raises(ValueError, match="completed diagnostic attempt differs"):
        publish_failed_attempt(**arguments)
    assert storage.objects == original_objects
    assert storage.writes == original_writes


def test_interrupted_manifest_reserves_entire_file_set(tmp_path: Path) -> None:
    class InterruptedStorage(MemoryStorage):
        interrupted = False

        def put_bytes_conditional(self, payload, uri, *, if_none_match):
            if "/originals/" in uri and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("interrupted after manifest")
            return super().put_bytes_conditional(
                payload, uri, if_none_match=if_none_match
            )

    storage = InterruptedStorage()
    arguments = _arguments(tmp_path, storage)
    with pytest.raises(ConnectionError):
        publish_failed_attempt(**arguments)
    original_objects = dict(storage.objects)
    assert len(original_objects) == 1
    assert next(iter(original_objects)).endswith("attempt-manifest.json")
    conflicting = {
        **arguments,
        "files": {"extra.log": arguments["files"]["worker.log"]},
    }
    with pytest.raises(ValueError, match="readback differs"):
        publish_failed_attempt(**conflicting)
    assert storage.objects == original_objects
    receipt = publish_failed_attempt(**arguments)
    assert receipt["status"] == "failed"
    assert storage.writes[-1].endswith("failure-diagnostic.json")


def test_competing_writer_cannot_add_different_originals(tmp_path: Path) -> None:
    class CompetingStorage(MemoryStorage):
        def put_bytes_conditional(self, payload, uri, *, if_none_match):
            result = super().put_bytes_conditional(
                payload, uri, if_none_match=if_none_match
            )
            if uri.endswith("attempt-manifest.json"):
                competing = {
                    **arguments,
                    "files": {"extra.log": arguments["files"]["worker.log"]},
                }
                with pytest.raises(ValueError, match="readback differs"):
                    publish_failed_attempt(**competing)
            return result

    storage = CompetingStorage()
    arguments = _arguments(tmp_path, storage)
    publish_failed_attempt(**arguments)
    assert not any(uri.endswith("extra.log") for uri in storage.objects)


def test_completed_receipt_requires_original_readback(tmp_path: Path) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    receipt = publish_failed_attempt(**arguments)
    storage.corrupt_readback_for = receipt["files"]["worker.log"]["uri"]
    original_writes = list(storage.writes)
    with pytest.raises(ValueError, match="original readback differs"):
        publish_failed_attempt(**arguments)
    assert storage.writes == original_writes


def test_file_swap_to_symlink_is_rejected(tmp_path: Path, monkeypatch) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    source = arguments["files"]["worker.log"]
    target = tmp_path / "unselected"
    target.write_text("not selected")
    original_open = os.open

    def swapped_open(path, flags):
        source.unlink()
        source.symlink_to(target)
        return original_open(path, flags)

    monkeypatch.setattr("npa.workflows.attempt_diagnostics.os.open", swapped_open)
    with pytest.raises(ValueError, match="regular file"):
        publish_failed_attempt(**arguments)
    assert storage.writes == []


def test_fifo_is_rejected_without_reading(tmp_path: Path) -> None:
    storage = MemoryStorage()
    arguments = _arguments(tmp_path, storage)
    source = arguments["files"]["worker.log"]
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular file"):
        publish_failed_attempt(**arguments)
    assert storage.writes == []

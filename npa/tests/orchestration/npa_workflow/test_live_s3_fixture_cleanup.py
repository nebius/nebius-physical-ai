"""Live fixture cleanup preserves evidence and cannot target another run."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path

from botocore.exceptions import ClientError
import pytest


RUN_ID = "live-infra-" + "a" * 32
PREFIX = f"npa-workflow-e2e/{RUN_ID}/"


@pytest.fixture
def cleanup(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    module = importlib.import_module("tests.e2e.test_npa_workflow_live_infra")
    return module._archive_and_cleanup_live_prefix


class Storage:
    def __init__(self, *, broken_read=None, broken_delete=None, outside_listing=False):
        self.objects = {
            PREFIX + "manifest.json": b'{"status":"failed","evidence":"synthetic"}\n',
            PREFIX + "state.json": b'{"status":"complete"}\n',
            "npa-workflow-e2e/live-infra-" + "b" * 32 + "/manifest.json": b"other run",
        }
        self.broken_read = broken_read
        self.broken_delete = broken_delete
        self.outside_listing = outside_listing
        self.calls = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        yield self.list_objects_v2(**kwargs)

    def list_objects_v2(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"Contents": [{"Key": key} for key in self.objects
                             if key.startswith(kwargs["Prefix"]) or self.outside_listing]}

    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        if kwargs["Key"] == self.broken_read:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    def delete_object(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if kwargs["Key"] == self.broken_delete:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "DeleteObject")
        del self.objects[kwargs["Key"]]

    def head_object(self, **kwargs):
        self.calls.append(("head", kwargs))
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}


def test_archives_complete_private_bytes_before_scoped_cleanup(cleanup, tmp_path):
    storage = Storage()
    original = dict(storage.objects)
    evidence = tmp_path / "private-evidence"
    result = cleanup(storage, bucket="example-bucket", run_id=RUN_ID, evidence_dir=evidence)

    assert result["all_owned_objects_absent"] and not result["errors"]
    assert len(result["objects"]) == 2
    assert evidence.stat().st_mode & 0o077 == 0
    assert (evidence / "cleanup.json").stat().st_mode & 0o077 == 0
    for row in result["objects"]:
        archive = evidence / row["archive_file"]
        assert archive.read_bytes() == original[row["key"]]
        assert row["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert archive.stat().st_mode & 0o077 == 0
        assert row["archived"] and row["deleted"]
        get = next(i for i, (op, args) in enumerate(storage.calls) if op == "get" and args["Key"] == row["key"])
        delete = next(i for i, (op, args) in enumerate(storage.calls) if op == "delete" and args["Key"] == row["key"])
        assert get < delete
    assert list(storage.objects.values()) == [b"other run"]
    assert all(args["Prefix"] == PREFIX for op, args in storage.calls if op == "list")


@pytest.mark.parametrize("operation", ["read", "delete"])
def test_failure_preserves_recovery_evidence_and_cleans_other_owned_keys(cleanup, tmp_path, operation):
    failed_key = PREFIX + "manifest.json"
    storage = Storage(**{"broken_" + operation: failed_key})
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        cleanup(storage, bucket="example-bucket", run_id=RUN_ID, evidence_dir=tmp_path)

    result = json.loads((tmp_path / "cleanup.json").read_text())
    failed, successful = result["objects"]
    assert failed["key"] == failed_key and not failed["deleted"]
    assert failed["archived"] is (operation == "delete")
    assert successful["archived"] and successful["deleted"]
    assert failed_key in storage.objects and PREFIX + "state.json" not in storage.objects
    assert len(storage.objects) == 2  # failed owned object plus untouched other run
    assert result["errors"] and not result["all_owned_objects_absent"]
    if operation == "read":
        assert not any(op == "delete" and args["Key"] == failed_key for op, args in storage.calls)


def test_unexpected_outside_key_aborts_before_archival_or_delete(cleanup, tmp_path):
    storage = Storage(outside_listing=True)
    with pytest.raises(RuntimeError, match="inventory failed"):
        cleanup(storage, bucket="example-bucket", run_id=RUN_ID, evidence_dir=tmp_path)
    assert len(storage.objects) == 3
    assert all(op == "list" for op, _ in storage.calls)
    assert json.loads((tmp_path / "cleanup.json").read_text())["errors"]


@pytest.mark.parametrize("run_id", ["", "live-infra-", "../other-run", RUN_ID + "/parent"])
def test_invalid_scope_is_rejected_before_any_storage_call(cleanup, tmp_path, run_id):
    storage = Storage()
    with pytest.raises(ValueError, match="UUID"):
        cleanup(storage, bucket="example-bucket", run_id=run_id, evidence_dir=tmp_path)
    assert storage.calls == []

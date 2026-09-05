"""Artifact verification must finish before the client signals Ray session stop."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest


@pytest.fixture
def recipe(monkeypatch):
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("finish_worker", "finish", "submit", "validation")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    worker = importlib.import_module("finish_worker")
    client = importlib.import_module("finish")
    yield SimpleNamespace(worker=worker, client=client)
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.corrupt_reads = False

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, Metadata):
        assert IfNoneMatch == "*"
        self.puts.append((Bucket, Key))
        if (Bucket, Key) in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Bucket, Key] = Body.read() if hasattr(Body, "read") else Body

    def get_object(self, *, Bucket, Key):
        body = self.objects[Bucket, Key]
        return {"Body": io.BytesIO(body + b"corruption" if self.corrupt_reads else body)}


def test_upload_selects_all_regular_files_and_verifies_idempotent_manifest(recipe, tmp_path):
    root = tmp_path / "outputs"
    (root / "lance").mkdir(parents=True)
    (root / "report.json").write_text('{"rows":2}')
    (root / "lance" / "shard.lance").write_bytes(b"real artifact bytes")
    storage = MemoryS3()
    first = recipe.worker.upload_artifacts(storage, root, "s3://example-bucket/run/artifacts")
    again = recipe.worker.upload_artifacts(storage, root, "s3://example-bucket/run/artifacts")
    assert first == again
    assert first["file_count"] == 2 and first["all_objects_read_after_write_verified"]
    manifest = storage.objects["example-bucket", "run/artifacts/manifest.json"]
    assert hashlib.sha256(manifest).hexdigest() == first["manifest_sha256"]
    assert [row["path"] for row in json.loads(manifest)["files"]] == ["lance/shard.lance", "report.json"]


def test_hash_failure_prevents_manifest_publication(recipe, tmp_path):
    (tmp_path / "artifact").write_bytes(b"output")
    storage = MemoryS3()
    storage.corrupt_reads = True
    with pytest.raises(ValueError, match="read-after-write"):
        recipe.worker.upload_artifacts(storage, tmp_path, "s3://example-bucket/artifacts")
    assert ("example-bucket", "artifacts/manifest.json") not in storage.objects


@pytest.mark.parametrize("link_directory", [False, True])
def test_symlink_selection_fails_before_any_upload(recipe, tmp_path, link_directory):
    root, outside = tmp_path / "outputs", tmp_path / "private"
    root.mkdir()
    outside.mkdir()
    (outside / "data").write_bytes(b"outside")
    (root / "link").symlink_to(outside if link_directory else outside / "data", target_is_directory=link_directory)
    storage = MemoryS3()
    with pytest.raises(ValueError, match="Symbolic links"):
        recipe.worker.upload_artifacts(storage, root, "s3://example-bucket/artifacts")
    assert storage.puts == []


def test_parent_symlink_swap_cannot_escape_selected_root(recipe, tmp_path):
    root, outside = tmp_path / "outputs", tmp_path / "private"
    (root / "part").mkdir(parents=True)
    outside.mkdir()
    (outside / "data").write_bytes(b"outside")
    (root / "part").rmdir()
    (root / "part").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        recipe.worker.open_relative(root, "part/data")
    with pytest.raises(ValueError, match="relative"):
        recipe.worker.open_relative(root, "../private/data")


@pytest.mark.parametrize("uri", ["https://example.com/data", "s3://bucket/", "s3://bucket/a/../b", "s3://bucket/prefix?token=value"])
def test_destination_requires_safe_specific_s3_scope(recipe, uri):
    with pytest.raises(ValueError):
        recipe.worker.parse_s3(uri)


class Jobs:
    def __init__(self, report, status="SUCCEEDED"):
        self.report = report
        self.status = status
        self.observed_terminal = False

    def submit_job(self, **kwargs):
        self.submission = kwargs
        self.identity = kwargs["submission_id"]
        assert set(path.name for path in Path(kwargs["runtime_env"]["working_dir"]).iterdir()) == {"finish_worker.py"}
        assert set(kwargs["runtime_env"]) == {"working_dir"}
        assert "stop" not in kwargs["entrypoint"]
        return self.identity

    def get_job_info(self, identity):
        assert identity == self.identity
        self.observed_terminal = True
        return SimpleNamespace(status=self.status)

    def get_job_logs(self, identity):
        assert identity == self.identity
        return "RAY_CLIP_ARTIFACTS " + json.dumps(self.report)


def arguments(tmp_path):
    return SimpleNamespace(artifact_uri="s3://example-bucket/run/artifacts", stop_uri="s3://example-bucket/run/control/finished.json",
                           output_path="/outputs/run", python="/app/env/bin/python", evidence_dir=str(tmp_path))


def test_stop_marker_is_client_only_after_terminal_job_and_manifest_proof(recipe, tmp_path):
    storage = MemoryS3()
    payload = b'{"files":[]}'
    digest = hashlib.sha256(payload).hexdigest()
    storage.objects["example-bucket", "run/artifacts/manifest.json"] = payload
    jobs = Jobs({"manifest_uri": "s3://example-bucket/run/artifacts/manifest.json", "manifest_sha256": digest,
                 "manifest_bytes": len(payload), "all_objects_read_after_write_verified": True})
    original_put = storage.put_object
    def observed_put(**kwargs):
        assert jobs.observed_terminal
        original_put(**kwargs)
    storage.put_object = observed_put
    result = recipe.client.finish(jobs, storage, arguments(tmp_path))
    marker = json.loads(storage.objects["example-bucket", "run/control/finished.json"])
    assert marker == {"finished": True, "manifest_sha256": digest}
    assert result["stop_marker_read_after_write_verified"]
    assert json.loads((tmp_path / "finish.json").read_text())["status"] == "SUCCEEDED"


@pytest.mark.parametrize("status", ["FAILED", "STOPPED"])
def test_failed_or_cancelled_upload_job_never_writes_stop(recipe, tmp_path, status):
    storage = MemoryS3()
    jobs = Jobs({}, status)
    with pytest.raises(RuntimeError, match="finish was not signaled"):
        recipe.client.finish(jobs, storage, arguments(tmp_path))
    assert storage.puts == []
    assert not json.loads((tmp_path / "finish.json").read_text())["stop_marker_written"]


def test_client_manifest_hash_failure_never_writes_stop(recipe, tmp_path):
    storage = MemoryS3()
    storage.objects["example-bucket", "run/artifacts/manifest.json"] = b"corrupt"
    jobs = Jobs({"manifest_uri": "s3://example-bucket/run/artifacts/manifest.json", "manifest_sha256": "0" * 64,
                 "manifest_bytes": 7, "all_objects_read_after_write_verified": True})
    with pytest.raises(ValueError, match="read-after-write"):
        recipe.client.finish(jobs, storage, arguments(tmp_path))
    assert storage.puts == []


def test_stop_marker_cannot_overlap_artifact_prefix(recipe, tmp_path):
    args = arguments(tmp_path)
    args.stop_uri = args.artifact_uri + "/stop.json"
    with pytest.raises(ValueError, match="outside"):
        recipe.client.finish(Jobs({}), MemoryS3(), args)


def test_marker_read_failure_preserves_ambiguous_write_evidence(recipe, tmp_path):
    storage = MemoryS3()
    payload = b"manifest"
    digest = hashlib.sha256(payload).hexdigest()
    storage.objects["example-bucket", "run/artifacts/manifest.json"] = payload
    jobs = Jobs({"manifest_uri": "s3://example-bucket/run/artifacts/manifest.json", "manifest_sha256": digest,
                 "manifest_bytes": len(payload), "all_objects_read_after_write_verified": True})
    original_get = storage.get_object
    def fail_marker_read(**kwargs):
        if kwargs["Key"].endswith("finished.json"):
            raise OSError("read response lost")
        return original_get(**kwargs)
    storage.get_object = fail_marker_read
    with pytest.raises(OSError, match="response lost"):
        recipe.client.finish(jobs, storage, arguments(tmp_path))
    saved = json.loads((tmp_path / "finish.json").read_text())
    assert saved["stop_marker_write_attempted"]
    assert saved["stop_marker_written"] is None
    assert saved["stop_marker_read_after_write_verified"] is False

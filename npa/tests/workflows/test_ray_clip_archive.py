# Exercise complete-tree immutability and restore failure boundaries with real local formats.
"""CPU regressions; vectors here are explicit synthetic fixtures, not CLIP inference."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading

import pytest

from npa.clients.storage import StoragePreconditionFailed

PREFIX = "s3://test-bucket/clip-test"


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("archive", "archive_inventory", "embed", "worker", "validation")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    yield importlib.import_module("archive"), importlib.import_module("embed")
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _checksums(root):
    files = {str(path.relative_to(root)): _hash(path.read_bytes())
             for path in root.rglob("*") if path.is_file() and path.name not in {"sha256.json", "SHA256SUMS"}}
    if (root / "sha256.json").exists():
        _json(root / "sha256.json", files)
        files["sha256.json"] = _hash((root / "sha256.json").read_bytes())
    (root / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(files.items())))


@pytest.fixture
def source(modules, tmp_path):
    _, embed = modules
    root = tmp_path / "source"
    root.mkdir(mode=0o700)
    rows = embed.worker.preprocess_shard(list(range(6)))["rows"]
    for index, row in enumerate(rows):
        row["vector"] = [float(position == index) for position in range(512)]
    report = embed.save_results(root, rows, 6)
    report["source_sha256"] = {name: "a" * 64 for name in embed.SOURCE_FILES}
    report["actors"] = [{"node_id": "synthetic-node", "source_sha256": report["source_sha256"]}]
    report["ray_nodes"] = 1
    _json(root / "report.json", report)
    embed.write_hashes(root)
    return root


@pytest.fixture
def advanced(source):
    import pyarrow.parquet as pq

    report = json.loads((source / "report.json").read_text())
    report.update(schema_version="npa.ray-clip-development.v1", batch_size=2, shards=3,
                  execution_fingerprint="e" * 64, model_revision="b" * 40,
                  source_sha256="c" * 64, gpu_actors=1, retrieval_queries=5,
                  embedding_sha256=report.pop("parquet_sha256"))
    report.update(application_sha256="a" * 64, validation_sha256="b" * 64, udf_sha256="d" * 64)
    report["model_initializations"] = [{field: report[field] for field in (
        "application_sha256", "validation_sha256", "udf_sha256", "source_sha256", "model_revision", "execution_fingerprint")}]
    report["recovery"] = None
    table = pq.read_table(source / "embeddings.parquet")
    for column, field in (("input_sha256", "input_hash"), ("processed_sha256", "processed_hash")):
        report[field] = _hash(json.dumps(table[column].to_pylist(), separators=(",", ":")).encode())
    for index, start in enumerate(range(0, 6, 2)):
        directory = source / "shards" / f"{index:06d}"
        directory.mkdir(parents=True)
        shard = table.slice(start, 2)
        pq.write_table(shard, directory / "embeddings.parquet")
        identity = {"record_ids": shard["record_id"].to_pylist(),
                    "source_sha256": report["source_sha256"], "model_revision": report["model_revision"],
                    "execution_fingerprint": report["execution_fingerprint"]}
        for column, field in (("input_sha256", "input_hash"), ("processed_sha256", "processed_hash")):
            identity[field] = _hash(json.dumps(shard[column].to_pylist(), separators=(",", ":")).encode())
        _json(directory / "commit.json", {"identity": identity, "rows": len(shard),
                                         "parquet_sha256": _hash((directory / "embeddings.parquet").read_bytes())})
    _json(source / "report.json", report)
    _json(source / "retrieval.json", [{"query_id": value, "top_ids": [value, *[other for other in range(6) if other != value]][:5]}
                                      for value in [0, 1, 3, 4, 5]])
    _json(source / "execution.json", {"execution_fingerprint": report["execution_fingerprint"]})
    _json(source / "actor-cleanup.json", {"errors": [], "attempted": 1})
    (source / "sha256.json").unlink()
    _checksums(source)
    return source


class Store:
    """Exact object store with atomic conditional creation and observable calls."""

    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()
        self.reads = []
        self.hook = None

    def put_bytes_conditional(self, content, uri, *, if_none_match):
        assert if_none_match is True
        with self.lock:
            if uri in self.objects:
                raise StoragePreconditionFailed("exists")
            self.objects[uri] = content
        if self.hook:
            self.hook(uri)
        return '"opaque-etag"'

    def read_bytes_with_etag(self, uri):
        with self.lock:
            self.reads.append(uri)
            if uri not in self.objects:
                return None
            return self.objects[uri], '"opaque-etag"'


@pytest.fixture
def store():
    return Store()


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind", ["source", "advanced"])
def test_complete_formats_retry_and_restore(modules, store, request, kind, tmp_path):
    archive, _ = modules
    source = request.getfixturevalue(kind)
    original = _bytes(source)
    receipt = archive.archive(source, PREFIX, store)
    first = dict(store.objects)
    assert receipt == archive.archive(source, PREFIX, store)
    assert first == store.objects
    # No source/driver is present during restore, and unlisted remote files are ignored.
    shutil.rmtree(source)
    store.objects[f"{PREFIX}/unlisted"] = b"do not restore"
    result = tmp_path / "restored"
    assert archive.restore(PREFIX, result, receipt["manifest_sha256"], store) == receipt
    assert _bytes(result) == original
    assert result.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o077 == 0 for path in result.rglob("*") if path.is_file())
    assert f"{PREFIX}/unlisted" not in store.reads


def test_concurrent_identical_publishers(modules, source, store):
    archive, _ = modules
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(archive.archive, source, PREFIX, store) for _ in range(2)]
        assert futures[0].result() == futures[1].result()


def test_conflicting_publish_preserves_every_existing_byte(modules, source, store):
    archive, _ = modules
    archive.archive(source, PREFIX, store)
    existing = dict(store.objects)
    report = json.loads((source / "report.json").read_text())
    report["annotation"] = "different complete result metadata"
    _json(source / "report.json", report)
    _checksums(source)
    with pytest.raises(ValueError, match="conflicting"):
        archive.archive(source, PREFIX, store)
    assert store.objects == existing


def test_interrupted_upload_has_no_completion_and_retry_works(modules, source, store):
    archive, _ = modules
    def interrupt(_uri):
        raise OSError("injected interrupted upload")
    store.hook = interrupt
    with pytest.raises(OSError):
        archive.archive(source, PREFIX, store)
    assert f"{PREFIX}/complete.json" not in store.objects
    store.hook = None
    archive.archive(source, PREFIX, store)
    assert f"{PREFIX}/complete.json" in store.objects


@pytest.mark.parametrize("mutation", ["bytes", "added", "removed", "linked", "root"])
def test_source_mutation_never_publishes_completion(modules, source, store, mutation):
    archive, _ = modules
    def change(_uri):
        store.hook = None
        path = source / "preview.png"
        if mutation == "bytes":
            path.write_bytes(b"changed")
        elif mutation == "added":
            (source / "extra").write_bytes(b"extra")
        elif mutation == "removed":
            path.unlink()
        elif mutation == "linked":
            path.unlink()
            path.symlink_to("report.json")
        else:
            source.rename(source.with_name("old-source"))
            source.mkdir()
    store.hook = change
    with pytest.raises((ValueError, OSError)):
        archive.archive(source, PREFIX, store)
    assert f"{PREFIX}/complete.json" not in store.objects


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "traversal", "json-duplicate", "secondary", "symlink", "hardlink", "fifo", "symlink-dir"])
def test_bad_source_is_rejected_before_any_upload(modules, source, store, damage):
    archive, _ = modules
    if damage == "missing":
        (source / "preview.png").unlink()
    elif damage == "extra":
        (source / "unexpected").write_bytes(b"x")
    elif damage in {"duplicate", "traversal"}:
        path = source / "SHA256SUMS"
        extra = path.read_text().splitlines()[0] if damage == "duplicate" else "a" * 64 + "  ../escape"
        path.write_text(path.read_text() + extra + "\n")
    elif damage == "json-duplicate":
        path = source / "report.json"
        path.write_text(path.read_text().rstrip()[:-1] + ', "records": 3}')
        _checksums(source)
    elif damage == "secondary":
        _json(source / "sha256.json", {})
        # Keep the outer checksum honest while making the secondary inventory wrong.
        lines = (source / "SHA256SUMS").read_text().splitlines()
        lines = [(_hash((source / "sha256.json").read_bytes()) + "  sha256.json")
                 if line.endswith("  sha256.json") else line for line in lines]
        (source / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    elif damage == "symlink":
        (source / "link").symlink_to("report.json")
    elif damage == "hardlink":
        os.link(source / "report.json", source / "linked")
    elif damage == "fifo":
        os.mkfifo(source / "fifo")
    else:
        (source / "linked-dir").symlink_to(source / "images", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        archive.archive(source, PREFIX, store)
    assert store.objects == {}


@pytest.mark.parametrize("field", ["execution", "cleanup", "shard", "count"])
def test_inconsistent_advanced_result_is_not_complete(modules, advanced, store, field):
    archive, _ = modules
    if field == "execution":
        _json(advanced / "execution.json", {"execution_fingerprint": "f" * 64})
    elif field == "cleanup":
        _json(advanced / "actor-cleanup.json", {"errors": ["failed"], "attempted": 1})
    elif field == "shard":
        (advanced / "shards/000001/commit.json").unlink()
    else:
        report = json.loads((advanced / "report.json").read_text())
        report["shards"] += 1
        _json(advanced / "report.json", report)
    _checksums(advanced)
    with pytest.raises((ValueError, OSError)):
        archive.archive(advanced, PREFIX, store)
    assert store.objects == {}


@pytest.mark.parametrize("damage", ["missing", "corrupt", "completion", "trusted-hash"])
def test_failed_restore_has_no_destination_or_staging(modules, source, store, tmp_path, damage):
    archive, _ = modules
    receipt = archive.archive(source, PREFIX, store)
    target = f"{PREFIX}/files/preview.png"
    expected = receipt["manifest_sha256"]
    if damage == "missing":
        del store.objects[target]
    elif damage == "corrupt":
        store.objects[target] = b"broken"
    elif damage == "completion":
        del store.objects[f"{PREFIX}/complete.json"]
    else:
        expected = "a" * 64
    with pytest.raises(ValueError):
        archive.restore(PREFIX, tmp_path / "result", expected, store)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".clip-restore-*"))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/./b", "a\\b", "a\nline"])
def test_unsafe_manifest_is_rejected_before_payload_reads(modules, store, tmp_path, name):
    archive, _ = modules
    payload = archive.canonical({"schema": archive.SCHEMA, "format": "basic",
                                 "files": {name: {"size": 0, "sha256": _hash(b"")}}})
    store.objects[f"{PREFIX}/complete.json"] = payload
    with pytest.raises(ValueError):
        archive.restore(PREFIX, tmp_path / "result", _hash(payload), store)
    assert store.reads == [f"{PREFIX}/complete.json"]


def test_empty_destination_created_at_publication_is_preserved(modules, source, store, tmp_path, monkeypatch):
    archive, _ = modules
    receipt = archive.archive(source, PREFIX, store)
    target = tmp_path / "result"
    expose = archive._expose
    def race(parent, staging, destination):
        target.mkdir()
        expose(parent, staging, destination)
    monkeypatch.setattr(archive, "_expose", race)
    with pytest.raises(FileExistsError):
        archive.restore(PREFIX, target, receipt["manifest_sha256"], store)
    assert list(target.iterdir()) == []
    assert not list(tmp_path.glob(".clip-restore-*"))


def test_linked_source_ancestor_and_nonprivate_restore_parent_refused(modules, source, store, tmp_path):
    archive, _ = modules
    link = tmp_path / "linked"
    link.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(OSError):
        archive.archive(link / source.name, PREFIX, store)
    receipt = archive.archive(source, PREFIX, store)
    parent = tmp_path / "public"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        archive.restore(PREFIX, parent / "result", receipt["manifest_sha256"], store)


def test_existing_completion_does_not_hide_corruption(modules, source, store):
    archive, _ = modules
    archive.archive(source, PREFIX, store)
    store.objects[f"{PREFIX}/files/preview.png"] = b"corrupt"
    with pytest.raises(ValueError, match="conflicting"):
        archive.archive(source, PREFIX, store)
    assert store.objects[f"{PREFIX}/files/preview.png"] == b"corrupt"


@pytest.mark.parametrize("damage", ["duplicate-retrieval", "missing-sources", "actor-sources", "preview"])
def test_rehashed_inconsistent_basic_artifacts_are_rejected(modules, source, store, damage):
    archive, _ = modules
    report = json.loads((source / "report.json").read_text())
    if damage == "duplicate-retrieval":
        retrieval = json.loads((source / "retrieval.json").read_text())
        retrieval[0]["top_ids"] = [0] * 5
        report["retrieval"] = retrieval
        _json(source / "retrieval.json", retrieval)
    elif damage == "missing-sources":
        del report["source_sha256"]
    elif damage == "actor-sources":
        report["actors"][0]["source_sha256"]["embed.py"] = "b" * 64
    else:
        from PIL import Image

        Image.new("RGB", (448, 1344), color="black").save(source / "preview.png")
    _json(source / "report.json", report)
    _checksums(source)
    with pytest.raises(ValueError):
        archive.archive(source, PREFIX, store)
    assert not store.objects


@pytest.mark.parametrize("damage", ["missing", "different", "wrong-shard", "unreported"])
def test_advanced_recovery_artifact_must_match_report_and_first_shard(modules, advanced, store, damage):
    archive, _ = modules
    report = json.loads((advanced / "report.json").read_text())
    recovery = {"parquet_sha256": _hash((advanced / "shards/000000/embeddings.parquet").read_bytes())}
    report["recovery"] = recovery
    if damage == "wrong-shard":
        recovery["parquet_sha256"] = "f" * 64
    if damage != "missing":
        _json(advanced / "recovery.json", {**recovery, **({"extra": True} if damage == "different" else {})})
    if damage == "unreported":
        report["recovery"] = None
    _json(advanced / "report.json", report)
    _checksums(advanced)
    with pytest.raises(ValueError, match="recovery|Recovery"):
        archive.archive(advanced, PREFIX, store)
    assert not store.objects


def test_payload_corrupted_after_its_first_readback_prevents_completion(modules, source, store):
    archive, _ = modules
    first = None
    def corrupt_previous(uri):
        nonlocal first
        if first is None:
            first = uri
        else:
            store.objects[first] = b"corrupted after initial readback"
            store.hook = None
    store.hook = corrupt_previous
    with pytest.raises(ValueError, match="changed before completion"):
        archive.archive(source, PREFIX, store)
    assert f"{PREFIX}/complete.json" not in store.objects


def test_manifest_rejects_duplicate_keys_and_file_directory_collisions(modules, store, tmp_path):
    archive, _ = modules
    payload = b'{"schema":"npa.ray-clip-archive.v1","schema":"npa.ray-clip-archive.v1","files":{},"format":"basic"}'
    store.objects[f"{PREFIX}/complete.json"] = payload
    with pytest.raises(ValueError, match="Duplicate"):
        archive.restore(PREFIX, tmp_path / "result", _hash(payload), store)
    entry = {"sha256": _hash(b""), "size": 0}
    payload = archive.canonical({"schema": archive.SCHEMA, "format": "basic", "files": {"a": entry, "a/b": entry}})
    store.objects[f"{PREFIX}/complete.json"] = payload
    with pytest.raises(ValueError, match="conflicts with a directory"):
        archive.restore(PREFIX, tmp_path / "result", _hash(payload), store)
    assert not (tmp_path / "result").exists()


def test_cli_uses_shared_functions_and_redacts_storage_errors(modules, monkeypatch, capsys, source, store):
    archive, _ = modules
    monkeypatch.setattr(archive.StorageClient, "from_environment", lambda: store)
    assert archive.main(["archive", "--input-path", str(source), "--output-path", PREFIX]) == 0
    assert json.loads(capsys.readouterr().out)["files"] > 0
    def denied():
        raise OSError("sensitive-provider-address")
    monkeypatch.setattr(archive.StorageClient, "from_environment", denied)
    assert archive.main(["archive", "--input-path", str(source), "--output-path", PREFIX]) == 1
    assert "sensitive-provider-address" not in capsys.readouterr().err

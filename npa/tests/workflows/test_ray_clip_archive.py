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
import struct
import sys
import threading
from types import SimpleNamespace

import pytest

from npa.clients.storage import StoragePreconditionFailed

PREFIX = "s3://test-bucket/clip-test"


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("archive", "archive_inventory", "archive_lance", "embed", "worker", "validation")
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


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/./b", "a\\b", "a\nline", "a?query", "a#fragment"])
def test_unsafe_manifest_is_rejected_before_payload_reads(modules, store, tmp_path, name):
    archive, _ = modules
    payload = archive.canonical({"schema": archive.SCHEMA, "format": "basic",
                                 "files": {name: {"size": 0, "sha256": _hash(b"")}}})
    store.objects[f"{PREFIX}/complete.json"] = payload
    with pytest.raises(ValueError):
        archive.restore(PREFIX, tmp_path / "result", _hash(payload), store)
    assert store.reads == [f"{PREFIX}/complete.json"]


@pytest.mark.parametrize("suffix", ["?query", "#fragment"])
def test_url_ambiguous_source_names_are_rejected_before_upload(modules, source, store, suffix):
    from npa.clients.storage import _parse_bucket_uri

    archive, _ = modules
    assert _parse_bucket_uri(f"{PREFIX}/files/preview.png{suffix}") == _parse_bucket_uri(f"{PREFIX}/files/preview.png")
    (source / ("preview.png" + suffix)).write_bytes((source / "preview.png").read_bytes())
    _checksums(source)
    with pytest.raises(ValueError, match="Unsafe"):
        archive.archive(source, PREFIX, store)
    assert not store.objects


@pytest.mark.parametrize("actors,attempted", [(0, 0), (-1, -1), (True, True), ("all", "all"), (1, True), (2, 2)])
def test_cleanup_counts_require_real_initialized_actors(modules, advanced, store, actors, attempted):
    archive, _ = modules
    report = json.loads((advanced / "report.json").read_text())
    report["gpu_actors"] = actors
    _json(advanced / "report.json", report)
    _json(advanced / "actor-cleanup.json", {"errors": [], "attempted": attempted})
    _checksums(advanced)
    with pytest.raises(ValueError, match="cleanup"):
        archive.archive(advanced, PREFIX, store)
    assert not store.objects


def test_completed_actor_recovery_retains_replacement_initialization(modules, advanced, store, tmp_path):
    archive, _ = modules
    report = json.loads((advanced / "report.json").read_text())
    recovery = {"parquet_sha256": _hash((advanced / "shards/000000/embeddings.parquet").read_bytes())}
    report["recovery"] = recovery
    report["model_initializations"].append(dict(report["model_initializations"][0]))
    _json(advanced / "report.json", report)
    _json(advanced / "recovery.json", recovery)
    _checksums(advanced)
    receipt = archive.archive(advanced, PREFIX, store)
    restored = tmp_path / "recovered"
    assert archive.restore(PREFIX, restored, receipt["manifest_sha256"], store) == receipt
    assert _bytes(restored) == _bytes(advanced)


def test_overflowing_json_number_is_rejected_before_upload(modules, source, store):
    archive, _ = modules
    report = source / "report.json"
    report.write_text(report.read_text().rstrip()[:-1] + ', "application_seconds": 1e999}')
    _checksums(source)
    with pytest.raises(ValueError, match="Nonfinite"):
        archive.archive(source, PREFIX, store)
    assert not store.objects


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_restore_parent_swap_cannot_report_success(modules, source, store, tmp_path, replacement):
    archive, _ = modules
    receipt = archive.archive(source, PREFIX, store)
    parent = tmp_path / "publish"
    parent.mkdir(mode=0o700)
    moved = tmp_path / "moved"
    read = store.read_bytes_with_etag
    def race(uri):
        parent.rename(moved)
        if replacement == "symlink":
            parent.symlink_to(moved, target_is_directory=True)
        else:
            parent.mkdir(mode=0o700)
        store.read_bytes_with_etag = read
        return read(uri)
    store.read_bytes_with_etag = race
    with pytest.raises((ValueError, OSError)):
        archive.restore(PREFIX, parent / "result", receipt["manifest_sha256"], store)
    assert not (parent / "result").exists() and not (moved / "result").exists()
    assert not list(moved.iterdir())


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


def test_live_cleanup_attempts_all_owned_objects_and_retains_failed_ledger(tmp_path):
    path = Path(__file__).parents[1] / "e2e/test_ray_clip_archive_live.py"
    spec = importlib.util.spec_from_file_location("clip_archive_live_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attempted = []
    def delete_object(*, Bucket, Key):
        attempted.append(Key)
        if Key.endswith("first"):
            raise OSError("private-provider-detail")
    client = SimpleNamespace(s3=SimpleNamespace(delete_object=delete_object),
                             read_bytes_with_etag=lambda uri: None)
    evidence = tmp_path / "objects.json"
    storage = module._RecordedStorage(client, evidence)
    first, second = f"{PREFIX}/first", f"{PREFIX}/second"
    storage.owned = {first: {"created": True}, second: {"created": True}}
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        module._cleanup_created(storage)
    assert attempted == ["clip-test/first", "clip-test/second"]
    result = json.loads(evidence.read_text())
    assert result[first] == {"created": True, "cleanup_error_type": "OSError"}
    assert result[second]["deleted_and_absent"] is True
    assert "private-provider-detail" not in evidence.read_text()


@pytest.mark.parametrize("vector_kind", ["variable", "boolean"])
@pytest.mark.parametrize("operation", ["archive", "restore"])
def test_lance_schema_must_support_the_validated_vectors(modules, source, store, tmp_path, vector_kind, operation):
    import lancedb
    import pyarrow as pa
    import pyarrow.parquet as pq

    archive, _ = modules
    table = pq.read_table(source / "embeddings.parquet")
    rows = table.to_pylist()
    vector_type = pa.list_(pa.float32())
    if vector_kind == "boolean":
        vector_type = pa.list_(pa.bool_(), 512)
        for row in rows:
            row["vector"] = [bool(value) for value in row["vector"]]
    schema = table.schema.set(3, pa.field("vector", vector_type))
    malformed = pa.Table.from_pylist(rows, schema=schema)
    # Python row equality alone hides both changes to the native Arrow schema.
    assert malformed.to_pylist() == table.to_pylist()
    shutil.rmtree(source / "lance")
    persisted = lancedb.connect(str(source / "lance")).create_table("embeddings", data=malformed, schema=schema)
    assert not persisted.schema.equals(table.schema, check_metadata=False)
    assert persisted.to_arrow().to_pylist() == table.to_pylist()
    _checksums(source)
    if operation == "archive":
        with pytest.raises(ValueError, match="schemas differ"):
            archive.archive(source, PREFIX, store)
        assert not store.objects
        return
    # A content-bound but malformed remote tree must also fail during restore.
    files = {name: {"size": len(data), "sha256": _hash(data)} for name, data in _bytes(source).items()}
    manifest = archive.canonical({"schema": archive.SCHEMA, "format": "basic", "files": files})
    store.objects.update({f"{PREFIX}/files/{name}": data for name, data in _bytes(source).items()})
    store.objects[f"{PREFIX}/complete.json"] = manifest
    with pytest.raises(ValueError, match="schemas differ"):
        archive.restore(PREFIX, tmp_path / "unusable", _hash(manifest), store)
    assert not (tmp_path / "unusable").exists() and not list(tmp_path.glob(".clip-restore-*"))


def _reject_before_lance_open(archive, source, store, tmp_path, operation, monkeypatch):
    import lancedb

    def forbidden(*args, **kwargs):
        raise AssertionError("Lance reader ran before validating local references")

    monkeypatch.setattr(lancedb, "connect", forbidden)
    if operation == "archive":
        with pytest.raises(ValueError, match="Lance metadata"):
            archive.archive(source, PREFIX, store)
        assert not store.objects
    else:
        files = {name: {"size": len(data), "sha256": _hash(data)} for name, data in _bytes(source).items()}
        manifest = archive.canonical({"schema": archive.SCHEMA, "format": "basic", "files": files})
        store.objects.update({f"{PREFIX}/files/{name}": data for name, data in _bytes(source).items()})
        store.objects[f"{PREFIX}/complete.json"] = manifest
        with pytest.raises(ValueError, match="Lance metadata"):
            archive.restore(PREFIX, tmp_path / "unusable", _hash(manifest), store)
        assert not (tmp_path / "unusable").exists() and not list(tmp_path.glob(".clip-restore-*"))


@pytest.mark.parametrize("operation", ["archive", "restore"])
@pytest.mark.parametrize("local_decoy", [False, True])
def test_shallow_lance_references_rejected_before_reader(modules, source, store, tmp_path,
                                                        operation, local_decoy, monkeypatch):
    import lancedb
    import pyarrow.parquet as pq

    archive, _ = modules
    outside = tmp_path / "outside"
    shutil.move(source / "lance", outside)
    database = lancedb.connect(str(source / "lance"))
    clone = database.clone_table("embeddings", str(outside / "embeddings.lance"), is_shallow=True)
    expected = pq.read_table(source / "embeddings.parquet")
    assert clone.schema.equals(expected.schema, check_metadata=False)
    assert clone.to_arrow().to_pylist() == expected.to_pylist()
    assert not list((source / "lance").rglob("data/*.lance"))
    if local_decoy:
        # Presence of plausible local data is insufficient to prove references.
        shutil.copytree(outside / "embeddings.lance/data", source / "lance/embeddings.lance/data")
    shutil.rmtree(outside)
    # The invalid external dependency survived fixture setup; local decoys do not fix it.
    with pytest.raises(Exception):
        lancedb.connect(str(source / "lance")).open_table("embeddings").to_arrow()
    _checksums(source)
    _reject_before_lance_open(archive, source, store, tmp_path, operation, monkeypatch)


@pytest.mark.parametrize("identifier", [1, 2**32, 2**64 - 1])
def test_initial_lance_fragment_ids_are_contiguous_uint32(modules, source, identifier):
    from archive_lance import _fragments

    def integer(value):
        encoded = bytearray()
        while value >= 128:
            encoded.append((value & 127) | 128)
            value >>= 7
        return bytes(encoded) + bytes([value])

    path = next((source / "lance/embeddings.lance/data").glob("*.lance"))
    name = path.name.encode()
    data = (b"\x0a" + integer(len(name)) + name + b"\x12\x04\x00\x01\x02\x03"
            + b"\x1a\x04\x00\x01\x02\x03\x20\x02\x30" + integer(path.stat().st_size))
    fragment = b"\x12" + integer(len(data)) + data + b"\x20\x06"
    prefix = "lance/embeddings.lance/"
    files = {prefix + "data/" + path.name: {"size": path.stat().st_size}}
    assert _fragments([fragment], files, prefix, 6)[1] == 0
    with pytest.raises(ValueError, match="Lance metadata"):
        _fragments([b"\x08" + integer(identifier) + fragment], files, prefix, 6)


@pytest.mark.parametrize("operation", ["archive", "restore"])
@pytest.mark.parametrize("mutation", [
    "base_paths", "index", "reader_flags", "schema_metadata", "branch", "duplicate_version",
    "unknown_field", "wrong_wire", "zero_field", "truncated_length", "overlong_integer",
    "overflow_integer", "footer", "trailing", "transaction", "extra_version", "missing_data", "extra_index",
])
def test_lance_metadata_rejected_before_reader(modules, source, store, tmp_path, operation, mutation, monkeypatch):
    archive, _ = modules
    lance_root = source / "lance/embeddings.lance"
    path = next((lance_root / "_versions").glob("*.manifest"))
    content = path.read_bytes()
    offset = struct.unpack("<Q", content[-16:-8])[0]
    # Actual writer output plus independent protobuf wire records. Honest outer
    # checksums alone must never authorize a Lance reader for these variants.
    additions = {
        "base_paths": b"\x92\x01\x00", "index": b"\x30\x00", "reader_flags": b"\x48\x01",
        "schema_metadata": b"\x2a\x00", "branch": b"\xa2\x01\x00", "duplicate_version": b"\x18\x01",
        "unknown_field": b"\xf8\x07\x01", "wrong_wire": b"\x1d\x00\x00\x00\x00",
        "zero_field": b"\x00\x00", "truncated_length": b"\x92\x01\x80",
        "overlong_integer": b"\x18\x81\x00", "overflow_integer": b"\x18" + b"\xff" * 9 + b"\x02",
    }
    if mutation in additions:
        payload = content[offset + 4:-16] + additions[mutation]
        path.write_bytes(content[:offset] + struct.pack("<I", len(payload)) + payload + content[-16:])
    elif mutation == "footer":
        path.write_bytes(content[:-8] + b"\x01" + content[-7:])
    elif mutation == "trailing":
        path.write_bytes(content[:-16] + b"unexpected" + content[-16:])
    elif mutation == "transaction":
        transaction = next((lance_root / "_transactions").glob("*.txn"))
        transaction.write_bytes(transaction.read_bytes() + b"\x08\x01")
    elif mutation == "extra_version":
        (path.parent / "2.manifest").write_bytes(content)
    elif mutation == "missing_data":
        next((lance_root / "data").glob("*.lance")).unlink()
    else:
        (lance_root / "_indices").mkdir()
        (lance_root / "_indices/unknown.idx").write_bytes(b"index")
    _checksums(source)
    _reject_before_lance_open(archive, source, store, tmp_path, operation, monkeypatch)

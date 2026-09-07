# Validate factual Rerun conversion using explicit synthetic artifact fixtures.
"""These tests exercise real Parquet/PNG/RRD bytes, without claiming CUDA inference."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
import sys
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("report", "embed", "application", "validation", "worker")
    previous = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    loaded = [importlib.import_module(n) for n in names[:4]]
    yield loaded
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(previous)


def _dump(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def result(modules, tmp_path):
    converter, embed, _, _ = modules
    root = tmp_path / "result"
    root.mkdir()
    rows = embed.worker.preprocess_shard(list(range(6)))["rows"]
    for index, row in enumerate(rows):
        row["vector"] = [float(i == index) for i in range(512)]
    report = embed.save_results(root, rows, 6)
    sources = {name: "a" * 64 for name in embed.SOURCE_FILES}
    actor = {"source_sha256": sources, "model_revision": "b" * 40,
             "weight_sha256": "c" * 64, "model_config_sha256": "d" * 64,
             "model_load_seconds": 0.2, "node_id": "private-node-do-not-export",
             "imported_paths": {"private": "/private/source"}}
    report.update({"source_sha256": sources, "actors": [actor], "crop_policy": "left",
                   "preprocessors": [{"source_sha256": sources["worker.py"], "pid": "private-process"}],
                   "timings_seconds": {name: 0.5 for name in converter._BASIC_TIMINGS}})
    _dump(root / "report.json", report)
    embed.write_hashes(root)
    return root


@pytest.fixture
def advanced(modules, result):
    converter, embed, application, validation = modules
    root = result
    basic = json.loads((root / "report.json").read_text())
    table = pq.read_table(root / "embeddings.parquet")
    actor = {field: "a" * 64 for field in converter._ADVANCED_SOURCES.values()}
    actor.update({"instance_id": "private-initial-instance", "execution_fingerprint": "e" * 64,
                  "model_revision": "b" * 40, "model_config_sha256": "d" * 64,
                  "model_load_seconds": 0.2, "model_files": [{"path": "weights.bin", "sha256": "c" * 64},
                                                            {"path": "config.json", "sha256": "d" * 64}],
                  "precision": "float32", "python": "3.12.3", "ray": "2.58.0", "torch": "2.12.1",
                  "cuda": "13.0", "transformers": "5.10.2", "pyarrow": "23.0.1", "lancedb": "0.30.2",
                  "gpu_capability": [12, 0]})
    fingerprint = validation.canonical_hash({field: actor[field] for field in application.FINGERPRINT_FIELDS})
    actor["execution_fingerprint"] = fingerprint
    replacement = {**actor, "instance_id": "private-replacement-instance"}
    commits = []
    for index in range(3):
        path = root / "shards" / f"{index:06d}"
        path.mkdir(parents=True)
        shard = table.slice(index * 2, 2)
        pq.write_table(shard, path / "embeddings.parquet")
        commit = {"identity": {"record_ids": shard["record_id"].to_pylist(),
                    "input_hash": validation.canonical_hash(shard["input_sha256"].to_pylist()),
                    "processed_hash": validation.canonical_hash(shard["processed_sha256"].to_pylist()),
                    "source_sha256": "a" * 64, "model_revision": "b" * 40, "execution_fingerprint": fingerprint},
                  "rows": 2, "parquet_sha256": embed.sha256(path / "embeddings.parquet"),
                  "preprocessor": {"source_sha256": "a" * 64, "source_path": "/private/worker.py"},
                  "preprocess_seconds": 0.1, "inference": {"instance_id": actor["instance_id"] if index == 0 else replacement["instance_id"],
                    "start_monotonic_ns": 1000000000, "end_monotonic_ns": 1200000000, "inference_seconds": 0.2}}
        _dump(path / "commit.json", commit)
        commits.append(commit)
    recovery = {"old_instance": actor["instance_id"], "new_instance": replacement["instance_id"],
                "checkpoint_reused": True, "model_reloaded": True, "replay_inference_calls": 0,
                "parquet_sha256": commits[0]["parquet_sha256"]}
    queries = []
    for index in sorted({0, 1, 3, 4, 5}):
        queries.append({"query_id": index, "top_ids": [index] + [i for i in range(6) if i != index][:4]})
    report = {k: v for k, v in basic.items() if k in ("records", "lance_rows", "vector_bytes_sha256", "crop_policy")}
    report.update({field: "a" * 64 for field in converter._ADVANCED_SOURCES.values()})
    report.update({"schema_version": "npa.ray-clip-development.v1", "execution_fingerprint": fingerprint,
                   "input_hash": validation.canonical_hash(table["input_sha256"].to_pylist()),
                   "processed_hash": validation.canonical_hash(table["processed_sha256"].to_pylist()),
                   "mean_embedding": np.asarray(table["vector"].to_pylist(), dtype=np.float32).mean(axis=0, dtype=np.float64).tolist(),
                   "model_revision": "b" * 40, "model_initializations": [actor, replacement],
                   "final_actors": [{**replacement, "inference_calls": 2}],
                   "gpu_actors": 1, "shards": 3, "batch_size": 2, "embedding_sha256": basic["parquet_sha256"],
                   "retrieval_queries": len(queries), "recovery": recovery,
                   "concurrency_observation": {"participants": 1, "overlap": False, "events": [
                       {"instance_id": replacement["instance_id"], "event": "start", "monotonic_ns": 9000000000},
                       {"instance_id": replacement["instance_id"], "event": "finish", "monotonic_ns": 9300000000}]},
                   "concurrent_actor_inference_observed": False,
                   **{key: 0.5 for key in converter._ADVANCED_TIMINGS},
                   "preprocessing_task_seconds_sum": 0.3, "inference_actor_seconds_sum": 0.6})
    _dump(root / "retrieval.json", queries)
    _dump(root / "report.json", report)
    _dump(root / "execution.json", {"execution_fingerprint": fingerprint})
    _dump(root / "recovery.json", recovery)
    _dump(root / "actor-cleanup.json", {"errors": [], "attempted": 1})
    application._write_cleanup_artifacts(root, [], 1)
    # Advanced producer has no JSON manifest.
    (root / "sha256.json").unlink()
    application._write_cleanup_artifacts(root, [], 1)
    return root


def _chunks(path, entity):
    from rerun.recording import load_recording
    return [c.to_record_batch() for c in load_recording(path).chunks() if str(c.entity_path) == entity]


@pytest.mark.parametrize("fixture_name", ["result", "advanced"])
def test_recording_decodes_source_indices_vectors_images_and_provenance(modules, request, fixture_name, tmp_path):
    converter = modules[0]
    root = request.getfixturevalue(fixture_name)
    path = tmp_path / "report.rrd"
    receipt = converter.convert(root, path, run_id="synthetic-test")
    assert receipt["records"] == 6
    assert receipt["entity_rows"]["/vectors/embedding"] == 6
    assert receipt["rrd_sha256"] == converter._sha(path)
    norms = pa.concat_tables([pa.Table.from_batches([b]) for b in _chunks(path, "/vectors/norm")])
    assert norms["record_id"].to_pylist() == list(range(6))
    assert norms["Scalars:scalars"].to_pylist() == [[1.0]] * 6
    vectors = _chunks(path, "/vectors/embedding")
    assert sum(len(b) for b in vectors) == 6
    images = _chunks(path, "/images/crop")
    assert [i for b in images for i in b.column("record_id").to_pylist()] == list(range(6))
    text = _chunks(path, "/provenance/run")[0].column("TextDocument:text")[0].as_py()[0]
    assert json.loads(text)["report_sha256"] == converter._sha(root / "report.json")
    assert b"private-node-do-not-export" not in path.read_bytes()
    assert "private" not in json.dumps(receipt)


def test_advanced_uses_coordinator_clock_and_static_recovery(modules, advanced, tmp_path):
    path = tmp_path / "advanced.rrd"
    modules[0].convert(advanced, path, run_id="synthetic-test")
    events = _chunks(path, "/concurrency/actors/1/active")
    ticks = [value.value for batch in events for value in batch.column("coordinator_elapsed")]
    assert ticks == [0, 300000000]
    active = [v[0] for b in events for v in b.column("Scalars:scalars").to_pylist()]
    assert active == [1.0, 0.0]
    shards = _chunks(path, "/checkpoints/materialized")
    assert [i for b in shards for i in b.column("shard_index").to_pylist()] == [0, 1, 2]
    recovery = _chunks(path, "/recovery/checkpoint_replay")[0]
    assert "coordinator_elapsed" not in recovery.schema.names
    assert "shard_index" not in recovery.schema.names


@pytest.mark.parametrize("damage", ["missing", "checksum", "duplicate", "traversal", "symlink", "extra", "partial"])
def test_manifest_failures_publish_nothing(modules, result, tmp_path, damage):
    path = result / "SHA256SUMS"
    if damage == "missing":
        (result / "images/000000-crop.png").unlink()
    elif damage == "checksum":
        (result / "report.json").write_text("{}")
    elif damage == "duplicate":
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    elif damage == "traversal":
        path.write_text(path.read_text() + "a" * 64 + "  ../secret\n")
    elif damage == "symlink":
        (result / "link").symlink_to(tmp_path)
    elif damage == "extra":
        (result / "unexpected").write_text("incomplete output")
    else:
        (result / "report.json").unlink()
        modules[1].write_hashes(result)
    with pytest.raises(ValueError):
        modules[0].convert(result, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()


@pytest.mark.parametrize("damage", ["ids", "nan", "dimension", "norm", "image", "retrieval", "source", "timing"])
def test_semantic_corruption_rejected_even_after_rehash(modules, result, tmp_path, damage):
    converter, embed, *_ = modules
    report = json.loads((result / "report.json").read_text())
    table = pq.read_table(result / "embeddings.parquet")
    rows = table.to_pylist()
    if damage in ("ids", "nan", "dimension", "norm"):
        if damage == "ids":
            rows[-1]["record_id"] = 0
        else:
            rows[0]["vector"] = {"nan": [float("nan")] * 512, "dimension": [1.0], "norm": [0.0] * 512}[damage]
        pq.write_table(pa.Table.from_pylist(rows, schema=None if damage == "dimension" else table.schema), result / "embeddings.parquet")
        report["parquet_sha256"] = converter._sha(result / "embeddings.parquet")
    elif damage == "image":
        (result / "images/000000-crop.png").write_bytes((result / "images/000001-crop.png").read_bytes())
    elif damage == "retrieval":
        queries = json.loads((result / "retrieval.json").read_text())
        queries[0]["top_ids"][0] = 600
        report["retrieval"] = queries
        _dump(result / "retrieval.json", queries)
    elif damage == "source":
        report["actors"][0]["source_sha256"]["embed.py"] = "f" * 64
    else:
        report["timings_seconds"]["application"] = -1
    _dump(result / "report.json", report)
    embed.write_hashes(result)
    with pytest.raises(ValueError):
        converter.convert(result, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()


@pytest.mark.parametrize("damage", ["commit", "identity", "shard", "event", "recovery", "cleanup"])
def test_advanced_corruption_rejected_after_rehash(modules, advanced, tmp_path, damage):
    converter, _, app, *_ = modules
    root = advanced
    if damage == "commit":
        (root / "shards/000001/commit.json").unlink()
    elif damage == "identity":
        path = root / "shards/000001/commit.json"
        value = json.loads(path.read_text())
        value["identity"]["execution_fingerprint"] = "f" * 64
        _dump(path, value)
    elif damage == "shard":
        (root / "shards/000001/embeddings.parquet").write_bytes((root / "shards/000000/embeddings.parquet").read_bytes())
    elif damage == "cleanup":
        _dump(root / "actor-cleanup.json", {"attempted": 1, "errors": ["failure"]})
    else:
        path = root / "report.json"
        value = json.loads(path.read_text())
        if damage == "event":
            value["concurrency_observation"]["events"].pop()
        else:
            value["recovery"]["replay_inference_calls"] = 1
            _dump(root / "recovery.json", value["recovery"])
        _dump(path, value)
    # Refresh only the manifest, preserving intentionally damaged cleanup evidence.
    lines = [f"{converter._sha(p)}  {p.relative_to(root)}\n" for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (root / "SHA256SUMS").write_text("".join(lines))
    with pytest.raises(ValueError):
        converter.convert(root, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()


def test_writer_failure_and_output_conflict_preserve_input(modules, result, tmp_path, monkeypatch):
    converter = modules[0]
    before = converter._manifest(result)
    output = tmp_path / "report.rrd"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        converter.convert(result, output, run_id="test")
    assert output.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="outside"):
        converter.convert(result, result / "new.rrd", run_id="test")
    def fail(*args):
        raise RuntimeError("sink failed")
    monkeypatch.setattr(converter, "_inspect", fail)
    with pytest.raises(RuntimeError, match="sink failed"):
        converter.convert(result, tmp_path / "new.rrd", run_id="test")
    assert not (tmp_path / "new.rrd").exists()
    assert not list(tmp_path.glob(".clip-*"))
    assert converter._manifest(result) == before


def test_retrieval_extra_fields_cannot_leak_into_recording(modules, result, tmp_path):
    converter, embed, *_ = modules
    queries = json.loads((result / "retrieval.json").read_text())
    queries[0]["private_detail"] = "never-export-this-field"
    report = json.loads((result / "report.json").read_text())
    report["retrieval"] = queries
    _dump(result / "retrieval.json", queries)
    _dump(result / "report.json", report)
    embed.write_hashes(result)
    path = tmp_path / "report.rrd"
    converter.convert(result, path, run_id="test")
    decoded = [json.loads(row[0]) for b in _chunks(path, "/retrieval/top_ids")
               for row in b.column("TextDocument:text").to_pylist()]
    assert all(set(query) == {"query_id", "top_ids"} for query in decoded)


def test_reversed_recovery_lineage_is_rejected(modules, advanced, tmp_path):
    converter, _, app, *_ = modules
    report = json.loads((advanced / "report.json").read_text())
    recovery = report["recovery"]
    recovery["old_instance"], recovery["new_instance"] = recovery["new_instance"], recovery["old_instance"]
    _dump(advanced / "report.json", report)
    _dump(advanced / "recovery.json", recovery)
    app._write_cleanup_artifacts(advanced, [], 1)
    with pytest.raises(ValueError, match="recovery"):
        converter.convert(advanced, tmp_path / "invalid.rrd", run_id="test")


@pytest.mark.parametrize("damage", ["weights", "fingerprint", "config"])
def test_model_identity_cannot_be_invented(modules, advanced, tmp_path, damage):
    converter, _, app, *_ = modules
    report = json.loads((advanced / "report.json").read_text())
    for actor in report["model_initializations"]:
        if damage == "weights":
            actor["model_files"] = [entry for entry in actor["model_files"] if entry["path"] == "config.json"]
        elif damage == "fingerprint":
            actor["torch"] = "changed-runtime"
        else:
            actor["model_config_sha256"] = "f" * 64
    _dump(advanced / "report.json", report)
    app._write_cleanup_artifacts(advanced, [], 1)
    with pytest.raises(ValueError):
        converter.convert(advanced, tmp_path / "invalid.rrd", run_id="test")


@pytest.mark.parametrize("damage", ["vector", "image", "event", "recovery", "identity"])
def test_decoded_verification_rejects_writer_content_drift(modules, advanced, tmp_path, monkeypatch, damage):
    converter = modules[0]
    original = converter._write
    def corrupt(path, data, run_id):
        changed = copy.deepcopy(data)
        if damage == "vector":
            changed["matrix"] = changed["matrix"][::-1].copy()
        elif damage == "image":
            changed["frames"][0][0][:] = 0
        elif damage == "event":
            changed["events"] = []
        elif damage == "recovery":
            changed["recovery"] = None
        else:
            run_id = "incorrect-recording"
        original(path, changed, run_id)
    monkeypatch.setattr(converter, "_write", corrupt)
    with pytest.raises(ValueError, match="Decoded RRD"):
        converter.convert(advanced, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()


@pytest.mark.parametrize("damage", ["string_vector", "report_list", "invalid_numeric_metric"])
def test_cli_rejects_malformed_values_without_echo_or_traceback(modules, result, tmp_path, damage):
    converter, embed, *_ = modules
    sentinel = "synthetic-private-value-must-stay-private"
    report = json.loads((result / "report.json").read_text())
    if damage == "string_vector":
        table = pq.read_table(result / "embeddings.parquet")
        vectors = pa.array([[sentinel] * 512] * len(table), type=pa.list_(pa.string(), 512))
        table = table.set_column(table.schema.get_field_index("vector"), "vector", vectors)
        pq.write_table(table, result / "embeddings.parquet")
        report["parquet_sha256"] = converter._sha(result / "embeddings.parquet")
    elif damage == "report_list":
        report = [sentinel]
    else:
        report["timings_seconds"]["application"] = {sentinel: sentinel}
    _dump(result / "report.json", report)
    embed.write_hashes(result)
    output = tmp_path / "invalid.rrd"
    run = subprocess.run([sys.executable, converter.__file__, "--input-path", str(result),
                          "--output-path", str(output), "--run-id", "test"], capture_output=True, text=True)
    assert run.returncode == 1
    assert sentinel not in run.stdout + run.stderr
    assert "Traceback" not in run.stderr
    assert "CLIP report conversion failed" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize("fixture_name", ["result", "advanced"])
@pytest.mark.parametrize("damage", ["missing_table", "missing_data", "empty_manifest"])
def test_lance_payload_is_required_even_after_manifest_refresh(modules, request, fixture_name, damage, tmp_path):
    converter, embed, app, *_ = modules
    root = request.getfixturevalue(fixture_name)
    lance = root / "lance/embeddings.lance"
    if damage == "missing_table":
        shutil.rmtree(lance)
    elif damage == "missing_data":
        shutil.rmtree(lance / "data")
    else:
        for path in (lance / "_versions").glob("*.manifest"):
            path.write_bytes(b"")
    if fixture_name == "result":
        embed.write_hashes(root)
    else:
        app._write_cleanup_artifacts(root, [], 1)
    with pytest.raises(ValueError, match="Lance"):
        converter.convert(root, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()


@pytest.mark.parametrize("damage", ["killed_actor", "inference_calls"])
def test_recovery_cannot_attribute_later_work_to_killed_actor(modules, advanced, tmp_path, damage):
    converter, _, app, *_ = modules
    report = json.loads((advanced / "report.json").read_text())
    if damage == "killed_actor":
        path = advanced / "shards/000001/commit.json"
        commit = json.loads(path.read_text())
        commit["inference"]["instance_id"] = report["recovery"]["old_instance"]
        _dump(path, commit)
    else:
        report["final_actors"][0]["inference_calls"] += 1
        _dump(advanced / "report.json", report)
    app._write_cleanup_artifacts(advanced, [], 1)
    with pytest.raises(ValueError, match="replacement|inference calls"):
        converter.convert(advanced, tmp_path / "invalid.rrd", run_id="test")


@pytest.mark.parametrize("damage", ["oversized_png", "duration_overflow"])
def test_cli_normalizes_malformed_image_and_duration(modules, advanced, tmp_path, damage):
    converter, _, app, *_ = modules
    if damage == "oversized_png":
        path = advanced / "preview.png"
        png = bytearray(path.read_bytes())
        png[16:24] = struct.pack(">II", 20000000, 10)
        png[29:33] = struct.pack(">I", zlib.crc32(png[12:29]))
        path.write_bytes(png)
    else:
        report = json.loads((advanced / "report.json").read_text())
        events = report["concurrency_observation"]["events"]
        events[-1]["monotonic_ns"] = events[0]["monotonic_ns"] + 2**63
        _dump(advanced / "report.json", report)
    app._write_cleanup_artifacts(advanced, [], 1)
    output = tmp_path / "invalid.rrd"
    run = subprocess.run([sys.executable, converter.__file__, "--input-path", str(advanced),
                          "--output-path", str(output), "--run-id", "test"], capture_output=True, text=True)
    assert run.returncode == 1
    assert not run.stdout
    assert "CLIP report conversion failed" in run.stderr
    assert "Traceback" not in run.stderr
    assert str(advanced) not in run.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".clip-*.rrd"))


@pytest.mark.parametrize("fixture_name", ["result", "advanced"])
@pytest.mark.parametrize("damage", ["missing", "contradictory"])
def test_preprocessor_identity_must_match_inference_source(modules, request, fixture_name, damage, tmp_path):
    converter, embed, app, *_ = modules
    root = request.getfixturevalue(fixture_name)
    path = root / ("report.json" if fixture_name == "result" else "shards/000001/commit.json")
    value = json.loads(path.read_text())
    field = "preprocessors" if fixture_name == "result" else "preprocessor"
    if damage == "missing":
        value.pop(field)
    else:
        preprocessor = value[field][0] if fixture_name == "result" else value[field]
        preprocessor["source_sha256"] = "f" * 64
    _dump(path, value)
    if fixture_name == "result":
        embed.write_hashes(root)
    else:
        app._write_cleanup_artifacts(root, [], 1)
    with pytest.raises(ValueError, match="[Pp]reprocessor|malformed"):
        converter.convert(root, tmp_path / "invalid.rrd", run_id="test")
    assert not (tmp_path / "invalid.rrd").exists()

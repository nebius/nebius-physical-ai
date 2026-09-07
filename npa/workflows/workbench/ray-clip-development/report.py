# Convert persisted native Ray CLIP results into a factual, portable Rerun recording.
"""Review downloaded CLIP results without Ray, CUDA, model downloads, or a service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

APPLICATION_ID = "npa.ray-clip-development"
_HASH = re.compile(r"[0-9a-f]{64}")
_BASIC_SOURCES = ("embed.py", "worker.py", "npa_lancedb_bdd100k_udfs.py")
_ADVANCED_SOURCES = {
    "application.py": "application_sha256", "worker.py": "source_sha256",
    "validation.py": "validation_sha256", "npa_lancedb_bdd100k_udfs.py": "udf_sha256",
}
_BASIC_TIMINGS = (
    "cluster_connect_and_model_ready", "preprocessing_submission",
    "preprocessing_and_inference_wall", "preprocessing_task_sum", "inference_actor_sum",
    "aggregation_and_artifacts", "application",
)
_ADVANCED_TIMINGS = (
    "cluster_connect_and_actor_ready_seconds", "preprocessing_and_inference_wall_seconds",
    "preprocessing_task_seconds_sum", "inference_actor_seconds_sum", "aggregation_seconds",
    "application_seconds",
)


class _InvalidResult(ValueError):
    """A safe, converter-authored validation message suitable for ordinary output."""


def _require(condition, message):
    if not condition:
        raise _InvalidResult(message)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def _json(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_pairs)


def _hash(value):
    _require(isinstance(value, str) and _HASH.fullmatch(value), "Invalid SHA-256 identity")
    return value


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "Expected a finite nonnegative measurement")
    return value


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "Invalid integer count or index")
    return value


def _manifest(root):
    """Verify complete tree coverage before parsing any source report or image."""
    _require(root.is_dir() and not root.is_symlink(), "Input must be a regular result directory")
    paths = list(root.rglob("*"))
    _require(all(not p.is_symlink() for p in paths), "Symlinks are not result artifacts")
    _require(all(p.is_dir() or p.is_file() for p in paths), "Nonregular result artifact")
    listed = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        _require("  " in line, "Malformed checksum manifest")
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        _require(name and not relative.is_absolute() and ".." not in relative.parts
                 and str(relative) == name and "\\" not in name and name != "SHA256SUMS",
                 "Unsafe checksum manifest path")
        _require(name not in listed, "Duplicate checksum manifest entry")
        listed[name] = _hash(digest)
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()} - {"SHA256SUMS"}
    _require(set(listed) == actual and listed, "Checksum manifest does not cover the complete result tree")
    for name, digest in listed.items():
        _require(_sha(root / name) == digest, "Artifact checksum mismatch")
    if "sha256.json" in listed:
        _require(_json(root / "sha256.json") == {k: v for k, v in listed.items() if k != "sha256.json"},
                 "JSON checksum manifest disagrees")
    return listed


def _vectors(table, count):
    _require(set(table.column_names) == {"record_id", "input_sha256", "processed_sha256", "vector"},
             "Unexpected embedding table columns")
    ids = table["record_id"].to_pylist()
    _require(all(type(i) is int for i in ids) and ids == list(range(count)),
             "Missing, duplicate, unordered or unexpected record IDs")
    _require(table.schema.field("vector").type == pa.list_(pa.float32(), 512),
             "CLIP vectors must use the producer's fixed 512-element float32 Arrow type")
    matrix = np.asarray(table["vector"].to_pylist(), dtype=np.float32)
    _require(matrix.shape == (count, 512) and np.isfinite(matrix).all(), "Invalid CLIP vector dimensions or values")
    _require(np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4), "CLIP vectors are not normalized")
    for column in ("input_sha256", "processed_sha256"):
        for value in table[column].to_pylist():
            _hash(value)
    return matrix


def _sources(report, advanced):
    if advanced:
        result = {name: _hash(report[field]) for name, field in _ADVANCED_SOURCES.items()}
    else:
        _require(set(report["source_sha256"]) == set(_BASIC_SOURCES), "Incomplete source identity")
        result = {name: _hash(report["source_sha256"][name]) for name in _BASIC_SOURCES}
    return result


def _actors(report, advanced, sources):
    actors = report["model_initializations" if advanced else "actors"]
    _require(isinstance(actors, list) and actors, "Missing model actor provenance")
    safe = []
    for actor in actors:
        source = ({name: actor[field] for name, field in _ADVANCED_SOURCES.items()}
                  if advanced else actor["source_sha256"])
        _require(source == sources, "Actor and driver source identities disagree")
        revision = actor["model_revision"]
        _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "Invalid model revision")
        weights = ({entry["path"]: _hash(entry["sha256"]) for entry in actor["model_files"]}
                   if advanced else {"pytorch_model.bin": _hash(actor["weight_sha256"])})
        if advanced:
            files = actor["model_files"]
            _require(len(weights) == len(files), "Duplicate model file identity")
            _require(any(name.endswith((".bin", ".safetensors")) for name in weights), "Missing model weight identity")
            _require(weights.get("config.json") == actor["model_config_sha256"], "Model config identity disagrees")
            for name in weights:
                relative = PurePosixPath(name)
                _require(name and not relative.is_absolute() and ".." not in relative.parts
                         and str(relative) == name and "\\" not in name, "Unsafe model file identity")
        # File names and runtime text can contain private data. Export only byte identities.
        model_hashes = sorted(weights.values())
        _require(model_hashes, "Missing model byte identities")
        safe.append({"actor_index": len(safe), "model_revision": revision,
                     "model_config_sha256": _hash(actor["model_config_sha256"]),
                     "model_file_sha256": model_hashes,
                     "model_load_seconds": _number(actor["model_load_seconds"])})
    identity_keys = ("model_revision", "model_config_sha256", "model_file_sha256")
    _require(all(all(a[k] == safe[0][k] for k in identity_keys) for a in safe),
             "Actors disagree on model identity")
    return actors, safe


def _images(root, table, count):
    frames = []
    for index in range(min(count, 8)):
        pair = []
        for kind, column in (("original", "input_sha256"), ("crop", "processed_sha256")):
            path = root / "images" / f"{index:06d}-{kind}.png"
            _require(_sha(path) == table[column][index].as_py(), "Preview image differs from embedded input")
            with Image.open(path) as image:
                image.load()
                _require(image.mode == "RGB", "Preview image must decode as RGB")
                pair.append(np.asarray(image).copy())
        frames.append(pair)
    with Image.open(root / "preview.png") as image:
        image.load()
        _require(image.mode == "RGB" and image.size == (448, 224 * len(frames)), "Invalid contact sheet")
        expected = Image.new("RGB", image.size)
        for row, pair in enumerate(frames):
            for col, pixels in enumerate(pair):
                expected.paste(Image.fromarray(pixels).resize((224, 224)), (224 * col, 224 * row))
        _require(np.array_equal(np.asarray(image), np.asarray(expected)), "Contact sheet disagrees with preview images")
    return frames


def _retrieval(root, report, matrix, advanced):
    queries = _json(root / "retrieval.json")
    count = len(matrix)
    expected = sorted({0, count // 4, count // 2, 3 * count // 4, count - 1}) if advanced else sorted({0, count // 2, count - 1})
    _require(isinstance(queries, list) and [q["query_id"] for q in queries] == expected,
             "Incomplete retrieval query coverage")
    for query in queries:
        query_id = _integer(query["query_id"])
        hits = query["top_ids"]
        _require(isinstance(hits, list) and len(hits) == min(5, count)
                 and all(type(i) is int and 0 <= i < count for i in hits)
                 and len(set(hits)) == len(hits) and query_id in hits, "Invalid retrieval record IDs")
        # Check ranking with tolerance for tied/float32 scores without re-running the model.
        vectors = matrix.astype(np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        distance = 1 - vectors @ vectors[query_id]
        observed = distance[hits]
        _require(np.all(np.diff(observed) >= -1e-5)
                 and observed.max() <= np.sort(distance)[len(hits) - 1] + 1e-5,
                 "Retrieval ranking disagrees with persisted vectors")
    if advanced:
        _require(report["retrieval_queries"] == len(queries), "Retrieval count disagrees")
    else:
        _require(report["retrieval"] == queries, "Retrieval report disagrees")
    return [{"query_id": q["query_id"], "top_ids": q["top_ids"]} for q in queries]


def _advanced(root, report, table, actors):
    """Verify shard materialization and preserve separate coordinator clock semantics."""
    from application import FINGERPRINT_FIELDS

    fingerprint = _hash(report["execution_fingerprint"])
    _require(_json(root / "execution.json") == {"execution_fingerprint": fingerprint}, "Execution marker disagrees")
    _require(_json(root / "actor-cleanup.json") == {"errors": [], "attempted": report["gpu_actors"]},
             "Result has missing or failed actor cleanup")
    instances = [a["instance_id"] for a in actors]
    _require(all(isinstance(i, str) and i for i in instances) and len(set(instances)) == len(instances),
             "Invalid actor identities")
    for actor in actors:
        identity = {field: actor[field] for field in FINGERPRINT_FIELDS}
        _require(_canonical(identity) == fingerprint, "Execution fingerprint does not match actor provenance")
        _require(actor["execution_fingerprint"] == fingerprint and actor["model_revision"] == report["model_revision"],
                 "Actor execution identity disagrees")
    final = report["final_actors"]
    final_instances = [a["instance_id"] for a in final]
    participants = _integer(report["gpu_actors"], 1)
    _require(len(final_instances) == participants and len(set(final_instances)) == participants
             and all(i in instances for i in final_instances), "Final actor membership disagrees")
    for actor in final:
        original = actors[instances.index(actor["instance_id"])]
        _require(all(value == actor.get(key) for key, value in original.items() if key != "inference_calls"),
                 "Final actor provenance disagrees")
        _integer(actor["inference_calls"])
    count = _integer(report["shards"], 1)
    batch_size = _integer(report["batch_size"], 1)
    _require(count == math.ceil(len(table) / batch_size), "Shard count disagrees with dataset")
    commits = []
    for index in range(count):
        path = root / "shards" / f"{index:06d}"
        commit = _json(path / "commit.json")
        shard = pq.read_table(path / "embeddings.parquet")
        expected = table.slice(index * batch_size, batch_size)
        _require(shard.equals(expected), "Committed shard differs from aggregate embeddings")
        identity = {"record_ids": shard["record_id"].to_pylist(),
                    "input_hash": _canonical(shard["input_sha256"].to_pylist()),
                    "processed_hash": _canonical(shard["processed_sha256"].to_pylist()),
                    "source_sha256": report["source_sha256"], "model_revision": report["model_revision"],
                    "execution_fingerprint": fingerprint}
        _require(commit["identity"] == identity and commit["rows"] == len(shard), "Checkpoint identity disagrees")
        _require(commit["parquet_sha256"] == _sha(path / "embeddings.parquet"), "Checkpoint hash disagrees")
        measurement = commit["inference"]
        _require(measurement["instance_id"] in instances, "Checkpoint references unknown actor")
        start = _integer(measurement["start_monotonic_ns"])
        end = _integer(measurement["end_monotonic_ns"])
        seconds = _number(measurement["inference_seconds"])
        _require(end >= start and math.isclose(seconds, (end - start) / 1e9, abs_tol=1e-9), "Invalid worker timing")
        commits.append({"rows": len(shard), "actor_index": instances.index(measurement["instance_id"]),
                        "inference_seconds": seconds, "preprocess_seconds": _number(commit["preprocess_seconds"]),
                        "parquet_sha256": _hash(commit["parquet_sha256"])})
    _require(set(p.name for p in (root / "shards").iterdir()) == {f"{i:06d}" for i in range(count)},
             "Unexpected shard checkpoint")
    for field, key in (("inference_actor_seconds_sum", "inference_seconds"),
                       ("preprocessing_task_seconds_sum", "preprocess_seconds")):
        _require(math.isclose(report[field], sum(c[key] for c in commits), rel_tol=1e-9), "Shard timing sum disagrees")
    observation = report["concurrency_observation"]
    events, active, started, finished, overlap = [], set(), set(), set(), False
    previous = 0
    origin = None
    for event in observation["events"]:
        instance, kind = event["instance_id"], event["event"]
        timestamp = _integer(event["monotonic_ns"])
        _require(timestamp >= previous and instance in instances, "Invalid coordinator event")
        previous = timestamp
        if origin is None:
            origin = timestamp
        if kind == "start":
            _require(instance not in started, "Duplicate coordinator start")
            active.add(instance)
            started.add(instance)
            overlap |= len(active) > 1
        else:
            _require(kind == "finish" and instance in active, "Unpaired coordinator finish")
            active.remove(instance)
            finished.add(instance)
        events.append({"actor_index": instances.index(instance), "active": int(kind == "start"),
                       "nanoseconds": timestamp - origin})
    _require(started == set(final_instances), "Coordinator actors disagree with final actors")
    _require(not active and started == finished and len(started) == participants
             and observation["participants"] == participants and observation["overlap"] is overlap
             and report["concurrent_actor_inference_observed"] is overlap, "Incomplete concurrency observation")
    recovery = report["recovery"]
    safe_recovery = None
    if recovery is not None:
        _require(_json(root / "recovery.json") == recovery, "Recovery file disagrees with report")
        old, new = recovery["old_instance"], recovery["new_instance"]
        _require(old in instances and new in instances and old != new
                 and recovery["checkpoint_reused"] is True and recovery["model_reloaded"] is True
                 and type(recovery["replay_inference_calls"]) is int and recovery["replay_inference_calls"] == 0
                 and recovery["parquet_sha256"] == commits[0]["parquet_sha256"]
                 and instances.index(old) == commits[0]["actor_index"]
                 and old not in final_instances and new in final_instances
                 and len(actors) == participants + 1, "Invalid recovery evidence")
        safe_recovery = {"old_actor_index": instances.index(old), "new_actor_index": instances.index(new),
                         "checkpoint_index": 0, "checkpoint_reused": True, "replay_inference_calls": 0,
                         "model_reloaded": True, "parquet_sha256": commits[0]["parquet_sha256"]}
    else:
        _require(not (root / "recovery.json").exists() and len(actors) == participants, "Unexpected recovery evidence")
    return commits, events, safe_recovery


def _load(root):
    manifest = _manifest(root)
    report = _json(root / "report.json")
    _require(isinstance(report, dict), "CLIP report must be a JSON object")
    lance = root / "lance" / "embeddings.lance"
    for directory, pattern in (("data", "*.lance"), ("_versions", "*.manifest")):
        _require(any(p.is_file() and p.stat().st_size > 0 for p in (lance / directory).glob(pattern)),
                 "Missing nonempty Lance result data or version manifest")
    schema = report.get("schema_version")
    _require(schema in (None, "npa.ray-clip-development.v1"), "Unsupported CLIP report schema")
    advanced = schema is not None
    count = _integer(report["records"], 1)
    _require(report["lance_rows"] == count, "Report row count disagrees")
    table = pq.read_table(root / "embeddings.parquet")
    matrix = _vectors(table, count)
    parquet_key = "embedding_sha256" if advanced else "parquet_sha256"
    _require(report[parquet_key] == manifest["embeddings.parquet"], "Report Parquet hash disagrees")
    _require(report["vector_bytes_sha256"] == hashlib.sha256(matrix.tobytes()).hexdigest(), "Vector byte hash disagrees")
    _require(report["crop_policy"] in ("left", "right"), "Unsupported crop policy")
    sources = _sources(report, advanced)
    actors, safe_actors = _actors(report, advanced, sources)
    timings = ({k: _number(report[k]) for k in _ADVANCED_TIMINGS} if advanced
               else {k: _number(report["timings_seconds"][k]) for k in _BASIC_TIMINGS})
    if advanced:
        _require(report["input_hash"] == _canonical(table["input_sha256"].to_pylist())
                 and report["processed_hash"] == _canonical(table["processed_sha256"].to_pylist()),
                 "Report input identity disagrees")
        mean = np.asarray(report["mean_embedding"], dtype=np.float64)
        _require(mean.shape == (512,) and np.array_equal(mean, matrix.mean(axis=0, dtype=np.float64)),
                 "Report mean embedding disagrees")
    frames = _images(root, table, count)
    queries = _retrieval(root, report, matrix, advanced)
    commits, events, recovery = _advanced(root, report, table, actors) if advanced else ([], [], None)
    return {"matrix": matrix, "frames": frames, "queries": queries, "actors": safe_actors,
            "timings": timings, "commits": commits, "events": events, "recovery": recovery,
            "provenance": {"producer": "application.py" if advanced else "embed.py",
                           "converter_sha256": _sha(Path(__file__)),
                           "report_sha256": manifest["report.json"], "source_sha256": sources,
                           "input_manifest_sha256": _sha(root / "SHA256SUMS"),
                           "parquet_sha256": manifest["embeddings.parquet"],
                           "vector_bytes_sha256": report["vector_bytes_sha256"],
                           "crop_policy": report["crop_policy"], "records": count,
                           "limitations": "Procedural images; no semantic accuracy claim. Dataset and shard indices are not time. "
                           "Timing totals overlap and are not additive. Coordinator time covers one observed wave with RPC edges; "
                           "worker clocks are not aligned. Recovery has no timestamp. The supplied run label is not Jobs status proof. "
                           "Lance files are checksum verified; this converter reads Parquet and saved retrieval results."}}


def _write(path, data, run_id):
    import rerun as rr
    import rerun.blueprint as rrb

    recording = rr.RecordingStream(APPLICATION_ID, recording_id=run_id)
    recording.save(path)
    try:
        blueprint = rrb.Blueprint(rrb.Vertical(
            rrb.Horizontal(rrb.Spatial2DView(origin="images/original"), rrb.Spatial2DView(origin="images/crop")),
            rrb.Horizontal(rrb.TimeSeriesView(origin="vectors/norm"), rrb.TextDocumentView(origin="provenance/run")),
        ), collapse_panels=True)
        recording.send_blueprint(blueprint)
        recording.log("provenance/run", rr.TextDocument(json.dumps(data["provenance"], sort_keys=True)), static=True)
        for name, seconds in data["timings"].items():
            recording.log("timings_seconds/" + name, rr.Scalars(seconds), static=True)
        for actor in data["actors"]:
            root = f"actors/{actor['actor_index']}"
            recording.log(root + "/model", rr.TextDocument(json.dumps(actor, sort_keys=True)), static=True)
            recording.log(root + "/model_load_seconds", rr.Scalars(actor["model_load_seconds"]), static=True)
        if data["recovery"] is not None:
            recording.log("recovery/checkpoint_replay", rr.TextDocument(json.dumps(data["recovery"], sort_keys=True)), static=True)
        matrix = data["matrix"]
        recording.send_columns("vectors/norm", indexes=[rr.TimeColumn("record_id", sequence=np.arange(len(matrix)))],
                               columns=rr.Scalars.columns(scalars=np.linalg.norm(matrix, axis=1)), strict=True)
        for index, vector in enumerate(matrix):
            recording.set_time("record_id", sequence=index)
            recording.log("vectors/embedding", rr.Tensor(vector), strict=True)
        for index, pair in enumerate(data["frames"]):
            recording.set_time("record_id", sequence=index)
            for kind, pixels in zip(("original", "crop"), pair, strict=True):
                recording.log("images/" + kind, rr.Image(pixels), strict=True)
        for query in data["queries"]:
            recording.set_time("record_id", sequence=query["query_id"])
            recording.log("retrieval/top_ids", rr.TextDocument(json.dumps(query)), strict=True)
        recording.disable_timeline("record_id")
        for index, commit in enumerate(data["commits"]):
            recording.set_time("shard_index", sequence=index)
            recording.log("checkpoints/materialized", rr.Scalars(commit["rows"]), strict=True)
            recording.log("checkpoints/lineage", rr.TextDocument(json.dumps(commit, sort_keys=True)), strict=True)
            for name in ("inference_seconds", "preprocess_seconds"):
                recording.log("shards/" + name, rr.Scalars(commit[name]), strict=True)
        recording.disable_timeline("shard_index")
        for event in data["events"]:
            recording.set_time("coordinator_elapsed", duration=np.timedelta64(event["nanoseconds"], "ns"))
            recording.log(f"concurrency/actors/{event['actor_index']}/active", rr.Scalars(event["active"]), strict=True)
    finally:
        recording.flush()
        recording.disconnect()


def _inspect(path, data, run_id):
    """Decode every exported fact and require exact values, identity and index coverage."""
    from rerun.recording import load_recording

    recording = load_recording(path)
    _require(recording.application_id() == APPLICATION_ID and recording.recording_id() == run_id,
             "Decoded RRD recording identity differs")
    chunks = {}
    counts = {}
    for chunk in recording.chunks():
        entity = str(chunk.entity_path)
        counts[entity] = counts.get(entity, 0) + chunk.num_rows
        chunks.setdefault(entity, []).append(chunk.to_record_batch())

    def values(entity, column):
        return [value for batch in chunks.get("/" + entity, []) for value in batch.column(column).to_pylist()]

    def documents(entity):
        return [json.loads(row[0]) for row in values(entity, "TextDocument:text")]

    def scalars(entity, expected, timeline=None, indices=None):
        observed = [row[0] for row in values(entity, "Scalars:scalars")]
        _require(np.array_equal(observed, expected), "Decoded RRD scalar values differ: " + entity)
        if timeline is not None:
            actual = values(entity, timeline)
            if timeline == "coordinator_elapsed":
                actual = [value.value for value in actual]
            _require(actual == indices, "Decoded RRD index coverage differs: " + entity)

    matrix = data["matrix"]
    indices = list(range(len(matrix)))
    scalars("vectors/norm", np.linalg.norm(matrix, axis=1), "record_id", indices)
    _require(documents("provenance/run") == [data["provenance"]], "Decoded RRD provenance differs")
    tensor = values("vectors/embedding", "Tensor:data")
    _require(values("vectors/embedding", "record_id") == indices
             and all(len(row) == 1 and row[0]["shape"] == [512] for row in tensor)
             and np.array_equal(np.asarray([row[0]["buffer"] for row in tensor], dtype=np.float32), matrix),
             "Decoded RRD vectors differ")
    for column, kind in enumerate(("original", "crop")):
        entity = "images/" + kind
        _require(values(entity, "record_id") == list(range(len(data["frames"]))), "Decoded RRD image indices differ")
        buffers = values(entity, "Image:buffer")
        formats = values(entity, "Image:format")
        _require(len(buffers) == len(formats) == len(data["frames"]), "Decoded RRD image coverage differs")
        for index, (buffer, image_format) in enumerate(zip(buffers, formats, strict=True)):
            pixels = data["frames"][index][column]
            expected_format = {"width": pixels.shape[1], "height": pixels.shape[0],
                               "pixel_format": None, "color_model": 2, "channel_datatype": 6}
            _require(image_format == [expected_format] and len(buffer) == 1
                     and bytes(buffer[0]) == pixels.tobytes(), "Decoded RRD image pixels differ")
    _require(documents("retrieval/top_ids") == data["queries"]
             and values("retrieval/top_ids", "record_id") == [q["query_id"] for q in data["queries"]],
             "Decoded RRD retrieval differs")
    for name, seconds in data["timings"].items():
        scalars("timings_seconds/" + name, [seconds])
    for actor in data["actors"]:
        entity = f"actors/{actor['actor_index']}"
        _require(documents(entity + "/model") == [actor], "Decoded RRD model provenance differs")
        scalars(entity + "/model_load_seconds", [actor["model_load_seconds"]])
    commits = data["commits"]
    if commits:
        indices = list(range(len(commits)))
        scalars("checkpoints/materialized", [c["rows"] for c in commits], "shard_index", indices)
        _require(documents("checkpoints/lineage") == commits, "Decoded RRD checkpoint lineage differs")
        for name in ("inference_seconds", "preprocess_seconds"):
            scalars("shards/" + name, [c[name] for c in commits], "shard_index", indices)
    for actor_index in sorted({e["actor_index"] for e in data["events"]}):
        events = [e for e in data["events"] if e["actor_index"] == actor_index]
        scalars(f"concurrency/actors/{actor_index}/active", [e["active"] for e in events],
                "coordinator_elapsed", [e["nanoseconds"] for e in events])
    expected_recovery = [data["recovery"]] if data["recovery"] is not None else []
    _require(documents("recovery/checkpoint_replay") == expected_recovery, "Decoded RRD recovery lineage differs")
    return {"entity_rows": counts, "records": len(matrix), "decoded_provenance": "matched"}


def convert(input_path: Path, output_path: Path, *, run_id: str) -> dict:
    """Validate one complete persisted CLIP result and publish a new RRD atomically.

    Args:
        input_path: Downloaded embed.py or application.py result directory.
        output_path: New .rrd file outside the immutable input tree.
        run_id: Public-safe native Jobs submission label used as recording identity.
    Returns:
        Sanitized source/output hashes and independently decoded entity row counts.
    Raises:
        ValueError: Input is incomplete, inconsistent or corrupt, or output is unsafe.
        OSError: Artifact access or atomic output publication fails.
    """
    root = Path(input_path)
    output = Path(output_path)
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id), "Use a public-safe run label")
    _require(output.suffix == ".rrd" and not output.resolve().is_relative_to(root.resolve()),
             "Output must be a new .rrd outside the input tree")
    _require(not output.exists() and not output.is_symlink(), "Output already exists")
    try:
        data = _load(root)
    except _InvalidResult:
        raise
    except (ValueError, KeyError, TypeError, IndexError, OSError, AttributeError, OverflowError) as error:
        raise _InvalidResult("Incomplete or malformed persisted CLIP result") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".clip-", suffix=".rrd", dir=output.parent)
    os.close(descriptor)
    path = Path(temporary)
    try:
        _write(path, data, run_id)
        inspection = _inspect(path, data, run_id)
        _manifest(root)
        _require(_sha(root / "SHA256SUMS") == data["provenance"]["input_manifest_sha256"],
                 "Input result changed during conversion")
        digest = _sha(path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        # Same-directory hard link provides atomic, no-overwrite publication under races.
        os.link(path, output)
    finally:
        path.unlink(missing_ok=True)
    return {"schema_version": "npa.ray-clip-rrd.v1", "application_id": APPLICATION_ID,
            "recording_id": run_id, "rrd_sha256": digest, "source": data["provenance"], **inspection}


def main(argv: list[str] | None = None) -> int:
    """Run the local converter; JSON stdout is emitted only after decoded validation.

    Args:
        argv: Explicit arguments, or None to read the process command line.
    Returns:
        Zero on successful conversion; argparse exits nonzero on invalid input.
    Raises:
        OSError: A recording cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True, help="Downloaded complete CLIP result directory")
    parser.add_argument("--output-path", type=Path, required=True, help="New .rrd outside the input directory")
    parser.add_argument("--run-id", required=True, help="Public-safe native Jobs submission label")
    args = parser.parse_args(argv)
    try:
        result = convert(args.input_path, args.output_path, run_id=args.run_id)
    except _InvalidResult as error:
        parser.exit(1, f"CLIP report conversion failed: {error}\n")
    except (ValueError, OSError, RuntimeError):
        parser.exit(1, "CLIP report conversion failed: recording could not be written or verified\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

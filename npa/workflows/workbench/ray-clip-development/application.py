"""Real Workbench CLIP inference with driver-owned durable shard checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
import uuid

import validation
import worker


def model_snapshot_files(checkpoint: Path) -> list[dict]:
    """Hash model artifacts without per-download Hugging Face bookkeeping."""
    files = []
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_file():
            files.append({"path": relative.as_posix(), "sha256": validation.file_hash(path)})
    return files


def load_workbench_udf():
    """Import the canonical source bytes delivered in this job's working_dir."""
    # Ray serializes __main__ actors by value, including the driver's __file__.
    # Imported modules resolve through each worker's own runtime_env directory.
    path = Path(worker.__file__).with_name("npa_lancedb_bdd100k_udfs.py")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("The Jobs working_dir lacks a regular Workbench CLIP UDF")
    spec = importlib.util.spec_from_file_location("npa_lancedb_bdd100k_udfs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The Jobs working_dir lacks the real Workbench CLIP UDF")
    udf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = udf
    spec.loader.exec_module(udf)
    return udf


class ClipActor:
    def __init__(self, model_path: str, model_revision: str):
        import ray
        import torch
        from importlib.metadata import version

        started = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("This recipe requires real CUDA inference")
        checkpoint = Path(model_path)
        if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
            raise ValueError("model-path must be a previously fetched CLIP snapshot")
        self.udf = load_workbench_udf()
        udf_path = Path(self.udf.__file__)
        self.udf.CLIP_MODEL_NAME = str(checkpoint.resolve())
        self.udf._clip_components(device="cuda:0", precision="float32")
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - started
        fingerprint_started = time.perf_counter()
        model_files = model_snapshot_files(checkpoint)
        if not any(row["path"].endswith((".safetensors", ".bin")) for row in model_files):
            raise ValueError("Model snapshot has no verifiable PyTorch weights")
        self.revision = model_revision
        self.info = {
            "instance_id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "python": sys.version.split()[0],
            "ray": ray.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": version("transformers"),
            "pyarrow": version("pyarrow"),
            "lancedb": version("lancedb"),
            "source_sha256": worker.source_hash(),
            "worker_module_path": str(Path(worker.__file__).resolve()),
            "application_sha256": validation.file_hash(Path(worker.__file__).with_name("application.py")),
            "udf_sha256": validation.file_hash(udf_path),
            "udf_module_path": str(udf_path.resolve()),
            "model_revision": model_revision,
            "model_config_sha256": validation.file_hash(checkpoint / "config.json"),
            "model_load_seconds": model_load_seconds,
            "model_files": model_files,
            "precision": "float32",
            "validation_sha256": validation.file_hash(Path(validation.__file__)),
            "model_initializations": 1,
        }
        keys = ("source_sha256", "application_sha256", "validation_sha256", "udf_sha256",
                "model_revision", "model_files", "precision", "python", "ray", "torch", "cuda",
                "transformers", "pyarrow", "lancedb", "gpu_capability")
        self.info["execution_fingerprint"] = validation.canonical_hash({key: self.info[key] for key in keys})
        self.info["fingerprint_verification_seconds"] = time.perf_counter() - fingerprint_started
        self.inference_calls = 0

    def status(self) -> dict:
        return {**self.info, "inference_calls": self.inference_calls}

    def infer(self, shard: dict, barrier=None) -> tuple[object, dict]:
        import numpy as np
        import pyarrow as pa
        import torch
        import ray

        if shard["source_sha256"] != worker.source_hash():
            raise ValueError("CPU and GPU workers imported different application revisions")
        if barrier is not None:
            ray.get(barrier.arrive.remote(self.info["instance_id"]))
            ray.get(barrier.start.remote(self.info["instance_id"]))
        torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        vectors = self.udf.udf_clip_embedding(
            pa.record_batch({"image_bytes": [row["image_bytes"] for row in shard["rows"]]}),
            device="cuda:0",
            precision="float32",
        )
        torch.cuda.synchronize()
        end_ns = time.monotonic_ns()
        if barrier is not None:
            ray.get(barrier.finish.remote(self.info["instance_id"]))
        matrix = np.asarray(vectors.to_pylist(), dtype=np.float32)
        if matrix.shape != (len(shard["rows"]), 512) or not np.isfinite(matrix).all():
            raise ValueError("Invalid CLIP vectors")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
            raise ValueError("CLIP vectors are not normalized")
        self.inference_calls += 1
        return vectors, {
            "instance_id": self.info["instance_id"],
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "inference_seconds": (end_ns - start_ns) / 1e9,
        }


class InferenceBarrier:
    """Observe one inference wave using a single clock, including across nodes."""

    def __init__(self, participants: int):
        import asyncio

        self.participants = participants
        self.arrived = set()
        self.active = set()
        self.ready = asyncio.Event()
        self.overlap = False
        self.events = []

    async def arrive(self, instance: str):
        self.arrived.add(instance)
        if len(self.arrived) == self.participants:
            self.ready.set()
        await self.ready.wait()

    async def start(self, instance: str):
        self.active.add(instance)
        self.overlap |= len(self.active) > 1
        self.events.append({"instance_id": instance, "event": "start", "monotonic_ns": time.monotonic_ns()})

    async def finish(self, instance: str):
        self.events.append({"instance_id": instance, "event": "finish", "monotonic_ns": time.monotonic_ns()})
        self.active.remove(instance)

    async def status(self):
        return {"participants": len(self.arrived), "overlap": self.overlap, "events": self.events}


def commit_shard(path: Path, shard: dict, vectors, measurement: dict, model_revision: str, execution_fingerprint: str) -> dict:
    """Only the driver writes checkpoints, so workers need no shared filesystem."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = validation.checkpoint_identity(shard, model_revision, execution_fingerprint)
    cached = validation.read_checkpoint(path, identity)
    if cached is not None:
        return {**cached, "checkpoint_reused": True}
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "record_id": [row["record_id"] for row in shard["rows"]],
        "input_sha256": [row["input_sha256"] for row in shard["rows"]],
        "processed_sha256": [row["processed_sha256"] for row in shard["rows"]],
        "vector": vectors,
    })
    temporary = path / f".{uuid.uuid4().hex}.parquet"
    try:
        pq.write_table(table, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path / "embeddings.parquet")
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "identity": identity, "rows": len(shard["rows"]),
        "parquet_sha256": validation.file_hash(path / "embeddings.parquet"),
        "preprocess_seconds": shard["preprocess_seconds"],
        "preprocessor": {key: shard[key] for key in ("source_sha256", "source_path", "pid", "node_id")},
        "inference": measurement, "checkpoint_reused": False,
    }
    validation.atomic_json(path / "commit.json", receipt)
    return receipt


def submit_shard(actor, shard: dict, directory: Path, model_revision: str, execution_fingerprint: str, barrier=None):
    """The retry boundary checks a committed identity before dispatching GPU work."""
    cached = validation.read_checkpoint(directory, validation.checkpoint_identity(shard, model_revision, execution_fingerprint))
    if cached is not None:
        return {**cached, "checkpoint_reused": True}, None
    return None, actor.infer.remote(shard, barrier)


def aggregate(output: Path, receipts: list[dict], records: int) -> dict:
    import lancedb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    tables = []
    for index, receipt in enumerate(receipts):
        path = output / "shards" / f"{index:06d}"
        validation.read_checkpoint(path, receipt["identity"])
        tables.append(pq.read_table(path / "embeddings.parquet"))
    table = pa.concat_tables(tables).sort_by("record_id")
    ids = table["record_id"].to_pylist()
    validation.verify_ids(ids, records)
    vectors = np.asarray(table["vector"].to_pylist(), dtype=np.float32)
    database = lancedb.connect(str(output / "lance"))
    lance_table = database.create_table("embeddings", table, mode="overwrite")
    if lance_table.count_rows() != records:
        raise ValueError("Lance persistence lost rows")
    query_ids = list(dict.fromkeys([0, records // 4, records // 2, 3 * records // 4, records - 1]))
    retrievals = []
    for query_id in query_ids:
        hits = lance_table.search(vectors[query_id]).metric("cosine").limit(5).to_list()
        if query_id not in [row["record_id"] for row in hits]:
            raise ValueError(f"Lance self-query missed record {query_id}")
        retrievals.append({"query_id": query_id, "top_ids": [row["record_id"] for row in hits]})
    pq.write_table(table, output / "embeddings.parquet")
    validation.atomic_json(output / "retrieval.json", retrievals)
    return {
        "records": records,
        "input_hash": validation.canonical_hash(table["input_sha256"].to_pylist()),
        "processed_hash": validation.canonical_hash(table["processed_sha256"].to_pylist()),
        "embedding_sha256": validation.file_hash(output / "embeddings.parquet"),
        "vector_bytes_sha256": __import__("hashlib").sha256(vectors.tobytes()).hexdigest(),
        "mean_embedding": vectors.mean(axis=0, dtype=np.float64).tolist(),
        "lance_rows": lance_table.count_rows(),
        "retrieval_queries": len(query_ids),
        "aggregation_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--records", type=int, default=2048)
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recovery-check", action="store_true")
    parser.add_argument("--cancellation-probe", action="store_true")
    parser.add_argument("--compare-baseline-path", help="Prior baseline output on the Jobs driver node")
    args = parser.parse_args(argv)
    shards = validation.partitions(args.records, args.batch_size)
    if args.actors < 1 or len(shards) < args.actors + 1:
        raise ValueError("Need positive actors and a first checkpoint plus one batch per actor")
    output = Path(args.output_path)
    if not output.is_absolute():
        raise ValueError("output-path must be an absolute run-owned driver path outside working_dir")
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Keep output-path outside Ray's cached working_dir")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Use a new output directory for each application job")
    address = os.environ["NPA_RAY_APP_ADDRESS"]
    if address == "auto" or address.rsplit(":", 1)[-1] == "6380":
        raise ValueError("Explicit application Ray address required; management Ray is forbidden")
    # ray.init also reads RAY_ADDRESS. Set it to the verified application address.
    os.environ["RAY_ADDRESS"] = address
    started = time.perf_counter()
    ray.init(address=address, namespace=f"clip-{uuid.uuid4().hex}")
    gpu_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if ray.cluster_resources().get("GPU", 0) < args.actors:
        raise ValueError("Application Ray has fewer GPUs than requested actors")
    actor_type = ray.remote(num_gpus=1, num_cpus=1, scheduling_strategy="SPREAD")(ClipActor)
    actors = [actor_type.remote(args.model_path, args.model_revision) for _ in range(args.actors)]
    all_initializations = []
    barrier = None
    try:
        all_initializations.extend(ray.get([actor.status.remote() for actor in actors]))
        fingerprints = {info["execution_fingerprint"] for info in all_initializations}
        if len(fingerprints) != 1:
            raise ValueError("GPU workers disagree on source, model weights, UDF, or runtime fingerprint")
        fingerprint = next(iter(fingerprints))
        validation.verify_execution(output, fingerprint)
        model_ready = time.perf_counter()
        if len({(info["node_id"], tuple(info["gpu_ids"])) for info in all_initializations}) != args.actors:
            raise ValueError("GPU actors did not receive distinct GPU allocations")
        prepare = ray.remote(num_cpus=1)(worker.preprocess_shard)
        first = ray.get(prepare.remote(shards[0]))
        first_path = output / "shards" / "000000"
        receipt, first_pending = submit_shard(actors[0], first, first_path, args.model_revision, fingerprint)
        if receipt is None:
            vectors, measurement = ray.get(first_pending)
            receipt = commit_shard(first_path, first, vectors, measurement, args.model_revision, fingerprint)
        print("RAY_CLIP_FIRST_CHECKPOINT", flush=True)
        recovery = None
        if args.recovery_check:
            old = all_initializations[0]
            ray.kill(actors[0], no_restart=True)
            actors[0] = actor_type.remote(args.model_path, args.model_revision)
            new = ray.get(actors[0].status.remote())
            if new["execution_fingerprint"] != fingerprint:
                raise ValueError("Replacement actor changed execution fingerprint")
            all_initializations.append(new)
            replayed, replay_pending = submit_shard(actors[0], first, first_path, args.model_revision, fingerprint)
            after = ray.get(actors[0].status.remote())
            if replay_pending is not None or not replayed["checkpoint_reused"] or after["inference_calls"] != 0:
                raise ValueError("Committed shard was not reused after actor replacement")
            if replayed["parquet_sha256"] != receipt["parquet_sha256"]:
                raise ValueError("Recovery changed a committed shard")
            recovery = {
                "operation": "kill exact owned actor after committed shard, replace, replay shard",
                "old_instance": old["instance_id"], "new_instance": new["instance_id"],
                "checkpoint_reused": True, "replay_inference_calls": after["inference_calls"],
                "model_reloaded": True, "parquet_sha256": receipt["parquet_sha256"],
            }
            validation.atomic_json(output / "recovery.json", recovery)
        if args.cancellation_probe:
            validation.atomic_json(output / "cancellation-ready.json", {"first_shard": receipt, "actors": all_initializations})
            # Continue real inference until the client stops this exact Jobs submission.
            while True:
                ray.get(actors[0].infer.remote(first))
        prepared = ray.get([prepare.remote(shard) for shard in shards[1:]])
        barrier = ray.remote(num_cpus=0)(InferenceBarrier).remote(args.actors)
        pending = [actors[index % args.actors].infer.remote(shard, barrier if index <= args.actors else None)
                   for index, shard in enumerate(prepared, 1)]
        receipts = [receipt]
        for index, (shard, future) in enumerate(zip(prepared, pending, strict=True), 1):
            vectors, measurement = ray.get(future)
            receipts.append(commit_shard(output / "shards" / f"{index:06d}", shard, vectors, measurement, args.model_revision, fingerprint))
        work_done = time.perf_counter()
        final_actors = ray.get([actor.status.remote() for actor in actors])
        result = aggregate(output, receipts, args.records)
        if args.compare_baseline_path:
            result["full_vector_comparison"] = validation.compare_vectors(
                Path(args.compare_baseline_path) / "embeddings.parquet", output / "embeddings.parquet",
                changed=worker.CROP_POLICY == "right",
            )
        concurrency = ray.get(barrier.status.remote())
        ray.kill(barrier, no_restart=True)
        barrier = None
        if args.actors > 1 and not concurrency["overlap"]:
            raise ValueError("Multi-GPU inference intervals never overlapped")
        report = {
            "schema_version": "npa.ray-clip-development.v1",
            **result,
            "source_sha256": worker.source_hash(),
            "application_sha256": validation.file_hash(Path(__file__)),
            "validation_sha256": validation.file_hash(Path(validation.__file__)),
            "udf_sha256": validation.file_hash(Path(__file__).with_name("npa_lancedb_bdd100k_udfs.py")),
            "crop_policy": worker.CROP_POLICY,
            "model_revision": args.model_revision,
            "execution_fingerprint": fingerprint,
            "physical_nodes": len({info["node_id"] for info in all_initializations}),
            "ray_gpu_nodes_available": len(gpu_nodes), "gpu_actors": args.actors,
            "batch_size": args.batch_size, "shards": len(shards),
            "model_initializations": all_initializations,
            "final_actors": final_actors, "recovery": recovery,
            "concurrent_actor_inference_observed": concurrency["overlap"],
            "concurrency_observation": concurrency,
            "concurrency_timing_boundary": "coordinator receives start before CUDA inference and finish after CUDA synchronize; includes RPC edges",
            "cluster_connect_and_actor_ready_seconds": model_ready - started,
            "preprocessing_and_inference_wall_seconds": work_done - model_ready,
            "preprocessing_task_seconds_sum": sum(item["preprocess_seconds"] for item in receipts),
            "inference_actor_seconds_sum": sum(item["inference"]["inference_seconds"] for item in receipts),
            "application_seconds": time.perf_counter() - started,
            "image_builds": 0,
        }
        validation.atomic_json(output / "report.json", report)
        print("RAY_CLIP_REPORT " + __import__("json").dumps(report, sort_keys=True), flush=True)
        return 0
    finally:
        original_failure = sys.exc_info()[0] is not None
        cleanup_errors = cleanup_actors(ray, actors + ([barrier] if barrier is not None else []))
        try:
            validation.atomic_json(output / "actor-cleanup.json", {"errors": cleanup_errors, "attempted": len(actors)})
        except Exception as exc:
            cleanup_errors.append({"operation": "write cleanup receipt", "error_type": type(exc).__name__})
        if cleanup_errors:
            print("RAY_CLIP_CLEANUP " + __import__("json").dumps(cleanup_errors), flush=True)
            if not original_failure:
                raise RuntimeError("Application actor cleanup failed; see cleanup receipt")


def cleanup_actors(ray, actors: list) -> list[dict]:
    """Attempt every owned actor and driver shutdown, preserving prior failures."""
    errors = []
    for index, actor in enumerate(actors):
        try:
            ray.kill(actor, no_restart=True)
        except Exception as exc:
            errors.append({"operation": "kill actor", "actor_index": index, "error_type": type(exc).__name__})
    try:
        ray.shutdown()
    except Exception as exc:
        errors.append({"operation": "driver shutdown", "error_type": type(exc).__name__})
    return errors


if __name__ == "__main__":
    raise SystemExit(main())

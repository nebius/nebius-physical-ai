# Responsibility: Run distributed Workbench CLIP inference and verify recoverable shard outputs.
"""Real Workbench CLIP inference with driver-owned durable shard checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import validation
import worker


FINGERPRINT_FIELDS = (
    "source_sha256", "application_sha256", "validation_sha256", "udf_sha256",
    "model_revision", "model_files", "precision", "python", "ray", "torch", "cuda",
    "transformers", "pyarrow", "lancedb", "gpu_capability",
)


def model_snapshot_files(checkpoint: Path) -> list[dict]:
    """Fingerprint model files while excluding download bookkeeping.

    Args:
        checkpoint: Directory containing the prepared model snapshot.
    Returns:
        Relative filenames and SHA-256 hashes in filename order.
    Raises:
        OSError: A model file cannot be read.
    """
    files = []
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_file():
            files.append({"path": relative.as_posix(), "sha256": validation.file_hash(path)})
    return files


def load_workbench_udf():
    """Import the canonical Workbench UDF from this worker's Jobs package.

    Args:
        None.
    Returns:
        The imported Workbench image-embedding module.
    Raises:
        RuntimeError: The Jobs package lacks a regular, importable UDF file.
        ImportError: A dependency required by the UDF cannot be imported.
    """
    # Imported modules resolve through each worker's own runtime_env directory.
    path = Path(worker.__file__).with_name("npa_lancedb_bdd100k_udfs.py")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("The Jobs working_dir lacks a regular Workbench CLIP UDF")
    specification = importlib.util.spec_from_file_location("npa_lancedb_bdd100k_udfs", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("The Jobs working_dir lacks the real Workbench CLIP UDF")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_cuda_model(checkpoint):
    """Load the actual Workbench CLIP implementation on the allocated GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires real CUDA inference")
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise ValueError("model-path must be a previously fetched CLIP snapshot")
    module = load_workbench_udf()
    module.CLIP_MODEL_NAME = str(checkpoint.resolve())
    module._clip_components(device="cuda:0", precision="float32")
    torch.cuda.synchronize()
    return module


def _source_provenance(module):
    """Identify bytes imported by this process instead of a driver's neighbors."""
    return {
        "source_sha256": worker.source_hash(),
        "worker_module_path": str(Path(worker.__file__).resolve()),
        "application_sha256": validation.file_hash(Path(__file__)),
        "application_module_path": str(Path(__file__).resolve()),
        "udf_sha256": validation.file_hash(Path(module.__file__)),
        "udf_module_path": str(Path(module.__file__).resolve()),
        "validation_sha256": validation.file_hash(Path(validation.__file__)),
    }


def _runtime_provenance():
    """Record the GPU allocation and dependency versions of an actor."""
    from importlib.metadata import version
    import ray
    import torch

    context = ray.get_runtime_context()
    return {
        "instance_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "node_id": context.get_node_id(),
        "gpu_ids": context.get_accelerator_ids().get("GPU", []),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "python": sys.version.split()[0],
        "ray": ray.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": version("transformers"),
        "pyarrow": version("pyarrow"),
        "lancedb": version("lancedb"),
    }


def _model_provenance(checkpoint, revision, load_seconds):
    """Record actual model bytes independently of the requested revision label."""
    files = model_snapshot_files(checkpoint)
    weights = [entry for entry in files if entry["path"].endswith((".safetensors", ".bin"))]
    if not weights:
        raise ValueError("Model snapshot has no verifiable PyTorch weights")
    return {
        "model_revision": revision,
        "model_config_sha256": validation.file_hash(checkpoint / "config.json"),
        "model_load_seconds": load_seconds,
        "model_files": files,
        "precision": "float32",
        "model_initializations": 1,
    }


def _embed_shard(module, shard):
    """Use the real Workbench UDF on each decoded image in the shard."""
    import pyarrow

    images = [row["image_bytes"] for row in shard["rows"]]
    batch = pyarrow.record_batch({"image_bytes": images})
    return module.udf_clip_embedding(batch, device="cuda:0", precision="float32")


def _verify_vectors(vectors, records):
    """Reject incomplete, non-finite or unnormalized CLIP results."""
    import numpy

    matrix = numpy.asarray(vectors.to_pylist(), dtype=numpy.float32)
    if matrix.shape != (records, 512) or not numpy.isfinite(matrix).all():
        raise ValueError("Invalid CLIP vectors")
    lengths = numpy.linalg.norm(matrix, axis=1)
    if not numpy.allclose(lengths, 1.0, atol=1e-4):
        raise ValueError("CLIP vectors are not normalized")


class ClipActor:
    """Keep one real CLIP model resident across batches on one Ray GPU.

    Args:
        model_path: Prepared local model snapshot directory.
        model_revision: Immutable public revision used during preparation.
    Returns:
        An actor implementation for ray.remote(num_gpus=1).
    Raises:
        RuntimeError: CUDA or the canonical UDF is unavailable.
        ValueError: The model snapshot is missing or unverifiable.
    """

    def __init__(self, model_path: str, model_revision: str):
        """Load the model and fingerprint this actor's imports and runtime.

        Args:
            model_path: Prepared model snapshot directory.
            model_revision: Immutable snapshot revision label.
        Returns:
            None.
        Raises:
            RuntimeError: CUDA or the Workbench UDF is unavailable.
            ValueError: Model configuration or weights cannot be verified.
        """
        started = time.perf_counter()
        checkpoint = Path(model_path)
        self.udf = _load_cuda_model(checkpoint)
        load_seconds = time.perf_counter() - started
        verification_started = time.perf_counter()
        self.revision = model_revision
        self.info = _source_provenance(self.udf)
        self.info.update(_runtime_provenance())
        self.info.update(_model_provenance(checkpoint, model_revision, load_seconds))
        identity = {field: self.info[field] for field in FINGERPRINT_FIELDS}
        self.info["execution_fingerprint"] = validation.canonical_hash(identity)
        self.info["fingerprint_verification_seconds"] = time.perf_counter() - verification_started
        self.inference_calls = 0

    def status(self) -> dict:
        """Return immutable provenance and the number of completed batches.

        Args:
            None.
        Returns:
            Actor provenance plus the current inference call count.
        Raises:
            None.
        """
        return {**self.info, "inference_calls": self.inference_calls}

    def infer(self, shard: dict, barrier=None) -> tuple[object, dict]:
        """Embed a preprocessed shard using the resident CUDA model.

        Args:
            shard: Images and CPU-worker source provenance.
            barrier: Optional Ray actor observing concurrent inference.
        Returns:
            Arrow vectors and synchronized inference timing.
        Raises:
            ValueError: Source provenance or returned vectors are invalid.
            RuntimeError: CUDA inference fails.
        """
        import ray
        import torch

        if shard["source_sha256"] != worker.source_hash():
            raise ValueError("CPU and GPU workers imported different application revisions")
        if barrier is not None:
            ray.get(barrier.arrive.remote(self.info["instance_id"]))
            ray.get(barrier.start.remote(self.info["instance_id"]))
        torch.cuda.synchronize()
        started = time.monotonic_ns()
        vectors = _embed_shard(self.udf, shard)
        torch.cuda.synchronize()
        finished = time.monotonic_ns()
        if barrier is not None:
            ray.get(barrier.finish.remote(self.info["instance_id"]))
        _verify_vectors(vectors, len(shard["rows"]))
        self.inference_calls += 1
        return vectors, {"instance_id": self.info["instance_id"],
                         "start_monotonic_ns": started, "end_monotonic_ns": finished,
                         "inference_seconds": (finished - started) / 1e9}


class InferenceBarrier:
    """Observe overlapping actor calls using one coordinator clock.

    Args:
        participants: Number of GPU actors in the observed inference wave.
    Returns:
        An asynchronous actor implementation for ray.remote.
    Raises:
        None.
    """

    def __init__(self, participants: int):
        """Create the rendezvous and its inference observation state.

        Args:
            participants: Actors required before the first wave starts.
        Returns:
            None.
        Raises:
            None.
        """
        import asyncio

        self.participants = participants
        self.arrived = set()
        self.active = set()
        self.ready = asyncio.Event()
        self.overlap = False
        self.events = []

    async def arrive(self, instance: str):
        """Wait until all participating actors reach the inference wave.

        Args:
            instance: Unique model actor instance identity.
        Returns:
            None after all participants arrive.
        Raises:
            asyncio.CancelledError: The waiting coroutine is cancelled.
        """
        self.arrived.add(instance)
        if len(self.arrived) == self.participants:
            self.ready.set()
        await self.ready.wait()

    async def start(self, instance: str):
        """Record the coordinator's observation of an inference start.

        Args:
            instance: Model actor beginning its CUDA call.
        Returns:
            None.
        Raises:
            None.
        """
        self.active.add(instance)
        if len(self.active) > 1:
            self.overlap = True
        self.events.append({"instance_id": instance, "event": "start", "monotonic_ns": time.monotonic_ns()})

    async def finish(self, instance: str):
        """Record completion after the actor synchronizes its CUDA call.

        Args:
            instance: Model actor whose inference completed.
        Returns:
            None.
        Raises:
            KeyError: The actor had no corresponding active start.
        """
        self.events.append({"instance_id": instance, "event": "finish", "monotonic_ns": time.monotonic_ns()})
        self.active.remove(instance)

    async def status(self):
        """Return concurrency observations without combining worker clocks.

        Args:
            None.
        Returns:
            Participant count, observed overlap and coordinator events.
        Raises:
            None.
        """
        return {"participants": len(self.arrived), "overlap": self.overlap, "events": self.events}


def _write_shard_vectors(directory, shard, vectors):
    """Publish one complete Parquet file before its commit marker appears."""
    import pyarrow
    import pyarrow.parquet as parquet

    table = pyarrow.table({
        "record_id": [row["record_id"] for row in shard["rows"]],
        "input_sha256": [row["input_sha256"] for row in shard["rows"]],
        "processed_sha256": [row["processed_sha256"] for row in shard["rows"]],
        "vector": vectors,
    })
    temporary = directory / f".{uuid.uuid4().hex}.parquet"
    try:
        parquet.write_table(table, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "embeddings.parquet")
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_receipt(directory, shard, measurement, identity):
    """Describe the committed bytes and the workers that produced them."""
    provenance_fields = ("source_sha256", "source_path", "pid", "node_id")
    return {
        "identity": identity,
        "rows": len(shard["rows"]),
        "parquet_sha256": validation.file_hash(directory / "embeddings.parquet"),
        "preprocess_seconds": shard["preprocess_seconds"],
        "preprocessor": {field: shard[field] for field in provenance_fields},
        "inference": measurement,
        "checkpoint_reused": False,
    }


def commit_shard(path: Path, shard: dict, vectors, measurement: dict,
                 model_revision: str, execution_fingerprint: str) -> dict:
    """Commit a driver-owned shard or reuse an identical verified checkpoint.

    Args:
        path: Checkpoint directory on the driver.
        shard: Preprocessed rows and their provenance.
        vectors: Arrow CLIP vectors for those rows.
        measurement: Completed actor inference timing.
        model_revision: Immutable model revision.
        execution_fingerprint: Source, model and runtime identity.
    Returns:
        A durable checkpoint receipt.
    Raises:
        ValueError: Existing checkpoint identity or data differs.
        OSError: Checkpoint bytes or commit metadata cannot be written.
    """
    identity = validation.checkpoint_identity(shard, model_revision, execution_fingerprint)
    cached = validation.read_checkpoint(path, identity)
    if cached is not None:
        return {**cached, "checkpoint_reused": True}
    path.mkdir(parents=True, exist_ok=True)
    _write_shard_vectors(path, shard, vectors)
    receipt = _checkpoint_receipt(path, shard, measurement, identity)
    validation.atomic_json(path / "commit.json", receipt)
    return receipt


def submit_shard(actor, shard: dict, directory: Path, model_revision: str,
                 execution_fingerprint: str, barrier=None):
    """Reuse a committed shard before scheduling another GPU inference call.

    Args:
        actor: Native Ray model actor.
        shard: Preprocessed image records.
        directory: Driver-owned checkpoint directory.
        model_revision: Immutable model revision.
        execution_fingerprint: Source, model and runtime identity.
        barrier: Optional concurrency observation actor.
    Returns:
        A cached receipt and no future, or no receipt and a Ray future.
    Raises:
        ValueError: A committed checkpoint does not match this execution.
        OSError: Checkpoint data cannot be read.
    """
    identity = validation.checkpoint_identity(shard, model_revision, execution_fingerprint)
    cached = validation.read_checkpoint(directory, identity)
    if cached is not None:
        return {**cached, "checkpoint_reused": True}, None
    return None, actor.infer.remote(shard, barrier)


def _read_completed_table(output, receipts, records):
    """Require verified shards to contain exactly the requested record IDs."""
    import pyarrow
    import pyarrow.parquet as parquet

    tables = []
    for index, receipt in enumerate(receipts):
        directory = output / "shards" / f"{index:06d}"
        committed = validation.read_checkpoint(directory, receipt["identity"])
        if committed is None:
            raise ValueError("A completed shard lacks its commit marker")
        tables.append(parquet.read_table(directory / "embeddings.parquet"))
    table = pyarrow.concat_tables(tables).sort_by("record_id")
    validation.verify_ids(table["record_id"].to_pylist(), records)
    return table


def _persist_lance_table(output, table, records):
    """Persist the complete vector dataset in a locally inspectable Lance table."""
    import lancedb

    database = lancedb.connect(str(output / "lance"))
    lance_table = database.create_table("embeddings", table, mode="overwrite")
    if lance_table.count_rows() != records:
        raise ValueError("Lance persistence lost rows")
    return lance_table


def _check_retrievals(lance_table, vectors, records):
    """Verify representative self-queries against the persisted table."""
    query_ids = [0, records // 4, records // 2, 3 * records // 4, records - 1]
    retrievals = []
    for query_id in dict.fromkeys(query_ids):
        hits = lance_table.search(vectors[query_id]).metric("cosine").limit(5).to_list()
        top_ids = [row["record_id"] for row in hits]
        if query_id not in top_ids:
            raise ValueError(f"Lance self-query missed record {query_id}")
        retrievals.append({"query_id": query_id, "top_ids": top_ids})
    return retrievals


def _preview_images(row):
    """Recreate preview bytes only when they match the actual embedded inputs."""
    import hashlib

    original = worker.render_record(row["record_id"])
    processed = worker.preprocess_image(original)
    original_hash = hashlib.sha256(original).hexdigest()
    processed_hash = hashlib.sha256(processed).hexdigest()
    if original_hash != row["input_sha256"] or processed_hash != row["processed_sha256"]:
        raise ValueError("Rendered preview differs from the embedded inputs")
    return (("original", original), ("crop", processed))


def _render_preview_row(preview, images, index, row):
    """Decode the verified RGB inputs into visible originals and crops."""
    import io
    from PIL import Image

    for column, (label, image_bytes) in enumerate(_preview_images(row)):
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError("Expected decoded RGB inputs")
            image.save(images / f"{index:06d}-{label}.png")
            preview.paste(image.resize((224, 224)), (224 * column, 224 * index))


def _render_previews(output, table, records):
    """Create a small image contact sheet alongside the complete vectors."""
    from PIL import Image

    images = output / "images"
    images.mkdir(exist_ok=True)
    preview = Image.new("RGB", (448, 224 * min(8, records)))
    for index, row in enumerate(table.slice(0, 8).to_pylist()):
        _render_preview_row(preview, images, index, row)
    preview.save(output / "preview.png")


def _aggregation_report(output, table, vectors, retrievals, started, lance_rows):
    """Summarize persisted bytes separately from application execution timing."""
    import hashlib
    import numpy

    return {
        "records": len(table),
        "input_hash": validation.canonical_hash(table["input_sha256"].to_pylist()),
        "processed_hash": validation.canonical_hash(table["processed_sha256"].to_pylist()),
        "embedding_sha256": validation.file_hash(output / "embeddings.parquet"),
        "vector_bytes_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "mean_embedding": vectors.mean(axis=0, dtype=numpy.float64).tolist(),
        "lance_rows": lance_rows,
        "retrieval_queries": len(retrievals),
        "aggregation_seconds": time.perf_counter() - started,
    }


def aggregate(output: Path, receipts: list[dict], records: int) -> dict:
    """Persist complete Parquet/Lance outputs and verify retrieval and RGB inputs.

    Args:
        output: Driver-owned result directory.
        receipts: Ordered committed shard receipts.
        records: Expected number of distinct records.
    Returns:
        Artifact identities, completeness and aggregation measurements.
    Raises:
        ValueError: Checkpoint, completeness, retrieval or preview checks fail.
        OSError: An input or output artifact cannot be accessed.
    """
    import numpy
    import pyarrow.parquet as parquet

    started = time.perf_counter()
    table = _read_completed_table(output, receipts, records)
    vectors = numpy.asarray(table["vector"].to_pylist(), dtype=numpy.float32)
    lance_table = _persist_lance_table(output, table, records)
    retrievals = _check_retrievals(lance_table, vectors, records)
    parquet.write_table(table, output / "embeddings.parquet")
    validation.atomic_json(output / "retrieval.json", retrievals)
    _render_previews(output, table, records)
    lance_rows = lance_table.count_rows()
    return _aggregation_report(output, table, vectors, retrievals, started, lance_rows)


def _arguments(argv):
    """Expose ordinary application options to the native Ray Jobs entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-path", default="/tmp/npa-clip-model")
    parser.add_argument("--model-revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    parser.add_argument("--records", type=int, default=2048)
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recovery-check", action="store_true")
    parser.add_argument("--cancellation-probe", action="store_true")
    parser.add_argument("--compare-baseline-path", help="Prior baseline output on the Jobs driver node")
    return parser.parse_args(argv)


def _validate_output_directory(output):
    """Keep durable output separate from Ray's cached application source."""
    if not output.is_absolute():
        raise ValueError("output-path must be an absolute run-owned driver path outside working_dir")
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Keep output-path outside Ray's cached working_dir")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Use a new output directory for each application job")


def _verify_actor_allocations(initializations, actors):
    """Require matching execution identities on distinct Ray GPU allocations."""
    fingerprints = {information["execution_fingerprint"] for information in initializations}
    if len(fingerprints) != 1:
        raise ValueError("GPU workers disagree on source, model weights, UDF, or runtime fingerprint")
    allocations = {(information["node_id"], tuple(information["gpu_ids"])) for information in initializations}
    if len(allocations) != actors:
        raise ValueError("GPU actors did not receive distinct GPU allocations")
    return next(iter(fingerprints))


def _recovery_receipt(old, replacement, replay, pending, committed):
    """Prove actor replacement reused an unchanged committed shard."""
    if pending is not None or not replay["checkpoint_reused"] or replacement["inference_calls"] != 0:
        raise ValueError("Committed shard was not reused after actor replacement")
    if replay["parquet_sha256"] != committed["parquet_sha256"]:
        raise ValueError("Recovery changed a committed shard")
    return {
        "operation": "kill exact owned actor after committed shard, replace, replay shard",
        "old_instance": old["instance_id"],
        "new_instance": replacement["instance_id"],
        "checkpoint_reused": True,
        "replay_inference_calls": replacement["inference_calls"],
        "model_reloaded": True,
        "parquet_sha256": committed["parquet_sha256"],
    }


class _InferenceSession:
    """Coordinate driver-owned application stages within one native Ray Job."""

    def __init__(self, arguments, shards):
        """Retain application state so lifecycle stages have explicit names."""
        import ray

        self.ray = ray
        self.arguments = arguments
        self.shards = shards
        self.output = Path(arguments.output_path)
        _validate_output_directory(self.output)
        self.actors = []
        self.initializations = []
        self.barrier = None
        self.recovery = None
        self.receipts = []

    def _connect(self):
        """Connect only to the application cluster injected by Ray Jobs."""
        from application import ClipActor, InferenceBarrier

        address = os.environ.get("RAY_ADDRESS", "")
        if not address.endswith(":6381") or "://" in address or not address[:-5]:
            raise ValueError("Submit through application Ray Jobs on GCS port 6381")
        self.started = time.perf_counter()
        self.ray.init(address=address, namespace=f"clip-{uuid.uuid4().hex}")
        self.gpu_nodes = [node for node in self.ray.nodes()
                          if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
        if self.ray.cluster_resources().get("GPU", 0) < self.arguments.actors:
            raise ValueError("Application Ray has fewer GPUs than requested actors")
        self.clip_actor = self.ray.remote(num_gpus=1, num_cpus=1, scheduling_strategy="SPREAD")(ClipActor)
        self.inference_barrier = self.ray.remote(num_cpus=0)(InferenceBarrier)
        self.prepare_shard = self.ray.remote(num_cpus=1)(worker.preprocess_shard)

    def _new_actor(self):
        """Allocate one model actor using the prepared immutable snapshot."""
        return self.clip_actor.remote(self.arguments.model_path, self.arguments.model_revision)

    def _initialize_actors(self):
        """Verify model/runtime agreement before any output is committed."""
        self.actors = [self._new_actor() for _ in range(self.arguments.actors)]
        self.initializations = self.ray.get([actor.status.remote() for actor in self.actors])
        self.fingerprint = _verify_actor_allocations(self.initializations, self.arguments.actors)
        validation.verify_execution(self.output, self.fingerprint)
        self.model_ready = time.perf_counter()

    def _commit_first_shard(self):
        """Establish a real checkpoint for the recovery or cancellation check."""
        self.first_shard = self.ray.get(self.prepare_shard.remote(self.shards[0]))
        self.first_path = self.output / "shards" / "000000"
        receipt, pending = submit_shard(self.actors[0], self.first_shard, self.first_path,
                                       self.arguments.model_revision, self.fingerprint)
        if receipt is None:
            vectors, measurement = self.ray.get(pending)
            receipt = commit_shard(self.first_path, self.first_shard, vectors, measurement,
                                   self.arguments.model_revision, self.fingerprint)
        self.receipts.append(receipt)
        print("RAY_CLIP_FIRST_CHECKPOINT", flush=True)

    def _recover_actor(self):
        """Replace only the failed application actor and replay its checkpoint."""
        if not self.arguments.recovery_check:
            return
        old = self.initializations[0]
        self.ray.kill(self.actors[0], no_restart=True)
        self.actors[0] = self._new_actor()
        replacement = self.ray.get(self.actors[0].status.remote())
        if replacement["execution_fingerprint"] != self.fingerprint:
            raise ValueError("Replacement actor changed execution fingerprint")
        self.initializations.append(replacement)
        replay, pending = submit_shard(self.actors[0], self.first_shard, self.first_path,
                                      self.arguments.model_revision, self.fingerprint)
        after_replay = self.ray.get(self.actors[0].status.remote())
        self.recovery = _recovery_receipt(old, after_replay, replay, pending, self.receipts[0])
        validation.atomic_json(self.output / "recovery.json", self.recovery)

    def _run_until_cancelled(self):
        """Continue real inference until the customer stops this exact Ray Job."""
        if not self.arguments.cancellation_probe:
            return
        ready = {"first_shard": self.receipts[0], "actors": self.initializations}
        validation.atomic_json(self.output / "cancellation-ready.json", ready)
        while True:
            self.ray.get(self.actors[0].infer.remote(self.first_shard))

    def _schedule_batches(self, prepared):
        """Schedule concurrent GPU batches and observe the first inference wave."""
        self.barrier = self.inference_barrier.remote(self.arguments.actors)
        pending = []
        for index, shard in enumerate(prepared, 1):
            actor = self.actors[index % self.arguments.actors]
            observation = None
            if index <= self.arguments.actors:
                observation = self.barrier
            pending.append(actor.infer.remote(shard, observation))
        return pending

    def _infer_remaining_shards(self):
        """Preprocess partitions, run concurrent inference and commit all outputs."""
        prepared = self.ray.get([self.prepare_shard.remote(shard) for shard in self.shards[1:]])
        pending = self._schedule_batches(prepared)
        for index, (shard, future) in enumerate(zip(prepared, pending, strict=True), 1):
            vectors, measurement = self.ray.get(future)
            directory = self.output / "shards" / f"{index:06d}"
            receipt = commit_shard(directory, shard, vectors, measurement,
                                   self.arguments.model_revision, self.fingerprint)
            self.receipts.append(receipt)
        self.work_done = time.perf_counter()
        self.final_actors = self.ray.get([actor.status.remote() for actor in self.actors])

    def _check_concurrency(self):
        """Require actual observed overlap instead of inferring it from SPREAD."""
        observation = self.ray.get(self.barrier.status.remote())
        self.ray.kill(self.barrier, no_restart=True)
        self.barrier = None
        if self.arguments.actors > 1 and not observation["overlap"]:
            raise ValueError("Multi-GPU inference intervals never overlapped")
        return observation

    def _execution_report(self):
        """Attach driver and every actor's source and execution identities."""
        return {
            "schema_version": "npa.ray-clip-development.v1",
            "source_sha256": worker.source_hash(),
            "application_sha256": validation.file_hash(Path(__file__)),
            "validation_sha256": validation.file_hash(Path(validation.__file__)),
            "udf_sha256": validation.file_hash(Path(__file__).with_name("npa_lancedb_bdd100k_udfs.py")),
            "crop_policy": worker.CROP_POLICY,
            "model_revision": self.arguments.model_revision,
            "execution_fingerprint": self.fingerprint,
            "ray_nodes": len({information["node_id"] for information in self.initializations}),
            "ray_gpu_nodes_available": len(self.gpu_nodes),
            "gpu_actors": self.arguments.actors,
            "batch_size": self.arguments.batch_size,
            "shards": len(self.shards),
            "model_initializations": self.initializations,
            "final_actors": self.final_actors,
            "recovery": self.recovery,
        }

    def _timing_report(self, observation):
        """Separate coordinator observations, actor sums and driver wall time."""
        return {
            "concurrent_actor_inference_observed": observation["overlap"],
            "concurrency_observation": observation,
            "concurrency_timing_boundary": "coordinator receives start before CUDA inference and finish after CUDA synchronize; includes RPC edges",
            "cluster_connect_and_actor_ready_seconds": self.model_ready - self.started,
            "preprocessing_and_inference_wall_seconds": self.work_done - self.model_ready,
            "preprocessing_task_seconds_sum": sum(receipt["preprocess_seconds"] for receipt in self.receipts),
            "inference_actor_seconds_sum": sum(receipt["inference"]["inference_seconds"] for receipt in self.receipts),
            "application_seconds": time.perf_counter() - self.started,
        }

    def _write_report(self):
        """Publish verified useful results before the application actors close."""
        result = aggregate(self.output, self.receipts, self.arguments.records)
        if self.arguments.compare_baseline_path:
            baseline = Path(self.arguments.compare_baseline_path) / "embeddings.parquet"
            result["full_vector_comparison"] = validation.compare_vectors(
                baseline, self.output / "embeddings.parquet", changed=worker.CROP_POLICY == "right")
        observation = self._check_concurrency()
        report = self._execution_report()
        report.update(result)
        report.update(self._timing_report(observation))
        validation.atomic_json(self.output / "report.json", report)
        fields = ("records", "lance_rows", "retrieval_queries", "crop_policy", "ray_nodes",
                  "gpu_actors", "application_seconds", "concurrent_actor_inference_observed")
        visible_report = {field: report[field] for field in fields}
        print("RAY_CLIP_REPORT " + json.dumps(visible_report, sort_keys=True), flush=True)

    def _close(self, original_failure):
        """Attempt every owned actor and preserve prior failures during cleanup."""
        owned_actors = list(self.actors)
        if self.barrier is not None:
            owned_actors.append(self.barrier)
        errors = cleanup_actors(self.ray, owned_actors)
        try:
            _write_cleanup_artifacts(self.output, errors, len(self.actors))
        except Exception as error:
            errors.append({"operation": "write cleanup receipt", "error_type": type(error).__name__})
        if not errors:
            return
        print("RAY_CLIP_CLEANUP " + json.dumps(errors), flush=True)
        if not original_failure:
            raise RuntimeError("Application actor cleanup failed; see cleanup receipt")


def _write_cleanup_artifacts(output, errors, attempted):
    """Preserve cleanup results and portable hashes before worker storage closes."""
    validation.atomic_json(output / "actor-cleanup.json", {"errors": errors, "attempted": attempted})
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{validation.file_hash(path)}  {path.relative_to(output)}\n")
    (output / "SHA256SUMS").write_text("".join(entries))


def main(argv: list[str] | None = None) -> int:
    """Run the advanced CLIP application inside a standard Ray Jobs submission.

    Args:
        argv: Application options, or None to read command-line arguments.
    Returns:
        Zero after all workload and artifact checks pass.
    Raises:
        ValueError: Inputs, provenance or correctness checks fail.
        RuntimeError: GPU execution, Ray coordination or actor cleanup fails.
        OSError: Model, checkpoint or output storage is inaccessible.
    """
    arguments = _arguments(argv)
    shards = validation.partitions(arguments.records, arguments.batch_size)
    if arguments.actors < 1 or len(shards) < arguments.actors + 1:
        raise ValueError("Need positive actors and a first checkpoint plus one batch per actor")
    session = _InferenceSession(arguments, shards)
    try:
        session._connect()
        session._initialize_actors()
        session._commit_first_shard()
        session._recover_actor()
        session._run_until_cancelled()
        session._infer_remaining_shards()
        session._write_report()
        return 0
    finally:
        session._close(original_failure=sys.exc_info()[0] is not None)


def cleanup_actors(ray, actors: list) -> list[dict]:
    """Attempt every owned actor shutdown even after an earlier cleanup failure.

    Args:
        ray: Connected Ray module or a test implementation of its lifecycle API.
        actors: Exact application actors owned by this Job.
    Returns:
        Descriptions of cleanup failures; an empty list means all calls passed.
    Raises:
        None. Ray cleanup exceptions are returned as evidence.
    """
    errors = []
    for index, actor in enumerate(actors):
        try:
            ray.kill(actor, no_restart=True)
        except Exception as error:
            errors.append({"operation": "kill actor", "actor_index": index,
                           "error_type": type(error).__name__})
    try:
        ray.shutdown()
    except Exception as error:
        errors.append({"operation": "driver shutdown", "error_type": type(error).__name__})
    return errors


if __name__ == "__main__":
    raise SystemExit(main())

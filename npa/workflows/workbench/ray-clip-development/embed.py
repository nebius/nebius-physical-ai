# Run CLIP inference with Ray actors and persist inspectable image/vector artifacts.
"""Embed rendered RGB images with Workbench CLIP using ordinary Ray Core actors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import worker

MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WEIGHT_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
SOURCE_FILES = ("embed.py", "worker.py", "npa_lancedb_bdd100k_udfs.py")


def sha256(path: Path) -> str:
    """Hash a source, model or artifact file without loading it into memory.

    Args:
        path: File whose exact bytes identify its contents.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            content = stream.read(1024 * 1024)
            if not content:
                break
            digest.update(content)
    return digest.hexdigest()


def source_hashes() -> dict:
    """Identify the declared application and Workbench source files.

    Args:
        None.
    Returns:
        Source filenames mapped to their SHA-256 digests.
    Raises:
        ValueError: A required application or Workbench module is missing or linked.
        OSError: A delivered source file cannot be read.
    """
    directory = Path(__file__).parent
    hashes = {}
    for name in SOURCE_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Ray working_dir must contain the application, worker and canonical Workbench UDF")
        hashes[name] = sha256(path)
    return hashes


def application_gcs_address() -> str:
    """Require the application GCS address supplied to the Ray Jobs driver.

    Args:
        None; reads the upstream RAY_ADDRESS environment variable.
    Returns:
        Explicit application cluster address on port 6381.
    Raises:
        ValueError: The address is missing or could select a management runtime.
    """
    address = os.environ.get("RAY_ADDRESS", "")
    if not address.endswith(":6381") or "://" in address or not address[:-5]:
        raise ValueError("Submit through the application Ray Jobs server; its GCS must use port 6381")
    return address


def _imported_source_receipt(workbench) -> dict:
    """Reject stale imports even when different files exist in working_dir."""
    paths = {
        "embed.py": __file__,
        "worker.py": worker.__file__,
        "npa_lancedb_bdd100k_udfs.py": workbench.__file__,
    }
    hashes = {name: sha256(Path(path)) for name, path in paths.items()}
    if hashes != source_hashes():
        raise ValueError("Imported application/UDF modules differ from the Jobs working_dir")
    return {"source_sha256": hashes, "imported_paths": paths}


def _load_pinned_model(workbench, checkpoint: Path) -> dict:
    """Measure weight verification separately from CUDA model initialization."""
    import torch

    started = time.perf_counter()
    weight_hash = sha256(checkpoint / "pytorch_model.bin")
    if weight_hash != WEIGHT_SHA256:
        raise ValueError("CLIP weights do not match the pinned public snapshot")
    verification_seconds = time.perf_counter() - started
    started = time.perf_counter()
    workbench.CLIP_MODEL_NAME = str(checkpoint)
    workbench._clip_components(device="cuda:0", precision="float32")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    return {
        "model_revision": MODEL_REVISION,
        "weight_sha256": weight_hash,
        "model_config_sha256": sha256(checkpoint / "config.json"),
        "model_verification_seconds": verification_seconds,
        "model_load_seconds": load_seconds,
    }


def _cuda_actor_receipt() -> dict:
    """Report actual process, GPU placement and dependency versions."""
    import importlib.metadata

    import ray
    import torch

    packages = ("ray", "torch", "transformers", "lancedb", "pyarrow", "numpy", "pillow")
    versions = {package: importlib.metadata.version(package) for package in packages}
    context = ray.get_runtime_context()
    return {
        "node_id": context.get_node_id(),
        "gpu_ids": context.get_accelerator_ids().get("GPU", []),
        "pid": os.getpid(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "versions": versions,
    }


class ClipModel:
    """Keep one CUDA model loaded for successive batches in a Ray actor.

    Args:
        model_path: Directory containing the pinned public CLIP snapshot.
    Returns:
        A model actor implementation with source and runtime provenance.
    Raises:
        RuntimeError: CUDA is unavailable or model initialization fails.
        ValueError: Imported source or model weights differ from their pins.
        OSError: Source or model files cannot be read.
    """

    def __init__(self, model_path: str):
        """Verify source and load one pinned CLIP model onto this actor's GPU.

        Args:
            model_path: Local snapshot prepared in this SkyPilot pod.
        Returns:
            None.
        Raises:
            RuntimeError: CUDA is unavailable or model initialization fails.
            ValueError: Source or model hashes differ from expected bytes.
            OSError: Source or model files cannot be read.
        """
        import npa_lancedb_bdd100k_udfs as workbench
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This example requires an actual CUDA GPU")
        self.receipt = _imported_source_receipt(workbench)
        self.receipt.update(_load_pinned_model(workbench, Path(model_path)))
        self.receipt.update(_cuda_actor_receipt())
        self.workbench = workbench
        self.calls = 0

    def status(self) -> dict:
        """Return actor provenance and its completed inference-call count.

        Args:
            None.
        Returns:
            A receipt describing this actor and its model initialization.
        Raises:
            None.
        """
        return {**self.receipt, "inference_calls": self.calls}

    def infer(self, shard: dict) -> dict:
        """Embed one prepared image batch through the real Workbench UDF.

        Args:
            shard: Image rows and provenance returned by preprocess_shard.
        Returns:
            Embedding rows, timing and CPU-worker provenance.
        Raises:
            ValueError: Preprocessor source differs from this actor's source.
            RuntimeError: CUDA inference fails.
        """
        import pyarrow
        import torch

        if shard["source_sha256"] != worker.source_hash():
            raise ValueError("Preprocessor and GPU actor imported different source")
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = pyarrow.record_batch({"image_bytes": [row["image_bytes"] for row in shard["rows"]]})
        vectors = self.workbench.udf_clip_embedding(batch, device="cuda:0", precision="float32")
        torch.cuda.synchronize()
        self.calls += 1
        rows = []
        for row, vector in zip(shard["rows"], vectors.to_pylist(), strict=True):
            rows.append({**row, "vector": vector})
        return {
            "rows": rows,
            "inference_seconds": time.perf_counter() - started,
            "preprocessing_seconds": shard["preprocess_seconds"],
            "preprocessor": {key: shard[key] for key in ("source_sha256", "node_id", "pid")},
        }


def _complete_normalized_vectors(rows: list[dict], records: int):
    """Reject incomplete or malformed results before creating output artifacts."""
    import numpy

    if [row["record_id"] for row in rows] != list(range(records)):
        raise ValueError("Output has missing, duplicate or unexpected record IDs")
    vectors = numpy.asarray([row["vector"] for row in rows], dtype=numpy.float32)
    if vectors.shape != (records, 512) or not numpy.isfinite(vectors).all():
        raise ValueError("Expected finite, normalized 512-dimensional CLIP vectors")
    if not numpy.allclose(numpy.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Expected finite, normalized 512-dimensional CLIP vectors")
    return vectors


def _persist_vector_tables(output: Path, rows: list[dict], vectors):
    """Persist one typed vector table in Parquet and Lance formats."""
    import lancedb
    import pyarrow
    import pyarrow.parquet

    table = pyarrow.table({
        "record_id": [row["record_id"] for row in rows],
        "input_sha256": [row["input_sha256"] for row in rows],
        "processed_sha256": [row["processed_sha256"] for row in rows],
        "vector": pyarrow.array(vectors.tolist(), type=pyarrow.list_(pyarrow.float32(), 512)),
    })
    pyarrow.parquet.write_table(table, output / "embeddings.parquet")
    database = lancedb.connect(str(output / "lance"))
    return database.create_table("embeddings", table)


def _verify_retrieval(table, vectors, records: int) -> list[dict]:
    """Check that persisted vectors can retrieve their own image records."""
    if table.count_rows() != records:
        raise ValueError("Lance table lost output rows")
    retrieval = []
    for query_id in sorted({0, records // 2, records - 1}):
        hits = table.search(vectors[query_id]).metric("cosine").limit(5).to_list()
        record_ids = [hit["record_id"] for hit in hits]
        if query_id not in record_ids:
            raise ValueError("Lance self-retrieval missed its query image")
        retrieval.append({"query_id": query_id, "top_ids": record_ids})
    return retrieval


def _preview_inputs(row: dict) -> tuple[bytes, bytes]:
    """Require preview bytes to match the exact images embedded by the GPU."""
    original = worker.render_record(row["record_id"])
    original_hash = hashlib.sha256(original).hexdigest()
    crop_hash = hashlib.sha256(row["image_bytes"]).hexdigest()
    if original_hash != row["input_sha256"] or crop_hash != row["processed_sha256"]:
        raise ValueError("Preview bytes differ from the images embedded by the GPU")
    return original, row["image_bytes"]


def _save_preview_image(content: bytes, destination: Path, preview, position: tuple) -> None:
    """Decode RGB bytes before including them in the inspectable contact sheet."""
    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError("Expected decoded RGB inputs")
        image.save(destination)
        preview.paste(image.resize((224, 224)), position)


def _save_previews(output: Path, rows: list[dict], records: int) -> None:
    """Show original images beside the crops sent to CLIP."""
    from PIL import Image

    samples = output / "images"
    samples.mkdir()
    preview = Image.new("RGB", (448, 224 * min(records, 8)))
    for index, row in enumerate(rows[:8]):
        original, crop = _preview_inputs(row)
        inputs = (("original", original), ("crop", crop))
        for column, (name, content) in enumerate(inputs):
            destination = samples / f"{row['record_id']:06d}-{name}.png"
            position = (224 * column, 224 * index)
            _save_preview_image(content, destination, preview, position)
    preview.save(output / "preview.png")


def save_results(output: Path, rows: list[dict], records: int) -> dict:
    """Persist complete vectors, decoded RGB previews and verified retrieval.

    Args:
        output: Existing directory reserved for this submission's artifacts.
        rows: Completed image rows carrying vectors and input hashes.
        records: Expected total number of unique, sequential record IDs.
    Returns:
        Artifact counts, retrieval results and vector/Parquet hashes.
    Raises:
        ValueError: Completeness, vector, image or retrieval validation fails.
        OSError: Artifacts cannot be read or written.
    """
    rows = sorted(rows, key=lambda row: row["record_id"])
    vectors = _complete_normalized_vectors(rows, records)
    table = _persist_vector_tables(output, rows, vectors)
    retrieval = _verify_retrieval(table, vectors, records)
    _save_previews(output, rows, records)
    (output / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
    return {
        "records": records,
        "lance_rows": table.count_rows(),
        "retrieval": retrieval,
        "vector_bytes_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "parquet_sha256": sha256(output / "embeddings.parquet"),
    }


def write_hashes(output: Path) -> None:
    """Write manifests that let a reader verify every downloaded artifact.

    Args:
        output: Directory containing this submission's completed artifacts.
    Returns:
        None; writes sha256.json and the standard SHA256SUMS manifest.
    Raises:
        OSError: An artifact or manifest cannot be read or written.
    """
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"sha256.json", "SHA256SUMS"}:
            files[str(path.relative_to(output))] = sha256(path)
    (output / "sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    files["sha256.json"] = sha256(output / "sha256.json")
    checksum_lines = []
    for name, digest in sorted(files.items()):
        checksum_lines.append(f"{digest}  {name}\n")
    (output / "SHA256SUMS").write_text("".join(checksum_lines))


def _parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    """Keep the command-line interface independent from the execution stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default="/tmp/npa-clip-model")
    options = parser.parse_args(arguments)
    if min(options.records, options.actors, options.batch_size) < 1:
        raise ValueError("records, actors and batch-size must be positive")
    output = Path(options.output_path)
    source_directory = Path(__file__).resolve().parent
    if not output.is_absolute() or output.resolve().is_relative_to(source_directory) or output.exists():
        raise ValueError("Use a new absolute output directory outside Ray working_dir")
    return options


def _start_model_actors(options: argparse.Namespace, expected_sources: dict) -> list:
    """Create GPU actors from the imported module and verify their delivered source."""
    import ray
    # Module import makes each actor resolve its own working_dir, not __main__ bytes.
    from embed import ClipModel

    if ray.cluster_resources().get("GPU", 0) < options.actors:
        raise ValueError("The application cluster has fewer GPUs than requested actors")
    Path(options.output_path).mkdir(parents=True)
    clip_actor = ray.remote(num_gpus=1, num_cpus=1, scheduling_strategy="SPREAD")(ClipModel)
    models = [clip_actor.remote(options.model_path) for _ in range(options.actors)]
    receipts = ray.get([model.status.remote() for model in models])
    if any(receipt["source_sha256"] != expected_sources for receipt in receipts):
        raise ValueError("GPU actors imported different application source")
    return models


def _preprocess_batches(options: argparse.Namespace) -> list:
    """Partition record IDs into ordinary Ray CPU tasks for GPU inference."""
    import ray

    prepare = ray.remote(num_cpus=1)(worker.preprocess_shard)
    batches = []
    for start in range(0, options.records, options.batch_size):
        stop = min(start + options.batch_size, options.records)
        record_ids = list(range(start, stop))
        batches.append(prepare.remote(record_ids))
    return batches


def _write_execution_report(options, models, results, expected_sources, boundaries) -> dict:
    """Persist workload evidence using the same artifacts a reader inspects."""
    import ray

    output = Path(options.output_path)
    rows = []
    for result in results:
        rows.extend(result["rows"])
    artifacts = save_results(output, rows, options.records)
    actors = ray.get([model.status.remote() for model in models])
    started, ready, prepared, inferred = boundaries
    timings = {
        "cluster_connect_and_model_ready": ready - started,
        "preprocessing_submission": prepared - ready,
        "preprocessing_and_inference_wall": inferred - ready,
        "preprocessing_task_sum": sum(result["preprocessing_seconds"] for result in results),
        "inference_actor_sum": sum(result["inference_seconds"] for result in results),
        "aggregation_and_artifacts": time.perf_counter() - inferred,
        "application": time.perf_counter() - started,
    }
    report = {
        **artifacts,
        "source_sha256": expected_sources,
        "crop_policy": worker.CROP_POLICY,
        "actors": actors,
        "ray_nodes": len({actor["node_id"] for actor in actors}),
        "preprocessors": [result["preprocessor"] for result in results],
        "timings_seconds": timings,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_hashes(output)
    print(f"Embedded {options.records} RGB images on {len(actors)} CUDA actor(s).")
    print(f"Open {output}/preview.png; vectors: embeddings.parquet and lance/embeddings.lance")
    print(json.dumps({"retrieval": artifacts["retrieval"], "timings_seconds": timings}))
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the plain Ray application submitted through the upstream Jobs CLI.

    Args:
        argv: Command-line arguments; None reads the process arguments.
    Returns:
        Zero after validated artifacts and provenance have been persisted.
    Raises:
        ValueError: Source, resources, options or output validation fails.
        RuntimeError: Ray or CUDA cannot execute the application.
        OSError: Source, model or artifact files cannot be accessed.
    """
    import ray

    options = _parse_arguments(argv)
    expected_sources = source_hashes()
    started = time.perf_counter()
    ray.init(address=application_gcs_address())
    try:
        models = _start_model_actors(options, expected_sources)
        ready = time.perf_counter()
        shards = _preprocess_batches(options)
        prepared = time.perf_counter()
        submissions = []
        for index, shard in enumerate(shards):
            model = models[index % options.actors]
            submissions.append(model.infer.remote(shard))
        results = ray.get(submissions)
        inferred = time.perf_counter()
        boundaries = (started, ready, prepared, inferred)
        _write_execution_report(options, models, results, expected_sources, boundaries)
        return 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

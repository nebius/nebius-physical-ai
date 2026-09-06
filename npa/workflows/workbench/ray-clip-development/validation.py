# Responsibility: Verify CLIP source provenance, recoverable checkpoints and persisted vector comparisons.
"""Independent completeness and source/output checks for the Ray CLIP recipe."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

SOURCE_HASH_FIELDS = {
    "application.py": "application_sha256",
    "worker.py": "source_sha256",
    "validation.py": "validation_sha256",
    "npa_lancedb_bdd100k_udfs.py": "udf_sha256",
}
MODEL_RUNTIME_FIELDS = (
    "model_revision", "model_files", "model_config_sha256", "udf_sha256",
    "precision", "gpu_capability", "python", "ray", "torch", "cuda",
    "transformers", "pyarrow", "lancedb",
)


def canonical_hash(value: object) -> str:
    """Fingerprint JSON data independently of dictionary insertion order.

    Args:
        value: JSON-serializable provenance or identity data.
    Returns:
        The SHA-256 hexadecimal digest of canonical JSON bytes.
    Raises:
        TypeError: The value cannot be serialized as JSON.
    """
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    encoded = serialized.encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    """Fingerprint a complete file without loading it all into memory.

    Args:
        path: File whose actual bytes must be verified.
    Returns:
        The file's SHA-256 hexadecimal digest.
    Raises:
        OSError: The file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish complete JSON metadata through an atomic file replacement.

    Args:
        path: Destination metadata file.
        value: JSON-serializable metadata.
    Returns:
        None.
    Raises:
        TypeError: Metadata cannot be serialized as JSON.
        OSError: Temporary or destination storage cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def partitions(records: int, batch_size: int) -> list[list[int]]:
    """Partition every record ID into ordered, bounded inference batches.

    Args:
        records: Positive number of records to process.
        batch_size: Maximum positive number of records per shard.
    Returns:
        Ordered batches covering IDs from zero through records minus one.
    Raises:
        ValueError: Either requested size is nonpositive.
    """
    if records < 1 or batch_size < 1:
        raise ValueError("records and batch_size must be positive")
    batches = []
    for start in range(0, records, batch_size):
        end = min(start + batch_size, records)
        batches.append(list(range(start, end)))
    return batches


def verify_ids(ids: list[int], count: int) -> None:
    """Reject missing, duplicated or unexpected output record IDs.

    Args:
        ids: Persisted record IDs in any order.
        count: Expected complete number of records.
    Returns:
        None when the IDs form the exact expected dataset.
    Raises:
        ValueError: The dataset has missing, duplicated or unexpected IDs.
    """
    if sorted(ids) != list(range(count)):
        raise ValueError("Output has missing, duplicate, or unexpected record IDs")


def checkpoint_identity(shard: dict, model_revision: str, execution_fingerprint: str) -> dict:
    """Bind a checkpoint to its inputs, source, model and execution environment.

    Args:
        shard: Preprocessed rows and imported worker source hash.
        model_revision: Immutable model snapshot revision.
        execution_fingerprint: Verified source, weights and runtime identity.
    Returns:
        Identity fields that must match before a checkpoint can be replayed.
    Raises:
        KeyError: The shard lacks required provenance fields.
    """
    return {
        "record_ids": [row["record_id"] for row in shard["rows"]],
        "input_hash": canonical_hash([row["input_sha256"] for row in shard["rows"]]),
        "processed_hash": canonical_hash([row["processed_sha256"] for row in shard["rows"]]),
        "source_sha256": shard["source_sha256"],
        "model_revision": model_revision,
        "execution_fingerprint": execution_fingerprint,
    }


def verify_execution(directory: Path, fingerprint: str) -> None:
    """Associate an output directory with exactly one execution identity.

    Args:
        directory: Existing driver-owned output directory.
        fingerprint: Expected source, model and runtime identity.
    Returns:
        None after the identity is verified or first recorded.
    Raises:
        ValueError: Existing output belongs to another or unknown execution.
        OSError: Output metadata cannot be read or written.
    """
    marker = directory / "execution.json"
    expected = {"execution_fingerprint": fingerprint}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("Output directory belongs to a different execution fingerprint")
        return
    if any(directory.iterdir()):
        raise ValueError("Nonempty output directory has no execution fingerprint")
    atomic_json(marker, expected)


def read_checkpoint(directory: Path, identity: dict) -> dict | None:
    """Read a committed checkpoint only after identity and byte verification.

    Args:
        directory: Driver-owned shard checkpoint directory.
        identity: Required source, input, model and runtime identity.
    Returns:
        The verified receipt, or None if no commit marker exists.
    Raises:
        ValueError: Stored identity or committed Parquet bytes differ.
        OSError: A committed checkpoint cannot be read.
    """
    marker = directory / "commit.json"
    if not marker.exists():
        return None
    receipt = json.loads(marker.read_text())
    if receipt["identity"] != identity:
        raise ValueError("Checkpoint identity differs from source/input/model")
    if file_hash(directory / "embeddings.parquet") != receipt["parquet_sha256"]:
        raise ValueError("Checkpoint data hash mismatch")
    return receipt


def verify_submitted_sources(report: dict, manifest: dict) -> None:
    """Match the driver and every model actor's imports to submitted source bytes.

    Args:
        report: Application report containing actor initialization provenance.
        manifest: Client-side filename-to-SHA-256 mapping.
    Returns:
        None when all required imported modules match the submitted package.
    Raises:
        ValueError: Required provenance is absent or a source hash differs.
    """
    actors = report.get("model_initializations")
    if not isinstance(actors, list) or not actors:
        raise ValueError("Application report has no model initialization provenance")
    for filename, field in SOURCE_HASH_FIELDS.items():
        expected = manifest.get(filename)
        if not expected or report.get(field) != expected:
            raise ValueError(f"Application imported a different {filename} source than submitted")
        if any(actor.get(field) != expected for actor in actors):
            raise ValueError(f"An actor imported a different {filename} source than submitted")


def _verify_fixed_comparison_inputs(baseline, changed, restored):
    """Ensure the crop edit is the only intended input change between Jobs."""
    fields = ("records", "input_hash", "model_revision", "application_sha256",
              "validation_sha256", "udf_sha256")
    for field in fields:
        if baseline[field] != changed[field] or baseline[field] != restored[field]:
            raise ValueError(f"Comparison changed fixed input {field}")


def _actor_model_runtime(actor, report):
    """Require each actor to identify its actual weights and dependency runtime."""
    if any(field not in actor for field in MODEL_RUNTIME_FIELDS):
        raise ValueError("Model initialization lacks actual model/runtime provenance")
    if actor["model_revision"] != report["model_revision"]:
        raise ValueError("Actor model revision differs from the application report")
    return {field: actor[field] for field in MODEL_RUNTIME_FIELDS}


def _verify_matching_runtime(reference, observed):
    """Identify runtime drift explicitly instead of comparing only revision labels."""
    if reference is None:
        return observed
    if observed == reference:
        return reference
    changed_fields = sorted(field for field in MODEL_RUNTIME_FIELDS if observed[field] != reference[field])
    raise ValueError("Comparison changed actual model/runtime: " + ", ".join(changed_fields))


def _verify_all_actor_runtimes(reports):
    """Check every initial and replacement actor across all compared Jobs."""
    reference = None
    for report in reports:
        manifest = {name: report.get(field) for name, field in SOURCE_HASH_FIELDS.items()}
        verify_submitted_sources(report, manifest)
        for actor in report["model_initializations"]:
            observed = _actor_model_runtime(actor, report)
            reference = _verify_matching_runtime(reference, observed)
    return reference


def _verify_crop_restoration(baseline, changed, restored):
    """Require changed crop source and inputs to return to their baseline bytes."""
    if baseline["source_sha256"] != restored["source_sha256"]:
        raise ValueError("Restored worker source does not match baseline")
    if baseline["source_sha256"] == changed["source_sha256"]:
        raise ValueError("Changed worker source did not change")
    if baseline["processed_hash"] == changed["processed_hash"]:
        raise ValueError("Changed preprocessing did not change model inputs")
    if baseline["processed_hash"] != restored["processed_hash"]:
        raise ValueError("Restoration did not restore model inputs")


def _compare_mean_embeddings(baseline, changed, restored):
    """Measure the expected crop effect and its numerical restoration."""
    import numpy

    original = numpy.asarray(baseline["mean_embedding"])
    edited = numpy.asarray(changed["mean_embedding"])
    repeated = numpy.asarray(restored["mean_embedding"])
    difference = float(numpy.linalg.norm(original - edited))
    restoration_error = float(numpy.max(numpy.abs(original - repeated)))
    if difference <= 0.01:
        raise ValueError("CLIP output did not meaningfully change with crop revision")
    if restoration_error > 1e-5:
        raise ValueError("Restored embeddings exceed floating-point tolerance")
    return {"changed_mean_embedding_l2": difference, "restored_max_absolute_error": restoration_error}


def compare_reports(baseline: dict, changed: dict, restored: dict) -> dict:
    """Verify a source-edit experiment changed real inference and then restored it.

    Args:
        baseline: Successful report before the crop edit.
        changed: Successful report with the intended crop edit.
        restored: Successful report after restoring the original source.
    Returns:
        Mean-vector differences and the fixed model/runtime fingerprint.
    Raises:
        ValueError: Inputs, imported code, model/runtime or output checks fail.
        KeyError: A required report field is absent.
    """
    _verify_fixed_comparison_inputs(baseline, changed, restored)
    reference = _verify_all_actor_runtimes((baseline, changed, restored))
    _verify_crop_restoration(baseline, changed, restored)
    comparison = _compare_mean_embeddings(baseline, changed, restored)
    comparison["fixed_model_runtime_sha256"] = canonical_hash(reference)
    return comparison


def _read_aligned_vector_tables(baseline_path, current_path):
    """Require full persisted vectors to refer to the same rendered input records."""
    import pyarrow.parquet as parquet

    baseline = parquet.read_table(baseline_path).sort_by("record_id")
    current = parquet.read_table(current_path).sort_by("record_id")
    for column in ("record_id", "input_sha256"):
        if baseline[column].to_pylist() != current[column].to_pylist():
            raise ValueError("Baseline comparison changed record IDs or rendered inputs")
    return baseline, current


def _compare_vector_matrices(baseline, current, changed):
    """Apply change and restoration thresholds to every persisted vector."""
    import numpy

    differences = numpy.linalg.norm(baseline - current, axis=1)
    changed_fraction = float(numpy.mean(differences > 0.01))
    maximum_error = float(numpy.max(numpy.abs(baseline - current)))
    if changed and changed_fraction < 0.99:
        raise ValueError("Changed crop did not change at least 99% of persisted vectors")
    if not changed and not numpy.allclose(baseline, current, rtol=0, atol=1e-5):
        raise ValueError("Restored persisted vectors exceed numerical tolerance")
    mode = "restored"
    if changed:
        mode = "changed"
    return {"compared_vectors": len(baseline), "fraction_l2_change_above_0_01": changed_fraction,
            "max_absolute_error": maximum_error, "mode": mode}


def compare_vectors(baseline_path: Path, current_path: Path, *, changed: bool) -> dict:
    """Compare every persisted vector rather than a mean or selected sample.

    Args:
        baseline_path: Baseline embeddings Parquet file.
        current_path: Changed or restored embeddings Parquet file.
        changed: Whether this run should differ meaningfully from the baseline.
    Returns:
        Complete vector counts, change fraction and maximum absolute error.
    Raises:
        ValueError: Input identity or expected vector differences do not match.
        OSError: A persisted embeddings file cannot be read.
    """
    import numpy

    baseline, current = _read_aligned_vector_tables(baseline_path, current_path)
    original = numpy.asarray(baseline["vector"].to_pylist(), dtype=numpy.float32)
    repeated = numpy.asarray(current["vector"].to_pylist(), dtype=numpy.float32)
    return _compare_vector_matrices(original, repeated, changed)

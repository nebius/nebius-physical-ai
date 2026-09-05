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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
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
    if records < 1 or batch_size < 1:
        raise ValueError("records and batch_size must be positive")
    return [list(range(start, min(start + batch_size, records))) for start in range(0, records, batch_size)]


def verify_ids(ids: list[int], count: int) -> None:
    if sorted(ids) != list(range(count)):
        raise ValueError("Output has missing, duplicate, or unexpected record IDs")


def checkpoint_identity(shard: dict, model_revision: str, execution_fingerprint: str) -> dict:
    return {
        "record_ids": [row["record_id"] for row in shard["rows"]],
        "input_hash": canonical_hash([row["input_sha256"] for row in shard["rows"]]),
        "processed_hash": canonical_hash([row["processed_sha256"] for row in shard["rows"]]),
        "source_sha256": shard["source_sha256"],
        "model_revision": model_revision,
        "execution_fingerprint": execution_fingerprint,
    }


def verify_execution(directory: Path, fingerprint: str) -> None:
    marker = directory / "execution.json"
    expected = {"execution_fingerprint": fingerprint}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("Output directory belongs to a different execution fingerprint")
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty output directory has no execution fingerprint")
        atomic_json(marker, expected)


def read_checkpoint(directory: Path, identity: dict) -> dict | None:
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
    """Match the driver and every actor's imports to the client's source bytes."""
    actors = report.get("model_initializations")
    if not isinstance(actors, list) or not actors:
        raise ValueError("Application report has no model initialization provenance")
    for filename, field in SOURCE_HASH_FIELDS.items():
        expected = manifest.get(filename)
        if not expected or report.get(field) != expected:
            raise ValueError(f"Application imported a different {filename} source than submitted")
        if any(actor.get(field) != expected for actor in actors):
            raise ValueError(f"An actor imported a different {filename} source than submitted")


def compare_reports(baseline: dict, changed: dict, restored: dict) -> dict:
    import numpy as np

    for key in ("records", "input_hash", "model_revision", "application_sha256", "validation_sha256", "udf_sha256"):
        if not (baseline[key] == changed[key] == restored[key]):
            raise ValueError(f"Comparison changed fixed input {key}")
    reference = None
    for report in (baseline, changed, restored):
        verify_submitted_sources(report, {name: report.get(field) for name, field in SOURCE_HASH_FIELDS.items()})
        for actor in report["model_initializations"]:
            if any(field not in actor for field in MODEL_RUNTIME_FIELDS):
                raise ValueError("Model initialization lacks actual model/runtime provenance")
            observed = {field: actor[field] for field in MODEL_RUNTIME_FIELDS}
            if actor["model_revision"] != report["model_revision"]:
                raise ValueError("Actor model revision differs from the application report")
            if reference is None:
                reference = observed
            elif observed != reference:
                changed_fields = sorted(field for field in MODEL_RUNTIME_FIELDS if observed[field] != reference[field])
                raise ValueError("Comparison changed actual model/runtime: " + ", ".join(changed_fields))
    if baseline["source_sha256"] != restored["source_sha256"]:
        raise ValueError("Restored worker source does not match baseline")
    if baseline["source_sha256"] == changed["source_sha256"]:
        raise ValueError("Changed worker source did not change")
    if baseline["processed_hash"] == changed["processed_hash"]:
        raise ValueError("Changed preprocessing did not change model inputs")
    if baseline["processed_hash"] != restored["processed_hash"]:
        raise ValueError("Restoration did not restore model inputs")
    left, right, again = [np.asarray(report["mean_embedding"]) for report in (baseline, changed, restored)]
    change = float(np.linalg.norm(left - right))
    restore = float(np.max(np.abs(left - again)))
    if change <= 0.01:
        raise ValueError("CLIP output did not meaningfully change with crop revision")
    if restore > 1e-5:
        raise ValueError("Restored embeddings exceed floating-point tolerance")
    return {"changed_mean_embedding_l2": change, "restored_max_absolute_error": restore,
            "fixed_model_runtime_sha256": canonical_hash(reference)}


def compare_vectors(baseline_path: Path, current_path: Path, *, changed: bool) -> dict:
    """Compare every persisted vector, not merely a mean or sample."""
    import numpy as np
    import pyarrow.parquet as pq

    baseline, current = [pq.read_table(path).sort_by("record_id") for path in (baseline_path, current_path)]
    for column in ("record_id", "input_sha256"):
        if baseline[column].to_pylist() != current[column].to_pylist():
            raise ValueError("Baseline comparison changed record IDs or rendered inputs")
    left, right = [np.asarray(table["vector"].to_pylist(), dtype=np.float32) for table in (baseline, current)]
    differences = np.linalg.norm(left - right, axis=1)
    changed_fraction = float(np.mean(differences > 0.01))
    maximum = float(np.max(np.abs(left - right)))
    if changed and changed_fraction < 0.99:
        raise ValueError("Changed crop did not change at least 99% of persisted vectors")
    if not changed and not np.allclose(left, right, rtol=0, atol=1e-5):
        raise ValueError("Restored persisted vectors exceed numerical tolerance")
    return {"compared_vectors": len(left), "fraction_l2_change_above_0_01": changed_fraction,
            "max_absolute_error": maximum, "mode": "changed" if changed else "restored"}

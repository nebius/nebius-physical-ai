"""Backfill BDD100K-derived columns into LanceDB tables."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

try:
    from .bdd100k_import import DEFAULT_LANCE_URI, DEFAULT_TABLE, validate_lance_uri, validate_table
    from .bdd100k_udfs import BDD100K_UDFS, UDFSpec, udf_is_duplicate
except ImportError:  # pragma: no cover - used by the copied Docker module.
    from npa_lancedb_bdd100k_import import DEFAULT_LANCE_URI, DEFAULT_TABLE, validate_lance_uri, validate_table
    from npa_lancedb_bdd100k_udfs import BDD100K_UDFS, UDFSpec, udf_is_duplicate


DEFAULT_BATCH_SIZE = 256
DEFAULT_GPU_BATCH_SIZE = 32
DEFAULT_GPU_DEVICE = "cuda:0"
DEFAULT_DHASH_HAMMING_THRESHOLD = 5


class BackfillError(RuntimeError):
    """Base class for LanceDB backfill failures."""


class BackfillValidationError(BackfillError, ValueError):
    """Raised when a backfill request is invalid."""


class UnknownUDFError(BackfillValidationError):
    """Raised when the requested UDF is not registered."""


class MissingDependencyError(BackfillError):
    """Raised when a UDF dependency has not been backfilled."""


class BackfillTableNotFoundError(BackfillError, FileNotFoundError):
    """Raised when the requested LanceDB table does not exist."""


class BackfillWriteError(BackfillError):
    """Raised when LanceDB cannot read or write backfill data."""


class GPUOOMAtMinimumBatchError(BackfillError):
    """Raised when a GPU UDF cannot run even at batch size 1."""


@dataclass(frozen=True)
class BackfillResult:
    """Result returned by API, CLI, and SDK backfill calls."""

    table: str
    lance_uri: str
    rows_updated: int
    rows_skipped: int
    table_version_before: int | None
    table_version_after: int | None
    udf: str
    output_column: str
    column_added: bool
    duration_ms: int
    manifest_sha256: str
    gpu_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        return asdict(self)


def backfill_column(
    table_uri: str | None = None,
    table_name: str | None = None,
    udf_name: str | None = None,
    *,
    lance_uri: str | None = None,
    table: str | None = None,
    udf: str | None = None,
    batch_size: int | None = None,
    force: bool = False,
    force_recompute: bool | None = None,
    dhash_hamming_threshold: int = DEFAULT_DHASH_HAMMING_THRESHOLD,
    device: str = DEFAULT_GPU_DEVICE,
    precision: str | None = None,
) -> BackfillResult:
    """Backfill one BDD100K UDF output column into a LanceDB table."""
    start = time.perf_counter()
    resolved_uri = validate_lance_uri(lance_uri if lance_uri is not None else table_uri or DEFAULT_LANCE_URI)
    resolved_table = validate_table(table if table is not None else table_name or DEFAULT_TABLE)
    resolved_udf = _resolve_udf(udf if udf is not None else udf_name)
    resolved_batch_size = _resolve_batch_size(batch_size, resolved_udf)
    resolved_force = force if force_recompute is None else force_recompute
    resolved_device = _resolve_device(device, resolved_udf)
    resolved_precision = _resolve_precision(precision)
    if resolved_batch_size < 1:
        raise BackfillValidationError("batch_size must be positive")
    if dhash_hamming_threshold < 0 or dhash_hamming_threshold > 64:
        raise BackfillValidationError("dhash_hamming_threshold must be between 0 and 64")

    db = _connect_lancedb(resolved_uri)
    table_obj = _open_table(db, resolved_table)
    table_version_before = _table_version(table_obj)
    schema = _schema(table_obj)
    _validate_dependencies(schema, resolved_udf)
    _validate_input_columns(schema, resolved_udf)

    column_added = False
    if resolved_udf.output_column not in schema.names:
        _add_output_column(table_obj, resolved_udf)
        column_added = True

    if resolved_udf.name == "is_duplicate":
        rows_updated, rows_skipped = _backfill_is_duplicate(
            table_obj,
            resolved_udf,
            force=resolved_force,
            dhash_hamming_threshold=dhash_hamming_threshold,
        )
    elif resolved_udf.gpu:
        rows_updated, rows_skipped = _run_gpu_udf(
            table_obj,
            resolved_udf,
            batch_size=resolved_batch_size,
            force=resolved_force,
            device=resolved_device,
            precision=resolved_precision,
        )
    else:
        rows_updated, rows_skipped = _backfill_batches(table_obj, resolved_udf, batch_size=resolved_batch_size, force=resolved_force)

    table_version_after = _table_version(table_obj)
    manifest_sha256 = _manifest_sha256(table_obj, resolved_udf.output_column)
    return BackfillResult(
        table=resolved_table,
        lance_uri=resolved_uri,
        rows_updated=rows_updated,
        rows_skipped=rows_skipped,
        table_version_before=table_version_before,
        table_version_after=table_version_after,
        udf=resolved_udf.name,
        output_column=resolved_udf.output_column,
        column_added=column_added,
        duration_ms=int((time.perf_counter() - start) * 1000),
        manifest_sha256=manifest_sha256,
        gpu_used=resolved_udf.gpu,
    )


def _resolve_udf(name: str | None) -> UDFSpec:
    value = (name or "").strip()
    if not value:
        raise UnknownUDFError("udf is required")
    spec = BDD100K_UDFS.get(value)
    if spec is None:
        valid = ", ".join(sorted(BDD100K_UDFS))
        raise UnknownUDFError(f"unknown UDF {value!r}; expected one of {valid}")
    return spec


def _resolve_batch_size(batch_size: int | None, spec: UDFSpec) -> int:
    if batch_size is not None:
        return batch_size
    return DEFAULT_GPU_BATCH_SIZE if spec.gpu else DEFAULT_BATCH_SIZE


def _resolve_device(device: str, spec: UDFSpec) -> str:
    value = (device or "").strip()
    if spec.gpu and not value:
        raise BackfillValidationError("device is required for GPU UDFs")
    return value


def _resolve_precision(precision: str | None) -> str | None:
    value = (precision or "").strip().lower()
    if not value:
        return None
    if value not in {"float16", "float32"}:
        raise BackfillValidationError("precision must be 'float16' or 'float32'")
    return value


def _connect_lancedb(lance_uri: str):
    try:
        import lancedb
    except ImportError as exc:
        raise BackfillWriteError("Backfill requires the lancedb package") from exc
    try:
        return lancedb.connect(lance_uri)
    except Exception as exc:
        raise BackfillWriteError(f"failed to connect to LanceDB URI {lance_uri}: {exc}") from exc


def _open_table(db: Any, table_name: str) -> Any:
    try:
        if table_name not in _list_tables(db):
            raise BackfillTableNotFoundError(f"LanceDB table not found: {table_name}")
        return db.open_table(table_name)
    except BackfillTableNotFoundError:
        raise
    except Exception as exc:
        raise BackfillWriteError(f"failed to open LanceDB table {table_name}: {exc}") from exc


def _list_tables(db: Any) -> list[str]:
    table_names = getattr(db, "table_names", None)
    if callable(table_names):
        return _normalize_table_names(table_names())
    return _normalize_table_names(db.list_tables())


def _normalize_table_names(values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, tuple | list) and value:
            names.append(str(value[0]))
        elif hasattr(value, "name"):
            names.append(str(value.name))
        else:
            names.append(str(value))
    return names


def _schema(table_obj: Any) -> pa.Schema:
    schema = getattr(table_obj, "schema", None)
    if callable(schema):
        schema = schema()
    if not isinstance(schema, pa.Schema):
        raise BackfillWriteError("LanceDB table schema is not a PyArrow schema")
    return schema


def _table_version(table_obj: Any) -> int | None:
    value = getattr(table_obj, "version", None)
    if value is None:
        return None
    try:
        return int(value() if callable(value) else value)
    except (TypeError, ValueError):
        return None


def _validate_input_columns(schema: pa.Schema, spec: UDFSpec) -> None:
    missing = [name for name in spec.input_columns if name not in schema.names]
    if missing:
        joined = ", ".join(missing)
        raise BackfillValidationError(f"table is missing required input column(s) for {spec.name}: {joined}")


def _validate_dependencies(schema: pa.Schema, spec: UDFSpec) -> None:
    missing = [name for name in spec.dependencies if name not in schema.names]
    if missing:
        joined = ", ".join(missing)
        raise MissingDependencyError(f"{spec.name} requires backfilling dependency column(s) first: {joined}")


def _add_output_column(table_obj: Any, spec: UDFSpec) -> None:
    try:
        table_obj.add_columns([pa.field(spec.output_column, spec.output_type)])
    except Exception as exc:
        raise BackfillWriteError(f"failed to add output column {spec.output_column}: {exc}") from exc


def _backfill_batches(table_obj: Any, spec: UDFSpec, *, batch_size: int, force: bool) -> tuple[int, int]:
    rows_updated = 0
    rows_skipped = 0
    columns = _select_columns(spec)
    try:
        batches = table_obj.search().select(columns).to_batches(batch_size=batch_size)
        for batch in batches:
            output = spec.function(batch)
            ids, values, skipped = _updates_for_batch(batch, spec.output_column, output, force=force)
            rows_skipped += skipped
            if ids:
                _write_updates(table_obj, ids, values, spec)
                rows_updated += len(ids)
    except BackfillError:
        raise
    except Exception as exc:
        raise BackfillWriteError(f"failed to backfill {spec.name}: {exc}") from exc
    return rows_updated, rows_skipped


def _run_gpu_udf(
    table_obj: Any,
    spec: UDFSpec,
    *,
    batch_size: int,
    force: bool,
    device: str,
    precision: str | None,
) -> tuple[int, int]:
    rows_updated = 0
    rows_skipped = 0
    columns = _select_columns(spec)
    try:
        batches = table_obj.search().select(columns).to_batches(batch_size=batch_size)
        for batch in batches:
            updated, skipped = _backfill_gpu_batch(
                table_obj,
                spec,
                batch,
                batch_size=batch_size,
                force=force,
                device=device,
                precision=precision,
            )
            rows_updated += updated
            rows_skipped += skipped
    except BackfillError:
        raise
    except Exception as exc:
        raise BackfillWriteError(f"failed to backfill GPU UDF {spec.name}: {exc}") from exc
    return rows_updated, rows_skipped


def _backfill_gpu_batch(
    table_obj: Any,
    spec: UDFSpec,
    batch: pa.RecordBatch,
    *,
    batch_size: int,
    force: bool,
    device: str,
    precision: str | None,
) -> tuple[int, int]:
    rows_updated = 0
    rows_skipped = 0
    offset = 0
    current_batch_size = max(1, min(batch_size, batch.num_rows or 1))
    while offset < batch.num_rows:
        chunk_size = min(current_batch_size, batch.num_rows - offset)
        chunk = batch.slice(offset, chunk_size)
        try:
            output = spec.function(chunk, device=device, precision=precision)
        except Exception as exc:
            if not _is_gpu_oom(exc):
                raise
            _clear_gpu_cache()
            if chunk_size <= 1:
                raise GPUOOMAtMinimumBatchError(
                    f"HALT_GPU_OOM_AT_MINIMUM_BATCH: {spec.name} failed on {device} with batch_size=1"
                ) from exc
            current_batch_size = max(1, chunk_size // 2)
            continue
        ids, values, skipped = _updates_for_batch(chunk, spec.output_column, output, force=force)
        rows_skipped += skipped
        if ids:
            _write_updates(table_obj, ids, values, spec)
            rows_updated += len(ids)
        offset += chunk_size
    return rows_updated, rows_skipped


def _backfill_is_duplicate(
    table_obj: Any,
    spec: UDFSpec,
    *,
    force: bool,
    dhash_hamming_threshold: int,
) -> tuple[int, int]:
    try:
        columns = _select_columns(spec)
        arrow = table_obj.search().select(columns).to_arrow()
        if arrow.column("dhash").null_count:
            raise MissingDependencyError("is_duplicate requires non-null dhash values; run dhash backfill first")
        sorted_table = arrow.sort_by([("image_id", "ascending")])
        batches = sorted_table.to_batches(max_chunksize=max(1, len(sorted_table)))
        batch = batches[0] if batches else pa.record_batch([], schema=sorted_table.schema)
        output = udf_is_duplicate(batch, dhash_column="dhash", hamming_threshold=dhash_hamming_threshold)
        ids, values, skipped = _updates_for_batch(batch, spec.output_column, output, force=force)
        if ids:
            _write_updates(table_obj, ids, values, spec)
        return len(ids), skipped
    except BackfillError:
        raise
    except Exception as exc:
        raise BackfillWriteError(f"failed to backfill {spec.name}: {exc}") from exc


def _select_columns(spec: UDFSpec) -> list[str]:
    names = ["image_id", *spec.input_columns, spec.output_column]
    selected: list[str] = []
    for name in names:
        if name not in selected:
            selected.append(name)
    return selected


def _updates_for_batch(
    batch: pa.RecordBatch,
    output_column: str,
    output: pa.Array,
    *,
    force: bool,
) -> tuple[list[str], list[Any], int]:
    image_ids = batch.column("image_id").to_pylist()
    existing = batch.column(output_column)
    computed = output.to_pylist()
    update_ids: list[str] = []
    update_values: list[Any] = []
    skipped = 0
    for index, image_id in enumerate(image_ids):
        should_update = force or not existing[index].is_valid
        if should_update:
            update_ids.append(str(image_id))
            update_values.append(computed[index])
        else:
            skipped += 1
    return update_ids, update_values, skipped


def _write_updates(table_obj: Any, image_ids: list[str], values: list[Any], spec: UDFSpec) -> None:
    update_table = pa.table(
        {
            "image_id": pa.array(image_ids, type=pa.string()),
            spec.output_column: pa.array(values, type=spec.output_type),
        }
    )
    try:
        table_obj.merge_insert("image_id").when_matched_update_all().execute(update_table)
    except Exception as exc:
        raise BackfillWriteError(f"failed to write {spec.output_column} updates: {exc}") from exc


def _is_gpu_oom(exc: Exception) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


def _clear_gpu_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def _manifest_sha256(table_obj: Any, output_column: str) -> str:
    try:
        rows = table_obj.search().select(["image_id", output_column]).to_arrow().to_pylist()
    except Exception as exc:
        raise BackfillWriteError(f"failed to read {output_column} manifest values: {exc}") from exc
    digest = hashlib.sha256()
    digest.update(output_column.encode("utf-8"))
    digest.update(b"\n")
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        digest.update(str(row["image_id"]).encode("utf-8"))
        digest.update(b"\t")
        digest.update(json.dumps(row.get(output_column), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "BackfillError",
    "BackfillResult",
    "BackfillTableNotFoundError",
    "BackfillValidationError",
    "BackfillWriteError",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DHASH_HAMMING_THRESHOLD",
    "DEFAULT_GPU_BATCH_SIZE",
    "DEFAULT_GPU_DEVICE",
    "GPUOOMAtMinimumBatchError",
    "MissingDependencyError",
    "UnknownUDFError",
    "backfill_column",
]

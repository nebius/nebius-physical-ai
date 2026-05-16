"""BDD100K ingestion for the LanceDB workbench tool."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pyarrow as pa
from PIL import Image, ImageDraw

DEFAULT_TABLE = "bdd100k"
DEFAULT_LANCE_URI = "s3://YOUR_S3_BUCKET/lancedb/bdd100k/"
DEFAULT_SPLITS = ("train", "val")
VALID_SPLITS = ("train", "val", "test")
BATCH_SIZE = 200

BDD_WEATHER = (
    "clear",
    "partly cloudy",
    "overcast",
    "rainy",
    "snowy",
    "foggy",
    "undefined",
)
BDD_SCENE = (
    "city street",
    "highway",
    "residential",
    "parking lot",
    "tunnel",
    "gas stations",
    "undefined",
)
BDD_TIMEOFDAY = ("daytime", "night", "dawn/dusk", "undefined")
BDD_DETECTION_CATEGORIES = (
    "traffic light",
    "traffic sign",
    "car",
    "bus",
    "truck",
    "person",
    "rider",
    "bike",
    "motor",
    "train",
)
LABEL_FILE_CANDIDATES = (
    "det_{split}.json",
    "bdd100k_labels_images_{split}.json",
)


class BDD100KImportError(RuntimeError):
    """Base class for BDD100K import failures."""


class BDD100KValidationError(BDD100KImportError, ValueError):
    """Raised when request fields are invalid."""


class BDD100KSourceError(BDD100KImportError, FileNotFoundError):
    """Raised when a BDD100K source bundle cannot be read."""


class BDD100KWriteError(BDD100KImportError):
    """Raised when LanceDB cannot write the generated table."""


@dataclass(frozen=True)
class BDD100KImportResult:
    """Result returned by API, CLI, and SDK BDD100K imports."""

    table: str
    lance_uri: str
    table_uri: str
    rows_per_split: dict[str, int]
    total_rows: int
    table_version_before: int | None
    table_version_after: int | None
    table_version: int | None
    manifest_sha256: str
    row_checksum_sha256: str
    splits: list[str]
    synthetic: int | None
    synthetic_seed: int | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""
        return asdict(self)


@dataclass(frozen=True)
class _BatchPayload:
    rows: list[dict[str, Any]]
    batch: pa.RecordBatch


@dataclass(frozen=True)
class _S3Location:
    bucket: str
    prefix: str


def bdd100k_schema() -> pa.Schema:
    """Return the BDD100K LanceDB PyArrow schema."""
    return pa.schema(
        [
            pa.field("image_id", pa.string()),
            pa.field("image_bytes", pa.large_binary()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("weather", pa.string()),
            pa.field("scene", pa.string()),
            pa.field("timeofday", pa.string()),
            pa.field("timestamp", pa.timestamp("ms")),
            pa.field("ann_categories", pa.list_(pa.string())),
            pa.field("ann_bboxes", pa.list_(pa.list_(pa.float32()))),
            pa.field("ann_occluded", pa.list_(pa.bool_())),
            pa.field("split", pa.string()),
        ]
    )


def schema_summary() -> dict[str, str]:
    """Return field names mapped to PyArrow type strings."""
    return {field.name: str(field.type) for field in bdd100k_schema()}


def import_bdd100k(
    *,
    source: str = "",
    table: str = DEFAULT_TABLE,
    lance_uri: str = DEFAULT_LANCE_URI,
    synthetic: int | None = None,
    synthetic_seed: int | None = None,
    splits: Iterable[str] | None = None,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    write: bool = True,
) -> BDD100KImportResult:
    """Import BDD100K detection rows into a LanceDB table."""
    split_values = validate_splits(splits)
    table_name = validate_table(table)
    target_uri = validate_lance_uri(lance_uri)
    if limit is not None and limit < 1:
        raise BDD100KValidationError("limit must be a positive integer when provided")
    if synthetic is not None and synthetic < 1:
        raise BDD100KValidationError("synthetic must be a positive integer when provided")
    if synthetic is None and not source.strip():
        raise BDD100KSourceError("source is required when synthetic is not set")
    if batch_size < 1:
        raise BDD100KValidationError("batch_size must be positive")

    batches = (
        synthetic_batches(
            synthetic,
            splits=split_values,
            seed=synthetic_seed,
            batch_size=batch_size,
        )
        if synthetic is not None
        else source_batches(
            source,
            splits=split_values,
            limit=limit,
            batch_size=batch_size,
        )
    )

    rows_per_split = {split: 0 for split in split_values}
    checksum_entries: list[tuple[str, str, str]] = []
    table_version_before: int | None = None
    table_version_after: int | None = None
    table_obj: Any = None
    db: Any = None

    if write:
        db = _connect_lancedb(target_uri)

    for payload in batches:
        for row in payload.rows:
            split = str(row["split"])
            rows_per_split[split] = rows_per_split.get(split, 0) + 1
            checksum_entries.append(
                (
                    str(row["image_id"]),
                    split,
                    hashlib.sha256(bytes(row["image_bytes"])).hexdigest(),
                )
            )
        if write:
            arrow_table = pa.Table.from_batches([payload.batch], schema=bdd100k_schema())
            if table_obj is None:
                table_obj, table_version_before = _open_or_create_table(
                    db,
                    table_name,
                    arrow_table,
                )
            else:
                table_obj.add(arrow_table)

    total_rows = sum(rows_per_split.values())
    if total_rows == 0:
        raise BDD100KSourceError("BDD100K import produced no rows")
    if write:
        if table_obj is None:
            raise BDD100KWriteError("LanceDB table was not created")
        table_version_after = _table_version(table_obj)

    row_checksum = manifest_checksum(checksum_entries)
    return BDD100KImportResult(
        table=table_name,
        lance_uri=target_uri,
        table_uri=f"{target_uri.rstrip('/')}/{table_name}.lance",
        rows_per_split=rows_per_split,
        total_rows=total_rows,
        table_version_before=table_version_before,
        table_version_after=table_version_after,
        table_version=table_version_after,
        manifest_sha256=row_checksum,
        row_checksum_sha256=row_checksum,
        splits=split_values,
        synthetic=synthetic,
        synthetic_seed=synthetic_seed,
        source="" if synthetic is not None else source,
    )


def validate_splits(splits: Iterable[str] | None) -> list[str]:
    """Validate and normalize split names."""
    raw = list(splits) if splits is not None else list(DEFAULT_SPLITS)
    if not raw:
        raw = list(DEFAULT_SPLITS)
    normalized: list[str] = []
    for split in raw:
        value = str(split).strip().lower()
        if value not in VALID_SPLITS:
            valid = ", ".join(VALID_SPLITS)
            raise BDD100KValidationError(f"invalid split {split!r}; expected one of {valid}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def validate_table(table: str) -> str:
    """Validate a LanceDB table name enough for the wrapper surface."""
    value = table.strip()
    if not value:
        raise BDD100KValidationError("table is required")
    if not (value[0].isalpha() or value[0] == "_"):
        raise BDD100KValidationError("table must start with a letter or underscore")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if any(char not in allowed for char in value):
        raise BDD100KValidationError("table contains unsupported characters")
    return value


def validate_lance_uri(lance_uri: str) -> str:
    """Validate the LanceDB target URI."""
    value = lance_uri.strip()
    if not value:
        raise BDD100KValidationError("lance_uri is required")
    if value.startswith("s3://"):
        parsed = urlparse(value)
        if not parsed.netloc:
            raise BDD100KValidationError("lance_uri S3 URI must include a bucket")
        return value
    if Path(value).is_absolute() or value.startswith("."):
        return value
    return value


def synthetic_batches(
    total_rows: int,
    *,
    splits: Iterable[str] | None = None,
    seed: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> Iterable[_BatchPayload]:
    """Yield deterministic synthetic BDD100K-shaped batches."""
    split_values = validate_splits(splits)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for split, count in _distribute_rows(total_rows, split_values).items():
        for index in range(count):
            rows.append(_synthetic_row(split=split, index=index, rng=rng))
            if len(rows) >= batch_size:
                yield _batch_from_rows(rows)
                rows = []
    if rows:
        yield _batch_from_rows(rows)


def source_batches(
    source: str,
    *,
    splits: Iterable[str] | None = None,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> Iterable[_BatchPayload]:
    """Yield batches from a local or S3 BDD100K detection bundle."""
    split_values = validate_splits(splits)
    source_value = source.strip()
    if not source_value:
        raise BDD100KSourceError("source is required")
    if source_value.startswith(("http://", "https://")):
        raise BDD100KSourceError("HTTP BDD100K bundles are not supported yet; use a local path or s3:// URI")
    if source_value.startswith("s3://"):
        yield from _s3_source_batches(source_value, splits=split_values, limit=limit, batch_size=batch_size)
        return
    yield from _local_source_batches(source_value, splits=split_values, limit=limit, batch_size=batch_size)


def manifest_checksum(entries: Iterable[tuple[str, str, str]]) -> str:
    """Hash a sorted image-id checksum chain."""
    digest = hashlib.sha256()
    for image_id, split, image_hash in sorted(entries):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(split.encode("utf-8"))
        digest.update(b"\t")
        digest.update(image_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _batch_from_rows(rows: list[dict[str, Any]]) -> _BatchPayload:
    table = pa.Table.from_pylist(rows, schema=bdd100k_schema())
    batches = table.to_batches(max_chunksize=len(rows))
    return _BatchPayload(rows=list(rows), batch=batches[0])


def _distribute_rows(total_rows: int, splits: list[str]) -> dict[str, int]:
    weights = {"train": 70, "val": 20, "test": 10}
    total_weight = sum(weights[split] for split in splits)
    exact = {split: total_rows * weights[split] / total_weight for split in splits}
    counts = {split: int(exact[split]) for split in splits}
    remainder = total_rows - sum(counts.values())
    for split in sorted(splits, key=lambda value: exact[value] - counts[value], reverse=True)[:remainder]:
        counts[split] += 1
    return counts


def _synthetic_row(*, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    width = 720
    height = 480
    image = Image.new(
        "RGB",
        (width, height),
        (rng.randrange(32, 224), rng.randrange(32, 224), rng.randrange(32, 224)),
    )
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x1 = rng.randrange(0, width - 1)
        y1 = rng.randrange(0, height - 1)
        x2 = min(width, x1 + rng.randrange(8, 160))
        y2 = min(height, y1 + rng.randrange(8, 120))
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
            width=2,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    box_count = rng.randrange(1, 6)
    categories: list[str] = []
    boxes: list[list[float]] = []
    occluded: list[bool] = []
    for _ in range(box_count):
        x1 = rng.uniform(0, width * 0.85)
        y1 = rng.uniform(0, height * 0.85)
        x2 = rng.uniform(x1 + 1.0, width)
        y2 = rng.uniform(y1 + 1.0, height)
        categories.append(rng.choice(BDD_DETECTION_CATEGORIES))
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        occluded.append(bool(rng.randrange(0, 2)))

    return {
        "image_id": f"{split}-{index:06d}.jpg",
        "image_bytes": buffer.getvalue(),
        "width": width,
        "height": height,
        "weather": rng.choice(BDD_WEATHER),
        "scene": rng.choice(BDD_SCENE),
        "timeofday": rng.choice(BDD_TIMEOFDAY),
        "timestamp": None,
        "ann_categories": categories,
        "ann_bboxes": boxes,
        "ann_occluded": occluded,
        "split": split,
    }


def _local_source_batches(
    source: str,
    *,
    splits: list[str],
    limit: int | None,
    batch_size: int,
) -> Iterable[_BatchPayload]:
    root = Path(source)
    if not root.exists():
        raise BDD100KSourceError(f"source path does not exist: {source}")
    if not root.is_dir():
        raise BDD100KSourceError(f"source must be a BDD100K directory: {source}")
    image_index = _local_image_index(root)
    rows: list[dict[str, Any]] = []
    for split in splits:
        labels_path = _find_local_label_file(root, split)
        labels = _load_json_array(labels_path.read_bytes(), str(labels_path))
        emitted = 0
        for entry in labels:
            row = _row_from_label_entry(
                entry,
                split=split,
                image_lookup=lambda image_id, index=image_index: _read_local_image_bytes(index, image_id),
            )
            rows.append(row)
            emitted += 1
            if len(rows) >= batch_size:
                yield _batch_from_rows(rows)
                rows = []
            if limit is not None and emitted >= limit:
                break
    if rows:
        yield _batch_from_rows(rows)


def _s3_source_batches(
    source: str,
    *,
    splits: list[str],
    limit: int | None,
    batch_size: int,
) -> Iterable[_BatchPayload]:
    location = _parse_s3_uri(source)
    client = _s3_client()
    keys = _list_s3_keys(client, location)
    image_index = {
        Path(key).name: key
        for key in keys
        if key.lower().endswith((".jpg", ".jpeg"))
    }
    rows: list[dict[str, Any]] = []
    for split in splits:
        label_key = _find_s3_label_key(keys, split)
        labels = _load_json_array(_read_s3_bytes(client, location.bucket, label_key), f"s3://{location.bucket}/{label_key}")
        emitted = 0
        for entry in labels:
            row = _row_from_label_entry(
                entry,
                split=split,
                image_lookup=lambda image_id, index=image_index: _read_s3_bytes(
                    client,
                    location.bucket,
                    _require_image_key(index, image_id),
                ),
            )
            rows.append(row)
            emitted += 1
            if len(rows) >= batch_size:
                yield _batch_from_rows(rows)
                rows = []
            if limit is not None and emitted >= limit:
                break
    if rows:
        yield _batch_from_rows(rows)


def _row_from_label_entry(
    entry: Any,
    *,
    split: str,
    image_lookup,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise BDD100KSourceError("label entries must be JSON objects")
    image_id = _image_id_from_entry(entry)
    image_bytes = image_lookup(Path(image_id).name)
    if image_bytes is None:
        raise BDD100KSourceError(f"image not found for BDD100K label entry: {image_id}")
    width, height = _image_size(image_bytes, entry)
    categories, bboxes, occluded = _annotations_from_entry(entry)
    attrs = entry.get("attributes") if isinstance(entry.get("attributes"), dict) else {}
    return {
        "image_id": image_id,
        "image_bytes": image_bytes,
        "width": width,
        "height": height,
        "weather": _string_value(attrs.get("weather"), "undefined"),
        "scene": _string_value(attrs.get("scene"), "undefined"),
        "timeofday": _string_value(attrs.get("timeofday"), "undefined"),
        "timestamp": _timestamp_from_entry(entry),
        "ann_categories": categories,
        "ann_bboxes": bboxes,
        "ann_occluded": occluded,
        "split": split,
    }


def _image_id_from_entry(entry: dict[str, Any]) -> str:
    for key in ("image_id", "name", "file_name", "filename"):
        value = entry.get(key)
        if value:
            return str(value)
    raise BDD100KSourceError("label entry is missing image_id/name/file_name")


def _annotations_from_entry(entry: dict[str, Any]) -> tuple[list[str], list[list[float]], list[bool]]:
    categories: list[str] = []
    bboxes: list[list[float]] = []
    occluded: list[bool] = []
    labels = entry.get("labels") or entry.get("annotations") or []
    if not isinstance(labels, list):
        return categories, bboxes, occluded
    for label in labels:
        if not isinstance(label, dict):
            continue
        category = label.get("category") or label.get("label") or label.get("name")
        bbox = _bbox_from_label(label)
        if category is None or bbox is None:
            continue
        attrs = label.get("attributes") if isinstance(label.get("attributes"), dict) else {}
        categories.append(str(category))
        bboxes.append(bbox)
        occluded.append(bool(attrs.get("occluded", label.get("occluded", False))))
    return categories, bboxes, occluded


def _bbox_from_label(label: dict[str, Any]) -> list[float] | None:
    box2d = label.get("box2d")
    if isinstance(box2d, dict):
        try:
            return [float(box2d[key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            return None
    bbox = label.get("bbox") or label.get("box")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return [float(value) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None
    return None


def _image_size(image_bytes: bytes, entry: dict[str, Any]) -> tuple[int, int]:
    width = entry.get("width")
    height = entry.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    with Image.open(io.BytesIO(image_bytes)) as image:
        return int(image.width), int(image.height)


def _timestamp_from_entry(entry: dict[str, Any]) -> datetime | None:
    for key in ("timestamp", "timestamp_ms", "time"):
        value = entry.get(key)
        if value is None:
            continue
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return _parse_timestamp(float(stripped))
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
    return None


def _string_value(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _find_local_label_file(root: Path, split: str) -> Path:
    for pattern in LABEL_FILE_CANDIDATES:
        basename = pattern.format(split=split)
        matches = sorted(path for path in root.rglob(basename) if path.is_file())
        if matches:
            return matches[0]
    names = ", ".join(pattern.format(split=split) for pattern in LABEL_FILE_CANDIDATES)
    raise BDD100KSourceError(f"no BDD100K label file found for split {split}; expected {names}")


def _find_s3_label_key(keys: list[str], split: str) -> str:
    for pattern in LABEL_FILE_CANDIDATES:
        basename = pattern.format(split=split)
        matches = sorted(key for key in keys if Path(key).name == basename)
        if matches:
            return matches[0]
    names = ", ".join(pattern.format(split=split) for pattern in LABEL_FILE_CANDIDATES)
    raise BDD100KSourceError(f"no BDD100K label object found for split {split}; expected {names}")


def _local_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            index.setdefault(path.name, path)
    return index


def _read_local_image_bytes(index: dict[str, Path], image_id: str) -> bytes:
    path = index.get(Path(image_id).name)
    if path is None:
        raise BDD100KSourceError(f"image not found for BDD100K label entry: {image_id}")
    return path.read_bytes()


def _load_json_array(raw: bytes, label: str) -> list[Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BDD100KSourceError(f"invalid JSON in {label}: {exc.msg}") from exc
    if not isinstance(data, list):
        raise BDD100KSourceError(f"BDD100K labels must be a JSON array: {label}")
    return data


def _parse_s3_uri(uri: str) -> _S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BDD100KSourceError(f"invalid S3 URI: {uri}")
    return _S3Location(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise BDD100KSourceError("Reading s3:// sources requires boto3") from exc
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client("s3", endpoint_url=endpoint_url)


def _list_s3_keys(client: Any, location: _S3Location) -> list[str]:
    prefix = location.prefix.rstrip("/")
    if prefix:
        prefix = f"{prefix}/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=location.bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    if not keys:
        raise BDD100KSourceError(f"S3 source contains no objects: s3://{location.bucket}/{location.prefix}")
    return keys


def _read_s3_bytes(client: Any, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _require_image_key(index: dict[str, str], image_id: str) -> str:
    key = index.get(Path(image_id).name)
    if key is None:
        raise BDD100KSourceError(f"image object not found for BDD100K label entry: {image_id}")
    return key


def _connect_lancedb(lance_uri: str):
    try:
        import lancedb
    except ImportError as exc:
        raise BDD100KWriteError("Writing BDD100K imports requires the lancedb package") from exc
    try:
        return lancedb.connect(lance_uri)
    except Exception as exc:
        raise BDD100KWriteError(f"failed to connect to LanceDB URI {lance_uri}: {exc}") from exc


def _open_or_create_table(db: Any, table_name: str, data: pa.Table) -> tuple[Any, int | None]:
    try:
        exists = table_name in _list_tables(db)
        if exists:
            table_obj = db.open_table(table_name)
            before = _table_version(table_obj)
            table_obj.add(data)
            return table_obj, before
        table_obj = db.create_table(table_name, data=data, mode="create")
        return table_obj, None
    except Exception as exc:
        raise BDD100KWriteError(f"failed to write LanceDB table {table_name}: {exc}") from exc


def _table_version(table_obj: Any) -> int | None:
    value = getattr(table_obj, "version", None)
    if value is None:
        return None
    try:
        return int(value() if callable(value) else value)
    except (TypeError, ValueError):
        return None


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

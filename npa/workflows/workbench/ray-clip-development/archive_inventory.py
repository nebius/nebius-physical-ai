# Validate completed native CLIP result bytes without importing the GPU application.
"""Local format checks shared by CLIP archive and verified restore."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def digest(content: bytes) -> str:
    """Return the SHA-256 identity of exact bytes."""
    return hashlib.sha256(content).hexdigest()


def canonical(value: object) -> bytes:
    """Encode deterministic, finite JSON metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_name(name: str) -> str:
    """Reject ambiguous, nonportable or escaping relative file names."""
    if (not isinstance(name, str) or not name or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe result file name")
    return name


def require_digest(value: str) -> str:
    """Require one canonical SHA-256 digest."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA-256 digest")
    return value


def _unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON name")
        result[name] = value
    return result


def read_json(content: bytes):
    """Decode JSON without silently accepting duplicate names or nonfinite data."""
    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(content, object_pairs_hook=_unique_pairs, parse_constant=invalid_constant)


def _checksums(root, files):
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("Malformed SHA256SUMS entry")
        name = safe_name(line[66:])
        if name in expected:
            raise ValueError("Duplicate checksum name")
        expected[name] = require_digest(line[:64])
    observed = {name: entry["sha256"] for name, entry in files.items() if name != "SHA256SUMS"}
    if expected != observed:
        raise ValueError("Checksum inventory is incomplete or differs from actual bytes")
    if "sha256.json" in files:
        secondary = read_json((root / "sha256.json").read_bytes())
        if secondary != {name: value for name, value in expected.items() if name != "sha256.json"}:
            raise ValueError("The two checksum inventories disagree")


def _table(root, report):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(root / "embeddings.parquet").sort_by("record_id")
    expected_schema = pa.schema([("record_id", pa.int64()), ("input_sha256", pa.string()),
                                 ("processed_sha256", pa.string()),
                                 ("vector", pa.list_(pa.float32(), 512))])
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise ValueError("Unexpected CLIP table schema")
    count = report["records"]
    if (type(count) is not int or count < 1 or report["lance_rows"] != count
            or table["record_id"].to_pylist() != list(range(count))):
        raise ValueError("Incomplete or duplicate result rows")
    for field in ("input_sha256", "processed_sha256"):
        for value in table[field].to_pylist():
            require_digest(value)
    vectors = np.asarray(table["vector"].to_pylist(), dtype=np.float32)
    if (vectors.shape != (count, 512) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Invalid CLIP vectors")
    if digest(vectors.tobytes()) != report["vector_bytes_sha256"]:
        raise ValueError("Report vector hash differs")
    return table


def _lance_and_previews(root, table, files):
    import lancedb
    from PIL import Image

    database = lancedb.connect(str(root / "lance"))
    persisted = database.open_table("embeddings").to_arrow().sort_by("record_id")
    if persisted.to_pylist() != table.to_pylist():
        raise ValueError("Lance and Parquet rows differ")
    expected_images = set()
    contact = Image.new("RGB", (448, 224 * min(8, len(table))))
    for index, row in enumerate(table.slice(0, 8).to_pylist()):
        for column, (suffix, field) in enumerate((("original", "input_sha256"), ("crop", "processed_sha256"))):
            name = f"images/{row['record_id']:06d}-{suffix}.png"
            expected_images.add(name)
            if files[name]["sha256"] != row[field]:
                raise ValueError("Preview input bytes differ from embedded images")
            with Image.open(root / name) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError("Preview input is not RGB")
                if image.size != ((256, 256) if suffix == "original" else (224, 224)):
                    raise ValueError("Preview input dimensions differ")
                contact.paste(image.resize((224, 224)), (224 * column, 224 * index))
    if {name for name in files if name.startswith("images/")} != expected_images:
        raise ValueError("Preview inventory differs")
    with Image.open(root / "preview.png") as image:
        image.load()
        if image.mode != "RGB" or image.size != contact.size or image.tobytes() != contact.tobytes():
            raise ValueError("Invalid contact sheet")


def _provenance(root, report, files, advanced):
    if advanced:
        import validation

        sources = {name: require_digest(report[field]) for name, field in validation.SOURCE_HASH_FIELDS.items()}
        validation.verify_submitted_sources(report, sources)
        for actor in report["model_initializations"]:
            if (actor["execution_fingerprint"] != report["execution_fingerprint"]
                    or actor["model_revision"] != report["model_revision"]):
                raise ValueError("Actor execution provenance differs")
        recovery = report["recovery"]
        if recovery is None:
            if "recovery.json" in files:
                raise ValueError("Unexpected recovery artifact")
        elif (not isinstance(recovery, dict) or "recovery.json" not in files
              or read_json((root / "recovery.json").read_bytes()) != recovery
              or recovery["parquet_sha256"] != files["shards/000000/embeddings.parquet"]["sha256"]):
            raise ValueError("Recovery artifact is missing or inconsistent")
    else:
        sources = report["source_sha256"]
        if not isinstance(sources, dict) or set(sources) != {"embed.py", "worker.py", "npa_lancedb_bdd100k_udfs.py"}:
            raise ValueError("Basic source provenance is incomplete")
        for value in sources.values():
            require_digest(value)
        actors = report["actors"]
        if not isinstance(actors, list) or not actors or any(actor["source_sha256"] != sources for actor in actors):
            raise ValueError("Basic actor source provenance differs")
        if report["ray_nodes"] != len({actor["node_id"] for actor in actors}):
            raise ValueError("Basic actor inventory differs")


def _advanced(root, report, table, files):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fingerprint = require_digest(report["execution_fingerprint"])
    if read_json((root / "execution.json").read_bytes()) != {"execution_fingerprint": fingerprint}:
        raise ValueError("Execution identity differs")
    cleanup = read_json((root / "actor-cleanup.json").read_bytes())
    if cleanup["errors"] != [] or cleanup["attempted"] != report["gpu_actors"]:
        raise ValueError("Advanced result has no successful actor cleanup")
    count, size = report["records"], report["batch_size"]
    if type(size) is not int or size < 1:
        raise ValueError("Invalid shard size")
    expected = set()
    shards = []
    for index, start in enumerate(range(0, count, size)):
        prefix = f"shards/{index:06d}"
        expected.update({f"{prefix}/commit.json", f"{prefix}/embeddings.parquet"})
        receipt = read_json((root / prefix / "commit.json").read_bytes())
        shard = pq.read_table(root / prefix / "embeddings.parquet")
        ids = list(range(start, min(count, start + size)))
        identity = receipt["identity"]
        if (shard["record_id"].to_pylist() != ids or receipt["rows"] != len(ids)
                or identity != {
                    "record_ids": ids,
                    "input_hash": digest(canonical(shard["input_sha256"].to_pylist())),
                    "processed_hash": digest(canonical(shard["processed_sha256"].to_pylist())),
                    "source_sha256": report["source_sha256"],
                    "model_revision": report["model_revision"],
                    "execution_fingerprint": fingerprint,
                }
                or receipt["parquet_sha256"] != files[f"{prefix}/embeddings.parquet"]["sha256"]):
            raise ValueError("Shard identity, rows or committed bytes differ")
        shards.append(shard)
    if (report["shards"] != len(shards)
            or {name for name in files if name.startswith("shards/")} != expected
            or pa.concat_tables(shards).sort_by("record_id").to_pylist() != table.to_pylist()):
        raise ValueError("Shard inventory differs from completed table")
    for field, column in (("input_hash", "input_sha256"), ("processed_hash", "processed_sha256")):
        if report[field] != digest(canonical(table[column].to_pylist())):
            raise ValueError("Report input identity differs")


def _validate_result(root, files):
    _checksums(root, files)
    report = read_json((root / "report.json").read_bytes())
    advanced = report.get("schema_version") == "npa.ray-clip-development.v1"
    if not advanced and ("schema_version" in report or "sha256.json" not in files):
        raise ValueError("Unknown completed CLIP result format")
    _provenance(root, report, files, advanced)
    table = _table(root, report)
    hash_field = "embedding_sha256" if advanced else "parquet_sha256"
    if report[hash_field] != files["embeddings.parquet"]["sha256"]:
        raise ValueError("Report Parquet identity differs")
    retrieval = read_json((root / "retrieval.json").read_bytes())
    count = len(table)
    ids = sorted({0, count // 2, count - 1})
    if advanced:
        ids = list(dict.fromkeys([0, count // 4, count // 2, 3 * count // 4, count - 1]))
    if [row["query_id"] for row in retrieval] != ids:
        raise ValueError("Incomplete retrieval inventory")
    for row in retrieval:
        if (type(row["query_id"]) is not int or row["query_id"] not in row["top_ids"]
                or len(row["top_ids"]) != min(count, 5)
                or len(set(row["top_ids"])) != len(row["top_ids"])
                or any(type(value) is not int or value not in range(len(table)) for value in row["top_ids"])):
            raise ValueError("Invalid retrieval result")
    if advanced:
        if report["retrieval_queries"] != len(retrieval):
            raise ValueError("Report retrieval count differs")
        _advanced(root, report, table, files)
    elif report["retrieval"] != retrieval or any(name.startswith("shards/") for name in files):
        raise ValueError("Basic result metadata differs")
    _lance_and_previews(root, table, files)
    return "advanced" if advanced else "basic"


def validate_result(root: Path, files: dict) -> str:
    """Validate a private frozen tree against the existing basic/advanced formats.

    Args:
        root: Private staging tree whose files were safely copied and hashed.
        files: Complete relative file names mapped to size and SHA-256.
    Returns:
        The recognized basic or advanced result format.
    Raises:
        ValueError: Inventories, completion evidence or actual formats disagree.
        OSError: A required file cannot be read.
    """
    try:
        return _validate_result(root, files)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Incomplete or malformed CLIP result metadata") from error

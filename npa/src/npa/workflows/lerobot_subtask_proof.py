"""Verify reviewed LeRobot v3 subtask labels and publish one row-level proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

PROOF_SCHEMA = "npa.lerobot.subtask_proof.v1"
REQUIRED_FRAME_COLUMNS = {
    "episode_index",
    "frame_index",
    "timestamp",
    "subtask_index",
}


class LeRobotSubtaskProofError(ValueError):
    """Raised when a dataset cannot prove complete LeRobot subtask labels.

    Args:
        message: Human-readable contract failure.

    Returns:
        None.

    Raises:
        None.
    """


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotSubtaskProofError("LeRobot meta/info.json is missing or invalid") from exc
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise LeRobotSubtaskProofError("LeRobot info.json must contain a features object")
    feature = info["features"].get("subtask_index")
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise LeRobotSubtaskProofError("dataset must declare LeRobot codebase_version v3")
    if not isinstance(feature, dict) or feature.get("dtype") != "int64":
        raise LeRobotSubtaskProofError("LeRobot info.json must declare int64 subtask_index")
    return info


def _read_catalog(root: Path) -> dict[int, str]:
    path = root / "meta" / "subtasks.parquet"
    if not path.is_file():
        raise LeRobotSubtaskProofError("LeRobot dataset is missing meta/subtasks.parquet")
    table = pq.read_table(path)
    required = {"subtask", "subtask_index"}
    if not required.issubset(table.column_names):
        raise LeRobotSubtaskProofError("subtask catalog is missing required columns")
    if table.schema.field("subtask_index").type != pa.int64():
        raise LeRobotSubtaskProofError("subtask catalog index must be int64")
    catalog: dict[int, str] = {}
    for row in table.select(sorted(required)).to_pylist():
        try:
            index = int(row["subtask_index"])
            label = str(row["subtask"] or "").strip()
        except (TypeError, ValueError) as exc:
            raise LeRobotSubtaskProofError("subtask catalog has an invalid entry") from exc
        if index < 0 or not label or index in catalog or label in catalog.values():
            raise LeRobotSubtaskProofError("subtask catalog has an invalid or duplicate entry")
        catalog[index] = label
    if not catalog:
        raise LeRobotSubtaskProofError("subtask catalog is empty")
    return catalog


def _normalize_frame(row: Mapping[str, Any], source_file: str, digest: str) -> dict[str, Any]:
    try:
        normalized = {
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "subtask_index": int(row["subtask_index"]),
            "source_data_file": source_file,
            "source_parquet_sha256": digest,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise LeRobotSubtaskProofError("LeRobot frame has invalid subtask fields") from exc
    numeric = (normalized["episode_index"], normalized["frame_index"], normalized["timestamp"])
    if (
        any(value < 0 for value in numeric)
        or not math.isfinite(normalized["timestamp"])
        or normalized["subtask_index"] < 0
    ):
        raise LeRobotSubtaskProofError("every LeRobot frame must have a nonnegative subtask_index")
    return normalized


def _read_frames(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise LeRobotSubtaskProofError("LeRobot dataset has no data parquet files")
    frames: list[dict[str, Any]] = []
    for path in paths:
        schema = pq.read_schema(path)
        if not REQUIRED_FRAME_COLUMNS.issubset(schema.names):
            missing = sorted(REQUIRED_FRAME_COLUMNS.difference(schema.names))
            raise LeRobotSubtaskProofError(f"LeRobot data parquet is missing columns: {missing}")
        for column in ("episode_index", "frame_index", "subtask_index"):
            if schema.field(column).type != pa.int64():
                raise LeRobotSubtaskProofError(f"LeRobot frame column {column} must be int64")
        table = pq.read_table(path, columns=sorted(REQUIRED_FRAME_COLUMNS))
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        frames.extend(
            _normalize_frame(row, relative, digest)
            for row in table.select(sorted(REQUIRED_FRAME_COLUMNS)).to_pylist()
        )
    frames.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
    return frames


def _validate_frames(frames: list[dict[str, Any]], catalog: Mapping[int, str]) -> None:
    if not frames:
        raise LeRobotSubtaskProofError("LeRobot dataset has no frames")
    identities: set[tuple[int, int]] = set()
    for row in frames:
        identity = (row["episode_index"], row["frame_index"])
        if identity in identities:
            raise LeRobotSubtaskProofError(f"duplicate LeRobot frame identity: {identity}")
        identities.add(identity)
        if row["subtask_index"] not in catalog:
            raise LeRobotSubtaskProofError(
                f"unknown subtask_index {row['subtask_index']} at frame {identity}"
            )


def _summary(frames: list[dict[str, Any]], catalog: Mapping[int, str]) -> dict[str, int]:
    segments = 0
    previous: tuple[int, int] | None = None
    for row in frames:
        current = (row["episode_index"], row["subtask_index"])
        segments += current != previous
        previous = current
    return {
        "episode_count": len({row["episode_index"] for row in frames}),
        "frame_count": len(frames),
        "labeled_frame_count": len(frames),
        "unlabeled_frame_count": 0,
        "subtask_count": len(catalog),
        "segment_count": segments,
    }


def _select_proof(
    frames: list[dict[str, Any]], catalog: Mapping[int, str], expected_label: str
) -> dict[str, Any]:
    selected = next(
        (row for row in frames if not expected_label or catalog[row["subtask_index"]] == expected_label),
        None,
    )
    if selected is None:
        raise LeRobotSubtaskProofError(f"expected subtask label was not found: {expected_label}")
    proof = dict(selected)
    proof["subtask"] = catalog[proof["subtask_index"]]
    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()
    proof["row_sha256"] = hashlib.sha256(canonical).hexdigest()
    return proof


def _validate_destinations(dataset_uri: str, proof_uri: str) -> None:
    source, destination = urlparse(dataset_uri), urlparse(proof_uri)
    if source.scheme not in ("", "s3") or destination.scheme not in ("", "s3"):
        raise LeRobotSubtaskProofError("dataset and proof locations must be local paths or S3 URIs")
    if destination.scheme == "s3":
        if not destination.netloc or not destination.path.strip("/") or proof_uri.endswith("/"):
            raise LeRobotSubtaskProofError("proof_uri must name an exact S3 object")
        if source.netloc == destination.netloc and destination.path.startswith(source.path.rstrip("/") + "/"):
            raise LeRobotSubtaskProofError("proof_uri must be outside the source dataset")
    elif not source.scheme:
        root = Path(dataset_uri).expanduser().resolve()
        if Path(proof_uri).expanduser().resolve().is_relative_to(root):
            raise LeRobotSubtaskProofError("proof_uri must be outside the source dataset")


def _materialize(dataset_uri: str, temporary_root: Path, storage_client: Any | None) -> Path:
    if not dataset_uri.startswith("s3://"):
        return Path(dataset_uri).expanduser().resolve()
    root = temporary_root / "dataset"
    client = storage_client or _storage()
    prefix = dataset_uri.rstrip("/")
    for relative in ("meta/info.json", "meta/subtasks.parquet"):
        client.download_file(f"{prefix}/{relative}", str(root / relative))
    client.download_directory(f"{prefix}/data/", str(root / "data"))
    return root


def _build_proof(root: Path, expected_label: str) -> dict[str, Any]:
    info = _read_info(root)
    catalog = _read_catalog(root)
    frames = _read_frames(root)
    _validate_frames(frames, catalog)
    summary = _summary(frames, catalog)
    for declared, measured in (("total_frames", "frame_count"), ("total_episodes", "episode_count")):
        if info.get(declared) != summary[measured]:
            raise LeRobotSubtaskProofError(f"LeRobot {declared} does not match the observed dataset")
    return {
        "schema": PROOF_SCHEMA,
        "status": "verified",
        "source_format": f"lerobot-{info['codebase_version']}",
        "source_catalog_sha256": _sha256(root / "meta" / "subtasks.parquet"),
        "expected_subtask": expected_label or None,
        "catalog": [{"subtask_index": index, "subtask": catalog[index]} for index in sorted(catalog)],
        "summary": summary,
        "proof": _select_proof(frames, catalog, expected_label),
    }


def _publish(
    payload: dict[str, Any], uri: str, *, temporary_root: Path, storage_client: Any | None
) -> str:
    staged = temporary_root / "subtask-proof.json"
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if uri.startswith("s3://"):
        return str((storage_client or _storage()).upload_file(str(staged), uri))
    destination = Path(uri).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(staged.read_bytes())
    return str(destination)


def prove_lerobot_subtasks(
    dataset_uri: str,
    proof_uri: str,
    *,
    expected_label: str = "",
    storage_client: Any | None = None,
) -> dict[str, Any]:
    """Validate complete labels and publish a proof from a real LeRobot frame row.

    Args:
        dataset_uri: Local directory or S3 prefix containing the reviewed dataset.
        proof_uri: Local file or exact S3 object URI for the JSON proof.
        expected_label: Optional label that the selected proof row must resolve to.
        storage_client: Optional compatible storage client for tests or callers.

    Returns:
        The validated proof document written to ``proof_uri``.

    Raises:
        LeRobotSubtaskProofError: If metadata, coverage, or catalog resolution fails.
    """

    _validate_destinations(dataset_uri, proof_uri)
    with tempfile.TemporaryDirectory(prefix="npa-lerobot-subtask-proof-") as temporary:
        temporary_root = Path(temporary)
        root = _materialize(dataset_uri, temporary_root, storage_client)
        payload = _build_proof(root, expected_label)
        payload.update({"source_dataset_uri": dataset_uri, "artifact_uri": proof_uri})
        _publish(payload, proof_uri, temporary_root=temporary_root, storage_client=storage_client)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--proof-uri", required=True)
    parser.add_argument("--expected-label", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the LeRobot subtask proof publisher from an argv-safe entry point.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after the proof is published; validation failures propagate.

    Raises:
        LeRobotSubtaskProofError: The input fails the subtask contract.
    """

    proof = prove_lerobot_subtasks(**vars(_parser().parse_args(argv)))
    print(json.dumps(proof, sort_keys=True))
    return 0


__all__ = [
    "LeRobotSubtaskProofError",
    "PROOF_SCHEMA",
    "main",
    "prove_lerobot_subtasks",
]


if __name__ == "__main__":  # pragma: no cover - exercised through workflow argv
    raise SystemExit(main())

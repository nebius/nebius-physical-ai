"""Verify Gymnasium-Robotics public corresponding-source delivery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable

MAX_METADATA_BYTES = 1_048_576
MAX_SOURCE_ARCHIVE_BYTES = 21_474_836_480
READ_CHUNK_BYTES = 1_048_576
ACCEPTED_RECORD = Path(__file__).with_name("gymnasium_robotics_image_manifest.json")
SOURCE_LOCK = (
    Path(__file__).resolve().parents[3]
    / "docker/workbench/gymnasium-robotics/corresponding-source.lock.json"
)


class CorrespondingSourceError(RuntimeError):
    """Report a fail-closed corresponding-source delivery contract violation.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrespondingSourceError(message)


def _read_regular_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CorrespondingSourceError(f"cannot open {label}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
        _require(metadata.st_size <= MAX_METADATA_BYTES, f"{label} is too large")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, READ_CHUNK_BYTES):
            chunks.append(chunk)
        content = b"".join(chunks)
        _require(len(content) == metadata.st_size, f"{label} changed while reading")
        return content
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    content = _read_regular_file(path, label)
    try:
        payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorrespondingSourceError(f"{label} is not valid JSON") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload, content


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    _require(set(value) == fields, f"{label} fields are incomplete or unsupported")


def _digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
        f"{label} must be an exact sha256 digest",
    )
    return value


def _sha256_hex(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        f"{label} must be an exact SHA-256",
    )
    return value


def _validate_lock(lock: dict[str, Any]) -> dict[str, Any]:
    _require(
        lock.get("schema") == "npa.gymnasium-robotics.baked-corresponding-source-lock.v2",
        "corresponding-source lock schema is unsupported",
    )
    _require(lock.get("status") == "complete", "corresponding-source lock is incomplete")
    _require(
        lock.get("public_corresponding_source_delivery") == "accepted-public-immutable",
        "public corresponding-source delivery is not accepted",
    )
    deliveries = lock.get("deliveries")
    _require(isinstance(deliveries, list) and len(deliveries) == 1, "lock requires one delivery")
    delivery = _mapping(deliveries[0], "locked delivery")
    _sha256_hex(delivery.get("binary_manifest_sha256"), "binary manifest")
    _sha256_hex(delivery.get("build_materials_sha256"), "build materials")
    _sha256_hex(delivery.get("source_manifest_sha256"), "source manifest")
    _require(bool(delivery.get("artifacts")), "locked delivery has no artifacts")
    return delivery


def _validate_reference(reference: Any, artifact_sha256: str) -> str:
    _require(isinstance(reference, str), "source artifact reference must be a string")
    parsed = urllib.parse.urlsplit(reference)
    _require(parsed.scheme == "https", "source artifact reference must use HTTPS")
    _require(bool(parsed.hostname), "source artifact reference has no host")
    _require(not parsed.username and not parsed.password, "source artifact reference embeds credentials")
    _require(not parsed.query and not parsed.fragment, "source artifact reference must be immutable")
    hostname = str(parsed.hostname).lower()
    _require(hostname != "localhost" and not hostname.endswith(".local"), "source artifact host is not public")
    _require(artifact_sha256 in parsed.path.lower(), "source artifact reference is not digest-addressed")
    return reference


def _validate_record(
    record: dict[str, Any], lock: dict[str, Any], lock_bytes: bytes, subject: dict[str, str]
) -> dict[str, Any]:
    _exact_fields(
        record,
        {"format", "status", "tool", "source_revision", "image", "corresponding_source"},
        "accepted record",
    )
    _require(record["format"] == "npa_gymnasium_robotics_accepted_image_manifest_v1", "accepted record format is unsupported")
    _require(record["status"] == "accepted-for-publication", "publication record is not accepted")
    _require(record["tool"] == "gymnasium-robotics", "publication record has the wrong subject")
    _require(record["source_revision"] == subject["source_revision"], "source revision does not match")
    _validate_image(_mapping(record["image"], "image binding"), subject)
    source = _mapping(record["corresponding_source"], "corresponding-source binding")
    _exact_fields(source, {"lock_sha256", "delivery", "artifact"}, "corresponding-source binding")
    _require(source["lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest(), "corresponding-source lock digest does not match")
    delivery = _validate_lock(lock)
    _require(source["delivery"] == delivery, "corresponding-source delivery does not match the lock")
    artifact = _mapping(source["artifact"], "source artifact")
    _validate_artifact(artifact, delivery)
    return artifact


def _validate_image(image: dict[str, Any], subject: dict[str, str]) -> None:
    _exact_fields(
        image,
        {"digest", "platform", "platform_manifest_digest", "config_digest"},
        "image binding",
    )
    platform = _mapping(image["platform"], "image platform")
    _exact_fields(platform, {"os", "architecture"}, "image platform")
    _require(platform == {"os": "linux", "architecture": "amd64"}, "image platform is not linux/amd64")
    for field in ("digest", "platform_manifest_digest", "config_digest"):
        _digest(image[field], f"image {field}")
        _require(image[field] == subject[field], f"image {field} does not match")


def _validate_artifact(artifact: dict[str, Any], delivery: dict[str, Any]) -> None:
    _exact_fields(
        artifact,
        {"reference", "sha256", "size_bytes", "media_type", "anonymous", "immutable", "contents_manifest_sha256"},
        "source artifact",
    )
    artifact_sha256 = _sha256_hex(artifact["sha256"], "source artifact")
    _validate_reference(artifact["reference"], artifact_sha256)
    _require(artifact["anonymous"] is True, "source artifact is not anonymous")
    _require(artifact["immutable"] is True, "source artifact is not immutable")
    _require(artifact["media_type"] == "application/zstd", "source artifact media type is unsupported")
    size = artifact["size_bytes"]
    _require(isinstance(size, int) and 0 < size <= MAX_SOURCE_ARCHIVE_BYTES, "source artifact size is invalid")
    _require(
        artifact["contents_manifest_sha256"] == delivery["source_manifest_sha256"],
        "source artifact contents manifest does not match the lock",
    )


def _open_anonymous(request: urllib.request.Request, timeout: float) -> BinaryIO:
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def _verify_download(artifact: dict[str, Any], opener: Callable[..., BinaryIO]) -> None:
    request = urllib.request.Request(
        artifact["reference"], headers={"Accept": "application/octet-stream"}
    )
    expected_size = artifact["size_bytes"]
    digest = hashlib.sha256()
    total = 0
    try:
        with opener(request, timeout=60) as response:
            _require(getattr(response, "status", 200) == 200, "anonymous source retrieval failed")
            _require(response.geturl() == artifact["reference"], "source artifact redirected")
            while chunk := response.read(READ_CHUNK_BYTES):
                total += len(chunk)
                _require(total <= expected_size, "source artifact exceeds accepted size")
                digest.update(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise CorrespondingSourceError("anonymous source retrieval failed") from exc
    _require(total == expected_size, "source artifact size does not match")
    _require(digest.hexdigest() == artifact["sha256"], "source artifact digest does not match")


def verify_corresponding_source_delivery(
    record_path: Path,
    lock_path: Path,
    *,
    source_revision: str,
    image_digest: str,
    platform_manifest_digest: str,
    config_digest: str,
    opener: Callable[..., BinaryIO] = _open_anonymous,
) -> dict[str, Any]:
    """Validate and anonymously retrieve one accepted corresponding-source delivery.

    Args:
        record_path: Accepted Gymnasium image/publication record.
        lock_path: Exact corresponding-source lock used by the image build.
        source_revision: Full source Git revision represented by the image.
        image_digest: Immutable top-level development image digest.
        platform_manifest_digest: Immutable linux/amd64 manifest digest.
        config_digest: Immutable image configuration digest.
        opener: Anonymous HTTPS opener, injectable for hermetic tests.

    Returns:
        The validated accepted publication record.

    Raises:
        CorrespondingSourceError: Any record, binding, or retrieval is invalid.
    """
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", source_revision)), "source revision is invalid")
    subject = {
        "source_revision": source_revision,
        "digest": _digest(image_digest, "image digest"),
        "platform_manifest_digest": _digest(platform_manifest_digest, "platform manifest"),
        "config_digest": _digest(config_digest, "image config"),
    }
    record, _ = _load_json(record_path, "accepted publication record")
    lock, lock_bytes = _load_json(lock_path, "corresponding-source lock")
    artifact = _validate_record(record, lock, lock_bytes, subject)
    _verify_download(artifact, opener)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--platform-manifest-digest", required=True)
    parser.add_argument("--config-digest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the corresponding-source delivery verifier.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero after complete verification.

    Raises:
        CorrespondingSourceError: The delivery contract is not satisfied.
    """
    args = _parser().parse_args(argv)
    verify_corresponding_source_delivery(
        args.record,
        args.lock,
        source_revision=args.source_revision,
        image_digest=args.image_digest,
        platform_manifest_digest=args.platform_manifest_digest,
        config_digest=args.config_digest,
    )
    print("passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

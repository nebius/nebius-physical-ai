"""Conditionally stage and read back one pinned NCore qualification source."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.errors import NpaError
from npa.workbench.ncore_staging import private_staging_directory
from npa.workbench.nurec.s3_probe import _snapshot


STAGING_FORMAT = "npa_ncore_source_staging_v1"
DATASET_REPOSITORY = "nvidia/PhysicalAI-NuRec-PPISP"
DATASET_REVISION = "2521064a3af6ab1c1caa2ba1b01ddde7eecded69"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_MEMBER = "colmap/struktur28_colmap.zip"


class NcoreSourceStagingError(NpaError):
    """Pinned source staging or exact read-back failed."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _destination(uri: str) -> tuple[str, str, str]:
    parsed = urlparse(str(uri).strip())
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or key.endswith("/")
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise NcoreSourceStagingError("source destination must be an exact S3 object")
    parent = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    if not parent:
        raise NcoreSourceStagingError("source destination must use a run-owned prefix")
    return parsed.netloc, key, parent


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def stage_source_archive(
    source_path: Path,
    destination_uri: str,
    *,
    expected_sha256: str,
    receipt_path: Path,
    scratch_dir: Path,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Conditionally create, independently download, and hash one source object."""
    from botocore.exceptions import ClientError
    from npa.clients.storage import StorageClient

    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_nlink != 1
    ):
        raise NcoreSourceStagingError("source archive must be one private regular file")
    if (
        len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
        or _sha_file(source_path) != expected_sha256
    ):
        raise NcoreSourceStagingError("source archive SHA-256 differs")
    bucket, key, prefix = _destination(destination_uri)
    client = storage_client or StorageClient.from_environment()
    before = _snapshot(client, bucket, prefix)
    if before:
        raise NcoreSourceStagingError("source staging prefix is not fresh")
    response_loss_recovered = False
    try:
        with source_path.open("rb") as stream:
            response = client.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentType="application/zip",
                IfNoneMatch="*",
            )
        etag = str(response.get("ETag") or "").strip()
        if not etag:
            raise NcoreSourceStagingError("source staging returned no ETag")
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = int(
            exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
        )
        if code in {
            "412",
            "PreconditionFailed",
            "ConditionalRequestConflict",
        } or status in {
            409,
            412,
        }:
            raise NcoreSourceStagingError(
                "source destination was concurrently created"
            ) from exc
        response_loss_recovered = True
        etag = ""
    except Exception:
        response_loss_recovered = True
        etag = ""
    try:
        with private_staging_directory(scratch_dir, prefix="source-readback-") as root:
            readback = Path(root) / "source.zip"
            client.download_file(destination_uri, str(readback))
            if (
                readback.stat().st_size != source_path.stat().st_size
                or _sha_file(readback) != expected_sha256
            ):
                raise NcoreSourceStagingError("staged source read-back differs")
        head = client.s3.head_object(Bucket=bucket, Key=key)
        observed_etag = str(head.get("ETag") or "").strip()
        if not observed_etag or (etag and observed_etag != etag):
            raise NcoreSourceStagingError("staged source ETag differs")
        etag = observed_etag
    except NcoreSourceStagingError:
        raise
    except Exception as exc:
        raise NcoreSourceStagingError("staged source read-back failed") from exc
    attribution_uri = f"s3://{bucket}/{prefix}attribution.json"
    attribution = {
        "dataset": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "member": DATASET_MEMBER,
        "sha256": expected_sha256,
        "creator": "NVIDIA",
        "license": DATASET_LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": (
            f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/tree/"
            f"{DATASET_REVISION}"
        ),
        "changes": (
            "Unmodified source archive; qualification converts the full struktur28 "
            "capture to NCore V4 and derives the NRE rig edge."
        ),
        "selected_capture": "struktur28",
        "source_counts": {"images": 518, "cameras": 3, "points": 163453},
    }
    attribution_bytes = (
        json.dumps(attribution, indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        try:
            attribution_etag = client.put_bytes_conditional(
                attribution_bytes,
                attribution_uri,
                if_none_match=True,
                content_type="application/json",
            )
        except Exception:
            recovered = client.read_bytes_with_etag(attribution_uri)
            if recovered is None or recovered[0] != attribution_bytes:
                raise
            attribution_etag = recovered[1]
            response_loss_recovered = True
        recovered = client.read_bytes_with_etag(attribution_uri)
        if (
            recovered is None
            or recovered[0] != attribution_bytes
            or recovered[1] != attribution_etag
        ):
            raise NcoreSourceStagingError("source attribution read-back differs")
    except Exception as exc:
        try:
            client.s3.delete_object(Bucket=bucket, Key=key, IfMatch=etag)
        except Exception as cleanup_exc:
            raise NcoreSourceStagingError(
                "source attribution failed and source cleanup did not converge"
            ) from cleanup_exc
        raise NcoreSourceStagingError("source attribution staging failed") from exc
    after = _snapshot(client, bucket, prefix)
    key_sha256 = hashlib.sha256(key.encode()).hexdigest()
    matching = [item for item in after if item["key_sha256"] == key_sha256]
    attribution_key_sha256 = hashlib.sha256(
        f"{prefix}attribution.json".encode()
    ).hexdigest()
    attribution_matching = [
        item for item in after if item["key_sha256"] == attribution_key_sha256
    ]
    if (
        len(after) != 2
        or len(matching) != 1
        or matching[0]["bytes"] != source_path.stat().st_size
        or len(attribution_matching) != 1
        or attribution_matching[0]["bytes"] != len(attribution_bytes)
    ):
        raise NcoreSourceStagingError("source staging inventory differs")
    receipt = {
        "format": STAGING_FORMAT,
        "status": "pass",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "source_sha256": expected_sha256,
        "source_bytes": source_path.stat().st_size,
        "destination_uri_sha256": hashlib.sha256(destination_uri.encode()).hexdigest(),
        "object_key_sha256": key_sha256,
        "etag_sha256": hashlib.sha256(etag.encode()).hexdigest(),
        "attribution_sha256": hashlib.sha256(attribution_bytes).hexdigest(),
        "attribution_etag_sha256": hashlib.sha256(
            attribution_etag.encode()
        ).hexdigest(),
        "conditional_create": True,
        "response_loss_recovered": response_loss_recovered,
        "exact_readback": True,
        "enumerated_exactly_once": True,
        "before_objects": 0,
        "after_objects": 2,
    }
    _write_private(receipt_path, receipt)
    return receipt

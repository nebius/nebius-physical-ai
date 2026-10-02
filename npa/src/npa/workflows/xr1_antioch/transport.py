"""Connect an operator-owned Antioch project to run-scoped Nebius S3 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import urlsplit

from npa.clients.antioch import antioch_environment
from npa.clients.storage import StorageClient


def _worker(project: Path, request: dict) -> dict:
    command = [
        "antioch",
        "service",
        "exec",
        "--no-stream",
        "--no-tty",
        "--",
        "python",
        "-c",
        Path(__file__).with_name("artifact_worker.py").read_text(),
    ]
    result = subprocess.run(
        command,
        cwd=project,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=antioch_environment(),
    )
    if result.returncode:
        raise RuntimeError(
            f"Antioch artifact worker failed with exit {result.returncode}"
        )
    return json.loads(result.stdout)


def _destination(destination: str) -> tuple[str, str]:
    location = urlsplit(destination)
    prefix = location.path.strip("/")
    if (
        location.scheme != "s3"
        or not location.netloc
        or not prefix
        or location.query
        or location.fragment
    ):
        raise ValueError("Artifact destination must be a run-scoped S3 prefix")
    if ".." in PurePosixPath(prefix).parts:
        raise ValueError("Artifact destination cannot traverse its run prefix")
    return location.netloc, prefix


def _readback(client, bucket: str, prefix: str, manifest: dict) -> dict:
    verified = {}
    for name, expected in manifest.items():
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/{name}")
        digest, size = hashlib.sha256(), 0
        with response["Body"] as body:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(f"S3 readback differs from the Antioch artifact: {name}")
        verified[name] = expected
    return verified


def publish_artifacts(
    project: Path, remote_root: str, destination: str, storage: StorageClient
) -> dict:
    """Upload native artifacts and verify their complete bytes from Nebius S3.

    Args:
        project: Operator-owned local Antioch project directory.
        remote_root: Explicit output directory within its session.
        destination: Fresh run-scoped S3 prefix.
        storage: Authenticated operator storage client; credentials stay on the operator.
    Returns:
        Verified artifact identities and byte counts, without infrastructure identifiers.
    Raises:
        ValueError: Paths, manifests, or readback hashes are invalid.
        FileExistsError: Destination contains artifacts from another manifest.
        RuntimeError: Antioch transfer fails.
    """
    bucket, prefix = _destination(destination)
    client = storage.s3
    manifest = _worker(project, {"operation": "manifest", "root": remote_root})
    pending = _pending_files(client, bucket, prefix, manifest)
    transfers = _presigned_transfers(client, bucket, prefix, pending)
    result = _worker(
        project, {"operation": "upload", "root": remote_root, "transfers": transfers}
    )
    if set(result["uploaded"]) != set(pending):
        raise ValueError("Antioch did not upload every artifact")
    return {
        "files": _readback(client, bucket, prefix, manifest),
        "s3_readback_verified": True,
    }


def _pending_files(client, bucket: str, prefix: str, manifest: dict) -> dict:
    existing = set()
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix + "/"
    )
    for page in pages:
        existing.update(
            row["Key"][len(prefix) + 1 :] for row in page.get("Contents", [])
        )
    if not existing.issubset(manifest):
        raise FileExistsError("S3 prefix contains files outside this artifact manifest")
    if existing:
        _readback(client, bucket, prefix, {name: manifest[name] for name in existing})
    return {name: entry for name, entry in manifest.items() if name not in existing}


def _presigned_transfers(client, bucket: str, prefix: str, manifest: dict) -> dict:
    result = {}
    for name, entry in manifest.items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Remote manifest contains an out-of-root artifact")
        parameters = {
            "Bucket": bucket,
            "Key": f"{prefix}/{name}",
            "Metadata": {"sha256": entry["sha256"]},
        }
        result[name] = {
            **entry,
            "url": client.generate_presigned_url(
                "put_object", Params=parameters, ExpiresIn=3600
            ),
        }
    return result


def fetch_inputs(
    project: Path,
    remote_root: str,
    source: str,
    manifest: dict,
    storage: StorageClient,
    source_bundle: str | None = None,
) -> dict:
    """Make Antioch download and verify approved inputs directly from Nebius S3.

    Args:
        project: Operator-owned Antioch project.
        remote_root: Fresh destination inside its session.
        source: Run-scoped S3 input prefix.
        manifest: Relative file names mapped to approved SHA-256 digests and sizes.
        storage: Operator storage client; static credentials remain local.
        source_bundle: Optional verified ZIP file to extract as runtime source.
    Returns:
        Remote receipt of independently verified bytes.
    Raises:
        ValueError: An input path or returned receipt is invalid.
        RuntimeError: Antioch download or hash verification failed.
    """
    bucket, prefix = _destination(source)
    files = {}
    for name, identity in manifest.items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Input manifest contains an out-of-root path")
        files[name] = {
            **identity,
            "url": storage.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": f"{prefix}/{name}"},
                ExpiresIn=3600,
            ),
        }
    request = {"root": remote_root, "files": files, "source_bundle": source_bundle}
    code = Path(__file__).with_name("bootstrap_worker.py").read_text()
    result = subprocess.run(
        [
            "antioch",
            "service",
            "exec",
            "--no-stream",
            "--no-tty",
            "--",
            "python",
            "-c",
            code,
        ],
        cwd=project,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=antioch_environment(),
    )
    if result.returncode:
        raise RuntimeError(
            f"Antioch input verification failed with exit {result.returncode}"
        )
    receipt = json.loads(result.stdout)
    if receipt.get("files") != manifest or receipt.get("download_verified") is not True:
        raise ValueError("Antioch receipt differs from the approved input identities")
    return receipt

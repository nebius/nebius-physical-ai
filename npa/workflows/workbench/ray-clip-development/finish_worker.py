"""Upload and verify a completed driver's artifacts; never stop the Ray session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from urllib.parse import urlparse


def parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if (parsed.scheme != "s3" or not parsed.netloc or parsed.username or parsed.password
            or parsed.port or parsed.query or parsed.fragment or not key
            or any(part in {"", ".", ".."} for part in key.rstrip("/").split("/"))):
        raise ValueError("Expected an S3 URI with a bucket and a nonempty safe object prefix")
    return parsed.netloc, key.rstrip("/")


def s3_client():
    import boto3

    endpoint = (os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
                or os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("S3_ENDPOINT_URL")
                or os.environ.get("NPA_STORAGE_ENDPOINT"))
    return boto3.client("s3", endpoint_url=endpoint or None)


def inventory(root: Path) -> list[tuple[str, int, int, int]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("output-path must be an absolute regular directory, not a symlink")
    if root.resolve() == Path("/"):
        raise ValueError("Filesystem root cannot be an artifact directory")
    selected = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Symbolic links are forbidden in the artifact tree")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Only regular artifact files are supported")
        relative = path.relative_to(root).as_posix()
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("Artifact escaped its selected root")
        selected.append((relative, info.st_dev, info.st_ino, info.st_size))
    if not selected:
        raise ValueError("Artifact directory is empty")
    return selected


def open_relative(root: Path, relative: str):
    """Open using no-follow directory descriptors to reject parent-link races."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(part in {".", ".."} for part in parts):
        raise ValueError("Artifact path must stay relative to its selected root")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    return os.fdopen(file_descriptor, "rb")


def stream_hash(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def verify_object(client, bucket: str, key: str, sha256: str, size: int) -> None:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        observed_hash, observed_size = stream_hash(body)
    finally:
        body.close()
    if observed_hash != sha256 or observed_size != size:
        raise ValueError("S3 read-after-write object hash or size mismatch")


def put_immutable(client, bucket: str, key: str, body, sha256: str, size: int) -> None:
    from botocore.exceptions import ClientError

    try:
        client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": sha256})
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}:
            raise
        # An identical existing object is an idempotent retry; anything else fails.
    verify_object(client, bucket, key, sha256, size)


def upload_artifacts(client, root: Path, artifact_uri: str) -> dict:
    bucket, prefix = parse_s3(artifact_uri)
    selected = inventory(root)
    files = []
    for relative, device, inode, size in selected:
        with open_relative(root, relative) as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size) != (device, inode, size):
                raise ValueError("Artifact changed after inventory")
            digest, read_size = stream_hash(stream)
            if read_size != size:
                raise ValueError("Artifact size changed while hashing")
            stream.seek(0)
            put_immutable(client, bucket, f"{prefix}/files/{relative}", stream, digest, size)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Artifact changed while uploading")
        files.append({"path": relative, "sha256": digest, "size_bytes": size})
    if inventory(root) != selected:
        raise ValueError("Artifact tree changed while uploading")
    manifest = {"schema_version": "npa.ray-clip-artifacts.v1", "files": files,
                "file_count": len(files), "total_bytes": sum(item["size_bytes"] for item in files)}
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    put_immutable(client, bucket, f"{prefix}/manifest.json", payload, digest, len(payload))
    return {"manifest_uri": f"s3://{bucket}/{prefix}/manifest.json", "manifest_sha256": digest,
            "manifest_bytes": len(payload), "file_count": len(files), "total_bytes": manifest["total_bytes"],
            "all_objects_read_after_write_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--artifact-uri", required=True)
    args = parser.parse_args(argv)
    result = upload_artifacts(s3_client(), Path(args.output_path), args.artifact_uri)
    print("RAY_CLIP_ARTIFACTS " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

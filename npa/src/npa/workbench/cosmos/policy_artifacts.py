"""Content-verified local/S3 handoffs for native policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient, safe_s3_download_target
from npa.workbench.dataset.storage import read_json_uri, uri_join, write_json_uri


@contextmanager
def policy_workspace(output_path: str, phase: str) -> Iterator[Path]:
    """Retain private native diagnostics when setup or execution fails.

    Args:
        output_path: Run-scoped private output prefix.
        phase: Native execution phase name.
    Returns:
        Context manager yielding a fresh temporary directory.
    Raises:
        Exception: Original execution error, after attempting diagnostic publication.
    """
    with tempfile.TemporaryDirectory(prefix=f"npa-policy-{phase}-") as temporary:
        root = Path(temporary)
        try:
            yield root
        except Exception as exc:
            diagnostics = root / "failure-diagnostics"
            diagnostics.mkdir()
            for source in list(root.rglob("*.log")):
                if source.is_symlink() or source.is_relative_to(diagnostics):
                    continue
                target = diagnostics / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            (diagnostics / "traceback.log").write_text(traceback.format_exc())
            try:
                publish_bundle(diagnostics, uri_join(output_path, "failure"),
                               {"status": "failed", "phase": phase, "error_type": type(exc).__name__}, "failure.json")
            except Exception as publication_error:
                logging.getLogger(__name__).warning(
                    "Diagnostic publication also failed (%s)", type(publication_error).__name__
                )
            raise


def file_digest(path: Path) -> str:
    """Hash one artifact without loading model weights into memory.

    Args:
        path: Local regular file.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: File cannot be read.
    """
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def publish_bundle(root: Path, output_path: str, report: dict[str, Any], name: str) -> dict[str, Any]:
    """Publish artifact bytes before writing the completion manifest.

    Args:
        root: Directory containing only output artifacts.
        output_path: Local directory or S3 prefix.
        report: Metadata to bind to the artifacts.
        name: Completion manifest filename.
    Returns:
        Manifest including hashes and absolute artifact URIs.
    Raises:
        OSError: Publication failed or a local output already exists.
        ValueError: A symlink occurs in the artifact tree.
    """
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("checkpoint publication does not follow symlinks")
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    report = dict(report, artifacts=[{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": file_digest(path), "uri": uri_join(output_path, path.relative_to(root).as_posix()),
    } for path in files])
    if output_path.startswith("s3://"):
        StorageClient.from_environment().upload_directory(str(root), output_path)
    else:
        shutil.copytree(root, output_path)
    write_json_uri(uri_join(output_path, name), report)
    return report


def materialize_bundle(input_path: str, target: Path, schema: str) -> dict[str, Any]:
    """Fetch exactly the files named by a completed, hash-bound artifact manifest.

    Args:
        input_path: Local or S3 completion manifest.
        target: Fresh local staging directory.
        schema: Expected manifest schema.
    Returns:
        Verified manifest.
    Raises:
        ValueError: Schema, containment, file size, or content hash mismatch.
        OSError: Artifact retrieval failed.
    """
    report = read_json_uri(input_path)
    if report.get("schema") != schema or report.get("status") != "succeeded":
        raise ValueError("expected a completed native policy artifact manifest")
    entries = report.get("artifacts", [])
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("artifact manifest must contain unique files")
    for entry in entries:
        path = safe_s3_download_target(target, entry["path"], "")
        path.parent.mkdir(parents=True, exist_ok=True)
        uri = entry["uri"]
        if uri != uri_join(input_path.rsplit("/", 1)[0], entry["path"]):
            raise ValueError("artifact URI must remain in the manifest's own prefix")
        if uri.startswith("s3://"):
            StorageClient.from_environment().download_file(uri, str(path))
        else:
            shutil.copyfile(uri, path)
        if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError("policy artifact content verification failed")
    return report


def write_local_json(path: Path, data: dict[str, Any]) -> None:
    """Write human-readable local evidence.

    Args:
        path: Destination file.
        data: JSON-compatible evidence.
    Returns:
        None.
    Raises:
        OSError: Destination is not writable.
    """
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

"""Recover completed Physis stages and retain original artifacts when publication fails."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from npa.workflows.physis_lang_artifacts import materialize, publish, write_json


def reuse_completed(args, temporary: Path) -> bool:
    """Verify a completed destination before repeating expensive stage execution.

    Args:
        args: Original stage arguments, including the run-scoped destination.
        temporary: Disposable directory for storage readback.
    Returns:
        True only for complete artifacts from the same invocation.
    Raises:
        ValueError: Existing artifacts are incomplete or belong to another request.
        OSError: Local readback fails; storage errors also propagate.
    """
    destination = args.output_path
    existing = Path(destination)
    if destination.startswith("s3://"):
        from npa.clients.storage import StorageClient

        existing = temporary / "existing"
        StorageClient.from_environment().download_directory(destination, str(existing))
    if not existing.exists() or not any(existing.iterdir()):
        return False
    if not (existing / "checksums.json").is_file():
        raise ValueError(
            "Incomplete Physis destination; publish retained output before retrying"
        )
    materialize(str(existing), existing)
    receipt = existing / "stage.json"
    if not receipt.is_file() or json.loads(receipt.read_text()) != vars(args):
        raise ValueError("Physis destination belongs to a different stage request")
    return True


def _failure_bundle(args, root: Path, error: Exception) -> Path:
    failure = root / "failure"
    failure.mkdir()
    (root / "output").rename(failure / "output")
    write_json(
        failure / "failure.json",
        {"status": "failed", "stage": args.stage, "error_type": type(error).__name__},
    )
    return failure


def preserve_failure(args, root: Path, error: Exception) -> bool:
    """Publish failure evidence, retaining local output if storage also fails.

    Args:
        args: Stage arguments with the original output destination.
        root: Private working directory containing original stage outputs.
        error: Original stage or publication exception.
    Returns:
        Whether the caller must retain the working directory for recovery.
    Raises:
        None; reporting failures never replace the original exception.
    """
    output = root / "output"
    if not output.exists():
        return False
    destination = args.output_path.rstrip("/") + "-failed/" + root.name + "/"
    try:
        failure = _failure_bundle(args, root, error)
        publish(failure, destination)
    except Exception:
        print(
            f"Physis failure evidence retained at {root}", file=sys.stderr, flush=True
        )
        return True
    print(
        f"Physis failure evidence verified at {destination}",
        file=sys.stderr,
        flush=True,
    )
    return False

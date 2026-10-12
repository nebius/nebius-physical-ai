"""Executable wrong-source control for NCore COLMAP qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from npa.errors import NpaError
from npa.workbench.nurec.colmap import (
    ColmapConversionRequest,
    NcoreConversionError,
    convert_colmap,
)
from npa.workbench.nurec.s3_probe import _prefix, _snapshot


CONTROL_FORMAT = "npa_ncore_wrong_source_control_v1"


class NcoreSourceControlError(NpaError):
    """The pre-conversion wrong-source control was not proven."""


def _wrong_sha256(expected: str) -> str:
    if len(expected) != 64 or any(
        value not in "0123456789abcdef" for value in expected
    ):
        raise NcoreSourceControlError("expected source SHA-256 is invalid")
    replacement = "0" if expected[0] != "0" else "1"
    return replacement + expected[1:]


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


def run_wrong_source_control(
    request: ColmapConversionRequest,
    *,
    expected_archive_sha256: str,
    receipt_path: Path,
    storage_client: Any = None,
    converter: Any = convert_colmap,
) -> dict[str, Any]:
    """Require digest rejection before extraction/native execution and zero output."""
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    bucket, prefix = _prefix(request.output_path.rstrip("/") + "/")
    before = _snapshot(client, bucket, prefix)
    if before:
        raise NcoreSourceControlError("wrong-source output prefix is not fresh")
    wrong = _wrong_sha256(expected_archive_sha256)
    controlled = request.model_copy(update={"expected_archive_sha256": wrong})
    try:
        converter(controlled, storage_client=client)
    except NcoreConversionError as exc:
        if exc.phase != "source_digest_pre_extract":
            raise NcoreSourceControlError(
                "converter failed outside the pre-extraction digest gate"
            ) from exc
    else:
        raise NcoreSourceControlError("wrong source digest was accepted")
    after = _snapshot(client, bucket, prefix)
    if after:
        raise NcoreSourceControlError("wrong-source control produced output objects")
    receipt = {
        "format": CONTROL_FORMAT,
        "status": "pass",
        "input_uri_sha256": hashlib.sha256(request.input_path.encode()).hexdigest(),
        "output_scope_sha256": hashlib.sha256(request.output_path.encode()).hexdigest(),
        "expected_archive_sha256": expected_archive_sha256,
        "supplied_wrong_sha256": wrong,
        "failure_phase": "source_digest_pre_extract",
        "native_started": False,
        "before_output_objects": 0,
        "after_output_objects": 0,
        "before_inventory_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "after_inventory_sha256": hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _write_private(receipt_path, receipt)
    return receipt

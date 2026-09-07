"""Stage inference inputs inside the service's explicit S3 read boundary."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.workbench.storage_scope import (
    AuthorizedUri,
    StorageAuthorizationError,
    StorageScope,
    authorize_uri,
)

_INPUT_FIELDS = {
    ("vision_path",), ("sound_path",), ("action_path",), ("prompt_path",),
    ("negative_prompt_file",),
    *((kind, "control_path") for kind in ("edge", "blur", "depth", "seg", "wsm")),
}


def _download_uri(target: AuthorizedUri) -> str:
    """Refuse decoded keys that the storage URI parser would reinterpret."""
    uri = f"s3://{target.bucket}/{target.key}"
    parsed = urlparse(uri)
    if (
        parsed.netloc != target.bucket
        or parsed.path.lstrip("/") != target.key
        or parsed.query
        or parsed.fragment
    ):
        raise StorageAuthorizationError("conditioning S3 key cannot be downloaded exactly")
    return uri


def validate_sample_s3_keys(raw: dict[str, Any]) -> None:
    """Check the shared staging key contract before a client submits inference."""
    for field in _INPUT_FIELDS:
        value: Any = raw
        for part in field:
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, str) and value.startswith("s3://"):
            _download_uri(authorize_uri(value, operation="read"))


def stage_sample_inputs(
    raw: dict[str, Any], destination: Path, *, scope: StorageScope,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Authorize every input before allowing upstream validators to touch it.

    HTTP URLs and server-local paths are not an inference capability. Operators
    stage media into the service's allowed S3 roots; only those exact objects
    are downloaded into a fresh request directory. Defaults files are refused
    because upstream merges their contents into the sample after validation.
    """
    staged = copy.deepcopy(raw)
    inputs: list[tuple[dict[str, Any], str, Any]] = []

    def inspect(value: Any, location: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                field = (*location, key)
                if key in {"defaults_file", "output_dir"} and child is not None:
                    raise StorageAuthorizationError("sample filesystem configuration is server-owned")
                if field in _INPUT_FIELDS and child is not None:
                    if not isinstance(child, str) or not child.startswith("s3://"):
                        raise StorageAuthorizationError("sample inputs require an authorized s3:// object")
                    target = scope.authorize(child, operation="read")
                    if not target.key or child.endswith("/"):
                        raise StorageAuthorizationError("sample inputs must identify a single S3 object")
                    _download_uri(target)
                    inputs.append((value, key, target))
                elif key.endswith(("_path", "_file", "_dir", "_url")) and child is not None:
                    raise StorageAuthorizationError("unsupported sample file input")
                else:
                    inspect(child, field)
        elif isinstance(value, list):
            for child in value:
                inspect(child, location)

    inspect(staged)
    if not inputs:
        return staged
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    client = storage_client or StorageClient.from_environment()
    for ordinal, (container, key, target) in enumerate(inputs):
        suffix = Path(target.key).suffix
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix):
            suffix = ".bin"
        local = destination / f"{ordinal}{suffix}"
        client.download_file(_download_uri(target), str(local))
        container[key] = str(local)
    return staged

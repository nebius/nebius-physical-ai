"""Read and write LTX-2.5 provenance artifacts on S3 or a local path.

The provenance manifest has to survive the hop between two containers that
never share a filesystem: the GPU pod that generates video writes it, and a
later state in the same workflow reads it back before a trainer is allowed near
the artifacts. That makes S3 the transport, and it makes "the manifest could not
be read" a case the gate has to be able to distinguish from "the manifest says
no" — see :func:`load_manifest`, which returns ``None`` rather than inventing an
empty manifest that would look permissive to a careless caller.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MANIFEST_FILENAME = "ltx2_provenance.json"
GATE_REPORT_FILENAME = "ltx2_gate.json"
#: What the GPU container itself wrote, via `ltx-runtime provenance`. This is the
#: declaration the generation actually ran under, as opposed to whatever a later
#: CPU state happens to have in its own environment.
DECLARATION_FILENAME = "ltx2_5_declaration.json"


def _storage() -> Any:
    # Deferred: a run over local paths must not need object-storage credentials.
    from npa.clients.storage import LazyStorageClient

    return LazyStorageClient()


def is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


def local_path(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def resolve_uri(uri: str, *, filename: str) -> str:
    """Append *filename* when *uri* names a directory or prefix rather than a file."""

    if uri.endswith("/"):
        return uri + filename
    if Path(local_path(uri)).suffix == ".json":
        return uri
    return uri.rstrip("/") + "/" + filename


def load_manifest(
    uri: str, *, storage: Any | None = None, filename: str = MANIFEST_FILENAME
) -> Any | None:
    """Return the JSON document at *uri*, or ``None`` if it cannot be read.

    Every failure — absent object, unreadable object, malformed JSON — collapses
    to ``None`` on purpose. The gate treats ``None`` as a refusal, so the caller
    never has to decide which read errors are safe to ignore.

    *filename* is what a prefix resolves to; the run's own declaration is written
    under a different name than the manifest the gate later reads.
    """

    resolved = resolve_uri(uri, filename=filename)
    if not is_remote(resolved):
        path = Path(local_path(resolved))
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    store = storage if storage is not None else _storage()
    bucket, key = _split(resolved)
    try:
        body = store.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def write_json(
    payload: dict[str, Any], uri: str, *, filename: str, storage: Any | None = None
) -> str:
    """Write *payload* to *uri* (S3 or local) and return the destination URI."""

    resolved = resolve_uri(uri, filename=filename)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if is_remote(resolved):
        store = storage if storage is not None else _storage()
        with tempfile.TemporaryDirectory(prefix="npa-ltx2-") as tmp:
            local = Path(tmp) / Path(_split(resolved)[1]).name
            local.write_text(body, encoding="utf-8")
            return str(store.upload_file(str(local), resolved))
    path = Path(local_path(resolved))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


__all__ = [
    "DECLARATION_FILENAME",
    "GATE_REPORT_FILENAME",
    "MANIFEST_FILENAME",
    "is_remote",
    "load_manifest",
    "local_path",
    "resolve_uri",
    "write_json",
]

"""Immutable, concurrency-safe runtime cache for the OpenPI Polaris checkpoint.

The public policy image contains this bootstrap but never the checkpoint.  The
checkpoint is a directory of public GCS objects whose *generations* are the
immutable revision.  A mutable ``gs://`` prefix is therefore accepted only when
its canonical generation manifest matches the pinned digest below.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CACHE_FORMAT = "npa.openpi.gcs-generation-cache.v1"
OBJECT_CACHE_FORMAT = "npa.openpi.gcs-object-cache.v1"
OPENPI_TERMS_ENV = "NPA_OPENPI_ACCEPT_GEMMA_TERMS"
OPENPI_TERMS_ACCEPTED_VALUE = "YES"
PROVIDER = "gcs"
BUCKET = "openpi-assets"
OBJECT_PREFIX = "checkpoints/polaris/pi05_droid_jointpos_polaris/"
ARTIFACT = "checkpoints/polaris/pi05_droid_jointpos_polaris"
EXPECTED_MANIFEST_SHA256 = (
    "8b97388a0bbe5913210255d9e77c8ff925562ec7b40213b4525226c6d5885218"
)
EXPECTED_OBJECT_COUNT = 27
EXPECTED_TOTAL_SIZE = 12_434_530_837
TOKENIZER_BUCKET = "big_vision"
TOKENIZER_OBJECT = "paligemma_tokenizer.model"
TOKENIZER_GENERATION = "1711547605575873"
TOKENIZER_SIZE = 4_264_023
TOKENIZER_MD5 = "FCCtyYVnIKVZ6KhyhLGV4g=="
TOKENIZER_CRC32C = "cKDEzw=="
TOKENIZER_MANIFEST_SHA256 = (
    "1e0c7ffebf2ed3b8e861f008d0aa9fa6304b39ac43254a14d332e0f992ad661b"
)
DEFAULT_CACHE_ROOT = "/opt/npa-model-cache/openpi"
OPENPI_DATA_DIRNAME = "openpi-data"
READY_MARKER = ".npa-ready.json"


class OpenPICacheError(RuntimeError):
    """Raised when upstream access or immutable cache verification fails."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_terms(environ: Mapping[str, str] | None = None) -> None:
    source = os.environ if environ is None else environ
    if source.get(OPENPI_TERMS_ENV) != OPENPI_TERMS_ACCEPTED_VALUE:
        raise OpenPICacheError(
            "OpenPI cache population requires the exact run-scoped Gemma terms acceptance"
        )


def _manifest_url(page_token: str = "") -> str:
    query = {"prefix": OBJECT_PREFIX, "maxResults": "1000"}
    if page_token:
        query["pageToken"] = page_token
    return f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urlencode(query)}"


def _tokenizer_metadata_url() -> str:
    encoded = TOKENIZER_OBJECT.replace("/", "%2F")
    return f"https://storage.googleapis.com/storage/v1/b/{TOKENIZER_BUCKET}/o/{encoded}"


def _default_read_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "npa-openpi-cache/1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - pinned public GCS API
        value = json.load(response)
    if not isinstance(value, dict):
        raise OpenPICacheError("upstream GCS metadata response is not an object")
    return value


def fetch_generation_manifest(
    read_json: Callable[[str], Mapping[str, Any]] = _default_read_json,
) -> list[dict[str, object]]:
    """Fetch and normalize the exact GCS generation manifest."""

    page_token = ""
    records: list[dict[str, object]] = []
    while True:
        payload = read_json(_manifest_url(page_token))
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise OpenPICacheError("upstream GCS metadata has an invalid items field")
        for raw in items:
            if not isinstance(raw, Mapping):
                raise OpenPICacheError(
                    "upstream GCS metadata contains a malformed object"
                )
            name = str(raw.get("name", ""))
            generation = str(raw.get("generation", ""))
            md5_hash = str(raw.get("md5Hash", ""))
            crc32c = str(raw.get("crc32c", ""))
            if not name.startswith(OBJECT_PREFIX) or not generation or not md5_hash:
                raise OpenPICacheError(
                    "upstream checkpoint object metadata is incomplete"
                )
            records.append(
                {
                    "name": name,
                    "generation": generation,
                    "size": int(raw.get("size", -1)),
                    "md5Hash": md5_hash,
                    "crc32c": crc32c,
                }
            )
        page_token = str(payload.get("nextPageToken", ""))
        if not page_token:
            break
    records.sort(key=lambda item: str(item["name"]))
    return records


def fetch_tokenizer_record(
    read_json: Callable[[str], Mapping[str, Any]] = _default_read_json,
) -> dict[str, object]:
    """Fetch the exact tokenizer object's immutable GCS generation metadata."""

    raw = read_json(_tokenizer_metadata_url())
    record = {
        "name": str(raw.get("name", "")),
        "generation": str(raw.get("generation", "")),
        "size": int(raw.get("size", -1)),
        "md5Hash": str(raw.get("md5Hash", "")),
        "crc32c": str(raw.get("crc32c", "")),
    }
    verify_tokenizer_record(record)
    return record


def manifest_sha256(records: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_json(list(records))).hexdigest()


def verify_upstream_manifest(records: Sequence[Mapping[str, object]]) -> None:
    digest = manifest_sha256(records)
    total_size = sum(int(item["size"]) for item in records)
    if (
        digest != EXPECTED_MANIFEST_SHA256
        or len(records) != EXPECTED_OBJECT_COUNT
        or total_size != EXPECTED_TOTAL_SIZE
    ):
        raise OpenPICacheError(
            "upstream OpenPI checkpoint revision does not match the pinned generation manifest"
        )


def verify_tokenizer_record(record: Mapping[str, object]) -> None:
    digest = manifest_sha256([record])
    if (
        record.get("name") != TOKENIZER_OBJECT
        or record.get("generation") != TOKENIZER_GENERATION
        or int(record.get("size", -1)) != TOKENIZER_SIZE
        or record.get("md5Hash") != TOKENIZER_MD5
        or record.get("crc32c") != TOKENIZER_CRC32C
        or digest != TOKENIZER_MANIFEST_SHA256
    ):
        raise OpenPICacheError(
            "upstream PaliGemma tokenizer does not match the pinned GCS generation"
        )


def cache_identity_root(cache_root: str | Path) -> Path:
    """Return provider/artifact/revision/format-keyed immutable directory."""

    return (
        Path(cache_root)
        / PROVIDER
        / BUCKET
        / ARTIFACT
        / f"generation-manifest-sha256-{EXPECTED_MANIFEST_SHA256}"
        / "v1"
    )


def checkpoint_path(cache_root: str | Path) -> Path:
    return cache_identity_root(cache_root) / "checkpoint"


def tokenizer_identity_root(cache_root: str | Path) -> Path:
    return (
        openpi_data_home(cache_root)
        / ".npa-identities"
        / PROVIDER
        / TOKENIZER_BUCKET
        / TOKENIZER_OBJECT
        / f"generation-{TOKENIZER_GENERATION}"
        / "v1"
    )


def tokenizer_object_path(cache_root: str | Path) -> Path:
    return tokenizer_identity_root(cache_root) / "object"


def tokenizer_alias_path(cache_root: str | Path) -> Path:
    """Path OpenPI's upstream downloader derives from its immutable gs:// URL."""

    return openpi_data_home(cache_root) / TOKENIZER_BUCKET / TOKENIZER_OBJECT


def checkpoint_assets_alias_path(cache_root: str | Path) -> Path:
    """Path OpenPI derives when resolving the checkpoint's normalization assets."""

    return openpi_data_home(cache_root) / BUCKET / ARTIFACT / "assets"


def checkpoint_assets_identity_root(cache_root: str | Path) -> Path:
    return (
        openpi_data_home(cache_root)
        / ".npa-identities"
        / PROVIDER
        / BUCKET
        / ARTIFACT
        / f"generation-manifest-sha256-{EXPECTED_MANIFEST_SHA256}"
        / "v1"
    )


def checkpoint_assets_path(cache_root: str | Path) -> Path:
    return checkpoint_assets_identity_root(cache_root) / "assets"


def openpi_data_home(cache_root: str | Path) -> Path:
    return Path(cache_root) / OPENPI_DATA_DIRNAME


def _ensure_relative_alias(alias: Path, target: Path, *, description: str) -> None:
    """Publish one fail-closed alias whose target stays within the mounted cache."""

    if not target.is_dir() and not target.is_file():
        raise OpenPICacheError(f"{description} target is missing")
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, alias.parent)
    if os.path.lexists(alias):
        if not alias.is_symlink() or os.readlink(alias) != relative_target:
            raise OpenPICacheError(
                f"refusing to overwrite an existing {description} alias"
            )
        return
    temporary_alias = alias.with_name(
        f".{alias.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    os.symlink(relative_target, temporary_alias)
    os.rename(temporary_alias, alias)


def _relative_path(record: Mapping[str, object]) -> Path:
    name = str(record["name"])
    relative = Path(name.removeprefix(OBJECT_PREFIX))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise OpenPICacheError("upstream checkpoint contains an unsafe object name")
    return relative


def _file_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _verify_identity(
    identity: Path,
    records: Sequence[Mapping[str, object]],
    *,
    require_ready: bool = True,
) -> Path:
    """Verify marker, exact file set, sizes, and every upstream MD5 checksum."""

    verify_upstream_manifest(records)
    marker = identity / READY_MARKER
    if require_ready:
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OpenPICacheError("OpenPI cache has no valid ready marker") from exc
        expected_marker = {
            "format": CACHE_FORMAT,
            "provider": PROVIDER,
            "bucket": BUCKET,
            "artifact": ARTIFACT,
            "revision": EXPECTED_MANIFEST_SHA256,
            "object_count": EXPECTED_OBJECT_COUNT,
            "total_size_bytes": EXPECTED_TOTAL_SIZE,
        }
        if metadata != expected_marker:
            raise OpenPICacheError(
                "OpenPI cache ready marker does not match its identity"
            )
    checkpoint = identity / "checkpoint"
    expected_paths = {_relative_path(item) for item in records}
    actual_paths = (
        {
            path.relative_to(checkpoint)
            for path in checkpoint.rglob("*")
            if path.is_file()
        }
        if checkpoint.is_dir()
        else set()
    )
    if actual_paths != expected_paths:
        raise OpenPICacheError("OpenPI cache file set is incomplete or unexpected")
    for record in records:
        path = checkpoint / _relative_path(record)
        if path.stat().st_size != int(record["size"]):
            raise OpenPICacheError("OpenPI cache object size mismatch")
        if _file_md5_base64(path) != str(record["md5Hash"]):
            raise OpenPICacheError("OpenPI cache object checksum mismatch")
    return checkpoint


def verify_cache(
    cache_root: str | Path,
    records: Sequence[Mapping[str, object]],
    *,
    require_ready: bool = True,
) -> Path:
    """Verify the pinned immutable identity below a cache root."""

    return _verify_identity(
        cache_identity_root(cache_root), records, require_ready=require_ready
    )


def verify_tokenizer_cache(
    cache_root: str | Path, record: Mapping[str, object]
) -> Path:
    verify_tokenizer_record(record)
    identity = tokenizer_identity_root(cache_root)
    marker = identity / READY_MARKER
    expected_marker = {
        "format": OBJECT_CACHE_FORMAT,
        "provider": PROVIDER,
        "bucket": TOKENIZER_BUCKET,
        "artifact": TOKENIZER_OBJECT,
        "revision": TOKENIZER_GENERATION,
        "manifest_sha256": TOKENIZER_MANIFEST_SHA256,
        "size_bytes": TOKENIZER_SIZE,
    }
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenPICacheError(
            "PaliGemma tokenizer cache has no valid ready marker"
        ) from exc
    if metadata != expected_marker:
        raise OpenPICacheError(
            "PaliGemma tokenizer ready marker does not match its identity"
        )
    object_path = tokenizer_object_path(cache_root)
    if not object_path.is_file() or object_path.stat().st_size != TOKENIZER_SIZE:
        raise OpenPICacheError("PaliGemma tokenizer cache object size mismatch")
    if _file_md5_base64(object_path) != TOKENIZER_MD5:
        raise OpenPICacheError("PaliGemma tokenizer cache object checksum mismatch")
    alias = tokenizer_alias_path(cache_root)
    if not alias.is_symlink() or alias.resolve() != object_path.resolve():
        raise OpenPICacheError(
            "PaliGemma tokenizer cache alias is missing or mismatched"
        )
    return object_path


def verify_runtime_cache(
    cache_root: str | Path,
    records: Sequence[Mapping[str, object]],
    tokenizer_record: Mapping[str, object],
) -> Path:
    checkpoint = verify_cache(cache_root, records)
    verify_tokenizer_cache(cache_root, tokenizer_record)
    assets = verify_checkpoint_assets_cache(cache_root, records)
    alias = checkpoint_assets_alias_path(cache_root)
    if not alias.is_symlink() or alias.resolve() != assets.resolve():
        raise OpenPICacheError(
            "OpenPI checkpoint normalization-assets alias is missing or mismatched"
        )
    return checkpoint


def _checkpoint_assets_records(
    records: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    selected = [
        record for record in records if _relative_path(record).parts[0] == "assets"
    ]
    if not selected:
        raise OpenPICacheError("OpenPI checkpoint contains no normalization assets")
    return selected


def verify_checkpoint_assets_cache(
    cache_root: str | Path, records: Sequence[Mapping[str, object]]
) -> Path:
    selected = _checkpoint_assets_records(records)
    identity = checkpoint_assets_identity_root(cache_root)
    marker = identity / READY_MARKER
    expected_marker = {
        "format": CACHE_FORMAT,
        "provider": PROVIDER,
        "bucket": BUCKET,
        "artifact": f"{ARTIFACT}/assets",
        "revision": EXPECTED_MANIFEST_SHA256,
        "object_count": len(selected),
        "total_size_bytes": sum(int(record["size"]) for record in selected),
    }
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenPICacheError(
            "OpenPI normalization-assets cache has no valid ready marker"
        ) from exc
    if metadata != expected_marker:
        raise OpenPICacheError(
            "OpenPI normalization-assets ready marker does not match its identity"
        )
    assets = checkpoint_assets_path(cache_root)
    expected_paths = {Path(*_relative_path(record).parts[1:]) for record in selected}
    actual_paths = (
        {path.relative_to(assets) for path in assets.rglob("*") if path.is_file()}
        if assets.is_dir()
        else set()
    )
    if actual_paths != expected_paths:
        raise OpenPICacheError(
            "OpenPI normalization-assets file set is incomplete or unexpected"
        )
    for record in selected:
        path = assets / Path(*_relative_path(record).parts[1:])
        if path.stat().st_size != int(record["size"]):
            raise OpenPICacheError("OpenPI normalization-assets size mismatch")
        if _file_md5_base64(path) != str(record["md5Hash"]):
            raise OpenPICacheError("OpenPI normalization-assets checksum mismatch")
    return assets


def _download_url(record: Mapping[str, object], *, bucket: str = BUCKET) -> str:
    encoded_name = str(record["name"]).replace("/", "%2F")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/"
        f"{encoded_name}?alt=media&generation={record['generation']}"
    )


def _default_download(record: Mapping[str, object], destination: Path) -> None:
    request = Request(
        _download_url(record), headers={"User-Agent": "npa-openpi-cache/1"}
    )
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def _default_tokenizer_download(
    record: Mapping[str, object], destination: Path
) -> None:
    request = Request(
        _download_url(record, bucket=TOKENIZER_BUCKET),
        headers={"User-Agent": "npa-openpi-cache/1"},
    )
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def _populate_tokenizer(
    cache_root: Path,
    record: Mapping[str, object],
    download: Callable[[Mapping[str, object], Path], None],
) -> bool:
    try:
        verify_tokenizer_cache(cache_root, record)
        return False
    except OpenPICacheError:
        pass
    identity = tokenizer_identity_root(cache_root)
    identity.parent.mkdir(parents=True, exist_ok=True)
    if identity.exists():
        shutil.rmtree(identity)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{identity.name}.tmp-", dir=identity.parent)
    )
    try:
        destination = temporary / "object"
        download(record, destination)
        if destination.stat().st_size != TOKENIZER_SIZE:
            raise OpenPICacheError("downloaded PaliGemma tokenizer size mismatch")
        if _file_md5_base64(destination) != TOKENIZER_MD5:
            raise OpenPICacheError("downloaded PaliGemma tokenizer checksum mismatch")
        marker = {
            "format": OBJECT_CACHE_FORMAT,
            "provider": PROVIDER,
            "bucket": TOKENIZER_BUCKET,
            "artifact": TOKENIZER_OBJECT,
            "revision": TOKENIZER_GENERATION,
            "manifest_sha256": TOKENIZER_MANIFEST_SHA256,
            "size_bytes": TOKENIZER_SIZE,
        }
        (temporary / READY_MARKER).write_bytes(_canonical_json(marker) + b"\n")
        os.rename(temporary, identity)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _ensure_relative_alias(
        tokenizer_alias_path(cache_root),
        tokenizer_object_path(cache_root),
        description="PaliGemma tokenizer cache",
    )
    verify_tokenizer_cache(cache_root, record)
    return True


def _populate_checkpoint_assets(
    cache_root: Path, records: Sequence[Mapping[str, object]]
) -> bool:
    try:
        verify_checkpoint_assets_cache(cache_root, records)
        return False
    except OpenPICacheError:
        pass
    selected = _checkpoint_assets_records(records)
    identity = checkpoint_assets_identity_root(cache_root)
    identity.parent.mkdir(parents=True, exist_ok=True)
    if identity.exists():
        shutil.rmtree(identity)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{identity.name}.tmp-", dir=identity.parent)
    )
    try:
        destination_root = temporary / "assets"
        source_root = checkpoint_path(cache_root)
        for record in selected:
            relative = _relative_path(record)
            source = source_root / relative
            destination = destination_root / Path(*relative.parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        marker = {
            "format": CACHE_FORMAT,
            "provider": PROVIDER,
            "bucket": BUCKET,
            "artifact": f"{ARTIFACT}/assets",
            "revision": EXPECTED_MANIFEST_SHA256,
            "object_count": len(selected),
            "total_size_bytes": sum(int(record["size"]) for record in selected),
        }
        (temporary / READY_MARKER).write_bytes(_canonical_json(marker) + b"\n")
        published_assets = temporary / "assets"
        for record in selected:
            relative = _relative_path(record)
            path = published_assets / Path(*relative.parts[1:])
            if path.stat().st_size != int(record["size"]):
                raise OpenPICacheError(
                    "copied OpenPI normalization-assets size mismatch"
                )
            if _file_md5_base64(path) != str(record["md5Hash"]):
                raise OpenPICacheError(
                    "copied OpenPI normalization-assets checksum mismatch"
                )
        os.rename(temporary, identity)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_checkpoint_assets_cache(cache_root, records)
    return True


def populate_cache(
    cache_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    read_json: Callable[[str], Mapping[str, Any]] = _default_read_json,
    download: Callable[[Mapping[str, object], Path], None] = _default_download,
    tokenizer_download: Callable[
        [Mapping[str, object], Path], None
    ] = _default_tokenizer_download,
) -> tuple[Path, bool]:
    """Populate once under an advisory lock and atomically publish the cache."""

    _require_terms(environ)
    records = fetch_generation_manifest(read_json)
    verify_upstream_manifest(records)
    tokenizer_record = fetch_tokenizer_record(read_json)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    # OpenPI calls chmod on OPENPI_DATA_HOME even for a fully warm cache.  Set
    # the requested mode on a warmer-owned child while the init container has a
    # RW mount. CSI volume roots are commonly group-writable but not owned by
    # the pod user, so changing the root mode would incorrectly require root.
    data_home = openpi_data_home(root)
    data_home.mkdir(parents=True, exist_ok=True)
    data_home.chmod(0o777)
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{EXPECTED_MANIFEST_SHA256}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        populated = False
        try:
            verify_cache(root, records)
        except OpenPICacheError:
            identity = cache_identity_root(root)
            identity.parent.mkdir(parents=True, exist_ok=True)
            if identity.exists():
                shutil.rmtree(identity)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{identity.name}.tmp-", dir=identity.parent)
            )
            try:
                target = temporary / "checkpoint"
                for record in records:
                    destination = target / _relative_path(record)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    download(record, destination)
                    if destination.stat().st_size != int(record["size"]):
                        raise OpenPICacheError("downloaded OpenPI object size mismatch")
                    if _file_md5_base64(destination) != str(record["md5Hash"]):
                        raise OpenPICacheError(
                            "downloaded OpenPI object checksum mismatch"
                        )
                marker = {
                    "format": CACHE_FORMAT,
                    "provider": PROVIDER,
                    "bucket": BUCKET,
                    "artifact": ARTIFACT,
                    "revision": EXPECTED_MANIFEST_SHA256,
                    "object_count": EXPECTED_OBJECT_COUNT,
                    "total_size_bytes": EXPECTED_TOTAL_SIZE,
                }
                (temporary / READY_MARKER).write_bytes(_canonical_json(marker) + b"\n")
                _verify_identity(temporary, records)
                os.rename(temporary, identity)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            verify_cache(root, records)
            populated = True
        populated = (
            _populate_tokenizer(root, tokenizer_record, tokenizer_download) or populated
        )
        populated = _populate_checkpoint_assets(root, records) or populated
        _ensure_relative_alias(
            checkpoint_assets_alias_path(root),
            checkpoint_assets_path(root),
            description="OpenPI checkpoint normalization-assets cache",
        )
        return verify_runtime_cache(root, records, tokenizer_record), populated


def preflight(*, environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Prove terms and exact upstream revision access without downloading weights."""

    _require_terms(environ)
    records = fetch_generation_manifest()
    verify_upstream_manifest(records)
    fetch_tokenizer_record()
    return {
        "status": "passed",
        "provider": PROVIDER,
        "artifact": ARTIFACT,
        "revision": EXPECTED_MANIFEST_SHA256,
        "object_count": EXPECTED_OBJECT_COUNT,
        "total_size_bytes": EXPECTED_TOTAL_SIZE,
        "tokenizer_generation": TOKENIZER_GENERATION,
        "tokenizer_size_bytes": TOKENIZER_SIZE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "warm", "path", "verify"))
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if args.command == "warm":
        path, populated = populate_cache(args.cache_root)
        print(
            json.dumps(
                {"status": "populated" if populated else "reused", "path": str(path)}
            )
        )
        return 0
    records = fetch_generation_manifest()
    verify_upstream_manifest(records)
    tokenizer_record = fetch_tokenizer_record()
    path = verify_runtime_cache(args.cache_root, records, tokenizer_record)
    if args.command == "path":
        print(path)
    else:
        print(json.dumps({"status": "passed", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify Gymnasium-Robotics public corresponding-source delivery evidence."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable

import zstandard

MAX_METADATA_BYTES = 1_048_576
MAX_SOURCE_ARCHIVE_BYTES = 21_474_836_480
MAX_SOURCE_ARCHIVE_EXPANDED_BYTES = 25_769_803_776
MAX_SOURCE_ARCHIVE_MEMBERS = 4_096
READ_CHUNK_BYTES = 1_048_576
TAR_BLOCK_BYTES = 512
SOURCE_MANIFEST_MEMBER = "source-manifest.json"
SOURCE_MANIFEST_SCHEMA = "npa.gymnasium-robotics.corresponding-source-manifest.v1"
# A publication origin must be added here by a separate, evidence-backed review.
# The empty current contract keeps the pre-registration candidate fail closed.
REVIEWED_PUBLIC_DELIVERY_ORIGINS: frozenset[str] = frozenset()
ACCEPTED_RECORD = Path(__file__).with_name("gymnasium_robotics_image_manifest.json")
SOURCE_LOCK = (
    Path(__file__).resolve().parents[3]
    / "docker/workbench/gymnasium-robotics/corresponding-source.lock.json"
)
_ApprovedAddress = tuple[int, str]
_ACCEPTED_RECORD_FIELDS = {
    "format",
    "status",
    "tool",
    "source_revision",
    "image",
    "corresponding_source",
}
_SOURCE_ARTIFACT_FIELDS = {
    "reference",
    "sha256",
    "size_bytes",
    "media_type",
    "anonymous",
    "immutable",
    "contents_manifest_sha256",
}


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
        lock.get("schema")
        == "npa.gymnasium-robotics.baked-corresponding-source-lock.v2",
        "corresponding-source lock schema is unsupported",
    )
    _require(
        lock.get("status") == "complete", "corresponding-source lock is incomplete"
    )
    _require(
        lock.get("public_corresponding_source_delivery") == "accepted-public-immutable",
        "public corresponding-source delivery is not accepted",
    )
    deliveries = lock.get("deliveries")
    _require(
        isinstance(deliveries, list) and len(deliveries) == 1,
        "lock requires one delivery",
    )
    delivery = _mapping(deliveries[0], "locked delivery")
    _sha256_hex(delivery.get("binary_manifest_sha256"), "binary manifest")
    _sha256_hex(delivery.get("build_materials_sha256"), "build materials")
    _sha256_hex(delivery.get("source_manifest_sha256"), "source manifest")
    _require(bool(delivery.get("artifacts")), "locked delivery has no artifacts")
    return delivery


def _resolved_address(answer: Any) -> _ApprovedAddress:
    try:
        family = answer[0]
        raw_address = answer[4][0]
        address = ipaddress.ip_address(str(raw_address).split("%", 1)[0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise CorrespondingSourceError(
            "source artifact hostname resolution is malformed"
        ) from exc
    expected_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    _require(
        family == expected_family,
        "source artifact hostname resolution is malformed",
    )
    return int(family), str(address)


def _globally_routable(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    return (
        parsed.is_global
        and not parsed.is_loopback
        and not parsed.is_private
        and not parsed.is_link_local
        and not parsed.is_reserved
        and not parsed.is_multicast
        and not parsed.is_unspecified
    )


def _public_addresses(
    hostname: str,
    resolver: Callable[..., list[tuple[Any, ...]]],
) -> tuple[_ApprovedAddress, ...]:
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            answers = resolver(hostname, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise CorrespondingSourceError(
                "source artifact hostname resolution failed"
            ) from exc
        _require(bool(answers), "source artifact hostname resolution failed")
        addresses = [_resolved_address(answer) for answer in answers]
    else:
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        addresses = [(int(family), str(literal))]
    _require(
        all(_globally_routable(address) for _, address in addresses),
        "source artifact destination is not globally routable",
    )
    return tuple(dict.fromkeys(addresses))


def _validate_reference(
    reference: Any,
    artifact_sha256: str,
    *,
    reviewed_origins: frozenset[str],
    resolver: Callable[..., list[tuple[Any, ...]]],
) -> tuple[_ApprovedAddress, ...]:
    _require(isinstance(reference, str), "source artifact reference must be a string")
    parsed = urllib.parse.urlsplit(reference)
    _require(parsed.scheme == "https", "source artifact reference must use HTTPS")
    _require(bool(parsed.hostname), "source artifact reference has no host")
    _require(
        not parsed.username and not parsed.password,
        "source artifact reference embeds credentials",
    )
    _require(
        not parsed.query and not parsed.fragment,
        "source artifact reference must be immutable",
    )
    hostname = str(parsed.hostname).lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise CorrespondingSourceError(
            "source artifact reference has an invalid port"
        ) from exc
    _require(port is None, "source artifact reference uses an unsupported port")
    formatted_host = f"[{hostname}]" if ":" in hostname else hostname
    origin = f"https://{formatted_host}"
    _require(
        origin in reviewed_origins,
        "source artifact origin is not in the reviewed public delivery contract",
    )
    approved_addresses = _public_addresses(hostname, resolver)
    _require(
        artifact_sha256 in parsed.path.lower(),
        "source artifact reference is not digest-addressed",
    )
    return approved_addresses


def _validated_record_source(
    record: dict[str, Any],
    lock_bytes: bytes,
    subject: dict[str, str],
) -> dict[str, Any]:
    _exact_fields(record, _ACCEPTED_RECORD_FIELDS, "accepted record")
    _require(
        record["format"] == "npa_gymnasium_robotics_accepted_image_manifest_v1",
        "accepted record format is unsupported",
    )
    _require(
        record["status"] == "accepted-for-publication",
        "publication record is not accepted",
    )
    _require(
        record["tool"] == "gymnasium-robotics",
        "publication record has the wrong subject",
    )
    _require(
        record["source_revision"] == subject["source_revision"],
        "source revision does not match",
    )
    _validate_image(_mapping(record["image"], "image binding"), subject)
    source = _mapping(record["corresponding_source"], "corresponding-source binding")
    _exact_fields(
        source, {"lock_sha256", "delivery", "artifact"}, "corresponding-source binding"
    )
    _require(
        source["lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest(),
        "corresponding-source lock digest does not match",
    )
    return source


def _validate_record(
    record: dict[str, Any],
    lock: dict[str, Any],
    lock_bytes: bytes,
    subject: dict[str, str],
    *,
    reviewed_origins: frozenset[str],
    resolver: Callable[..., list[tuple[Any, ...]]],
) -> tuple[dict[str, Any], tuple[_ApprovedAddress, ...]]:
    source = _validated_record_source(record, lock_bytes, subject)
    delivery = _validate_lock(lock)
    _require(
        source["delivery"] == delivery,
        "corresponding-source delivery does not match the lock",
    )
    artifact = _mapping(source["artifact"], "source artifact")
    approved_addresses = _validate_artifact(
        artifact,
        delivery,
        reviewed_origins=reviewed_origins,
        resolver=resolver,
    )
    return artifact, approved_addresses


def _validate_image(image: dict[str, Any], subject: dict[str, str]) -> None:
    _exact_fields(
        image,
        {"digest", "platform", "platform_manifest_digest", "config_digest"},
        "image binding",
    )
    platform = _mapping(image["platform"], "image platform")
    _exact_fields(platform, {"os", "architecture"}, "image platform")
    _require(
        platform == {"os": "linux", "architecture": "amd64"},
        "image platform is not linux/amd64",
    )
    for field in ("digest", "platform_manifest_digest", "config_digest"):
        _digest(image[field], f"image {field}")
        _require(image[field] == subject[field], f"image {field} does not match")


def _validate_artifact(
    artifact: dict[str, Any],
    delivery: dict[str, Any],
    *,
    reviewed_origins: frozenset[str],
    resolver: Callable[..., list[tuple[Any, ...]]],
) -> tuple[_ApprovedAddress, ...]:
    _exact_fields(artifact, _SOURCE_ARTIFACT_FIELDS, "source artifact")
    artifact_sha256 = _sha256_hex(artifact["sha256"], "source artifact")
    approved_addresses = _validate_reference(
        artifact["reference"],
        artifact_sha256,
        reviewed_origins=reviewed_origins,
        resolver=resolver,
    )
    _require(artifact["anonymous"] is True, "source artifact is not anonymous")
    _require(artifact["immutable"] is True, "source artifact is not immutable")
    _require(
        artifact["media_type"] == "application/zstd",
        "source artifact media type is unsupported",
    )
    size = artifact["size_bytes"]
    _require(
        isinstance(size, int) and 0 < size <= MAX_SOURCE_ARCHIVE_BYTES,
        "source artifact size is invalid",
    )
    _require(
        artifact["contents_manifest_sha256"] == delivery["source_manifest_sha256"],
        "source artifact contents manifest does not match the lock",
    )
    return approved_addresses


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise CorrespondingSourceError("source artifact redirected")


def _connect_approved_address(
    approved_address: _ApprovedAddress,
    port: int,
    timeout: object,
    source_address: tuple[str, int] | None,
) -> socket.socket:
    family, address = approved_address
    connection = socket.socket(family, socket.SOCK_STREAM)
    try:
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            connection.settimeout(timeout)  # type: ignore[arg-type]
        if source_address:
            connection.bind(source_address)
        destination = (
            (address, port, 0, 0)
            if family == socket.AF_INET6
            else (address, port)
        )
        connection.connect(destination)
        return connection
    except OSError:
        connection.close()
        raise


def _connect_approved_addresses(
    approved_addresses: tuple[_ApprovedAddress, ...],
    port: int,
    timeout: object,
    source_address: tuple[str, int] | None,
) -> socket.socket:
    last_error: OSError | None = None
    for approved_address in approved_addresses:
        try:
            return _connect_approved_address(
                approved_address, port, timeout, source_address
            )
        except OSError as exc:
            last_error = exc
    raise OSError("all approved source artifact addresses failed") from last_error


def _close_failed_connection(connection: socket.socket | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        # Cleanup must not replace the connection or TLS failure being reported.
        return


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        approved_addresses: tuple[_ApprovedAddress, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(host, **kwargs)
        self._approved_addresses = approved_addresses
        self._reviewed_destination = (self.host, self.port)
        self._create_connection = self._connect_approved

    def _connect_approved(
        self,
        address: tuple[str, int],
        timeout: object = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        _require(self._tunnel_host is None, "HTTPS tunneling is not allowed")
        _require(
            address == self._reviewed_destination
            and (self.host, self.port) == self._reviewed_destination,
            "HTTPS destination changed",
        )
        return _connect_approved_addresses(
            self._approved_addresses, self.port, timeout, source_address
        )

    def connect(self) -> None:
        try:
            super().connect()
        except (CorrespondingSourceError, OSError, ValueError):
            failed_connection = self.sock
            self.sock = None
            _close_failed_connection(failed_connection)
            raise


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, approved_addresses: tuple[_ApprovedAddress, ...]) -> None:
        super().__init__()
        self._approved_addresses = approved_addresses

    def https_open(self, request: urllib.request.Request) -> BinaryIO:
        def connection(host: str, **kwargs: Any) -> _PinnedHTTPSConnection:
            return _PinnedHTTPSConnection(
                host,
                approved_addresses=self._approved_addresses,
                **kwargs,
            )

        return self.do_open(connection, request, context=self._context)


def _open_anonymous(
    request: urllib.request.Request,
    timeout: float,
    *,
    approved_addresses: tuple[_ApprovedAddress, ...],
) -> BinaryIO:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedHTTPSHandler(approved_addresses),
        _RejectRedirects(),
    )
    return opener.open(request, timeout=timeout)  # noqa: S310


def _tar_number(field: bytes, label: str) -> int:
    value = field.rstrip(b"\0 ").lstrip(b" ")
    _require(
        bool(value) and all(48 <= byte <= 55 for byte in value),
        f"source archive {label} is malformed",
    )
    return int(value, 8)


def _safe_archive_path(value: str, *, directory: bool, label: str) -> str:
    if directory:
        value = value.rstrip("/")
    path = PurePosixPath(value)
    _require(
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and str(path) == value,
        f"{label} path is unsafe",
    )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CorrespondingSourceError(f"{label} path is not UTF-8") from exc
    _require(len(encoded) <= 255, f"{label} path is too long")
    return value


def _tar_path(header: bytes, *, directory: bool) -> str:
    name = header[0:100].split(b"\0", 1)[0]
    prefix = header[345:500].split(b"\0", 1)[0]
    raw = prefix + (b"/" if prefix and name else b"") + name
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorrespondingSourceError(
            "source archive member path is not UTF-8"
        ) from exc
    return _safe_archive_path(
        decoded,
        directory=directory,
        label="source archive member",
    )


def _manifest_files(raw: bytes) -> list[tuple[str, int, str]]:
    _require(len(raw) <= MAX_METADATA_BYTES, "source manifest is too large")
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorrespondingSourceError("source manifest is not valid JSON") from exc
    payload = _mapping(payload, "source manifest")
    _exact_fields(payload, {"schema", "files"}, "source manifest")
    _require(
        payload["schema"] == SOURCE_MANIFEST_SCHEMA,
        "source manifest schema is unsupported",
    )
    files = payload["files"]
    _require(
        isinstance(files, list) and 0 < len(files) < MAX_SOURCE_ARCHIVE_MEMBERS,
        "source manifest file inventory is invalid",
    )
    observed: set[str] = set()
    result: list[tuple[str, int, str]] = []
    for index, value in enumerate(files):
        item = _mapping(value, f"source manifest file {index}")
        _exact_fields(
            item, {"path", "size_bytes", "sha256"}, f"source manifest file {index}"
        )
        path = item["path"]
        _require(isinstance(path, str), "source manifest path must be a string")
        normalized = _safe_archive_path(
            path,
            directory=False,
            label="source manifest member",
        )
        _require(
            normalized != SOURCE_MANIFEST_MEMBER, "source manifest may not list itself"
        )
        folded = normalized.casefold()
        _require(folded not in observed, "source manifest contains a duplicate path")
        observed.add(folded)
        size = item["size_bytes"]
        _require(
            isinstance(size, int) and size >= 0, "source manifest file size is invalid"
        )
        digest = _sha256_hex(item["sha256"], "source manifest file")
        result.append((normalized, size, digest))
    return result


def _read_exact(
    stream: BinaryIO,
    size: int,
    expanded: list[int],
    *,
    allow_eof: bool = False,
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_eof and not chunks:
                return None
            raise CorrespondingSourceError("source archive is truncated")
        expanded[0] += len(chunk)
        _require(
            expanded[0] <= MAX_SOURCE_ARCHIVE_EXPANDED_BYTES,
            "source archive exceeds its expansion limit",
        )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _verify_source_archive(archive: BinaryIO, expected_manifest_sha256: str) -> None:
    expanded = [0]
    seen: set[str] = set()
    observed_files: list[tuple[str, int, str]] = []
    manifest_raw: bytes | None = None
    member_count = 0
    try:
        with zstandard.ZstdDecompressor(max_window_size=64 * 1024 * 1024).stream_reader(
            archive, read_across_frames=True
        ) as decoded:
            while True:
                header = _read_exact(decoded, TAR_BLOCK_BYTES, expanded, allow_eof=True)
                _require(header is not None, "source archive has no TAR terminator")
                if header == b"\0" * TAR_BLOCK_BYTES:
                    second = _read_exact(decoded, TAR_BLOCK_BYTES, expanded)
                    _require(
                        second == b"\0" * TAR_BLOCK_BYTES,
                        "source archive TAR terminator is malformed",
                    )
                    while trailing := decoded.read(READ_CHUNK_BYTES):
                        expanded[0] += len(trailing)
                        _require(
                            expanded[0] <= MAX_SOURCE_ARCHIVE_EXPANDED_BYTES,
                            "source archive exceeds its expansion limit",
                        )
                        _require(
                            not trailing.strip(b"\0"),
                            "source archive has trailing data",
                        )
                    break
                member_count += 1
                _require(
                    member_count <= MAX_SOURCE_ARCHIVE_MEMBERS,
                    "source archive has too many members",
                )
                _require(
                    header[257:263] == b"ustar\0" and header[263:265] == b"00",
                    "source archive TAR format is unsupported",
                )
                expected_checksum = _tar_number(header[148:156], "header checksum")
                actual_checksum = sum(header[:148]) + (32 * 8) + sum(header[156:])
                _require(
                    expected_checksum == actual_checksum,
                    "source archive TAR checksum does not match",
                )
                kind = header[156:157]
                _require(
                    kind in {b"\0", b"0", b"5"},
                    "source archive contains a link or special member",
                )
                directory = kind == b"5"
                name = _tar_path(header, directory=directory)
                folded = name.casefold()
                _require(folded not in seen, "source archive contains a duplicate path")
                seen.add(folded)
                size = _tar_number(header[124:136], "member size")
                _require(
                    not directory or size == 0, "source archive directory has a body"
                )
                digest = hashlib.sha256()
                captured: list[bytes] | None = (
                    [] if name == SOURCE_MANIFEST_MEMBER else None
                )
                remaining = size
                while remaining:
                    chunk_size = min(remaining, READ_CHUNK_BYTES)
                    chunk = _read_exact(decoded, chunk_size, expanded)
                    assert chunk is not None
                    digest.update(chunk)
                    if captured is not None:
                        _require(
                            size <= MAX_METADATA_BYTES, "source manifest is too large"
                        )
                        captured.append(chunk)
                    remaining -= len(chunk)
                padding = (-size) % TAR_BLOCK_BYTES
                if padding:
                    padded = _read_exact(decoded, padding, expanded)
                    _require(
                        padded == b"\0" * padding,
                        "source archive member padding is malformed",
                    )
                if directory:
                    continue
                if captured is not None:
                    _require(
                        manifest_raw is None,
                        "source archive contains multiple source manifests",
                    )
                    manifest_raw = b"".join(captured)
                else:
                    observed_files.append((name, size, digest.hexdigest()))
    except zstandard.ZstdError as exc:
        raise CorrespondingSourceError(
            "source archive Zstandard stream is malformed"
        ) from exc
    _require(manifest_raw is not None, "source archive has no source manifest")
    _require(
        hashlib.sha256(manifest_raw).hexdigest() == expected_manifest_sha256,
        "source archive manifest identity does not match the lock",
    )
    _require(
        _manifest_files(manifest_raw) == observed_files,
        "source archive members do not match the locked source manifest",
    )


def _read_and_verify_download(
    response: BinaryIO,
    archive: BinaryIO,
    artifact: dict[str, Any],
    expected_manifest_sha256: str,
) -> None:
    _require(
        getattr(response, "status", 200) == 200,
        "anonymous source retrieval failed",
    )
    _require(response.geturl() == artifact["reference"], "source artifact redirected")
    expected_size = artifact["size_bytes"]
    digest = hashlib.sha256()
    total = 0
    while chunk := response.read(READ_CHUNK_BYTES):
        total += len(chunk)
        _require(total <= expected_size, "source artifact exceeds accepted size")
        digest.update(chunk)
        archive.write(chunk)
    archive.flush()
    archive.seek(0)
    _require(total == expected_size, "source artifact size does not match")
    _require(
        digest.hexdigest() == artifact["sha256"],
        "source artifact digest does not match",
    )
    _verify_source_archive(archive, expected_manifest_sha256)


def _verify_download(
    artifact: dict[str, Any],
    expected_manifest_sha256: str,
    approved_addresses: tuple[_ApprovedAddress, ...],
    opener: Callable[..., BinaryIO],
) -> None:
    host = urllib.parse.urlsplit(artifact["reference"]).netloc
    request = urllib.request.Request(
        artifact["reference"],
        headers={"Accept": "application/octet-stream", "Host": host},
    )
    try:
        with (
            tempfile.TemporaryFile() as archive,
            opener(
                request,
                timeout=60,
                approved_addresses=approved_addresses,
            ) as response,
        ):
            _read_and_verify_download(
                response,
                archive,
                artifact,
                expected_manifest_sha256,
            )
    except CorrespondingSourceError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise CorrespondingSourceError("anonymous source retrieval failed") from exc


def _image_subject(
    source_revision: str,
    image_digest: str,
    platform_manifest_digest: str,
    config_digest: str,
) -> dict[str, str]:
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", source_revision)),
        "source revision is invalid",
    )
    return {
        "source_revision": source_revision,
        "digest": _digest(image_digest, "image digest"),
        "platform_manifest_digest": _digest(
            platform_manifest_digest, "platform manifest"
        ),
        "config_digest": _digest(config_digest, "image config"),
    }


def _verified_record(
    record_path: Path,
    lock_path: Path,
    subject: dict[str, str],
    opener: Callable[..., BinaryIO],
) -> dict[str, Any]:
    record, _ = _load_json(record_path, "accepted publication record")
    lock, lock_bytes = _load_json(lock_path, "corresponding-source lock")
    artifact, approved_addresses = _validate_record(
        record,
        lock,
        lock_bytes,
        subject,
        reviewed_origins=REVIEWED_PUBLIC_DELIVERY_ORIGINS,
        resolver=socket.getaddrinfo,
    )
    delivery = _validate_lock(lock)
    _verify_download(
        artifact,
        delivery["source_manifest_sha256"],
        approved_addresses,
        opener,
    )
    return record


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
        record_path: Accepted publication record.
        lock_path: Exact source lock used by the build.
        source_revision: Full represented Git revision.
        image_digest: Immutable image digest.
        platform_manifest_digest: Immutable platform digest.
        config_digest: Immutable configuration digest.
        opener: Address-bound anonymous HTTPS opener, injectable for hermetic tests.

    Returns:
        The validated accepted publication record.

    Raises:
        CorrespondingSourceError: Any record, binding, or retrieval is invalid.
    """
    subject = _image_subject(
        source_revision, image_digest, platform_manifest_digest, config_digest
    )
    return _verified_record(record_path, lock_path, subject, opener)


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

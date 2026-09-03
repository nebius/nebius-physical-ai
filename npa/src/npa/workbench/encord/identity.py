"""Exact identity helpers for Encord storage items.

Display fields are deliberately absent from this module. Identity comes
from namespaced metadata, a complete normalized object URL, or an explicit
operator sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlsplit, urlunsplit

from npa.workbench.encord.schemas import EncordToolError, IdentitySidecarRow


def canonical_s3_uri(bucket: str, key: str) -> str:
    bucket = bucket.strip()
    if not bucket or any(char in bucket for char in "/\\\x00:@?#"):
        raise EncordToolError("S3 bucket must be a nonempty bucket name")
    if not key or key.endswith("/") or "\\" in key or "\x00" in key:
        raise EncordToolError("S3 object key must identify one unambiguous object")
    if key.startswith("/") or any(part in {"", ".", ".."} for part in key.split("/")):
        raise EncordToolError("S3 object key contains an ambiguous path form")
    # The SDK returns literal object keys. Encode the percent sign instead of
    # interpreting percent triplets, so the key ``a%2Fb`` never aliases ``a/b``.
    encoded = quote(key, safe="/~:@!$&'()*+,;=-._")
    return f"s3://{bucket}/{encoded}"


def normalize_object_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EncordToolError("object URL must be an absolute HTTP(S) URL")
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise EncordToolError("object URL contains an invalid port") from exc
    if port:
        host = f"{host}:{port}"
    normalized_path = _normalize_url_path(parsed.path)
    segments = normalized_path.split("/")[1:]
    if (
        not normalized_path.startswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise EncordToolError("object URL contains an ambiguous path")
    return urlunsplit((parsed.scheme.lower(), host, normalized_path, "", ""))


def _normalize_url_path(path: str) -> str:
    """Normalize only unreserved escapes while preserving reserved identity."""

    unreserved = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    normalized: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character == "%":
            if index + 2 >= len(path) or not {
                path[index + 1],
                path[index + 2],
            } <= hexadecimal:
                raise EncordToolError("object URL contains an invalid percent escape")
            value = int(path[index + 1 : index + 3], 16)
            decoded = chr(value)
            if decoded in {"\\", "\x00"}:
                raise EncordToolError("object URL contains an ambiguous path")
            normalized.append(decoded if decoded in unreserved else f"%{value:02X}")
            index += 3
            continue
        if character in {"\\", "\x00"}:
            raise EncordToolError("object URL contains an ambiguous path")
        normalized.append(quote(character, safe="/~:@!$&'()*+,;=-._"))
        index += 1
    return "".join(normalized)


def metadata_identity(item: Any) -> tuple[str, str]:
    raw = getattr(item, "client_metadata", None) or {}
    if not isinstance(raw, Mapping):
        return "", ""
    namespace = raw.get("npa") or {}
    if not isinstance(namespace, Mapping):
        return "", ""
    source_uri = str(namespace.get("source_uri") or "").strip()
    record_id = str(namespace.get("record_id") or "").strip()
    return source_uri, record_id


def object_url_identity(item: Any) -> str:
    for attr in ("object_url", "objectUrl", "url", "file_url"):
        raw = getattr(item, attr, "")
        if raw:
            try:
                return normalize_object_url(str(raw))
            except EncordToolError:
                return ""
    return ""


@dataclass(frozen=True)
class IdentityCandidate:
    item_uuid: str
    source_uri: str = ""
    record_id: str = ""
    object_url: str = ""

    @classmethod
    def from_item(cls, item: Any) -> "IdentityCandidate":
        source_uri, record_id = metadata_identity(item)
        item_uuid = str(
            getattr(item, "uuid", "") or getattr(item, "item_uuid", "") or ""
        ).strip()
        return cls(
            item_uuid=item_uuid,
            source_uri=source_uri,
            record_id=record_id,
            object_url=object_url_identity(item),
        )


@dataclass(frozen=True)
class IdentityResolution:
    item_uuid: str = ""
    signal: str = ""
    error_code: str = ""
    error: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.item_uuid) and not self.error_code


def resolve_exact_identity(
    *,
    source_uri: str,
    record_id: str,
    submitted_object_url: str,
    candidates: Iterable[Any],
    sidecar: IdentitySidecarRow | None = None,
) -> IdentityResolution:
    expected_url = (
        normalize_object_url(submitted_object_url) if submitted_object_url else ""
    )
    views = [IdentityCandidate.from_item(candidate) for candidate in candidates]
    views = [view for view in views if view.item_uuid]

    matched: list[tuple[IdentityCandidate, str]] = []
    for view in views:
        signal = ""
        if record_id and view.record_id == record_id and view.source_uri == source_uri:
            signal = "record_id_metadata"
        elif view.source_uri == source_uri:
            signal = "source_uri_metadata"
        elif expected_url and view.object_url == expected_url:
            signal = "object_url"
        if not signal:
            continue
        matched.append((view, signal))

    if sidecar is not None:
        if sidecar.source_uri != source_uri:
            return IdentityResolution(
                error_code="identity_sidecar_mismatch",
                error="sidecar source URI does not match the discovered object",
            )
        if record_id and sidecar.record_id and sidecar.record_id != record_id:
            return IdentityResolution(
                error_code="identity_sidecar_mismatch",
                error="sidecar record ID does not match the discovered object",
            )
        if sidecar.item_uuid:
            matched.append(
                (
                    IdentityCandidate(
                        item_uuid=sidecar.item_uuid,
                        source_uri=sidecar.source_uri,
                        record_id=sidecar.record_id,
                    ),
                    "sidecar",
                )
            )

    uuids = {view.item_uuid for view, _ in matched}
    conflicts: list[str] = []
    for view in views:
        if view.item_uuid not in uuids:
            continue
        if view.source_uri and view.source_uri != source_uri:
            conflicts.append(f"{view.item_uuid}:source_uri")
        if record_id and view.record_id and view.record_id != record_id:
            conflicts.append(f"{view.item_uuid}:record_id")
        if expected_url and view.object_url and view.object_url != expected_url:
            conflicts.append(f"{view.item_uuid}:object_url")
    if conflicts or len(uuids) > 1:
        detail = ", ".join(sorted(conflicts)) or "multiple exact UUID candidates"
        return IdentityResolution(
            error_code="identity_conflict",
            error=f"exact identity signals conflict: {detail}",
        )
    if not matched:
        return IdentityResolution(
            error_code="identity_unresolved",
            error="no exact metadata, object URL, or sidecar identity matched",
        )
    best_order = {"record_id_metadata": 0, "source_uri_metadata": 1, "object_url": 2, "sidecar": 3}
    view, signal = sorted(matched, key=lambda entry: best_order[entry[1]])[0]
    return IdentityResolution(item_uuid=view.item_uuid, signal=signal)

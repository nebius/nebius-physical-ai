"""Pure committed-generation resolver shared by NPA and the shipped agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


JOURNAL_SCHEMA = "npa.sim2real.mutable_publication.v1"
JOURNAL_SUFFIX = "/reports/.sim2real-publication.json"
REPORT_SUFFIX = "/reports/sim2real-report.json"
RRD_SUFFIX = "/reports/sim2real.rrd"
MCAP_SUFFIX = "/reports/sim2real.mcap"
STAGE14_SUFFIX = "/components/stage_14.json"
RESERVED_SUFFIXES = (
    REPORT_SUFFIX,
    RRD_SUFFIX,
    MCAP_SUFFIX,
    STAGE14_SUFFIX,
)


class PublicationConflict(RuntimeError):
    """Raised when publication state cannot identify one complete generation."""


@dataclass(frozen=True)
class PublicationObject:
    """One canonical target and its immutable committed source."""

    canonical_uri: str
    immutable_uri: str | None
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CommittedPublicationSnapshot:
    """One journal read resolving every reserved target as a unit."""

    root: str
    journal_uri: str
    journaled: bool
    transaction_id: str
    objects: dict[str, PublicationObject]

    def target(self, canonical_uri: str) -> PublicationObject:
        if publication_root_from_canonical_uri(canonical_uri) != self.root:
            raise ValueError("publication target belongs to a different run")
        try:
            return self.objects[canonical_uri]
        except KeyError as exc:
            raise ValueError("publication target is not reserved") from exc

    def resolve(self, canonical_uri: str) -> str | None:
        target = self.target(canonical_uri)
        return target.immutable_uri


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def valid_publication_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value not in {".", ".."}
        and all(char.isascii() and (char.isalnum() or char in "._-") for char in value)
    )


def publication_root_from_lock_uri(lock_uri: str) -> str:
    if not isinstance(lock_uri, str) or not lock_uri.endswith(JOURNAL_SUFFIX):
        raise PublicationConflict(f"publication journal URI is invalid: {lock_uri}")
    root = lock_uri[: -len(JOURNAL_SUFFIX)]
    if not root:
        raise PublicationConflict(f"publication journal URI is invalid: {lock_uri}")
    return root


def publication_root_from_canonical_uri(canonical_uri: str) -> str:
    if not isinstance(canonical_uri, str):
        raise ValueError("publication target is not a reserved canonical URI")
    for suffix in RESERVED_SUFFIXES:
        if canonical_uri.endswith(suffix):
            root = canonical_uri[: -len(suffix)]
            if root:
                return root
    raise ValueError("publication target is not a reserved canonical URI")


def canonical_publication_uri(object_uri: str) -> str:
    """Map a reserved alias or immutable publication object to its alias."""

    try:
        publication_root_from_canonical_uri(object_uri)
    except ValueError:
        pass
    else:
        return object_uri
    if not isinstance(object_uri, str):
        raise ValueError("publication object is not a reserved URI")
    generation_marker = "/reports/generations/"
    if generation_marker in object_uri:
        root, relative = object_uri.split(generation_marker, 1)
        parts = relative.split("/")
        filenames = {
            REPORT_SUFFIX.rsplit("/", 1)[-1],
            RRD_SUFFIX.rsplit("/", 1)[-1],
            MCAP_SUFFIX.rsplit("/", 1)[-1],
        }
        if (
            root
            and len(parts) == 2
            and valid_publication_id(parts[0])
            and parts[1] in filenames
        ):
            return f"{root}/reports/{parts[1]}"
    history_marker = "/components/history/stage_14/"
    if history_marker in object_uri:
        root, leaf = object_uri.split(history_marker, 1)
        if (
            root
            and len(leaf) == 69
            and leaf.endswith(".json")
            and all(char in "0123456789abcdef" for char in leaf[:-5])
        ):
            return f"{root}{STAGE14_SUFFIX}"
    raise ValueError("publication object is not a reserved URI")


def reserved_publication_uris(root: str) -> tuple[str, str, str, str]:
    return tuple(f"{root}{suffix}" for suffix in RESERVED_SUFFIXES)  # type: ignore[return-value]


def _validate_present_object(
    item: dict[str, Any],
    *,
    canonical_uri: str,
    root: str,
) -> str | None:
    digest = item.get("sha256")
    size = item.get("size_bytes")
    immutable_uri = item.get("immutable_uri")
    if (
        set(item) != {"uri", "state", "sha256", "size_bytes", "immutable_uri"}
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or type(size) is not int
        or size <= 0
        or not isinstance(immutable_uri, str)
        or not immutable_uri
        or immutable_uri == canonical_uri
    ):
        raise PublicationConflict(
            f"publication transaction object is invalid: {canonical_uri}"
        )
    if canonical_uri.endswith(STAGE14_SUFFIX):
        prefix = f"{root}/components/history/stage_14/"
        suffix = immutable_uri.removeprefix(prefix)
        valid = (
            immutable_uri.startswith(prefix)
            and len(suffix) == 69
            and suffix.endswith(".json")
            and all(char in "0123456789abcdef" for char in suffix[:-5])
        )
        generation_id = None
    else:
        prefix = f"{root}/reports/generations/"
        generation_path = immutable_uri.removeprefix(prefix)
        parts = generation_path.split("/")
        valid = (
            immutable_uri.startswith(prefix)
            and len(parts) == 2
            and valid_publication_id(parts[0])
            and parts[1] == canonical_uri.rsplit("/", 1)[-1]
        )
        generation_id = parts[0] if valid else None
    if not valid:
        raise PublicationConflict(
            f"publication immutable source is outside its run: {immutable_uri}"
        )
    return generation_id


def parse_publication_journal(payload_bytes: bytes, *, lock_uri: str) -> dict[str, Any]:
    """Decode and validate one complete four-target publication journal."""

    try:
        payload = json.loads(
            payload_bytes,
            object_pairs_hook=_strict_json_object,
        )
    except (TypeError, ValueError) as exc:
        raise PublicationConflict(
            f"publication transaction journal is invalid: {lock_uri}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "transaction_id",
        "attempt_id",
        "state",
        "objects",
    }:
        raise PublicationConflict(
            f"publication transaction journal is invalid: {lock_uri}"
        )
    transaction_id = payload.get("transaction_id")
    attempt_id = payload.get("attempt_id")
    objects = payload.get("objects")
    if (
        payload.get("schema") != JOURNAL_SCHEMA
        or payload.get("state") not in {"publishing", "committed"}
        or not valid_publication_id(transaction_id)
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 32
        or any(char not in "0123456789abcdef" for char in attempt_id)
        or not isinstance(objects, list)
    ):
        raise PublicationConflict(
            f"publication transaction journal is incomplete: {lock_uri}"
        )
    root = publication_root_from_lock_uri(lock_uri)
    reserved = set(reserved_publication_uris(root))
    seen: set[str] = set()
    generation_ids: set[str] = set()
    for item in objects:
        if not isinstance(item, dict):
            raise PublicationConflict(
                f"publication transaction journal is invalid: {lock_uri}"
            )
        uri = item.get("uri")
        if not isinstance(uri, str) or uri not in reserved or uri in seen:
            raise PublicationConflict(
                f"publication transaction journal is invalid: {lock_uri}"
            )
        seen.add(uri)
        if item.get("state") == "absent":
            if set(item) != {"uri", "state"} or not uri.endswith(MCAP_SUFFIX):
                raise PublicationConflict(
                    f"publication transaction journal is invalid: {lock_uri}"
                )
        elif item.get("state") == "present":
            generation_id = _validate_present_object(
                item,
                canonical_uri=uri,
                root=root,
            )
            if generation_id is not None:
                generation_ids.add(generation_id)
        else:
            raise PublicationConflict(
                f"publication transaction journal is invalid: {lock_uri}"
            )
    if len(generation_ids) > 1:
        raise PublicationConflict(
            f"publication transaction mixes immutable generations: {lock_uri}"
        )
    if generation_ids and generation_ids != {str(transaction_id)}:
        raise PublicationConflict(
            "publication transaction identity disagrees with its generation: "
            f"{lock_uri}"
        )
    if seen != reserved:
        raise PublicationConflict(
            f"publication transaction journal is incomplete: {lock_uri}"
        )
    return payload


def resolve_committed_publication(
    read_journal: Callable[[str], bytes | None],
    canonical_uri: str,
) -> CommittedPublicationSnapshot:
    """Read one journal and resolve all reserved aliases from that snapshot."""

    root = publication_root_from_canonical_uri(canonical_uri)
    journal_uri = f"{root}{JOURNAL_SUFFIX}"
    payload_bytes = read_journal(journal_uri)
    canonical_uris = reserved_publication_uris(root)
    if payload_bytes is None:
        return CommittedPublicationSnapshot(
            root=root,
            journal_uri=journal_uri,
            journaled=False,
            transaction_id="",
            objects={
                uri: PublicationObject(uri, uri, "", -1) for uri in canonical_uris
            },
        )
    payload = parse_publication_journal(payload_bytes, lock_uri=journal_uri)
    if payload["state"] != "committed":
        raise PublicationConflict(
            f"publication generation is not committed: {journal_uri}"
        )
    objects: dict[str, PublicationObject] = {}
    for item in payload["objects"]:
        uri = str(item["uri"])
        if item["state"] == "absent":
            objects[uri] = PublicationObject(uri, None, "", 0)
        else:
            objects[uri] = PublicationObject(
                uri,
                str(item["immutable_uri"]),
                str(item["sha256"]),
                int(item["size_bytes"]),
            )
    return CommittedPublicationSnapshot(
        root=root,
        journal_uri=journal_uri,
        journaled=True,
        transaction_id=str(payload["transaction_id"]),
        objects=objects,
    )

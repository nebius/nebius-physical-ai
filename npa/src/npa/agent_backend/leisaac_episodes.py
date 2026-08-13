"""Bounded, S3-only LeIsaac episode discovery and media helpers.

The public agent imports this module directly.  It deliberately accepts only the
dataset prefix attested by the selected LeIsaac session manifest: callers cannot
submit buckets, prefixes, object keys, or filesystem paths.  Every list and body
read has an explicit bound, while media bodies remain streaming S3 objects.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable
from urllib.parse import quote, urlparse


MAX_PAGE_SIZE = 50
MAX_LIST_KEYS = 100
MAX_JSON_BYTES = 1024 * 1024
MAX_TIMELINE_BYTES = 16 * 1024 * 1024
MAX_TIMELINE_ROWS = 5000
MAX_MEDIA_BYTES = 2 * 1024 * 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024
MAX_S3_READ_WORKERS = 8

_EPISODE_ID = re.compile(r"(?:0|[1-9][0-9]{0,8})")
_VERSION_ID = re.compile(r"v[0-9]{6}-[a-f0-9]{32}")
_CAMERA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_FILTER_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:-]{0,127}")
_CURSOR = re.compile(r"[^\x00-\x1f\x7f]{1,4096}")
_SHA256 = re.compile(r"[a-f0-9]{64}")


class EpisodeStoreError(RuntimeError):
    """A safe, status-bearing episode API failure."""

    def __init__(
        self,
        detail: str,
        *,
        status_code: int = 400,
        skippable_legacy: bool = False,
    ):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.skippable_legacy = skippable_legacy


class RangeNotSatisfiable(EpisodeStoreError):
    """The requested HTTP byte range cannot be served."""

    def __init__(self, size: int):
        super().__init__("requested media range is not satisfiable", status_code=416)
        self.size = max(0, int(size))


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    size: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ObjectRef:
    key: str
    sha256: str
    size: int
    name: str


def parse_http_range(value: str, size: int) -> ByteRange | None:
    """Parse one RFC 9110 byte range, including suffix/open-ended forms."""

    total = int(size)
    if total < 0:
        raise ValueError("object size cannot be negative")
    raw = str(value or "").strip()
    if not raw:
        return None
    if not raw.startswith("bytes=") or "," in raw:
        raise RangeNotSatisfiable(total)
    spec = raw[6:]
    if "-" not in spec:
        raise RangeNotSatisfiable(total)
    start_raw, end_raw = spec.split("-", 1)
    if not start_raw:
        if not end_raw.isdigit():
            raise RangeNotSatisfiable(total)
        suffix = int(end_raw)
        if suffix <= 0 or total <= 0:
            raise RangeNotSatisfiable(total)
        start = max(0, total - suffix)
        return ByteRange(start=start, end=total - 1, size=total)
    if not start_raw.isdigit() or (end_raw and not end_raw.isdigit()):
        raise RangeNotSatisfiable(total)
    start = int(start_raw)
    if start >= total:
        raise RangeNotSatisfiable(total)
    end = total - 1 if not end_raw else int(end_raw)
    if end < start:
        raise RangeNotSatisfiable(total)
    return ByteRange(start=start, end=min(end, total - 1), size=total)


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_from_ns(value: Any) -> str:
    try:
        stamp = int(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if stamp <= 0:
        return ""
    try:
        return (
            datetime.fromtimestamp(stamp / 1_000_000_000, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError):
        return ""


def _safe_filter(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if text and not _FILTER_TEXT.fullmatch(text):
        raise EpisodeStoreError(f"invalid {name} filter")
    return text


def _safe_date(value: Any, name: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = _utc(text)
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise EpisodeStoreError(f"invalid {name} filter") from exc
    return parsed


def _read_body(body: Any, limit: int, detail: str) -> bytes:
    try:
        data = body.read(limit + 1)
    except Exception as exc:
        raise EpisodeStoreError(detail, status_code=502) from exc
    if not isinstance(data, (bytes, bytearray)) or len(data) > limit:
        raise EpisodeStoreError(detail, status_code=502)
    return bytes(data)


def _json_body(response: dict[str, Any], limit: int, detail: str) -> dict[str, Any]:
    try:
        body = _read_body(response["Body"], limit, detail)
        payload = json.loads(body)
    except EpisodeStoreError as exc:
        raise EpisodeStoreError(
            detail, status_code=502, skippable_legacy=True
        ) from exc
    except (KeyError, UnicodeDecodeError, ValueError) as exc:
        raise EpisodeStoreError(
            detail, status_code=502, skippable_legacy=True
        ) from exc
    if not isinstance(payload, dict):
        raise EpisodeStoreError(detail, status_code=502, skippable_legacy=True)
    checksum = hashlib.sha256(body).hexdigest()
    metadata = response.get("Metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    declared = str(metadata.get("sha256") or "")
    if declared and (not _SHA256.fullmatch(declared) or declared != checksum):
        raise EpisodeStoreError(detail, status_code=502, skippable_legacy=True)
    # S3-compatible gateways do not all return user metadata on GET.  The JSON
    # body is already read under a strict bound, so retain an independently
    # computed digest for the version/player surface in every environment.
    metadata["sha256"] = checksum
    response["Metadata"] = metadata
    return payload


class EpisodeStore:
    """Read immutable LeIsaac episode commits beneath one attested S3 prefix."""

    def __init__(
        self,
        client: Any,
        dataset_uri: str,
        *,
        allowed_buckets: Iterable[str],
        run_id: str,
    ) -> None:
        parsed = urlparse(str(dataset_uri or ""))
        prefix = parsed.path.strip("/")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not prefix
            or any(part in {"", ".", ".."} for part in prefix.split("/"))
        ):
            raise EpisodeStoreError(
                "selected LeIsaac dataset URI is invalid", status_code=502
            )
        allowed = {str(item) for item in allowed_buckets if str(item)}
        if parsed.netloc not in allowed:
            raise EpisodeStoreError(
                "selected LeIsaac dataset is outside configured agent storage",
                status_code=403,
            )
        self.client = client
        self.bucket = parsed.netloc
        self.prefix = prefix
        self.dataset_uri = f"s3://{self.bucket}/{self.prefix}"
        self.run_id = str(run_id)
        self._episode_listing_key = re.compile(
            re.escape(f"{self.prefix}/commits") + r"/episode-([0-9]{6})\.json"
        )

    def _key(self, suffix: str) -> str:
        normalized = str(suffix or "").strip("/")
        if not normalized or any(
            part in {"", ".", ".."} for part in normalized.split("/")
        ):
            raise EpisodeStoreError("invalid dataset object path")
        return f"{self.prefix}/{normalized}"

    def _get_json(
        self, key: str, detail: str, *, limit: int = MAX_JSON_BYTES
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", None)
            error = response.get("Error", {}) if isinstance(response, dict) else {}
            code = str(error.get("Code") or "") if isinstance(error, dict) else ""
            missing = isinstance(exc, KeyError) or code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }
            raise EpisodeStoreError(
                detail,
                status_code=404 if missing else 502,
                skippable_legacy=missing,
            ) from exc
        return _json_body(response, limit, detail), response

    def _version_listing_entry(
        self, version_prefix: str
    ) -> dict[str, Any] | None:
        version_id = version_prefix.rstrip("/").split("/")[-1]
        if not _VERSION_ID.fullmatch(version_id):
            return None
        try:
            manifest, raw = self._get_json(
                self._key(f"versions/{version_id}/npa-dataset.json"),
                "immutable dataset version manifest is unreadable",
            )
        except EpisodeStoreError as exc:
            if exc.skippable_legacy:
                return None
            raise
        if (
            manifest.get("schema") != "npa.leisaac.dataset.v1"
            or str(manifest.get("version") or "") != version_id
            or str(manifest.get("output_prefix") or "").rstrip("/")
            != self.dataset_uri
        ):
            return None
        metadata_checksum = str((raw.get("Metadata") or {}).get("sha256") or "")
        try:
            episode_count = int(manifest.get("episode_count") or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        return {
            "version_id": version_id,
            "dataset_uri": str(manifest.get("dataset_uri") or ""),
            "created_at": str(manifest.get("created_at") or ""),
            "episode_count": episode_count,
            "lerobot_version": str(manifest.get("lerobot_version") or ""),
            "manifest_checksum": metadata_checksum
            if _SHA256.fullmatch(metadata_checksum)
            else "",
        }

    def _object_ref(self, payload: Any, name: str) -> ObjectRef:
        if not isinstance(payload, dict):
            raise EpisodeStoreError(
                "episode artifact manifest is invalid", status_code=502
            )
        key = str(payload.get("key") or "")
        sha256 = str(payload.get("sha256") or "")
        raw_size = payload.get("bytes")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise EpisodeStoreError(
                "episode artifact manifest is invalid", status_code=502
            )
        size = raw_size
        expected = f"{self.prefix}/episodes/"
        if (
            not key.startswith(expected)
            or any(part in {"", ".", ".."} for part in key.split("/"))
            or not _SHA256.fullmatch(sha256)
            or size < 0
            or size > MAX_MEDIA_BYTES
        ):
            raise EpisodeStoreError(
                "episode artifact escaped the dataset prefix", status_code=502
            )
        return ObjectRef(key=key, sha256=sha256, size=size, name=name)

    @staticmethod
    def _page_size(value: Any) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EpisodeStoreError("invalid page size") from exc
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise EpisodeStoreError(f"page size must be between 1 and {MAX_PAGE_SIZE}")
        return limit

    @staticmethod
    def _cursor(value: Any) -> str:
        cursor = str(value or "")
        if cursor and not _CURSOR.fullmatch(cursor):
            raise EpisodeStoreError("invalid pagination cursor")
        return cursor

    def list_versions(self, *, limit: Any = 20, cursor: Any = "") -> dict[str, Any]:
        page_size = self._page_size(limit)
        continuation = self._cursor(cursor)
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": self._key("versions") + "/",
            "Delimiter": "/",
            "MaxKeys": page_size,
        }
        if continuation:
            kwargs["ContinuationToken"] = continuation
        try:
            response = self.client.list_objects_v2(**kwargs)
        except Exception as exc:
            raise EpisodeStoreError(
                "could not list immutable dataset versions", status_code=502
            ) from exc
        prefixes = [
            str(item.get("Prefix") or "") for item in response.get("CommonPrefixes", [])
        ]
        if not prefixes:
            # Some S3-compatible fakes/gateways omit CommonPrefixes.  Derive at
            # most MaxKeys distinct version roots from the bounded response.
            prefixes = sorted(
                {
                    "/".join(str(item.get("Key") or "").split("/")[:-1]) + "/"
                    for item in response.get("Contents", [])
                    if str(item.get("Key") or "").endswith("/npa-dataset.json")
                }
            )[:page_size]
        selected_prefixes = prefixes[:page_size]
        workers = min(MAX_S3_READ_WORKERS, len(selected_prefixes))
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                entries = executor.map(self._version_listing_entry, selected_prefixes)
                versions = [entry for entry in entries if entry is not None]
        else:
            versions = []
        return {
            "versions": versions,
            "next_cursor": str(response.get("NextContinuationToken") or ""),
            "bounded": True,
            "page_size": page_size,
        }

    def _commit(self, episode_index: int) -> tuple[dict[str, Any], dict[str, Any], str]:
        key = self._key(f"commits/episode-{episode_index:06d}.json")
        commit, response = self._get_json(key, "episode commit was not found")
        if (
            commit.get("schema") != "npa.leisaac.episode-commit.v1"
            or int(commit.get("episode_index", -1)) != episode_index
            or not isinstance(commit.get("metadata"), dict)
            or not isinstance(commit.get("objects"), dict)
        ):
            raise EpisodeStoreError("episode commit is malformed", status_code=502)
        metadata = commit["metadata"]
        if str(metadata.get("run_id") or "") != self.run_id:
            raise EpisodeStoreError(
                "episode does not belong to the selected run", status_code=404
            )
        return commit, response, key

    @staticmethod
    def _provenance(metadata: dict[str, Any]) -> dict[str, str]:
        provenance = metadata.get("provenance")
        raw: dict[str, Any] = provenance if isinstance(provenance, dict) else {}
        robot = str(raw.get("robot") or metadata.get("robot") or "SO101")
        scene = str(
            raw.get("scene") or metadata.get("scene") or metadata.get("task") or ""
        )
        device = str(raw.get("device") or metadata.get("teleop_device") or "keyboard")
        task = str(raw.get("task") or metadata.get("task") or "")
        bundle = str(raw.get("bundle") or metadata.get("bundle") or "stock")
        return {
            "robot": robot,
            "scene": scene,
            "device": device,
            "task": task,
            "bundle": bundle,
        }

    def _summary(self, commit: dict[str, Any], commit_key: str) -> dict[str, Any]:
        metadata = commit["metadata"]
        provenance = self._provenance(metadata)
        return {
            "episode_index": int(commit["episode_index"]),
            "episode_id": str(commit.get("episode_uuid") or ""),
            "task": str(metadata.get("task") or ""),
            "environment_id": str(metadata.get("environment_id") or ""),
            "environment_index": int(metadata.get("environment_index") or 0),
            "outcome": str(metadata.get("outcome") or ""),
            "recorded_at": str(metadata.get("recorded_at") or ""),
            "committed_at": str(commit.get("committed_at") or ""),
            "frame_count": int(metadata.get("frame_count") or 0),
            "fps": int(metadata.get("fps") or 0),
            "robot": provenance["robot"],
            "scene": provenance["scene"],
            "device": provenance["device"],
            "configuration_task": provenance["task"],
            "bundle": provenance["bundle"],
            "commit_uri": f"s3://{self.bucket}/{commit_key}",
        }

    def list_episodes(
        self,
        *,
        limit: Any = 20,
        cursor: Any = "",
        version_id: str = "",
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        page_size = self._page_size(limit)
        continuation = self._cursor(cursor)
        query = filters if isinstance(filters, dict) else {}
        expected = {
            name: _safe_filter(query.get(name), name)
            for name in ("task", "environment", "outcome", "robot", "scene", "device")
        }
        date_from = _safe_date(query.get("date_from"), "date from")
        date_to = _safe_date(query.get("date_to"), "date to")
        if date_from and date_to and date_from > date_to:
            raise EpisodeStoreError("date from must not be after date to")
        version_commit_keys: set[str] | None = None
        version_episode_count: int | None = None
        if version_id:
            version = self._version_manifest(version_id)
            commits = version.get("episode_commits")
            if isinstance(commits, list):
                version_commit_keys = {
                    str(uri).removeprefix(f"s3://{self.bucket}/")
                    for uri in commits
                    if str(uri).startswith(f"s3://{self.bucket}/{self.prefix}/commits/")
                }
            else:
                version_episode_count = int(version["episode_count"])
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": self._key("commits/episode-"),
            # Consume every object in the S3 page before returning its opaque
            # continuation token.  Asking S3 for more than the API page size
            # and then stopping at page_size would silently skip the unread
            # keys because the token advances past the entire S3 response.
            "MaxKeys": min(MAX_LIST_KEYS, page_size),
        }
        if continuation:
            kwargs["ContinuationToken"] = continuation
        try:
            response = self.client.list_objects_v2(**kwargs)
        except Exception as exc:
            raise EpisodeStoreError(
                "could not list immutable episodes", status_code=502
            ) from exc
        candidates: list[tuple[str, int]] = []
        for item in response.get("Contents", []):
            key = str(item.get("Key") or "")
            match = self._episode_listing_key.fullmatch(key)
            if not match:
                continue
            if version_commit_keys is not None and key not in version_commit_keys:
                continue
            if version_episode_count is not None and int(match.group(1)) >= version_episode_count:
                continue
            candidates.append((key, int(match.group(1))))

        def load(candidate: tuple[str, int]) -> tuple[dict[str, Any], str] | None:
            key, index = candidate
            commit, _raw = self._get_json(key, "episode commit is unreadable")
            if (
                commit.get("schema") != "npa.leisaac.episode-commit.v1"
                or int(commit.get("episode_index", -1)) != index
                or not isinstance(commit.get("metadata"), dict)
                or not isinstance(commit.get("objects"), dict)
            ):
                return None
            if str(commit["metadata"].get("run_id") or "") != self.run_id:
                return None
            return self._summary(commit, key), key

        workers = min(MAX_S3_READ_WORKERS, len(candidates))
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = list(executor.map(load, candidates))
        else:
            loaded = []
        episodes: list[dict[str, Any]] = []
        loaded_count = 0
        for entry in loaded:
            if entry is None:
                continue
            loaded_count += 1
            summary, _key = entry
            summary["dataset_version"] = version_id
            values = {
                "task": summary["task"],
                "environment": summary["environment_id"],
                "outcome": summary["outcome"],
                "robot": summary["robot"],
                "scene": summary["scene"],
                "device": summary["device"],
            }
            if any(
                expected[name] and values[name] != expected[name] for name in expected
            ):
                continue
            recorded = _utc(summary["recorded_at"]) or _utc(summary["committed_at"])
            if date_from and (recorded is None or recorded < date_from):
                continue
            if date_to and (recorded is None or recorded > date_to):
                continue
            episodes.append(summary)
            if len(episodes) >= page_size:
                break
        return {
            "episodes": episodes,
            "next_cursor": str(response.get("NextContinuationToken") or ""),
            "source_count": len(candidates),
            "loaded_count": loaded_count,
            "filtered_count": max(0, loaded_count - len(episodes)),
            "skipped_count": max(0, len(candidates) - loaded_count),
            "has_more_pages": bool(response.get("NextContinuationToken")),
            "bounded": True,
            "page_size": page_size,
            "filters": {key: value for key, value in expected.items() if value},
            "dataset_version": version_id,
        }

    def _version_manifest(self, version_id: str) -> dict[str, Any]:
        if not _VERSION_ID.fullmatch(version_id):
            raise EpisodeStoreError("invalid immutable dataset version")
        manifest, _response = self._get_json(
            self._key(f"versions/{version_id}/npa-dataset.json"),
            "immutable dataset version was not found",
        )
        commits = manifest.get("episode_commits")
        parent_linked = manifest.get("index_layout") == "parent-linked-v2"
        if (
            manifest.get("schema") != "npa.leisaac.dataset.v1"
            or str(manifest.get("version") or "") != version_id
            or not (
                (isinstance(commits, list) and len(commits) <= MAX_TIMELINE_ROWS)
                or (
                    parent_linked
                    and isinstance(manifest.get("new_episode_commit"), str)
                    and 0 < int(manifest.get("episode_count", 0)) <= MAX_TIMELINE_ROWS
                )
            )
            or str(manifest.get("output_prefix") or "").rstrip("/") != self.dataset_uri
        ):
            raise EpisodeStoreError(
                "immutable dataset version is malformed", status_code=502
            )
        return manifest

    def _validate_version(
        self, version_id: str, episode_index: int, commit_key: str
    ) -> dict[str, Any]:
        manifest = self._version_manifest(version_id)
        expected_uri = f"s3://{self.bucket}/{commit_key}"
        commits = manifest.get("episode_commits")
        if isinstance(commits, list):
            included = expected_uri in {str(item) for item in commits}
        else:
            included = 0 <= episode_index < int(manifest["episode_count"])
        if not included:
            raise EpisodeStoreError(
                "episode is not part of that immutable version", status_code=404
            )
        return manifest

    def _timeline(
        self, commit: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], ObjectRef, str]:
        records_ref = self._object_ref(commit["objects"].get("records"), "records")
        if records_ref.size > MAX_TIMELINE_BYTES:
            raise EpisodeStoreError(
                "episode timeline exceeds the bounded player limit", status_code=413
            )
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=records_ref.key)
        except Exception as exc:
            raise EpisodeStoreError(
                "episode timeline is unavailable", status_code=502
            ) from exc
        body = _read_body(
            response.get("Body"), MAX_TIMELINE_BYTES, "episode timeline is unreadable"
        )
        digest = hashlib.sha256(body).hexdigest()
        if digest != records_ref.sha256:
            raise EpisodeStoreError(
                "episode timeline checksum mismatch", status_code=502
            )
        rows: list[dict[str, Any]] = []
        previous_mono = -1
        for line in body.splitlines():
            if not line.strip():
                continue
            if len(rows) >= MAX_TIMELINE_ROWS:
                raise EpisodeStoreError(
                    "episode timeline exceeds the bounded row limit", status_code=413
                )
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise EpisodeStoreError(
                    "episode timeline is malformed", status_code=502
                ) from exc
            if not isinstance(raw, dict):
                raise EpisodeStoreError(
                    "episode timeline is malformed", status_code=502
                )
            try:
                mono_ns = int(str(raw.get("monotonic_ns") or ""))
                wall_ns = int(str(raw.get("wall_clock_ns") or ""))
            except (TypeError, ValueError, OverflowError) as exc:
                raise EpisodeStoreError(
                    "episode timeline timestamps are malformed", status_code=502
                ) from exc
            if mono_ns <= previous_mono or wall_ns <= 0:
                raise EpisodeStoreError(
                    "episode timeline timestamps are not monotonic", status_code=502
                )
            previous_mono = mono_ns
            rows.append(
                {
                    "frame_index": int(raw.get("source_frame_index", len(rows))),
                    "sim_step": int(raw.get("sim_step", 0)),
                    "timestamp": float(
                        raw.get(
                            "timestamp",
                            len(rows) / max(1, int(commit["metadata"].get("fps") or 1)),
                        )
                    ),
                    "monotonic_ns": str(mono_ns),
                    "wall_clock_ns": str(wall_ns),
                    "action": raw.get("action"),
                    "observation_state": raw.get("observation.state"),
                    "reward": raw.get("reward"),
                    "success": bool(raw.get("success", False)),
                    "terminated": bool(raw.get("terminated", False)),
                    "truncated": bool(raw.get("truncated", False)),
                    "done": bool(raw.get("done", False)),
                    "reset_reason": str(raw.get("reset_reason") or ""),
                    "frame_sha256": str(raw.get("frame_sha256") or ""),
                }
            )
        if len(rows) < 2:
            raise EpisodeStoreError(
                "episode timeline has fewer than two aligned frames", status_code=502
            )
        return rows, records_ref, "verified"

    def _camera_refs(self, commit: dict[str, Any]) -> dict[str, ObjectRef]:
        objects = commit["objects"]
        result: dict[str, ObjectRef] = {}
        videos = objects.get("videos")
        if isinstance(videos, dict):
            for camera, payload in videos.items():
                camera_id = str(camera or "")
                if _CAMERA_ID.fullmatch(camera_id):
                    result[camera_id] = self._object_ref(payload, f"video-{camera_id}")
        if not result and isinstance(objects.get("video"), dict):
            result["primary"] = self._object_ref(objects["video"], "video")
        if not result:
            raise EpisodeStoreError(
                "episode has no playable video artifact", status_code=502
            )
        return result

    def detail(self, episode_id: str, *, version_id: str = "") -> dict[str, Any]:
        if not _EPISODE_ID.fullmatch(str(episode_id or "")):
            raise EpisodeStoreError("invalid episode ID")
        index = int(episode_id)
        commit, response, commit_key = self._commit(index)
        version: dict[str, Any] | None = None
        if version_id:
            version = self._validate_version(version_id, index, commit_key)
        rows, records_ref, records_state = self._timeline(commit)
        camera_refs = self._camera_refs(commit)
        metadata = commit["metadata"]
        summary = self._summary(commit, commit_key)
        start_ns = int(rows[0]["wall_clock_ns"])
        end_ns = int(rows[-1]["wall_clock_ns"])
        timeline_duration = max(
            0.0, float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"])
        )
        commit_checksum = str((response.get("Metadata") or {}).get("sha256") or "")
        artifacts: list[dict[str, Any]] = []
        for name, payload in commit["objects"].items():
            if name in {"video", "videos", "frames"} or not _ARTIFACT_ID.fullmatch(
                str(name)
            ):
                continue
            try:
                ref = self._object_ref(payload, str(name))
            except EpisodeStoreError:
                # Unknown nested artifact collections stay discoverable as an
                # explicit fallback record without accepting a user-supplied key.
                artifacts.append(
                    {"name": str(name), "kind": "unknown", "download_url": ""}
                )
                continue
            artifacts.append(
                {
                    "name": name,
                    "kind": "timeline" if name == "records" else "download",
                    "bytes": ref.size,
                    "sha256": ref.sha256,
                    "download_url": self._download_url(index, str(name), version_id),
                }
            )
        cameras = [
            {
                "id": camera,
                "label": "Primary camera"
                if len(camera_refs) == 1
                else camera.replace("_", " ").title(),
                "bytes": ref.size,
                "sha256": ref.sha256,
                "media_url": self._media_url(index, camera, version_id),
            }
            for camera, ref in camera_refs.items()
        ]
        return {
            **summary,
            "run_id": self.run_id,
            "dataset_uri": self.dataset_uri,
            "dataset_version": str(version.get("version") if version else version_id),
            "dataset_version_uri": str(version.get("dataset_uri") if version else ""),
            "start_timestamp": _iso_from_ns(start_ns),
            "end_timestamp": _iso_from_ns(end_ns),
            "duration_seconds": timeline_duration,
            "timeline_rows": len(rows),
            "timeline_checksum": records_ref.sha256,
            "timeline_checksum_state": records_state,
            "commit_checksum": commit_checksum
            if _SHA256.fullmatch(commit_checksum)
            else "",
            "provenance": metadata.get("provenance")
            if isinstance(metadata.get("provenance"), dict)
            else {},
            "source_uris": {
                "commit": summary["commit_uri"],
                "dataset_version": str(version.get("dataset_uri") if version else ""),
            },
            "markers": {
                "outcome": summary["outcome"],
                "success_frames": [
                    row["frame_index"] for row in rows if row["success"]
                ],
                "reset_frames": [
                    row["frame_index"] for row in rows if row["reset_reason"]
                ],
            },
            "cameras": cameras,
            "camera_mode": "synchronized-two-camera"
            if len(cameras) >= 2
            else "legacy-single-camera",
            "timeline_url": self._timeline_url(index, version_id),
            "artifacts": artifacts,
            "export": {
                "records_url": next(
                    (
                        item["download_url"]
                        for item in artifacts
                        if item["name"] == "records"
                    ),
                    "",
                ),
                "metadata_url": next(
                    (
                        item["download_url"]
                        for item in artifacts
                        if item["name"] == "metadata"
                    ),
                    "",
                ),
                "dataset_version_uri": str(
                    version.get("dataset_uri") if version else ""
                ),
            },
        }

    def timeline(self, episode_id: str, *, version_id: str = "") -> dict[str, Any]:
        if not _EPISODE_ID.fullmatch(str(episode_id or "")):
            raise EpisodeStoreError("invalid episode ID")
        index = int(episode_id)
        commit, _response, commit_key = self._commit(index)
        if version_id:
            self._validate_version(version_id, index, commit_key)
        rows, ref, state = self._timeline(commit)
        return {
            "episode_index": index,
            "rows": rows,
            "row_count": len(rows),
            "sha256": ref.sha256,
            "checksum_state": state,
            "bounded": True,
        }

    def media_ref(
        self, episode_id: str, camera_id: str, *, version_id: str = ""
    ) -> ObjectRef:
        if not _EPISODE_ID.fullmatch(str(episode_id or "")) or not _CAMERA_ID.fullmatch(
            str(camera_id or "")
        ):
            raise EpisodeStoreError("invalid episode or camera ID")
        index = int(episode_id)
        commit, _response, commit_key = self._commit(index)
        if version_id:
            self._validate_version(version_id, index, commit_key)
        cameras = self._camera_refs(commit)
        if camera_id not in cameras:
            raise EpisodeStoreError("episode camera was not found", status_code=404)
        return cameras[camera_id]

    def download_ref(
        self, episode_id: str, artifact_id: str, *, version_id: str = ""
    ) -> ObjectRef:
        if not _EPISODE_ID.fullmatch(
            str(episode_id or "")
        ) or not _ARTIFACT_ID.fullmatch(str(artifact_id or "")):
            raise EpisodeStoreError("invalid episode or artifact ID")
        index = int(episode_id)
        commit, _response, commit_key = self._commit(index)
        if version_id:
            self._validate_version(version_id, index, commit_key)
        payload = commit["objects"].get(artifact_id)
        if artifact_id in {"video", "videos", "frames"} or not isinstance(
            payload, dict
        ):
            raise EpisodeStoreError("episode artifact was not found", status_code=404)
        return self._object_ref(payload, artifact_id)

    def _base_query(self, version_id: str) -> str:
        suffix = "&version_id=" + quote(version_id, safe="") if version_id else ""
        return "?run_id=" + quote(self.run_id, safe="") + suffix

    def _media_url(self, index: int, camera: str, version_id: str) -> str:
        return (
            f"/api/leisaac/episodes/{index}/media/{quote(camera, safe='')}"
            + self._base_query(version_id)
        )

    def _timeline_url(self, index: int, version_id: str) -> str:
        return f"/api/leisaac/episodes/{index}/timeline" + self._base_query(version_id)

    def _download_url(self, index: int, artifact: str, version_id: str) -> str:
        return (
            f"/api/leisaac/episodes/{index}/download/{quote(artifact, safe='')}"
            + self._base_query(version_id)
        )


def iter_s3_body(body: Any, length: int):
    """Yield exactly ``length`` bytes and always close the S3 response body."""

    remaining = int(length)
    try:
        while remaining > 0:
            chunk = body.read(min(STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                raise IOError("S3 media body ended before Content-Length")
            data = bytes(chunk)
            remaining -= len(data)
            yield data
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()

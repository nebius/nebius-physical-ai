"""Bounded S3 discovery, playback, range, and timeline tests for LeIsaac."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from npa.agent_backend.leisaac import LEISAAC_MEDIA_PORT, LEISAAC_SIGNAL_PORT
from npa.agent_backend.leisaac_episodes import (
    ByteRange,
    EpisodeStore,
    EpisodeStoreError,
    RangeNotSatisfiable,
    parse_http_range,
)
from npa.agent_backend.leisaac_registry import REGISTRY_FINGERPRINT
from npa.agent_backend.leisaac_routes import LeIsaacDeps, register_leisaac_routes


VERSION_ID = "v000001-" + "b" * 32
RUN_ID = "episode-browser-run"
PREFIX = "datasets/leisaac-browser"


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.list_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def put(self, key: str, body: bytes, *, checksum: str = "") -> dict:
        digest = checksum or hashlib.sha256(body).hexdigest()
        self.objects[key] = (body, {"sha256": digest})
        return {"key": key, "sha256": digest, "bytes": len(body)}

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(dict(kwargs))
        prefix = str(kwargs.get("Prefix") or "")
        maximum = int(kwargs.get("MaxKeys") or 1000)
        offset = int(kwargs.get("ContinuationToken") or 0)
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        delimiter = kwargs.get("Delimiter")
        if delimiter:
            choices = sorted(
                {
                    prefix
                    + key[len(prefix) :].split(str(delimiter), 1)[0]
                    + str(delimiter)
                    for key in keys
                    if str(delimiter) in key[len(prefix) :]
                }
            )
            page = choices[offset : offset + maximum]
            result = {"CommonPrefixes": [{"Prefix": item} for item in page]}
        else:
            page = keys[offset : offset + maximum]
            result = {
                "Contents": [
                    {"Key": item, "Size": len(self.objects[item][0])} for item in page
                ]
            }
        next_offset = offset + len(page)
        result["IsTruncated"] = next_offset < (len(choices) if delimiter else len(keys))
        if result["IsTruncated"]:
            result["NextContinuationToken"] = str(next_offset)
        return result

    def get_object(self, **kwargs):
        self.get_calls.append(dict(kwargs))
        body, metadata = self.objects[str(kwargs["Key"])]
        raw_range = str(kwargs.get("Range") or "")
        if raw_range:
            start_raw, end_raw = raw_range.removeprefix("bytes=").split("-", 1)
            body = body[int(start_raw) : int(end_raw) + 1]
        return {
            "Body": io.BytesIO(body),
            "Metadata": dict(metadata),
            "ContentLength": len(body),
        }

    def head_object(self, **kwargs):
        body, metadata = self.objects[str(kwargs["Key"])]
        return {"ContentLength": len(body), "Metadata": dict(metadata)}


def _records() -> bytes:
    rows = []
    for index in range(3):
        rows.append(
            {
                "source_frame_index": index,
                "sim_step": index + 10,
                "timestamp": index / 16,
                "monotonic_ns": 1_000_000_000 + index * 62_500_000,
                "wall_clock_ns": 1_800_000_000_000_000_000 + index * 62_500_000,
                "action": [float(index)] * 8,
                "observation.state": [float(index)] * 6,
                "reward": float(index == 2),
                "success": index == 2,
                "terminated": index == 2,
                "truncated": False,
                "done": index == 2,
                "reset_reason": "success" if index == 2 else "",
                "frame_sha256": f"{index + 1:064x}",
            }
        )
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


def _fixture(*, two_cameras: bool = False, unknown: bool = False):
    s3 = FakeS3()
    records = s3.put(f"{PREFIX}/episodes/000000-uuid/records.jsonl", _records())
    metadata_bytes = b'{"schema":"npa.leisaac.episode.v1"}\n'
    metadata_ref = s3.put(f"{PREFIX}/episodes/000000-uuid/episode.json", metadata_bytes)
    primary = s3.put(
        f"{PREFIX}/episodes/000000-uuid/primary.mp4", b"primary-video-bytes"
    )
    objects: dict = {"records": records, "metadata": metadata_ref, "frames": []}
    if two_cameras:
        secondary = s3.put(
            f"{PREFIX}/episodes/000000-uuid/secondary.mp4", b"secondary-video-bytes"
        )
        objects["videos"] = {"workspace": primary, "wrist": secondary}
    else:
        objects["video"] = primary
    if unknown:
        objects["calibration"] = s3.put(
            f"{PREFIX}/episodes/000000-uuid/calibration.bin", b"unknown-calibration"
        )
    commit = {
        "schema": "npa.leisaac.episode-commit.v1",
        "episode_index": 0,
        "episode_uuid": "uuid",
        "committed_at": "2026-08-06T01:00:00Z",
        "metadata": {
            "schema": "npa.leisaac.episode.v1",
            "episode_uuid": "uuid",
            "run_id": RUN_ID,
            "task": "LeIsaac-SO101-LiftCube-v0",
            "environment_id": "table-b",
            "environment_index": 1,
            "seed": 47,
            "outcome": "success",
            "frame_count": 3,
            "fps": 16,
            "recorded_at": "2026-08-06T00:59:59Z",
            "provenance": {
                "robot": "custom-so101",
                "scene": "custom-table",
                "device": "spacemouse",
                "bundle": "bundle-sha256",
            },
        },
        "objects": objects,
    }
    commit_key = f"{PREFIX}/commits/episode-000000.json"
    commit_body = (json.dumps(commit, sort_keys=True) + "\n").encode()
    s3.put(commit_key, commit_body)
    version = {
        "schema": "npa.leisaac.dataset.v1",
        "version": VERSION_ID,
        "dataset_uri": f"s3://bucket/{PREFIX}/versions/{VERSION_ID}",
        "output_prefix": f"s3://bucket/{PREFIX}",
        "created_at": "2026-08-06T01:00:01Z",
        "episode_count": 1,
        "episode_commits": [f"s3://bucket/{commit_key}"],
        "lerobot_version": "0.5.1",
    }
    s3.put(
        f"{PREFIX}/versions/{VERSION_ID}/npa-dataset.json",
        (json.dumps(version, sort_keys=True) + "\n").encode(),
    )
    return s3, commit, version


def _store(s3: FakeS3) -> EpisodeStore:
    return EpisodeStore(
        s3,
        f"s3://bucket/{PREFIX}",
        allowed_buckets=["bucket"],
        run_id=RUN_ID,
    )


@pytest.mark.parametrize(
    "header,expected",
    [
        ("", None),
        ("bytes=2-5", ByteRange(2, 5, 10)),
        ("bytes=6-", ByteRange(6, 9, 10)),
        ("bytes=-4", ByteRange(6, 9, 10)),
        ("bytes=-40", ByteRange(0, 9, 10)),
        ("bytes=2-99", ByteRange(2, 9, 10)),
    ],
)
def test_http_range_full_suffix_and_open_ended(header, expected) -> None:
    assert parse_http_range(header, 10) == expected


@pytest.mark.parametrize(
    "header",
    ["items=1-2", "bytes=", "bytes=9-2", "bytes=10-", "bytes=-0", "bytes=1-2,4-5"],
)
def test_http_range_rejects_unsatisfiable_and_multi_range(header: str) -> None:
    with pytest.raises(RangeNotSatisfiable) as exc_info:
        parse_http_range(header, 10)
    assert exc_info.value.size == 10


def test_episode_listing_versions_filters_and_bounded_pagination() -> None:
    s3, _commit, _version = _fixture()
    store = _store(s3)

    versions = store.list_versions(limit=5)
    assert [item["version_id"] for item in versions["versions"]] == [VERSION_ID]
    assert versions["bounded"] is True
    episodes = store.list_episodes(
        limit=5,
        filters={
            "task": "LeIsaac-SO101-LiftCube-v0",
            "environment": "table-b",
            "outcome": "success",
            "robot": "custom-so101",
            "scene": "custom-table",
            "device": "spacemouse",
            "date_from": "2026-08-06",
            "date_to": "2026-08-07",
        },
    )
    assert [item["episode_index"] for item in episodes["episodes"]] == [0]
    assert episodes["episodes"][0]["bundle"] == "bundle-sha256"
    assert all(call["MaxKeys"] <= 100 for call in s3.list_calls)
    episode_list_call = next(
        call for call in s3.list_calls if call["Prefix"].endswith("commits/episode-")
    )
    assert episode_list_call["MaxKeys"] == 5
    assert (
        store.list_episodes(limit=5, filters={"outcome": "failure"})["episodes"] == []
    )
    version_episodes = store.list_episodes(limit=5, version_id=VERSION_ID)
    assert [item["dataset_version"] for item in version_episodes["episodes"]] == [
        VERSION_ID
    ]
    with pytest.raises(EpisodeStoreError):
        store.list_episodes(limit=5, version_id="../../latest")


def test_filtered_empty_s3_page_returns_truthful_continuation_signal() -> None:
    s3, commit, _version = _fixture()
    second = json.loads(json.dumps(commit))
    second["episode_index"] = 1
    second["metadata"]["outcome"] = "success"
    s3.put(
        f"{PREFIX}/commits/episode-000001.json",
        (json.dumps(second, sort_keys=True) + "\n").encode(),
    )
    first = _store(s3).list_episodes(limit=1, filters={"outcome": "failure"})
    assert first["episodes"] == []
    assert first["next_cursor"]
    assert first["has_more_pages"] is True
    assert first["source_count"] == first["loaded_count"] == 1
    assert first["filtered_count"] == 1

    second_page = _store(s3).list_episodes(
        limit=1,
        cursor=first["next_cursor"],
        filters={"outcome": "success"},
    )
    assert [item["episode_index"] for item in second_page["episodes"]] == [1]


def test_json_checksums_are_computed_when_gateway_omits_metadata() -> None:
    s3, _commit, _version = _fixture()
    commit_key = f"{PREFIX}/commits/episode-000000.json"
    version_key = f"{PREFIX}/versions/{VERSION_ID}/npa-dataset.json"
    commit_body, _commit_metadata = s3.objects[commit_key]
    version_body, _version_metadata = s3.objects[version_key]
    s3.objects[commit_key] = (commit_body, {})
    s3.objects[version_key] = (version_body, {})

    store = _store(s3)
    assert store.detail("0")["commit_checksum"] == hashlib.sha256(
        commit_body
    ).hexdigest()
    assert store.list_versions()["versions"][0]["manifest_checksum"] == (
        hashlib.sha256(version_body).hexdigest()
    )

    s3.objects[commit_key] = (commit_body, {"sha256": "0" * 64})
    with pytest.raises(EpisodeStoreError, match="episode commit was not found"):
        store.detail("0")


def test_episode_and_version_listing_reads_are_ordered_and_bounded_concurrent() -> None:
    base, commit, version = _fixture()

    class TrackingS3(FakeS3):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.guard = threading.Lock()

        def get_object(self, **kwargs):
            with self.guard:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.005)
                return super().get_object(**kwargs)
            finally:
                with self.guard:
                    self.active -= 1

    s3 = TrackingS3()
    s3.objects.update(base.objects)
    for index in range(12):
        item = dict(commit)
        item["episode_index"] = index
        item["episode_uuid"] = f"uuid-{index}"
        body = (json.dumps(item, sort_keys=True) + "\n").encode()
        s3.put(f"{PREFIX}/commits/episode-{index:06d}.json", body)
        version_id = f"v{index + 1:06d}-" + f"{index + 1:x}" * 32
        manifest = dict(version)
        manifest.update(
            version=version_id,
            dataset_uri=f"s3://bucket/{PREFIX}/versions/{version_id}",
        )
        s3.put(
            f"{PREFIX}/versions/{version_id}/npa-dataset.json",
            (json.dumps(manifest, sort_keys=True) + "\n").encode(),
        )

    store = _store(s3)
    episodes = store.list_episodes(limit=12)["episodes"]
    assert [item["episode_index"] for item in episodes] == list(range(12))
    versions = store.list_versions(limit=12)["versions"]
    assert [item["version_id"] for item in versions] == sorted(
        item["version_id"] for item in versions
    )
    assert 1 < s3.maximum_active <= 8


def test_aggregate_versions_skip_only_legacy_unreadable_objects() -> None:
    s3, _commit, _version = _fixture()
    malformed_id = "v000002-" + "c" * 32
    s3.put(f"{PREFIX}/versions/{malformed_id}/npa-dataset.json", b"{")
    store = _store(s3)

    assert [item["version_id"] for item in store.list_versions()["versions"]] == [
        VERSION_ID
    ]
    with pytest.raises(EpisodeStoreError, match="not found"):
        store._version_manifest(malformed_id)


def test_aggregate_versions_do_not_hide_auth_failures() -> None:
    s3, _commit, _version = _fixture()

    class AccessDenied(RuntimeError):
        response = {"Error": {"Code": "AccessDenied"}}

    original = s3.get_object

    def denied(**kwargs):
        if "/versions/" in str(kwargs["Key"]):
            raise AccessDenied("denied")
        return original(**kwargs)

    s3.get_object = denied  # type: ignore[method-assign]
    with pytest.raises(EpisodeStoreError, match="unreadable") as exc_info:
        _store(s3).list_versions()
    assert exc_info.value.status_code == 502


def test_episode_listing_does_not_hide_worker_auth_failure() -> None:
    s3, _commit, _version = _fixture()

    class AccessDenied(RuntimeError):
        response = {"Error": {"Code": "AccessDenied"}}

    original = s3.get_object

    def denied(**kwargs):
        if "/commits/" in str(kwargs["Key"]):
            raise AccessDenied("denied")
        return original(**kwargs)

    s3.get_object = denied  # type: ignore[method-assign]
    with pytest.raises(EpisodeStoreError, match="unreadable") as exc_info:
        _store(s3).list_episodes()
    assert exc_info.value.status_code == 502


def test_episode_detail_timeline_two_camera_and_unknown_download_fallback() -> None:
    s3, _commit, _version = _fixture(two_cameras=True, unknown=True)
    store = _store(s3)

    detail = store.detail("0", version_id=VERSION_ID)
    assert detail["camera_mode"] == "synchronized-two-camera"
    assert [item["id"] for item in detail["cameras"]] == ["workspace", "wrist"]
    assert detail["timeline_rows"] == 3
    assert detail["timeline_checksum_state"] == "verified"
    assert detail["dataset_version"] == VERSION_ID
    fallback = next(
        item for item in detail["artifacts"] if item["name"] == "calibration"
    )
    assert fallback["kind"] == "download"
    assert "/download/calibration" in fallback["download_url"]
    timeline = store.timeline("0", version_id=VERSION_ID)
    assert timeline["rows"][2]["success"] is True
    assert timeline["rows"][2]["reset_reason"] == "success"


def test_legacy_single_camera_is_explicit_and_traversal_is_rejected() -> None:
    s3, _commit, _version = _fixture()
    store = _store(s3)
    assert store.detail("0")["camera_mode"] == "legacy-single-camera"
    for episode_id in ("../0", "0/../../etc", "uuid"):
        with pytest.raises(EpisodeStoreError):
            store.detail(episode_id)
    with pytest.raises(EpisodeStoreError):
        store.media_ref("0", "../primary")
    with pytest.raises(EpisodeStoreError):
        store.detail("0", version_id="../../versions/latest")
    with pytest.raises(EpisodeStoreError):
        EpisodeStore(
            s3,
            "s3://other/private",
            allowed_buckets=["bucket"],
            run_id=RUN_ID,
        )


def _manifest() -> dict:
    now = datetime.now(timezone.utc)
    nonce = "a" * 64
    return {
        "schema": "npa.leisaac.session.v2",
        "run_id": RUN_ID,
        "provider": "nebius-kubernetes",
        "task": "LeIsaac-SO101-LiftCube-v0",
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "teleop_device": "keyboard",
        "environment": {"id": "table-b", "index": 1, "seed": 47, "num_envs": 1},
        "dataset": {
            "output_path": f"s3://bucket/{PREFIX}",
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
        "signal_host": "8.8.8.8",
        "signal_port": LEISAAC_SIGNAL_PORT,
        "media_host": "1.1.1.1",
        "media_server": "1.1.1.1",
        "media_port": LEISAAC_MEDIA_PORT,
        "service_url": "http://8.8.8.8:8080",
        "session_nonce": nonce,
        "session_attestation": hashlib.sha256(
            f"npa-leisaac-session:{nonce}".encode()
        ).hexdigest(),
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "source_version": "0.4.0",
        "source_commit": "1" * 40,
        "isaac_sim_version": "5.1.0.0",
        "isaac_lab_version": "2.3.2.post1",
        "image": "registry.example/npa-leisaac@sha256:" + "2" * 64,
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }


def test_authenticated_episode_routes_stream_ranges_and_return_416() -> None:
    s3, _commit, _version = _fixture(two_cameras=True, unknown=True)
    manifest = _manifest()
    app = FastAPI()
    register_leisaac_routes(
        app,
        LeIsaacDeps(
            load_state=lambda: {"leisaac": {"run_id": RUN_ID}},
            resolve_manifest=lambda run_id: manifest if run_id == RUN_ID else None,
            http_get=lambda *_args, **_kwargs: None,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
            s3_client=lambda: (s3, {}),
            s3_buckets=lambda _client, _settings: ["bucket"],
        ),
    )
    client = TestClient(app)
    headers = {"x-forwarded-proto": "https", "sec-fetch-site": "same-origin"}

    listed = client.get(f"/leisaac/episodes?run_id={RUN_ID}", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["episodes"][0]["episode_index"] == 0
    detail = client.get(
        f"/leisaac/episodes/0?run_id={RUN_ID}&version_id={VERSION_ID}", headers=headers
    )
    assert detail.status_code == 200
    media_url = detail.json()["cameras"][0]["media_url"].removeprefix("/api")
    full = client.get(media_url, headers=headers)
    assert full.status_code == 200
    assert full.content == b"primary-video-bytes"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"].startswith("video/mp4")
    partial = client.get(media_url, headers={**headers, "range": "bytes=2-7"})
    assert partial.status_code == 206
    assert partial.content == b"imary-"
    assert partial.headers["content-range"] == "bytes 2-7/19"
    suffix = client.get(media_url, headers={**headers, "range": "bytes=-5"})
    assert suffix.status_code == 206
    assert suffix.content == b"bytes"
    rejected = client.get(media_url, headers={**headers, "range": "bytes=99-"})
    assert rejected.status_code == 416
    assert rejected.headers["content-range"] == "bytes */19"
    assert (
        client.get(
            media_url, headers={**headers, "sec-fetch-site": "cross-site"}
        ).status_code
        == 403
    )
    assert client.get(
        f"/leisaac/episodes/../../etc/passwd?run_id={RUN_ID}", headers=headers
    ).status_code in {404, 400}

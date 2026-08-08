from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from npa.workbench.leisaac import dataset as leisaac_dataset
from npa.agent_backend.leisaac_registry import (
    DEFAULT_TASK,
    REGISTRY_FINGERPRINT,
    registry_payload,
    resolve_configuration,
    validate_num_envs,
    validate_task,
)
from npa.workbench.leisaac.dataset import (
    DatasetError,
    EpisodeRecorder,
    S3DatasetStore,
    extract_step,
    resolve_s3_endpoint,
)


def _jpeg(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1280, 720), color=color).save(output, format="JPEG")
    return output.getvalue()


def _step(index: int) -> dict:
    return {
        "observation.state": [float(index)] * 6,
        "action": [float(index) / 10] * 8,
        "reward": float(index),
        "terminated": False,
        "truncated": False,
        "done": False,
        "sim_step": index,
        "monotonic_ns": 1_000_000_000 + index,
        "wall_clock_ns": 2_000_000_000 + index,
        "input_id": index,
        "input_key": "W",
    }


def test_registry_is_the_honest_two_task_sequential_contract() -> None:
    payload = registry_payload()
    assert payload["fingerprint"] == REGISTRY_FINGERPRINT
    assert {task["task"] for task in payload["tasks"]} == {
        "LeIsaac-SO101-PickOrange-v0",
        "LeIsaac-SO101-LiftCube-v0",
    }
    assert payload["environment_model"] == "named-sequential"
    assert payload["max_parallel_environments"] == 1
    assert payload["default_task"] == "LeIsaac-SO101-LiftCube-v0"
    assert DEFAULT_TASK == payload["default_task"]
    assert validate_task("LeIsaac-SO101-LiftCube-v0").endswith("LiftCube-v0")
    with pytest.raises(ValueError, match="unsupported"):
        validate_task("made-up")
    with pytest.raises(ValueError, match="exactly one"):
        validate_num_envs(2)


def test_registry_resolves_real_defaults_and_cumulative_custom_overrides() -> None:
    defaults = resolve_configuration()
    assert defaults["schema"] == "npa.leisaac.configuration.v1"
    assert defaults["robot"]["id"] == "so101_follower"
    assert defaults["scene"]["id"] == "table_with_cube"
    assert defaults["device"]["id"] == "browser_keyboard_so101"
    assert defaults["task"]["id"] == "LeIsaac-SO101-LiftCube-v0"
    assert defaults["custom_bundle_count"] == 0
    assert {
        defaults[kind]["source"] for kind in ("robot", "scene", "device")
    } == {"built-in-runtime"}

    custom = resolve_configuration(
        selected_bundles={
            "robot": {
                "bundle_sha256": "a" * 64,
                "name": "custom-so101",
                "entrypoint": "robot.usda",
            },
            "device": {
                "bundle_sha256": "b" * 64,
                "name": "custom-device",
                "entrypoint": "device.json",
            },
        }
    )
    assert custom["robot"]["source"] == "uploaded-bundle"
    assert custom["device"]["source"] == "uploaded-bundle"
    assert custom["scene"] == defaults["scene"]
    assert custom["task"] == defaults["task"]
    assert custom["custom_bundle_count"] == 2


def test_extract_step_uses_real_environment_return_values() -> None:
    result = (
        {"policy": {"joint_pos": np.arange(6, dtype=np.float32)[None, :]}},
        np.array([0.75], dtype=np.float32),
        np.array([False]),
        np.array([True]),
        {"real": True},
    )
    record = extract_step(result, np.arange(8, dtype=np.float32)[None, :], sim_step=17)
    assert record["observation.state"] == pytest.approx(list(range(6)))
    assert record["action"] == pytest.approx(list(range(8)))
    assert record["reward"] == pytest.approx(0.75)
    assert record["terminated"] is False
    assert record["truncated"] is True
    assert record["done"] is True
    assert record["sim_step"] == 17
    assert record["monotonic_ns"] > 0 and record["wall_clock_ns"] > 0


def test_recorder_requires_outcome_and_atomically_finalizes(tmp_path: Path) -> None:
    published = []

    def publish(path: Path, metadata: dict) -> dict:
        published.append((path, metadata, path.joinpath("records.jsonl").read_text()))
        return {
            "episode_index": 3,
            "completed_episode_count": 4,
            "dataset_version_uri": "s3://bucket/demos/versions/v000004-test",
        }

    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task=DEFAULT_TASK,
        environment_id="table-a",
        environment_index=2,
        seed=7,
        run_id="run-1",
        source_commit="1" * 40,
        publisher=publish,
    )
    recorder.start()
    for index, color in ((1, (200, 10, 10)), (2, (10, 200, 10))):
        recorder.observe(_step(index))
        recorder.frame(_jpeg(color))
    with pytest.raises(DatasetError, match="mark success or failure"):
        recorder.finalize()
    recorder.mark("success")
    result = recorder.finalize()
    status = recorder.status()
    assert result["episode_index"] == 3
    assert status["state"] == "idle"
    assert status["active_episode"] is None
    assert status["last_episode_index"] == 3
    assert status["completed_episode_count"] == 4
    assert status["last_outcome"] == "success"
    assert len(published) == 1
    rows = [json.loads(line) for line in published[0][2].splitlines()]
    assert [row["sim_step"] for row in rows] == [1, 2]
    assert all(row["task"] == DEFAULT_TASK for row in rows)
    assert rows[-1]["success"] is True
    assert rows[-1]["reset_reason"] == "success"
    assert rows[0]["timestamp"] == 0.0


def test_recorder_commits_only_complete_synchronized_camera_groups(
    tmp_path: Path,
) -> None:
    published: list[Path] = []

    def publish(path: Path, _metadata: dict) -> dict:
        published.append(path)
        return {
            "episode_index": 0,
            "completed_episode_count": 1,
            "dataset_version_uri": "s3://bucket/demos/versions/v000001-dual",
        }

    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task=DEFAULT_TASK,
        environment_id="table-a",
        environment_index=0,
        seed=7,
        run_id="run-dual",
        source_commit="1" * 40,
        camera_ids=("workspace", "overview"),
        provenance={
            "robot": "so101_follower",
            "scene": "table_with_cube",
            "device": "browser_keyboard_so101",
            "task": DEFAULT_TASK,
            "bundle": "built-in",
        },
        publisher=publish,
    )
    recorder.start()
    recorder.observe(_step(1))
    recorder.frame(_jpeg((200, 10, 10)), camera_id="workspace", capture_group="g1")
    assert recorder.status()["frame_count"] == 0
    recorder.frame(_jpeg((10, 10, 200)), camera_id="overview", capture_group="g1")
    assert recorder.status()["frame_count"] == 1
    recorder.observe(_step(2))
    recorder.frame(_jpeg((10, 200, 10)), camera_id="overview", capture_group="g2")
    recorder.frame(_jpeg((200, 200, 10)), camera_id="workspace", capture_group="g2")
    recorder.mark("failure")
    recorder.finalize()

    episode = published[0]
    assert len(list((episode / "frames").glob("frame-*.jpg"))) == 2
    assert len(list((episode / "frames-overview").glob("frame-*.jpg"))) == 2
    metadata = json.loads((episode / "episode.json").read_text())
    assert metadata["cameras"] == ["workspace", "overview"]
    assert metadata["task"] == DEFAULT_TASK
    assert metadata["provenance"]["robot"] == "so101_follower"
    assert metadata["provenance"]["scene"] == "table_with_cube"
    assert metadata["provenance"]["device"] == "browser_keyboard_so101"
    assert metadata["provenance"]["task"] == DEFAULT_TASK
    assert metadata["provenance"]["bundle"] == "built-in"
    rows = [
        json.loads(line)
        for line in (episode / "records.jsonl").read_text().splitlines()
    ]
    assert all(
        set(row["camera_frame_sha256"]) == {"workspace", "overview"} for row in rows
    )
    assert rows[-1]["success"] is False
    assert rows[-1]["reset_reason"] == "failure"


def test_recorder_can_retry_a_failed_immutable_upload(tmp_path: Path) -> None:
    attempts = 0
    observed_metadata = []

    def publish(_path: Path, metadata: dict) -> dict:
        nonlocal attempts
        attempts += 1
        observed_metadata.append(dict(metadata))
        if attempts == 1:
            raise DatasetError("temporary object-store failure")
        return {
            "episode_index": 0,
            "completed_episode_count": 1,
            "dataset_version_uri": "s3://bucket/demos/versions/v000001-retry",
        }

    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task="LeIsaac-SO101-PickOrange-v0",
        environment_id="counter-a",
        environment_index=0,
        seed=7,
        run_id="run-1",
        source_commit="1" * 40,
        publisher=publish,
    )
    recorder.start()
    for index, color in ((1, (200, 10, 10)), (2, (10, 200, 10))):
        recorder.observe(_step(index))
        recorder.frame(_jpeg(color))
    recorder.mark("failure")
    with pytest.raises(DatasetError, match="temporary"):
        recorder.finalize()
    assert recorder.status()["state"] == "upload-failed"
    assert recorder.finalize()["episode_index"] == 0
    assert recorder.status()["state"] == "idle"
    assert observed_metadata[0] == observed_metadata[1]
    assert observed_metadata[0]["recorded_at"]


def test_recorder_retries_failed_authoritative_status_recovery(tmp_path: Path) -> None:
    attempts = 0

    def recover() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DatasetError("temporary latest pointer failure")
        return {
            "completed_episode_count": 0,
            "last_episode_index": None,
            "last_outcome": "",
            "last_upload_status": "never",
            "dataset_version_uri": "",
            "last_episode_commit_uri": "",
        }

    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task="LeIsaac-SO101-PickOrange-v0",
        environment_id="counter-a",
        environment_index=0,
        seed=7,
        run_id="run-recover",
        source_commit="1" * 40,
        publisher=lambda _path, _metadata: {},
        status_loader=recover,
    )
    assert "Dataset state recovery failed" in recorder.status()["last_error"]

    recorder.start()

    assert attempts == 2
    assert recorder.status()["state"] == "recording"
    assert recorder.status()["last_error"] == ""


def test_recorder_command_ids_are_recoverable_and_idempotent(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task="LeIsaac-SO101-PickOrange-v0",
        environment_id="counter-a",
        environment_index=0,
        seed=7,
        run_id="run-command-ids",
        source_commit="1" * 40,
        publisher=lambda _path, _metadata: {},
    )
    request_id = "start-command-1"
    recorder.pending_command_path.write_text(
        json.dumps({"request_id": request_id, "command": "start"}),
        encoding="utf-8",
    )
    recorder.control_path.write_text(
        "".join(
            json.dumps({"request_id": request_id, "command": "start"}) + "\n"
            for _ in range(2)
        ),
        encoding="utf-8",
    )

    recorder.process_commands()

    status = recorder.status()
    assert status["state"] == "recording"
    assert status["active_episode"]
    assert status["last_command_id"] == request_id
    assert status["last_command"] == "start"
    assert status["command_revision"] == 1
    assert status["pending_command_id"] == ""
    assert not recorder.pending_command_path.exists()

    recorder.control_path.write_text(
        json.dumps({"request_id": "invalid-mark", "command": "mark-success"}) + "\n",
        encoding="utf-8",
    )
    recorder._control_offset = 0
    recorder.process_commands()
    assert recorder.status()["state"] == "outcome-pending"
    assert recorder.status()["pending_outcome"] == "success"

    with recorder.control_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"request_id": request_id, "command": "start"}) + "\n")
    recorder.process_commands()
    assert recorder.status()["state"] == "outcome-pending"
    assert recorder.status()["command_revision"] == 2
    assert recorder.status()["processed_commands"] == {
        request_id: "start",
        "invalid-mark": "mark-success",
    }

    recovered = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task="LeIsaac-SO101-PickOrange-v0",
        environment_id="counter-a",
        environment_index=0,
        seed=7,
        run_id="run-command-ids",
        source_commit="1" * 40,
        publisher=lambda _path, _metadata: {},
    )
    assert (
        recovered.status()["processed_commands"]
        == recorder.status()["processed_commands"]
    )
    recovered.process_commands()
    assert recovered.status()["command_revision"] == 2


def test_recorder_command_publication_does_not_block_simulator_loop(
    tmp_path: Path,
) -> None:
    publication_started = threading.Event()
    release_publication = threading.Event()
    reset_calls: list[str] = []

    def publish(_path: Path, _metadata: dict) -> dict:
        publication_started.set()
        assert release_publication.wait(timeout=5)
        return {
            "episode_index": 0,
            "completed_episode_count": 1,
            "dataset_version_uri": "s3://bucket/demos/versions/v000001-async",
            "episode_commit_uri": "s3://bucket/demos/episodes/episode-000000.json",
        }

    recorder = EpisodeRecorder(
        root=tmp_path,
        output_uri="s3://bucket/demos",
        task="LeIsaac-SO101-PickOrange-v0",
        environment_id="counter-a",
        environment_index=0,
        seed=7,
        run_id="run-async-finalize",
        source_commit="1" * 40,
        publisher=publish,
    )
    recorder.start()
    for index, color in ((1, (200, 10, 10)), (2, (10, 200, 10))):
        recorder.observe(_step(index))
        recorder.frame(_jpeg(color))
    recorder.mark("success")
    request_id = "async-finalize-1"
    recorder.pending_command_path.write_text(
        json.dumps({"request_id": request_id, "command": "finalize"}),
        encoding="utf-8",
    )
    with recorder.control_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"request_id": request_id, "command": "finalize"}) + "\n"
        )

    recorder.process_commands(reset=lambda: reset_calls.append("reset"))

    assert publication_started.wait(timeout=2)
    assert recorder.status()["state"] == "uploading"
    assert recorder.status()["last_command_id"] == request_id
    assert recorder.status()["pending_command_id"] == ""
    assert reset_calls == []
    # Polling commands while the object-store publisher is blocked must return;
    # this is the same call site as Isaac's render/control loop.
    recorder.process_commands(reset=lambda: reset_calls.append("unexpected"))
    assert reset_calls == []

    release_publication.set()
    future = recorder._finalize_future
    assert future is not None
    future.result(timeout=2)
    recorder.process_commands(reset=lambda: reset_calls.append("unexpected"))

    assert reset_calls == ["reset"]
    assert recorder.status()["state"] == "idle"
    assert recorder.status()["completed_episode_count"] == 1
    recorder.shutdown()


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.fail_next_latest_precondition = False
        self.latest_precondition_failures = 0

    def put_object(
        self,
        *,
        Bucket,
        Key,
        Body,
        Metadata=None,
        IfNoneMatch=None,
        IfMatch=None,
        **_kwargs,
    ):
        target = (Bucket, Key)
        if IfNoneMatch == "*" and target in self.objects:
            raise RuntimeError("precondition failed")
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        if IfMatch is not None:
            current = self.objects.get(target)
            current_etag = (
                '"' + __import__("hashlib").sha256(current[0]).hexdigest() + '"'
                if current is not None
                else ""
            )
            if current_etag != IfMatch:
                raise RuntimeError("precondition failed")
        if Key.endswith("/latest.json") and self.fail_next_latest_precondition:
            self.fail_next_latest_precondition = False
            self.latest_precondition_failures += 1
            raise RuntimeError("precondition failed")
        self.objects[target] = (data, dict(Metadata or {}))
        return {"ETag": '"' + __import__("hashlib").sha256(data).hexdigest() + '"'}

    def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
        contents = [
            {"Key": key, "Size": len(value[0])}
            for (bucket, key), value in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def get_object(self, *, Bucket, Key):
        data = self.objects[(Bucket, Key)][0]
        return {
            "Body": io.BytesIO(data),
            "ETag": '"' + __import__("hashlib").sha256(data).hexdigest() + '"',
        }

    def head_object(self, *, Bucket, Key):
        data, metadata = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Metadata": dict(metadata)}

    def download_file(self, bucket, key, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(self.objects[(bucket, key)][0])


def _episode_dir(
    root: Path, task: str, environment: str, number: int
) -> tuple[Path, dict]:
    episode = root / f"episode-{number}"
    frames = episode / "frames"
    frames.mkdir(parents=True)
    records = []
    for index, color in ((1, (200, 20, 20)), (2, (20, 200, 20)), (3, (20, 20, 200))):
        jpeg = _jpeg(color)
        (frames / f"frame-{index - 1:06d}.jpg").write_bytes(jpeg)
        row = _step(index)
        row.update(
            {
                "frame_sha256": __import__("hashlib").sha256(jpeg).hexdigest(),
                "task": task,
                "environment_id": environment,
                "environment_index": number,
                "seed": 40 + number,
            }
        )
        records.append(row)
    (episode / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    metadata = {
        "schema": "npa.leisaac.episode.v1",
        "episode_uuid": f"episode-{number}",
        "run_id": f"run-{number}",
        "task": task,
        "environment_id": environment,
        "environment_index": number,
        "seed": 40 + number,
        "outcome": "success" if number == 0 else "failure",
        "frame_count": len(records),
        "fps": 16,
        "source_commit": "1" * 40,
        "recorded_at": "2026-08-05T00:00:00Z",
    }
    (episode / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
    return episode, metadata


def _dual_episode_dir(root: Path) -> tuple[Path, dict]:
    episode, metadata = _episode_dir(
        root, "LeIsaac-SO101-PickOrange-v0", "kitchen-dual", 0
    )
    overview = episode / "frames-overview"
    overview.mkdir()
    for index, color in enumerate(((20, 20, 80), (40, 40, 120), (60, 60, 180))):
        (overview / f"frame-{index:06d}.jpg").write_bytes(_jpeg(color))
    metadata["cameras"] = ["workspace", "overview"]
    metadata["provenance"] = {
        "robot": "custom-so101",
        "scene": "custom-table",
        "device": "spacemouse",
        "bundle": "sha256-bundle",
    }
    (episode / "episode.json").write_text(json.dumps(metadata), encoding="utf-8")
    return episode, metadata


def test_s3_store_publishes_faststart_synchronized_two_camera_artifacts(
    tmp_path: Path,
) -> None:
    fake = _FakeS3()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=fake)
    result = store.publish_episode(*_dual_episode_dir(tmp_path))
    commit = json.loads(
        fake.objects[("bucket", "demos/leisaac/commits/episode-000000.json")][0]
    )
    assert set(commit["objects"]["videos"]) == {"workspace", "overview"}
    assert set(commit["media"]) == {"workspace", "overview"}
    assert all(item["codec"] == "h264" for item in commit["media"].values())
    assert all(item["frames"] == 3 for item in commit["media"].values())
    assert (
        commit["media"]["workspace"]["timestamps"]
        == commit["media"]["overview"]["timestamps"]
    )
    assert set(commit["objects"]["frames_by_camera"]) == {"workspace", "overview"}
    for camera, ref in commit["objects"]["videos"].items():
        content = fake.objects[("bucket", ref["key"])][0]
        assert content.find(b"moov") < content.find(b"mdat"), camera
        assert __import__("hashlib").sha256(content).hexdigest() == ref["sha256"]
    assert result["dataset_version_uri"].startswith(
        "s3://bucket/demos/leisaac/versions/"
    )


def test_packaged_ffmpeg_fallback_decodes_and_validates_media_without_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_which = leisaac_dataset.shutil.which
    monkeypatch.setattr(
        leisaac_dataset.shutil,
        "which",
        lambda name: None if name in {"ffmpeg", "ffprobe"} else original_which(name),
    )
    fake = _FakeS3()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=fake)
    result = store.publish_episode(*_dual_episode_dir(tmp_path))
    commit = json.loads(
        fake.objects[("bucket", "demos/leisaac/commits/episode-000000.json")][0]
    )
    assert result["episode_index"] == 0
    assert set(commit["media"]) == {"workspace", "overview"}
    assert all(item["codec"] == "h264" for item in commit["media"].values())
    assert all(item["pix_fmt"] == "yuv420p" for item in commit["media"].values())
    assert all(item["frames"] == 3 for item in commit["media"].values())
    assert all((item["width"], item["height"]) == (1280, 720) for item in commit["media"].values())
    assert commit["media"]["workspace"]["timestamps"] == commit["media"]["overview"]["timestamps"]


def test_s3_store_resumes_episode_numbers_and_publishes_lerobot_v3(
    tmp_path: Path,
) -> None:
    fake = _FakeS3()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=fake)
    first = _episode_dir(tmp_path, "LeIsaac-SO101-PickOrange-v0", "kitchen-a", 0)
    second = _episode_dir(tmp_path, "LeIsaac-SO101-LiftCube-v0", "table-b", 1)
    result0 = store.publish_episode(*first)
    retried0 = store.publish_episode(*first)
    fake.fail_next_latest_precondition = True
    result1 = store.publish_episode(*second)
    assert result0["episode_index"] == 0
    assert retried0["episode_index"] == 0
    assert retried0["completed_episode_count"] == 1
    assert retried0["dataset_version_uri"] == result0["dataset_version_uri"]
    assert result1["episode_index"] == 1
    assert result1["completed_episode_count"] == 2
    assert result0["dataset_version_uri"] != result1["dataset_version_uri"]
    assert fake.latest_precondition_failures == 1
    resumed = store.resume_status()
    assert resumed == {
        "completed_episode_count": 2,
        "last_episode_index": 1,
        "last_outcome": "failure",
        "last_upload_status": "uploaded",
        "dataset_version_uri": result1["dataset_version_uri"],
        "last_episode_commit_uri": (
            "s3://bucket/demos/leisaac/commits/episode-000001.json"
        ),
    }
    recovered_recorder = EpisodeRecorder(
        root=tmp_path / "recovered-recorder",
        output_uri="s3://bucket/demos/leisaac",
        task="LeIsaac-SO101-LiftCube-v0",
        environment_id="table-b",
        environment_index=1,
        seed=41,
        run_id="run-recovered",
        source_commit="1" * 40,
        publisher=store.publish_episode,
        status_loader=store.resume_status,
    )
    assert recovered_recorder.status()["completed_episode_count"] == 2
    assert recovered_recorder.status()["last_outcome"] == "failure"
    assert (
        recovered_recorder.status()["dataset_version_uri"]
        == result1["dataset_version_uri"]
    )
    commits = [
        key for bucket, key in fake.objects if bucket == "bucket" and "/commits/" in key
    ]
    assert commits == [
        "demos/leisaac/commits/episode-000000.json",
        "demos/leisaac/commits/episode-000001.json",
    ]
    version_prefix = result1["dataset_version_uri"].split("s3://bucket/", 1)[1]
    info = json.loads(fake.objects[("bucket", f"{version_prefix}/meta/info.json")][0])
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert info["total_tasks"] == 2
    assert info["features"]["observation.state"]["names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert info["features"]["action"]["names"][-2:] == [
        "delta_shoulder_pan",
        "delta_gripper",
    ]
    assert info["features"]["observation.images.front"]["info"]["video.codec"] == "h264"
    assert info["features"]["observation.images.front"]["shape"] == [720, 1280, 3]
    tasks_bytes = fake.objects[("bucket", f"{version_prefix}/meta/tasks.parquet")][0]
    tasks = pq.read_table(io.BytesIO(tasks_bytes))
    assert tasks["task"].to_pylist() == [
        "LeIsaac-SO101-LiftCube-v0",
        "LeIsaac-SO101-PickOrange-v0",
    ]
    assert tasks["task_index"].to_pylist() == [0, 1]
    parquet_bytes = fake.objects[
        ("bucket", f"{version_prefix}/data/chunk-000/file-001.parquet")
    ][0]
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.num_rows == 3
    assert table["environment.id"].to_pylist() == ["table-b"] * 3
    assert table["success"].to_pylist() == [False, False, False]
    assert table["reset_reason"].to_pylist()[-1] == "failure"

    first_only = store._commits()[:1]
    stale_result = store._publish_version(first_only)
    latest = json.loads(fake.objects[("bucket", "demos/leisaac/latest.json")][0])
    assert latest["episode_count"] == 2
    assert stale_result["episode_count"] == 2
    assert stale_result["dataset_uri"] == result1["dataset_version_uri"]


def test_s3_endpoint_precedence_is_explicit_then_env_then_config_then_primary() -> None:
    config = "https://config.example"
    env = {
        "AWS_ENDPOINT_URL_S3": "https://s3-env.example/",
        "NEBIUS_S3_ENDPOINT": "https://nebius-env.example",
        "AWS_ENDPOINT_URL": "https://aws-env.example",
    }
    assert (
        resolve_s3_endpoint(
            "https://explicit.example/", config_endpoint=config, environ=env
        )
        == "https://explicit.example"
    )
    assert (
        resolve_s3_endpoint(config_endpoint=config, environ=env)
        == "https://s3-env.example"
    )
    del env["AWS_ENDPOINT_URL_S3"]
    assert (
        resolve_s3_endpoint(config_endpoint=config, environ=env)
        == "https://nebius-env.example"
    )
    env.clear()
    assert resolve_s3_endpoint(config_endpoint=config, environ=env) == config
    assert resolve_s3_endpoint(environ={}) == "https://storage.eu-north1.nebius.cloud"


def test_immutable_object_upload_is_idempotent_but_rejects_colliding_bytes(
    tmp_path: Path,
) -> None:
    fake = _FakeS3()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=fake)
    path = tmp_path / "object.bin"
    path.write_bytes(b"first immutable bytes")
    first = store._put_file("episodes/fixed/object.bin", path)
    assert store._put_file("episodes/fixed/object.bin", path) == first
    path.write_bytes(b"different immutable bytes")
    with pytest.raises(DatasetError, match="different bytes"):
        store._put_file("episodes/fixed/object.bin", path)


def test_immutable_object_upload_collision_is_atomic_under_a_race(
    tmp_path: Path,
) -> None:
    class AtomicFakeS3(_FakeS3):
        def __init__(self) -> None:
            super().__init__()
            self.start = threading.Barrier(2)
            self.lock = threading.Lock()

        def put_object(self, **kwargs):
            if kwargs.get("IfNoneMatch") == "*":
                self.start.wait(timeout=5)
            with self.lock:
                return super().put_object(**kwargs)

    fake = AtomicFakeS3()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=fake)
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"left contender")
    right.write_bytes(b"right contender")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(store._put_file, "raw/fixed.bin", path)
            for path in (left, right)
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except DatasetError as exc:
            outcomes.append(exc)

    assert sum(isinstance(item, dict) for item in outcomes) == 1
    errors = [item for item in outcomes if isinstance(item, DatasetError)]
    assert len(errors) == 1
    assert "different bytes" in str(errors[0])
    stored = fake.objects[("bucket", "demos/leisaac/raw/fixed.bin")][0]
    assert stored in {b"left contender", b"right contender"}


def test_multi_camera_publication_rejects_frame_count_misalignment(
    tmp_path: Path,
) -> None:
    episode, metadata = _dual_episode_dir(tmp_path)
    (episode / "frames-overview/frame-000002.jpg").unlink()
    store = S3DatasetStore("s3://bucket/demos/leisaac", client=_FakeS3())
    with pytest.raises(DatasetError, match="frames and timestamps"):
        store.publish_episode(episode, metadata)

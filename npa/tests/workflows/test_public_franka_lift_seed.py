from __future__ import annotations

import io
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.orchestration.npa_workflow.presets import (
    PUBLIC_FRANKA_LIFT_DATASET_ID,
    PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY,
    PUBLIC_FRANKA_LIFT_DATASET_REVISION,
)
from npa.workflows.sim2real.public_seed import (
    SOURCE_PATHS,
    PublicSeedError,
    stage_public_franka_lift,
)
from npa.workflows.sim2real.task_contract import (
    build_task_contract,
    validate_seed_dataset_manifest,
)


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local_file: str, uri: str) -> str:
        self.objects[uri] = Path(local_file).read_bytes()
        return uri


def _parquet_bytes(rows: int = 3) -> bytes:
    sink = io.BytesIO()
    values = [[float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for index in range(rows)]
    pq.write_table(
        pa.table({"actions": pa.array(values, type=pa.list_(pa.float32(), 7))}),
        sink,
    )
    return sink.getvalue()


def _video_bytes(tmp_path: Path, name: str) -> bytes:
    target = tmp_path / name
    with av.open(str(target), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (25, 75, 125, 225):
            array = np.full((16, 16, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return target.read_bytes()


def _fetcher(tmp_path: Path, *, revision: str = PUBLIC_FRANKA_LIFT_DATASET_REVISION):
    info = {
        "robot_type": "franka",
        "features": {
            "actions": {"dtype": "float32", "shape": [7]},
            "image": {"dtype": "video", "shape": [224, 224, 3]},
            "wrist_image": {"dtype": "video", "shape": [224, 224, 3]},
        },
        "source_rollout_meta": {"task": "Isaac-Lift-Cube-Franka-IK-Rel-v0"},
    }
    payloads = {
        SOURCE_PATHS["metadata"]: json.dumps(info).encode(),
        SOURCE_PATHS["tasks"]: b'{"task_index":0,"task":"lift the cube"}\n',
        SOURCE_PATHS["episodes"]: b'{"episode_index":0,"tasks":["lift the cube"],"length":3}\n',
        SOURCE_PATHS["actions"]: _parquet_bytes(),
        SOURCE_PATHS["main_video"]: _video_bytes(tmp_path, "main.mp4"),
        SOURCE_PATHS["wrist_video"]: _video_bytes(tmp_path, "wrist.mp4"),
    }
    api = {
        "sha": revision,
        "private": False,
        "gated": False,
        "cardData": {"license": "apache-2.0"},
        "siblings": [{"rfilename": path} for path in SOURCE_PATHS.values()],
    }

    def fetch(url: str) -> bytes:
        if "/api/datasets/" in url:
            return json.dumps(api).encode()
        for path, payload in payloads.items():
            if f"/{path}?download=true" in url:
                return payload
        raise AssertionError(f"unexpected hermetic fetch URL: {url}")

    return fetch


def test_stage_public_franka_lift_derives_real_subset_counts_and_hashes(
    tmp_path: Path,
) -> None:
    storage = MemoryStorage()
    result = stage_public_franka_lift(
        bucket="unit-bucket",
        run_id="unit-run",
        client=storage,  # type: ignore[arg-type]
        fetch=_fetcher(tmp_path),
    )

    assert result["dataset_id"] == PUBLIC_FRANKA_LIFT_DATASET_ID
    assert result["action_count"] == 3
    assert result["camera_observation_count"] == 4
    frame_names = sorted(
        uri.rsplit("/", 1)[-1]
        for uri in storage.objects
        if "/frames/" in uri
    )
    assert frame_names == [
        "camera-000.png",
        "camera-001.png",
        "camera-002.png",
        "camera-003.png",
    ]
    assert all(
        name.removeprefix("camera-").removesuffix(".png").isdigit()
        for name in frame_names
    )
    manifest = json.loads(storage.objects[result["seed_manifest_uri"]])
    frame_records = [
        record
        for record in manifest["source_provenance"]["objects"]
        if "/frames/" in record["uri"]
    ]
    assert [
        (record["source_camera"], record["source_frame_index"])
        for record in frame_records
    ] == [
        ("image", 0),
        ("image", 1),
        ("image", 2),
        ("image", 3),
    ]
    assert manifest["source_provenance"]["repository"] == PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY
    assert manifest["source_provenance"]["revision"] == PUBLIC_FRANKA_LIFT_DATASET_REVISION
    assert manifest["source_contract"]["action"]["dimensions"] == 7
    assert manifest["source_contract"]["cameras"] == ["image", "wrist_image"]
    assert manifest["compatibility_boundary"]["canonical_action"]["dimensions"] == 8
    assert manifest["compatibility_boundary"]["canonical_cameras"] == [
        "primary",
        "side",
        "overhead",
    ]
    assert manifest["compatibility_boundary"]["source_actions_reused_as_canonical_ppo_actions"] is False
    for record in manifest["source_provenance"]["objects"]:
        assert len(record["sha256"]) == 64
        assert record["bytes"] == len(storage.objects[record["uri"]])

    contract = build_task_contract(
        task_id=result["task_id"],
        dataset_id=result["dataset_id"],
        dataset_uri=result["trigger_uri"],
    )
    listed = [
        {
            "Bucket": "unit-bucket",
            "Key": uri.removeprefix("s3://unit-bucket/"),
            "Size": len(payload),
        }
        for uri, payload in storage.objects.items()
    ]
    proof = validate_seed_dataset_manifest(
        manifest, contract=contract, trigger_objects=listed
    )
    assert proof["preset"] == "public-franka-lift"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda api: api.update(sha="different"), "revision drifted"),
        (lambda api: api["cardData"].update(license="other"), "license mismatch"),
        (lambda api: api.update(gated=True), "anonymously accessible"),
    ],
)
def test_stage_public_franka_lift_fails_closed_on_repository_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    base = _fetcher(tmp_path)

    def fetch(url: str) -> bytes:
        payload = base(url)
        if "/api/datasets/" not in url:
            return payload
        api = json.loads(payload)
        mutation(api)
        return json.dumps(api).encode()

    with pytest.raises(PublicSeedError, match=message):
        stage_public_franka_lift(
            bucket="unit-bucket",
            run_id="unit-run",
            client=MemoryStorage(),  # type: ignore[arg-type]
            fetch=fetch,
        )


def test_stage_public_franka_lift_fails_closed_on_upload(tmp_path: Path) -> None:
    class BrokenStorage(MemoryStorage):
        def upload_file(self, local_file: str, uri: str) -> str:
            raise OSError("denied")

    with pytest.raises(PublicSeedError, match="upload failed"):
        stage_public_franka_lift(
            bucket="unit-bucket",
            run_id="unit-run",
            client=BrokenStorage(),  # type: ignore[arg-type]
            fetch=_fetcher(tmp_path),
        )


def test_stage_public_franka_lift_fails_closed_on_missing_source_path(
    tmp_path: Path,
) -> None:
    base = _fetcher(tmp_path)

    def fetch(url: str) -> bytes:
        payload = base(url)
        if "/api/datasets/" not in url:
            return payload
        api = json.loads(payload)
        api["siblings"] = [
            item
            for item in api["siblings"]
            if item["rfilename"] != SOURCE_PATHS["actions"]
        ]
        return json.dumps(api).encode()

    with pytest.raises(PublicSeedError, match="missing required paths"):
        stage_public_franka_lift(
            bucket="unit-bucket",
            run_id="unit-run",
            client=MemoryStorage(),  # type: ignore[arg-type]
            fetch=fetch,
        )


def test_stage_public_franka_lift_fails_closed_on_decode_failure(
    tmp_path: Path,
) -> None:
    base = _fetcher(tmp_path)

    def fetch(url: str) -> bytes:
        if SOURCE_PATHS["main_video"] in url:
            return b"not-a-video"
        return base(url)

    with pytest.raises(PublicSeedError, match="could not decode"):
        stage_public_franka_lift(
            bucket="unit-bucket",
            run_id="unit-run",
            client=MemoryStorage(),  # type: ignore[arg-type]
            fetch=fetch,
        )

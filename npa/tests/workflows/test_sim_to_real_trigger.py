from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import yaml

from npa.workbench import trigger as trigger_sdk
from npa.workflows.sim_to_real_trigger import (
    LocalWatermarkStore,
    PipelineLaunch,
    TriggerConfig,
    TriggerObject,
    list_lerobot_objects,
    run_once,
)


ROOT = Path(__file__).resolve().parents[3]
TRIGGER_YAML = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "sim-to-real-trigger.yaml"


class MissingObject(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}

    def add(
        self,
        bucket: str,
        key: str,
        body: bytes = b"",
        *,
        last_modified: datetime,
        etag: str = "etag",
    ) -> None:
        self.objects[(bucket, key)] = {
            "Body": body,
            "LastModified": last_modified,
            "ETag": etag,
            "Size": len(body),
        }

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ):
        del ContinuationToken
        contents = [
            {
                "Key": key,
                "LastModified": item["LastModified"],
                "ETag": item["ETag"],
                "Size": item["Size"],
            }
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"IsTruncated": False, "Contents": contents}

    def get_object(self, *, Bucket: str, Key: str):
        try:
            item = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise MissingObject() from exc
        return {"Body": BytesIO(item["Body"])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = {
            "Body": Body,
            "LastModified": datetime.now(timezone.utc),
            "ETag": "watermark",
            "Size": len(Body),
        }


@dataclass
class RecordingLauncher:
    launches: list[tuple[TriggerConfig, tuple[TriggerObject, ...]]]

    def launch(self, config: TriggerConfig, objects: tuple[TriggerObject, ...]) -> PipelineLaunch:
        self.launches.append((config, objects))
        return PipelineLaunch(
            run_id=f"run-{len(self.launches)}",
            status="launched",
            input_data_uri=config.input_data_uri,
        )


def _ts(second: int) -> datetime:
    return datetime(2026, 6, 4, 12, 0, second, tzinfo=timezone.utc)


def _config() -> TriggerConfig:
    return TriggerConfig(
        s3_endpoint="https://s3.example.invalid",
        s3_bucket="bucket",
        s3_prefix="datasets/lerobot-pusht/",
        pipeline_s3_prefix="sim-to-real/{run_id}",
    )


def test_lists_only_lerobot_format_objects_under_prefix() -> None:
    fake = FakeS3()
    fake.add("bucket", "datasets/lerobot-pusht/meta/info.json", b"{}", last_modified=_ts(1))
    fake.add(
        "bucket",
        "datasets/lerobot-pusht/data/chunk-000/episode_000000.parquet",
        b"parquet",
        last_modified=_ts(2),
    )
    fake.add("bucket", "datasets/lerobot-pusht/notes.txt", b"ignore", last_modified=_ts(3))
    fake.add("bucket", "other/meta/info.json", b"ignore", last_modified=_ts(4))

    objects = list_lerobot_objects(_config(), s3_client=fake)

    assert [obj.key for obj in objects] == [
        "datasets/lerobot-pusht/meta/info.json",
        "datasets/lerobot-pusht/data/chunk-000/episode_000000.parquet",
    ]


def test_run_once_launches_once_and_does_not_double_fire(tmp_path: Path) -> None:
    fake = FakeS3()
    fake.add("bucket", "datasets/lerobot-pusht/meta/info.json", b"{}", last_modified=_ts(1))
    store = LocalWatermarkStore(tmp_path / "watermark.json")
    launcher = RecordingLauncher([])

    first = run_once(_config(), s3_client=fake, watermark_store=store, launcher=launcher)
    second = run_once(_config(), s3_client=fake, watermark_store=store, launcher=launcher)

    assert first.status == "triggered"
    assert second.status == "idle"
    assert len(launcher.launches) == 1
    assert launcher.launches[0][0].s3_endpoint == "https://s3.example.invalid"
    assert launcher.launches[0][0].input_data_uri == "s3://bucket/datasets/lerobot-pusht/"

    fake.add(
        "bucket",
        "datasets/lerobot-pusht/data/chunk-000/episode_000001.parquet",
        b"new",
        last_modified=_ts(2),
    )
    third = run_once(_config(), s3_client=fake, watermark_store=store, launcher=launcher)

    assert third.status == "triggered"
    assert third.new_object_count == 1
    assert len(launcher.launches) == 2


def test_sdk_run_once_honors_byo_endpoint_config(tmp_path: Path) -> None:
    fake = FakeS3()
    fake.add("bucket", "datasets/lerobot-pusht/meta/info.json", b"{}", last_modified=_ts(1))
    launcher = RecordingLauncher([])

    result = trigger_sdk.run_once(
        s3_endpoint="https://byo-s3.example.invalid",
        s3_bucket="bucket",
        s3_prefix="datasets/lerobot-pusht/",
        watermark_uri=str(tmp_path / "watermark.json"),
        pipeline_render_only=True,
        s3_client=fake,
        launcher=launcher,
    )

    assert result.status == "triggered"
    assert launcher.launches[0][0].s3_endpoint == "https://byo-s3.example.invalid"
    assert launcher.launches[0][0].pipeline_render_only is True


def test_standalone_trigger_yaml_is_byo_endpoint_and_references_envs() -> None:
    docs = [doc for doc in yaml.safe_load_all(TRIGGER_YAML.read_text(encoding="utf-8")) if doc is not None]
    task = docs[0]
    text = TRIGGER_YAML.read_text(encoding="utf-8")

    assert task["envs"]["NPA_TRIGGER_S3_ENDPOINT"] == "https://s3.example.invalid"
    assert task["envs"]["NPA_TRIGGER_S3_BUCKET"] == "example-bucket"
    assert task["envs"]["NPA_TRIGGER_PIPELINE_S3_PREFIX"] == "sim-to-real/{run_id}"
    assert "--s3-endpoint" in task["run"]
    assert "--pipeline-input-data-uri" in task["run"]
    assert "--" + "down" not in text
    assert "auto" + "down" not in text.lower()
    assert "nebius.cloud" not in text

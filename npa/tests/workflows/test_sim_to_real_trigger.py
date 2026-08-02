from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path


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


def test_the_watcher_launches_the_spec_not_the_retired_pipeline_template() -> None:
    """`sim-to-real-pipeline.yaml` ran `npa.workflows.sim_to_real real-loop`.

    That module raises a `DeprecationWarning` pointing at the staged sim2real engine, so the
    template was retired rather than ported — wrapping a deprecated path in a new spec would
    make the new surface its home. Watching a bucket is NOT deprecated, so the watcher stays
    and now submits the staged loop's own spec.
    """

    from npa.workflows.sim_to_real_trigger import _pipeline_command

    config = TriggerConfig(
        s3_endpoint="https://s3.example.invalid",
        s3_bucket="example-bucket",
        s3_prefix="datasets/lerobot-pusht/",
        pipeline_bucket="example-bucket",
        pipeline_s3_prefix="sim-to-real/{run_id}",
        pipeline_input_data_uri="s3://example-bucket/datasets/lerobot-pusht/",
    )

    command = _pipeline_command(config, run_id="trigger-001")

    assert command[:4] == ["npa", "workbench", "workflow", "submit"]
    assert command[4].endswith("npa-workflows/sim2real-vlm-rl.yaml")
    assert "run_sim_to_real_pipeline" not in " ".join(command)
    # The trigger prefix the watch fired on is what the spec's first stage reads.
    assert "trigger_uri=s3://example-bucket/datasets/lerobot-pusht/" in command
    assert "bucket=example-bucket" in command
    assert "prefix=sim-to-real/trigger-001" in command


def test_render_only_validates_instead_of_submitting() -> None:
    """A dry run must not reach a cluster, and must not carry submit-only options."""

    from npa.workflows.sim_to_real_trigger import _pipeline_command

    config = TriggerConfig(
        s3_endpoint="https://s3.example.invalid",
        s3_bucket="example-bucket",
        s3_prefix="datasets/lerobot-pusht/",
        pipeline_render_only=True,
        sky_bin="/usr/local/bin/sky",
    )

    command = _pipeline_command(config, run_id="trigger-002")

    assert command[:4] == ["npa", "workbench", "workflow", "validate-spec"]
    assert "--submit-timeout" not in command
    assert "--controller-backend" not in command
    assert "--sky-bin" not in command


def test_the_retired_templates_are_gone() -> None:
    skypilot = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
    assert not (skypilot / "sim-to-real-trigger.yaml").exists()
    assert not (skypilot / "sim-to-real-pipeline.yaml").exists()

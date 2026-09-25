from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workflows.sim_to_real_trigger import (
    PipelineLaunch,
    SimToRealTriggerError,
    TriggerConfig,
    TriggerResult,
    TriggerWatermark,
    list_lerobot_objects,
    run_once,
)


runner = CliRunner()


def _object(key: str, second: int) -> dict[str, Any]:
    return {
        "Key": key,
        "LastModified": datetime(2026, 6, 4, 12, 0, second, tzinfo=timezone.utc),
        "ETag": f'"etag-{second}"',
        "Size": second,
    }


def _trigger_config() -> TriggerConfig:
    return TriggerConfig(
        s3_endpoint="https://s3.example.invalid",
        s3_bucket="bucket",
        s3_prefix="datasets/lerobot/",
    )


class FallbackS3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = iter(pages)
        self.calls: list[dict[str, str]] = []

    def list_objects_v2(self, **kwargs: str) -> dict[str, Any]:
        self.calls.append(kwargs)
        return next(self.pages)


class NativePaginatorS3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages

    def get_paginator(self, operation: str):
        assert operation == "list_objects_v2"
        pages = self.pages

        class Paginator:
            def paginate(self, **kwargs: str):
                assert kwargs == {"Bucket": "bucket", "Prefix": "datasets/lerobot"}
                return iter(pages)

        return Paginator()


class RecordingStore:
    def __init__(self) -> None:
        self.saved: list[TriggerWatermark] = []

    def load(self) -> TriggerWatermark:
        return TriggerWatermark()

    def save(self, watermark: TriggerWatermark) -> None:
        self.saved.append(watermark)


class RejectLaunch:
    def launch(self, config, objects):
        raise AssertionError("malformed discovery must not launch a workflow")


def test_workbench_trigger_help() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "trigger", "--help"])

    assert result.exit_code == 0
    assert "retrigger Workbench workflows" in result.output


def test_workbench_trigger_run_passes_byo_endpoint_and_paths(monkeypatch) -> None:
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return TriggerResult(
            status="triggered",
            watched_uri=config.input_data_uri,
            watermark_uri=config.effective_watermark_uri,
            new_object_count=1,
            new_objects=(),
            launch=PipelineLaunch(
                run_id="run-1",
                status="launched",
                input_data_uri=config.input_data_uri,
            ),
            watermark=TriggerWatermark(
                cursor_last_modified="2026-06-04T12:00:00Z", launches=1
            ),
            generated_at="2026-06-04T12:00:01Z",
        )

    monkeypatch.setattr("npa.cli.workbench.trigger.run_trigger_once", fake_run)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "trigger",
            "run",
            "--s3-endpoint",
            "https://byo-s3.example.invalid",
            "--s3-bucket",
            "bucket",
            "--s3-prefix",
            "datasets/lerobot-pusht/",
            "--watermark-uri",
            "s3://bucket/datasets/lerobot-pusht/.npa/watermark.json",
            "--pipeline-s3-prefix",
            "sim-to-real/{run_id}",
            "--pipeline-input-data-uri",
            "s3://bucket/datasets/lerobot-pusht/",
            "--pipeline-render-only",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "triggered"
    assert captured["config"].s3_endpoint == "https://byo-s3.example.invalid"
    assert captured["config"].s3_bucket == "bucket"
    assert captured["config"].s3_prefix == "datasets/lerobot-pusht/"
    assert captured["config"].pipeline_s3_prefix == "sim-to-real/{run_id}"
    assert captured["config"].pipeline_render_only is True


@pytest.mark.parametrize(
    "malformed_pages, expected_message",
    [
        (
            [
                {
                    "IsTruncated": True,
                    "Contents": [_object("datasets/lerobot/meta/info.json", 1)],
                }
            ],
            "did not provide a continuation token",
        ),
        (
            [
                {
                    "IsTruncated": True,
                    "NextContinuationToken": "page-2",
                    "Contents": [_object("datasets/lerobot/meta/info.json", 1)],
                },
                {
                    "IsTruncated": True,
                    "NextContinuationToken": "page-2",
                    "Contents": [
                        _object("datasets/lerobot/data/chunk-000/episode.parquet", 2)
                    ],
                },
            ],
            "repeated a continuation token",
        ),
    ],
)
def test_malformed_fallback_pagination_cannot_launch_or_save_watermark(
    malformed_pages, expected_message
) -> None:
    client = FallbackS3(malformed_pages)
    store = RecordingStore()

    with pytest.raises(SimToRealTriggerError, match=expected_message):
        run_once(
            _trigger_config(),
            s3_client=client,
            watermark_store=store,
            launcher=RejectLaunch(),
        )

    assert store.saved == []
    assert len(client.calls) == len(malformed_pages)


def test_fallback_single_page_preserves_filtering_and_ordering() -> None:
    client = FallbackS3(
        [
            {
                "IsTruncated": False,
                "Contents": [
                    _object("datasets/lerobot/data/chunk-000/episode.parquet", 2),
                    _object("datasets/lerobot/notes.txt", 3),
                    _object("datasets/lerobot/meta/info.json", 1),
                ],
            }
        ]
    )

    objects = list_lerobot_objects(_trigger_config(), s3_client=client)

    assert [item.key for item in objects] == [
        "datasets/lerobot/meta/info.json",
        "datasets/lerobot/data/chunk-000/episode.parquet",
    ]


def test_fallback_multiple_pages_preserve_filtering_and_ordering() -> None:
    client = FallbackS3(
        [
            {
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
                "Contents": [
                    _object("datasets/lerobot/data/chunk-000/episode.parquet", 2),
                    _object("datasets/lerobot/notes.txt", 3),
                ],
            },
            {
                "IsTruncated": False,
                "Contents": [_object("datasets/lerobot/meta/info.json", 1)],
            },
        ]
    )

    objects = list_lerobot_objects(_trigger_config(), s3_client=client)

    assert client.calls[1]["ContinuationToken"] == "page-2"
    assert [item.key for item in objects] == [
        "datasets/lerobot/meta/info.json",
        "datasets/lerobot/data/chunk-000/episode.parquet",
    ]


def test_native_paginator_preserves_filtering_and_ordering() -> None:
    client = NativePaginatorS3(
        [
            {
                "Contents": [
                    _object("datasets/lerobot/data/chunk-000/episode.parquet", 2),
                    _object("datasets/lerobot/notes.txt", 3),
                ]
            },
            {"Contents": [_object("datasets/lerobot/meta/info.json", 1)]},
        ]
    )

    objects = list_lerobot_objects(_trigger_config(), s3_client=client)

    assert [item.key for item in objects] == [
        "datasets/lerobot/meta/info.json",
        "datasets/lerobot/data/chunk-000/episode.parquet",
    ]

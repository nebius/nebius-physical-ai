from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workflows.sim_to_real_trigger import PipelineLaunch, TriggerResult, TriggerWatermark


runner = CliRunner()


def test_workbench_trigger_help() -> None:
    result = runner.invoke(app, ["workbench", "trigger", "--help"])

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
            watermark=TriggerWatermark(cursor_last_modified="2026-06-04T12:00:00Z", launches=1),
            generated_at="2026-06-04T12:00:01Z",
        )

    monkeypatch.setattr("npa.cli.workbench.trigger.run_trigger_once", fake_run)

    result = runner.invoke(
        app,
        [
            "workbench",
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

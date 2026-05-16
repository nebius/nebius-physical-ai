from __future__ import annotations

from typer.testing import CliRunner

from npa.cli.main import app
from npa.viz.adapters.lerobot_to_rerun import RerunAdapterError


runner = CliRunner()


def test_convert_help_smoke() -> None:
    result = runner.invoke(app, ["convert", "--help"])

    assert result.exit_code == 0
    assert "standalone formats" in result.output
    assert "lerobot-to-rrd" in result.output
    assert "lerobot-to-mp4" in result.output


def test_convert_lerobot_to_rrd_help_smoke() -> None:
    result = runner.invoke(app, ["convert", "lerobot-to-rrd", "--help"])

    assert result.exit_code == 0
    assert "--input-path" in result.output
    assert "--output-path" in result.output
    assert "--predictions-path" in result.output


def test_convert_lerobot_to_rrd_dispatches_input_only(mocker) -> None:
    lerobot = mocker.patch("npa.cli.convert.lerobot_to_rrd.lerobot_to_rerun")
    overlay = mocker.patch("npa.cli.convert.lerobot_to_rrd.groot_predictions_to_rerun")

    result = runner.invoke(
        app,
        [
            "convert",
            "lerobot-to-rrd",
            "--input-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/visuals/out.rrd",
            "--duration",
            "5",
        ],
    )

    assert result.exit_code == 0
    lerobot.assert_called_once_with(
        "s3://bucket/dataset/",
        "s3://bucket/visuals/out.rrd",
        duration_s=5.0,
    )
    overlay.assert_not_called()
    assert "Conversion complete" in result.output


def test_convert_lerobot_to_rrd_dispatches_overlay(mocker) -> None:
    lerobot = mocker.patch("npa.cli.convert.lerobot_to_rrd.lerobot_to_rerun")
    overlay = mocker.patch("npa.cli.convert.lerobot_to_rrd.groot_predictions_to_rerun")

    result = runner.invoke(
        app,
        [
            "convert",
            "lerobot-to-rrd",
            "--input-path",
            "/tmp/dataset",
            "--output-path",
            "/tmp/overlay.rrd",
            "--predictions-path",
            "/tmp/predictions.json",
        ],
    )

    assert result.exit_code == 0
    overlay.assert_called_once_with(
        "/tmp/predictions.json",
        "/tmp/dataset",
        "/tmp/overlay.rrd",
        duration_s=None,
    )
    lerobot.assert_not_called()


def test_convert_lerobot_to_rrd_maps_adapter_error(mocker) -> None:
    mocker.patch(
        "npa.cli.convert.lerobot_to_rrd.lerobot_to_rerun",
        side_effect=RerunAdapterError("bad recording"),
    )

    result = runner.invoke(
        app,
        [
            "convert",
            "lerobot-to-rrd",
            "--input-path",
            "/tmp/dataset",
            "--output-path",
            "/tmp/out.rrd",
        ],
    )

    assert result.exit_code == 1
    assert "bad recording" in result.output

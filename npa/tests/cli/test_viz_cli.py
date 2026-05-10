from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.viz.lerobot import (
    VizRenderResult,
    _finalize_output_path,
    _materialize_lerobot_path,
    _materialize_predictions_path,
    _prepare_output_path,
)


runner = CliRunner()


def test_viz_help_smoke() -> None:
    result = runner.invoke(app, ["viz", "--help"])

    assert result.exit_code == 0
    assert "visualization" in result.output.lower()
    assert "lerobot" in result.output


def test_viz_lerobot_help_smoke() -> None:
    result = runner.invoke(app, ["viz", "lerobot", "--help"])

    assert result.exit_code == 0
    assert "--backend" in result.output
    assert "--layout" in result.output
    assert "--resolution" in result.output


def test_lerobot_cli_dispatches_to_matplotlib_backend(tmp_path: Path, mocker) -> None:
    output = tmp_path / "out.mp4"
    render = mocker.patch(
        "npa.cli.viz.lerobot.render_lerobot",
        return_value=VizRenderResult(
            local_path=output,
            saved_to=str(output),
            duration_s=5.0,
            resolution=(640, 360),
            fps=12,
            frame_count=60,
        ),
    )

    result = runner.invoke(
        app,
        [
            "viz",
            "lerobot",
            "--input-path",
            str(tmp_path / "dataset"),
            "--backend",
            "matplotlib",
            "--layout",
            "single",
            "--output-path",
            str(output),
            "--duration",
            "5",
            "--resolution",
            "640x360",
            "--fps",
            "12",
        ],
    )

    assert result.exit_code == 0
    render.assert_called_once_with(
        input_path=str(tmp_path / "dataset"),
        backend="matplotlib",
        predictions_path=None,
        layout="single",
        output_path=str(output),
        duration_s=5.0,
        resolution="640x360",
        fps=12,
        title="",
    )


def test_lerobot_cli_dispatches_layout_values(tmp_path: Path, mocker) -> None:
    output = tmp_path / "out.mp4"
    render = mocker.patch(
        "npa.cli.viz.lerobot.render_lerobot",
        return_value=VizRenderResult(
            local_path=output,
            saved_to=str(output),
            duration_s=1.0,
            resolution=(1280, 720),
            fps=30,
            frame_count=30,
        ),
    )

    result = runner.invoke(
        app,
        [
            "viz",
            "lerobot",
            "--input-path",
            str(tmp_path / "dataset"),
            "--predictions-path",
            str(tmp_path / "predictions.json"),
            "--layout",
            "overlay",
        ],
    )

    assert result.exit_code == 0
    assert render.call_args.kwargs["layout"] == "overlay"
    assert render.call_args.kwargs["predictions_path"] == str(tmp_path / "predictions.json")


def test_rerun_backend_gives_clean_error() -> None:
    result = runner.invoke(
        app,
        [
            "viz",
            "lerobot",
            "--input-path",
            "missing",
            "--backend",
            "rerun",
        ],
    )

    assert result.exit_code == 1
    assert "Rerun backend is not implemented yet" in result.output


def test_s3_input_and_predictions_are_materialized(tmp_path: Path, mocker) -> None:
    storage = mocker.Mock()
    storage.download_directory.return_value = str(tmp_path / "downloaded-dataset")
    storage.download_path.return_value = str(tmp_path / "downloaded-predictions")
    from_env = mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=storage)
    temp_dirs = []

    dataset = _materialize_lerobot_path("s3://bucket/dataset/", temp_dirs)
    predictions = _materialize_predictions_path("s3://bucket/predictions/", temp_dirs)

    assert dataset == tmp_path / "downloaded-dataset"
    assert predictions == tmp_path / "downloaded-predictions"
    storage.download_directory.assert_called_once_with("s3://bucket/dataset/", temp_dirs[0].name)
    storage.download_path.assert_called_once_with("s3://bucket/predictions/", temp_dirs[1].name)
    assert from_env.call_count == 2
    for temp_dir in temp_dirs:
        temp_dir.cleanup()


def test_output_path_local_and_s3_upload(tmp_path: Path, mocker) -> None:
    temp_dirs = []
    local_output = _prepare_output_path(str(tmp_path / "local" / "out.mp4"), temp_dirs)
    assert local_output == tmp_path / "local" / "out.mp4"
    assert local_output.parent.exists()

    s3_local = _prepare_output_path("s3://bucket/visuals/out.mp4", temp_dirs)
    s3_local.write_bytes(b"mp4")
    storage = mocker.Mock()
    storage.upload_file.return_value = "s3://bucket/visuals/out.mp4"
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=storage)

    assert _finalize_output_path(s3_local, "s3://bucket/visuals/out.mp4") == "s3://bucket/visuals/out.mp4"
    storage.upload_file.assert_called_once_with(str(s3_local), "s3://bucket/visuals/out.mp4")
    for temp_dir in temp_dirs:
        temp_dir.cleanup()


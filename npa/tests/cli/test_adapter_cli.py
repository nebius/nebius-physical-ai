from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from npa.adapter.sim_to_lerobot import AdapterError
from npa.cli.main import app


runner = CliRunner()


def test_adapter_convert_help() -> None:
    result = runner.invoke(app, ["adapter", "convert", "--help"])

    assert result.exit_code == 0
    assert "Convert Genesis/sim demo numpy arrays" in result.output


def test_adapter_convert_dispatches_to_adapter(
    tmp_path: Path, mocker
) -> None:
    input_dir = tmp_path / "demos"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    convert_mock = mocker.patch(
        "npa.adapter.sim_to_lerobot.convert", return_value=output_dir
    )

    result = runner.invoke(
        app,
        [
            "adapter",
            "convert",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--fps",
            "30",
            "--robot",
            "testbot",
            "--task",
            "test task",
        ],
    )

    assert result.exit_code == 0
    convert_mock.assert_called_once_with(
        input_dir,
        output_dir,
        fps=30,
        robot_type="testbot",
        task="test task",
    )


def test_adapter_convert_accepts_standard_path_aliases(
    tmp_path: Path, mocker
) -> None:
    input_dir = tmp_path / "demos"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    convert_mock = mocker.patch(
        "npa.adapter.sim_to_lerobot.convert", return_value=output_dir
    )

    result = runner.invoke(
        app,
        [
            "adapter",
            "convert",
            "--input-path",
            str(input_dir),
            "--output-path",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    convert_mock.assert_called_once()
    assert convert_mock.call_args.args[:2] == (input_dir, output_dir)


def test_adapter_convert_missing_input_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "adapter",
            "convert",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "Input directory does not exist" in result.output


def test_adapter_convert_adapter_error_exits(tmp_path: Path, mocker) -> None:
    input_dir = tmp_path / "demos"
    input_dir.mkdir()
    mocker.patch(
        "npa.adapter.sim_to_lerobot.convert",
        side_effect=AdapterError("bad demos"),
    )

    result = runner.invoke(
        app,
        [
            "adapter",
            "convert",
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "bad demos" in result.output

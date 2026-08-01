"""CLI coverage for `npa workbench foxglove` (no network, no infra)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.foxglove import FOXGLOVE_EMBED_SDK_VERSION

runner = CliRunner()


def _make_run(tmp_path: Path) -> Path:
    from PIL import Image

    root = tmp_path / "run"
    (root / "camera" / "front").mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (8, 8), (index * 40, 60, 90)).save(
            root / "camera" / "front" / f"{index}.png"
        )
    (root / "metrics.json").write_text('{"success_rate": 0.5}', encoding="utf-8")
    return root


def test_foxglove_help_lists_commands() -> None:
    result = runner.invoke(app, ["workbench", "foxglove", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("convert-run", "inspect", "install-sdk", "config"):
        assert command in result.output


def test_convert_run_help_documents_synthetic_timestamps() -> None:
    result = runner.invoke(app, ["workbench", "foxglove", "convert-run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--input-path" in result.output
    assert "--output-path" in result.output
    assert "fps" in result.output.lower()


def test_config_reports_embed_settings(monkeypatch) -> None:
    monkeypatch.setenv("NPA_FOXGLOVE_EMBED_SRC", "https://foxglove.internal.example/")
    monkeypatch.setenv("NPA_FOXGLOVE_ORG_SLUG", "acme")
    result = runner.invoke(app, ["workbench", "foxglove", "config", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["embed_src"] == "https://foxglove.internal.example/"
    assert payload["org_slug"] == "acme"
    assert payload["sdk_version"] == FOXGLOVE_EMBED_SDK_VERSION
    assert payload["service_port"] == 8099
    assert "MIT-licensed" in payload["note"]


def test_convert_run_and_inspect_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    pytest.importorskip("PIL")
    root = _make_run(tmp_path)
    output = tmp_path / "session.mcap"

    convert = runner.invoke(
        app,
        [
            "workbench",
            "foxglove",
            "convert-run",
            "--input-path",
            str(root),
            "--output-path",
            str(output),
            "--run-id",
            "cli-run",
            "--output",
            "json",
        ],
    )
    assert convert.exit_code == 0, convert.output
    summary = json.loads(convert.output)
    assert summary["frames"] == 2
    assert summary["metrics"] == 1
    assert summary["timestamps"] == "synthetic-fps"
    assert output.is_file()

    inspected = runner.invoke(
        app,
        ["workbench", "foxglove", "inspect", "--input-path", str(output), "--output", "json"],
    )
    assert inspected.exit_code == 0, inspected.output
    info = json.loads(inspected.output)
    assert info["message_count"] == 3
    assert info["schemas"]["/camera/front"] == "foxglove.CompressedImage"
    assert info["metadata"]["npa"]["run_id"] == "cli-run"


def test_convert_run_fails_cleanly_on_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "foxglove",
            "convert-run",
            "--input-path",
            str(tmp_path / "missing"),
            "--output-path",
            str(tmp_path / "out.mcap"),
        ],
    )
    assert result.exit_code == 1
    assert "MCAP conversion failed" in result.output


def test_inspect_rejects_non_mcap(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mcap"
    fake.write_bytes(b"definitely not mcap")
    result = runner.invoke(
        app, ["workbench", "foxglove", "inspect", "--input-path", str(fake)]
    )
    assert result.exit_code == 1
    assert "MCAP inspect failed" in result.output


def test_install_sdk_surfaces_script_failure(tmp_path: Path, monkeypatch) -> None:
    # No network in unit tests: point at an unreachable registry and assert the
    # command fails loudly instead of pretending the assets are installed.
    result = runner.invoke(
        app,
        [
            "workbench",
            "foxglove",
            "install-sdk",
            "--dest",
            str(tmp_path / "sdk"),
            "--registry",
            "http://127.0.0.1:9",
        ],
    )
    assert result.exit_code == 1
    assert "install failed" in result.output.lower()
    assert not (tmp_path / "sdk").exists()

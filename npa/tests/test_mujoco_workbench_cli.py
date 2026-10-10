"""MuJoCo CLI path-contract coverage that does not require MuJoCo itself."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.workbench import mujoco as mujoco_cli
from npa.workbench.mujoco_manip.envs import MujocoManipEnv
from npa.workflows.byof import mujoco_pipeline


def test_cli_rejects_local_output_path(tmp_path) -> None:
    result = CliRunner().invoke(
        mujoco_cli.app,
        [
            "--task",
            "reach",
            "--policy",
            "scripted:expert",
            "--output-path",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code != 0
    assert "expects an S3 URI" in result.output


def test_cli_uses_canonical_s3_output_path(monkeypatch) -> None:
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"summary": {"success_rate": 1.0, "successes": 1, "episodes": 1}}

    monkeypatch.setattr(mujoco_pipeline, "run", fake_run)
    result = CliRunner().invoke(
        mujoco_cli.app,
        [
            "--task",
            "reach",
            "--policy",
            "scripted:expert",
            "--output-path",
            "s3://bucket/runs/841/report.json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["output_path"] == "s3://bucket/runs/841/report.json"


def test_pipeline_serializes_to_s3_path_without_mujoco(monkeypatch) -> None:
    writes = []

    def fake_write(path, payload, *, content_type):
        writes.append((path, payload, content_type))
        return path

    monkeypatch.setattr(mujoco_pipeline, "write_bytes", fake_write)
    assert (
        mujoco_pipeline._write_json_path(
            "s3://bucket/runs/841/report.json", {"status": "completed"}
        )
        == "s3://bucket/runs/841/report.json"
    )
    assert writes == [
        (
            "s3://bucket/runs/841/report.json",
            b'{\n  "status": "completed"\n}\n',
            "application/json",
        )
    ]


def test_render_rgb_preserves_unexpected_errors_and_records_headless_ones() -> None:
    class HeadlessRenderer:
        def __init__(self, *args) -> None:
            raise RuntimeError("EGL is unavailable")

    env = object.__new__(MujocoManipEnv)
    env._renderer = None
    env._render_error = None
    env._mujoco = SimpleNamespace(Renderer=HeadlessRenderer)
    env.model = object()
    env.data = object()

    assert env.render_rgb() is None
    assert env.render_error == "RuntimeError: EGL is unavailable"

    class BrokenRenderer:
        def __init__(self, *args) -> None:
            raise AssertionError("programming error")

    env._mujoco = SimpleNamespace(Renderer=BrokenRenderer)
    with pytest.raises(AssertionError, match="programming error"):
        env.render_rgb()

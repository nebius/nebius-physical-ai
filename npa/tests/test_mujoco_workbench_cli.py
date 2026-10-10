"""MuJoCo CLI path-contract coverage that does not require MuJoCo itself."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.workbench import mujoco_manip
from npa.cli.workbench import mujoco as mujoco_cli
from npa.workbench.mujoco_manip.envs import MujocoManipEnv
from npa.workflows.byof import artifact_io
from npa.workflows.byof import mujoco_pipeline


class _ConditionalStorage:
    """In-memory create-only store with the production retry interface."""

    def __init__(self) -> None:
        self.attempts = 0
        self.objects: dict[str, bytes] = {}

    def put_bytes_conditional(self, payload, bucket_uri, **_kwargs):
        self.attempts += 1
        if bucket_uri in self.objects:
            raise artifact_io.StoragePreconditionFailed("already exists")
        self.objects[bucket_uri] = payload
        return '"etag"'

    def read_bytes_with_etag(self, output_path):
        payload = self.objects.get(output_path)
        return None if payload is None else (payload, '"etag"')


class _FakeMujocoEnv:
    """One-step deterministic environment for the real pipeline producer."""

    def reset(self, seed=None):
        return [float(seed or 0)]

    def step(self, _action):
        return [1.0], 1.0, True, {"success": True, "tilt": 0.0, "tip_pos": [0.0]}


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


def test_pipeline_s3_retry_is_exact_across_changed_utc_seconds(monkeypatch) -> None:
    """The real producer retries only identical serialized MuJoCo reports."""
    storage = _ConditionalStorage()
    monkeypatch.setattr(
        artifact_io.StorageClient,
        "from_environment",
        staticmethod(lambda: storage),
    )
    monkeypatch.setattr(
        mujoco_manip,
        "make_env",
        lambda *_args, **_kwargs: _FakeMujocoEnv(),
    )
    monkeypatch.setattr(
        mujoco_manip,
        "get_policy",
        lambda *_args, **_kwargs: lambda _obs: [0.0],
    )
    clock = {"utc": "2026-10-10T15:00:00Z"}
    monkeypatch.setattr(time, "strftime", lambda *_args: clock["utc"])
    common = {
        "task": "reach",
        "policy": "scripted:expert",
        "output_path": "s3://evidence-bucket/runs/841/mujoco/report.json",
        "episodes": 1,
        "seed": 4,
        "max_steps": 2,
    }

    first = mujoco_pipeline.run(**common)
    saved_objects = dict(storage.objects)
    clock["utc"] = "2026-10-10T15:00:01Z"
    second = mujoco_pipeline.run(**common)

    assert "created_utc" not in first
    assert first == second
    assert storage.objects == saved_objects
    assert storage.attempts == 2
    with pytest.raises(artifact_io.StorageError, match="non-identical artifact"):
        mujoco_pipeline.run(**{**common, "seed": 5})


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

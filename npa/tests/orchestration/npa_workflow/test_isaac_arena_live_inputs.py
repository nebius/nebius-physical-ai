"""Verify Arena live input staging with synthetic HDF5 and in-memory storage."""

from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workbench.isaac_arena.runtime import build_evaluation_argv


class _Storage:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.readback_override: bytes | None = None

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[bucket, key] = Path(filename).read_bytes()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        body = self.objects[Bucket, Key]
        if self.readback_override is not None:
            body = self.readback_override
        return {"Body": BytesIO(body)}


@pytest.fixture
def live_inputs(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "e2e/npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("unit_arena_live_helpers", path)
    assert spec and spec.loader
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    monkeypatch.delenv("NPA_E2E_S3_PREFIX", raising=False)
    replay = tmp_path / "synthetic-replay.hdf5"
    with h5py.File(replay, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("actions", data=np.arange(12).reshape(4, 3))
        episode.create_group("initial_state")
    payload = replay.read_bytes()
    monkeypatch.setattr(
        helpers, "ISAAC_ARENA_REPLAY_SHA256", hashlib.sha256(payload).hexdigest()
    )
    download = Mock(side_effect=lambda destination: destination.write_bytes(payload))
    monkeypatch.setattr(helpers, "_download_isaac_arena_replay", download)
    storage = _Storage()
    factory = Mock(return_value=storage)
    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", factory
    )
    return helpers, storage, download, factory, payload


@pytest.mark.parametrize("prefix", [None, "unit-authorized/Exact_root.v1"])
def test_arena_live_worker_reads_the_hash_verified_run_scoped_seed(
    live_inputs, monkeypatch, tmp_path, prefix
):
    helpers, storage, download, _factory, payload = live_inputs
    if prefix is not None:
        monkeypatch.setenv("NPA_E2E_S3_PREFIX", prefix)
    name = "isaac-arena-evaluation-rtxpro.yaml"
    path = helpers.materialize_live_spec(
        tmp_path, name, bucket="unit-bucket", run_id="unit-run"
    )
    helpers.seed_live_workflow_inputs(
        spec_name=name, bucket="unit-bucket", run_id="unit-run"
    )
    root = prefix or "npa-workflow-e2e/unit-run"
    key = f"{root}/isaac-arena-evaluation-rtxpro/input/gr1-open-microwave.hdf5"
    assert storage.objects == {("unit-bucket", key): payload}
    plan = build_plan(load_spec(path), run_id="unit-run")
    argv = plan.steps[0].argv
    assert argv[argv.index("--input-path") + 1] == f"s3://unit-bucket/{key}"
    assert not Path(download.call_args.args[0]).exists()


def test_arena_live_seed_rejects_changed_download_before_write(
    live_inputs, monkeypatch
):
    helpers, storage, _download, _factory, _payload = live_inputs
    monkeypatch.setattr(
        helpers,
        "_download_isaac_arena_replay",
        lambda path: path.write_bytes(b"changed"),
    )
    with pytest.raises(pytest.fail.Exception, match="differs from the pinned"):
        helpers.seed_live_workflow_inputs(
            spec_name="isaac-arena-evaluation-rtxpro.yaml",
            bucket="unit-bucket",
            run_id="unit-run",
        )
    assert storage.objects == {}


def test_arena_live_seed_rejects_incomplete_hdf5_before_write(live_inputs, monkeypatch):
    helpers, storage, download, _factory, _payload = live_inputs
    buffer = BytesIO()
    with h5py.File(buffer, "w") as dataset:
        dataset.create_group("data/demo_0")
    payload = buffer.getvalue()
    download.side_effect = lambda path: path.write_bytes(payload)
    monkeypatch.setattr(
        helpers, "ISAAC_ARENA_REPLAY_SHA256", hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(
        pytest.fail.Exception, match="missing actions or its recorded initial state"
    ):
        helpers.seed_live_workflow_inputs(
            spec_name="isaac-arena-evaluation-rtxpro.yaml",
            bucket="unit-bucket",
            run_id="unit-run",
        )
    assert storage.objects == {}


def test_arena_live_seed_rejects_changed_readback(live_inputs):
    helpers, storage, _download, _factory, _payload = live_inputs
    storage.readback_override = b"changed"
    with pytest.raises(pytest.fail.Exception, match="readback differs"):
        helpers.seed_live_workflow_inputs(
            spec_name="isaac-arena-evaluation-rtxpro.yaml",
            bucket="unit-bucket",
            run_id="unit-run",
        )


def test_arena_live_seed_rejects_invalid_prefix_before_io(live_inputs, monkeypatch):
    helpers, storage, download, factory, _payload = live_inputs
    monkeypatch.setenv("NPA_E2E_S3_PREFIX", "../outside")
    with pytest.raises(ValueError, match="NPA_E2E_S3_PREFIX"):
        helpers.seed_live_workflow_inputs(
            spec_name="isaac-arena-evaluation-rtxpro.yaml",
            bucket="unit-bucket",
            run_id="unit-run",
        )
    factory.assert_not_called()
    download.assert_not_called()
    assert storage.objects == {}


@pytest.mark.parametrize("extra_object", ["", "tomato_soup_can"])
def test_replay_workflow_preserves_optional_scene_selection_through_cli(
    monkeypatch, tmp_path, extra_object
):
    path = (
        Path(__file__).resolve().parents[4]
        / "workflows/testing/isaac-arena-evaluation-rtxpro.yaml"
    )
    spec = load_spec(path)
    assert spec.config["object"] == ""
    spec.config["object"] = extra_object
    planned = build_plan(spec, run_id="unit-scene-binding").steps[0].argv
    replay = tmp_path / "recording.hdf5"
    with h5py.File(replay, "w") as dataset:
        dataset.create_group("data/demo_0")
    native = []

    def execute(request):
        native.extend(
            build_evaluation_argv(request, output_dir=tmp_path, local_input=replay)
        )
        return {"status": "test-only-argv"}

    monkeypatch.setattr("npa.cli.workbench.isaac_arena.evaluate", execute)
    result = CliRunner().invoke(app, planned[1:])
    assert result.exit_code == 0, result.output
    assert native[native.index("--device") + 1] == "cpu"
    assert "--record_viewport_video" in native
    if extra_object:
        assert native[native.index("--object") + 1] == extra_object
    else:
        assert "--object" not in native

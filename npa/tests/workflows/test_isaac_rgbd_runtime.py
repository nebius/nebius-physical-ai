"""Verify Isaac startup, synchronization, and failure boundaries without a GPU."""

from __future__ import annotations

import builtins
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from npa.workflows.isaac_rgbd import runtime
from npa.workflows.isaac_rgbd.fixture import write_fixture


def test_cli_failure_survives_native_atexit_that_forces_success():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import atexit, os; "
            "from npa.workflows.isaac_rgbd import cli; "
            "atexit.register(lambda: os._exit(0)); "
            "cli.main = lambda: (_ for _ in ()).throw(ValueError('native failure')); "
            "cli._entrypoint()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "ValueError: native failure" in result.stderr


def _modules(monkeypatch, names):
    result = {}
    for name in names:
        result[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, result[name])
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(result[parent], child, result[name])
    return result


@pytest.mark.parametrize("installed", [None, "5.1.0.0", "6.1.0.0"])
def test_wrong_or_missing_isaac_runtime_fails_closed(monkeypatch, installed):
    def version(_name):
        if installed is None:
            raise PackageNotFoundError("isaacsim")
        return installed

    monkeypatch.setattr(runtime, "version", version)
    with pytest.raises(RuntimeError, match="requires"):
        runtime._runtime_version()


def _mock_capture(monkeypatch, events):
    modules = _modules(
        monkeypatch, ["isaacsim", "omni", "omni.replicator", "omni.replicator.core"]
    )

    def start(_config):
        events.append("start")
        return SimpleNamespace(close=lambda: events.append("close"))

    modules["isaacsim"].SimulationApp = start
    original_import = builtins.__import__

    def ordered_import(name, *args, **kwargs):
        if name.startswith(("omni.", "pxr")):
            assert "start" in events
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", ordered_import)
    monkeypatch.setattr(runtime, "_runtime_version", lambda: None)
    monkeypatch.setattr(runtime, "_provenance", lambda *args: {})
    monkeypatch.setattr(
        runtime, "_render_frames", lambda *args: events.append("render") or []
    )
    monkeypatch.setattr(
        runtime, "_finalize", lambda *args: events.append("validate") or {}
    )


@pytest.mark.parametrize("failure", [False, True])
def test_app_startup_and_publication_preserve_failure(monkeypatch, tmp_path, failure):
    events = []
    _mock_capture(monkeypatch, events)
    request = write_fixture(tmp_path / "input")

    def publish(*_args):
        events.append("publish")
        if failure:
            raise OSError("publication denied")

    if failure:
        with pytest.raises(OSError, match="denied"):
            runtime.capture_local(
                tmp_path / "input",
                request,
                tmp_path / "output",
                scope="procedural-room",
                publish=publish,
            )
        assert events == ["start", "render", "validate", "publish"]
    else:
        runtime.capture_local(
            tmp_path / "input",
            request,
            tmp_path / "output",
            scope="procedural-room",
            publish=publish,
        )
        assert events == ["start", "render", "validate", "publish", "close"]


def _fake_timeline(events):
    state = {"time": 0}

    def set_time(value):
        state["time"] = value
        events.append(("time", value))

    return SimpleNamespace(
        pause=lambda: None,
        set_end_time=lambda value: None,
        set_current_time=set_time,
        commit=lambda: None,
        get_current_time=lambda: state["time"],
    )


def _fake_render_loop(monkeypatch, events):
    modules = _modules(monkeypatch, ["omni", "omni.timeline"])
    timeline = _fake_timeline(events)
    modules["omni.timeline"].get_timeline_interface = lambda: timeline
    monkeypatch.setattr(runtime, "_open_scene", lambda *args: None)
    monkeypatch.setattr(
        runtime, "_create_camera", lambda stage, camera, rep: {"camera": camera}
    )
    monkeypatch.setattr(
        runtime,
        "_set_poses",
        lambda sensors, sample: events.append(("pose", sample["timestamp_ns"])),
    )
    monkeypatch.setattr(
        runtime,
        "_snapshot",
        lambda sensor: events.append(("view", sensor["camera"]["id"])) or {},
    )
    monkeypatch.setattr(
        runtime, "_write_frame", lambda *args: events.append(("write", args[2]))
    )
    rep = SimpleNamespace(
        orchestrator=SimpleNamespace(
            set_capture_on_play=lambda value: None,
            step=lambda **kwargs: events.append(("render", kwargs)),
        )
    )
    app = SimpleNamespace(update=lambda: events.append(("update",)))
    return app, rep


def test_one_render_step_precedes_all_views_before_next_sample(monkeypatch, tmp_path):
    request = write_fixture(tmp_path / "input")
    events = []
    app, rep = _fake_render_loop(monkeypatch, events)
    runtime._render_frames(app, tmp_path, request, tmp_path, rep)
    for index in range(3):
        batch = events[index * 9 : (index + 1) * 9]
        assert [item[0] for item in batch] == [
            "time",
            "pose",
            "update",
            "render",
            "view",
            "view",
            "view",
            "view",
            "write",
        ]
        assert batch[3][1] == {
            "delta_time": 0,
            "pause_timeline": True,
            "rt_subframes": 4,
            "wait_for_render": True,
        }
        assert [item[1] for item in batch[4:8]] == ["front", "left", "rear", "right"]


def test_snapshot_requires_real_sensor_metadata(monkeypatch):
    sensor = {
        "annotators": {
            "CameraParams": SimpleNamespace(get_data=lambda: {}),
            "ReferenceTime": SimpleNamespace(get_data=lambda: {}),
        }
    }
    with pytest.raises(ValueError, match="annotations"):
        runtime._snapshot(sensor)


@pytest.mark.parametrize(
    "workflow", ["multicamera-rgbd-capture", "multicamera-rgbd-warehouse"]
)
def test_workflow_uses_isaac_image_route_and_cpu_validator(workflow):
    from npa.orchestration.npa_workflow.skypilot_render import tool_image_key
    from npa.orchestration.npa_workflow.spec import load_spec

    path = Path(__file__).resolve().parents[3] / f"workflows/testing/{workflow}.yaml"
    spec = load_spec(path)
    capture, validate = spec.states["capture"], spec.states["validate"]
    assert tool_image_key(capture.tool_ref) == "isaac-lab"
    assert not tool_image_key(validate.tool_ref)
    assert capture.next == "validate" and validate.terminal
    if "prepare" in spec.states:
        assert tool_image_key(spec.states["prepare"].tool_ref) == "isaac-lab"
        assert spec.states["prepare"].next == "capture"


@pytest.mark.parametrize(
    "workflow", ["multicamera-rgbd-capture", "multicamera-rgbd-warehouse"]
)
def test_render_stages_branch_source_in_gpu_and_cpu_tasks(monkeypatch, workflow):
    import yaml
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
    from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit

    source = "s3://example-bucket/staged-source/npa"
    monkeypatch.setenv("NPA_SRC_S3_URI", source)
    path = Path(__file__).resolve().parents[3] / f"workflows/testing/{workflow}.yaml"
    prepared = prepare_npa_workflow_for_submit(
        path, run_id="source-proof", render_options=SkypilotRenderOptions()
    )
    try:
        docs = [
            doc
            for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if doc
        ]
        capture, validation = docs[-2:]
        if workflow.endswith("warehouse"):
            prepare = docs[1]
            assert "prepare-reference" in prepare["run"]
            assert prepare["envs"]["NPA_SRC_OVERLAY"] == "1"
            assert "/isaac-sim/python.sh" in prepare["setup"]
        assert capture["envs"]["NPA_SRC_OVERLAY"] == "1"
        assert capture["envs"]["NPA_SRC_S3_URI"] == source
        assert "/isaac-sim/python.sh" in capture["setup"]
        assert 'if [ "$NPA_SRC_OVERLAY" = "1" ]' in capture["setup"]
        assert validation["envs"]["NPA_SRC_S3_URI"] == source
        assert "image_id" not in validation["resources"]
        assert "npa-src" in validation["setup"]
    finally:
        prepared.temp_dir.cleanup()

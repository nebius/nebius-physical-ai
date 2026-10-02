"""Preserve native LiDAR capture initialization without narrowing explicit selections."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.nurec import ncore_initialization as initialization, nurec


@pytest.fixture
def native_capture(tmp_path, monkeypatch):
    meta = tmp_path / "sequence.json"
    meta.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [
                    {
                        "path": "native.zarr.itar",
                        "components": {
                            "cameras": {"camera1": {}, "camera2": {}},
                            "lidars": {"virtual_lidar": {}},
                        },
                    }
                ],
            }
        )
    )
    (tmp_path / "npa-rig.json").write_text('{"reference_camera":"camera2"}')
    monkeypatch.setattr(initialization, "_point_readers", lambda _: {})
    return meta


def plan(meta, **kwargs):
    return initialization.plan_initialization(nurec.NurecConfig(**kwargs), str(meta))


def test_native_lidar_defaults_to_declared_rig_camera_with_notice(native_capture):
    before = {p.name: p.read_bytes() for p in native_capture.parent.iterdir()}
    with pytest.warns(UserWarning, match="only rig reference camera 'camera2'"):
        config, evidence = plan(native_capture)
    assert config.camera_ids == ("camera2",)
    assert config.lidar_ids == ("virtual_lidar",)
    assert config.extra_overrides == () and evidence == {}
    assert before == {p.name: p.read_bytes() for p in native_capture.parent.iterdir()}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_ids": ("camera1", "camera2")},
        {"extra_overrides": ("dataset.camera_ids=[camera1,camera2]",)},
    ],
)
def test_explicit_multiple_cameras_are_never_silently_restricted(
    native_capture, kwargs
):
    with pytest.raises(nurec.NurecError, match="PointCloudsComponent.*--camera-id"):
        plan(native_capture, **kwargs)


def test_explicit_single_camera_keeps_native_initializer(native_capture):
    config, evidence = plan(native_capture, camera_ids=("camera1",))
    assert config.camera_ids == ("camera1",) and evidence == {}
    assert config.extra_overrides == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lidar_ids": ("none",)},
        {"extra_overrides": ("dataset.lidar_ids=[]",)},
    ],
)
def test_disabled_native_lidar_does_not_trigger_camera_fallback(native_capture, kwargs):
    with pytest.raises(nurec.NurecError, match="nonempty NCore point inventory"):
        plan(native_capture, **kwargs)


def test_missing_reference_camera_fails_before_native_execution(native_capture):
    native_capture.with_name("npa-rig.json").write_text(
        '{"reference_camera":"missing"}'
    )
    with pytest.raises(nurec.NurecError, match="reference camera is absent"):
        plan(native_capture)


def test_point_cloud_capture_keeps_all_cameras(native_capture, monkeypatch):
    class Points:
        pcs_count = 1

        def get_pc_xyz(self, index):
            return [(1.0, 2.0, 3.0)]

    monkeypatch.setattr(initialization, "_point_readers", lambda _: {"sfm": Points()})
    config, evidence = plan(native_capture)
    assert config.camera_ids == ("camera1", "camera2")
    assert evidence["point_count"] == 1
    assert evidence["initializer"] == "accumulated-point-cloud"


def test_custom_initialization_preserves_full_camera_selection(native_capture):
    override = "model.layers.background.initialization.num_point_cloud_points=123"
    config, evidence = plan(native_capture, extra_overrides=(override,))
    assert config.camera_ids == ("camera1", "camera2")
    assert config.extra_overrides == (override,) and evidence == {}


def test_cli_json_stays_valid_and_native_argv_uses_reference_camera(native_capture):
    with pytest.warns(UserWarning, match="other capture cameras"):
        result = CliRunner().invoke(
            app,
            [
                "workbench",
                "nurec",
                "reconstruct",
                "--ncore-json",
                str(native_capture),
                "--out-dir",
                str(native_capture.parent / "output"),
                "--dry-run",
                "--output",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert "dataset.camera_ids=['camera2']" in report["command"]
    assert "dataset.lidar_ids=['virtual_lidar']" in report["command"]
    assert not any("accumulated_point_cloud" in arg for arg in report["command"])
    assert not Path(native_capture.parent / "output").exists()

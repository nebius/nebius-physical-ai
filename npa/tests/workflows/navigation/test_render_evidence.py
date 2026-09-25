"""Reject stale native camera/time evidence and preserve the rendering context."""

from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import render_evidence as evidence


class Settings(dict):
    def set(self, name, value):
        self[name] = value

    def destroy_item(self, name):
        del self[name]


def test_render_settings_restore_on_setup_failure():
    settings = Settings(enabled=False, capture=True)
    with pytest.raises(RuntimeError, match="setup failed"):
        with evidence._settings(
            settings, {"enabled": True, "capture": False, "new": 7}
        ):
            assert settings == {"enabled": True, "capture": False, "new": 7}
            raise RuntimeError("setup failed")
    assert settings == {"enabled": False, "capture": True}


def camera_values():
    return {
        "view_transform": np.eye(4),
        "aperture": np.asarray([20.955, 15.71625]),
        "offset": np.zeros(2),
        "focal": np.asarray(18.0),
        "resolution": np.asarray([640, 480]),
    }


@pytest.mark.parametrize("name", list(camera_values()))
def test_camera_rejects_stale_pose_and_intrinsics(name):
    expected = camera_values()
    actual = {key: value.copy() for key, value in expected.items()}
    actual[name] = actual[name] + 0.25
    with pytest.raises(RuntimeError, match=f"camera {name}"):
        evidence._check_camera(actual, expected)


def test_camera_rejects_nonfinite_and_wrong_shape():
    expected = camera_values()
    actual = camera_values()
    actual["view_transform"][0, 0] = np.nan
    with pytest.raises(RuntimeError, match="view_transform"):
        evidence._check_camera(actual, expected)
    actual["view_transform"] = np.ones((3, 4))
    with pytest.raises(RuntimeError, match="view_transform"):
        evidence._check_camera(actual, expected)


def test_native_clock_retains_post_probe_origin():
    reference = {"referenceTimeNumerator": 241, "referenceTimeDenominator": 10}
    assert evidence._reference_time(reference, 24.1) == {
        "numerator": 241,
        "denominator": 10,
    }
    with pytest.raises(RuntimeError, match="stale rendered reference time"):
        evidence._reference_time(reference, 0.0)
    with pytest.raises(RuntimeError, match="stale rendered reference time"):
        evidence._reference_time(reference, 24.3)


@pytest.mark.parametrize("numerator,denominator", [(1.5, 1), (1, 0), (1, -1), (1, 1.2)])
def test_reference_time_requires_valid_integer_ratio(numerator, denominator):
    with pytest.raises(RuntimeError, match="integer rational"):
        evidence._reference_time(
            {
                "referenceTimeNumerator": numerator,
                "referenceTimeDenominator": denominator,
            },
            1,
        )


@pytest.mark.parametrize(
    "name",
    ["root_transforms", "root_velocities", "joint_positions", "joint_velocities"],
)
def test_uncached_native_comparison_detects_changed_state(name):
    torch = pytest.importorskip("torch")
    before = {"clocks": {"physics_seconds": 24.0}, "state": {name: torch.zeros(4, 7)}}
    after = {
        "clocks": before["clocks"].copy(),
        "state": {name: before["state"][name].clone()},
    }
    evidence._check_frozen(before, after)
    after["state"][name][3, 6] = 0.1
    with pytest.raises(RuntimeError, match=f"uncached native PhysX {name}"):
        evidence._check_frozen(before, after)


def test_native_comparison_rejects_clock_advance():
    with pytest.raises(RuntimeError, match="advanced native clocks"):
        evidence._check_frozen(
            {"clocks": {"physics_step_count": 40}},
            {"clocks": {"physics_step_count": 41}},
        )


def test_expected_camera_uses_real_gf_row_vector_transform():
    pxr = pytest.importorskip("pxr.UsdGeom")
    from pxr import Gf, Usd

    stage = Usd.Stage.CreateInMemory()
    camera = pxr.Camera.Define(stage, "/Camera")
    camera.CreateFocalLengthAttr(18)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.71625)
    transform = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(3, 4, 5), Gf.Vec3d(1, 2, 0), Gf.Vec3d(0, 0, 1)
    )
    expected = evidence._expected_camera(camera, transform.GetInverse())
    assert np.allclose(expected["view_transform"], np.asarray(transform))
    evidence._check_camera(expected, expected)


def test_frozen_context_restores_physics_on_render_exception(monkeypatch):
    import sys

    settings = Settings({"/app/player/playSimulations": True})
    module = SimpleNamespace(get_settings=lambda: settings)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=module))
    monkeypatch.setitem(sys.modules, "carb.settings", module)
    monkeypatch.setattr(
        evidence, "_native_snapshot", lambda env: {"state": {}, "clocks": {}}
    )
    with pytest.raises(RuntimeError, match="renderer failed"):
        with evidence.frozen_physics(object()):
            assert settings["/app/player/playSimulations"] is False
            raise RuntimeError("renderer failed")
    assert settings["/app/player/playSimulations"] is True


def test_capture_enables_actual_pinned_scheduler_and_restores(monkeypatch):
    import sys

    enabled = "/exts/omni.replicator.core/Orchestrator/enabled"
    capture = "/omni/replicator/captureOnPlay"
    settings = Settings({enabled: False, capture: True})
    module = SimpleNamespace(get_settings=lambda: settings)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=module))
    monkeypatch.setitem(sys.modules, "carb.settings", module)
    with evidence.capture_settings():
        assert settings == {
            enabled: True,
            capture: False,
            "/rtx/hydra/supportMultiTickRate": True,
            "/rtx/rendering/perSensorTickTlas": True,
        }
    assert settings == {enabled: False, capture: True}


def test_native_snapshot_copies_actual_physx_buffers(monkeypatch):
    import sys

    torch = pytest.importorskip("torch")
    roots = torch.zeros(4, 7)
    view = SimpleNamespace(
        get_root_transforms=lambda: roots,
        get_root_velocities=lambda: torch.zeros(4, 6),
        get_dof_positions=lambda: torch.zeros(4, 12),
        get_dof_velocities=lambda: torch.zeros(4, 12),
    )
    monkeypatch.setattr(
        evidence, "_native_clocks", lambda env: {"physics_seconds": 24.0}
    )
    monkeypatch.setitem(
        sys.modules, "warp", SimpleNamespace(to_torch=lambda value: value)
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(root_view=view)},
        sim=SimpleNamespace(get_physics_step_count=lambda: 4800),
    )
    before = evidence._native_snapshot(env)
    roots[3, 0] = 1.0
    after = evidence._native_snapshot(env)
    assert before["state"]["root_transforms"][3, 0] == 0
    with pytest.raises(RuntimeError, match="root_transforms"):
        evidence._check_frozen(before, after)


@pytest.fixture
def native_clock_api(monkeypatch):
    import sys

    values = {"seconds": 24.0, "steps": 4800, "fabric": 24.0}
    manager = SimpleNamespace(
        get_simulation_time=lambda: values["seconds"],
        get_num_physics_steps=lambda: values["steps"],
    )
    extension = SimpleNamespace(acquire_simulation_manager_interface=lambda: manager)
    attribute = SimpleNamespace(Get=lambda: values["fabric"])
    prim = SimpleNamespace(
        GetAttribute=lambda name: attribute if name == "omni:time" else None
    )
    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prim if path == "/ExternalSimulationTime" else None
    )
    stage_utils = SimpleNamespace(
        get_current_stage=lambda *, backend: stage if backend == "fabric" else None
    )
    timeline = SimpleNamespace(
        get_timeline_interface=lambda: SimpleNamespace(get_current_time=lambda: 0.0)
    )
    modules = {
        "omni": SimpleNamespace(timeline=timeline),
        "omni.timeline": timeline,
        "isaacsim.core.experimental.utils": SimpleNamespace(stage=stage_utils),
        "isaacsim.core.simulation_manager.impl.extension": extension,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    env = SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda: 4800))
    return env, values, extension, stage


def test_clock_uses_unpatched_existing_native_binding(native_clock_api):
    env, _, _, _ = native_clock_api
    assert evidence._native_clocks(env) == {
        "physics_seconds": 24.0,
        "simulation_manager_step_count": 4800,
        "fabric_seconds": 24.0,
        "timeline_seconds": 0.0,
        "physics_step_count": 4800,
    }


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("seconds", None, "numeric"),
        ("seconds", float("nan"), "finite"),
        ("seconds", 0, "did not advance"),
        ("steps", 0, "did not advance"),
        ("fabric", 23.8, "disagree"),
    ],
)
def test_clock_rejects_unobserved_or_inconsistent_events(
    native_clock_api, name, value, error
):
    env, values, _, _ = native_clock_api
    values[name] = value
    with pytest.raises(RuntimeError, match=error):
        evidence._native_clocks(env)


@pytest.mark.parametrize("missing", ["manager", "fabric"])
def test_clock_never_creates_missing_native_state(native_clock_api, missing):
    env, _, extension, stage = native_clock_api
    if missing == "manager":
        extension.acquire_simulation_manager_interface = lambda: None
    else:
        stage.GetPrimAtPath = lambda path: None
    with pytest.raises(RuntimeError, match="unavailable"):
        evidence._native_clocks(env)


def test_capture_preserves_clocks_even_when_metadata_rejects(monkeypatch):
    import sys

    settings = Settings({"/app/player/playSimulations": True})
    module = SimpleNamespace(get_settings=lambda: settings)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=module))
    monkeypatch.setitem(sys.modules, "carb.settings", module)
    snapshots = iter(
        [{"clocks": {"physics_seconds": 24}}, {"clocks": {"physics_seconds": 25}}]
    )
    monkeypatch.setattr(evidence, "_native_snapshot", lambda env: next(snapshots))
    with pytest.raises(RuntimeError, match="advanced native clocks") as failure:
        with evidence.frozen_physics(object()):
            raise RuntimeError("metadata rejected")
    assert str(failure.value.__context__) == "metadata rejected"
    assert settings["/app/player/playSimulations"] is True


def test_failed_frame_preserves_native_clock_and_render_mode():
    annotator = SimpleNamespace(
        get_data=lambda: {"referenceTimeNumerator": 14, "referenceTimeDenominator": 15}
    )
    clocks = {
        "physics_seconds": 24.01,
        "fabric_seconds": 24.01,
        "physics_step_count": 4802,
    }
    with pytest.raises(RuntimeError, match="stale rendered reference time") as failure:
        evidence._frame_time(
            {"ReferenceTime": annotator}, {"clocks": clocks}, {"multitick": True}
        )
    assert "fabric_seconds" in str(failure.value)
    assert "4802" in str(failure.value)
    assert "multitick" in str(failure.value)


def test_render_mode_rejects_legacy_frame_clock(monkeypatch):
    import sys

    settings = Settings(
        {
            "/rtx/hydra/supportMultiTickRate": True,
            "/rtx/rendering/perSensorTickTlas": True,
            "/exts/omni.replicator.core/Orchestrator/enabled": True,
            "/app/player/playSimulations": False,
        }
    )
    module = SimpleNamespace(get_settings=lambda: settings)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=module))
    monkeypatch.setitem(sys.modules, "carb.settings", module)
    assert evidence._render_settings() == settings
    settings["/rtx/hydra/supportMultiTickRate"] = False
    with pytest.raises(RuntimeError, match="native capture settings changed"):
        evidence._render_settings()

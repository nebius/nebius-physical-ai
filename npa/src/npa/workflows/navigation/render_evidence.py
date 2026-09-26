"""Qualify native rollout frames against renderer metadata and uncached PhysX state."""

from contextlib import contextmanager
from fractions import Fraction
import json

import numpy as np


@contextmanager
def capture_settings():
    """Enable the pinned Lab preset's disabled scheduler only for capture.

    Args:
        None.
    Returns:
        Context restoring the prior Replicator settings, including on setup failure.
    Raises:
        RuntimeError: Kit settings cannot be applied.
    """
    import carb.settings

    settings = carb.settings.get_settings()
    prefix = "/exts/omni.replicator.core/"
    changes = {
        prefix + "Orchestrator/enabled": True,
        "/omni/replicator/captureOnPlay": False,
        "/rtx/hydra/supportMultiTickRate": True,
        "/rtx/rendering/perSensorTickTlas": True,
    }
    original = {key: settings.get(key) for key in changes}
    print(
        json.dumps(
            {"capture_settings_before": original, "capture_settings_requested": changes}
        ),
        flush=True,
    )
    with _settings(settings, changes):
        yield


@contextmanager
def _settings(settings, changes):
    original = {key: settings.get(key) for key in changes}
    try:
        for key, value in changes.items():
            settings.set(key, value)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                settings.destroy_item(key)
            else:
                settings.set(key, value)


def _native_snapshot(env):
    import warp as wp

    view = env.scene["robot"].root_view
    arrays = {
        "root_transforms": view.get_root_transforms(),
        "root_velocities": view.get_root_velocities(),
        "joint_positions": view.get_dof_positions(),
        "joint_velocities": view.get_dof_velocities(),
    }
    state = {
        name: wp.to_torch(value).detach().clone() for name, value in arrays.items()
    }
    return {"state": state, "clocks": _native_clocks(env)}


def _native_clocks(env):
    import omni.timeline
    from isaacsim.core.experimental.utils import stage as stage_utils
    from isaacsim.core.simulation_manager.impl.extension import (
        acquire_simulation_manager_interface,
    )

    # Lab replaces the public manager class; this accessor retains Kit's binding.
    manager = acquire_simulation_manager_interface()
    stage = stage_utils.get_current_stage(backend="fabric")
    prim = stage.GetPrimAtPath("/ExternalSimulationTime")
    clock = prim.GetAttribute("omni:time") if prim else None
    if manager is None or not clock:
        raise RuntimeError("native simulation-manager or Fabric clock is unavailable")
    clocks = {
        "physics_seconds": manager.get_simulation_time(),
        "simulation_manager_step_count": manager.get_num_physics_steps(),
        "fabric_seconds": clock.Get(),
        "timeline_seconds": omni.timeline.get_timeline_interface().get_current_time(),
        "physics_step_count": env.sim.get_physics_step_count(),
    }
    if not all(isinstance(value, (int, float)) for value in clocks.values()):
        raise RuntimeError(f"native render clocks must be numeric: {clocks}")
    if not all(np.isfinite(list(clocks.values()))):
        raise RuntimeError(f"native render clocks must be finite: {clocks}")
    if clocks["physics_seconds"] <= 0 or clocks["simulation_manager_step_count"] <= 0:
        raise RuntimeError(
            f"native physics events did not advance before capture: {clocks}"
        )
    if abs(clocks["physics_seconds"] - clocks["fabric_seconds"]) > 1e-9:
        raise RuntimeError(f"native physics and Fabric clocks disagree: {clocks}")
    return clocks


def _check_frozen(before, after):
    if before["clocks"] != after["clocks"]:
        raise RuntimeError(
            f"capture advanced native clocks: {before['clocks']} -> {after['clocks']}"
        )
    for name, value in before["state"].items():
        changed = after["state"][name]
        if (
            not value.isfinite().all()
            or not changed.isfinite().all()
            or not value.equal(changed)
        ):
            raise RuntimeError(f"capture changed uncached native PhysX {name}")


@contextmanager
def frozen_physics(env):
    """Keep Kit updates from advancing physics and verify the native buffers.

    Args:
        env: Actual unwrapped Isaac Lab environment.
    Returns:
        Native state and clocks before rendering; equality is checked on exit.
    Raises:
        RuntimeError: Rendering changes any native root/joint state or clock.
    """
    import carb.settings

    before = _native_snapshot(env)
    with _settings(
        carb.settings.get_settings(), {"/app/player/playSimulations": False}
    ):
        try:
            yield before
        finally:
            _check_frozen(before, _native_snapshot(env))


def _camera_values(parameters):
    return {
        "view_transform": np.asarray(parameters["cameraViewTransform"]).reshape(4, 4),
        "aperture": np.asarray(parameters["cameraAperture"]),
        "offset": np.asarray(parameters["cameraApertureOffset"]),
        "focal": np.asarray(parameters["cameraFocalLength"]),
        "resolution": np.asarray(parameters["renderProductResolution"]),
    }


def _expected_camera(camera, transform):
    return {
        "view_transform": np.linalg.inv(np.asarray(transform)),
        "aperture": np.asarray(
            [
                camera.GetHorizontalApertureAttr().Get(),
                camera.GetVerticalApertureAttr().Get(),
            ]
        ),
        "offset": np.asarray(
            [
                camera.GetHorizontalApertureOffsetAttr().Get(),
                camera.GetVerticalApertureOffsetAttr().Get(),
            ]
        ),
        "focal": np.asarray(camera.GetFocalLengthAttr().Get()),
        "resolution": np.asarray([640, 480]),
    }


def _check_camera(observed, expected):
    for name, value in expected.items():
        actual = observed[name]
        if (
            actual.shape != value.shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, value, atol=1e-5, rtol=1e-5)
        ):
            raise RuntimeError(
                f"stale or invalid rendered camera {name}: "
                f"observed={actual.tolist()}, expected={value.tolist()}"
            )


def _reference_time(reference, seconds):
    numerator = reference["referenceTimeNumerator"]
    denominator = reference["referenceTimeDenominator"]
    if (
        not isinstance(numerator, (int, np.integer))
        or not isinstance(denominator, (int, np.integer))
        or denominator <= 0
    ):
        raise RuntimeError("renderer reference time must be an integer rational")
    observed = Fraction(int(numerator), int(denominator))
    if abs(observed - Fraction(float(seconds))) > Fraction(1, 1_000_000_000):
        raise RuntimeError(
            f"stale rendered reference time {observed}; native physics={seconds}"
        )
    return {"numerator": int(numerator), "denominator": int(denominator)}


def renderer_evidence(annotators, camera, transform, native):
    """Bind actual renderer calibration and rational time to the frozen native state.

    Args:
        annotators: Native CameraParams and ReferenceTime annotators.
        camera: Authored USD perspective camera.
        transform: Authored Gf camera-to-world row-vector transform.
        native: Uncached PhysX snapshot and native clocks before capture.
    Returns:
        JSON-compatible observed and expected calibration and clock evidence.
    Raises:
        RuntimeError: Rendered calibration or time differs from native state.
    """
    observed = _camera_values(annotators["CameraParams"].get_data())
    expected = _expected_camera(camera, transform)
    _check_camera(observed, expected)
    settings = _render_settings()
    reference = _frame_time(annotators, native, settings)
    return {
        "native_clocks": native["clocks"],
        "native_clock_source": "isaacsim.core.simulation_manager.native_step_events",
        "reference_time": reference,
        "observed_camera": {key: value.tolist() for key, value in observed.items()},
        "expected_camera": {key: value.tolist() for key, value in expected.items()},
        "native_physics_unchanged": True,
        "frozen_render_passes": 2,
        "render_settings": settings,
    }


def _render_settings():
    import carb.settings

    # Lab's custom app omits these two sensor settings from Isaac's base app.
    expected = {
        "/rtx/hydra/supportMultiTickRate": True,
        "/rtx/rendering/perSensorTickTlas": True,
        "/exts/omni.replicator.core/Orchestrator/enabled": True,
        "/app/player/playSimulations": False,
    }
    settings = carb.settings.get_settings()
    observed = {key: settings.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"native capture settings changed: {observed}")
    return observed


def _frame_time(annotators, native, settings):
    try:
        return _reference_time(
            annotators["ReferenceTime"].get_data(), native["clocks"]["physics_seconds"]
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}; native clocks={native['clocks']}; render settings={settings}"
        ) from error

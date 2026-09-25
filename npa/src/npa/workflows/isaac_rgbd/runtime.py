"""Render calibrated static USD rigs with Isaac Sim 6.0.1 Replicator after Kit startup."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from npa.workflows.isaac_capture import _simulation_app_lifecycle

from .contract import (
    ISAAC_SIM_VERSION,
    _contained,
    _json_bytes,
    _sha256,
    validate_request,
)
from .dataset import _finalize, _write_frame
from .geometry import _optical_to_usd, _usd_camera_parameters
from .scene import _check_scene_files


def _runtime_version():
    try:
        installed = version("isaacsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "RGB-D capture requires /isaac-sim/python.sh in the pinned Isaac runtime"
        ) from exc
    if installed != ISAAC_SIM_VERSION:
        raise RuntimeError(
            f"RGB-D capture requires Isaac Sim {ISAAC_SIM_VERSION}; found {installed}"
        )


def _open_scene(root, request):
    import omni.usd
    from pxr import UsdGeom

    _check_scene_files(root, request)
    context = omni.usd.get_context()
    if not context.open_stage(str(_contained(root, request["scene"]))):
        raise ValueError("Isaac could not open the supplied USD scene")
    stage = context.get_stage()
    if stage.GetCompositionErrors():
        raise ValueError(
            "USD scene has composition errors; refusing partial scene capture"
        )
    if (
        UsdGeom.GetStageUpAxis(stage) != "Z"
        or UsdGeom.GetStageMetersPerUnit(stage) != 1
    ):
        raise ValueError(
            "scene must declare Z up and metersPerUnit=1; convert it before capture"
        )
    if stage.GetPrimAtPath("/NpaRgbdCapture"):
        raise ValueError("scene uses the reserved /NpaRgbdCapture prim")
    # Package root layers are read-only; rig opinions belong to this capture session.
    stage.SetEditTarget(stage.GetSessionLayer())
    return stage


def _create_camera(stage, camera, rep):
    from pxr import Gf, UsdGeom

    path = f"/NpaRgbdCapture/{camera['id']}"
    prim = UsdGeom.Camera.Define(stage, path)
    settings = _usd_camera_parameters(camera)
    prim.CreateProjectionAttr("perspective")
    prim.CreateFocalLengthAttr(settings["focal"])
    prim.CreateHorizontalApertureAttr(settings["aperture"][0])
    prim.CreateVerticalApertureAttr(settings["aperture"][1])
    prim.CreateHorizontalApertureOffsetAttr(settings["offset"][0])
    prim.CreateVerticalApertureOffsetAttr(settings["offset"][1])
    prim.CreateClippingRangeAttr(Gf.Vec2f(*camera["depth_range_m"]))
    prim.CreateFStopAttr(0.0)
    transform_op = UsdGeom.Xformable(prim).AddTransformOp()
    product = rep.create.render_product(path, (camera["width"], camera["height"]))
    annotators = {}
    for name in ("rgb", "distance_to_image_plane", "CameraParams", "ReferenceTime"):
        annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
        annotator.attach(product)
        annotators[name] = annotator
    return {
        "camera": camera,
        "transform": transform_op,
        "product": product,
        "annotators": annotators,
    }


def _set_poses(sensors, sample):
    from pxr import Gf

    for sensor in sensors:
        world = np.asarray(sample["T_world_rig"]) @ np.asarray(
            sensor["camera"]["T_rig_camera"]
        )
        # USD/Gf stores row-vector transforms, unlike the dataset's column vectors.
        sensor["transform"].Set(Gf.Matrix4d(_optical_to_usd(world).T.tolist()))


def _snapshot(sensor):
    annotators = sensor["annotators"]
    parameters = annotators["CameraParams"].get_data()
    reference = annotators["ReferenceTime"].get_data()
    if not parameters or not reference:
        raise ValueError(
            "renderer did not return calibration/reference-time annotations"
        )
    return {
        "rgb": np.asarray(annotators["rgb"].get_data()).copy(),
        "depth": np.asarray(annotators["distance_to_image_plane"].get_data()).copy(),
        "render_reference_time": {
            "numerator": int(reference["referenceTimeNumerator"]),
            "denominator": int(reference["referenceTimeDenominator"]),
        },
        "render_calibration": {
            "aperture": np.asarray(parameters["cameraAperture"]).tolist(),
            "offset": np.asarray(parameters["cameraApertureOffset"]).tolist(),
            "focal": float(parameters["cameraFocalLength"]),
            "view_transform": np.asarray(parameters["cameraViewTransform"])
            .reshape(4, 4)
            .tolist(),
            "resolution": np.asarray(parameters["renderProductResolution"]).tolist(),
        },
    }


def _render_frames(app, root, request, output, rep):
    import omni.timeline

    stage = _open_scene(root, request)
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.set_end_time(request["trajectory"][-1]["timestamp_ns"] / 1e9 + 1)
    rep.orchestrator.set_capture_on_play(False)
    cameras = sorted(request["cameras"], key=lambda camera: camera["id"])
    sensors = [_create_camera(stage, camera, rep) for camera in cameras]
    frames = []
    for index, sample in enumerate(request["trajectory"]):
        timeline.set_current_time(sample["timestamp_ns"] / 1e9)
        timeline.commit()
        _set_poses(sensors, sample)
        app.update()
        rep.orchestrator.step(
            delta_time=0.0, pause_timeline=True, rt_subframes=4, wait_for_render=True
        )
        snapshots = {sensor["camera"]["id"]: _snapshot(sensor) for sensor in sensors}
        frames.append(
            _write_frame(output, request, index, snapshots, timeline.get_current_time())
        )
    return frames


def _provenance(input_root, request, scope):
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = manager.get_enabled_extension_id("omni.replicator.core")
    return {
        "request_sha256": hashlib.sha256(_json_bytes(request)).hexdigest(),
        "input_manifest_sha256": _sha256(input_root / "request.json"),
        "scope": scope,
        "replicator_version": manager.get_extension_dict(extension_id)["package"][
            "version"
        ],
    }


def capture_local(input_root, request, output_root, *, scope, publish):
    """Render every requested view and publish only after decoded-data validation.

    Args:
        input_root: Private, hash-verified input bundle directory.
        request: Calibrated capture request.
        output_root: Empty local directory for captured data.
        scope: supplied-usd or procedural-room validation scope.
        publish: Callback receiving output directory and completed manifest before Kit closes.

    Returns:
        Completed dataset manifest if Kit shutdown returns.

    Raises:
        RuntimeError: The supported Isaac interpreter is unavailable.
        ValueError: Inputs, sensor data, or decoded output validation fail.
        Exception: Renderer or publication failure; Kit close cannot mask the failure.
    """
    validate_request(request)
    _runtime_version()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
    with _simulation_app_lifecycle(app):
        import omni.replicator.core as rep

        provenance = _provenance(input_root, request, scope)
        frames = _render_frames(app, input_root, request, output_root, rep)
        manifest = _finalize(output_root, request, frames, provenance)
        publish(output_root, manifest)
    return manifest

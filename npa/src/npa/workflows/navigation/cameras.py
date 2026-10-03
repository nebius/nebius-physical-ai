"""Measure fresh native RGB-D peer invariance and reject frozen render buffers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np

from npa.workflows.navigation.measure import verify_reset
from npa.workflows.navigation.visibility import hide_robot_geometry


def validate_frame(rgb, depth, config) -> dict:
    """Require explicit finite, nonempty RGB and metric depth buffers.

    Args:
        rgb: Native uint8 RGB/RGBA array.
        depth: Native floating depth image in metres.
        config: Validated camera dimensions and tolerances.
    Returns:
        Copies of normalized RGB and finite positive depth.
    Raises:
        ValueError: Dtype, dimensions, finite range or content is invalid.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.dtype != np.uint8 or rgb.shape not in (
        (config.height, config.width, 3),
        (config.height, config.width, 4),
    ):
        raise ValueError("RGB must be uint8 with the declared H,W,3 or H,W,4 shape")
    if depth.shape == (config.height, config.width, 1):
        depth = depth[..., 0]
    if depth.shape != (config.height, config.width) or depth.dtype.kind != "f":
        raise ValueError("depth must be floating point with declared H,W shape")
    if not np.isfinite(depth).all() or (depth <= 0).any() or not rgb[..., :3].any():
        raise ValueError("RGB/depth buffers are empty or contain invalid depth")
    return {
        "rgb": rgb[..., :3].astype(float) / 255,
        "depth": depth.astype(float).copy(),
    }


def compare_frames(baseline: dict, hidden: dict, moved: dict, config) -> dict:
    """Evaluate peer invariance and camera-motion controls for both modalities.

    Args:
        baseline: Validated focal frames with peers parked.
        hidden: Validated focal frames after peers enter the field of view.
        moved: Validated focal frames after moving the observing camera.
        config: Camera tolerances and minimum changing-pixel fraction.
    Returns:
        Measured deltas, changed fractions and hashes for retained frames.
    Raises:
        ValueError: Peers appear, buffers freeze or positive controls fail.
    """
    evidence = {}
    for key, tolerance in (
        ("rgb", config.rgb_tolerance),
        ("depth", config.depth_tolerance_m),
    ):
        arrays = [frame[key] for frame in (baseline, hidden, moved)]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("camera comparisons require finite frames")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("camera frame shapes changed")
        delta = float(np.max(np.abs(arrays[0] - arrays[1])))
        changed = np.abs(arrays[1] - arrays[2]) > tolerance
        if key == "rgb":
            changed = changed.any(axis=-1)
        fraction = float(changed.mean())
        if delta > tolerance or fraction < config.minimum_changed_fraction:
            raise ValueError(
                f"{key}: peer visibility or frozen/ineffective camera positive control"
            )
        evidence[key] = {
            "maximum_peer_delta": delta,
            "changed_fraction": fraction,
            "frame_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        }
    return evidence


@contextmanager
def _capture(camera_path, config):
    import omni.replicator.core as rep

    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(camera_path, (config.width, config.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb.attach([product])
    depth.attach([product])

    def frame():
        rep.orchestrator.step(delta_time=0.0, wait_for_render=True)
        return validate_frame(rgb.get_data(), depth.get_data(), config)

    try:
        yield frame
    finally:
        rgb.detach([product])
        depth.detach([product])
        product.destroy()


@contextmanager
def _translated_camera(stage, path, translation):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    order = camera.GetXformOpOrderAttr().Get()
    op = camera.AddTranslateOp(opSuffix="npaNavigationProbe")
    op.Set(Gf.Vec3d(*translation))
    try:
        yield
    finally:
        camera.GetXformOpOrderAttr().Set(order)
        camera.GetPrim().RemoveProperty(op.GetOpName())


def _require_peer_in_view(stage, camera_path, peer):
    from pxr import UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetCamera()
    point = np.array([*peer.position_m, 1.0])
    clip = (
        point
        @ np.asarray(camera.frustum.ComputeViewMatrix())
        @ np.asarray(camera.frustum.ComputeProjectionMatrix())
    )
    if clip[3] <= 0 or np.any(np.abs(clip[:3] / clip[3]) >= 1):
        raise ValueError("peer positive placement is outside observing camera frustum")


def probe_cameras(adapter, env, recipe, output) -> dict:
    """Capture native RGB-D controls with all self and peer geometry hidden.

    Args:
        adapter: Task module supplying reset/measurement and USD paths.
        env: Native Isaac environment on an RT-core GPU.
        recipe: Validated camera recipe.
        output: Directory for actual paired RGB/depth arrays.
    Returns:
        Measured camera evidence, including fresh render count and frame hashes.
    Raises:
        ValueError: Visibility, reset, field-of-view or image controls fail.
        RuntimeError: Native rendering fails.
    """
    config, probe = recipe.camera, recipe.probe
    parked = [probe.free] + [probe.parked] * (recipe.num_envs - 1)
    verify_reset(adapter, env, parked, probe.tolerance)
    inventory = hide_robot_geometry(adapter, env, recipe)
    baseline, hidden, moved = _paired_frames(
        adapter, env, recipe, inventory["camera_prims"][0]
    )
    evidence = compare_frames(baseline, hidden, moved, config)
    _save_frames(output, baseline, hidden, moved)
    return {
        "visibility": inventory,
        "measurements": evidence,
        "fresh_render_steps": 3,
        "observer_robot_index": 0,
        "all_camera_throughput_verified": False,
    }


def _paired_frames(adapter, env, recipe, camera):
    config, probe = recipe.camera, recipe.probe
    with _capture(camera, config) as frame:
        baseline = frame()
        verify_reset(
            adapter,
            env,
            [probe.free] + [config.peer_in_view] * (recipe.num_envs - 1),
            probe.tolerance,
        )
        hide_robot_geometry(adapter, env, recipe)
        _require_peer_in_view(env.unwrapped.sim.stage, camera, config.peer_in_view)
        hidden = frame()
        with _translated_camera(env.unwrapped.sim.stage, camera, config.translation_m):
            moved = frame()
    return baseline, hidden, moved


def _save_frames(output, baseline, hidden, moved):
    np.savez_compressed(
        output / "camera-probes.npz",
        **{
            f"{name}_{key}": value
            for name, frames in (
                ("parked", baseline),
                ("peer", hidden),
                ("moved", moved),
            )
            for key, value in frames.items()
        },
    )

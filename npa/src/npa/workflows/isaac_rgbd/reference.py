"""Collect NVIDIA's public warehouse and prepare a substantial four-camera capture."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from uuid import uuid4

import numpy as np

from .contract import REQUEST_SCHEMA, _contained, _json_bytes, _sha256, validate_request
from .fixture import _fixture_camera
from .scene import _check_scene_files

_SOURCE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
_LICENSE_URL = "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/licenses.html"
_WAYPOINTS = [(-13, 6), (-13, 26), (-8, 26), (-8, 8), (-3, 8), (-3, 26)]


def _reference_cameras():
    cameras = []
    focal = 1280 / (2 * np.tan(np.deg2rad(50)))
    for name, direction in (
        ("front", [1, 0, 0]),
        ("left", [0, 1, 0]),
        ("rear", [-1, 0, 0]),
        ("right", [0, -1, 0]),
    ):
        camera = _fixture_camera(name, direction)
        camera.update(
            width=1280,
            height=720,
            intrinsics=[[focal, 0, 639.5], [0, focal, 359.5], [0, 0, 1]],
            depth_range_m=[0.05, 100.0],
        )
        cameras.append(camera)
    return cameras


def _reference_trajectory():
    samples = []
    for start, end in zip(_WAYPOINTS[:-1], _WAYPOINTS[1:], strict=True):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        forward = (end - start) / np.linalg.norm(end - start)
        rotation = [
            [forward[0], -forward[1], 0],
            [forward[1], forward[0], 0],
            [0, 0, 1],
        ]
        count = int(np.ceil(np.linalg.norm(end - start) / 0.25))
        for position in np.linspace(start, end, count, endpoint=False):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [*position, 1.5]
            samples.append(
                {
                    "timestamp_ns": len(samples) * 250_000_000,
                    "T_world_rig": pose.tolist(),
                }
            )
    last = copy.deepcopy(samples[-1])
    last["timestamp_ns"] += 250_000_000
    last["T_world_rig"][0][3], last["T_world_rig"][1][3] = _WAYPOINTS[-1]
    samples.append(last)
    return samples


def _collect_warehouse(app, root):
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.kit.usd.collect")
    app.update()
    from omni.kit.usd.collect import Collector, CollectorFailureOptions

    failures = (
        CollectorFailureOptions.EXTERNAL_USD_REFERENCES
        | CollectorFailureOptions.OTHER_EXTERNAL_REFERENCES
    )
    collector = Collector(
        _SOURCE_URL, str(root), flat_collection=True, failure_options=failures
    )
    task = asyncio.ensure_future(collector.collect())
    while not task.done():
        app.update()
    success, scene = task.result()
    if not success:
        raise ValueError("warehouse dependency collection failed")
    mapping = collector.get_source_target_url_mapping()
    # The collector's cache includes worker-local paths; retain relative provenance instead.
    cache = Path(collector.collect_mapping_file_url)
    if cache.is_relative_to(root) and cache.is_file():
        cache.unlink()
    collector.destroy()
    return Path(scene), mapping


def _write_reference_request(root, scene, mapping):
    sources = {
        Path(target).resolve().relative_to(root.resolve()).as_posix(): source
        for source, target in mapping.items()
    }
    provenance = {
        "source_url": _SOURCE_URL,
        "source_license_url": _LICENSE_URL,
        "scope": "public_authored_warehouse_reference",
        "source_by_collected_file": sources,
        "route_distance_m": 66.0,
        "sample_spacing_m": 0.25,
        "route_collision_free_verified": False,
    }
    (root / "reference.json").write_bytes(_json_bytes(provenance))
    request = {
        "schema": REQUEST_SCHEMA,
        "scene": scene.resolve().relative_to(root.resolve()).as_posix(),
        "files": {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in root.rglob("*")
            if path.is_file()
        },
        "pointcloud": True,
        "cameras": _reference_cameras(),
        "trajectory": _reference_trajectory(),
    }
    validate_request(request)
    _check_scene_files(root, request)
    return request


def _publish_reference(root, request, destination, storage):
    from .transport import _s3_uri

    destination = _s3_uri(destination)
    validate_request(request)
    _check_scene_files(root, request)
    commit = destination + "/request.json"
    if storage.read_bytes_with_etag(commit) is not None:
        raise ValueError("reference input already exists; use a fresh output prefix")
    attempt = "inputs/" + uuid4().hex
    published = copy.deepcopy(request)
    published["files"] = {
        attempt + "/" + name: digest for name, digest in request["files"].items()
    }
    published["scene"] = attempt + "/" + request["scene"]
    for name in sorted(request["files"]):
        storage.upload_file(
            str(_contained(root, name)), destination + "/" + attempt + "/" + name
        )
    storage.put_bytes_conditional(
        _json_bytes(published),
        commit,
        if_none_match=True,
        content_type="application/json",
    )


def prepare_reference(output_path, root):
    """Collect the public warehouse and commit a complete four-camera input bundle.

    Args:
        output_path: Fresh operator S3 prefix for request.json and vendor assets.
        root: Empty private directory for collected files.

    Returns:
        Input request after successful immutable S3 publication.

    Raises:
        RuntimeError: The pinned Isaac runtime or collection extension is unavailable.
        ValueError: Collection, static auditing, containment or publication fails.
        Exception: Native collector or storage failure propagates before Kit teardown.
    """
    from npa.clients.storage import StorageClient
    from npa.workflows.isaac_capture import _simulation_app_lifecycle
    from .runtime import _runtime_version
    from .transport import _s3_uri

    _s3_uri(output_path)
    _runtime_version()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("reference preparation requires an empty directory")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    with _simulation_app_lifecycle(app):
        scene, mapping = _collect_warehouse(app, root)
        request = _write_reference_request(root, scene, mapping)
        _publish_reference(root, request, output_path, StorageClient.from_environment())
    return request

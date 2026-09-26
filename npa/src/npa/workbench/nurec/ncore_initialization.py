"""Prepare native NRE multi-camera initialization from unchanged NCore SfM points."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from npa.workbench.nurec.nurec import (
    DEFAULT_CONFIG_NAME,
    NO_LIDAR_SENTINEL,
    NurecConfig,
    NurecError,
    ncore_sensor_ids,
    read_rig_sidecar,
    verify_ncore_input,
)

_INITIALIZATION = "model.layers.background.initialization"
_INITIALIZATION_GROUP = f"model/gaussians/initialization@{_INITIALIZATION}"


def plan_initialization(
    config: NurecConfig, ncore_json: str
) -> tuple[NurecConfig, dict[str, Any]]:
    """Resolve sensors and plan the native initializer without writing files.

    Args:
        config: Resolved caller configuration, including explicit overrides.
        ncore_json: Local NCore sequence metadata.
    Returns:
        Configuration with native overrides and optional initialization evidence.
    Raises:
        NurecError: The selected input or point inventory is invalid.
    """
    cameras, lidars = ncore_sensor_ids(ncore_json)
    config = replace(
        config,
        camera_ids=config.camera_ids or cameras,
        lidar_ids=config.lidar_ids or lidars or (NO_LIDAR_SENTINEL,),
    )
    if not _needs_accumulated_points(config, ncore_json):
        return config, {}
    selected = _selected_cameras(config)
    if len(selected) < 2:
        return config, {}
    target = config.resolved_out_dir / "initialization" / "ncore-sfm.ply"
    sequence_dir = Path(ncore_json).resolve().parent
    if (sequence_dir / "conversion.json").exists() and target.resolve().is_relative_to(
        sequence_dir
    ):
        raise NurecError(
            "initialization output must be outside the NCore sequence directory"
        )
    evidence = _initialization_evidence(config, ncore_json, target, selected)
    overrides = _native_overrides(target, evidence["point_count"])
    return replace(
        config, extra_overrides=(*overrides, *config.extra_overrides)
    ), evidence


def _selected_cameras(config: NurecConfig) -> tuple[str, ...]:
    import yaml

    selected = config.camera_ids
    for override in config.extra_overrides:
        key, separator, value = override.partition("=")
        if key.lstrip("+") != "dataset.camera_ids" or not separator:
            continue
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise NurecError(
                "dataset.camera_ids override must be a camera ID list"
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise NurecError("dataset.camera_ids override must be a camera ID list")
        selected = tuple(parsed)
    return selected


def _needs_accumulated_points(config: NurecConfig, ncore_json: str) -> bool:
    if config.config_name != DEFAULT_CONFIG_NAME:
        return False
    for override in config.extra_overrides:
        key = override.partition("=")[0].lstrip("+~")
        if key.startswith(_INITIALIZATION) or key == _INITIALIZATION_GROUP:
            return False
    return bool(read_rig_sidecar(ncore_json).get("reference_camera"))


def _initialization_evidence(
    config: NurecConfig, ncore_json: str, target: Path, cameras: tuple[str, ...]
) -> dict[str, Any]:
    report_path = Path(ncore_json).parent / "conversion.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    count, source, legacy_sources = _point_inventory(ncore_json, report)
    if type(count) is not int or count <= 0:
        raise NurecError("initialization requires a nonempty NCore point inventory")
    return {
        "status": "planned",
        "recipe": config.config_name,
        "image": config.image,
        "initializer": "accumulated-point-cloud",
        "camera_ids": list(cameras),
        "point_cloud_path": str(target),
        "point_count": count,
        "reference_frame": "world",
        "source": source,
        **({"legacy_sources": legacy_sources} if legacy_sources else {}),
        "ncore_meta_sha256": _sha256(Path(ncore_json)),
        "conversion_report_sha256": _sha256(report_path) if report else "",
        "sampling": "all source points; no optional random near/far points",
    }


def _point_inventory(
    ncore_json: str, report: dict[str, Any]
) -> tuple[Any, str, dict[str, str]]:
    source = "NCore PointCloudsComponent SfM XYZ and rgb"
    if report:
        return report.get("counts", {}).get("points"), source, {}
    readers = _point_readers(ncore_json)
    count = sum(
        len(points.get_pc_xyz(index))
        for points in readers.values()
        for index in range(points.pcs_count)
    )
    legacy_sources = _legacy_sources(readers)
    if legacy_sources:
        source = readers["virtual_lidar"].source_description
    return count, source, legacy_sources


def _native_overrides(target: Path, count: int) -> tuple[str, ...]:
    return (
        f"{_INITIALIZATION_GROUP}=accumulated_point_cloud",
        f"{_INITIALIZATION}.point_cloud_path={json.dumps(str(target))}",
        f"{_INITIALIZATION}.num_point_cloud_points={count}",
        f"{_INITIALIZATION}.num_near_points=0",
        f"{_INITIALIZATION}.num_far_points=0",
    )


def _point_readers(ncore_json: str) -> dict[str, Any]:
    try:
        from ncore.data.v4 import PointCloudsComponent, SequenceComponentGroupsReader
        from upath import UPath
    except ImportError as exc:
        raise NurecError(
            "multi-camera initialization requires the public NVIDIA NCore V4 reader"
        ) from exc
    reader = SequenceComponentGroupsReader([UPath(ncore_json)])
    points = reader.open_component_readers(PointCloudsComponent.Reader)
    if points:
        return points
    from npa.workbench.nurec.ncore_legacy_sfm import legacy_sfm_readers

    return legacy_sfm_readers(reader)


def _legacy_sources(readers: dict[str, Any]) -> dict[str, str]:
    return {
        name: points.source_sha256
        for name, points in readers.items()
        if getattr(points, "source_sha256", "")
    }


def export_initialization(ncore_json: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Write all verified world XYZ/RGB points as a native NRE PLY input.

    Args:
        ncore_json: Local NCore sequence metadata.
        plan: Evidence returned by plan_initialization.
    Returns:
        Export evidence with PLY hash and decoded component counts.
    Raises:
        NurecError: Geometry, colors, counts or the source inventory are invalid.
        OSError: The reconstruction output cannot be written.
    """
    if not plan:
        return {}
    _verify_source(ncore_json, plan)
    clouds, components = _read_clouds(ncore_json, plan.get("legacy_sources", {}))
    count = sum(len(xyz) for xyz, _ in clouds)
    if not count or count != plan["point_count"]:
        raise NurecError(
            "decoded initialization point count differs from the NCore inventory"
        )
    _verify_source(ncore_json, plan)
    target = Path(plan["point_cloud_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    _store_ply(target, clouds, count)
    _verify_source(ncore_json, plan)
    evidence = {
        **plan,
        "status": "exported",
        "point_cloud_sha256": _sha256(target),
        "point_cloud_bytes": target.stat().st_size,
        "components": components,
    }
    _store_evidence(target.with_suffix(".json"), evidence)
    return evidence


def _verify_source(ncore_json: str, plan: dict[str, Any]) -> None:
    verify_ncore_input(ncore_json)
    if _sha256(Path(ncore_json)) != plan["ncore_meta_sha256"]:
        raise NurecError("NCore metadata changed during initialization export")
    expected = plan["conversion_report_sha256"]
    if expected and _sha256(Path(ncore_json).parent / "conversion.json") != expected:
        raise NurecError("conversion inventory changed during initialization export")
    if plan.get("legacy_sources"):
        _verify_legacy_sources(_point_readers(ncore_json), plan["legacy_sources"])


def _verify_legacy_sources(readers: dict[str, Any], expected: dict[str, str]) -> None:
    if _legacy_sources(readers) != expected:
        raise NurecError(
            "legacy SfM source points changed since initialization planning"
        )


def _read_clouds(
    ncore_json: str,
    expected_legacy_sources: dict[str, str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    clouds = []
    components = []
    readers = _point_readers(ncore_json)
    _verify_legacy_sources(readers, expected_legacy_sources)
    for name, points in readers.items():
        for index in range(points.pcs_count):
            xyz, rgb = _read_cloud(points, index)
            clouds.append((xyz, rgb))
            components.append(
                {"component": name, "index": index, "point_count": len(xyz)}
            )
    return clouds, components


def _read_cloud(points: Any, index: int) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points.get_pc_xyz(index))
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not len(xyz)
        or xyz.dtype.kind not in "fiu"
        or not np.isfinite(xyz).all()
        or points.get_pc_reference_frame_id(index) != "world"
    ):
        raise NurecError("initialization requires nonempty finite world XYZ points")
    if "rgb" not in points.attribute_names:
        raise NurecError("initialization point cloud is missing RGB colors")
    rgb = np.asarray(points.get_pc_attribute(index, "rgb"))
    if rgb.dtype != np.uint8 or rgb.shape != xyz.shape:
        raise NurecError("initialization requires one uint8 RGB color per point")
    return xyz.copy(), rgb.copy()


def _store_ply(
    target: Path, clouds: list[tuple[np.ndarray, np.ndarray]], count: int
) -> None:
    try:
        with target.open("xb") as stream:
            _write_ply(stream.write, clouds, count)
    except FileExistsError:
        digest = hashlib.sha256()
        _write_ply(digest.update, clouds, count)
        if (
            target.is_symlink()
            or not target.is_file()
            or _sha256(target) != digest.hexdigest()
        ):
            raise NurecError(
                "existing initialization PLY differs; use a fresh output directory"
            )


def _store_evidence(target: Path, evidence: dict[str, Any]) -> None:
    content = json.dumps(evidence, indent=2) + "\n"
    try:
        with target.open("x") as stream:
            stream.write(content)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_text() != content:
            raise NurecError(
                "existing initialization evidence differs; use a fresh output directory"
            )


def _write_ply(
    write: Any, clouds: list[tuple[np.ndarray, np.ndarray]], count: int
) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    write(header.encode("ascii"))
    dtype = np.dtype([("xyz", "<f8", (3,)), ("rgb", "u1", (3,))])
    for xyz, rgb in clouds:
        vertices = np.empty(len(xyz), dtype=dtype)
        vertices["xyz"], vertices["rgb"] = xyz, rgb
        write(vertices.tobytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

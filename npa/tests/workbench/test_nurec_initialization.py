"""Exercise full-camera native initialization and immutable XYZ/RGB export."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.nurec import colmap, ncore_initialization as initialization, nurec


class PointReader:
    pcs_count = 1
    attribute_names = ("rgb",)

    def __init__(self, xyz, rgb):
        self.xyz = np.asarray(xyz, dtype=np.float32)
        self.rgb = np.asarray(rgb, dtype=np.uint8)
        self.reference_frame = "world"

    def get_pc_xyz(self, index):
        return self.xyz

    def get_pc_attribute(self, index, attribute):
        assert attribute == "rgb"
        return self.rgb

    def get_pc_reference_frame_id(self, index):
        return self.reference_frame


def seal_conversion(root, count=3, image_count=518):
    members = [
        {"path": path.name, "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(root.iterdir())
        if path.name not in (colmap.CONVERSION_REPORT, colmap.PUBLICATION_CLAIM)
    ]
    report = root / colmap.CONVERSION_REPORT
    report.write_text(json.dumps({
        "schema_version": 1, "status": "ok", "engine": "nvidia-ncore-colmap",
        "ncore_meta": "sequence.json", "poses_component_group": "npa_rig",
        "members": members,
        "counts": {"cameras": 3, "images": image_count, "poses": image_count, "points": count},
        "publication": {"mode": "immutable-prefix-v1", "claim": colmap.PUBLICATION_CLAIM},
    }))
    (root / colmap.PUBLICATION_CLAIM).write_text(json.dumps({
        "schema_version": 1, "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }))
    colmap.verify_conversion_inventory(root)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    root = tmp_path / "sequence"
    root.mkdir()
    meta = root / "sequence.json"
    meta.write_text(json.dumps({
        "version": "v4", "component_stores": [{
            "path": "data.zarr.itar",
            "components": {"cameras": {f"camera{i}": {} for i in (1, 2, 3)}},
        }],
    }))
    (root / "data.zarr.itar").write_bytes(b"synthetic point-reader fixture")
    (root / "npa-rig.json").write_text('{"reference_camera":"camera2"}')
    seal_conversion(root)
    readers = {
        "sfm1": PointReader([[1.25, -2, 3], [4, 5, 6]], [[255, 0, 5], [6, 7, 8]]),
        "sfm2": PointReader([[9, 10, 11]], [[12, 13, 14]]),
    }
    monkeypatch.setattr(initialization, "_point_readers", lambda _: readers)
    return meta, readers


def reconstruct(meta, out, **kwargs):
    return nurec.reconstruct_scene(
        nurec.NurecConfig(out_dir=out, **kwargs), ncore_json=str(meta), dry_run=True,
    )


def test_cli_plans_all_three_cameras_without_writing_or_reading_points(capture, tmp_path, monkeypatch):
    meta, _ = capture
    before = {p.name: p.read_bytes() for p in meta.parent.iterdir()}
    monkeypatch.setattr(initialization, "_point_readers", lambda _: pytest.fail("dry-run decoded points"))
    result = CliRunner().invoke(app, [
        "workbench", "nurec", "reconstruct", "--ncore-json", str(meta),
        "--out-dir", str(tmp_path / "out"), "--dry-run", "--output", "json",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    command = payload["command"]
    assert "dataset.camera_ids=['camera1','camera2','camera3']" in command
    assert "model/gaussians/initialization@model.layers.background.initialization=accumulated_point_cloud" in command
    assert "model.layers.background.initialization.num_point_cloud_points=3" in command
    assert "model.layers.background.initialization.num_near_points=0" in command
    assert "model.layers.background.initialization.num_far_points=0" in command
    assert not any("max_epochs=" in arg or "n_samples_per_epoch=" in arg for arg in command)
    assert payload["initialization"]["status"] == "planned"
    assert not (tmp_path / "out").exists()
    assert before == {p.name: p.read_bytes() for p in meta.parent.iterdir()}


@pytest.mark.parametrize("kwargs", [
    {"camera_ids": ("camera3",)},
    {"extra_overrides": ("dataset.camera_ids=[camera3]",)},
])
def test_single_camera_selection_retains_native_sfm(capture, tmp_path, kwargs):
    result = reconstruct(capture[0], tmp_path / "out", **kwargs)
    assert not result.initialization
    assert not any("accumulated_point_cloud" in arg for arg in result.command)


@pytest.mark.parametrize("kwargs", [
    {"camera_ids": ("camera1", "camera3")},
    {"extra_overrides": ("dataset.camera_ids=[camera1,camera3]",)},
])
def test_explicit_multicamera_subset_uses_all_points(capture, tmp_path, kwargs):
    result = reconstruct(capture[0], tmp_path / "out", **kwargs)
    assert result.initialization["camera_ids"] == ["camera1", "camera3"]
    assert result.initialization["point_count"] == 3


@pytest.mark.parametrize("override", [
    "model/gaussians/initialization@model.layers.background.initialization=sfm_point_cloud",
    "model.layers.background.initialization.name=custom",
    "+model.layers.background.initialization.num_near_points=20",
])
def test_explicit_initialization_is_forwarded_unchanged(capture, tmp_path, override):
    result = reconstruct(capture[0], tmp_path / "out", extra_overrides=(override,))
    assert not result.initialization
    assert result.command[-1] == override


def test_custom_recipe_and_explicit_training_budget_are_preserved(capture, tmp_path):
    result = reconstruct(capture[0], tmp_path / "out", config_name="custom.yaml")
    assert not result.initialization
    result = reconstruct(
        capture[0], tmp_path / "out", max_epochs=4,
        extra_overrides=("dataset.n_samples_per_epoch=12345",),
    )
    assert result.initialization
    assert "trainer.max_epochs=4" in result.command
    assert result.command[-1] == "dataset.n_samples_per_epoch=12345"


def test_environment_camera_selection_survives_discovery(capture, tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_NUREC_CAMERA_IDS", "camera3")
    config = nurec.NurecConfig.from_env(out_dir=tmp_path / "out")
    result = nurec.reconstruct_scene(config, ncore_json=str(capture[0]), dry_run=True)
    assert "dataset.camera_ids=['camera3']" in result.command
    assert not result.initialization


def read_ply(path):
    header, body = path.read_bytes().split(b"end_header\n", 1)
    dtype = np.dtype([("xyz", "<f8", (3,)), ("rgb", "u1", (3,))])
    return header, np.frombuffer(body, dtype=dtype)


def test_real_export_preserves_all_xyz_rgb_and_inventory_before_launch(capture, tmp_path):
    meta, readers = capture
    before = {p.name: p.read_bytes() for p in meta.parent.iterdir()}
    out = tmp_path / "out"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        header, points = read_ply(out / "initialization/ncore-sfm.ply")
        assert b"element vertex 3\n" in header
        np.testing.assert_array_equal(points["xyz"], np.concatenate([p.xyz for p in readers.values()]))
        np.testing.assert_array_equal(points["rgb"], np.concatenate([p.rgb for p in readers.values()]))
        colmap.verify_conversion_inventory(meta.parent)
        (out / "nre/artifacts").mkdir(parents=True)
        (out / "nre/artifacts/last.usdz").write_bytes(b"test artifact")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = nurec.reconstruct_scene(
        nurec.NurecConfig(out_dir=out), ncore_json=str(meta), runner=run, export_gt=False,
    )
    assert result.ok and len(calls) == 1
    evidence = result.initialization
    assert evidence["status"] == "exported" and evidence["point_count"] == 3
    assert evidence["point_cloud_sha256"] == hashlib.sha256(Path(evidence["point_cloud_path"]).read_bytes()).hexdigest()
    assert json.loads((out / "initialization/ncore-sfm.json").read_text()) == evidence
    assert before == {p.name: p.read_bytes() for p in meta.parent.iterdir()}


@pytest.mark.parametrize("corruption, message", [
    ("nan", "finite world XYZ"), ("inf", "finite world XYZ"),
    ("shape", "finite world XYZ"), ("empty", "finite world XYZ"),
    ("frame", "finite world XYZ"), ("missing_rgb", "missing RGB"),
    ("rgb_float", "uint8 RGB"), ("rgb_count", "uint8 RGB"),
])
def test_invalid_point_data_fails_before_launch(capture, tmp_path, corruption, message):
    meta, readers = capture
    points = readers["sfm1"]
    if corruption in ("nan", "inf"):
        points.xyz[0, 0] = float(corruption)
    elif corruption == "shape":
        points.xyz = points.xyz[:, :2]
    elif corruption == "empty":
        points.xyz = points.xyz[:0]
    elif corruption == "frame":
        points.reference_frame = "rig"
    elif corruption == "missing_rgb":
        points.attribute_names = ()
    elif corruption == "rgb_float":
        points.rgb = points.rgb.astype(float)
    else:
        points.rgb = points.rgb[:1]
    with pytest.raises(nurec.NurecError, match=message):
        nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"), ncore_json=str(meta),
            runner=lambda *a, **k: pytest.fail("launched invalid input"),
        )
    assert not (tmp_path / "out").exists()


def test_missing_points_and_count_mismatch_fail_before_launch(capture, tmp_path):
    meta, readers = capture
    readers.clear()
    with pytest.raises(nurec.NurecError, match="count differs"):
        nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"), ncore_json=str(meta),
            runner=lambda *a, **k: pytest.fail("launched empty input"),
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("suffix", ["ply", "json"])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_initialization_rejects_nonregular_retry_outputs(capture, tmp_path, suffix, kind):
    import os

    target = tmp_path / "out/initialization" / f"ncore-sfm.{suffix}"
    target.parent.mkdir(parents=True)
    if kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(nurec.NurecError, match="existing initialization"):
        nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"), ncore_json=str(capture[0]),
            runner=lambda *a, **k: pytest.fail("launched unsafe initialization"),
        )


@pytest.mark.parametrize("reseal", [False, True])
def test_inventory_change_during_decode_is_rejected(capture, tmp_path, monkeypatch, reseal):
    meta, readers = capture

    def changed_reader(_):
        (meta.parent / "data.zarr.itar").write_bytes(b"changed source")
        if reseal:
            seal_conversion(meta.parent)
        return readers

    monkeypatch.setattr(initialization, "_point_readers", changed_reader)
    with pytest.raises(nurec.NurecError, match="inventory"):
        nurec.reconstruct_scene(
            nurec.NurecConfig(out_dir=tmp_path / "out"), ncore_json=str(meta),
            runner=lambda *a, **k: pytest.fail("launched changed source"),
        )
    assert not (tmp_path / "out").exists()


def test_initialization_cannot_write_inside_conversion_generation(capture):
    meta, _ = capture
    with pytest.raises(nurec.NurecError, match="outside the NCore"):
        reconstruct(meta, meta.parent / "out")
    colmap.verify_conversion_inventory(meta.parent)


def test_published_reconstruction_includes_initializer_evidence(capture, tmp_path):
    from npa.cli.nurec import _publish_reconstruction

    meta, _ = capture
    planned = reconstruct(meta, tmp_path / "out")
    evidence = initialization.export_initialization(str(meta), planned.initialization)
    from dataclasses import replace

    _publish_reconstruction(replace(planned, initialization=evidence), str(tmp_path / "published"))
    assert (tmp_path / "published/initialization/ncore-sfm.ply").is_file()
    assert (tmp_path / "published/initialization/ncore-sfm.json").is_file()


def public_ncore_sequence(root):
    v4 = pytest.importorskip("ncore.data.v4")
    from ncore.impl.common.transformations import HalfClosedInterval
    from ncore.impl.data.types import PointCloud
    from upath import UPath

    writer = v4.SequenceComponentGroupsWriter(
        output_dir_path=UPath(root), store_base_name="sequence", sequence_id="fixture",
        sequence_timestamp_interval_us=HalfClosedInterval(0, 2_000_000),
        generic_meta_data={}, store_type="itar",
    )
    write_public_cameras(writer, v4)
    points = writer.register_component_writer(
        v4.PointCloudsComponent.Writer, "sfm_points",
        coordinate_unit=PointCloud.CoordinateUnit.UNITLESS,
        attribute_schemas={"rgb": v4.PointCloudsComponent.AttributeSchema(
            transform_type=PointCloud.AttributeTransformType.INVARIANT,
            dtype=np.dtype("uint8"), shape_suffix=(3,),
        )},
    )
    xyz = np.array([[1.125, -2.25, 3.5], [7, 8, 9]], dtype=np.float32)
    rgb = np.array([[255, 0, 13], [4, 5, 6]], dtype=np.uint8)
    points.store_pc(xyz=xyz, reference_frame_id="world", reference_frame_timestamp_us=0, attributes={"rgb": rgb})
    paths = writer.finalize()
    meta = root / "sequence.json"
    meta.write_text(json.dumps(v4.SequenceComponentGroupsReader(paths).get_sequence_meta().to_dict()))
    (root / "npa-rig.json").write_text('{"reference_camera":"camera1"}')
    seal_conversion(root, count=2, image_count=6)
    return meta, xyz, rgb


def write_public_cameras(writer, v4):
    import io
    from PIL import Image
    from ncore.impl.data.types import OpenCVPinholeCameraModelParameters, ShutterType

    poses = writer.register_component_writer(v4.PosesComponent.Writer, "npa_rig")
    trajectory = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    trajectory[1, 0, 3] = 1
    timestamps = np.array([0, 1_000_000], dtype=np.uint64)
    poses.store_dynamic_pose("rig", "world", trajectory, timestamps, require_sequence_time_coverage=False)
    intrinsics = writer.register_component_writer(v4.IntrinsicsComponent.Writer, "default")
    model = OpenCVPinholeCameraModelParameters(
        resolution=np.array([4, 3], dtype=np.uint64), shutter_type=ShutterType.GLOBAL,
        focal_length=np.array([3, 3], dtype=np.float32),
        principal_point=np.array([2, 1.5], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32), tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    for camera_id in ("camera1", "camera2", "camera3"):
        intrinsics.store_camera_intrinsics(camera_id, model)
        poses.store_dynamic_pose(camera_id, "world", trajectory, timestamps, require_sequence_time_coverage=False)
        camera = writer.register_component_writer(v4.CameraSensorComponent.Writer, camera_id, group_name=camera_id)
        for timestamp in timestamps:
            image = io.BytesIO()
            Image.new("RGB", (4, 3), (20, 30, 40)).save(image, format="PNG")
            camera.store_frame(
                image_binary_data=image.getvalue(), image_format="png",
                frame_timestamps_us=np.array([timestamp, timestamp], dtype=np.uint64),
                generic_data={}, generic_meta_data={},
            )


def test_public_ncore_v4_points_round_trip_to_native_ply(tmp_path):
    meta, xyz, rgb = public_ncore_sequence(tmp_path / "sequence")
    result = reconstruct(meta, tmp_path / "out")
    assert result.initialization["camera_ids"] == ["camera1", "camera2", "camera3"]
    evidence = initialization.export_initialization(str(meta), result.initialization)
    header, vertices = read_ply(Path(evidence["point_cloud_path"]))
    assert b"element vertex 2\n" in header
    np.testing.assert_array_equal(vertices["xyz"], xyz)
    np.testing.assert_array_equal(vertices["rgb"], rgb)
    colmap.verify_conversion_inventory(meta.parent)


def test_native_ply_library_reads_the_exported_geometry_and_colors(capture, tmp_path):
    point_cloud_utils = pytest.importorskip("point_cloud_utils")
    meta, readers = capture
    result = reconstruct(meta, tmp_path / "out")
    evidence = initialization.export_initialization(str(meta), result.initialization)
    vertices = point_cloud_utils.load_triangle_mesh(evidence["point_cloud_path"]).vertex_data
    np.testing.assert_array_equal(vertices.positions, np.concatenate([p.xyz for p in readers.values()]))
    np.testing.assert_array_equal(vertices.colors[:, :3], np.concatenate([p.rgb for p in readers.values()]))


def test_retry_reuses_identical_initialization_but_rejects_changed_output(capture, tmp_path):
    meta, _ = capture
    result = reconstruct(meta, tmp_path / "out")
    first = initialization.export_initialization(str(meta), result.initialization)
    second = initialization.export_initialization(str(meta), result.initialization)
    assert first == second
    Path(first["point_cloud_path"]).write_bytes(b"changed output")
    with pytest.raises(nurec.NurecError, match="existing initialization PLY differs"):
        initialization.export_initialization(str(meta), result.initialization)

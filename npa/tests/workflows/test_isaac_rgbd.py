"""Exercise calibrated RGB-D math and hostile data without starting Isaac or S3."""

from __future__ import annotations

import copy
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from npa.workflows.isaac_rgbd import contract, dataset, geometry, reference, transport
from npa.workflows.isaac_rgbd.fixture import write_fixture


@pytest.fixture
def rig_request(tmp_path):
    value = write_fixture(tmp_path / "inputs")
    for camera in value["cameras"]:
        camera.update(
            width=6, height=4, intrinsics=[[4, 0, 2.5], [0, 5, 1.5], [0, 0, 1]]
        )
    return value


def _snapshot(camera, sample, index):
    # Unit-test arrays deliberately model annotated buffers, never live evidence.
    height, width = camera["height"], camera["width"]
    pose = np.asarray(sample["T_world_rig"]) @ camera["T_rig_camera"]
    optics = geometry._usd_camera_parameters(camera)
    depth = np.full((height, width), 2.0, dtype=np.float32)
    depth[0, :4] = [np.nan, np.inf, -1, 0]
    return {
        "rgb": np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4),
        "depth": depth,
        "render_reference_time": {"numerator": index + 1, "denominator": 10},
        "render_calibration": {
            "focal": optics["focal"],
            "aperture": optics["aperture"],
            "offset": optics["offset"],
            "resolution": [width, height],
            "view_transform": np.linalg.inv(geometry._optical_to_usd(pose)).T.tolist(),
        },
    }


@pytest.fixture
def captured(tmp_path, rig_request):
    root = tmp_path / "capture"
    root.mkdir()
    frames = []
    for index, sample in enumerate(rig_request["trajectory"]):
        snapshots = {
            camera["id"]: _snapshot(camera, sample, index)
            for camera in rig_request["cameras"]
        }
        frames.append(
            dataset._write_frame(
                root, rig_request, index, snapshots, sample["timestamp_ns"] / 1e9
            )
        )
    provenance = {
        "request_sha256": hashlib.sha256(contract._json_bytes(rig_request)).hexdigest(),
        "input_manifest_sha256": "a" * 64,
        "scope": "procedural-room",
        "replicator_version": "unit-test",
    }
    return root, dataset._finalize(root, rig_request, frames, provenance)


def test_backprojection_uses_axial_depth_and_rigid_world_transform():
    depth = np.array([[2, 0], [4, 2]], dtype=np.float32)
    colors = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    pose = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
    points, rgb = geometry.backproject(
        depth, colors, [[2, 0, 0], [0, 4, 0], [0, 0, 1]], pose
    )
    np.testing.assert_allclose(points, [[10, 20, 32], [9, 20, 34], [9.5, 21, 32]])
    np.testing.assert_array_equal(rgb, colors[[0, 1, 1], [0, 0, 1]])


def test_cpu_validation_binds_measured_native_material_identity(captured):
    root, manifest = captured
    declared = {
        "isaac_sim_version": "6.0.1.0",
        "image": "registry.example/isaac@sha256:" + "a" * 64,
        "library_sha256": "b" * 64,
        "modules": {
            "OmniPBR.mdl": {
                "path": "core/Base/OmniPBR.mdl",
                "sha256": "c" * 64,
            }
        },
    }
    manifest["request"]["runtime_materials"] = declared
    manifest["provenance"]["request_sha256"] = hashlib.sha256(
        contract._json_bytes(manifest["request"])
    ).hexdigest()
    with pytest.raises(ValueError, match="native MDL provenance"):
        dataset.validate_dataset(root, manifest)
    manifest["provenance"]["runtime_materials"] = copy.deepcopy(declared)
    assert dataset.validate_dataset(root, manifest)["validated"]
    manifest["provenance"]["runtime_materials"]["library_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="native MDL provenance"):
        dataset.validate_dataset(root, manifest)


def test_projection_parameters_preserve_off_center_non_square_calibration(rig_request):
    camera = rig_request["cameras"][0]
    camera["intrinsics"] = [[8, 0, 1], [0, 4, 2], [0, 0, 1]]
    optics = geometry._usd_camera_parameters(camera)
    assert optics == {"focal": 24, "aperture": [18, 24], "offset": [4.5, 3]}
    # Invert the physical film window; u,v denote centers, not image edges.
    for u, v in [(0, 0), (5, 3), (1, 2)]:
        x = ((u + 0.5) / 6 - 0.5) * 18 + 4.5
        y = (0.5 - (v + 0.5) / 4) * 24 + 3
        np.testing.assert_allclose([x / 24, -y / 24], [(u - 1) / 8, (v - 2) / 4])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_backprojection_rejects_unclean_depth(bad):
    with pytest.raises(ValueError):
        geometry.backproject(
            np.array([[bad]]), np.zeros((1, 1, 3), dtype=np.uint8), np.eye(3), np.eye(4)
        )


@pytest.mark.parametrize("depth", [np.zeros((1, 1)), np.array([["2"]]), np.ones(3)])
def test_backprojection_rejects_empty_or_wrong_shape(depth):
    with pytest.raises(ValueError):
        geometry.backproject(
            depth, np.zeros((1, 1, 3), dtype=np.uint8), np.eye(3), np.eye(4)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("height", 0),
        ("width", 2.5),
        ("id", "../escape"),
        ("intrinsics", [[0, 0, 1], [0, 1, 1], [0, 0, 1]]),
        ("intrinsics", [[4, 1, 1], [0, 4, 1], [0, 0, 1]]),
        ("intrinsics", [[4, 0, 1], [0, float("nan"), 1], [0, 0, 1]]),
        ("intrinsics", [[4, 0, 9], [0, 4, 1], [0, 0, 1]]),
        ("depth_range_m", [0, 20]),
        ("depth_range_m", [1, float("inf")]),
        ("depth_range_m", [3, 1]),
        ("T_rig_camera", np.zeros((4, 4)).tolist()),
    ],
)
def test_rejects_bad_calibration(rig_request, field, value):
    rig_request["cameras"][0][field] = value
    with pytest.raises(ValueError):
        contract.validate_request(rig_request)


@pytest.mark.parametrize(
    "rotation", [np.diag([-1, 1, 1]), np.diag([2, 1, 1]), np.full((3, 3), np.nan)]
)
def test_rejects_reflection_scale_nan_transforms(rig_request, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    rig_request["trajectory"][0]["T_world_rig"] = pose.tolist()
    with pytest.raises(ValueError):
        contract.validate_request(rig_request)


@pytest.mark.parametrize(
    "timestamps", [[0, 0, 1], [2, 1, 3], [-1, 0, 1], [False, 1, 2], [0, 1, 2**54]]
)
def test_rejects_invalid_timestamps(rig_request, timestamps):
    for sample, time in zip(rig_request["trajectory"], timestamps, strict=True):
        sample["timestamp_ns"] = time
    with pytest.raises(ValueError):
        contract.validate_request(rig_request)


@pytest.mark.parametrize(
    "mutation",
    ["no_cameras", "duplicate", "no_samples", "unknown", "no_scene", "no_files"],
)
def test_rejects_incomplete_requests(rig_request, mutation):
    if mutation == "no_cameras":
        rig_request["cameras"] = []
    elif mutation == "duplicate":
        rig_request["cameras"].append(rig_request["cameras"][0])
    elif mutation == "no_samples":
        rig_request["trajectory"] = []
    elif mutation == "unknown":
        rig_request["cameras"][0]["distortion"] = [0, 0]
    elif mutation == "no_scene":
        rig_request["scene"] = "missing.usda"
    else:
        rig_request["files"] = {}
    with pytest.raises(ValueError):
        contract.validate_request(rig_request)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "a/../escape",
        "a//b",
        "./a",
        "a\\b",
        "%2e%2e/x",
        "a?x",
        "https://host/a",
        ".hidden",
        "a\x00b",
    ],
)
def test_rejects_hostile_asset_paths(path, tmp_path):
    with pytest.raises(ValueError):
        contract._contained(tmp_path, path)


def test_rejects_symlinks_and_case_or_prefix_collisions(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        contract._contained(tmp_path, "link/file")
    for files in [{"a": "a" * 64, "a/b": "a" * 64}, {"A": "a" * 64, "a": "a" * 64}]:
        with pytest.raises(ValueError, match="colli"):
            contract._file_hashes(files)


@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_strict_json_rejects_nonfinite_and_duplicate_fields(tmp_path, text):
    path = tmp_path / "request.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        contract._read_json(path)


def test_decodes_complete_aligned_dataset_and_masks_invalid_depth(captured):
    root, manifest = captured
    report = dataset.validate_dataset(root)
    assert report == {
        "schema": "npa.isaac.rgbd.validation.v1",
        "validated": True,
        "frames": 3,
        "cameras": 4,
        "views": 12,
        "valid_depth_pixels": 240,
        "fused_points": 240,
    }
    assert [view["camera_id"] for view in manifest["frames"][0]["views"]] == [
        "front",
        "left",
        "rear",
        "right",
    ]
    paths = manifest["frames"][0]["views"][0]["artifacts"]
    depth = np.load(root / paths["depth"], allow_pickle=False)
    assert np.isfinite(depth).all() and (depth[0, :4] == 0).all()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_camera", "coverage"),
        ("reorder", "coverage"),
        ("timestamp", "timestamp"),
        ("pose", "extrinsics"),
        ("time", "synchronized"),
        ("stale", "stale"),
        ("missing_frame", "every trajectory"),
        ("empty", "every trajectory"),
        ("calibration", "calibration"),
        ("timeline", "timeline"),
    ],
)
def test_rejects_metadata_misalignment(captured, mutation, match):
    root, manifest = captured
    frame = manifest["frames"][0]
    if mutation == "missing_camera":
        frame["views"].pop()
    elif mutation == "reorder":
        frame["views"].reverse()
    elif mutation == "timestamp":
        frame["timestamp_ns"] += 1
    elif mutation == "pose":
        frame["views"][0]["T_world_camera"][0][3] += 1
    elif mutation == "time":
        frame["views"][0]["render_reference_time"]["numerator"] += 1
    elif mutation == "stale":
        for view in manifest["frames"][1]["views"]:
            view["render_reference_time"] = {"numerator": 1, "denominator": 10}
    elif mutation in {"empty", "missing_frame"}:
        manifest["frames"] = [] if mutation == "empty" else manifest["frames"][:-1]
    elif mutation == "calibration":
        frame["views"][0]["render_calibration"]["focal"] = 1
    else:
        frame["simulation_time_s"] = 0.5
    with pytest.raises(ValueError, match=match):
        dataset.validate_dataset(root, manifest)


def _replace_artifact(root, manifest, kind, data):
    name = manifest["frames"][0]["views"][0]["artifacts"][kind]
    path = root / name
    with path.open("wb") as stream:
        if kind == "points":
            np.savez(stream, **data)
        else:
            np.save(stream, data, allow_pickle=False)
    manifest["files"][name] = contract._sha256(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "nan",
        "empty",
        "mask",
        "shape",
        "negative",
        "clipped",
        "float64",
        "points",
        "colors",
    ],
)
def test_decoded_validation_catches_corruption_even_with_updated_hash(
    captured, mutation
):
    root, manifest = captured
    view = manifest["frames"][0]["views"][0]
    depth = np.load(root / view["artifacts"]["depth"], allow_pickle=False)
    if mutation == "mask":
        _replace_artifact(root, manifest, "mask", np.ones_like(depth, dtype=bool))
    elif mutation in {"points", "colors"}:
        with np.load(root / view["artifacts"]["points"], allow_pickle=False) as cloud:
            points, colors = cloud["xyz_world_m"], cloud["rgb"]
        if mutation == "points":
            points[0, 0] += 1
        else:
            colors[0, 0] += 1
        _replace_artifact(
            root, manifest, "points", {"xyz_world_m": points, "rgb": colors}
        )
    else:
        replacements = {"nan": np.nan, "empty": 0, "negative": -1, "clipped": 100}
        if mutation in replacements:
            depth[:] = replacements[mutation]
        elif mutation == "shape":
            depth = depth[:1]
        else:
            depth = depth.astype(np.float64)
        _replace_artifact(root, manifest, "depth", depth)
    with pytest.raises(ValueError):
        dataset.validate_dataset(root, manifest)


def test_missing_and_undecodable_data_never_validate(captured):
    root, manifest = captured
    name = manifest["frames"][0]["views"][0]["artifacts"]["rgb"]
    (root / name).write_bytes(b"not an image")
    with pytest.raises(ValueError, match="SHA256"):
        dataset.validate_dataset(root, manifest)
    manifest["files"][name] = contract._sha256(root / name)
    with pytest.raises(OSError):
        dataset.validate_dataset(root, manifest)
    (root / name).unlink()
    with pytest.raises(ValueError, match="missing"):
        dataset.validate_dataset(root, manifest)


def test_reference_time_is_exact_rational_and_rejects_bad_denominator():
    assert dataset._reference_time({"numerator": 1, "denominator": 3}) == Fraction(2, 6)
    for value in [0, -1, True, 1.0]:
        with pytest.raises(ValueError):
            dataset._reference_time({"numerator": 1, "denominator": value})


class _MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def read_bytes_with_etag(self, uri):
        return (self.objects[uri], "etag") if uri in self.objects else None

    def put_bytes_conditional(self, payload, uri, *, if_none_match, content_type):
        assert if_none_match and content_type == "application/json"
        if uri in self.objects:
            raise ValueError("already committed")
        self.objects[uri] = payload
        self.writes.append(uri)

    def upload_file(self, source, uri):
        self.objects[uri] = Path(source).read_bytes()
        self.writes.append(uri)

    def download_file(self, uri, destination):
        Path(destination).write_bytes(self.objects[uri])


def test_s3_roundtrip_decodes_remote_bytes_and_commits_manifest_last(
    captured, tmp_path
):
    root, manifest = captured
    storage = _MemoryStorage()
    prefix = "s3://test-bucket/run/capture"
    transport._publish(root, manifest, prefix, storage)
    assert storage.writes[-1] == prefix + "/manifest.json"
    assert all("/captures/" in uri for uri in storage.writes[:-1])
    report = transport.validate_s3(
        prefix + "/manifest.json",
        "s3://test-bucket/run/validation.json",
        tmp_path / "download",
        storage,
    )
    assert report["views"] == 12 and report["valid_depth_pixels"] == 240
    assert len(report["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="fresh run"):
        transport._publish(root, manifest, prefix, storage)


def test_failed_upload_never_commits_success(captured, monkeypatch):
    root, manifest = captured
    storage = _MemoryStorage()

    def fail(*_args):
        raise OSError("upload denied")

    monkeypatch.setattr(storage, "upload_file", fail)
    with pytest.raises(OSError, match="denied"):
        transport._publish(root, manifest, "s3://test-bucket/run/capture", storage)
    assert not storage.writes


def test_input_bundle_download_is_hashed_and_never_falls_back(rig_request, tmp_path):
    storage = _MemoryStorage()
    prefix = "s3://test-bucket/input"
    storage.objects[prefix + "/request.json"] = contract._json_bytes(rig_request)
    storage.objects[prefix + "/scene.usda"] = (
        tmp_path / "inputs/scene.usda"
    ).read_bytes()
    actual, scope = transport._acquire(
        prefix + "/request.json", tmp_path / "fetched", storage
    )
    assert actual == rig_request and scope == "supplied-usd"
    storage.objects[prefix + "/scene.usda"] = b"bad scene"
    with pytest.raises(ValueError, match="SHA256"):
        transport._acquire(prefix + "/request.json", tmp_path / "bad", storage)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket",
        "s3://bucket/../key",
        "s3://bucket/key?x",
        "s3://bucket/key#x",
        "https://host/key",
        "s3://user@bucket/key",
        "procedural://missing",
    ],
)
def test_invalid_remote_uris_fail_before_any_network(uri, tmp_path):
    with pytest.raises(ValueError):
        transport._acquire(uri, tmp_path / "fetch", _MemoryStorage())


def test_fixture_is_explicit_deterministic_and_not_overwritten(tmp_path):
    first = write_fixture(tmp_path / "first")
    assert first == write_fixture(tmp_path / "second")
    with pytest.raises(ValueError, match="empty"):
        write_fixture(tmp_path / "first")


def test_no_pointcloud_is_supported(rig_request, tmp_path):
    rig_request = copy.deepcopy(rig_request)
    rig_request["pointcloud"] = False
    camera, sample = rig_request["cameras"][0], rig_request["trajectory"][0]
    view = dataset._write_view(
        tmp_path, camera, sample, 0, _snapshot(camera, sample, 0), False
    )
    assert set(view["artifacts"]) == {"rgb", "depth", "mask"}


def test_fusion_preserves_all_views_and_original_pixel_indices(captured):
    root, manifest = captured
    frame = manifest["frames"][0]
    fused = frame["fused_cloud"]
    assert fused["camera_ids"] == ["front", "left", "rear", "right"]
    with np.load(root / fused["path"], allow_pickle=False) as cloud:
        assert cloud["xyz_world_m"].shape == (80, 3)
        np.testing.assert_array_equal(cloud["camera_index"], np.repeat(range(4), 20))
        np.testing.assert_array_equal(cloud["pixel_index"], np.tile(range(4, 24), 4))
        for index, view in enumerate(frame["views"]):
            with np.load(
                root / view["artifacts"]["points"], allow_pickle=False
            ) as points:
                np.testing.assert_array_equal(
                    cloud["xyz_world_m"][index * 20 : (index + 1) * 20],
                    points["xyz_world_m"],
                )


@pytest.mark.parametrize(
    "mutation", ["position", "color", "pixel", "camera", "extra", "missing", "nan"]
)
def test_fusion_decoded_provenance_refuses_rehashed_corruption(captured, mutation):
    root, manifest = captured
    name = manifest["frames"][0]["fused_cloud"]["path"]
    with np.load(root / name, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if mutation in {"extra", "missing"}:
        arrays = {
            key: np.concatenate([value, value[:1]])
            if mutation == "extra"
            else value[:-1]
            for key, value in arrays.items()
        }
    elif mutation == "nan":
        arrays["xyz_world_m"][0, 0] = np.nan
    else:
        key = {
            "position": "xyz_world_m",
            "color": "rgb",
            "pixel": "pixel_index",
            "camera": "camera_index",
        }[mutation]
        arrays[key][0] += 1
    np.savez(root / name, **arrays)
    manifest["files"][name] = contract._sha256(root / name)
    with pytest.raises(ValueError, match="fused cloud"):
        dataset.validate_dataset(root, manifest)


def test_fusion_binds_camera_order_and_artifact_hash(captured):
    root, manifest = captured
    fused = manifest["frames"][0]["fused_cloud"]
    fused["camera_ids"].reverse()
    with pytest.raises(ValueError, match="source camera order"):
        dataset.validate_dataset(root, manifest)
    fused["camera_ids"].reverse()
    (root / fused["path"]).write_bytes(b"corrupt fused points")
    with pytest.raises(ValueError, match="SHA256"):
        dataset.validate_dataset(root, manifest)


def test_reference_route_is_complete_and_distinct_from_qualification_fixture(tmp_path):
    root = tmp_path / "collected"
    write_fixture(root)
    (root / "request.json").unlink()
    scene = root / "scene.usda"
    request = reference._write_reference_request(
        root, scene, {reference._SOURCE_URL: str(scene)}
    )
    assert len(request["trajectory"]) == 265
    assert len(request["cameras"]) == 4
    assert {(camera["width"], camera["height"]) for camera in request["cameras"]} == {
        (1280, 720)
    }
    positions = np.asarray([sample["T_world_rig"] for sample in request["trajectory"]])[
        :, :3, 3
    ]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    np.testing.assert_allclose(distances, 0.25)
    assert distances.sum() == 66
    assert request["trajectory"][-1]["timestamp_ns"] == 66_000_000_000
    assert "reference.json" in request["files"]
    assert (
        contract._read_json(root / "reference.json")["route_collision_free_verified"]
        is False
    )


def test_reference_input_roundtrip_is_immutable_and_keeps_dependency_layout(tmp_path):
    root = tmp_path / "collected"
    write_fixture(root)
    (root / "request.json").unlink()
    request = reference._write_reference_request(root, root / "scene.usda", {})
    storage = _MemoryStorage()
    destination = "s3://test-bucket/reference"
    reference._publish_reference(root, request, destination, storage)
    assert storage.writes[-1] == destination + "/request.json"
    downloaded, scope = transport._acquire(
        destination + "/request.json", tmp_path / "download", storage
    )
    assert scope == "supplied-usd"
    assert downloaded["scene"].startswith("inputs/")
    assert downloaded["cameras"] == request["cameras"]
    with pytest.raises(ValueError, match="already exists"):
        reference._publish_reference(root, request, destination, storage)


def test_reference_upload_failure_does_not_commit_request(tmp_path, monkeypatch):
    root = tmp_path / "collected"
    request = write_fixture(root)
    storage = _MemoryStorage()

    def fail(*_args):
        raise OSError("reference upload failed")

    monkeypatch.setattr(storage, "upload_file", fail)
    with pytest.raises(OSError, match="upload failed"):
        reference._publish_reference(
            root, request, "s3://test-bucket/reference", storage
        )
    assert not storage.writes

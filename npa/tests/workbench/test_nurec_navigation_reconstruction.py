"""Exercise real RGB-D fusion, held-out depth checks, and portable collider export."""

import json

import numpy as np
from PIL import Image
import pytest

from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_capture import read_capture
from npa.workbench.nurec.navigation_reconstruction import reconstruct_capture
from npa.workbench.nurec.navigation_scene import prepare_scene


@pytest.fixture
def capture_bundle(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    frames = []
    for index in range(8):
        color = np.full((48, 64, 3), [80 + index, 120, 160], dtype=np.uint8)
        depth = np.full((48, 64), 10000, dtype=np.uint16)
        Image.fromarray(color).save(root / f"rgb_{index}.png")
        Image.fromarray(depth).save(root / f"depth_{index}.png")
        pose = np.eye(4)
        pose[0, 3] = index * 0.005
        frame = {
            "id": f"frame_{index}",
            "split": "validation" if index % 4 == 0 else "integration",
            "timestamp_s": index / 30,
            "camera_to_world": pose.tolist(),
        }
        for kind in ("rgb", "depth"):
            frame[kind] = f"{kind}_{index}.png"
            frame[f"{kind}_sha256"] = sha256(root / frame[kind])
        frames.append(frame)
    capture = {
        "schema": "npa.navigation.rgbd_capture.v1",
        "world": {"meters_per_unit": 1, "up_axis": "Z"},
        "camera_convention": "optical_x_right_y_down_z_forward",
        "intrinsics": {
            "width": 64,
            "height": 48,
            "fx": 60,
            "fy": 60,
            "cx": 31.5,
            "cy": 23.5,
            "distortion": [0, 0, 0, 0, 0],
        },
        "depth_units_per_meter": 5000,
        "depth_max_m": 4,
        "voxel_size_m": 0.025,
        "sdf_trunc_m": 0.1,
        "validation": {
            "pixel_stride": 8,
            "min_coverage": 0.9,
            "min_inlier_fraction": 0.9,
            "max_mean_error_m": 0.01,
            "distance_tolerance_m": 0.025,
        },
        "frames": frames,
    }
    (root / "capture.json").write_text(json.dumps(capture))
    return root


def _update(root, change):
    path = root / "capture.json"
    capture = json.loads(path.read_text())
    change(capture)
    path.write_text(json.dumps(capture))


def test_actual_tsdf_to_portable_collision_scene(capture_bundle, tmp_path):
    pytest.importorskip("open3d")
    pytest.importorskip("pxr")
    reconstruction = tmp_path / "reconstructed"
    report = reconstruct_capture(str(capture_bundle), str(reconstruction))
    assert report["engine"] == "open3d.pipelines.integration.ScalableTSDFVolume"
    assert report["integration_frames"] == 6
    assert report["validation_frames"] == 2
    assert report["triangles"] > 100
    assert report["depth_validation"]["mean_absolute_error_m"] < 0.01
    assert report["depth_validation"]["coverage"] >= 0.9
    assert len(report["ray_probes"]) == 2
    assembled = tmp_path / "assembled"
    scene = prepare_scene(str(reconstruction), str(assembled))
    assert scene["triangle_count"] == report["triangles"]
    assert scene["reconstruction_report_sha256"] == sha256(
        assembled / "reconstruction.json"
    )
    assert scene["capture_manifest_sha256"] == sha256(capture_bundle / "capture.json")
    assert scene["physics_validated"] is False
    with pytest.raises((ValueError, FileExistsError)):
        reconstruct_capture(str(capture_bundle), str(reconstruction))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c["intrinsics"].update(distortion=[0.1, 0, 0, 0, 0]), "undistorted"),
        (lambda c: c.update(depth_units_per_meter=0), "positive"),
        (
            lambda c: c.update(world={"meters_per_unit": 0.01, "up_axis": "Z"}),
            "metric Z-up",
        ),
        (lambda c: c["frames"][0].update(rgb_sha256="0" * 64), "hash differs"),
        (lambda c: c["frames"][0].update(rgb="../outside.png"), "bundle"),
        (
            lambda c: c["frames"][0].update(camera_to_world=(np.eye(4) * 2).tolist()),
            "row-vector",
        ),
        (lambda c: c["frames"][0].update(id=c["frames"][1]["id"]), "unique"),
        (lambda c: c["validation"].update(min_coverage=1.5), "fractions"),
        (lambda c: c["validation"].update(pixel_stride=100), "pixel_stride"),
    ],
)
def test_capture_rejects_ambiguous_or_changed_inputs(capture_bundle, change, message):
    _update(capture_bundle, change)
    with pytest.raises(ValueError, match=message):
        read_capture(capture_bundle)


def test_capture_rejects_duplicate_pair_across_split(capture_bundle):
    def duplicate(capture):
        original, replacement = capture["frames"][:2]
        for key in ("rgb", "depth", "rgb_sha256", "depth_sha256"):
            original[key] = replacement[key]

    _update(capture_bundle, duplicate)
    with pytest.raises(ValueError, match="held-out split"):
        read_capture(capture_bundle)


def test_actual_heldout_geometry_failure(capture_bundle, tmp_path):
    pytest.importorskip("open3d")

    def move_validation(capture):
        for frame in capture["frames"]:
            if frame["split"] == "validation":
                frame["camera_to_world"][0][3] = 20

    _update(capture_bundle, move_validation)
    with pytest.raises(ValueError, match="no observations or surface hits"):
        reconstruct_capture(str(capture_bundle), str(tmp_path / "bad"))
    assert not (tmp_path / "bad").exists()

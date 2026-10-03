"""Fetch and calibrate the complete, hash-pinned public TUM office RGB-D sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

import numpy as np
import requests
from scipy.spatial.transform import Rotation, Slerp

from npa.workbench.nurec.navigation_assets import contained_file, sha256
from npa.workbench.nurec.navigation_capture import read_capture
from npa.workflows.navigation.artifacts import publish, write_json

SAMPLE_NAME = "rgbd_dataset_freiburg3_long_office_household"
ARCHIVE_URL = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg3/{SAMPLE_NAME}.tgz"
ARCHIVE_SHA256 = "c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c"
ATTRIBUTION = (
    "TUM RGB-D benchmark, fr3/long_office_household; J. Sturm, N. Engelhard, "
    "F. Endres, W. Burgard, D. Cremers; A Benchmark for the Evaluation of RGB-D "
    "SLAM Systems, IROS 2012. CC BY 4.0. "
    "https://cvg.cit.tum.de/data/datasets/rgbd-dataset. "
    "Adaptation: associated RGB/depth and interpolated ground-truth poses; "
    "every fifth original RGB index withheld from fusion."
)


def _download(destination):
    with requests.get(ARCHIVE_URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with destination.open("xb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)


def _extract(archive, destination):
    if archive.is_symlink() or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("TUM sample archive differs from the pinned SHA-256")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as handle:
        names = set()
        for member in handle:
            path = PurePosixPath(member.name)
            if path.parts[:1] != (SAMPLE_NAME,) or ".." in path.parts:
                raise ValueError("TUM archive contains an escaping member")
            if member.name in names or not (member.isdir() or member.isfile()):
                raise ValueError("TUM archive requires unique regular files")
            names.add(member.name)
            target = destination.joinpath(*path.parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def _rows(root, name):
    return [
        line.split()
        for line in contained_file(root, name).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _pose(timestamp, poses, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation(timestamp).as_matrix()
    matrix[:3, 3] = [
        np.interp(timestamp, poses[:, 0], poses[:, axis]) for axis in (1, 2, 3)
    ]
    return matrix.tolist()


def _association(timestamp, poses, depth_times):
    nearest = int(np.argmin(abs(depth_times - timestamp)))
    difference = float(abs(depth_times[nearest] - timestamp))
    following = int(np.searchsorted(poses[:, 0], timestamp))
    valid = (
        0 < following < len(poses)
        and poses[following, 0] - poses[following - 1, 0] <= 0.05
        and difference <= 0.02
    )
    return nearest, difference, valid


def _frames(root, rgb, depth, poses):
    depth_times = np.array([float(row[0]) for row in depth])
    rotation = Slerp(poses[:, 0], Rotation.from_quat(poses[:, 4:8]))
    frames, excluded = [], []
    for index, (timestamp, image) in enumerate(rgb):
        timestamp = float(timestamp)
        nearest, difference, valid = _association(timestamp, poses, depth_times)
        if not valid:
            excluded.append({"source_index": index, "depth_offset_s": difference})
            continue
        depth_path = depth[nearest][1]
        frames.append(
            {
                "id": f"frame_{index:06d}",
                "split": "validation" if index % 5 == 0 else "integration",
                "timestamp_s": timestamp,
                "depth_timestamp_s": float(depth_times[nearest]),
                "camera_to_world": _pose(timestamp, poses, rotation),
                "rgb": image,
                "depth": depth_path,
                "rgb_sha256": sha256(contained_file(root, image)),
                "depth_sha256": sha256(contained_file(root, depth_path)),
            }
        )
    return frames, excluded


def _capture(root):
    rgb, depth = _rows(root, "rgb.txt"), _rows(root, "depth.txt")
    poses = np.array(_rows(root, "groundtruth.txt"), dtype=float)
    frames, excluded = _frames(root, rgb, depth, poses)
    capture = _calibration()
    capture.update(
        frames=frames,
        source={
            "dataset": "TUM RGB-D benchmark, fr3/long_office_household",
            "license": "CC-BY-4.0",
            "attribution": ATTRIBUTION,
            "url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "source_rgb_frames": len(rgb),
            "source_depth_frames": len(depth),
            "groundtruth_poses": len(poses),
            "excluded_frames": excluded,
            "pose_interpolation": "linear translation and quaternion SLERP; max bracket 50 ms",
            "rgb_depth_association": "nearest registered depth; max offset 20 ms",
            "split": "every fifth original RGB index held out",
        },
    )
    counts = {
        name: sum(row["split"] == name for row in frames)
        for name in ("integration", "validation")
    }
    if counts != {"integration": 1984, "validation": 501} or len(excluded) != 100:
        raise ValueError("TUM full-sequence association inventory changed")
    return capture


def _calibration():
    return {
        "schema": "npa.navigation.rgbd_capture.v1",
        "world": {"meters_per_unit": 1, "up_axis": "Z"},
        "camera_convention": "optical_x_right_y_down_z_forward",
        "intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 535.4,
            "fy": 539.2,
            "cx": 320.1,
            "cy": 247.6,
            "distortion": [0, 0, 0, 0, 0],
        },
        "depth_units_per_meter": 5000,
        "depth_max_m": 5.0,
        "voxel_size_m": 0.02,
        "sdf_trunc_m": 0.08,
        "validation": {
            "pixel_stride": 40,
            "min_coverage": 0.85,
            "min_inlier_fraction": 0.8,
            "max_mean_error_m": 0.08,
            "distance_tolerance_m": 0.1,
        },
    }


def prepare_capture(archive: Path, output: Path) -> dict:
    """Convert the pinned public archive into a complete calibrated capture.

    Args:
        archive: Cached archive whose bytes must match ARCHIVE_SHA256.
        output: Fresh local output directory.
    Returns:
        Validated capture including all 2,485 associated frames and attribution.
    Raises:
        ValueError: Archive containment, hash, association or calibration fails.
        OSError: Archive or output cannot be accessed.
    """
    _extract(Path(archive), Path(output))
    capture = _capture(Path(output))
    write_json(Path(output) / "capture.json", capture)
    (Path(output) / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n")
    return read_capture(Path(output))


def prepare_sample(output_path: str, archive_path: str | None = None) -> dict:
    """Fetch the full public sample and publish a verified, immutable input bundle.

    Args:
        output_path: Fresh local directory or run-scoped S3 publication prefix.
        archive_path: Optional cached local archive, verified against the same pin.
    Returns:
        Public sample identity, split counts and capture manifest hash.
    Raises:
        ValueError: Sample integrity, calibration or publication fails.
        requests.RequestException: The official public download fails.
        OSError: Temporary staging or publication fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-tum-sample-") as temporary:
        work = Path(temporary)
        archive = Path(archive_path) if archive_path else work / "sample.tgz"
        if archive_path is None:
            _download(archive)
        capture = prepare_capture(archive, work / "capture")
        report = {
            "schema": "npa.navigation.public_sample.v1",
            "archive_sha256": ARCHIVE_SHA256,
            "capture_sha256": sha256(work / "capture/capture.json"),
            "frames": len(capture["frames"]),
            "integration_frames": 1984,
            "validation_frames": 501,
            "excluded_frames": 100,
            "attribution": ATTRIBUTION,
        }
        write_json(work / "capture/sample.json", report)
        publish(work / "capture", output_path)
        return report


def main(argv=None):
    """Publish the public sample for the declarative scan-to-policy workflow.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero after verified publication.
    Raises:
        ValueError: Sample preparation fails.
        OSError: Input/output cannot be accessed.
        requests.RequestException: Public download fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--archive-path")
    print(json.dumps(prepare_sample(**vars(parser.parse_args(argv))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

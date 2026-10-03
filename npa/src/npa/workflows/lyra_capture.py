"""Prepare a calibrated video excerpt for Lyra and preserve measured validation data."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
from PIL import Image

from npa.workbench.nurec.navigation_capture import read_capture
from npa.workflows.lerobot_transfer_data import publish, write_json


def _video(root, frames, destination):
    first = np.asarray(Image.open(root / frames[0]["rgb"]).convert("RGB"))
    height, width = first.shape[:2]
    with av.open(str(destination), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "17"}
        for record in frames:
            pixels = np.asarray(Image.open(root / record["rgb"]).convert("RGB"))
            if pixels.shape != first.shape:
                raise ValueError(
                    "Capture dimensions changed within the selected excerpt"
                )
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _prepare(args, output):
    if args.input_path.is_file():
        _prepare_video(args.input_path, output)
        if args.calibration_path:
            _attach_calibration(args.calibration_path, output)
        return
    if args.calibration_path:
        raise ValueError("A capture directory already carries its calibration")
    capture = read_capture(args.input_path)
    frames = capture["frames"][args.start_frame : args.start_frame + args.frame_count]
    if len(frames) != args.frame_count or args.frame_count < 2:
        raise ValueError("Requested video excerpt is outside the calibrated capture")
    _video(args.input_path, frames, output / "capture.mp4")
    reference = deepcopy(capture)
    reference["frames"] = frames
    reference["excerpt"] = {
        "start_frame": args.start_frame,
        "frame_count": args.frame_count,
    }
    write_json(output / "capture.json", reference)
    indices = np.linspace(0, len(frames) - 1, min(128, len(frames)), dtype=int)
    depth = np.stack(
        [np.asarray(Image.open(args.input_path / frames[i]["depth"])) for i in indices]
    )
    np.savez_compressed(output / "measured-depth.npz", depth=depth, indices=indices)
    attribution = args.input_path / "ATTRIBUTION.txt"
    if attribution.is_file():
        (output / attribution.name).write_text(attribution.read_text())


def _prepare_video(source, output):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(stream))
        if count < 2 or not stream.average_rate:
            raise ValueError("Lyra requires a decodable video with at least two frames")
        metadata = {
            "schema": "npa.lyra-video.v1",
            "frames_count": count,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
            "metric_calibration": False,
        }
    shutil.copy2(source, output / "capture.mp4")
    write_json(output / "capture.json", metadata)
    (output / "ATTRIBUTION.txt").write_text(
        "Operator-supplied capture; source rights remain with its provider.\n"
    )


def _attach_calibration(path, output):
    video = json.loads((output / "capture.json").read_text())
    calibration = json.loads(path.read_text())
    if calibration.get("schema") != "npa.lyra-camera-calibration.v1":
        raise ValueError("Video calibration requires npa.lyra-camera-calibration.v1")
    if calibration.get("camera_convention") != "optical_x_right_y_down_z_forward":
        raise ValueError("Video calibration requires optical camera axes")
    if calibration.get("world") != {"meters_per_unit": 1, "up_axis": "Z"}:
        raise ValueError("Video calibration requires a metric Z-up world")
    frames = calibration.get("frames", [])
    if len(frames) != video["frames_count"]:
        raise ValueError("Calibration must contain one pose per decoded video frame")
    for frame in frames:
        pose = np.asarray(frame["camera_to_world"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Camera poses must be finite 4x4 matrices")
        rotation = pose[:3, :3]
        if (
            not np.array_equal(pose[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("Camera poses must be proper rigid transformations")
    calibration["video"] = video
    write_json(output / "capture.json", calibration)


def main():
    """Publish an explicit excerpt without using measured depth in Lyra inference.

    Args:
        None; reads the command line.
    Returns:
        None after publishing a checksummed input bundle.
    Raises:
        ValueError: Calibration, frames or selection is invalid.
        OSError: Capture files cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=320)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lyra-capture-") as temporary:
        output = Path(temporary)
        _prepare(args, output)
        publish(output, args.output_path)


if __name__ == "__main__":
    main()

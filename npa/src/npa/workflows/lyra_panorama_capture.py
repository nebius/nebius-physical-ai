"""Prepare stitched panoramas as a checksummed perspective video for Lyra 2."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import islice
import math
from pathlib import Path
import tempfile

import av
from PIL import Image

from npa.workflows.lerobot_transfer_data import file_sha256, publish, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--input-projection",
        choices=["equirectangular"],
        required=True,
        help="Declare a stitched, mono 2:1 panorama; raw fisheye is unsupported.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--h-fov", type=float, default=90)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Presentation rate for an image directory only.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--frame-count",
        type=int,
        help="Select an exact excerpt; default: all remaining frames.",
    )
    args = parser.parse_args()
    _validate_arguments(args)
    return args


def _validate_arguments(args):
    for size in (args.width, args.height):
        if size < 2 or size % 2 or size > 16382:
            raise ValueError("Output dimensions must be even and between 2 and 16382")
    if not math.isfinite(args.h_fov) or not 0 < args.h_fov < 180:
        raise ValueError("Perspective horizontal FOV must be finite and in (0, 180)")
    for angle, limit in ((args.yaw, 180), (args.pitch, 90)):
        if not math.isfinite(angle) or not -limit <= angle <= limit:
            raise ValueError("Yaw must be within +/-180 and pitch within +/-90 degrees")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Image presentation FPS must be finite and positive")
    if args.start_frame < 0 or (args.frame_count is not None and args.frame_count < 2):
        raise ValueError(
            "Start frame must be nonnegative; frame count must be at least 2"
        )
    if args.input_path.is_dir():
        return
    if args.input_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(
            "Export raw fisheye to stitched mono 2:1 MP4/MOV/MKV or JPG/PNG first"
        )


def _image_frames(root):
    paths = sorted(
        path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(
            "No exported JPG/PNG frames; raw fisheye needs calibrated stitching"
        )
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            frame = av.VideoFrame.from_image(image.convert("RGB"))
        yield frame, {"source_index": index, "source_sha256": file_sha256(path)}


def _video_frames(container):
    if not container.streams.video:
        raise ValueError("Input contains no video stream")
    for index, frame in enumerate(container.decode(video=0)):
        yield frame, {"source_index": index, "source_time_seconds": frame.time}


def _projection(frame, args):
    vertical_fov = math.degrees(
        2 * math.atan(args.height / args.width * math.tan(math.radians(args.h_fov) / 2))
    )
    filters = (
        f"input=equirect:output=flat:w={args.width * 2}:h={args.height * 2}"
        f":h_fov={args.h_fov}:v_fov={vertical_fov}:yaw={args.yaw}:pitch={args.pitch}"
        ":interp=lanczos"
    )
    graph = av.filter.Graph()
    source = graph.add_buffer(template=frame)
    projection = graph.add("v360", filters)
    # Supersampling followed by reduction limits aliasing from high-resolution input.
    reduction = graph.add("scale", f"w={args.width}:h={args.height}:flags=lanczos")
    sink = graph.add("buffersink")
    source.link_to(projection)
    projection.link_to(reduction)
    reduction.link_to(sink)
    graph.configure()
    return graph, vertical_fov


def _encode(args, frames, rate, destination):
    records, dimensions, graph = [], None, None
    stop = None if args.frame_count is None else args.start_frame + args.frame_count
    with av.open(str(destination), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = args.width, args.height
        stream.pix_fmt, stream.options = "yuv420p", {"crf": "17"}
        for frame, record in islice(frames, args.start_frame, stop):
            size = (frame.width, frame.height)
            if size[0] != 2 * size[1] or (dimensions and size != dimensions):
                raise ValueError(
                    "Stitched mono panoramas must have constant 2:1 dimensions"
                )
            dimensions = size
            frame = frame.reformat(format="rgb24")
            frame.pts, frame.time_base = len(records), 1 / rate
            if graph is None:
                graph, vertical_fov = _projection(frame, args)
            graph.push(frame)
            projected = graph.pull()
            projected.pts, projected.time_base = len(records), 1 / rate
            for packet in stream.encode(projected):
                container.mux(packet)
            records.append(record)
        for packet in stream.encode():
            container.mux(packet)
    if len(records) < 2 or (
        args.frame_count is not None and len(records) != args.frame_count
    ):
        raise ValueError(
            "Requested excerpt needs at least two available panorama frames"
        )
    return records, dimensions, vertical_fov


def _write_manifest(args, output, rate, prepared, source):
    records, dimensions, vertical_fov = prepared
    write_json(
        output / "capture.json",
        {
            "schema": "npa.lyra-video.v1",
            "frames_count": len(records),
            "width": args.width,
            "height": args.height,
            "fps": float(rate),
            "metric_calibration": False,
            "preprocessing": {
                "input_projection": args.input_projection,
                "output_projection": "perspective",
                "source_width": dimensions[0],
                "source_height": dimensions[1],
                "h_fov_degrees": args.h_fov,
                "v_fov_degrees": vertical_fov,
                "yaw_degrees": args.yaw,
                "pitch_degrees": args.pitch,
                "interpolation": "v360 Lanczos; 2x supersampling; Lanczos reduction",
                "pyav_version": av.__version__,
                "libavfilter_version": list(av.library_versions["libavfilter"]),
                "source": source,
                "frames": records,
            },
        },
    )
    (output / "ATTRIBUTION.txt").write_text(
        "Operator-supplied stitched panoramas; source rights remain with their provider.\n"
    )


def _prepare(args, output):
    if args.input_path.is_dir():
        rate = Fraction(str(args.fps))
        prepared = _encode(
            args, _image_frames(args.input_path), rate, output / "capture.mp4"
        )
        source = {
            "kind": "image-sequence",
            "ordering": "lexicographic filename",
            "timing": "presentation cadence only; capture timestamps unknown",
        }
    else:
        with av.open(str(args.input_path)) as container:
            if (
                not container.streams.video
                or not container.streams.video[0].average_rate
            ):
                raise ValueError(
                    "A decodable video with a known frame rate is required"
                )
            rate = container.streams.video[0].average_rate
            prepared = _encode(
                args, _video_frames(container), rate, output / "capture.mp4"
            )
        source = {
            "kind": "video",
            "sha256": file_sha256(args.input_path),
            "timing": "constant presentation cadence; original frame times retained",
        }
    _write_manifest(args, output, rate, prepared, source)


def main():
    """Publish one fixed perspective sequence from exported stitched panoramas.

    Args:
        None; reads the command line.
    Returns:
        None after publishing a checksummed local or S3 Lyra input bundle.
    Raises:
        ValueError: Projection, dimensions or requested excerpt is invalid.
        OSError: Input or output files cannot be accessed.
        av.FFmpegError: Decoding, projection or encoding fails.
    """
    args = _arguments()
    with tempfile.TemporaryDirectory(prefix="lyra-panorama-") as temporary:
        output = Path(temporary) / "prepared"
        output.mkdir(mode=0o700)
        _prepare(args, output)
        publish(output, args.output_path)


if __name__ == "__main__":
    main()

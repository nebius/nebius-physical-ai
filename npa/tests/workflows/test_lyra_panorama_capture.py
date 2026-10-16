"""Verify native panorama projection, temporal order and sealed Lyra handoff."""

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from PIL import Image
import pytest

from npa.workflows import lyra_panorama_capture as capture
from npa.workflows.lerobot_transfer_data import file_sha256, materialize


def _arguments(source, **changes):
    values = dict(
        input_path=source,
        input_projection="equirectangular",
        width=80,
        height=60,
        h_fov=90,
        yaw=0,
        pitch=0,
        fps=12,
        start_frame=0,
        frame_count=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _sphere_image():
    longitude = (np.arange(1024) + 0.5) / 1024 * 2 * np.pi - np.pi
    latitude = np.pi / 2 - (np.arange(512) + 0.5) / 512 * np.pi
    pixels = np.empty((512, 1024, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.rint(127.5 * (1 + np.sin(longitude)))
    pixels[:, :, 1] = np.rint(127.5 * (1 + np.sin(latitude[:, None])))
    pixels[:, :, 2] = np.rint(127.5 * (1 + np.cos(longitude)))
    return Image.fromarray(pixels)


@pytest.mark.parametrize(
    "yaw,pitch,expected",
    [
        (0, 0, [128, 128, 255]),
        (90, 0, [255, 128, 128]),
        (-90, 0, [0, 128, 128]),
        (180, 0, [128, 128, 0]),
        (0, 30, [128, 191, 255]),
    ],
)
def test_native_projection_preserves_orientation_and_wraps_seam(
    tmp_path, yaw, pitch, expected
):
    frame = av.VideoFrame.from_image(_sphere_image())
    frame.pts, frame.time_base = 0, Fraction(1, 12)
    graph, vertical_fov = capture._projection(
        frame, _arguments(tmp_path, yaw=yaw, pitch=pitch)
    )
    graph.push(frame)
    pixels = graph.pull().to_ndarray(format="rgb24")
    assert pixels.shape == (60, 80, 3)
    np.testing.assert_allclose(pixels[30, 40], expected, atol=5)
    assert vertical_fov == pytest.approx(73.73979529168804)
    if yaw == 180:
        assert np.max(pixels[25:35, 35:45, 2]) < 5


def test_fixed_perspective_has_correct_fov_and_no_stretched_panorama(tmp_path):
    frame = av.VideoFrame.from_image(_sphere_image())
    frame.pts, frame.time_base = 0, Fraction(1, 12)
    graph, _ = capture._projection(frame, _arguments(tmp_path))
    graph.push(frame)
    pixels = graph.pull().to_ndarray(format="rgb24")
    # A 90-degree pinhole view reaches +/-45 longitude, rather than panorama edges.
    np.testing.assert_allclose(pixels[30, 0], [38, 128, 218], atol=5)
    np.testing.assert_allclose(pixels[30, -1], [217, 128, 218], atol=5)


def _image_sequence(root, dimensions=(1024, 512)):
    root.mkdir()
    for index, color in enumerate(("red", "green", "blue", "white")):
        image = Image.new("RGB", dimensions, color)
        metadata = Image.Exif()
        metadata[315] = "private-fixture-attribution"
        image.save(root / f"frame-{index:04d}.jpg", exif=metadata)


def _decode(path):
    with av.open(str(path)) as container:
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
        return frames, container.streams.video[0].average_rate, dict(container.metadata)


def test_images_publish_a_verified_bundle_without_paths_or_exif(tmp_path, monkeypatch):
    source, output = tmp_path / "private-input", tmp_path / "lyra-input"
    _image_sequence(source)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lyra-panorama-capture",
            "--input-path",
            str(source),
            "--output-path",
            str(output),
            "--input-projection",
            "equirectangular",
            "--width",
            "80",
            "--height",
            "60",
            "--fps",
            "12",
            "--start-frame",
            "1",
            "--frame-count",
            "2",
        ],
    )
    capture.main()
    assert materialize(str(output), tmp_path / "unused") == output
    manifest_text = (output / "capture.json").read_text()
    manifest = json.loads(manifest_text)
    records = manifest["preprocessing"]["frames"]
    assert [record["source_index"] for record in records] == [1, 2]
    assert records[0]["source_sha256"] == file_sha256(source / "frame-0001.jpg")
    assert manifest["metric_calibration"] is False
    assert manifest["frames_count"] == 2
    frames, rate, metadata = _decode(output / "capture.mp4")
    assert rate == 12 and len(frames) == 2
    assert frames[0][30, 40, 1] > 100 and frames[1][30, 40, 2] > 240
    _assert_private_metadata_absent(source, manifest_text, metadata)


def _assert_private_metadata_absent(source, manifest_text, metadata):
    for private_value in (str(source), "frame-0001.jpg", "private-fixture-attribution"):
        assert private_value not in manifest_text
        assert private_value not in json.dumps(metadata)


def _video(path, rate, color_trc=2):
    with av.open(str(path), "w") as container:
        container.metadata["title"] = "private-fixture-attribution"
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.codec_context.color_trc = color_trc
        for color in ("red", "green", "blue", "white"):
            frame = av.VideoFrame.from_image(Image.new("RGB", (1024, 512), color))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("rate", [Fraction(30), Fraction(24000, 1001)])
def test_perspective_video_strips_source_metadata_and_preserves_rgb_order(
    tmp_path, rate
):
    from npa.workflows import lyra_capture

    source = tmp_path / "private-source.mp4"
    _video(source, rate)
    output = tmp_path / "prepared"
    output.mkdir()
    lyra_capture._prepare_video(source, output)
    frames, actual_rate, metadata = _decode(output / "capture.mp4")
    assert actual_rate == rate and len(frames) == 4
    assert all(frame.shape == (512, 1024, 3) for frame in frames)
    expected_colors = ([255, 0, 0], [0, 128, 0], [0, 0, 255], [255, 255, 255])
    for frame, expected in zip(frames, expected_colors):
        np.testing.assert_allclose(frame[256, 512], expected, atol=8)
    manifest_text = (output / "capture.json").read_text()
    _assert_private_metadata_absent(source, manifest_text, metadata)
    with av.open(str(source)) as original:
        assert original.metadata["title"] == "private-fixture-attribution"
    with av.open(str(output / "capture.mp4")) as cleaned:
        assert len(cleaned.streams) == 1 and cleaned.streams[0].type == "video"
        assert "private-fixture-attribution" not in json.dumps(
            dict(cleaned.streams[0].metadata)
        )


@pytest.mark.parametrize("color_trc", [16, 18], ids=["pq", "hlg"])
@pytest.mark.parametrize("entrypoint", ["panorama", "video"])
def test_signaled_hdr_requires_sdr_export_before_rgb_conversion(
    tmp_path, color_trc, entrypoint
):
    from npa.workflows import lyra_capture

    source = tmp_path / "hdr.mp4"
    _video(source, Fraction(30), color_trc=color_trc)
    with av.open(str(source)) as original:
        assert original.streams.video[0].codec_context.color_trc == color_trc
    output = tmp_path / "prepared"
    output.mkdir()
    with pytest.raises(ValueError, match="tone-mapped SDR export"):
        if entrypoint == "panorama":
            capture._prepare(_arguments(source), output)
        else:
            lyra_capture._prepare_video(source, output)
    assert list(output.iterdir()) == []


def test_video_preserves_rate_order_and_source_times(tmp_path):
    source = tmp_path / "panorama.mp4"
    _video(source, Fraction(24000, 1001))
    output = tmp_path / "prepared"
    output.mkdir()
    args = _arguments(source, start_frame=1, frame_count=2)
    capture._validate_arguments(args)
    capture._prepare(args, output)
    manifest = json.loads((output / "capture.json").read_text())
    records = manifest["preprocessing"]["frames"]
    assert [record["source_index"] for record in records] == [1, 2]
    assert [record["source_time_seconds"] for record in records] == pytest.approx(
        [1001 / 24000, 2002 / 24000]
    )
    assert manifest["preprocessing"]["source"]["sha256"] == file_sha256(source)
    frames, rate, metadata = _decode(output / "capture.mp4")
    assert rate == Fraction(24000, 1001) and len(frames) == 2
    assert frames[0][30, 40, 1] > 100 and frames[1][30, 40, 2] > 240
    assert "private-fixture-attribution" not in json.dumps(metadata)


def test_high_resolution_images_are_reduced_before_inference(tmp_path):
    source = tmp_path / "panoramas"
    source.mkdir()
    for index in range(2):
        Image.new("RGB", (8192, 4096), "red").save(source / f"frame-{index:04d}.png")
    output = tmp_path / "prepared"
    output.mkdir()
    capture._prepare(_arguments(source, width=640, height=480), output)
    manifest = json.loads((output / "capture.json").read_text())
    assert manifest["preprocessing"]["source_width"] == 8192
    frames, _, _ = _decode(output / "capture.mp4")
    assert all(frame.shape == (480, 640, 3) for frame in frames)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 81},
        {"height": 0},
        {"width": 16384},
        {"h_fov": 180},
        {"h_fov": float("nan")},
        {"h_fov": 0},
        {"pitch": 91},
        {"yaw": 181},
        {"yaw": float("inf")},
        {"fps": 0},
        {"fps": float("nan")},
        {"start_frame": -1},
        {"frame_count": 1},
    ],
)
def test_invalid_projection_options_fail_before_encoding(tmp_path, changes):
    with pytest.raises(ValueError):
        capture._validate_arguments(_arguments(tmp_path, **changes))


@pytest.mark.parametrize("suffix", [".insv", ".insp", ".dng"])
def test_raw_fisheye_requires_calibrated_export(tmp_path, suffix):
    with pytest.raises(ValueError, match="Export raw fisheye"):
        capture._validate_arguments(_arguments(tmp_path / f"raw{suffix}"))


@pytest.mark.parametrize(
    "dimensions,selection",
    [
        ((640, 480), {}),
        ((1024, 512), {"start_frame": 3}),
        ((1024, 512), {"frame_count": 5}),
    ],
)
def test_bad_panorama_or_unavailable_excerpt_is_not_published(
    tmp_path, monkeypatch, dimensions, selection
):
    source = tmp_path / "panoramas"
    _image_sequence(source, dimensions)
    args = _arguments(source, **selection)
    args.output_path = str(tmp_path / "not-published")
    monkeypatch.setattr(capture, "_arguments", lambda: args)
    with pytest.raises(ValueError):
        capture.main()
    assert not (tmp_path / "not-published").exists()


def test_changed_dimensions_are_rejected(tmp_path):
    source = tmp_path / "panoramas"
    _image_sequence(source)
    Image.new("RGB", (512, 256)).save(source / "frame-0002.jpg")
    with pytest.raises(ValueError, match="constant 2:1"):
        capture._encode(
            _arguments(source),
            capture._image_frames(source),
            Fraction(12),
            tmp_path / "capture.mp4",
        )


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    source, output = tmp_path / "panoramas", tmp_path / "lyra-input"
    _image_sequence(source)
    output.mkdir()
    (output / "keep.txt").write_text("existing evidence")
    args = _arguments(source)
    args.output_path = str(output)
    monkeypatch.setattr(capture, "_arguments", lambda: args)
    with pytest.raises(FileExistsError):
        capture.main()
    assert (output / "keep.txt").read_text() == "existing evidence"


class _ReadbackStorage:
    def __init__(self, failure):
        self.failure = failure
        self.private_parent = False

    def upload_directory(self, source, destination):
        if self.failure == "upload":
            raise OSError("synthetic upload failure")
        self.files = {
            path.name: path.read_bytes()
            for path in Path(source).iterdir()
            if path.is_file()
        }

    def download_directory(self, source, destination):
        root = Path(destination)
        self.private_parent = root.parent.stat().st_mode & 0o777 == 0o700
        root.mkdir()
        for name, content in self.files.items():
            (root / name).write_bytes(content)
        if self.failure == "download":
            raise OSError("synthetic download failure")
        if self.failure == "checksum":
            (root / "capture.mp4").write_bytes(b"synthetic mismatch")


@pytest.mark.parametrize("failure", [None, "upload", "download", "checksum"])
@pytest.mark.parametrize("entrypoint", ["panorama", "video"])
def test_s3_readback_stays_private_and_is_cleaned_on_every_exit(
    tmp_path, monkeypatch, failure, entrypoint
):
    from contextlib import nullcontext
    import tempfile

    from npa.workflows import lerobot_transfer_data

    source = tmp_path / "panoramas"
    _image_sequence(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    temporary_directory = tempfile.TemporaryDirectory
    monkeypatch.setattr(
        capture.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: temporary_directory(dir=scratch, **kwargs),
    )
    args = _arguments(source)
    args.output_path = "s3://synthetic-bucket/cleanup-regression/"
    module = capture
    if entrypoint == "video":
        from npa.workflows import lyra_capture

        prepared = tmp_path / "source-video"
        prepared.mkdir()
        capture._prepare(args, prepared)
        module = lyra_capture
        monkeypatch.setattr(
            "sys.argv",
            [
                "lyra_capture",
                "--input-path",
                str(prepared / "capture.mp4"),
                "--output-path",
                args.output_path,
            ],
        )
    else:
        monkeypatch.setattr(capture, "_arguments", lambda: args)
    storage = _ReadbackStorage(failure)
    monkeypatch.setattr(
        lerobot_transfer_data.StorageClient, "from_environment", lambda: storage
    )
    expected = ValueError if failure == "checksum" else OSError
    with pytest.raises(expected) if failure else nullcontext():
        module.main()
    assert list(scratch.iterdir()) == []
    if failure != "upload":
        assert storage.private_parent

"""NCore conversion contracts; infrastructure and converter execution are isolated."""

from __future__ import annotations

import json
import hashlib
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

from npa.workbench.nurec import colmap


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "a/../../escape", "a\\escape", "https://host/data"],
)
def test_archive_rejects_unsafe_members_before_extracting(tmp_path, name):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("valid", b"safe")
        bundle.writestr(name, b"unsafe")
    destination = tmp_path / "expanded"
    with pytest.raises(colmap.NcoreConversionError, match="archive"):
        colmap.extract_colmap_zip(archive, destination)
    assert not (destination / "valid").exists()


def test_archive_rejects_links_and_duplicates(tmp_path):
    for members in ("link", "duplicate"):
        archive = tmp_path / f"{members}.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            if members == "link":
                entry = zipfile.ZipInfo("link")
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, "../outside")
            else:
                bundle.writestr("same", b"one")
                with pytest.warns(UserWarning):
                    bundle.writestr("same", b"two")
        with pytest.raises(colmap.NcoreConversionError, match="archive"):
            colmap.extract_colmap_zip(archive, tmp_path / members)


def dataset(root: Path):
    (root / "sparse/0").mkdir(parents=True)
    (root / "images").mkdir()
    for name in ("cameras", "images", "points3D"):
        (root / "sparse/0" / f"{name}.bin").write_bytes(b"unit fixture")
    (root / "images/frame.jpg").write_bytes(b"unit fixture")
    return root


def test_discovery_rejects_ambiguous_datasets_and_honors_layout(tmp_path):
    a = dataset(tmp_path / "a")
    assert colmap.find_colmap_root(tmp_path, ".", "sparse/0", "images") == a
    dataset(tmp_path / "b")
    with pytest.raises(colmap.NcoreConversionError, match="multiple"):
        colmap.find_colmap_root(tmp_path, ".", "sparse/0", "images")
    assert colmap.find_colmap_root(tmp_path, "a", "sparse/0", "images") == a


@pytest.mark.parametrize(
    "reference",
    [
        "../outside.zarr.itar",
        "/outside.zarr.itar",
        "s3://other/store",
        "missing.zarr.itar",
    ],
)
def test_sequence_references_fail_closed(tmp_path, reference):
    meta = tmp_path / "sequence.json"
    meta.write_text(
        json.dumps({"version": "v4", "component_stores": [{"path": reference}]})
    )
    with pytest.raises(colmap.NcoreConversionError, match="component"):
        colmap.sequence_members(meta)


def test_sequence_rejects_symlink_store(tmp_path):
    (tmp_path / "real.zarr.itar").write_bytes(b"test")
    (tmp_path / "alias.zarr.itar").symlink_to(tmp_path / "real.zarr.itar")
    meta = tmp_path / "sequence.json"
    meta.write_text(
        json.dumps({"version": "v4", "component_stores": [{"path": "alias.zarr.itar"}]})
    )
    with pytest.raises(colmap.NcoreConversionError, match="component"):
        colmap.sequence_members(meta)


class Storage:
    def __init__(self, source):
        self.source = source
        self.uploads = {}
        self._s3 = self

    @property
    def s3(self):
        return self

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        base = f"s3://{Bucket}/"
        yield {
            "Contents": [
                {"Key": uri[len(base) :]}
                for uri in self.uploads
                if uri.startswith(base + Prefix)
            ]
        }

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        from botocore.exceptions import ClientError

        assert IfNoneMatch == "*"
        uri = f"s3://{Bucket}/{Key}"
        if uri in self.uploads:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.uploads[uri] = Body
        return {"ETag": '"claim"'}

    def put_bytes_conditional(self, *args, **kwargs):
        from npa.clients.storage import StorageClient

        return StorageClient.put_bytes_conditional(self, *args, **kwargs)

    def download_directory(self, uri, destination):
        shutil.copytree(self.source, destination, dirs_exist_ok=True)

    def download_file(self, uri, destination):
        shutil.copyfile(self.source, destination)

    def upload_file(self, source, uri):
        self.uploads[uri] = Path(source).read_bytes()


def fake_conversion(monkeypatch, tmp_path):
    monkeypatch.setattr(
        colmap, "_runtime_fingerprints", lambda: {"unit_test": "0" * 64}
    )
    source = dataset(tmp_path / "source")
    storage = Storage(source)
    events = []
    monkeypatch.setattr(
        colmap,
        "inspect_colmap_source",
        lambda *a, **kw: {
            "counts": {"images": 2, "cameras": 1, "poses": 2, "points": 3}
        },
    )

    def run(argv, **kwargs):
        events.append("converter")
        assert argv[0] == "/opt/ncore/bin/colmap-convert"
        assert "colmap-v4" in argv
        assert "--no-include-downsampled-images" in argv
        assert not any("epoch" in arg or "limit" in arg for arg in argv)
        root = Path(argv[argv.index("--root-dir") + 1])
        out = Path(argv[argv.index("--output-dir") + 1]) / root.name
        out.mkdir(parents=True)
        (out / "data.zarr.itar").write_bytes(b"not real ncore: unit boundary only")
        (out / f"{root.name}.json").write_text(
            json.dumps(
                {
                    "version": "v4",
                    "sequence_id": root.name,
                    "component_stores": [{"path": "data.zarr.itar"}],
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0)

    def validate(meta, source, **kw):
        events.append("validated")
        assert colmap.sequence_members(meta)
        return {"images": 2, "cameras": 1, "poses": 2, "points": 3}

    monkeypatch.setattr(colmap.subprocess, "run", run)
    monkeypatch.setattr(colmap, "validate_ncore_sequence", validate)
    return storage, events


@pytest.mark.parametrize("as_zip", [False, True])
def test_exact_self_contained_publication(monkeypatch, tmp_path, as_zip):
    storage, events = fake_conversion(monkeypatch, tmp_path)
    input_uri = "s3://test-bucket/input/"
    if as_zip:
        archive = tmp_path / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for file in storage.source.rglob("*"):
                if file.is_file():
                    bundle.write(
                        file, "scene/" + file.relative_to(storage.source).as_posix()
                    )
        storage.source = archive
        input_uri = "s3://test-bucket/input.zip"
    request = colmap.ColmapConversionRequest(
        input_path=input_uri,
        output_path="s3://test-bucket/exact",
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "scratch",
        rig_mode="preserve",
        include_downsampled_images=False,
    )
    result = colmap.convert_colmap(request, storage_client=storage)
    assert events == ["converter", "validated"]
    assert result["status"] == "ok"
    assert result["output_path"] == "s3://test-bucket/exact/"
    assert "s3://test-bucket/exact/data.zarr.itar" in storage.uploads
    report = json.loads(storage.uploads["s3://test-bucket/exact/conversion.json"])
    assert report["counts"]["images"] == 2
    assert report["converter"]["revision"] == colmap.NCORE_REVISION
    assert all(len(member["sha256"]) == 64 for member in report["members"])
    assert str(tmp_path) not in json.dumps(report)
    assert list(storage.uploads)[-1] == result["ncore_meta_uri"]


def test_validation_failure_publishes_nothing(monkeypatch, tmp_path):
    storage, _ = fake_conversion(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise colmap.NcoreConversionError("image decode failed")

    monkeypatch.setattr(colmap, "validate_ncore_sequence", fail)
    with pytest.raises(colmap.NcoreConversionError, match="image decode"):
        colmap.convert_colmap(
            colmap.ColmapConversionRequest(
                input_path="s3://test-bucket/input/",
                output_path="s3://test-bucket/output/",
                cache_dir=tmp_path / "cache",
                scratch_dir=tmp_path / "scratch",
                rig_mode="preserve",
                include_downsampled_images=False,
            ),
            storage_client=storage,
        )
    assert not storage.uploads


def test_converter_failure_is_private(monkeypatch, tmp_path):
    storage, _ = fake_conversion(monkeypatch, tmp_path)

    def fail(argv, **kwargs):
        kwargs["stderr"].write(b"private vendor diagnostic")
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(colmap.subprocess, "run", fail)
    with pytest.raises(colmap.NcoreConversionError, match="exit code 7") as error:
        colmap.convert_colmap(
            colmap.ColmapConversionRequest(
                input_path="s3://test-bucket/input/",
                output_path="s3://test-bucket/output/",
                cache_dir=tmp_path / "cache",
                scratch_dir=tmp_path / "scratch",
                rig_mode="preserve",
            ),
            storage_client=storage,
        )
    assert "private vendor" not in str(error.value)
    assert not storage.uploads


@pytest.mark.parametrize("field", ["cache_dir", "scratch_dir"])
def test_unsafe_staging_fails_before_download_or_conversion(monkeypatch, tmp_path, field):
    storage, events = fake_conversion(monkeypatch, tmp_path)
    outside = tmp_path / "private-name"
    outside.mkdir(mode=0o700)
    (outside / "keep").write_bytes(b"unchanged")
    link = tmp_path / "unsafe"
    link.symlink_to(outside, target_is_directory=True)

    def unexpected_download(*args, **kwargs):
        pytest.fail("download must not start with unsafe staging")

    monkeypatch.setattr(storage, "download_directory", unexpected_download)
    options = {"cache_dir": tmp_path / "cache", "scratch_dir": tmp_path / "scratch"}
    options[field] = link
    request = colmap.ColmapConversionRequest(
        input_path="s3://test-bucket/input/",
        output_path="s3://test-bucket/output/",
        **options,
    )
    with pytest.raises(colmap.NcoreConversionError, match="during staging") as error:
        colmap.convert_colmap(request, storage_client=storage)
    assert "private-name" not in str(error.value)
    assert not events and not storage.uploads
    assert list(outside.iterdir()) == [outside / "keep"]
    assert (outside / "keep").read_bytes() == b"unchanged"
    if field == "scratch_dir":
        assert not list(options["cache_dir"].iterdir())


def test_default_staging_conversion_uses_fresh_private_children(monkeypatch, tmp_path):
    from npa.workbench import ncore_staging as staging

    storage, events = fake_conversion(monkeypatch, tmp_path)
    defaults = {
        staging.DEFAULT_COLMAP_CACHE_DIR: tmp_path / "home" / "cache",
        staging.DEFAULT_COLMAP_SCRATCH_DIR: tmp_path / "home" / "scratch",
    }
    original_expanduser = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda path: defaults[path] if path in defaults else original_expanduser(path),
    )
    original_download = storage.download_directory
    downloaded = []

    def download(uri, destination):
        parent = Path(destination).parent
        assert parent.parent == defaults[staging.DEFAULT_COLMAP_CACHE_DIR]
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
        downloaded.append(parent)
        original_download(uri, destination)

    monkeypatch.setattr(storage, "download_directory", download)
    for generation in ("first", "second"):
        result = colmap.convert_colmap(
            colmap.ColmapConversionRequest(
                input_path="s3://test-bucket/input/",
                output_path=f"s3://test-bucket/{generation}/",
                rig_mode="preserve",
                include_downsampled_images=False,
            ),
            storage_client=storage,
        )
        assert result["status"] == "ok"
        for parent in defaults.values():
            assert stat.S_IMODE(parent.stat().st_mode) == 0o700
            assert not list(parent.iterdir())
    assert events == ["converter", "validated", "converter", "validated"]
    assert len(set(downloaded)) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_path", "/local/input"),
        ("output_path", "file:///local/out"),
        ("dataset_root", "../escape"),
        ("colmap_dir", "/etc"),
        ("images_dir", "s3://foreign"),
        ("rig_mode", "false"),
    ],
)
def test_request_rejects_invalid_paths_and_modes(field, value):
    args = {
        "input_path": "s3://test-bucket/input/",
        "output_path": "s3://test-bucket/output/",
        field: value,
    }
    with pytest.raises(ValueError):
        colmap.ColmapConversionRequest(**args)


def ncore_fixture(tmp_path, *, corruption=""):
    """Write real synthetic V4 stores with upstream, never a fake reader."""
    import io
    import numpy as np
    from PIL import Image

    v4 = pytest.importorskip("ncore.data.v4")
    from ncore.impl.common.transformations import HalfClosedInterval
    from ncore.impl.data.types import (
        OpenCVPinholeCameraModelParameters,
        PointCloud,
        ShutterType,
    )
    from upath import UPath

    writer = v4.SequenceComponentGroupsWriter(
        output_dir_path=UPath(tmp_path),
        store_base_name="fixture",
        sequence_id="fixture",
        sequence_timestamp_interval_us=HalfClosedInterval(0, 2_000_000),
        generic_meta_data={},
        store_type="itar",
    )
    poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0).astype(np.float32)
    poses[1, 0, 3] = 1
    expected_poses = poses.copy()
    if corruption == "pose":
        poses[1, 0, 3] = np.nan
    pose_writer = writer.register_component_writer(v4.PosesComponent.Writer, "default")
    pose_writer.store_dynamic_pose(
        "camera1",
        "world",
        poses,
        np.array([0, 1_000_000], dtype=np.uint64),
        require_sequence_time_coverage=False,
    )
    model = OpenCVPinholeCameraModelParameters(
        resolution=np.array([4, 3], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        focal_length=np.array([3, 3], dtype=np.float32),
        principal_point=np.array([2, 1.5], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    if corruption == "calibration":
        model.radial_coeffs[0] = np.nan
    writer.register_component_writer(
        v4.IntrinsicsComponent.Writer, "default"
    ).store_camera_intrinsics("camera1", model)
    camera = writer.register_component_writer(
        v4.CameraSensorComponent.Writer, "camera1", group_name="camera1"
    )
    for index in range(2):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 3), (20, index * 100, 50)).save(buffer, format="PNG")
        camera.store_frame(
            image_binary_data=b"not an image"
            if corruption == "image" and index == 1
            else buffer.getvalue(),
            image_format="png",
            frame_timestamps_us=np.array([index * 1_000_000] * 2, dtype=np.uint64),
            generic_data={"mask": np.zeros((6, 8), dtype=np.uint8)}
            if corruption == "mask"
            else {},
            generic_meta_data={},
        )
    points = writer.register_component_writer(
        v4.PointCloudsComponent.Writer,
        "sfm_points",
        coordinate_unit=PointCloud.CoordinateUnit.UNITLESS,
        attribute_schemas={},
    )
    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    if corruption == "points":
        xyz[1, 0] = np.inf
    points.store_pc(
        xyz=xyz,
        reference_frame_id="world",
        reference_frame_timestamp_us=0,
        attributes={},
    )
    paths = writer.finalize()
    meta = tmp_path / "fixture.json"
    meta.write_text(
        json.dumps(
            v4.SequenceComponentGroupsReader(paths).get_sequence_meta().to_dict()
        )
    )
    source = {
        "cameras": {
            "camera1": {
                "frames": [{"pose": pose.tolist()} for pose in expected_poses],
                "resolution": [4, 3],
                "focal_length": [3, 3],
                "principal_point": [2, 1.5],
                "model_type": "opencv-pinhole",
                "radial_coeffs": [0] * 6,
                "tangential_coeffs": [0] * 2,
                "thin_prism_coeffs": [0] * 4,
                "target": "world",
            }
        },
        "counts": {"cameras": 1, "images": 2, "poses": 2, "points": 2},
    }
    return meta, source


@pytest.mark.parametrize("derive", [False, True])
def test_real_v4_reader_decodes_all_data_and_relocates(tmp_path, derive):
    meta, source = ncore_fixture(tmp_path / "original")
    if derive:
        colmap._derive_in_place(meta, "camera1")
    relocated = tmp_path / "relocated"
    shutil.copytree(meta.parent, relocated)
    shutil.rmtree(meta.parent)
    counts = colmap.validate_ncore_sequence(
        relocated / meta.name, source, rig_mode="derive" if derive else "preserve"
    )
    assert counts == source["counts"]
    assert all(not path.is_symlink() for path in relocated.iterdir())


@pytest.mark.parametrize("corruption", ["pose", "calibration", "points", "image"])
def test_real_v4_reader_rejects_corrupt_data(tmp_path, corruption):
    meta, source = ncore_fixture(tmp_path, corruption=corruption)
    with pytest.raises(
        (colmap.NcoreConversionError, OSError), match="non-finite|image"
    ):
        colmap.validate_ncore_sequence(meta, source)


def test_real_v4_reader_rejects_lost_frames(tmp_path):
    meta, source = ncore_fixture(tmp_path)
    source["cameras"]["camera1"]["frames"].append(
        source["cameras"]["camera1"]["frames"][0]
    )
    with pytest.raises(colmap.NcoreConversionError, match="counts"):
        colmap.validate_ncore_sequence(meta, source)


def colmap_text_fixture(root, image_name="frame1.png"):
    from PIL import Image

    (root / "sparse/0").mkdir(parents=True)
    (root / "images").mkdir()
    (root / "sparse/0/cameras.txt").write_text("1 PINHOLE 4 3 3 3 2 1.5\n")
    (root / "sparse/0/images.txt").write_text(
        f"1 1 0 0 0 0 0 0 1 {image_name}\n0 0 1\n2 1 0 0 0 -1 0 0 1 frame2.png\n0 0 2\n"
    )
    (root / "sparse/0/points3D.txt").write_text(
        "1 1 2 3 20 30 40 0 1 0\n2 4 5 6 50 60 70 0 2 0\n3 0 0 0 0 0 0 0 1 0\n"
    )
    for index in (1, 2):
        Image.new("RGB", (4, 3), (index * 100, 20, 50)).save(
            root / f"images/frame{index}.png"
        )
    return root


@pytest.mark.parametrize("derive", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_pinned_upstream_colmap_code_and_actual_readers_round_trip(
    tmp_path, derive, binary
):
    """Real official conversion code on synthetic inputs; no GPU or live S3 claim."""
    pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner

    root = colmap_text_fixture(tmp_path / "capture")
    if binary:
        import struct

        sparse = root / "sparse/0"
        (sparse / "cameras.bin").write_bytes(
            struct.pack("<QIiQQ4d", 1, 1, 1, 4, 3, 3, 3, 2, 1.5)
        )
        with (sparse / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 2))
            for index in (1, 2):
                stream.write(
                    struct.pack("<I4d3dI", index, 1, 0, 0, 0, 1 - index, 0, 0, 1)
                )
                stream.write(f"frame{index}.png".encode() + b"\0")
                stream.write(struct.pack("<QddQ", 1, 0, 0, index))
        with (sparse / "points3D.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", 3))
            for index, xyz in enumerate(((1, 2, 3), (4, 5, 6), (0, 0, 0)), 1):
                stream.write(
                    struct.pack("<Q3d3BdQII", index, *xyz, 20, 30, 40, 0, 1, 1, 0)
                )
        for path in sparse.glob("*.txt"):
            path.unlink()
    request = colmap.ColmapConversionRequest(
        input_path="s3://test-bucket/input/",
        output_path="s3://test-bucket/out/",
        include_downsampled_images=False,
    )
    source = colmap.inspect_colmap_source(root, request)
    assert source["counts"] == {"cameras": 1, "images": 2, "poses": 2, "points": 2}
    assert source["origin_points_filtered"] == 1
    out = tmp_path / "output"
    result = CliRunner().invoke(
        converter.cli,
        [
            "--root-dir",
            str(root),
            "--output-dir",
            str(out),
            "colmap-v4",
            "--no-include-downsampled-images",
        ],
    )
    assert result.exit_code == 0, result.output
    meta = out / "capture/capture.json"
    if derive:
        colmap._derive_in_place(meta, "camera1")
    assert (
        colmap.validate_ncore_sequence(
            meta, source, rig_mode="derive" if derive else "preserve"
        )
        == source["counts"]
    )


def test_trueprice_reader_rejects_unsafe_image_reference(tmp_path):
    pytest.importorskip("pycolmap")
    root = colmap_text_fixture(tmp_path, image_name="../escape.png")
    request = colmap.ColmapConversionRequest(
        input_path="s3://test-bucket/input/", output_path="s3://test-bucket/out/"
    )
    with pytest.raises(ValueError, match="relative path"):
        colmap.inspect_colmap_source(root, request)


def complete_colmap_text_fixture(root):
    """Non-contiguous raw IDs, whitespace and an empty observation record."""
    from PIL import Image

    colmap_text_fixture(root)
    (root / "sparse/0/cameras.txt").write_text(
        "# cameras\n1 PINHOLE 4 3 3 3 2 1.5\n \t\n5 PINHOLE 4 3 4 4 2 1.5\n\n"
    )
    (root / "sparse/0/images.txt").write_text(
        "# images\n\n1 1 0 0 0 0 0 0 1 frame1.png\n0 0 1\n"
        "2 1 0 0 0 -1 0 0 1 frame2.png\n0 0 2 1 1 -1\n\n"
        "7 1 0 0 0 0 -1 0 5 frame7.png\n \t\n"
        "# later image\n11 1 0 0 0 -1 -1 0 5 frame11.png\n0 0 9\n"
    )
    (root / "sparse/0/points3D.txt").write_text(
        "# points\n1 1 2 3 20 30 40 0 1 0\n\n"
        "2 4 5 6 50 60 70 0 2 0\n3 0 0 0 0 0 0 0 1 0\n \t\n"
        "9 7 8 9 80 90 100 0 11 0\n"
    )
    for index in (7, 11):
        Image.new("RGB", (4, 3), (index * 20, 40, 60)).save(
            root / f"images/frame{index}.png"
        )
    return root


@pytest.mark.parametrize("derive", [False, True])
def test_official_text_roundtrip_preserves_blank_lines_and_empty_observations(
    tmp_path, derive
):
    pycolmap = pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner

    root = complete_colmap_text_fixture(tmp_path / "capture")
    scene = pycolmap.SceneManager(str(root / "sparse/0"))
    scene.load()
    assert set(scene.cameras) == {1, 5}
    assert set(scene.images) == {1, 2, 7, 11}
    assert list(scene.point3D_ids) == [1, 2, 3, 9]
    assert scene.points3D.shape == (4, 3)
    assert scene.images[7].points2D.shape == (0, 2)
    assert scene.images[7].point3D_ids.shape == (0,)
    assert scene.images[7].point3D_ids.dtype.name == "uint64"
    assert list(scene.images[2].point3D_ids) == [2, 2**64 - 1]
    source = colmap.inspect_colmap_source(root, publication_request(tmp_path))
    assert source["counts"] == {"cameras": 2, "images": 4, "poses": 4, "points": 3}
    assert source["source_points"] == 4
    assert source["origin_points_filtered"] == 1
    assert [f["name"] for f in source["cameras"]["camera5"]["frames"]] == [
        "frame7.png",
        "frame11.png",
    ]
    out = tmp_path / "output"
    result = CliRunner().invoke(
        converter.cli,
        ["--root-dir", str(root), "--output-dir", str(out), "colmap-v4"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    meta = out / "capture/capture.json"
    if derive:
        colmap._derive_in_place(meta, "camera1")
    assert (
        colmap.validate_ncore_sequence(
            meta, source, rig_mode="derive" if derive else "preserve"
        )
        == source["counts"]
    )


@pytest.mark.parametrize("component", ["cameras", "images", "points3D"])
@pytest.mark.parametrize("mutation", ["drop", "replace"])
def test_text_preflight_independently_checks_raw_ids_and_counts(
    tmp_path, monkeypatch, component, mutation
):
    pycolmap = pytest.importorskip("pycolmap")
    root = colmap_text_fixture(tmp_path / "capture")
    load = pycolmap.SceneManager.load

    def incomplete_load(scene):
        load(scene)
        if component == "points3D":
            if mutation == "drop":
                scene.point3D_ids = scene.point3D_ids[:-1]
                scene.points3D = scene.points3D[:-1]
            else:
                scene.point3D_ids[-1] = 99
        else:
            records = getattr(scene, component)
            value = records.pop(next(iter(records)))
            if mutation == "replace":
                records[99] = value

    monkeypatch.setattr(pycolmap.SceneManager, "load", incomplete_load)
    with pytest.raises(colmap.NcoreConversionError, match="raw.*" + component):
        colmap.inspect_colmap_source(root, publication_request(tmp_path))


@pytest.mark.parametrize(
    "component,contents",
    [
        ("images", "1 1 0 0 0 0 0 0 1 frame.png\n"),
        ("images", "1 1 0 0 0 0 0 0 1\n\n"),
        ("images", "1 1 0 0 0 0 0 0 1 frame.png\n0 0\n"),
        ("images", "1 1 0 0 0 0 0 0 1 frame.png\n0 0 bad\n"),
        ("images", "1 1 0 0 0 0 0 0 1 frame.png\n\n" * 2),
        ("cameras", "1 PINHOLE 4 3\n"),
        ("cameras", "1 PINHOLE 4 3 3 3 2 1.5\n" * 2),
        ("points3D", "1 1 2 3 20 30 40\n"),
        ("points3D", "1 1 2 3 20 30 40 0 1\n"),
        ("points3D", "1 1 2 3 20 30 40 0 1 0\n" * 2),
    ],
)
def test_raw_text_preflight_rejects_incomplete_or_duplicate_records(
    tmp_path, component, contents
):
    path = tmp_path / f"{component}.txt"
    path.write_text(contents)
    with pytest.raises(colmap.NcoreConversionError, match="COLMAP.*text"):
        colmap._raw_text_model_ids(path, component)


def test_raw_text_preflight_counts_empty_final_observations(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# comment\n\n7 1 0 0 0 0 0 0 1 frame.png\n\n")
    assert colmap._raw_text_model_ids(path, "images") == {7}


@pytest.mark.parametrize("ending", ["", "\n", "\n# missing observations\n"])
def test_text_preflight_and_official_converter_reject_incomplete_final_record(
    tmp_path, monkeypatch, ending
):
    pycolmap = pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner

    root = colmap_text_fixture(tmp_path / "capture")
    with (root / "sparse/0/images.txt").open("a") as stream:
        stream.write("7 1 0 0 0 0 -1 0 1 frame7.png" + ending)
    result = CliRunner().invoke(
        converter.cli,
        ["--root-dir", str(root), "--output-dir", str(tmp_path / "out"), "colmap-v4"],
    )
    assert result.exit_code != 0
    assert "incomplete COLMAP text image record" in str(result.exception)

    def must_not_load(*args, **kwargs):
        pytest.fail("incomplete raw text reached the shared upstream reader")

    monkeypatch.setattr(pycolmap.SceneManager, "load", must_not_load)
    with pytest.raises(colmap.NcoreConversionError, match="incomplete.*text.*image"):
        colmap.inspect_colmap_source(root, publication_request(tmp_path))


def masked_downsample_fixture(root, convention):
    import numpy as np
    from PIL import Image

    colmap_text_fixture(root)
    (root / "sparse/0/cameras.txt").write_text("1 PINHOLE 8 6 6 6 4 3\n")
    (root / "images_2").mkdir()
    masks = root / ("custom_masks" if convention == "explicit" else "masks")
    masks.mkdir()
    originals, downsampled = {}, {}
    for index in (1, 2):
        Image.new("RGB", (8, 6), (index * 100, 20, 50)).save(
            root / f"images/frame{index}.png"
        )
        # Deliberately differs from rounded calibration / 2.
        Image.new("RGB", (3, 2), (index * 100, 20, 50)).save(
            root / f"images_2/frame{index}.png"
        )
        original = (np.arange(48).reshape(6, 8) * 5 + index).astype(np.uint8)
        mask_path = masks / f"frame{index}.png"
        Image.fromarray(original).save(mask_path)
        originals[mask_path] = (mask_path.read_bytes(), original)
        downsampled[index] = original[np.ix_([1, 4], [1, 4, 6])]
        if convention == "per_camera":
            downsampled[index] = np.full((2, 3), 100 + index, dtype=np.uint8)
            Image.fromarray(downsampled[index]).save(
                root / f"images_2/frame{index}_mask.png"
            )
    return originals, downsampled


@pytest.mark.parametrize("convention", ["masks", "explicit", "per_camera"])
def test_official_masked_downsampling_preserves_pixels_and_originals(
    tmp_path, convention
):
    pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner
    import numpy as np
    from ncore.data.v4 import CameraSensorComponent, SequenceComponentGroupsReader
    from upath import UPath

    root = tmp_path / "capture"
    originals, downsampled = masked_downsample_fixture(root, convention)
    source = colmap.inspect_colmap_source(
        root,
        publication_request(tmp_path).model_copy(
            update={"include_downsampled_images": True}
        ),
    )
    out = tmp_path / "output"
    result = CliRunner().invoke(
        converter.cli,
        [
            "--root-dir",
            str(root),
            "--output-dir",
            str(out),
            "colmap-v4",
            "--include-downsampled-images",
        ]
        + (["--masks-dir", "custom_masks"] if convention == "explicit" else []),
    )
    assert result.exit_code == 0, (result.output, result.exception)
    meta = out / "capture/capture.json"
    assert colmap.validate_ncore_sequence(meta, source) == source["counts"]
    assert source["counts"] == {"cameras": 2, "images": 4, "poses": 4, "points": 2}
    cameras = SequenceComponentGroupsReader([UPath(meta)]).open_component_readers(
        CameraSensorComponent.Reader
    )
    for index, (path, (encoded, original)) in enumerate(originals.items(), 1):
        assert path.read_bytes() == encoded
        timestamp = (index - 1) * 1_000_000
        np.testing.assert_array_equal(
            cameras["camera1"].get_frame_generic_data(timestamp, "mask"), original
        )
        np.testing.assert_array_equal(
            cameras["camera1_2"].get_frame_generic_data(timestamp, "mask"),
            downsampled[index],
        )


@pytest.mark.parametrize("wrong", ["mask", "original_image"])
def test_official_masked_downsampling_rejects_unverified_source_dimensions(
    tmp_path, wrong
):
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner
    from PIL import Image

    root = tmp_path / "capture"
    masked_downsample_fixture(root, "masks")
    relative = "masks/frame1.png" if wrong == "mask" else "images/frame1.png"
    Image.new("L" if wrong == "mask" else "RGB", (7, 6)).save(root / relative)
    result = CliRunner().invoke(
        converter.cli,
        [
            "--root-dir",
            str(root),
            "--output-dir",
            str(tmp_path / "out"),
            "colmap-v4",
            "--include-downsampled-images",
        ],
    )
    assert result.exit_code != 0
    assert "dimensions" in str(result.exception)


def test_real_v4_reader_still_rejects_wrong_mask_dimensions(tmp_path):
    meta, source = ncore_fixture(tmp_path, corruption="mask")
    with pytest.raises(colmap.NcoreConversionError, match="mask dimensions"):
        colmap.validate_ncore_sequence(meta, source)


def test_binary_image_preflight_rejects_truncation_before_vendor_reader(tmp_path):
    import struct

    malformed = tmp_path / "images.bin"
    malformed.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<I4d3dI", 1, 1, 0, 0, 0, 0, 0, 0, 1)
        + b"unterminated-name"
    )
    with pytest.raises(colmap.NcoreConversionError, match="binary"):
        colmap.validate_binary_images(malformed)


def test_nre_discovers_sequence_meta_instead_of_conversion_report(tmp_path):
    from npa.workbench.nurec.nurec import find_ncore_json

    portable = tmp_path / "sequence.json"
    portable.write_text(
        json.dumps(
            {"version": "v4", "component_stores": [{"path": "capture.zarr.itar"}]}
        )
    )
    (tmp_path / "capture.zarr.itar").write_bytes(b"discovery fixture only")
    (tmp_path / "conversion.json").write_text('{"status":"ok"}')
    (tmp_path / "npa-rig.json").write_text('{"reference_camera":"camera1"}')
    assert find_ncore_json(tmp_path) == portable


def publication_request(tmp_path):
    return colmap.ColmapConversionRequest(
        input_path="s3://test-bucket/input/",
        output_path="s3://test-bucket/output/",
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "scratch",
        rig_mode="preserve",
        include_downsampled_images=False,
    )


def assert_published_inventory(storage, result):
    report = json.loads(storage.uploads[result["conversion_uri"]])
    for member in report["members"]:
        payload = storage.uploads[result["output_path"] + member["path"]]
        assert len(payload) == member["bytes"]
        assert hashlib.sha256(payload).hexdigest() == member["sha256"]


def test_republication_cannot_modify_committed_generation(monkeypatch, tmp_path):
    storage, _ = fake_conversion(monkeypatch, tmp_path)
    request = publication_request(tmp_path)
    first = colmap.convert_colmap(request, storage_client=storage)
    before = storage.uploads.copy()
    attempts = []

    def failed_replacement(source, uri):
        attempts.append(uri)
        if len(attempts) == 2:
            raise OSError("private source diagnostic")
        storage.uploads[uri] = b"different generation"

    monkeypatch.setattr(storage, "upload_file", failed_replacement)
    with pytest.raises(colmap.NcoreConversionError, match="destination|prefix"):
        colmap.convert_colmap(request, storage_client=storage)
    assert not attempts
    assert storage.uploads == before
    assert_published_inventory(storage, first)


@pytest.mark.parametrize(
    "existing", ["data.zarr.itar", "npa-rig.json", "sequence.json"]
)
def test_preexisting_unclaimed_destination_is_unchanged(
    monkeypatch, tmp_path, existing
):
    storage, _ = fake_conversion(monkeypatch, tmp_path)
    request = publication_request(tmp_path)
    uri = request.output_path + existing
    storage.uploads[uri] = b"previous generation"
    before = storage.uploads.copy()
    with pytest.raises(colmap.NcoreConversionError, match="destination|prefix"):
        colmap.convert_colmap(request, storage_client=storage)
    assert storage.uploads == before


def test_interleaved_writers_have_one_atomic_winner(monkeypatch, tmp_path):
    storage, _ = fake_conversion(monkeypatch, tmp_path)
    request = publication_request(tmp_path)
    put = storage.put_object
    winners = []

    def interleave(**kwargs):
        # Both writers reached PutObject after inspecting an empty destination.
        monkeypatch.setattr(storage, "put_object", put)
        winners.append(colmap.convert_colmap(request, storage_client=storage))
        return put(**kwargs)

    monkeypatch.setattr(storage, "put_object", interleave)
    with pytest.raises(colmap.NcoreConversionError, match="destination|prefix"):
        colmap.convert_colmap(request, storage_client=storage)
    assert len(winners) == 1
    assert_published_inventory(storage, winners[0])


def test_failed_publication_claim_is_not_reused(monkeypatch, tmp_path):
    storage, _ = fake_conversion(monkeypatch, tmp_path)
    request = publication_request(tmp_path)
    upload = storage.upload_file

    def fail(source, uri):
        if uri.endswith("conversion.json"):
            raise OSError("private source diagnostic")
        upload(source, uri)

    monkeypatch.setattr(storage, "upload_file", fail)
    with pytest.raises(colmap.NcoreConversionError, match="publication") as error:
        colmap.convert_colmap(request, storage_client=storage)
    assert "private source" not in str(error.value)
    assert request.output_path + "sequence.json" not in storage.uploads
    before = storage.uploads.copy()
    monkeypatch.setattr(storage, "upload_file", upload)
    with pytest.raises(colmap.NcoreConversionError, match="fresh.*prefix"):
        colmap.convert_colmap(request, storage_client=storage)
    assert storage.uploads == before


def write_inventory_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "sequence.json"
    meta.write_text(
        json.dumps({"version": "v4", "component_stores": [{"path": "data.zarr.itar"}]})
    )
    (root / "data.zarr.itar").write_bytes(b"synthetic generation A")
    report = {
        "schema_version": 1,
        "status": "ok",
        "engine": "nvidia-ncore-colmap",
        "ncore_meta": meta.name,
        "poses_component_group": "default",
        "members": colmap._inventory(root, [meta, root / "data.zarr.itar"]),
    }
    (root / "conversion.json").write_text(json.dumps(report))
    return root


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "corruption", ["", "shard", "meta", "extra-rig", "missing", "report", "claim-only"]
)
def test_materialization_verifies_conversion_inventory(tmp_path, remote, corruption):
    from npa.workbench.nurec.nurec import materialize_uri, NurecError

    source = write_inventory_fixture(tmp_path / "source")
    if corruption == "shard":
        (source / "data.zarr.itar").write_bytes(b"synthetic generation B")
    elif corruption == "meta":
        (source / "sequence.json").write_text("{}")
    elif corruption == "extra-rig":
        (source / "npa-rig.json").write_text('{"reference_camera":"stale"}')
    elif corruption == "missing":
        (source / "data.zarr.itar").unlink()
    elif corruption == "report":
        (source / "conversion.json").write_text("private invalid report")
    elif corruption == "claim-only":
        (source / "conversion.json").unlink()
        (source / ".npa-colmap-claim.json").write_text("{}")

    class Download:
        def download_path(self, uri, target):
            shutil.copytree(source, target, dirs_exist_ok=True)

    uri = "s3://test-bucket/input/" if remote else str(source)
    if corruption:
        with pytest.raises(
            NurecError, match="conversion|inventory|publication"
        ) as error:
            materialize_uri(uri, tmp_path / "download", storage_client=Download())
        assert "private invalid" not in str(error.value)
        assert str(source) not in str(error.value)
    else:
        result = materialize_uri(uri, tmp_path / "download", storage_client=Download())
        assert (result / "sequence.json").is_file()


def test_materialization_preserves_non_colmap_sequences(tmp_path):
    from npa.workbench.nurec.nurec import materialize_uri

    source = tmp_path / "ordinary-ncore"
    source.mkdir()
    (source / "capture.json").write_text('{"version":"v4"}')
    (source / "npa-rig.json").write_text('{"reference_camera":"camera1"}')
    assert materialize_uri(str(source), tmp_path / "unused") == source


@pytest.mark.parametrize("different_k", [False, True])
def test_official_downsampling_preserves_each_camera_calibration(tmp_path, different_k):
    pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner
    from PIL import Image
    import numpy as np
    from ncore.data.v4 import IntrinsicsComponent, SequenceComponentGroupsReader
    from upath import UPath

    root = colmap_text_fixture(tmp_path / "capture")
    second_k = "6 5 3 2" if different_k else "4 4 4 3"
    (root / "sparse/0/cameras.txt").write_text(
        "1 OPENCV 8 6 4 4 4 3 0.01 0.02 0.03 0.04\n"
        f"2 OPENCV 8 6 {second_k} 0.05 0.06 0.07 0.08\n"
    )
    (root / "sparse/0/images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame1.png\n0 0 1\n"
        "2 1 0 0 0 -1 0 0 1 frame2.png\n0 0 2\n"
        "3 1 0 0 0 0 -1 0 2 frame3.png\n0 0 1\n"
        "4 1 0 0 0 -1 -1 0 2 frame4.png\n0 0 2\n"
    )
    (root / "images_2").mkdir()
    for index in (1, 2, 3, 4):
        Image.new("RGB", (8, 6), (index * 50, 20, 50)).save(
            root / f"images/frame{index}.png"
        )
        Image.new("RGB", (4, 3), (index * 50, 20, 50)).save(
            root / f"images_2/frame{index}.png"
        )
    source = colmap.inspect_colmap_source(
        root,
        publication_request(tmp_path).model_copy(
            update={"include_downsampled_images": True}
        ),
    )
    out = tmp_path / "output"
    result = CliRunner().invoke(
        converter.cli,
        [
            "--root-dir",
            str(root),
            "--output-dir",
            str(out),
            "colmap-v4",
            "--include-downsampled-images",
        ],
    )
    assert result.exit_code == 0, result.output
    meta = out / "capture/capture.json"
    assert colmap.validate_ncore_sequence(meta, source) == source["counts"]
    assert source["counts"] == {"cameras": 4, "images": 8, "poses": 8, "points": 2}
    intrinsics = SequenceComponentGroupsReader([UPath(meta)]).open_component_readers(
        IntrinsicsComponent.Reader
    )["default"]
    for camera_id, radial, tangential in (
        ("camera1", [0.01, 0.02], [0.03, 0.04]),
        ("camera2", [0.05, 0.06], [0.07, 0.08]),
    ):
        for suffix in ("", "_2"):
            model = intrinsics.get_camera_model_parameters(camera_id + suffix)
            np.testing.assert_allclose(model.radial_coeffs, [*radial, 0, 0, 0, 0])
            np.testing.assert_allclose(model.tangential_coeffs, tangential)
        np.testing.assert_allclose(
            intrinsics.get_camera_model_parameters(camera_id + "_2").focal_length,
            intrinsics.get_camera_model_parameters(camera_id).focal_length / 2,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "opencv-fisheye"),
        ("radial_coeffs", [0.1, 0, 0, 0, 0, 0]),
        ("tangential_coeffs", [0.1, 0]),
        ("thin_prism_coeffs", [0.1, 0, 0, 0]),
    ],
)
def test_validator_rejects_finite_but_wrong_intrinsic_model(tmp_path, field, value):
    meta, source = ncore_fixture(tmp_path)
    source["cameras"]["camera1"][field] = value
    with pytest.raises(colmap.NcoreConversionError, match="calibration"):
        colmap.validate_ncore_sequence(meta, source)


@pytest.mark.parametrize(
    "model,parameters,radial,tangential",
    [
        ("SIMPLE_PINHOLE", "3 2 1.5", [0] * 6, [0, 0]),
        ("PINHOLE", "3 3 2 1.5", [0] * 6, [0, 0]),
        ("SIMPLE_RADIAL", "3 2 1.5 0.01", [0.01, 0, 0, 0, 0, 0], [0, 0]),
        ("RADIAL", "3 2 1.5 0.01 0.02", [0.01, 0.02, 0, 0, 0, 0], [0, 0]),
        (
            "OPENCV",
            "3 3 2 1.5 0.01 0.02 0.03 0.04",
            [0.01, 0.02, 0, 0, 0, 0],
            [0.03, 0.04],
        ),
        (
            "OPENCV_FISHEYE",
            "3 3 2 1.5 0.01 0.02 0.03 0.04",
            [0.01, 0.02, 0.03, 0.04],
            None,
        ),
    ],
)
def test_all_supported_source_models_preserve_distortion(
    tmp_path, model, parameters, radial, tangential
):
    pytest.importorskip("pycolmap")
    converter = pytest.importorskip("tools.data_converter.colmap.converter")
    from click.testing import CliRunner
    import numpy as np

    root = colmap_text_fixture(tmp_path / "capture")
    (root / "sparse/0/cameras.txt").write_text(f"1 {model} 4 3 {parameters}\n")
    source = colmap.inspect_colmap_source(root, publication_request(tmp_path))
    calibration = source["cameras"]["camera1"]
    assert calibration["colmap_camera_type"] == [
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
        "OPENCV_FISHEYE",
    ].index(model)
    np.testing.assert_allclose(calibration["radial_coeffs"], radial)
    if tangential is not None:
        np.testing.assert_allclose(calibration["tangential_coeffs"], tangential)
        assert calibration["thin_prism_coeffs"] == [0] * 4
    out = tmp_path / "output"
    result = CliRunner().invoke(
        converter.cli,
        [
            "--root-dir",
            str(root),
            "--output-dir",
            str(out),
            "colmap-v4",
            "--no-include-downsampled-images",
        ],
    )
    assert result.exit_code == 0, result.output
    meta = out / "capture/capture.json"
    assert colmap.validate_ncore_sequence(meta, source) == source["counts"]
    for index in range(len(radial)):
        original = calibration["radial_coeffs"][index]
        calibration["radial_coeffs"][index] += 0.1
        with pytest.raises(colmap.NcoreConversionError, match="calibration"):
            colmap.validate_ncore_sequence(meta, source)
        calibration["radial_coeffs"][index] = original


@pytest.mark.parametrize("fault", ["unsupported", "ambiguous", "legacy-race"])
def test_claim_failures_never_publish_members(monkeypatch, tmp_path, fault):
    storage, _ = fake_conversion(monkeypatch, tmp_path)
    request = publication_request(tmp_path)
    original = storage.put_object

    def fail(**kwargs):
        if fault == "unsupported":
            raise NotImplementedError("private provider diagnostic")
        result = original(**kwargs)
        if fault == "ambiguous":
            raise OSError("private provider diagnostic")
        storage.uploads[request.output_path + "data.zarr.itar"] = b"legacy writer"
        return result

    monkeypatch.setattr(storage, "put_object", fail)
    with pytest.raises(
        colmap.NcoreConversionError, match="publication|destination"
    ) as error:
        colmap.convert_colmap(request, storage_client=storage)
    assert "private provider" not in str(error.value)
    assert request.output_path + "sequence.json" not in storage.uploads
    assert request.output_path + "conversion.json" not in storage.uploads
    if fault == "unsupported":
        assert not storage.uploads
    else:
        before = storage.uploads.copy()
        monkeypatch.setattr(storage, "put_object", original)
        with pytest.raises(colmap.NcoreConversionError, match="fresh.*prefix"):
            colmap.convert_colmap(request, storage_client=storage)
        assert storage.uploads == before


@pytest.mark.parametrize(
    "corruption", ["", "missing-claim", "changed-claim", "changed-report"]
)
def test_claim_binds_published_inventory_before_materialization(
    monkeypatch, tmp_path, corruption
):
    from npa.workbench.nurec.nurec import materialize_uri

    storage, _ = fake_conversion(monkeypatch, tmp_path)
    result = colmap.convert_colmap(
        publication_request(tmp_path), storage_client=storage
    )
    root = tmp_path / "published"
    root.mkdir()
    for uri, payload in storage.uploads.items():
        (root / uri.rsplit("/", 1)[-1]).write_bytes(payload)
    if corruption == "missing-claim":
        (root / colmap.PUBLICATION_CLAIM).unlink()
    elif corruption == "changed-claim":
        (root / colmap.PUBLICATION_CLAIM).write_text(
            '{"schema_version":1,"report_sha256":"wrong"}'
        )
    elif corruption == "changed-report":
        report = json.loads((root / colmap.CONVERSION_REPORT).read_text())
        report["counts"]["images"] += 1
        (root / colmap.CONVERSION_REPORT).write_text(json.dumps(report))
    if corruption:
        with pytest.raises(colmap.NcoreConversionError, match="inventory"):
            materialize_uri(str(root), tmp_path / "unused")
    else:
        assert materialize_uri(str(root), tmp_path / "unused") == root
        assert result["objects"] == len(storage.uploads)


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "unsafe-path", "invalid-path-type", "unlisted-meta", "missing-rig"],
)
def test_inventory_rejects_invalid_or_incomplete_members(tmp_path, corruption):
    root = write_inventory_fixture(tmp_path / "source")
    path = root / colmap.CONVERSION_REPORT
    report = json.loads(path.read_text())
    if corruption == "duplicate":
        report["members"].append(report["members"][0])
    elif corruption == "unsafe-path":
        report["members"][0]["path"] = "../private-source"
    elif corruption == "invalid-path-type":
        report["members"][0]["path"] = 7
    elif corruption == "unlisted-meta":
        report["members"] = [
            item for item in report["members"] if item["path"] != "sequence.json"
        ]
    else:
        report["poses_component_group"] = "npa_rig"
    path.write_text(json.dumps(report))
    with pytest.raises(colmap.NcoreConversionError, match="inventory") as error:
        colmap.verify_conversion_inventory(root)
    assert "private-source" not in str(error.value)


def test_stale_local_metadata_cannot_complete_interrupted_remote_publication(
    monkeypatch, tmp_path
):
    from npa.workbench.nurec.nurec import materialize_uri

    storage, _ = fake_conversion(monkeypatch, tmp_path)
    colmap.convert_colmap(publication_request(tmp_path), storage_client=storage)
    target = tmp_path / "cache-destination"
    target.mkdir()
    for uri, payload in storage.uploads.items():
        (target / uri.rsplit("/", 1)[-1]).write_bytes(payload)
    previous = {path.name: path.read_bytes() for path in target.iterdir()}

    class InterruptedDownload:
        def download_path(self, uri, destination):
            # The new prefix has every byte except its final discovery commit.
            for name, payload in previous.items():
                if name != "sequence.json":
                    (Path(destination) / name).write_bytes(payload)

    with pytest.raises(colmap.NcoreConversionError, match="inventory"):
        materialize_uri(
            "s3://test-bucket/interrupted/",
            target,
            storage_client=InterruptedDownload(),
        )
    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous


def test_new_capture_does_not_merge_a_previous_cached_capture(tmp_path):
    from npa.clients.storage import StorageClient
    from npa.workbench.nurec.nurec import find_ncore_json, materialize_uri

    target = tmp_path / "cache"
    old = write_inventory_fixture(target / "old")
    metadata = json.dumps(
        {"version": "v4", "component_stores": [{"path": "new.zarr.itar"}]}
    ).encode()
    objects = {
        "new/capture/z-new.json": metadata,
        "new/capture/new.zarr.itar": b"synthetic preconverted capture",
    }

    class MemoryS3:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, *, Bucket, Prefix):
            assert Bucket == "test-bucket"
            yield {
                "Contents": [{"Key": key} for key in objects if key.startswith(Prefix)]
            }

        def download_file(self, bucket, key, destination):
            assert bucket == "test-bucket"
            Path(destination).write_bytes(objects[key])

    client = object.__new__(StorageClient)
    client._s3 = MemoryS3()
    staged = materialize_uri("s3://test-bucket/new/", target, storage_client=client)
    found = find_ncore_json(staged)
    assert found is not None
    assert found.name == "z-new.json"
    assert found.read_bytes() == metadata
    assert not (staged / "old").exists()
    assert (old / "sequence.json").exists()


@pytest.mark.parametrize("claimed", [False, True])
def test_empty_remote_prefix_never_returns_a_previous_cached_capture(tmp_path, claimed):
    import hashlib

    from npa.clients.storage import StorageClient
    from npa.workbench.nurec.nurec import find_ncore_json, materialize_uri

    cache = write_inventory_fixture(tmp_path / "cache")
    if claimed:
        report_path = cache / colmap.CONVERSION_REPORT
        report = json.loads(report_path.read_text())
        report["publication"] = {
            "mode": "immutable-prefix-v1",
            "claim": colmap.PUBLICATION_CLAIM,
        }
        report_path.write_text(json.dumps(report))
        (cache / colmap.PUBLICATION_CLAIM).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                }
            )
        )
    colmap.verify_conversion_inventory(cache)
    previous = {path.name: path.read_bytes() for path in cache.iterdir()}

    class EmptyS3:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, *, Bucket, Prefix):
            assert Bucket == "test-bucket"
            assert Prefix == "empty/"
            yield {}

    client = object.__new__(StorageClient)
    client._s3 = EmptyS3()
    staged = materialize_uri("s3://test-bucket/empty/", cache, storage_client=client)
    assert staged != cache
    assert find_ncore_json(staged) is None
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == previous

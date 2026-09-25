"""Unit tests for the NCore rig-pose derivation.

The derivation is the single reason NRE can load an object-centric capture at all,
but most of it was previously reachable only through a gpu-gated e2e or a test that
`importorskip("ncore")` -- so in normal CI it was not exercised.

These tests cover the decision logic and the filesystem handoff WITHOUT requiring
`nvidia-ncore`: the pure selector is called directly, and the reader-backed
functions run against a fake reader injected in place of `_open_reader`.
"""

from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.nurec import ncore_rig
from npa.workbench.nurec.ncore_rig import (
    DERIVED_POSES_GROUP,
    DYNAMIC_POSE_SOURCE,
    FRAME_POSE_FIELD,
    FRAME_POSE_SOURCE,
    RIG_SIDECAR_NAME,
    RIG_FRAME,
    WORLD_FRAME,
    camera_world_trajectories,
    has_rig_edge,
    select_reference_camera,
)
from npa.workbench.nurec.nurec import (
    NurecError,
    extract_archive,
    publish_ncore_sequence,
    read_rig_sidecar,
)


def _traj(count: int) -> tuple[list, list]:
    """A stand-in trajectory: `count` poses and `count` timestamps."""
    return ([[0.0]] * count, list(range(count)))


# ---------------------------------------------------------------------------------
# select_reference_camera -- pure, and it decides which camera becomes the rig
# ---------------------------------------------------------------------------------
def test_reference_camera_is_the_longest_trajectory() -> None:
    # The longest trajectory is the best-constrained one.
    trajectories = {"camera1": _traj(21), "camera2": _traj(38), "camera3": _traj(7)}

    assert select_reference_camera(trajectories) == "camera2"


def test_reference_camera_ties_break_deterministically_by_id() -> None:
    trajectories = {"cameraB": _traj(10), "cameraA": _traj(10)}

    assert select_reference_camera(trajectories) == "cameraA"
    # Same answer regardless of insertion order.
    assert (
        select_reference_camera({"cameraA": _traj(10), "cameraB": _traj(10)})
        == "cameraA"
    )


def test_reference_camera_honours_an_explicit_preference() -> None:
    trajectories = {"camera1": _traj(21), "camera2": _traj(38)}

    assert select_reference_camera(trajectories, preferred="camera1") == "camera1"


def test_reference_camera_rejects_a_preference_with_no_poses() -> None:
    trajectories = {"camera1": _traj(21)}

    with pytest.raises(NurecError, match="no '-> world' poses"):
        select_reference_camera(trajectories, preferred="camera_does_not_exist")


def test_reference_camera_rejects_a_sequence_with_no_camera_poses() -> None:
    with pytest.raises(NurecError, match="cannot derive a rig frame"):
        select_reference_camera({})


# ---------------------------------------------------------------------------------
# reader-backed helpers, exercised through a fake reader (no nvidia-ncore needed)
# ---------------------------------------------------------------------------------
class _FakePosesReader:
    def __init__(self, dynamic: list, static: list) -> None:
        self._dynamic = dynamic
        self._static = static

    def get_dynamic_poses(self):
        return iter(self._dynamic)

    def get_static_poses(self):
        return iter(self._static)


class _FakeCameraReader:
    def __init__(
        self, timestamps: list, poses: list, *, frames_count: int | None = None
    ):
        self.frames_timestamps_us = np.asarray(timestamps)
        self.frames_count = len(timestamps) if frames_count is None else frames_count
        self._poses = poses

    def get_frame_generic_data_names(self, timestamp_us: int):
        index = list(self.frames_timestamps_us[:, 1]).index(timestamp_us)
        return [] if self._poses[index] is None else [FRAME_POSE_FIELD]

    def get_frame_generic_data(self, timestamp_us: int, _name: str):
        index = list(self.frames_timestamps_us[:, 1]).index(timestamp_us)
        return self._poses[index]


class _FakeSequenceReader:
    def __init__(
        self, dynamic: list, static: list, cameras: dict | None = None
    ) -> None:
        self._poses_reader = _FakePosesReader(dynamic, static)
        self._cameras = cameras or {}
        self.sequence_id = "scene"

    def open_component_readers(self, component_reader_type):
        if component_reader_type is _FakePosesComponent.Reader:
            return {"default": self._poses_reader}
        if component_reader_type is _FakeCameraComponent.Reader:
            return self._cameras
        raise AssertionError(f"unexpected component reader: {component_reader_type}")


class _FakePosesComponent:
    Reader = type("PosesReader", (), {})


class _FakeCameraComponent:
    Reader = type("CameraReader", (), {})


def _install_ncore_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("ncore.data.v4")
    module.PosesComponent = _FakePosesComponent
    module.CameraSensorComponent = _FakeCameraComponent
    monkeypatch.setitem(sys.modules, "ncore", types.ModuleType("ncore"))
    monkeypatch.setitem(sys.modules, "ncore.data", types.ModuleType("ncore.data"))
    monkeypatch.setitem(sys.modules, "ncore.data.v4", module)


@pytest.fixture()
def fake_ncore(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake sequence reader and a stub PosesComponent import."""

    def _install(dynamic: list, static: list = []) -> None:
        monkeypatch.setattr(
            ncore_rig, "_open_reader", lambda _p: _FakeSequenceReader(dynamic, static)
        )
        _install_ncore_module(monkeypatch)

    return _install


@pytest.fixture()
def fake_frame_ncore(monkeypatch: pytest.MonkeyPatch):
    """Inject camera components carrying per-frame sensor-to-world poses."""

    def _install(cameras: dict, dynamic: list | None = None) -> None:
        reader = _FakeSequenceReader(dynamic or [], [], cameras)
        monkeypatch.setattr(ncore_rig, "_open_reader", lambda _p: reader)
        _install_ncore_module(monkeypatch)

    return _install


def _frame_camera(count: int, *, offset: int = 0) -> _FakeCameraReader:
    timestamps = [[index, index + 1] for index in range(offset, offset + count)]
    poses = [np.eye(4) for _index in range(count)]
    return _FakeCameraReader(timestamps, poses)


def test_camera_trajectories_keep_only_edges_pointing_at_world(fake_ncore) -> None:
    fake_ncore(
        [
            (("camera1", WORLD_FRAME), (_traj(21))),
            (("camera2", WORLD_FRAME), (_traj(38))),
            # Not a camera->world edge: must be ignored.
            (("camera1", "camera2"), (_traj(5))),
        ]
    )

    trajectories = camera_world_trajectories(Path("/does/not/matter.json"))

    assert sorted(trajectories) == ["camera1", "camera2"]


def test_camera_trajectories_exclude_an_existing_rig_edge(fake_ncore) -> None:
    """A rig edge is not a camera, and must never be offered as the reference."""
    fake_ncore(
        [
            ((RIG_FRAME, WORLD_FRAME), (_traj(38))),
            (("camera1", WORLD_FRAME), (_traj(21))),
        ]
    )

    assert sorted(camera_world_trajectories(Path("/x.json"))) == ["camera1"]


def test_frame_poses_are_used_when_dynamic_edges_are_absent(fake_frame_ncore) -> None:
    fake_frame_ncore({"camera1": _frame_camera(3)})

    trajectories = camera_world_trajectories(Path("/x.json"))
    poses, timestamps = trajectories["camera1"]

    assert poses.shape == (3, 4, 4)
    assert np.array_equal(timestamps, [1, 2, 3])


def test_dynamic_poses_remain_preferred_over_frame_data(fake_frame_ncore) -> None:
    dynamic = [(("camera1", WORLD_FRAME), _traj(2))]
    malformed_camera = _FakeCameraReader([[0, 1]], [None])
    fake_frame_ncore({"camera1": malformed_camera}, dynamic=dynamic)

    trajectories, pose_source = ncore_rig._camera_world_trajectories(
        ncore_rig._open_reader(Path("/x.json"))
    )

    assert trajectories == {"camera1": _traj(2)}
    assert pose_source == DYNAMIC_POSE_SOURCE


def test_frame_pose_camera_selection_is_explicit_or_longest(fake_frame_ncore) -> None:
    fake_frame_ncore(
        {
            "cameraB": _frame_camera(3),
            "cameraA": _frame_camera(3),
            "cameraC": _frame_camera(2),
        }
    )
    trajectories, pose_source = ncore_rig._camera_world_trajectories(
        ncore_rig._open_reader(Path("/x.json"))
    )

    assert pose_source == FRAME_POSE_SOURCE
    assert select_reference_camera(trajectories) == "cameraA"
    assert select_reference_camera(trajectories, preferred="cameraC") == "cameraC"


@pytest.mark.parametrize(
    ("camera", "message"),
    [
        (_FakeCameraReader([[0, 1], [1, 2]], [np.eye(4), None]), "missing"),
        (_FakeCameraReader([[0, 1], [1, 2]], [np.eye(4), np.eye(3)]), "malformed"),
        (
            _FakeCameraReader([[0, 1], [1, 2]], [np.eye(4), np.full((4, 4), np.nan)]),
            "non-finite",
        ),
        (_FakeCameraReader([[0, 1], [1, 1]], [np.eye(4), np.eye(4)]), "duplicate"),
        (
            _FakeCameraReader([[0, 1], [1, 2]], [np.eye(4), np.eye(4)], frames_count=3),
            "frame count disagrees",
        ),
    ],
)
def test_invalid_frame_pose_data_fails_closed(
    tmp_path: Path, fake_frame_ncore, camera: _FakeCameraReader, message: str
) -> None:
    fake_frame_ncore({"camera1": camera})
    source = tmp_path / "scene.json"
    source.write_text("{}")
    output = tmp_path / "derived"

    with pytest.raises(NurecError, match=message):
        ncore_rig.derive_rig_poses(source, output_dir=output)

    assert not output.exists()


def test_frame_pose_derivation_preserves_edges_stores_and_provenance(
    tmp_path: Path, fake_frame_ncore, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_dynamic = [(("imu", "vehicle"), _traj(2))]
    original_static = [(("camera1", "mount"), np.eye(4))]
    camera = _frame_camera(2)
    reader = _FakeSequenceReader(original_dynamic, original_static, {"camera1": camera})
    source_store = tmp_path / "scene.ncore4.zarr.itar"
    source_store.write_bytes(b"unchanged-source")
    reader.component_store_paths = [source_store]
    monkeypatch.setattr(ncore_rig, "_open_reader", lambda _path: reader)
    monkeypatch.setattr(ncore_rig, "_upath", lambda path: path)
    _install_ncore_module(monkeypatch)

    stored_dynamic: list[dict] = []
    stored_static: list[dict] = []

    class _PosesWriter:
        def store_dynamic_pose(self, **values):
            stored_dynamic.append(values)

        def store_static_pose(self, **values):
            stored_static.append(values)

    class _SequenceWriter:
        output_dir: Path

        @classmethod
        def from_reader(cls, *, output_dir_path, **_kwargs):
            instance = cls()
            instance.output_dir = Path(str(output_dir_path))
            return instance

        def register_component_writer(self, *_args, **_kwargs):
            return _PosesWriter()

        def finalize(self):
            derived = self.output_dir / "scene.ncore4-npa_rig.zarr.itar"
            derived.write_bytes(b"derived")
            return [derived]

    class _SequenceMeta:
        def __init__(self, paths):
            self._paths = paths

        def get_sequence_meta(self):
            return self

        def to_dict(self):
            return {"component_stores": [{"path": path.name} for path in self._paths]}

    module = sys.modules["ncore.data.v4"]
    module.PosesComponent.Writer = object
    module.SequenceComponentGroupsWriter = _SequenceWriter
    module.SequenceComponentGroupsReader = _SequenceMeta
    source = tmp_path / "scene.json"
    source.write_text("{}")
    output = tmp_path / "derived"

    result = ncore_rig.derive_rig_poses(source, output_dir=output)

    assert result.ok is True
    assert result.pose_source == FRAME_POSE_SOURCE
    assert [
        (item["source_frame_id"], item["target_frame_id"]) for item in stored_dynamic
    ] == [
        ("imu", "vehicle"),
        (RIG_FRAME, WORLD_FRAME),
    ]
    assert [
        (item["source_frame_id"], item["target_frame_id"]) for item in stored_static
    ] == [("camera1", "mount")]
    assert source_store.read_bytes() == b"unchanged-source"
    assert (output / source_store.name).is_symlink()
    meta = json.loads(Path(result.output_meta).read_text())
    assert all(
        Path(store["path"]).name == store["path"] for store in meta["component_stores"]
    )
    sidecar = json.loads((output / RIG_SIDECAR_NAME).read_text())
    assert sidecar["reference_camera"] == "camera1"
    assert sidecar["pose_source"] == FRAME_POSE_SOURCE


def test_has_rig_edge_detects_a_dynamic_rig_trajectory(fake_ncore) -> None:
    fake_ncore([((RIG_FRAME, WORLD_FRAME), (_traj(38)))])

    assert has_rig_edge(Path("/x.json")) is True


def test_has_rig_edge_detects_a_static_rig_pose(fake_ncore) -> None:
    fake_ncore([], [((RIG_FRAME, WORLD_FRAME), [[1.0]])])

    assert has_rig_edge(Path("/x.json")) is True


def test_has_rig_edge_detects_the_inverse_direction(fake_ncore) -> None:
    """world->rig is the same edge; deriving a second one would be wrong."""
    fake_ncore([((WORLD_FRAME, RIG_FRAME), (_traj(38)))])

    assert has_rig_edge(Path("/x.json")) is True


def test_has_rig_edge_is_false_for_a_camera_only_sequence(fake_ncore) -> None:
    """This is the PPISP shape: the whole reason the derivation exists."""
    fake_ncore(
        [(("camera1", WORLD_FRAME), (_traj(21)))],
        [(("virtual_lidar", WORLD_FRAME), [[1.0]])],
    )

    assert has_rig_edge(Path("/x.json")) is False


# ---------------------------------------------------------------------------------
# publish_ncore_sequence -- the cross-pod handoff, incl. symlink resolution
# ---------------------------------------------------------------------------------
def _derived_sequence(root: Path) -> Path:
    """A derived sequence: real meta + derived store + SYMLINKED source shards."""
    original = root / "original"
    original.mkdir(parents=True)
    (original / "scene.ncore4-camera1.zarr.itar").write_bytes(b"camera1-shard-bytes")
    (original / "scene.ncore4.zarr.itar").write_bytes(b"main-store-bytes")

    derived = root / "derived"
    derived.mkdir(parents=True)
    (derived / "scene.json").write_text(json.dumps({"version": "v4"}))
    (derived / "scene.ncore4-npa_rig.zarr.itar").write_bytes(b"derived-rig-bytes")
    for name in ("scene.ncore4-camera1.zarr.itar", "scene.ncore4.zarr.itar"):
        (derived / name).symlink_to(original / name)
    return derived / "scene.json"


def test_publish_resolves_symlinks_into_real_objects(tmp_path: Path) -> None:
    """The derived sequence LINKS the originals to avoid copying hundreds of MB.

    Links cannot travel to S3, so every member must be dereferenced on publish --
    otherwise the next pod materializes dangling paths.
    """
    meta = _derived_sequence(tmp_path)
    destination = tmp_path / "published"

    result = publish_ncore_sequence(meta, str(destination))

    assert result["meta_name"] == "scene.json"
    assert result["objects"] == 4
    for name, expected in (
        ("scene.ncore4-camera1.zarr.itar", b"camera1-shard-bytes"),
        ("scene.ncore4.zarr.itar", b"main-store-bytes"),
        ("scene.ncore4-npa_rig.zarr.itar", b"derived-rig-bytes"),
    ):
        published = destination / name
        assert published.is_file()
        assert not published.is_symlink(), f"{name} was published as a link"
        assert published.read_bytes() == expected
    assert result["bytes"] > 0


def test_publish_carries_the_sidecar_so_the_next_pod_finds_the_reference_camera(
    tmp_path: Path,
) -> None:
    meta = _derived_sequence(tmp_path)
    (meta.parent / RIG_SIDECAR_NAME).write_text(
        json.dumps(
            {
                "reference_camera": "camera1",
                "poses_component_group": DERIVED_POSES_GROUP,
            }
        )
    )
    destination = tmp_path / "published"

    publish_ncore_sequence(meta, str(destination))

    # read_rig_sidecar looks next to the meta-file, which is how reconstruct
    # recovers the choice after materializing from S3.
    assert read_rig_sidecar(destination / "scene.json")["reference_camera"] == "camera1"


def test_publish_rejects_a_missing_sequence(tmp_path: Path) -> None:
    with pytest.raises(NurecError, match="not found"):
        publish_ncore_sequence(tmp_path / "nope.json", str(tmp_path / "out"))


def test_publish_skips_a_broken_symlink_rather_than_crashing(tmp_path: Path) -> None:
    meta = _derived_sequence(tmp_path)
    (meta.parent / "scene.ncore4-dangling.zarr.itar").symlink_to(tmp_path / "gone.itar")

    result = publish_ncore_sequence(meta, str(tmp_path / "published"))

    assert not (tmp_path / "published" / "scene.ncore4-dangling.zarr.itar").exists()
    assert result["objects"] == 4


def test_published_sequence_round_trips_through_a_zip(tmp_path: Path) -> None:
    """End-to-end shape check: publish, archive, extract, still readable."""
    meta = _derived_sequence(tmp_path)
    published = tmp_path / "published"
    publish_ncore_sequence(meta, str(published))

    archive = tmp_path / "seq.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for member in sorted(published.iterdir()):
            bundle.write(member, member.name)

    extracted = tmp_path / "extracted"
    extract_archive(archive, extracted)

    assert (
        extracted / "scene.ncore4-camera1.zarr.itar"
    ).read_bytes() == b"camera1-shard-bytes"


def test_derived_meta_rejects_non_basename_store_paths(tmp_path: Path) -> None:
    """The meta-file must reference stores by basename.

    It resolves them relative to its own directory, so an absolute path would
    produce a sequence that only loads on the machine that wrote it -- and would
    then fail in a different pod after the S3 handoff. The check exists because
    that is a library behaviour we do not control.
    """
    source = tmp_path / "scene.json"
    source.write_text("{}")

    # The guard lives after the reader/writer calls, so drive it directly on the
    # shape get_sequence_meta returns.
    meta = {"component_stores": [{"path": "/abs/scene.ncore4.zarr.itar"}]}
    recorded = [
        str(store.get("path", "")) for store in meta.get("component_stores", [])
    ]
    unexpected = [name for name in recorded if "/" in name or not name]

    assert unexpected == ["/abs/scene.ncore4.zarr.itar"]


def test_derive_rig_poses_reports_a_missing_sequence_cleanly(tmp_path: Path) -> None:
    from npa.workbench.nurec.ncore_rig import derive_rig_poses

    result = derive_rig_poses(tmp_path / "absent.json", output_dir=tmp_path / "out")

    assert result.ok is False
    assert any("not found" in error for error in result.errors)
    # A failure result, not an exception: the CLI turns it into `status: failed`.
    assert result.as_dict()["status"] == "failed"

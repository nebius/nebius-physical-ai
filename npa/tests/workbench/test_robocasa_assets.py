"""Exercise actual ZIP extraction and MP4 encoding without a simulation runtime."""

from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

from npa.workbench.robocasa.asset_archives import extract_asset_archive
from npa.workbench.robocasa.capabilities import RoboCasaError, _write_video


def _archive(path: Path, entries: list[tuple[str | ZipInfo, bytes]]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return path


def test_regular_kitchen_archive_preserves_nested_bytes(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "assets.zip", [("fixture/mesh.obj", b"real asset bytes")])
    destination = tmp_path / "assets"
    extract_asset_archive(archive, destination)
    assert (destination / "fixture/mesh.obj").read_bytes() == b"real asset bytes"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "fixture/../../escape", "a\\b", "."])
def test_unsafe_archive_path_refuses_before_any_payload_write(tmp_path: Path, name: str) -> None:
    archive = _archive(tmp_path / "assets.zip", [("safe.txt", b"safe"), (name, b"bad")])
    destination = tmp_path / "assets"
    with pytest.raises(ValueError):
        extract_asset_archive(archive, destination)
    assert not destination.exists()


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_archive_link_and_special_files_are_rejected(tmp_path: Path, kind: int) -> None:
    member = ZipInfo("fixture")
    member.create_system = 3
    member.external_attr = (kind | 0o777) << 16
    archive = _archive(tmp_path / "assets.zip", [(member, b"../elsewhere")])
    with pytest.raises(ValueError, match="link or special"):
        extract_asset_archive(archive, tmp_path / "assets")


@pytest.mark.parametrize("ancestor", [False, True])
def test_existing_destination_symlinks_cannot_redirect_assets(tmp_path: Path, ancestor: bool) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "assets"
    if ancestor:
        destination.symlink_to(outside, target_is_directory=True)
        destination /= "nested"
    else:
        destination.mkdir()
        (destination / "fixture").symlink_to(outside, target_is_directory=True)
    archive = _archive(tmp_path / "assets.zip", [("fixture/mesh.obj", b"bad")])
    with pytest.raises(ValueError, match="symbolic link"):
        extract_asset_archive(archive, destination)
    assert not list(outside.iterdir())


def test_duplicate_normalized_archive_paths_are_rejected(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "assets.zip", [("a/b", b"one"), ("a/./b", b"two")])
    with pytest.raises(ValueError, match="repeats"):
        extract_asset_archive(archive, tmp_path / "assets")


def test_video_round_trip_preserves_frame_count_dimensions_and_motion(tmp_path: Path) -> None:
    import av

    frames = [np.full((48, 64, 3), value * 20, dtype=np.uint8) for value in range(8)]
    path = _write_video(frames, tmp_path / "rollout.mp4")
    with av.open(str(path)) as recording:
        decoded = [frame.to_ndarray(format="rgb24") for frame in recording.decode(video=0)]
    assert len(decoded) == len(frames)
    assert all(frame.shape == (48, 64, 3) for frame in decoded)
    assert float(decoded[-1].mean() - decoded[0].mean()) > 100


@pytest.mark.parametrize("frames", [[], [np.zeros((16, 16))], [np.zeros((16, 16, 3))]])
def test_invalid_video_frames_do_not_report_success(tmp_path: Path, frames: list) -> None:
    with pytest.raises(RoboCasaError):
        _write_video(frames, tmp_path / "rollout.mp4")


def test_encoder_failure_is_a_capability_failure(tmp_path: Path, monkeypatch) -> None:
    import av

    def fail_open(*args, **kwargs):
        raise OSError("synthetic encoder failure")

    monkeypatch.setattr(av, "open", fail_open)
    with pytest.raises(RoboCasaError, match="video encoding failed"):
        _write_video([np.zeros((16, 16, 3), dtype=np.uint8)], tmp_path / "rollout.mp4")


def test_encoder_that_writes_nothing_cannot_report_success(tmp_path: Path, monkeypatch) -> None:
    import av
    from unittest.mock import MagicMock

    container = MagicMock()
    container.__enter__.return_value = container
    container.add_stream.return_value.encode.return_value = []
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    with pytest.raises(RoboCasaError, match="video encoding failed"):
        _write_video([np.zeros((16, 16, 3), dtype=np.uint8)], tmp_path / "rollout.mp4")

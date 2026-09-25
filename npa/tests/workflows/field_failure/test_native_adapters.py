"""Check native bundle containment and measured evaluation translation."""

from __future__ import annotations

import io
import tarfile

import pytest

from npa.workflows.field_failure.native_artifacts import _archive, _extract
from npa.workflows.field_failure.native_policy import _episode_metrics


@pytest.mark.parametrize(
    "name", ["../outside", "/outside", "a/../../outside", "a\\outside"]
)
def test_capture_archive_cannot_escape_its_directory(tmp_path, name):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="contained regular"):
        _extract(archive, tmp_path / "decoded")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE]
)
def test_capture_archive_rejects_non_regular_members(tmp_path, kind):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("capture.json")
        member.type = kind
        member.linkname = "../outside"
        stream.addfile(member)
    with pytest.raises(ValueError, match="contained regular"):
        _extract(archive, tmp_path / "decoded")


def test_capture_archive_rejects_duplicate_member_names(tmp_path):
    archive = tmp_path / "capture.tar"
    with tarfile.open(archive, "w") as stream:
        for content in (b"first", b"later"):
            member = tarfile.TarInfo("capture.json")
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError, match="unique contained"):
        _extract(archive, tmp_path / "decoded")


def test_native_bundle_preserves_nested_measured_bytes(tmp_path):
    source = tmp_path / "source"
    (source / "depth").mkdir(parents=True)
    (source / "depth/frame.png").write_bytes(b"measured depth bytes")
    (source / "capture.json").write_text('{"calibration":"pinned"}')
    archive = tmp_path / "capture.tar"
    _archive(source, archive)
    target = tmp_path / "decoded"
    _extract(archive, target)
    assert (target / "depth/frame.png").read_bytes() == b"measured depth bytes"
    assert (target / "capture.json").read_bytes() == (
        source / "capture.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "success,contacts,termination",
    [(True, 0, "success"), (False, 4, "failure"), (False, 0, "timeout")],
)
def test_measured_episode_keeps_failures_in_the_comparison(
    success, contacts, termination
):
    scene = {"scenario_id": "held-room", "asset": {"sha256": "a" * 64}}
    row = {
        "seed": 29,
        "steps": 180,
        "success": success,
        "collision_steps": contacts,
        "peer_collision_steps": 0,
        "path_length_m": 5.25,
        "goal_distance_m": 0.2 if success else 1.4,
    }
    record = _episode_metrics(
        scene, row, [{"name": "success"}, {"name": "path_length_m"}]
    )
    assert record["termination"] == termination
    assert record["status"] == "completed"
    assert record["metrics"] == {"success": float(success), "path_length_m": 5.25}
    assert record["scene_sha256"] == "a" * 64
    assert record["seed"] == 29

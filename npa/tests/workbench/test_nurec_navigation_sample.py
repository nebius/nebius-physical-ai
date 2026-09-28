"""Check sample integrity, public calibration, measured support and honest HTML results."""

import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from npa.workbench.nurec import navigation_sample as sample
from npa.workbench.nurec import navigation_sample_cases as cases
from npa.workbench.nurec import navigation_sample_report as report
from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_publication import publish_immutable
from npa.workflows.navigation.artifacts import publish, write_json


def _archive(tmp_path, members):
    archive = tmp_path / "sample.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = 1 if kind == tarfile.REGTYPE else 0
            member.linkname = "../escape" if kind == tarfile.SYMTYPE else ""
            handle.addfile(member, io.BytesIO(b"x") if member.size else None)
    return archive


def test_archive_mismatch_cannot_extract(tmp_path):
    archive = _archive(tmp_path, [(sample.SAMPLE_NAME + "/rgb.txt", tarfile.REGTYPE)])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sample._extract(archive, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        (sample.SAMPLE_NAME + "/../../escape", tarfile.REGTYPE),
        ("/absolute/escape", tarfile.REGTYPE),
        (sample.SAMPLE_NAME + "/symbolic", tarfile.SYMTYPE),
        (sample.SAMPLE_NAME + "/hardlink", tarfile.LNKTYPE),
    ],
)
def test_archive_rejects_unsafe_members(tmp_path, monkeypatch, name, kind):
    archive = _archive(tmp_path, [(name, kind)])
    monkeypatch.setattr(sample, "ARCHIVE_SHA256", sha256(archive))
    with pytest.raises(ValueError, match="escaping|regular"):
        sample._extract(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_archive_rejects_duplicate_paths(tmp_path, monkeypatch):
    member = (sample.SAMPLE_NAME + "/rgb.txt", tarfile.REGTYPE)
    archive = _archive(tmp_path, [member, member])
    monkeypatch.setattr(sample, "ARCHIVE_SHA256", sha256(archive))
    with pytest.raises(ValueError, match="unique"):
        sample._extract(archive, tmp_path / "output")


def test_pose_interpolation_and_timestamp_gaps():
    poses = np.array([[0, 0, 0, 1, 0, 0, 0, 1], [0.04, 2, 0, 1, 0, 0, 1, 0]])
    rotations = Slerp(poses[:, 0], Rotation.from_quat(poses[:, 4:8]))
    matrix = np.array(sample._pose(0.02, poses, rotations))
    np.testing.assert_allclose(matrix[:3, 3], [1, 0, 1])
    np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert sample._association(0.02, poses, np.array([0.021]))[2]
    assert not sample._association(0.02, poses, np.array([0.05]))[2]
    poses[1, 0] = 0.10
    assert not sample._association(0.02, poses, np.array([0.021]))[2]


def test_original_indices_define_holdout_without_renumbering(tmp_path):
    for name in ("rgb.png", "depth.png"):
        (tmp_path / name).write_bytes(name.encode())
    times = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    rgb = [(str(value), "rgb.png") for value in times]
    depth = [(str(value), "depth.png") for value in times]
    poses = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0.04, 0, 0, 1, 0, 0, 0, 1],
            [0.08, 0, 0, 1, 0, 0, 0, 1],
        ]
    )
    frames, excluded = sample._frames(tmp_path, rgb, depth, poses)
    assert not excluded
    assert [row["id"] for row in frames if row["split"] == "validation"] == [
        "frame_000000",
        "frame_000005",
    ]
    assert all(row["rgb_sha256"] == sha256(tmp_path / "rgb.png") for row in frames)


def test_unknown_floor_does_not_become_supported(monkeypatch):
    monkeypatch.setattr(
        cases, "_rays", lambda scene, origins, directions: np.full(len(origins), np.inf)
    )
    supported, _ = cases._support(None, np.array([[0.0, 0.0]]))
    assert not supported.any()
    with pytest.raises(ValueError, match="connected support"):
        cases._free_grid(None)


def test_actual_rotated_reset_footprint_is_checked(monkeypatch):
    observations = []

    def rays(scene, origins, directions):
        observations.extend(origins)
        return np.full(len(origins), 0.6)

    monkeypatch.setattr(cases, "_rays", rays)
    pair = ({"xy_m": [0, 0], "floor_z_m": 0}, {"xy_m": [0, 2]})
    assert cases._supported_pairs(None, [pair]) == [pair]
    np.testing.assert_allclose(observations[1], [0.23, -0.35, 0.6], atol=1e-12)


def _published_reports(tmp_path, passed):
    scan, trained, scored = (tmp_path / name for name in ("scan", "trained", "scored"))
    for root in (scan, trained, scored):
        root.mkdir()
    write_json(
        scan / "reconstruction.json",
        {"integration_frames": 1984, "validation_frames": 501},
    )
    write_json(scan / "capture.json", {"public": "test-fixture"})
    write_json(
        trained / "training.json", {"iterations": 500, "checkpoint_sha256": "a" * 64}
    )
    write_json(
        scored / "evaluation.json",
        {
            "checkpoint_sha256": "a" * 64,
            "success_rate": 0.9 if passed else 0.0,
            "passed": passed,
            "episodes": [{"success": passed}],
        },
    )
    destinations = [
        tmp_path / name
        for name in ("scan-published", "train-published", "eval-published")
    ]
    publish_immutable(scan, str(destinations[0]))
    publish(trained, str(destinations[1]))
    publish(scored, str(destinations[2]))
    return [str(path) for path in destinations]


@pytest.mark.parametrize("passed", [True, False])
def test_report_uses_actual_evaluation_and_preserves_failure(tmp_path, passed):
    scan, training, evaluation = _published_reports(tmp_path, passed)
    output = tmp_path / "report"
    value = report.build_report(scan, evaluation, training, str(output))
    assert value["passed"] is passed
    assert value["success_rate"] == (0.9 if passed else 0.0)
    page = (output / "index.html").read_text()
    assert (
        "<strong>Passed</strong>" if passed else "<strong>Did not pass</strong>"
    ) in page
    assert "unseen buildings" in page
    assert (output / "evaluation.json").is_file()


def test_report_refuses_a_different_checkpoint(tmp_path):
    scan, training, evaluation = _published_reports(tmp_path, True)
    value = json.loads((Path(evaluation) / "evaluation.json").read_text())
    value["checkpoint_sha256"] = "b" * 64
    write_json(Path(evaluation) / "evaluation.json", value)
    with pytest.raises(ValueError, match="checksum"):
        report.build_report(scan, evaluation, training, str(tmp_path / "report"))

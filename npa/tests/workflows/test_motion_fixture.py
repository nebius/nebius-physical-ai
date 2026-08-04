"""Unit coverage for the SOMA-CSV motion fixture (no infrastructure, no numpy).

The assertions encode NVIDIA's upstream ``load_csv_motion`` contract, which is what the
retargeting tool ultimately feeds:

* ``joint_pos.csv`` -> ``(T, 29)`` after ``skiprows=1``;
* ``body_pos.csv``  -> ``(T, B*3)``, reshaped to ``(T, B, 3)``; body 0 is the pelvis;
* ``body_quat.csv`` -> ``(T, B*4)`` in ``wxyz``; body 0's quaternion is the root
  rotation, so it must be a **unit** quaternion (``scipy.Rotation.from_quat`` on a zero
  quaternion is meaningless).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from npa.workflows.motion_fixture import (
    CLIP_FILES,
    FIXTURE_SCHEMA,
    NUM_BODIES,
    NUM_DOF,
    MotionFixtureError,
    build_and_publish,
    build_clip,
    build_dataset,
    split_s3_prefix,
    upload_dataset,
)


def _rows(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)  # the converter uses skiprows=1
        return [[float(cell) for cell in row] for row in reader]


def test_clip_matches_the_upstream_loader_shapes(tmp_path: Path) -> None:
    meta = build_clip(tmp_path / "walk", frames=12)

    assert meta["frames"] == 12
    assert sorted(meta["files"]) == sorted(CLIP_FILES)
    joint = _rows(tmp_path / "walk" / "joint_pos.csv")
    body_pos = _rows(tmp_path / "walk" / "body_pos.csv")
    body_quat = _rows(tmp_path / "walk" / "body_quat.csv")

    assert len(joint) == len(body_pos) == len(body_quat) == 12
    assert all(len(row) == NUM_DOF for row in joint)
    assert all(len(row) == NUM_BODIES * 3 for row in body_pos)
    assert all(len(row) == NUM_BODIES * 4 for row in body_quat)


def test_root_quaternions_are_unit_length(tmp_path: Path) -> None:
    build_clip(tmp_path / "walk", frames=16)

    for row in _rows(tmp_path / "walk" / "body_quat.csv"):
        w, x, y, z = row[:4]  # body 0 = pelvis = root rotation
        assert math.isclose(math.sqrt(w * w + x * x + y * y + z * z), 1.0, abs_tol=1e-4)


def test_pelvis_actually_moves_and_stays_upright(tmp_path: Path) -> None:
    """A static clip would make the converter's rotation maths meaningless."""

    build_clip(tmp_path / "walk", frames=20, stride=0.9)
    rows = _rows(tmp_path / "walk" / "body_pos.csv")

    xs = [row[0] for row in rows]
    zs = [row[2] for row in rows]
    assert xs[-1] > xs[0], "the pelvis should advance along +X"
    assert xs == sorted(xs), "forward motion should be monotonic"
    assert len(set(zs)) == 1, "pelvis height should be constant for a walk clip"


def test_joint_angles_stay_inside_a_safe_envelope(tmp_path: Path) -> None:
    build_clip(tmp_path / "walk", frames=24, amplitude=0.25)

    values = [value for row in _rows(tmp_path / "walk" / "joint_pos.csv") for value in row]
    assert max(abs(value) for value in values) <= 0.25 + 1e-6
    # ...and are not all the same number.
    assert len(set(round(value, 4) for value in values)) > 10


def test_dataset_writes_multiple_distinct_clips(tmp_path: Path) -> None:
    meta = build_dataset(tmp_path, clips=("walk-forward", "stand-sway"), frames=10)

    assert meta["schema"] == FIXTURE_SCHEMA
    assert meta["clip_count"] == 2
    for name in ("walk-forward", "stand-sway"):
        for filename in CLIP_FILES:
            assert (tmp_path / name / filename).is_file()
    walk = _rows(tmp_path / "walk-forward" / "body_pos.csv")
    sway = _rows(tmp_path / "stand-sway" / "body_pos.csv")
    assert walk[-1][0] > sway[-1][0], "the sway clip should barely translate"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frames": 1}, "frames must be"),
        ({"amplitude": 0.0}, "amplitude must be"),
    ],
)
def test_build_clip_rejects_degenerate_input(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    with pytest.raises(MotionFixtureError, match=match):
        build_clip(tmp_path / "bad", **kwargs)


def test_build_dataset_requires_a_clip_name(tmp_path: Path) -> None:
    with pytest.raises(MotionFixtureError, match="at least one clip"):
        build_dataset(tmp_path, clips=())


@pytest.mark.parametrize("uri", ["", "bucket/prefix", "https://example.invalid/x"])
def test_split_s3_prefix_rejects_bad_input(uri: str) -> None:
    with pytest.raises(MotionFixtureError):
        split_s3_prefix(uri)


def test_split_s3_prefix_normalises_the_trailing_slash() -> None:
    assert split_s3_prefix("s3://bucket/motion") == ("bucket", "motion/")
    assert split_s3_prefix("s3://bucket/motion/") == ("bucket", "motion/")
    assert split_s3_prefix("s3://bucket") == ("bucket", "")


class _FakeS3:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        self.uploads.append((bucket, key))


def test_upload_preserves_the_clip_directory_layout(tmp_path: Path) -> None:
    build_dataset(tmp_path, clips=("walk-forward",), frames=4)
    client = _FakeS3()

    uploaded = upload_dataset(tmp_path, "s3://bucket/motion", client=client)

    keys = sorted(key for _, key in client.uploads)
    assert keys == [
        "motion/walk-forward/body_pos.csv",
        "motion/walk-forward/body_quat.csv",
        "motion/walk-forward/joint_pos.csv",
    ]
    assert len(uploaded) == 3


def test_upload_of_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MotionFixtureError, match="nothing to upload"):
        upload_dataset(tmp_path, "s3://bucket/motion", client=_FakeS3())


def test_build_and_publish_reports_the_uri(tmp_path: Path) -> None:
    client = _FakeS3()

    result = build_and_publish(
        output_dir=tmp_path,
        uri="s3://bucket/motion/",
        clips=("walk-forward",),
        frames=4,
        client=client,
    )

    assert result["uri"] == "s3://bucket/motion/"
    assert result["uploaded"] == 3
    assert client.uploads, "publishing must go through the injected client, not real S3"


def test_build_and_publish_without_a_uri_stays_local(tmp_path: Path) -> None:
    result = build_and_publish(output_dir=tmp_path, clips=("walk-forward",), frames=4)

    assert "uri" not in result and "uploaded" not in result

"""Unit tests for `npa.workflows.isaac_capture`.

Isaac Sim itself cannot run here, so these cover the parts that broke live and are checkable
without a simulator: what gets published, and when.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows import isaac_capture


def test_simulation_app_close_cannot_swallow_capture_failure() -> None:
    closed: list[bool] = []

    class FakeApp:
        def close(self) -> None:
            closed.append(True)

    with pytest.raises(RuntimeError, match="render failed"):
        with isaac_capture._simulation_app_lifecycle(FakeApp()):
            raise RuntimeError("render failed")

    assert closed == []


def test_simulation_app_closes_after_success() -> None:
    closed: list[bool] = []

    class FakeApp:
        def close(self) -> None:
            closed.append(True)

    with isaac_capture._simulation_app_lifecycle(FakeApp()):
        pass

    assert closed == [True]


def test_publish_runs_before_the_simulator_closes(monkeypatch, tmp_path: Path) -> None:
    """Live job 278: six frames, exit 0, nothing uploaded.

    Isaac environment or app teardown can terminate the process rather than returning, so the
    upload that used to run after teardown never happened — and the next stage failed with
    "No scene images found". The publish callback must fire first.
    """

    order: list[str] = []

    class FakeApp:
        def close(self) -> None:
            order.append("close")

    def fake_capture(*, task, output_dir, publish=None, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_00.png").write_bytes(b"png")
        summary = {"status": "success", "frames": ["frame_00.png"], "task": task}
        if publish is not None:
            order.append("publish")
            publish(summary)
        FakeApp().close()
        return summary

    uploaded: dict[str, str] = {}
    monkeypatch.setattr(isaac_capture, "_capture_frames", fake_capture)
    monkeypatch.setattr(
        isaac_capture,
        "_upload_tree",
        lambda local, uri: uploaded.setdefault("uri", uri) and {} or {"frame_00.png": uri},
    )

    assert isaac_capture.main(["--output-path", "s3://bucket/scene/"]) == 0
    assert order == ["publish", "close"], "publish must happen before the app tears down"
    assert uploaded["uri"] == "s3://bucket/scene/"


def test_upload_carries_the_summary_next_to_the_frames(monkeypatch, tmp_path: Path) -> None:
    """Uploading only *.png stranded isaac_capture_summary.json in the pod."""

    (tmp_path / "frame_00.png").write_bytes(b"png")
    (tmp_path / "isaac_capture_summary.json").write_text(json.dumps({"status": "success"}))

    sent: list[str] = []

    class FakeS3:
        def upload_file(self, local: str, bucket: str, key: str) -> None:
            sent.append(key)

    class FakeBoto:
        @staticmethod
        def client(*_args, **_kwargs):
            return FakeS3()

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto)

    isaac_capture._upload_tree(tmp_path, "s3://bucket/run/scene/")

    assert sorted(sent) == [
        "run/scene/frame_00.png",
        "run/scene/isaac_capture_summary.json",
    ]


def test_render_only_reports_the_resolved_settings_without_a_simulator(capsys) -> None:
    assert isaac_capture.main(["--output-path", "s3://bucket/scene/", "--render-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "Isaac-Lift-Cube-Franka-v0"
    assert payload["output_path"] == "s3://bucket/scene/"


def test_upload_rejects_a_non_s3_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        isaac_capture._upload_tree(tmp_path, "gs://bucket/scene/")


# ------------------------------------------------------------------ camera framing


def test_look_at_quaternion_actually_points_at_the_target() -> None:
    """Live job 280 rendered six perfect photographs of bare floor.

    The template borrowed a camera pose tuned for a different scene, so the capture succeeded
    and the reasoner honestly reported "a tiled floor ... no visible objects". Framing is the
    one part of this stage that can be checked without a simulator, so it is checked here:
    rotate the camera's own +X axis by the returned quaternion and it must point from the eye
    towards the target. Job 281 then aimed 90 degrees off because the frame was assumed to be
    OpenGL's -Z-forward instead of Isaac's world/REP-103 +X-forward, and returned six pictures
    of the ground receding to a horizon.
    """

    import math

    def rotate(q, v):
        w, x, y, z = q
        # v' = q * v * q^-1, written out.
        tx = 2.0 * (y * v[2] - z * v[1])
        ty = 2.0 * (z * v[0] - x * v[2])
        tz = 2.0 * (x * v[1] - y * v[0])
        return (
            v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx),
        )

    for eye, target in (
        (isaac_capture.DEFAULT_CAMERA_EYE, isaac_capture.DEFAULT_CAMERA_TARGET),
        ((2.0, 0.0, 0.5), (0.0, 0.0, 0.5)),
        ((0.0, -1.5, 1.5), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0)),  # straight down: the degenerate up-vector case
    ):
        quat = isaac_capture.look_at_quaternion(eye, target)
        assert math.isclose(sum(c * c for c in quat), 1.0, rel_tol=1e-6), "not a unit quaternion"

        # Isaac Lab's world convention is REP-103: the camera looks along its own +X.
        forward = rotate(quat, (1.0, 0.0, 0.0))
        wanted = [target[i] - eye[i] for i in range(3)]
        length = math.sqrt(sum(c * c for c in wanted))
        wanted = [c / length for c in wanted]
        dot = sum(forward[i] * wanted[i] for i in range(3))
        assert dot > 0.999, f"camera looks {dot:.4f} away from {target} when placed at {eye}"


def test_the_camera_name_matches_what_the_frame_extractor_looks_up() -> None:
    """Owning the pose must not cost us the extraction.

    `_isaac_extract_rgb_frame` finds the camera by scene key; renaming ours would make every
    capture return `None` and the stage would fail with "No frames captured" for a reason that
    has nothing to do with rendering.
    """

    from npa.workflows.sim2real.engine import HELDOUT_VIZ_CAMERA_NAME

    assert isaac_capture.CAPTURE_CAMERA_NAME == HELDOUT_VIZ_CAMERA_NAME


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", (1.6, 1.6, 1.3)),
        ("1,2,3", (1.0, 2.0, 3.0)),
        (" 0.5 , -1 , 2.25 ", (0.5, -1.0, 2.25)),
    ],
)
def test_parse_point(raw: str, expected: tuple[float, float, float]) -> None:
    assert isaac_capture.parse_point(raw, isaac_capture.DEFAULT_CAMERA_EYE) == expected


@pytest.mark.parametrize("raw", ["1,2", "a,b,c", "1,2,3,4"])
def test_parse_point_rejects_nonsense(raw: str) -> None:
    with pytest.raises(SystemExit):
        isaac_capture.parse_point(raw, isaac_capture.DEFAULT_CAMERA_EYE)

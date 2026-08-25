from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import npa.workflows.sim2real_viz as viz_module
from npa.workflows.sim2real_loop import generate_action_rollouts
from npa.workflows.sim2real_viz import (
    RerunUnavailableError,
    Sim2RealVizError,
    emit_sim2real_rerun,
    is_reference_stub_rollout,
)


class _FakeRecording:
    pass


class _FakeRerun:
    """In-memory Rerun sink that records every logged entity for assertions."""

    def __init__(self) -> None:
        self.logged: list[tuple[str, str]] = []
        self.logged_times: list[tuple[str, str, float]] = []
        self.times: list[float] = []
        self.current_time = 0.0
        self.saved_path: Path | None = None
        self.disconnected = False

    # Archetype factories ---------------------------------------------------
    def Scalars(self, value: float) -> dict[str, Any]:
        return {"kind": "scalar", "value": float(value)}

    def Image(self, array: Any) -> dict[str, Any]:
        return {"kind": "image", "shape": getattr(array, "shape", None)}

    def TextDocument(self, text: str, media_type: str = "") -> dict[str, Any]:
        return {"kind": "text", "text": text}

    def Boxes3D(self, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "boxes3d", **kwargs}

    def Points3D(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "points3d", "args": args, **kwargs}

    def LineStrips3D(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "lines3d", "args": args, **kwargs}

    # Recording lifecycle ---------------------------------------------------
    def RecordingStream(self, application_id: str) -> _FakeRecording:
        self.application_id = application_id
        return _FakeRecording()

    def save(
        self, path: Any, default_blueprint: Any = None, recording: Any = None
    ) -> None:
        self.saved_path = Path(path)
        Path(path).write_bytes(b"FAKE_RRD_CONTENT")

    def send_blueprint(self, blueprint: Any, recording: Any = None) -> None:
        return None

    def set_time_seconds(
        self, timeline: str, seconds: float, recording: Any = None
    ) -> None:
        self.current_time = float(seconds)
        self.times.append(float(seconds))

    def log(
        self, entity_path: str, archetype: dict[str, Any], recording: Any = None
    ) -> None:
        kind = archetype.get("kind", "?")
        self.logged.append((entity_path, kind))
        self.logged_times.append((entity_path, kind, self.current_time))

    def disconnect(self, recording: Any = None) -> None:
        self.disconnected = True


def _build_run_tree(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    actions_dir = tmp_path / "actions" / "train" / "outer-01" / "iter-01"
    rollouts = generate_action_rollouts(
        actions_dir, count=2, steps_per_rollout=3, seed=11, quality=0.4
    )
    eval_dir = tmp_path / "vlm_eval" / "train" / "outer-01" / "iter-01"
    signal_dir = tmp_path / "training_signal" / "train" / "outer-01" / "iter-01"
    eval_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)
    for rollout in rollouts:
        rollout_id = rollout.name
        per_step = [
            {
                "step": step,
                "critique_text": f"{rollout_id} step {step} drifted",
                "error_tags": ["minor_alignment"],
            }
            for step in range(3)
        ]
        (eval_dir / f"{rollout_id}.json").write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.vlm_eval.v1",
                    "rollout_id": rollout_id,
                    "success": False,
                    "score": 0.6,
                    "per_step": per_step,
                    "summary": "summary",
                }
            ),
            encoding="utf-8",
        )
        (signal_dir / f"{rollout_id}.json").write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.rl_signal.v1",
                    "rollout_id": rollout_id,
                    "per_step": [
                        {"step": step, "reward": 0.1 * step, "advantage": 0.05 * step}
                        for step in range(3)
                    ],
                }
            ),
            encoding="utf-8",
        )
    inner_evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": 1,
        "reward_trend": [0.2, 0.45],
        "iterations": [
            {
                "iteration": 1,
                "actions_dir": str(actions_dir),
                "vlm_eval_dir": str(eval_dir),
                "signal_dir": str(signal_dir),
                "mean_reward": 0.2,
                "update": {"loss_before": 1.0, "loss_after": 0.7},
                "policy_delta_vs_control": 0.1,
                "next_rollout_quality": 0.55,
            }
        ],
    }
    heldout_report = {
        "schema": "npa.sim2real.heldout_eval.v1",
        "success_rate": 0.5,
        "per_env": [
            {"env_id": "heldout-0000", "score": 0.7, "success": True},
            {"env_id": "heldout-0001", "score": 0.5, "success": False},
        ],
    }
    return inner_evidence, heldout_report


def test_emit_logs_frames_critiques_signal_and_heldout(
    monkeypatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))
    stage_components = [
        {
            "name": name,
            "tier": "SEAM" if stage == 12 else "WORKS",
            "evidence": f"stage {stage} evidence",
            "artifacts": {
                "job_name": f"s2r-stage-{stage:02d}"
                if stage in {3, 4, 7, 8, 9, 10}
                else "",
                "gpu_request": {"product": "NVIDIA-RTX-PRO-6000"}
                if stage in {3, 4, 7, 8, 9, 10, 14}
                else {},
                "remote": f"s3://bucket/run/stage-{stage:02d}.json",
            },
        }
        for stage, name in viz_module._CANONICAL_STAGE_COMPONENTS.items()
    ]

    rrd_path = tmp_path / "reports" / "sim2real.rrd"
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        stage_components=stage_components,
        outer_history=[
            {
                "checkpoint_uri": "s3://bucket/run/model_latest.pt",
                "resumed_from": "",
                "decision": {"decision": "promote_checkpoint", "success_rate": 1.0},
            }
        ],
        run_metadata={
            "run_id": "run",
            "policy_checkpoint": "s3://bucket/run/model_latest.pt",
            "candidate_s3_uri": "s3://bucket/run/checkpoints/candidate/candidate.json",
            "rrd_s3_uri": "s3://bucket/run/reports/sim2real.rrd",
            "artifact_root": "s3://bucket/run/",
            "viewer_command": "npa workbench sim2real rerun serve --run-id run",
            "orchestrator_job_name": "run",
            "orchestrator_node_product": "NVIDIA-RTX-PRO-6000",
        },
        output_rrd=rrd_path,
    )

    assert result.status == "written"
    assert rrd_path.exists() and rrd_path.stat().st_size > 0
    assert result.rollout_count == 2
    assert result.frame_count == 6
    assert result.heldout_env_count == 2
    assert fake.disconnected is True

    entities = [entity for entity, _kind in fake.logged]
    kinds = {entity: kind for entity, kind in fake.logged}
    # Rollout camera frames as image streams.
    assert any(e.endswith("/camera") and kinds[e] == "image" for e in entities)
    # 3D scene overview is the primary visual context.
    assert "world/table" in entities
    assert "world/cube" in entities
    assert "world/franka/joints" in entities
    assert "world/franka/links" in entities
    # VLM critique overlays.
    assert any(e.endswith("/critique") and kinds[e] == "text" for e in entities)
    assert "rollouts/summary/critique" in entities
    # RL signal scalar timeseries.
    assert "signal/reward" in entities
    assert "signal/advantage" in entities
    assert "signal/reward_trend" in entities
    assert "training/loss_before" in entities
    assert "training/loss_after" in entities
    assert "progress/inner_loop/iteration" in entities
    # Action trajectories per rollout step.
    assert any("/actions/dim_00" in e for e in entities)
    assert any(e.endswith("/actions/l2_norm") for e in entities)
    # Held-out scores.
    assert "heldout/success_rate" in entities
    assert "heldout/scores" in entities
    assert any(e.startswith("heldout/per_env/") for e in entities)
    # Full stage/tier/Job/GPU proof and deployable-policy access are first-class
    # viewer panels, with stage/outer-loop progress on the recording timeline.
    assert "summary/stage_progress" in entities
    assert "summary/policy_access" in entities
    assert "progress/stage_01/tier_works" in entities
    assert "progress/stage_14/evidence" in entities
    assert "progress/outer_loop/iteration" in entities
    assert "progress/outer_loop/decision" in entities

    counts = result.entity_counts
    assert counts["/signal/reward"] == 6
    assert counts["/world/franka/joints"] >= 1
    assert counts["/rollouts/iter_01/rollout-0000/actions/dim_00"] == 3
    assert counts["/heldout/scores"] == 2
    assert counts["/heldout/success_rate"] == 1
    assert counts["/summary/stage_progress"] == 1
    assert counts["/summary/policy_access"] == 1
    assert counts["/progress/stage_12/tier_works"] == 1


def test_progress_only_recording_is_allowed_with_stage_proof(
    monkeypatch, tmp_path: Path
) -> None:
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))

    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence={},
        heldout_report=None,
        stage_components=[
            {
                "name": "stage_01_trigger",
                "tier": "WORKS",
                "evidence": "started",
                "artifacts": {"duration_s": 0.1},
            }
        ],
        output_rrd=tmp_path / "reports" / "sim2real-progress.rrd",
        allow_progress_only=True,
    )

    assert result.status == "written"
    assert result.rollout_count == 0
    assert result.entity_counts["/progress/stage_01/tier_works"] == 1


def test_recording_loads_metrics_and_rollouts_from_every_outer_iteration(
    monkeypatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    for outer in (1, 2):
        payload = dict(inner_evidence)
        payload["outer_iteration"] = outer
        payload["iterations"] = [dict(inner_evidence["iterations"][0])]
        evidence_path = tmp_path / "inner_loop" / f"outer-{outer:02d}" / "evidence.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_rrd=tmp_path / "reports" / "sim2real.rrd",
    )

    entities = [entity for entity, _kind in fake.logged]
    assert result.rollout_count == 4
    assert any(entity.startswith("rollouts/outer_01/") for entity in entities)
    assert any(entity.startswith("rollouts/outer_02/") for entity in entities)
    assert result.entity_counts["/training/loss_after"] == 2


def test_emit_raises_when_rerun_unavailable(monkeypatch, tmp_path: Path) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)

    def _raise() -> Any:
        raise RerunUnavailableError("rerun-sdk is not installed")

    monkeypatch.setattr(viz_module, "_import_rerun", _raise)

    with pytest.raises(RerunUnavailableError):
        emit_sim2real_rerun(
            local_dir=tmp_path,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            output_rrd=tmp_path / "reports" / "sim2real.rrd",
        )


def test_emit_raises_when_no_content(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))

    with pytest.raises(Sim2RealVizError):
        emit_sim2real_rerun(
            local_dir=tmp_path,
            inner_evidence={"iterations": [], "reward_trend": []},
            heldout_report={"per_env": []},
            output_rrd=tmp_path / "reports" / "sim2real.rrd",
        )


def test_emit_mcap_roundtrip_camera_signal_critique(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    out = tmp_path / "reports" / "sim2real.mcap"
    result = viz_module.emit_sim2real_mcap(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=out,
    )

    assert result.status == "written"
    assert out.is_file() and out.stat().st_size > 0
    # 2 rollouts x 3 frames of raw .ppm camera dumps, all transcoded to PNG, plus
    # the first rollout's 3 frames mirrored onto the primary /camera topic (this run
    # has no held-out episodes, which would otherwise own that topic).
    assert result.camera_message_count == 9
    assert result.scalar_message_count > 0
    assert result.log_message_count > 0

    with open(out, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        topics = {channel.topic for channel in summary.channels.values()}
        schema_names = {schema.name for schema in summary.schemas.values()}
        first_camera = next(
            json.loads(message.data)
            for _schema, channel, message in reader.iter_messages()
            if channel.topic.endswith("/camera")
        )

    assert any(topic.endswith("/camera") for topic in topics)
    # The embedded viewer's default layout binds its Image panel to this one
    # well-known topic, so it must be populated even without held-out episodes.
    assert viz_module.MCAP_PRIMARY_CAMERA_TOPIC in topics
    assert "/signal/reward" in topics
    assert "/signal/advantage" in topics
    assert "/signal/reward_trend" in topics
    assert "/heldout/scores" in topics
    assert "/heldout/success_rate" in topics
    assert any(topic.endswith("/critique") for topic in topics)
    assert "foxglove.CompressedImage" in schema_names
    assert "foxglove.Log" in schema_names
    assert "npa.sim2real.Scalar" in schema_names
    # Raw .ppm rollout frames are transcoded to browser-decodable PNG.
    assert first_camera["format"] == "png"


def _set_first_rollout_capture_fps(tmp_path: Path, fps: float) -> None:
    manifest_path = (
        tmp_path
        / "actions"
        / "train"
        / "outer-01"
        / "iter-01"
        / "rollout-0000"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture"] = {"fps": fps, "width": 640, "height": 480}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_configured_capture_fps_controls_rrd_rollout_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    _set_first_rollout_capture_fps(tmp_path, 10.0)
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))
    emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_rrd=tmp_path / "reports" / "capture-fps.rrd",
    )
    times = [
        timestamp
        for entity, kind, timestamp in fake.logged_times
        if entity == "rollouts/iter_01/rollout-0000/camera" and kind == "image"
    ]
    assert times == pytest.approx([0.0, 0.1, 0.2])


def test_configured_capture_fps_controls_mcap_rollout_timestamps(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    _set_first_rollout_capture_fps(tmp_path, 10.0)
    out = tmp_path / "reports" / "capture-fps.mcap"
    viz_module.emit_sim2real_mcap(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=out,
    )
    with out.open("rb") as handle:
        messages = [
            message.log_time
            for _schema, channel, message in make_reader(handle).iter_messages()
            if channel.topic == "/rollouts/iter_01/rollout-0000/camera"
        ]
    assert messages == [0, 100_000_000, 200_000_000]


def _write_pointcloud_npz(
    tmp_path: Path,
    env_id: str = "env-0001",
    frames: int = 3,
    view: str | None = None,
) -> None:
    import numpy as np

    root = (
        tmp_path
        / "eval"
        / "heldout"
        / "renders"
        / viz_module.POINTCLOUD_SUBDIR
        / env_id
    )
    if view:
        root /= view
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(frames):
        xyz = rng.uniform(-0.5, 0.5, size=(500, 3)).astype("float32")
        rgb = rng.integers(0, 256, size=(500, 3), dtype="uint8")
        np.savez_compressed(root / f"cloud-{i:04d}.npz", xyz=xyz, rgb=rgb)


def test_emit_mcap_primary_camera_prefers_heldout_over_rollout_mirror(
    tmp_path: Path,
) -> None:
    """When held-out episodes exist they own the primary camera topic outright.

    The rollout fallback exists only for runs without held-out cameras; if both
    wrote to it the panel would interleave two unrelated, misaligned streams.
    """

    pytest.importorskip("mcap")

    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    renders_dir = tmp_path / "eval" / "heldout" / "renders" / "heldout-0000"
    _write_test_png(renders_dir / "camera-000.png", red=40, green=120, blue=200)
    _write_test_png(renders_dir / "camera-001.png", red=50, green=130, blue=210)
    heldout_report["render_manifest"] = {
        "schema": "npa.sim2real.heldout_renders.v1",
        "sim_backend": "isaac",
        "episodes": [
            {"env_id": "heldout-0000", "frames": ["camera-000.png", "camera-001.png"]}
        ],
    }

    result = viz_module.emit_sim2real_mcap(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=tmp_path / "reports" / "heldout.mcap",
    )

    assert result.channel_counts[viz_module.MCAP_PRIMARY_CAMERA_TOPIC] == 2


def test_pointcloud_message_packs_xyz_and_rgba() -> None:
    """The cloud must carry an opaque ``alpha`` channel alongside red/green/blue.

    The viewer only offers its ``rgba-fields`` colour mode when all four colour
    fields are declared. Without alpha that mode is unavailable and the 3D panel
    re-colours the cloud with a fallback colormap, losing the captured RGB.
    """

    import base64

    import numpy as np

    from npa.workbench.lichtblick import (
        _POINTCLOUD_ALPHA_OPAQUE,
        _POINTCLOUD_POINT_STRIDE,
        pointcloud_message,
    )

    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype="uint8")
    msg = pointcloud_message(xyz, rgb, stamp_ns=500_000_000, frame_id="sim2real")
    assert msg["point_stride"] == _POINTCLOUD_POINT_STRIDE
    assert [f["name"] for f in msg["fields"]] == [
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "alpha",
    ]
    # Declared offsets must match the packed layout the viewer will read.
    assert {f["name"]: f["offset"] for f in msg["fields"]}["alpha"] == 15
    raw = base64.b64decode(msg["data"])
    assert len(raw) == 2 * _POINTCLOUD_POINT_STRIDE
    # First point: xyz float32, then rgb uint8, then an opaque alpha.
    assert np.frombuffer(raw[0:12], dtype="<f4").tolist() == [1.0, 2.0, 3.0]
    assert list(raw[12:15]) == [10, 20, 30]
    assert raw[15] == _POINTCLOUD_ALPHA_OPAQUE
    # Every point is opaque, not just the first.
    assert list(raw[_POINTCLOUD_POINT_STRIDE + 12 : 2 * _POINTCLOUD_POINT_STRIDE]) == [
        40,
        50,
        60,
        _POINTCLOUD_ALPHA_OPAQUE,
    ]


def test_heldout_pointcloud_frames_reads_npz(tmp_path: Path) -> None:
    _write_pointcloud_npz(tmp_path, frames=3)
    frames = viz_module._heldout_pointcloud_frames(tmp_path)
    assert len(frames) == 3
    xyz, rgb = frames[0]
    assert xyz.shape[1] == 3 and rgb.shape[1] == 3
    assert xyz.dtype.name == "float32" and rgb.dtype.name == "uint8"


def test_heldout_pointcloud_frames_fuses_synchronized_camera_views(
    tmp_path: Path,
) -> None:
    _write_pointcloud_npz(tmp_path, frames=2, view="primary")
    _write_pointcloud_npz(tmp_path, frames=2, view="side")
    _write_pointcloud_npz(tmp_path, frames=2, view="overhead")
    frames = viz_module._heldout_pointcloud_frames(tmp_path)
    assert len(frames) == 2
    xyz, rgb = frames[0]
    assert xyz.shape == (1500, 3)
    assert rgb.shape == (1500, 3)


def test_emit_mcap_includes_pointclouds(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    heldout_report["policy_inference_provenance"] = {
        "checkpoint_uri": "s3://bucket/run/model_150.pt",
        "checkpoint_sha256": "d" * 64,
        "checkpoint_size_bytes": 98765,
        "loaded_for_inference": True,
        "stock_or_scripted_policy": False,
    }
    heldout_report["capture"] = {"width": 640, "height": 480, "fps": 10.0}
    _write_pointcloud_npz(tmp_path, frames=4)
    out = tmp_path / "reports" / "sim2real.mcap"
    result = viz_module.emit_sim2real_mcap(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=out,
    )
    assert result.pointcloud_message_count == 4
    # A coordinate transform must accompany the point cloud so a Foxglove-compatible
    # 3D panel has a defined frame to place it in (otherwise nothing renders).
    assert result.transform_message_count >= 1
    with open(out, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        topics = {channel.topic for channel in summary.channels.values()}
        schema_names = {schema.name for schema in summary.schemas.values()}
    assert "/heldout/points" in topics
    assert "foxglove.PointCloud" in schema_names
    assert "/tf" in topics
    assert "/provenance/heldout_policy" in topics
    assert "foxglove.FrameTransform" in schema_names


def test_emit_mcap_raises_when_mcap_unavailable(monkeypatch, tmp_path: Path) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)

    def _raise() -> Any:
        raise viz_module.McapUnavailableError("mcap is not installed")

    monkeypatch.setattr(viz_module, "_import_mcap", _raise)

    with pytest.raises(viz_module.McapUnavailableError):
        viz_module.emit_sim2real_mcap(
            local_dir=tmp_path,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            output_mcap=tmp_path / "reports" / "sim2real.mcap",
        )


def test_emit_mcap_raises_when_no_content(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    with pytest.raises(Sim2RealVizError):
        viz_module.emit_sim2real_mcap(
            local_dir=tmp_path,
            inner_evidence={"iterations": [], "reward_trend": []},
            heldout_report={"per_env": []},
            output_mcap=tmp_path / "reports" / "empty.mcap",
        )


def test_emit_rejects_synthetic_descriptor_only_recording(
    monkeypatch, tmp_path: Path
) -> None:
    _write_summary_artifacts(tmp_path)
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))

    with pytest.raises(Sim2RealVizError, match="no real rollout frames"):
        emit_sim2real_rerun(
            local_dir=tmp_path,
            inner_evidence={"iterations": [], "reward_trend": []},
            heldout_report={"per_env": []},
            output_rrd=tmp_path / "reports" / "sim2real.rrd",
        )

    assert any(entity.startswith("synthetic/") for entity, _kind in fake.logged)


def _write_test_png(path: Path, *, red: int, green: int, blue: int) -> None:
    import struct
    import zlib

    width = height = 64
    raw = bytearray()
    pixel = bytes([red, green, blue])
    for _row in range(height):
        raw.append(0)
        raw.extend(pixel * width)

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def test_real_augmentation_png_index_is_not_decoded_as_json(tmp_path: Path) -> None:
    frame = tmp_path / "augment" / "frames" / "frame-00000.png"
    _write_test_png(frame, red=12, green=34, blue=56)
    (frame.parent / "index.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": "frame-00000",
                        "uri": "s3://bucket/run/augment/frames/frame-00000.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "augment" / "manifest.json").write_text("{}", encoding="utf-8")

    samples = viz_module._augmentation_visual_samples(tmp_path)

    assert len(samples) == 1
    frame_id, payload, image = samples[0]
    assert frame_id == "frame-00000"
    assert payload["uri"].endswith("frame-00000.png")
    assert image is not None


def _write_summary_artifacts(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "sim2real-report.json").write_text(
        json.dumps({"run_id": "s2r-test", "status": "completed"}),
        encoding="utf-8",
    )
    (tmp_path / "outer_loop").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outer_loop" / "decision.json").write_text(
        json.dumps(
            {
                "decision": "promote_checkpoint",
                "success_rate": 1.0,
                "threshold": 0.5,
                "checkpoint_uri": "s3://bucket/run/model_latest.pt",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "augment" / "frames").mkdir(parents=True, exist_ok=True)
    (tmp_path / "augment" / "manifest.json").write_text(
        json.dumps(
            {
                "status": "executed",
                "mode": "descriptor_stub",
                "frame_count": 2,
                "image": "npa-cosmos2-transfer:test",
                "input_uri": "s3://bucket/input",
                "output_uri": "s3://bucket/augment",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "augment" / "frames" / "index.json").write_text(
        json.dumps({"frame_count": 2, "frames": [{"frame_id": "frame-00000"}]}),
        encoding="utf-8",
    )
    (tmp_path / "envs" / "manifest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "envs" / "manifest" / "split-manifest.json").write_text(
        json.dumps(
            {
                "raw_count": 10,
                "train_count": 8,
                "heldout_count": 2,
                "disjoint": True,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokens").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tokens" / "manifest.json").write_text(
        json.dumps({"train_env_count": 8, "heldout_env_count": 2}),
        encoding="utf-8",
    )
    env_sample = {
        "env_id": "env-00006",
        "physics": {"friction": 0.58, "lighting_lux": 700},
        "scene": {
            "simready_asset": "simready://warehouse/tabletop_v1",
            "augmented_frame_uri": "s3://bucket/augment/frame-00006.png",
        },
    }
    for rel in ("envs/train/envs.jsonl", "envs/heldout/envs.jsonl"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(env_sample) + "\n", encoding="utf-8")


def _gradient_rgb(width: int = 48, height: int = 32) -> Any:
    import numpy as np

    ys, xs = np.mgrid[0:height, 0:width]
    r = (xs * 255 // max(1, width - 1)).astype("uint8")
    g = (ys * 255 // max(1, height - 1)).astype("uint8")
    b = ((xs + ys) * 255 // max(1, width + height - 2)).astype("uint8")
    return np.dstack([r, g, b]).astype("uint8")


def test_decode_png_applies_row_filters(tmp_path: Path) -> None:
    """Regression: renders use Sub/Up/Paeth filters; ignoring them yields noise."""
    import io

    import numpy as np

    Image = pytest.importorskip("PIL.Image")

    arr = _gradient_rgb()
    # Pillow chooses adaptive per-row filters (not all-zero) for a gradient.
    buffer = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buffer, format="PNG")
    data = buffer.getvalue()

    # Confirm the encoder actually used non-zero filters, otherwise the test is moot.
    import struct
    import zlib

    idx = 8
    width = height = 0
    idat = bytearray()
    while idx + 8 <= len(data):
        length = struct.unpack("!I", data[idx : idx + 4])[0]
        ctype = data[idx + 4 : idx + 8]
        chunk = data[idx + 8 : idx + 8 + length]
        idx += 12 + length
        if ctype == b"IHDR":
            width, height = struct.unpack("!II", chunk[:8])
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    stride = width * 3 + 1
    filters = {raw[row * stride] for row in range(height)}
    assert filters - {0}, "expected Pillow to use non-zero row filters"

    # Manual fallback decoder must reconstruct the exact pixels (not noise).
    decoded = viz_module._decode_png_bytes(data)
    assert decoded is not None
    assert decoded.shape == arr.shape
    assert np.array_equal(decoded, arr)

    # Public reader (Pillow fast path) must also match.
    png_path = tmp_path / "gradient.png"
    png_path.write_bytes(data)
    via_reader = viz_module._read_png(png_path)
    assert via_reader is not None
    assert np.array_equal(via_reader, arr)


def _encode_png_with_filter(pixels, filter_type: int) -> bytes:
    """Encode an 8-bit RGB PNG forcing every scanline onto ``filter_type``.

    Pillow picks row filters adaptively, so it cannot be told to exercise a
    specific one. The fallback decoder has a separate branch per filter, and
    Average/Paeth are the hand-written recurrences, so each needs direct coverage.
    """

    import struct
    import zlib

    height, width, channels = pixels.shape
    assert channels == 3
    raw = bytearray()
    prev = [0] * (width * channels)
    for y in range(height):
        cur = [int(v) for v in pixels[y].reshape(-1)]
        line = bytearray()
        for i, value in enumerate(cur):
            left = cur[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 0:
                pred = 0
            elif filter_type == 1:
                pred = left
            elif filter_type == 2:
                pred = up
            elif filter_type == 3:
                pred = (left + up) // 2
            else:
                pred = viz_module._paeth(left, up, up_left)
            line.append((value - pred) & 0xFF)
        raw.append(filter_type)
        raw.extend(line)
        prev = cur

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _chunk(b"IEND", b"")
    )


def _write_filtered_rgb_png(path: Path, pixels: Any, filter_types: list[int]) -> None:
    import struct
    import zlib

    height, width, channels = pixels.shape
    assert channels == 3
    raw = bytearray()
    previous = [0] * (width * channels)
    for row_index in range(height):
        row = [int(value) for value in pixels[row_index].reshape(-1)]
        filter_type = filter_types[row_index % len(filter_types)]
        encoded = bytearray()
        for byte_index, value in enumerate(row):
            left = row[byte_index - channels] if byte_index >= channels else 0
            up = previous[byte_index]
            upper_left = (
                previous[byte_index - channels] if byte_index >= channels else 0
            )
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = viz_module._paeth(left, up, upper_left)
            else:
                raise AssertionError(filter_type)
            encoded.append((value - predictor) & 0xFF)
        raw.append(filter_type)
        raw.extend(encoded)
        previous = row

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _noisy_rgb(width: int = 24, height: int = 18):
    """Deterministic high-frequency image.

    A smooth gradient is a degenerate case for Paeth: with ``left`` and
    ``up_left`` equal along an axis-aligned ramp, the predictor picks ``up``
    whether or not ``up_left`` is read, so a decoder that ignored ``up_left``
    would still decode a gradient perfectly. Noise makes every term matter.
    """

    import numpy as np

    rng = np.random.default_rng(20260731)
    return rng.integers(0, 256, size=(height, width, 3), dtype="uint8")


@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("image", ["gradient", "noise"])
def test_decode_png_matches_pillow_for_every_row_filter(
    filter_type: int, image: str
) -> None:
    """Each filter branch must reconstruct exactly what Pillow does."""

    import io

    import numpy as np

    Image = pytest.importorskip("PIL.Image")

    expected = _gradient_rgb() if image == "gradient" else _noisy_rgb()
    data = _encode_png_with_filter(expected, filter_type)

    # Cross-check the hand-rolled encoder itself, so a bug there cannot make the
    # decoder look correct against a wrong reference.
    with Image.open(io.BytesIO(data)) as image:
        via_pillow = np.asarray(image.convert("RGB"), dtype="uint8")
    assert np.array_equal(via_pillow, expected), (
        f"filter {filter_type}: encoder is wrong"
    )

    decoded = viz_module._decode_png_bytes(data)
    assert decoded is not None
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, expected), (
        f"filter {filter_type}: fallback decode differs"
    )


def test_decode_png_matches_pillow_on_filter0(tmp_path: Path) -> None:
    import numpy as np

    _write_test_png(tmp_path / "solid.png", red=12, green=200, blue=77)
    decoded = viz_module._read_png(tmp_path / "solid.png")
    assert decoded is not None
    assert decoded.shape == (64, 64, 3)
    assert np.array_equal(decoded[0, 0], np.array([12, 200, 77], dtype="uint8"))
    assert int(decoded.mean()) == int(np.array([12, 200, 77]).mean())


def test_read_png_reconstructs_filtered_truecolor_rows(
    monkeypatch, tmp_path: Path
) -> None:
    import numpy as np

    monkeypatch.setattr(viz_module, "_read_png_with_pillow", lambda _path: None)
    pixels = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            [[11, 21, 31], [45, 55, 65], [76, 86, 96], [111, 121, 131]],
            [[12, 22, 32], [47, 57, 67], [81, 91, 101], [118, 128, 138]],
            [[13, 23, 33], [49, 59, 69], [88, 98, 108], [125, 135, 145]],
            [[14, 24, 34], [51, 61, 71], [95, 105, 115], [132, 142, 152]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "filtered.png"
    _write_filtered_rgb_png(path, pixels, [0, 1, 2, 3, 4])

    decoded = viz_module._read_png(path)

    assert decoded is not None
    assert np.array_equal(decoded, pixels)


def test_is_reference_stub_rollout_detects_reference_fixture(tmp_path: Path) -> None:
    actions_dir = tmp_path / "actions"
    rollouts = generate_action_rollouts(
        actions_dir, count=1, steps_per_rollout=2, seed=3, quality=0.5
    )
    frames = viz_module._rollout_frames(rollouts[0])
    assert is_reference_stub_rollout(rollouts[0], frames) is True

    rollout_dir = rollouts[0]
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.policy_rollout.v1",
                "rollout_id": rollout_dir.name,
                "camera_observations": ["camera-000.png"],
            }
        ),
        encoding="utf-8",
    )
    assert is_reference_stub_rollout(rollout_dir, frames) is False


def test_rollout_camera_frames_preserve_synchronized_named_views(
    tmp_path: Path,
) -> None:
    rollout_dir = tmp_path / "rollout-0000"
    views = {
        "primary": ["camera-000.png", "camera-001.png"],
        "side": ["camera-side-000.png", "camera-side-001.png"],
        "overhead": ["camera-overhead-000.png", "camera-overhead-001.png"],
    }
    for view_index, names in enumerate(views.values()):
        for frame_index, name in enumerate(names):
            _write_test_png(
                rollout_dir / name,
                red=20 + view_index * 40,
                green=60 + frame_index * 20,
                blue=160,
            )
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.action_rollout.v1",
                "camera_observations": views["primary"],
                "camera_views": views,
            }
        ),
        encoding="utf-8",
    )

    grouped = viz_module._rollout_camera_frames(rollout_dir)
    assert list(grouped) == ["primary", "side", "overhead"]
    assert all(len(frames) == 2 for frames in grouped.values())
    assert len(viz_module._rollout_frames(rollout_dir)) == 2


def test_emit_prefers_heldout_isaac_cameras_over_stub_rollouts(
    monkeypatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    _write_summary_artifacts(tmp_path)
    renders_dir = tmp_path / "eval" / "heldout" / "renders" / "heldout-0000"
    _write_test_png(renders_dir / "camera-000.png", red=40, green=120, blue=200)
    _write_test_png(renders_dir / "camera-001.png", red=50, green=130, blue=210)
    heldout_report["render_manifest"] = {
        "schema": "npa.sim2real.heldout_renders.v1",
        "sim_backend": "isaac",
        "isaac_task": "Isaac-Lift-Cube-Franka-v0",
        "episodes": [
            {
                "env_id": "heldout-0000",
                "frames": ["camera-000.png", "camera-001.png"],
            }
        ],
    }

    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_rrd=tmp_path / "reports" / "sim2real.rrd",
    )

    entities = [entity for entity, _kind in fake.logged]
    kinds = {entity: kind for entity, kind in fake.logged}
    assert result.heldout_frame_count == 2
    assert result.rollout_count == 0
    assert result.frame_count == 0
    assert "heldout/camera/heldout-0000/camera" in entities
    assert kinds["heldout/camera/heldout-0000/camera"] == "image"
    assert ("camera", "image", 0.0) in fake.logged_times
    assert not any(entity.startswith("world/franka") for entity in entities)
    assert not any(
        entity.startswith("rollouts/iter_01/rollout-") and entity.endswith("/camera")
        for entity in entities
    )
    assert "signal/reward_trend" in entities
    assert "heldout/success_rate" in entities
    assert "summary/run_success" in entities
    assert "summary/augmentation" in entities
    assert "summary/artifacts" in entities
    assert result.synthetic_frame_count > 0
    assert any(entity.startswith("synthetic/dataset/train/") for entity in entities)
    assert any(entity.startswith("synthetic/dataset/heldout/") for entity in entities)
    assert any(entity.startswith("synthetic/augmentation/") for entity in entities)
    assert "synthetic/preview" in entities
    visual_index = tmp_path / "reports" / "sim2real-visual-index.json"
    assert visual_index.is_file()
    index = json.loads(visual_index.read_text(encoding="utf-8"))
    assert index["success"]["decision"] == "promote_checkpoint"
    assert index["augmentation"]["frame_count"] == 2
    assert index["dataset"]["heldout_count"] == 2
    assert index["synthetic"]["dataset_sample_count"] == 2
    assert index["synthetic"]["dataset_descriptor_preview_count"] >= 2
    assert index["synthetic"]["augmentation_sample_count"] == 1


def test_emit_logs_synchronized_multiview_cameras_and_rotatable_scene(
    monkeypatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    renders_dir = tmp_path / "eval" / "heldout" / "renders" / "heldout-0000"
    views = {
        "primary": ["camera-000.png", "camera-001.png"],
        "side": ["camera-side-000.png", "camera-side-001.png"],
        "overhead": ["camera-overhead-000.png", "camera-overhead-001.png"],
    }
    for view_index, names in enumerate(views.values()):
        for frame_index, name in enumerate(names):
            _write_test_png(
                renders_dir / name,
                red=40 + view_index * 30,
                green=90 + frame_index * 20,
                blue=180,
            )
    heldout_report["render_manifest"] = {
        "schema": "npa.sim2real.heldout_renders.v1",
        "sim_backend": "isaac",
        "camera_views": list(views),
        "episodes": [
            {
                "env_id": "heldout-0000",
                "frames": views["primary"],
                "camera_views": views,
            }
        ],
    }
    heldout_report["capture"] = {
        "width": 640,
        "height": 480,
        "heldout_stride": 20,
        "png_compress_level": 3,
        "fps": 10.0,
    }
    heldout_report["camera_metadata"] = [
        {
            "name": name,
            "pose_frame": "isaac_world",
            "width": 640,
            "height": 480,
            "intrinsics_px": {
                "fx": 733.0,
                "fy": 733.0,
                "cx": 320.0,
                "cy": 240.0,
            },
        }
        for name in views
    ]
    heldout_report["policy_inference_provenance"] = {
        "backend": "isaac_rsl_rl_ppo",
        "checkpoint_uri": "s3://bucket/run/model_150.pt",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_size_bytes": 123456,
        "loaded_for_inference": True,
        "stock_or_scripted_policy": False,
    }
    for view in views:
        _write_pointcloud_npz(tmp_path, env_id="heldout-0000", frames=2, view=view)

    fake = _FakeRerun()
    rrb = _RecordingRRB()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, rrb))
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        run_metadata={
            "run_id": "run",
            "heldout_policy_checkpoint": "s3://bucket/run/model_150.pt",
            "heldout_policy_checkpoint_sha256": "c" * 64,
            "heldout_policy_checkpoint_size_bytes": 123456,
            "heldout_policy_loaded_for_inference": True,
            "runtime_parameters": {
                "capture": {"width": 640, "height": 480},
                "ppo": {
                    "num_envs": 1024,
                    "iterations": 150,
                    "steps_per_env": 24,
                },
            },
        },
        output_rrd=tmp_path / "reports" / "sim2real.rrd",
    )

    entities = {entity for entity, _kind in fake.logged}
    assert result.heldout_frame_count == 6
    assert result.pointcloud_frame_count == 2
    for view in views:
        assert f"heldout/camera/heldout-0000/{view}/camera" in entities
    assert "world/heldout/points" in entities
    assert "world/task_context/table" in entities
    assert "world/task_context/cube_start_region" in entities
    assert "world/task_context/goal_region" in entities
    assert "world/task_context/franka_home/links" in entities
    assert "world/task_context/provenance" in entities
    assert "summary/policy_access" in entities
    assert any(
        view["kind"] == "Spatial3DView"
        and view["origin"] == "world"
        and view["name"] == "Scene overview"
        for view in rrb.views
    )


def test_emit_logs_augmentation_previews_from_manifest_without_frame_index(
    monkeypatch, tmp_path: Path
) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    _write_summary_artifacts(tmp_path)
    (tmp_path / "augment" / "frames" / "index.json").unlink()

    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_rrd=tmp_path / "reports" / "sim2real.rrd",
    )

    entities = [entity for entity, _kind in fake.logged]
    assert result.synthetic_frame_count > 0
    assert any(
        entity.startswith("synthetic/augmentation/frame-") for entity in entities
    )
    index = json.loads(
        (tmp_path / "reports" / "sim2real-visual-index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["augmentation"]["frame_count"] == 2
    assert index["synthetic"]["augmentation_sample_count"] == 2
    assert index["synthetic"]["augmentation_descriptor_preview_count"] == 2


def test_heldout_render_step_indices_samples_evenly() -> None:
    from npa.workflows.sim2real.engine import _heldout_render_step_indices

    indices = _heldout_render_step_indices(120, max_frames=8)
    assert 0 in indices
    assert 119 in indices
    assert len(indices) <= 8


def test_build_heldout_render_manifest_from_png_tree(tmp_path: Path) -> None:
    from npa.workflows.sim2real.engine import _build_heldout_render_manifest

    env_dir = tmp_path / "heldout-0000"
    env_dir.mkdir(parents=True)
    (env_dir / "camera-000.png").write_bytes(b"png")
    (env_dir / "camera-001.png").write_bytes(b"png")
    manifest = _build_heldout_render_manifest(
        tmp_path,
        sim_backend="isaac",
        isaac_task="Isaac-Lift-Cube-Franka-v0",
    )
    assert manifest["episodes"][0]["env_id"] == "heldout-0000"
    assert manifest["episodes"][0]["frames"] == ["camera-000.png", "camera-001.png"]
    assert manifest["episodes"][0]["camera_views"] == {
        "primary": ["camera-000.png", "camera-001.png"]
    }


def test_build_heldout_render_manifest_groups_multi_camera_tree(tmp_path: Path) -> None:
    from npa.workflows.sim2real.engine import _build_heldout_render_manifest

    env_dir = tmp_path / "heldout-0000"
    env_dir.mkdir(parents=True)
    for name in (
        "camera-000.png",
        "camera-side-000.png",
        "camera-overhead-000.png",
    ):
        (env_dir / name).write_bytes(b"png")
    episode = _build_heldout_render_manifest(
        tmp_path,
        sim_backend="isaac",
        isaac_task="Isaac-Lift-Cube-Franka-v0",
    )["episodes"][0]
    assert episode["frames"] == ["camera-000.png"]
    assert episode["camera_views"] == {
        "primary": ["camera-000.png"],
        "overhead": ["camera-overhead-000.png"],
        "side": ["camera-side-000.png"],
    }


def test_usable_camera_frames_drops_blank_warmup() -> None:
    import numpy as np

    from npa.workflows.sim2real_viz import _usable_camera_frames

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    real = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert _usable_camera_frames([blank, real]) == [real]


def test_ensure_heldout_renders_builds_manifest_from_local_pngs(
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real.engine import _ensure_heldout_renders_for_viz
    from npa.workflows.sim2real.models import Sim2RealLoopConfig

    config = Sim2RealLoopConfig(run_id="sim2real-staged-20260616t032140z")
    renders_dir = tmp_path / "eval" / "heldout" / "renders" / "env-00003"
    _write_test_png(renders_dir / "camera-000.png", red=40, green=120, blue=200)
    heldout_report = {"success_rate": 1.0, "sim_backend": "isaac"}

    updated = _ensure_heldout_renders_for_viz(config, tmp_path, heldout_report)

    assert updated is not None
    assert updated["render_manifest"]["episodes"][0]["env_id"] == "env-00003"
    assert updated["render_manifest"]["episodes"][0]["frames"] == ["camera-000.png"]


class _RecordingRRB:
    """Records blueprint view construction so tests can assert structure."""

    class PanelState:
        Expanded = "expanded"

    def __init__(self) -> None:
        self.views: list[dict[str, Any]] = []

    def Spatial2DView(
        self, *, origin: str = "", contents: Any = None, name: str = "", **_: Any
    ) -> dict[str, Any]:
        view = {"kind": "Spatial2DView", "origin": origin, "name": name}
        self.views.append(view)
        return view

    def Spatial3DView(
        self, *, origin: str = "", contents: Any = None, name: str = "", **_: Any
    ) -> dict[str, Any]:
        view = {"kind": "Spatial3DView", "origin": origin, "name": name}
        self.views.append(view)
        return view

    def Grid(self, *args: Any, name: str = "", **_: Any) -> dict[str, Any]:
        return {"kind": "Grid", "name": name, "children": list(args)}

    def Vertical(self, *args: Any, **_: Any) -> dict[str, Any]:
        return {"kind": "Vertical", "children": list(args)}

    def Horizontal(self, *args: Any, **_: Any) -> dict[str, Any]:
        return {"kind": "Horizontal", "children": list(args)}

    def TextDocumentView(self, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "TextDocumentView", **kwargs}

    def TimeSeriesView(self, **kwargs: Any) -> dict[str, Any]:
        return {"kind": "TimeSeriesView", **kwargs}

    def TimePanel(self, **_: Any) -> dict[str, Any]:
        return {"kind": "TimePanel"}

    def Blueprint(self, *args: Any, **_: Any) -> dict[str, Any]:
        return {"kind": "Blueprint", "children": list(args)}


def test_build_blueprint_one_2d_view_per_heldout_env() -> None:
    rrb = _RecordingRRB()
    viz_module._build_blueprint(
        rrb, heldout_env_ids=["env-00006", "env-00009", "env-00018"]
    )
    assert rrb.views[0] == {
        "kind": "Spatial2DView",
        "origin": "camera",
        "name": "Isaac held-out simulation camera",
    }
    assert not any(
        v["kind"] == "Spatial3DView" and v["origin"] == "world" for v in rrb.views
    )
    heldout_origins = [
        v["origin"]
        for v in rrb.views
        if v["kind"] == "Spatial2DView" and v["origin"].startswith("heldout/camera/")
    ]
    assert heldout_origins == [
        "heldout/camera/env-00006/primary",
        "heldout/camera/env-00009/primary",
        "heldout/camera/env-00018/primary",
    ]


def test_build_blueprint_exposes_all_camera_angles_and_3d_scene() -> None:
    rrb = _RecordingRRB()
    viz_module._build_blueprint(
        rrb,
        heldout_env_ids=["env-00006"],
        heldout_camera_views=["primary", "side", "overhead"],
        has_3d_scene=True,
    )
    origins = {view["origin"] for view in rrb.views}
    assert {
        "heldout/camera/env-00006/primary",
        "heldout/camera/env-00006/side",
        "heldout/camera/env-00006/overhead",
    }.issubset(origins)
    assert any(
        view["kind"] == "Spatial3DView" and view["origin"] == "world"
        for view in rrb.views
    )


def test_build_blueprint_without_env_ids_keeps_single_camera_view() -> None:
    rrb = _RecordingRRB()
    viz_module._build_blueprint(rrb, has_heldout_cameras=False)
    # No per-env heldout views; falls back to the rollouts camera view.
    assert not any(v["origin"].startswith("heldout/camera/") for v in rrb.views)


def test_build_blueprint_with_synthetic_view() -> None:
    rrb = _RecordingRRB()
    viz_module._build_blueprint(
        rrb,
        heldout_env_ids=["env-00006"],
        has_synthetic_data=True,
    )
    origins = [view["origin"] for view in rrb.views if view["kind"] == "Spatial2DView"]
    assert "synthetic/preview" in origins
    assert "synthetic/dataset/train" in origins
    assert "synthetic/dataset/heldout" in origins
    assert "synthetic/augmentation" in origins


def test_log_heldout_cameras_time_aligns_envs() -> None:
    import numpy as np

    fake = _FakeRerun()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    episodes = [("env-a", [frame, frame, frame]), ("env-b", [frame, frame, frame])]
    counts: dict[str, int] = {}
    logged, end_seconds = viz_module._log_heldout_cameras(
        fake, None, episodes, counts, start_seconds=10.0
    )
    assert logged == 6
    # Both envs share the same time window -> env-a times == env-b times.
    assert fake.times[:3] == fake.times[3:6]
    assert fake.times[0] == 10.0
    assert end_seconds == 10.0 + 3 * viz_module.ROLLOUT_FRAME_SECONDS


def test_log_heldout_cameras_time_aligns_views() -> None:
    import numpy as np

    fake = _FakeRerun()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    episodes = [
        (
            "env-a",
            {
                "primary": [frame, frame],
                "side": [frame, frame],
                "overhead": [frame, frame],
            },
        )
    ]
    counts: dict[str, int] = {}
    logged, end_seconds = viz_module._log_heldout_cameras(
        fake, None, episodes, counts, start_seconds=3.0
    )
    assert logged == 6
    for view in ("primary", "side", "overhead"):
        assert counts[f"/heldout/camera/env-a/{view}/camera"] == 2
    assert end_seconds == 3.0 + 2 * viz_module.ROLLOUT_FRAME_SECONDS


def test_embodiment_summary_makes_custom_robot_parity_visible() -> None:
    markdown = viz_module._embodiment_markdown(
        {
            "embodiment": {
                "embodiment_digest": "a" * 64,
                "expected_action_dim": 8,
                "expected_observation_dim": 36,
                "resolved_usd_uri": "s3://private-run/resolved/robot.usd",
                "runtime_dimension_validation": "passed",
            }
        }
    )
    assert "a" * 64 in markdown
    assert "Action dimension: `8`" in markdown
    assert "Observation dimension: `36`" in markdown
    assert "Runtime parity: `passed`" in markdown

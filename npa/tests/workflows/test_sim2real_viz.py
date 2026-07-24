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
        self.times: list[float] = []
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

    def save(self, path: Any, default_blueprint: Any = None, recording: Any = None) -> None:
        self.saved_path = Path(path)
        Path(path).write_bytes(b"FAKE_RRD_CONTENT")

    def send_blueprint(self, blueprint: Any, recording: Any = None) -> None:
        return None

    def set_time_seconds(self, timeline: str, seconds: float, recording: Any = None) -> None:
        self.times.append(float(seconds))

    def log(self, entity_path: str, archetype: dict[str, Any], recording: Any = None) -> None:
        self.logged.append((entity_path, archetype.get("kind", "?")))

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
        "reward_trend": [0.2, 0.45],
        "iterations": [
            {
                "iteration": 1,
                "actions_dir": str(actions_dir),
                "vlm_eval_dir": str(eval_dir),
                "signal_dir": str(signal_dir),
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


def test_emit_logs_frames_critiques_signal_and_heldout(monkeypatch, tmp_path: Path) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
    fake = _FakeRerun()
    monkeypatch.setattr(viz_module, "_import_rerun", lambda: (fake, MagicMock()))

    rrd_path = tmp_path / "reports" / "sim2real.rrd"
    result = emit_sim2real_rerun(
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
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
    # Action trajectories per rollout step.
    assert any("/actions/dim_00" in e for e in entities)
    assert any(e.endswith("/actions/l2_norm") for e in entities)
    # Held-out scores.
    assert "heldout/success_rate" in entities
    assert "heldout/scores" in entities
    assert any(e.startswith("heldout/per_env/") for e in entities)

    counts = result.entity_counts
    assert counts["/signal/reward"] == 6
    assert counts["/world/franka/joints"] >= 1
    assert counts["/rollouts/iter_01/rollout-0000/actions/dim_00"] == 3
    assert counts["/heldout/scores"] == 2
    assert counts["/heldout/success_rate"] == 1


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
    # 2 rollouts x 3 frames of raw .ppm camera dumps, all transcoded to PNG.
    assert result.camera_message_count == 6
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


def test_is_reference_stub_rollout_detects_reference_fixture(tmp_path: Path) -> None:
    actions_dir = tmp_path / "actions"
    rollouts = generate_action_rollouts(actions_dir, count=1, steps_per_rollout=2, seed=3, quality=0.5)
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


def test_emit_prefers_heldout_isaac_cameras_over_stub_rollouts(monkeypatch, tmp_path: Path) -> None:
    inner_evidence, heldout_report = _build_run_tree(tmp_path)
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
    assert not any(
        entity.startswith("rollouts/iter_01/rollout-") and entity.endswith("/camera")
        for entity in entities
    )
    assert "signal/reward_trend" in entities
    assert "heldout/success_rate" in entities


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

    def Spatial2DView(self, *, origin: str = "", contents: Any = None, name: str = "", **_: Any) -> dict[str, Any]:
        view = {"kind": "Spatial2DView", "origin": origin, "name": name}
        self.views.append(view)
        return view

    def Spatial3DView(self, *, origin: str = "", contents: Any = None, name: str = "", **_: Any) -> dict[str, Any]:
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
    assert any(v["kind"] == "Spatial3DView" and v["origin"] == "world" for v in rrb.views)
    heldout_origins = [
        v["origin"] for v in rrb.views
        if v["kind"] == "Spatial2DView" and v["origin"].startswith("heldout/camera/")
    ]
    assert heldout_origins == [
        "heldout/camera/env-00006",
        "heldout/camera/env-00009",
        "heldout/camera/env-00018",
    ]


def test_build_blueprint_without_env_ids_keeps_single_camera_view() -> None:
    rrb = _RecordingRRB()
    viz_module._build_blueprint(rrb, has_heldout_cameras=False)
    # No per-env heldout views; falls back to the rollouts camera view.
    assert not any(
        v["origin"].startswith("heldout/camera/") for v in rrb.views
    )


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

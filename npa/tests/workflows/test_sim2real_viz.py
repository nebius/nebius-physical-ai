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

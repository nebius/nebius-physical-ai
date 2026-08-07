from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from npa.cli.isaac_lab.eval_runner import (
    EvalConfig,
    apply_success_metric,
    resolve_checkpoint,
    resolve_success_metric,
    write_eval_summary,
)


def test_resolve_checkpoint_prefers_portable_stable_weights(tmp_path: Path) -> None:
    stable = tmp_path / "npa_isaac_lab_checkpoint.pt"
    stable.write_bytes(b"weights")
    older = tmp_path / "logs" / "model_1.pt"
    older.parent.mkdir()
    older.write_bytes(b"older")

    resolved, checkpoint_format = resolve_checkpoint(tmp_path)

    assert resolved == stable
    assert checkpoint_format == "rsl_rl_checkpoint"


def test_resolve_checkpoint_repairs_stale_manifest_path(tmp_path: Path) -> None:
    stable = tmp_path / "npa_isaac_lab_checkpoint.pt"
    stable.write_bytes(b"weights")
    manifest = tmp_path / "npa_isaac_lab_checkpoint_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "npa_isaac_lab_rsl_rl_checkpoint_v1",
                "stable_checkpoint_path": "/old/vm/npa_isaac_lab_checkpoint.pt",
            }
        )
    )

    resolved, checkpoint_format = resolve_checkpoint(manifest)

    assert resolved == stable
    assert checkpoint_format == "npa_isaac_lab_rsl_rl_checkpoint_v1"


def test_resolve_checkpoint_rejects_manifest_without_weights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"checkpoint_path": "/old/vm/model_1.pt"}))

    with pytest.raises(FileNotFoundError, match="does not resolve"):
        resolve_checkpoint(manifest)


@pytest.mark.parametrize(
    ("episodes", "expected"),
    [
        ([{"native_success": True, "min_goal_distance_m": None}], "native-success"),
        ([{"native_success": None, "min_goal_distance_m": 0.1}], "goal-distance"),
        ([{"native_success": None, "min_goal_distance_m": None}], "survival"),
    ],
)
def test_resolve_success_metric_auto(episodes, expected: str) -> None:
    assert resolve_success_metric("auto", episodes) == expected


def test_resolve_success_metric_auto_requires_native_success_for_every_episode() -> None:
    episodes = [
        {"native_success": True, "min_goal_distance_m": 0.01},
        {"native_success": None, "min_goal_distance_m": 0.20},
    ]

    assert resolve_success_metric("auto", episodes) == "goal-distance"


def test_apply_goal_distance_and_survival_metrics() -> None:
    goal_episodes = [
        {"min_goal_distance_m": 0.04},
        {"min_goal_distance_m": 0.08},
    ]
    apply_success_metric(
        "goal-distance",
        goal_episodes,
        success_distance_m=0.05,
        task="Isaac-Lift-Cube-Franka-v0",
    )
    assert [episode["success"] for episode in goal_episodes] == [True, False]

    survival_episodes = [{"terminated": False}, {"terminated": True}]
    apply_success_metric(
        "survival",
        survival_episodes,
        success_distance_m=0.05,
        task="Isaac-Velocity-Flat-Anymal-C-v0",
    )
    assert [episode["success"] for episode in survival_episodes] == [True, False]


def test_eval_config_validates_quality_gate() -> None:
    config = EvalConfig(
        task="Isaac-Cartpole-v0",
        checkpoint=Path("/tmp/model.pt"),
        num_episodes=1,
        output_dir=Path("/tmp/eval"),
        min_success_rate=1.1,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        config.validate()


def test_eval_config_reads_video_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_ISAAC_EVAL_TASK", "Isaac-Cartpole-v0")
    monkeypatch.setenv("NPA_ISAAC_EVAL_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("NPA_ISAAC_EVAL_OUTPUT_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("NPA_ISAAC_EVAL_VIDEO", "yes")
    monkeypatch.setenv("NPA_ISAAC_EVAL_VIDEO_LENGTH", "123")
    monkeypatch.setenv("NPA_ISAAC_EVAL_VIDEO_FPS", "24")

    config = EvalConfig.from_environment()

    assert config.capture_video is True
    assert config.video_length == 123
    assert config.video_fps == 24
    assert config.video_dir == tmp_path / "eval" / "video"


def test_write_eval_summary_keeps_quality_failure_separate_from_runtime(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    config = EvalConfig(
        task="Isaac-Lift-Cube-Franka-v0",
        checkpoint=checkpoint,
        num_episodes=2,
        output_dir=tmp_path / "eval",
        success_metric="goal-distance",
        success_distance_m=0.05,
        min_success_rate=1.0,
    )
    episodes = [
        {"reward": 1.0, "min_goal_distance_m": 0.01},
        {"reward": 0.0, "min_goal_distance_m": 0.20},
    ]

    summary = write_eval_summary(
        config,
        episode_results=episodes,
        checkpoint_file=checkpoint,
        checkpoint_format="rsl_rl_checkpoint",
        device="cuda:0",
        started=time.time(),
    )

    assert summary["status"] == "success"
    assert summary["policy_loaded"] is True
    assert summary["success_rate"] == 0.5
    assert summary["passed"] is False

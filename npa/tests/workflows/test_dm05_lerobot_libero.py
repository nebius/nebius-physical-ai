"""Contract tests for the DM05 LeRobot checkpoint comparison workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows import dm05_lerobot_libero as workflow


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "dm05",
                "use_relative_actions": False,
                "add_state": False,
                "chunk_size": 10,
                "n_action_steps": 10,
                "input_features": {"observation.state": {"shape": [8]}},
                "output_features": {"action": {"shape": [7]}},
            }
        )
    )
    return path


def _native_eval(path: Path, role: str) -> None:
    av = pytest.importorskip("av")
    path.mkdir(parents=True)
    target = {"candidate": (49, 50, 50, 48), "baseline": (45, 46, 47, 44)}[role]
    tasks = []
    for suite, successes in zip(workflow.SUITES, target, strict=True):
        for task_id in range(10):
            values = [True] * 5
            if task_id == 0 and successes < 50:
                values[-1] = False
            if task_id == 1 and successes < 49:
                values[-2] = False
            if task_id == 2 and successes < 48:
                values[-3] = False
            tasks.append(
                {
                    "task_group": suite,
                    "task_id": task_id,
                    "metrics": {"successes": values},
                }
            )
    (path / "eval_info.json").write_text(
        json.dumps({"per_task": tasks, "per_group": {}, "overall": {}})
    )
    video = path / "videos" / "episode.mp4"
    video.parent.mkdir()
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 8, 8, "yuv420p"
        import numpy as np

        for index in range(2):
            frame = av.VideoFrame.from_ndarray(
                np.full((8, 8, 3), index * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_five_stage_contract_preserves_action_controller_boundary_and_real_artifacts(
    tmp_path, monkeypatch
):
    pytest.importorskip("rerun")
    native_run = workflow.subprocess.run
    checkpoints = {
        role: _checkpoint(tmp_path / role) for role in workflow.MODEL_REPOSITORIES
    }
    monkeypatch.setattr(
        workflow, "_download_checkpoint", lambda role, _: checkpoints[role]
    )

    def native(command, *args, **kwargs):
        if command[0] != "lerobot-eval":
            return native_run(command, *args, **kwargs)
        assert kwargs == {"check": True}
        output = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--output_dir=")
            )
        )
        _native_eval(
            output, "candidate" if "candidate" in str(command[1]) else "baseline"
        )

    monkeypatch.setattr(workflow.subprocess, "run", native)
    prepared, baseline, candidate, metrics, report = (
        tmp_path / name
        for name in ("prepared", "baseline-out", "candidate-out", "metrics", "report")
    )
    assert workflow.main(["prepare", "--output-path", str(prepared)]) == 0
    assert (
        workflow.main(
            [
                "rollout",
                "--input-path",
                str(prepared),
                "--output-path",
                str(baseline),
                "--checkpoint-role",
                "baseline",
            ]
        )
        == 0
    )
    assert (
        workflow.main(
            [
                "rollout",
                "--input-path",
                str(prepared),
                "--output-path",
                str(candidate),
                "--checkpoint-role",
                "candidate",
            ]
        )
        == 0
    )
    assert (
        workflow.main(
            [
                "metrics",
                "--input-path",
                str(prepared),
                "--baseline-path",
                str(baseline),
                "--candidate-path",
                str(candidate),
                "--output-path",
                str(metrics),
            ]
        )
        == 0
    )
    assert (
        workflow.main(
            [
                "report",
                "--input-path",
                str(prepared),
                "--baseline-path",
                str(baseline),
                "--candidate-path",
                str(candidate),
                "--metrics-path",
                str(metrics),
                "--run-id",
                "fixture-dm05",
                "--output-path",
                str(report),
            ]
        )
        == 0
    )
    protocol = json.loads((prepared / "protocol.json").read_text())
    result = json.loads((metrics / "metrics.json").read_text())
    assert protocol["action"]["model_representation"] == "absolute"
    assert protocol["action"]["environment_controller"] == "relative"
    assert result["candidate_successes"] == 197
    assert result["published_197_of_200_reproduced"] is True
    assert result["full_2000_episode_result"] is False
    assert (report / "comparison.mp4").is_file()
    assert (report / "comparison.rrd").is_file()
    assert (
        "metrics/libero_spatial/delta"
        in (report / "comparison.inspection.txt").read_text()
    )


def test_metrics_rejects_rollouts_that_do_not_consume_the_same_protocol(tmp_path):
    protocol = tmp_path / "protocol"
    workflow.prepare(protocol, seed=7, episodes_per_task=5)
    (protocol / "checksums.json").write_text("{}")
    for role in ("baseline", "candidate"):
        root = tmp_path / role
        root.mkdir()
        (root / "rollout.json").write_text(
            json.dumps({"checkpoint_role": role, "protocol_sha256": "different"})
        )
    with pytest.raises(ValueError, match="same sealed protocol"):
        workflow.calculate_metrics(
            protocol, tmp_path / "baseline", tmp_path / "candidate", tmp_path / "output"
        )


def test_rollout_command_is_the_published_native_protocol_with_exact_controller_boundary(
    tmp_path,
):
    protocol = workflow._protocol(seed=7, episodes_per_task=5)
    command = workflow.rollout_command(
        protocol, tmp_path / "checkpoint", tmp_path / "output", "cuda"
    )
    assert command[0] == "lerobot-eval"
    assert "--env.control_mode=relative" in command
    assert "--eval.n_episodes=5" in command
    assert "--seed=7" in command
    assert not any("use_relative_actions=true" in item for item in command)

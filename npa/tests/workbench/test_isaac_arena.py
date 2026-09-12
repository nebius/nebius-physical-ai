from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.isaac_arena import evaluate as sdk_evaluate
from npa.cli.entry import _is_isaac_arena_request
from npa.workbench.isaac_arena.runtime import (
    ARTIFACT_SCHEMA,
    ISAAC_ARENA_REVISION,
    IsaacArenaError,
    IsaacArenaRequest,
    build_evaluation_argv,
    evaluate,
)


def test_lightweight_console_route_is_exact() -> None:
    assert _is_isaac_arena_request(["workbench", "isaac-arena", "evaluate"])
    assert _is_isaac_arena_request(["workbench", "isaac-arena", "terms"])
    assert not _is_isaac_arena_request(["workbench", "isaac-lab"])
    assert not _is_isaac_arena_request(["isaac-arena", "evaluate"])


def test_dry_run_builds_real_pinned_upstream_argv(tmp_path: Path) -> None:
    result = evaluate(IsaacArenaRequest(output_path=str(tmp_path), dry_run=True))
    assert result["schema"] == ARTIFACT_SCHEMA
    assert result["status"] == "dry_run"
    assert result["upstream"]["revision"] == ISAAC_ARENA_REVISION
    argv = result["argv"]
    assert argv[0] == "/isaac-sim/python.sh"
    assert argv[1].endswith("isaaclab_arena/evaluation/policy_runner.py")
    assert argv[-1] == "cube_goal_pose"
    assert argv[argv.index("--num_episodes") + 1] == "1"
    assert "--headless" in argv
    assert "--record_viewport_video" not in argv

    configured = build_evaluation_argv(
        IsaacArenaRequest(
            output_path=str(tmp_path),
            embodiment="franka_ik",
            object_name="dex_cube",
        ),
        output_dir=tmp_path / "upstream",
    )
    environment_index = configured.index("cube_goal_pose")
    assert environment_index < configured.index("--embodiment")
    assert environment_index < configured.index("--object")


def test_policy_specific_inputs_are_fail_closed(tmp_path: Path) -> None:
    replay = tmp_path / "episode.hdf5"
    replay.write_bytes(b"test")
    argv = build_evaluation_argv(
        IsaacArenaRequest(
            output_path=str(tmp_path / "out"),
            policy_type="replay",
            input_path=str(replay),
        ),
        output_dir=tmp_path / "upstream",
        local_input=replay,
    )
    assert argv[argv.index("--replay_file_path") + 1] == str(replay)

    checkpoint = tmp_path / "checkpoint" / "model_100.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test")
    try:
        build_evaluation_argv(
            IsaacArenaRequest(
                output_path=str(tmp_path / "out"),
                policy_type="rsl_rl",
                input_path=str(checkpoint.parent),
            ),
            output_dir=tmp_path / "upstream",
            local_input=checkpoint.parent,
        )
    except IsaacArenaError as exc:
        assert "params/agent.yaml" in str(exc)
    else:  # pragma: no cover - the fail-closed assertion is load-bearing
        raise AssertionError("RSL-RL evaluation accepted an incomplete checkpoint")


def _fake_upstream(
    argv: list[str], **_kwargs: object
) -> subprocess.CompletedProcess[str]:
    output_root = Path(argv[argv.index("--output_base_dir") + 1])
    run = output_root / "2026-09-12_01-02-03"
    (run / "report").mkdir(parents=True)
    (run / "episode_results_rank0.jsonl").write_text(
        json.dumps(
            {
                "job_name": "policy_runner",
                "env_id": 0,
                "episode_in_env": 0,
                "success": False,
                "episode_length": 300,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "index.html").write_text("<html>Arena report</html>", encoding="utf-8")
    (run / "report" / "job_cube.html").write_text(
        "<html>episode detail</html>", encoding="utf-8"
    )
    if "--record_viewport_video" in argv:
        (run / "viewport-episode-0.mp4").write_bytes(b"real-video-fixture")
    return subprocess.CompletedProcess(argv, 0, stdout="Metrics: success_rate=0.0\n")


@patch(
    "npa.workbench.isaac_arena.runtime._probe_mp4",
    return_value={
        "codec": "h264",
        "width": 640,
        "height": 480,
        "duration_seconds": 2.0,
    },
)
@patch(
    "npa.workbench.isaac_arena.runtime._gpu_info",
    return_value={
        "available": True,
        "device_name": "test-gpu",
        "compute_capability": [10, 0],
    },
)
def test_execution_requires_scored_episode_report_and_requested_video(
    _gpu: object, _probe: object, tmp_path: Path
) -> None:
    result = evaluate(
        IsaacArenaRequest(
            output_path=str(tmp_path / "published"),
            record_video=True,
        ),
        runner=_fake_upstream,
    )
    assert result["status"] == "ok"
    assert result["summary"] == {
        "episodes": 1,
        "successes": 0,
        "success_rate": 0.0,
        "mean_episode_length": 300.0,
    }
    assert result["gpu"]["compute_capability"] == [10, 0]
    published = tmp_path / "published"
    manifest = json.loads((published / "result.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "upstream/2026-09-12_01-02-03/episode_results_rank0.jsonl" in paths
    assert "upstream/2026-09-12_01-02-03/index.html" in paths
    assert "upstream/2026-09-12_01-02-03/report/job_cube.html" in paths
    assert "upstream/2026-09-12_01-02-03/viewport-episode-0.mp4" in paths
    assert all(
        len(entry["sha256"]) == 64 and entry["bytes"] > 0
        for entry in manifest["artifacts"]
    )


def test_cli_sdk_and_terms_share_supported_contract(tmp_path: Path) -> None:
    cli = CliRunner().invoke(
        app,
        [
            "workbench",
            "isaac-arena",
            "evaluate",
            "--output-path",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["schema"] == ARTIFACT_SCHEMA
    assert (
        sdk_evaluate(output_path=str(tmp_path), dry_run=True)["schema"]
        == ARTIFACT_SCHEMA
    )

    terms = CliRunner().invoke(app, ["workbench", "isaac-arena", "terms"])
    assert terms.exit_code == 0, terms.output
    payload = json.loads(terms.output)
    assert payload["arena_source"]["license"] == "Apache-2.0"
    assert payload["arena_source"]["release_channel"] == "alpha"
    assert payload["isaac_sim_and_lab"]["baked"] is False

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.isaac_arena import evaluate as sdk_evaluate
from npa.cli.entry import _is_isaac_arena_request
from npa.workbench.isaac_arena.runtime import (
    ARTIFACT_SCHEMA,
    CAPABILITIES_SCHEMA,
    ISAAC_ARENA_REVISION,
    IsaacArenaError,
    IsaacArenaRequest,
    build_evaluation_argv,
    capabilities,
    evaluate,
    _input_evidence,
    _prepare_replay_execution_input,
    _probe_mp4,
)


def test_lightweight_console_route_is_exact() -> None:
    assert _is_isaac_arena_request(["workbench", "isaac-arena", "evaluate"])
    assert _is_isaac_arena_request(["workbench", "isaac-arena", "terms"])
    assert not _is_isaac_arena_request(["workbench", "isaac-lab"])
    assert not _is_isaac_arena_request(["isaac-arena", "evaluate"])


def test_capabilities_are_complete_and_honest() -> None:
    payload = capabilities()
    assert payload["schema"] == CAPABILITIES_SCHEMA
    assert payload["upstream"]["release_channel"] == "alpha"
    assert payload["upstream"]["production_supported"] is False
    assert {item["name"] for item in payload["policy_adapters"]} == {
        "replay",
        "rsl_rl",
        "zero_action",
    }
    replay = next(
        item for item in payload["policy_adapters"] if item["name"] == "replay"
    )
    assert {"implemented", "input_required"}.issubset(replay["npa_status"])
    assert "live_validated" not in replay["npa_status"]
    assert payload["live_validation"]["scope"] == "digest_bound_external_evidence"
    assert payload["live_validation"]["embedded_claims"] is False
    assert len(payload["environments"]) == 18
    assert {
        "cube_goal_pose",
        "gr1_open_microwave",
        "lift_object",
        "tabletop_sort_cubes",
    }.issubset({item["name"] for item in payload["environments"]})
    assert payload["environment_sources"]["graph_specs"]["npa_status"][0] == (
        "unsupported"
    )
    microwave = next(
        item for item in payload["environments"] if item["name"] == "gr1_open_microwave"
    )
    assert "input_required" in microwave["npa_status"]
    assert microwave["runtime_assets"] == [
        {
            "provider": "Lightwheel registry",
            "selector": "fixtures/Microwave039/USD",
            "delivery": "runtime_fetch",
            "baked": False,
        }
    ]
    lightwheel_sdk = next(
        item
        for item in payload["runtime_dependencies"]
        if item["name"] == "lightwheel-sdk"
    )
    assert lightwheel_sdk["version"] == "1.0.3"
    assert lightwheel_sdk["license"] == "Apache-2.0"
    assert lightwheel_sdk["baked"] is True
    lightwheel_assets = next(
        item
        for item in payload["runtime_dependencies"]
        if item["name"] == "Lightwheel registry assets"
    )
    assert lightwheel_assets["baked"] is False
    assert lightwheel_assets["redistribution"] is False
    assert payload["outputs"]["rerun_rrd"] is False

    cli = CliRunner().invoke(app, ["workbench", "isaac-arena", "capabilities"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == payload


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

    with pytest.raises(IsaacArenaError, match="no upstream scored task"):
        evaluate(
            IsaacArenaRequest(
                output_path=str(tmp_path),
                environment="gr1_table_multi_object_no_collision",
                dry_run=True,
            )
        )


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
    return subprocess.CompletedProcess(
        argv, 0, stdout="Metrics: {'success_rate': 0.0}\n"
    )


def _fake_moving_upstream(
    argv: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    _fake_upstream(argv, **kwargs)
    output_root = Path(argv[argv.index("--output_base_dir") + 1])
    results = next(output_root.rglob("episode_results_rank0.jsonl"))
    record = json.loads(results.read_text(encoding="utf-8"))
    record.update(
        {
            "success": True,
            "progress": {
                "overall_score": 1.0,
                "events": [{"step": 40, "predicate_name": "door_open"}],
            },
        }
    )
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return subprocess.CompletedProcess(
        argv,
        0,
        stdout="Metrics: {'success_rate': 1.0, 'revolute_joint_moved_rate': 1.0}\n",
    )


def _make_replay(path: Path, *, steps: int = 4) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = '{"env_name":"fixture","type":2}'
        data.attrs["total"] = steps
        episode = data.create_group("demo_0")
        episode.attrs["num_samples"] = steps
        episode.attrs["success"] = True
        episode.create_dataset(
            "actions",
            data=np.arange(steps * 3, dtype=np.float32).reshape(steps, 3) / 10,
        )
        initial = episode.create_group("initial_state")
        robot = initial.create_group("robot")
        robot.create_dataset("joint_pos", data=np.array([0.1, 0.2]))
        states = episode.create_group("states")
        states.create_dataset(
            "joint_pos", data=np.arange(steps * 2).reshape(steps, 2)
        )
        observations = episode.create_group("obs")
        observations.create_dataset(
            "large_unused_tensor", data=np.ones((steps, 8), dtype=np.float32)
        )


def test_replay_execution_input_is_minimal_hash_bound_and_horizon_complete(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "source.hdf5"
    private = tmp_path / "private"
    _make_replay(replay)
    evidence = _input_evidence(
        IsaacArenaRequest(
            output_path=str(tmp_path / "out"),
            policy_type="replay",
            input_path=str(replay),
        ),
        replay,
    )
    assert evidence is not None
    assert evidence["trajectory"]["recorded_success"] is True

    execution, normalized = _prepare_replay_execution_input(
        replay, private, target_steps=6, source_evidence=evidence
    )

    import h5py
    import numpy as np

    with h5py.File(execution, "r") as dataset:
        episode = dataset["data"]["demo_0"]
        assert set(episode) == {"actions", "initial_state"}
        assert episode["actions"].shape == (6, 3)
        np.testing.assert_array_equal(episode["actions"][-1], episode["actions"][-2])
        assert int(dataset["data"].attrs["total"]) == 6
        assert int(episode.attrs["num_samples"]) == 6
    assert normalized == {
        "strategy": "actions_initial_state_only_hold_final_action",
        "source_sha256": evidence["sha256"],
        "executed_sha256": normalized["executed_sha256"],
        "source_steps": 4,
        "executed_steps": 6,
        "held_final_action_steps": 2,
        "fields": ["actions", "initial_state"],
        "published": False,
    }
    assert len(normalized["executed_sha256"]) == 64
    with pytest.raises(IsaacArenaError, match="cannot truncate"):
        _prepare_replay_execution_input(
            replay, private, target_steps=3, source_evidence=evidence
        )


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
        "max_progress_score": 0.0,
        "progress_event_count": 0,
        "metrics": {"success_rate": 0.0},
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
    video = next(
        entry for entry in manifest["artifacts"] if entry["path"].endswith(".mp4")
    )
    assert video["video"]["binding"] == {
        "run_id": "2026-09-12_01-02-03",
        "upstream_run_directory": "2026-09-12_01-02-03",
        "policy_type": "zero_action",
        "input_sha256": "",
        "executed_input_sha256": "",
    }


@patch(
    "npa.workbench.isaac_arena.runtime._input_evidence",
    return_value={
        "kind": "replay_hdf5",
        "bytes": 100,
        "sha256": "a" * 64,
        "trajectory": {"meaningful": True, "episode": "demo_0"},
    },
)
@patch(
    "npa.workbench.isaac_arena.runtime._probe_mp4",
    return_value={
        "codec": "h264",
        "width": 640,
        "height": 480,
        "duration_seconds": 2.0,
        "motion": {"meaningful": True},
    },
)
@patch(
    "npa.workbench.isaac_arena.runtime._gpu_info",
    return_value={
        "available": True,
        "device_name": "test-gpu",
        "compute_capability": [12, 0],
    },
)
def test_replay_binds_nonzero_input_behavior_and_video_to_run(
    _gpu: object, _probe: object, _input: object, tmp_path: Path
) -> None:
    replay = tmp_path / "episode.hdf5"
    _make_replay(replay)
    result = evaluate(
        IsaacArenaRequest(
            output_path=str(tmp_path / "published"),
            environment="gr1_open_microwave",
            policy_type="replay",
            input_path=str(replay),
            record_video=True,
            run_id="arena-moving-run",
            replay_target_steps=6,
        ),
        runner=_fake_moving_upstream,
    )
    assert result["behavior"]["meaningful"] is True
    assert result["runtime"]["lightwheel_sdk"] == {
        "version": "1.0.3",
        "baked": True,
        "license": "Apache-2.0",
    }
    assert result["runtime"]["lightwheel_registry_assets"]["runtime_fetch"] is True
    assert result["runtime"]["lightwheel_registry_assets"]["redistribution"] is False
    assert result["summary"]["metrics"]["revolute_joint_moved_rate"] == 1.0
    video = next(
        entry for entry in result["artifacts"] if entry["path"].endswith(".mp4")
    )
    assert video["video"]["binding"]["run_id"] == "arena-moving-run"
    assert video["video"]["binding"]["input_sha256"] == "a" * 64
    assert len(video["video"]["binding"]["executed_input_sha256"]) == 64
    assert result["input"]["execution"]["source_steps"] == 4
    assert result["input"]["execution"]["executed_steps"] == 6
    assert not (tmp_path / "published" / "private").exists()
    assert all("replay-execution" not in entry["path"] for entry in result["artifacts"])
    assert result["request"]["input_path"] == "<operator-input>"
    assert result["request"]["output_path"] == "<operator-output>"


@patch(
    "npa.workbench.isaac_arena.runtime._input_evidence",
    return_value={
        "kind": "replay_hdf5",
        "bytes": 100,
        "sha256": "a" * 64,
        "trajectory": {"meaningful": True, "episode": "demo_0"},
    },
)
@patch(
    "npa.workbench.isaac_arena.runtime._gpu_info",
    return_value={
        "available": True,
        "device_name": "test-gpu",
        "compute_capability": [12, 0],
    },
)
def test_nonzero_policy_rejects_no_output_behavior(
    _gpu: object, _input: object, tmp_path: Path
) -> None:
    replay = tmp_path / "episode.hdf5"
    _make_replay(replay)
    with pytest.raises(IsaacArenaError, match="no positive task"):
        evaluate(
            IsaacArenaRequest(
                output_path=str(tmp_path / "published"),
                environment="gr1_open_microwave",
                policy_type="replay",
                input_path=str(replay),
            ),
            runner=_fake_upstream,
        )


def _make_test_video(path: Path, source: str) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def test_video_probe_accepts_temporal_motion(tmp_path: Path) -> None:
    video = tmp_path / "moving.mp4"
    _make_test_video(video, "testsrc2=size=320x240:rate=10:duration=2")
    metadata = _probe_mp4(video)
    assert metadata["codec"] == "h264"
    assert metadata["frame_count"] == 20
    assert metadata["motion"]["decoded_samples"] >= 4
    assert metadata["motion"]["changed_frame_pairs"] >= 2
    assert metadata["motion"]["meaningful"] is True


def test_video_probe_rejects_decodable_static_video(tmp_path: Path) -> None:
    video = tmp_path / "static.mp4"
    _make_test_video(video, "color=c=blue:size=320x240:rate=10:duration=2")
    with pytest.raises(IsaacArenaError, match="decodable but visually static"):
        _probe_mp4(video)


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
    assert payload["lightwheel_sdk"] == {
        "baked": True,
        "license": "Apache-2.0",
        "version": "1.0.3",
    }
    assert payload["lightwheel_registry_assets"]["baked"] is False
    assert payload["lightwheel_registry_assets"]["redistribution"] is False

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
    _prepare_viewport_graphics,
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
    assert payload["live_validation"]["readiness_sources"] == [
        "workflows/testing/isaac-arena-evaluation-rtxpro.readiness.json",
        "workflows/testing/isaac-arena-evaluation-b200.readiness.json",
        "npa/docker/workbench/blackwell-dc-images.json",
        "npa/src/npa/deploy/public_release_manifest.json",
    ]
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
    viewport_graphics = next(
        item
        for item in payload["runtime_dependencies"]
        if item["name"] == "NVIDIA viewport graphics userspace"
    )
    assert viewport_graphics["delivery"] == "runtime_fetch_exact_driver_match"
    assert viewport_graphics["installed_on_node"] is False
    assert viewport_graphics["redistribution"] is False
    assert "EGL ICD" in viewport_graphics["headless_icd"]
    assert payload["outputs"]["rerun_rrd"] is False

    cli = CliRunner().invoke(app, ["workbench", "isaac-arena", "capabilities"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == payload


def test_dry_run_builds_real_pinned_upstream_argv(tmp_path: Path) -> None:
    result = evaluate(IsaacArenaRequest(output_path=str(tmp_path), dry_run=True))
    assert result["schema"] == ARTIFACT_SCHEMA
    assert result["status"] == "dry_run"
    assert result["upstream"]["revision"] == ISAAC_ARENA_REVISION
    assert result["runtime"]["viewport_graphics"] == {
        "mode": "not_requested",
        "validated": False,
    }
    argv = result["argv"]
    assert argv[0] == "/isaac-sim/python.sh"
    assert argv[1].endswith("isaaclab_arena/evaluation/policy_runner.py")
    assert argv[-1] == "cube_goal_pose"
    assert argv[argv.index("--num_episodes") + 1] == "1"
    assert "--headless" in argv
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert "--record_viewport_video" not in argv

    cpu_argv = build_evaluation_argv(
        IsaacArenaRequest(output_path=str(tmp_path), execution_device="cpu"),
        output_dir=tmp_path / "cpu-upstream",
    )
    assert cpu_argv[cpu_argv.index("--device") + 1] == "cpu"

    with pytest.raises(IsaacArenaError, match="execution_device"):
        evaluate(
            IsaacArenaRequest(
                output_path=str(tmp_path), execution_device="cuda", dry_run=True
            )
        )

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


@pytest.mark.parametrize("policy,flag", [("replay", "--replay_file_path"), ("rsl_rl", "--checkpoint_path")])
def test_policy_dry_run_requires_no_input_download_or_local_file(tmp_path, policy, flag):
    request = IsaacArenaRequest(
        output_path=str(tmp_path / "out"), policy_type=policy,
        input_path=str(tmp_path / "absent-input"), dry_run=True,
    )
    with patch("npa.workbench.isaac_arena.runtime._local_input", side_effect=AssertionError("must not load")):
        result = evaluate(request)
    assert result["status"] == "dry_run"
    assert result["argv"][result["argv"].index(flag) + 1] == "<operator-input>"
    assert result["argv"].index(flag) < result["argv"].index("cube_goal_pose")
    assert not (tmp_path / "out").exists()


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
    _write_ground_truth(run, success=False)
    if "--record_viewport_video" in argv:
        (run / "viewport-episode-0.mp4").write_bytes(b"real-video-fixture")
    return subprocess.CompletedProcess(
        argv, 0, stdout="Metrics: {'success_rate': 0.0}\n"
    )


def _fake_moving_upstream(
    argv: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert env["NPA_ISAAC_ARENA_VIEWPORT_ONLY"] == "1"
    _fake_upstream(argv, **kwargs)
    output_root = Path(argv[argv.index("--output_base_dir") + 1])
    results = next(output_root.rglob("episode_results_rank0.jsonl"))
    record = json.loads(results.read_text(encoding="utf-8"))
    record.update(
        {
            "success": True,
            "episode_length": 4,
            "progress": {
                "overall_score": 1.0,
                "events": [{"step": 40, "predicate_name": "door_open"}],
            },
        }
    )
    results.write_text(json.dumps(record) + "\n", encoding="utf-8")
    run = results.parent
    _write_ground_truth(run, success=True, microwave=True)
    return subprocess.CompletedProcess(
        argv,
        0,
        stdout="Metrics: {'success_rate': 1.0, 'revolute_joint_moved_rate': 1.0}\n",
    )


def _fake_viewport_graphics(_root: Path, env: dict[str, str]) -> dict[str, object]:
    assert env["NPA_ISAAC_ARENA_VIEWPORT_ONLY"] == "1"
    return {
        "mode": "native",
        "validated": True,
        "runtime_fetch": False,
        "baked": False,
        "redistribution": False,
    }


def _fake_video_preparer(source: Path) -> tuple[Path, dict[str, object]]:
    target = source.with_name(f"{source.stem}-evidence-denoised.mp4")
    target.write_bytes(source.read_bytes())
    return target, {
        "kind": "test_spatiotemporal_denoise",
        "filter": "test",
        "source_path": source.name,
        "source_sha256": "b" * 64,
        "changes_simulator_outcome": False,
    }


def _write_ground_truth(run: Path, *, success: bool, microwave: bool = False) -> None:
    import h5py
    import numpy as np

    path = run / "simulator_ground_truth_rank0.hdf5"
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("success", data=np.array([success], dtype=bool))
        steps = 4 if microwave else 300
        episode.create_dataset("npa_video/initial_action_step", data=[[0]])
        episode.create_dataset("npa_video/action_step", data=np.arange(1, steps + 1).reshape(-1, 1))
        episode.create_dataset("npa_video/terminal_action_step", data=[[steps]])
        if microwave:
            trace = [0.2, 0.21, 0.36, 0.63, 0.81] if success else [0.2, 0.25, 0.1]
            episode.create_dataset(
                "revolute_joint_state",
                data=np.array(trace, dtype=np.float32).reshape(-1, 1),
            )
        else:
            episode.create_dataset(
                "object_linear_velocity",
                data=np.zeros((4, 3), dtype=np.float32),
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
        states.create_dataset("joint_pos", data=np.arange(steps * 2).reshape(steps, 2))
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
    assert evidence["trajectory"]["source_recorded_success"] is True
    assert evidence["trajectory"]["runtime_outcome_claim"] is False

    execution, normalized = _prepare_replay_execution_input(
        replay, private, source_evidence=evidence
    )

    import h5py
    import numpy as np

    with h5py.File(execution, "r") as dataset:
        episode = dataset["data"]["demo_0"]
        assert set(episode) == {"actions", "initial_state"}
        assert episode["actions"].shape == (4, 3)
        np.testing.assert_array_equal(
            episode["actions"],
            np.arange(12, dtype=np.float32).reshape(4, 3) / 10,
        )
        assert int(dataset["data"].attrs["total"]) == 4
        assert int(episode.attrs["num_samples"]) == 4
    assert normalized == {
        "strategy": "actions_initial_state_exact_replay",
        "source_sha256": evidence["sha256"],
        "executed_sha256": normalized["executed_sha256"],
        "source_steps": 4,
        "executed_steps": 4,
        "action_padding_steps": 0,
        "initial_state_application": "isaac_lab_reset_to_relative",
        "fields": ["actions", "initial_state"],
        "published": False,
    }
    assert len(normalized["executed_sha256"]) == 64


def test_viewport_graphics_prefers_valid_native_stack(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        stdout = "GPU0: NVIDIA" if argv[0] == "vulkaninfo" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    env: dict[str, str] = {}
    result = _prepare_viewport_graphics(tmp_path / "graphics", env, runner=fake_runner)

    assert result == {
        "mode": "native",
        "validated": True,
        "runtime_fetch": False,
        "baked": False,
        "redistribution": False,
    }
    assert len(calls) == 2
    assert all("apt-get" not in call for call in calls)
    assert "VK_ICD_FILENAMES" not in env


def test_viewport_graphics_extracts_exact_driver_match_privately(
    tmp_path: Path,
) -> None:
    driver_version = "580.173.02"
    candidate = f"{driver_version}-0ubuntu0.24.04.1"
    commands: list[list[str]] = []

    def fake_runner(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        env = kwargs.get("env")
        if argv[0].endswith("python") or argv[0].endswith("python3"):
            ready = isinstance(env, dict) and "VK_ICD_FILENAMES" in env
            return subprocess.CompletedProcess(
                argv, 0 if ready else 1, stdout="", stderr=""
            )
        if argv[0] == "vulkaninfo":
            return subprocess.CompletedProcess(
                argv, 0, stdout="GPU0: NVIDIA", stderr=""
            )
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{driver_version}\n", stderr=""
            )
        if argv[:2] == ["apt-cache", "policy"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"  Candidate: {candidate}\n", stderr=""
            )
        if "download" in argv and argv[0] == "apt-get":
            download_dir = Path(str(kwargs["cwd"]))
            (download_dir / "graphics.deb").write_bytes(b"signed-driver-package")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[:2] == ["dpkg-deb", "--field"]:
            values = {
                "Package": "libnvidia-gl-580-server",
                "Version": candidate,
                "Architecture": "amd64",
            }
            return subprocess.CompletedProcess(
                argv, 0, stdout=values[argv[-1]], stderr=""
            )
        if argv[:2] == ["dpkg-deb", "--extract"]:
            extracted = Path(argv[-1])
            library_dir = extracted / "usr/lib/x86_64-linux-gnu"
            library_dir.mkdir(parents=True)
            (library_dir / "libGLX_nvidia.so.0").write_bytes(b"glx")
            (library_dir / "libEGL_nvidia.so.0").write_bytes(b"egl")
            icd = extracted / "usr/share/vulkan/icd.d/nvidia_icd.json"
            icd.parent.mkdir(parents=True)
            icd.write_text(
                json.dumps(
                    {
                        "file_format_version": "1.0.1",
                        "ICD": {
                            "library_path": "libGLX_nvidia.so.0",
                            "api_version": "1.4.312",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    env = {"LD_LIBRARY_PATH": "/existing"}
    result = _prepare_viewport_graphics(tmp_path / "graphics", env, runner=fake_runner)

    assert result["mode"] == "runtime_package_extract"
    assert result["driver_version"] == driver_version
    assert result["package_version"] == candidate
    assert result["package_sha256"] == (
        "624a9cf63b28a6d18e71b9099f6f4be41b00e46b40d2a953ec9a547d15c95992"
    )
    assert result["installed_on_node"] is False
    assert result["published"] is False
    assert result["redistribution"] is False
    assert result["headless_icd_override"] is True
    assert result["icd_entrypoint"] == "libEGL_nvidia.so.0"
    assert len(result["package_manifest_sha256"]) == 64
    assert len(result["headless_icd_sha256"]) == 64
    assert env["LD_LIBRARY_PATH"].endswith(":/existing")
    assert env["VK_ICD_FILENAMES"] == env["VK_DRIVER_FILES"]
    generated_icd = Path(env["VK_ICD_FILENAMES"])
    assert generated_icd.stat().st_mode & 0o777 == 0o600
    assert json.loads(generated_icd.read_text(encoding="utf-8")) == {
        "file_format_version": "1.0.1",
        "ICD": {
            "library_path": "libEGL_nvidia.so.0",
            "api_version": "1.4.312",
        },
    }
    update = next(command for command in commands if "update" in command)
    download = next(command for command in commands if "download" in command)
    assert update[:2] == ["sudo", "apt-get"]
    assert download[0] == "apt-get"
    assert "Acquire::AllowInsecureRepositories=false" in update
    assert "APT::Get::AllowUnauthenticated=false" in update
    assert "Acquire::AllowInsecureRepositories=false" in download
    assert "APT::Get::AllowUnauthenticated=false" in download
    assert f"libnvidia-gl-580-server={candidate}" in download
    assert not any("install" in command for command in commands)


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {
            "file_format_version": "1.0.1",
            "ICD": {
                "library_path": "/opt/untrusted.so",
                "api_version": "1.4.312",
            },
        },
        {
            "file_format_version": "1.0.1",
            "ICD": {
                "library_path": "libGLX_nvidia.so.0",
                "api_version": "$(untrusted)",
            },
        },
    ],
)
def test_viewport_graphics_rejects_untrusted_packaged_icd(
    tmp_path: Path, manifest: dict[str, object]
) -> None:
    driver_version = "580.173.02"
    candidate = f"{driver_version}-0ubuntu0.24.04.1"

    def fake_runner(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if argv[0].endswith("python") or argv[0].endswith("python3"):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{driver_version}\n", stderr=""
            )
        if argv[:2] == ["apt-cache", "policy"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"  Candidate: {candidate}\n", stderr=""
            )
        if "download" in argv and argv[0] == "apt-get":
            download_dir = Path(str(kwargs["cwd"]))
            (download_dir / "graphics.deb").write_bytes(b"signed-driver-package")
        if argv[:2] == ["dpkg-deb", "--field"]:
            values = {
                "Package": "libnvidia-gl-580-server",
                "Version": candidate,
                "Architecture": "amd64",
            }
            return subprocess.CompletedProcess(
                argv, 0, stdout=values[argv[-1]], stderr=""
            )
        if argv[:2] == ["dpkg-deb", "--extract"]:
            extracted = Path(argv[-1])
            library_dir = extracted / "usr/lib/x86_64-linux-gnu"
            library_dir.mkdir(parents=True)
            (library_dir / "libGLX_nvidia.so.0").write_bytes(b"glx")
            (library_dir / "libEGL_nvidia.so.0").write_bytes(b"egl")
            icd = extracted / "usr/share/vulkan/icd.d/nvidia_icd.json"
            icd.parent.mkdir(parents=True)
            icd.write_text(json.dumps(manifest), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(IsaacArenaError, match="Vulkan metadata"):
        _prepare_viewport_graphics(tmp_path / "graphics", {}, runner=fake_runner)


def test_viewport_graphics_rejects_nonmatching_archive_version(
    tmp_path: Path,
) -> None:
    def fake_runner(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if argv[0].endswith("python") or argv[0].endswith("python3"):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                argv, 0, stdout="580.173.02\n", stderr=""
            )
        if argv[:2] == ["apt-cache", "policy"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="  Candidate: 580.999.01-0ubuntu1\n", stderr=""
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(IsaacArenaError, match="exact loaded driver version"):
        _prepare_viewport_graphics(tmp_path / "graphics", {}, runner=fake_runner)


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
@patch("npa.workbench.isaac_arena.runtime._verify_capture_evidence", return_value={"verified": True})
def test_execution_requires_scored_episode_report_and_requested_video(
    _capture: object, _gpu: object, _probe: object, tmp_path: Path
) -> None:
    result = evaluate(
        IsaacArenaRequest(
            output_path=str(tmp_path / "published"),
            record_video=True,
        ),
        runner=_fake_upstream,
        graphics_preparer=_fake_viewport_graphics,
        video_preparer=_fake_video_preparer,
    )
    assert result["status"] == "ok"
    assert result["summary"]["episodes"] == 1
    assert result["summary"]["successes"] == 0
    assert result["summary"]["success_rate"] == 0.0
    assert result["summary"]["mean_episode_length"] == 300.0
    assert result["summary"]["max_progress_score"] == 0.0
    assert result["summary"]["progress_event_count"] == 0
    assert result["summary"]["metrics"] == {"success_rate": 0.0}
    ground_truth = result["summary"]["simulator_ground_truth"]
    assert ground_truth["source"] == "current-run Arena metric-recorder HDF5"
    assert ground_truth["successes"] == 0
    assert ground_truth["task_motion"] is None
    assert result["gpu"]["compute_capability"] == [10, 0]
    published = tmp_path / "published"
    manifest = json.loads((published / "result.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "upstream/2026-09-12_01-02-03/episode_results_rank0.jsonl" in paths
    assert "upstream/2026-09-12_01-02-03/index.html" in paths
    assert "upstream/2026-09-12_01-02-03/report/job_cube.html" in paths
    assert "upstream/2026-09-12_01-02-03/viewport-episode-0.mp4" in paths
    assert (
        "upstream/2026-09-12_01-02-03/viewport-episode-0-evidence-denoised.mp4"
        in paths
    )
    assert (
        "upstream/2026-09-12_01-02-03/simulator_ground_truth_rank0.hdf5"
        in paths
    )
    assert all(
        len(entry["sha256"]) == 64 and entry["bytes"] > 0
        for entry in manifest["artifacts"]
    )
    video = next(entry for entry in manifest["artifacts"] if "video" in entry)
    assert video["video"]["binding"] == {
        "run_id": "2026-09-12_01-02-03",
        "upstream_run_directory": "2026-09-12_01-02-03",
        "policy_type": "zero_action",
        "input_sha256": "",
        "executed_input_sha256": "",
        "simulator_ground_truth_sha256": [ground_truth["files"][0]["sha256"]],
    }
    raw_video = next(
        entry for entry in manifest["artifacts"] if "visual_source" in entry
    )
    assert raw_video["visual_source"]["role"] == "raw_upstream_source"
    assert video["video"]["derivation"]["changes_simulator_outcome"] is False
    assert video["video"]["task_qualified"] is False


@patch("npa.workbench.isaac_arena.runtime._verify_capture_evidence", return_value={"verified": True})
@patch("npa.workbench.isaac_arena.runtime._probe_mp4", return_value={"motion": {"meaningful": True}})
@patch("npa.workbench.isaac_arena.runtime._gpu_info", return_value={"available": True})
def test_passive_baseline_success_is_never_task_qualified_video(_gpu, _probe, _capture, tmp_path):
    result = evaluate(
        IsaacArenaRequest(output_path=str(tmp_path / "out"), environment="gr1_open_microwave", record_video=True),
        runner=_fake_moving_upstream, graphics_preparer=_fake_viewport_graphics,
        video_preparer=_fake_video_preparer,
    )
    assert result["summary"]["success_rate"] == 1.0
    assert result["behavior"]["meaningful"] is False
    video = next(item for item in result["artifacts"] if "video" in item)
    assert video["video"]["task_qualified"] is False


@patch(
    "npa.workbench.isaac_arena.runtime._input_evidence",
    return_value={
        "kind": "replay_hdf5",
        "bytes": 100,
        "sha256": "a" * 64,
        "trajectory": {
            "episode": "demo_0",
            "nonzero_actions": True,
            "runtime_outcome_claim": False,
        },
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
@patch("npa.workbench.isaac_arena.runtime._verify_capture_evidence", return_value={"verified": True})
def test_replay_binds_nonzero_input_behavior_and_video_to_run(
    _capture: object, _gpu: object, _probe: object, _input: object, tmp_path: Path
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
        ),
        runner=_fake_moving_upstream,
        graphics_preparer=_fake_viewport_graphics,
        video_preparer=_fake_video_preparer,
    )
    assert result["behavior"]["meaningful"] is True
    assert result["runtime"]["lightwheel_sdk"] == {
        "version": "1.0.3",
        "baked": True,
        "license": "Apache-2.0",
    }
    assert result["runtime"]["execution_device"] == "cuda:0"
    assert result["runtime"]["viewport_renderer_gpu_required"] is True
    assert result["runtime"]["viewport_graphics"] == {
        "mode": "native",
        "validated": True,
        "runtime_fetch": False,
        "baked": False,
        "redistribution": False,
    }
    assert result["runtime"]["lightwheel_registry_assets"]["runtime_fetch"] is True
    assert result["runtime"]["lightwheel_registry_assets"]["redistribution"] is False
    assert result["summary"]["metrics"]["revolute_joint_moved_rate"] == 1.0
    video = next(entry for entry in result["artifacts"] if "video" in entry)
    assert video["video"]["binding"]["run_id"] == "arena-moving-run"
    assert video["video"]["simulator_capture"] == {"verified": True}
    assert video["video"]["task_qualified"] is True
    assert video["video"]["binding"]["input_sha256"] == "a" * 64
    assert len(video["video"]["binding"]["executed_input_sha256"]) == 64
    assert video["video"]["binding"]["simulator_ground_truth_sha256"] == [
        result["summary"]["simulator_ground_truth"]["files"][0]["sha256"]
    ]
    task_motion = result["summary"]["simulator_ground_truth"]["task_motion"]
    assert task_motion["task_success"] is True
    assert task_motion["initial_openness"] == pytest.approx(0.2)
    assert task_motion["final_openness"] == pytest.approx(0.81)
    assert task_motion["openness_delta"] == pytest.approx(0.61)
    assert result["input"]["execution"]["source_steps"] == 4
    assert result["input"]["execution"]["executed_steps"] == 4
    assert result["input"]["execution"]["action_padding_steps"] == 0
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
        "trajectory": {
            "episode": "demo_0",
            "nonzero_actions": True,
            "runtime_outcome_claim": False,
        },
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
def test_nonzero_policy_rejects_movement_without_task_success(
    _gpu: object, _input: object, tmp_path: Path
) -> None:
    replay = tmp_path / "episode.hdf5"
    _make_replay(replay)
    result = evaluate(
        IsaacArenaRequest(output_path=str(tmp_path / "scored"), environment="gr1_open_microwave",
                          policy_type="replay", input_path=str(replay)),
        runner=_fake_upstream,
    )
    assert result["summary"]["success_rate"] == 0.0
    assert result["behavior"]["meaningful"] is False
    assert (tmp_path / "scored" / "result.json").is_file()
    with pytest.raises(
        IsaacArenaError, match="no upstream-defined successful task episode"
    ):
        evaluate(
            IsaacArenaRequest(
                output_path=str(tmp_path / "published"),
                environment="gr1_open_microwave",
                policy_type="replay",
                input_path=str(replay),
                record_video=True,
            ),
            runner=_fake_upstream,
            graphics_preparer=_fake_viewport_graphics,
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
    with pytest.raises(IsaacArenaError, match="decodable but.*motion"):
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
    assert payload["nvidia_viewport_graphics_userspace"] == {
        "baked": False,
        "installation": False,
        "license": "NVIDIA driver package terms",
        "redistribution": False,
        "runtime_fetch": True,
        "scope": "viewport evaluation only",
        "source": "exact-driver-matched Ubuntu signed package",
    }

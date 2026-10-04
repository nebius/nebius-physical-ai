"""Contract tests for the DM05 LeRobot checkpoint comparison workflow."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from npa.workflows import dm05_lerobot_libero as workflow


def _checkpoint(path: Path, role: str = "candidate") -> Path:
    contract = workflow.CHECKPOINT_CONTRACTS[role]
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "dm05",
                "use_relative_actions": False,
                "add_state": contract["add_state"],
                "chunk_size": contract["chunk_size"],
                "n_action_steps": contract["n_action_steps"],
                "input_features": {
                    "observation.state": {"shape": [contract["state_dimension"]]}
                },
                "output_features": {
                    "action": {"shape": [contract["action_dimension"]]}
                },
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
        # This functional test supplies a synthetic, representation-matched
        # baseline solely to exercise all connected artifact stages.  The
        # released generic predecessor is tested below and must be rejected
        # before a live LIBERO comparison.
        role: _checkpoint(tmp_path / role, "candidate")
        for role in workflow.MODEL_REPOSITORIES
    }
    monkeypatch.setattr(
        workflow, "_download_checkpoint", lambda role, _: checkpoints[role]
    )
    # The release-contract rejection is covered independently below.  This
    # harness deliberately uses a representation-matched synthetic baseline
    # to exercise the five connected stages and their real artifact formats.
    monkeypatch.setattr(
        workflow, "_require_libero_checkpoint_contract", lambda role, checkpoint: None
    )
    monkeypatch.setattr(
        workflow,
        "_write_noninteractive_libero_config",
        lambda _: tmp_path / "libero-config",
    )
    runtime = {
        "source": workflow.DM05_IMPLEMENTATION,
        "config_class": "lerobot.policies.dm05.configuration_dm05.DM05Config",
        "policy_class": "lerobot.policies.dm05.modeling_dm05.DM05Policy",
    }
    monkeypatch.setattr(workflow, "_require_dm05_policy_runtime", lambda: runtime)

    def native(command, *args, **kwargs):
        if command[0] != "lerobot-eval":
            return native_run(command, *args, **kwargs)
        assert kwargs["check"] is True
        assert "LIBERO_CONFIG_PATH" in kwargs["env"]
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
    assert protocol["dm05_implementation"] == workflow.DM05_IMPLEMENTATION
    assert result["candidate_successes"] == 197
    assert result["published_197_of_200_reproduced"] is True
    assert result["full_2000_episode_result"] is False
    assert (report / "comparison.mp4").is_file()
    assert (report / "comparison.rrd").is_file()
    assert (
        "metrics/libero_spatial/delta"
        in (report / "comparison.inspection.txt").read_text()
    )
    for root in (baseline, candidate):
        rollout = json.loads((root / "rollout.json").read_text())
        assert rollout["dm05_runtime"] == runtime
        assert rollout["native_checkpoint_load_verified"] is True


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


def test_private_dm05_image_retains_bootstrap_attestation_and_telemetry_boundary():
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/lerobot/Dockerfile.dm05-validation"
    )
    instructions = dockerfile.read_text(encoding="utf-8")

    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"'
        in instructions
    )
    assert "/opt/lerobot/venv/bin/python -m pip uninstall -y wandb" in instructions


def test_dm05_runtime_manifest_is_exact_source_bound(tmp_path, monkeypatch):
    manifest = tmp_path / "dm05-runtime.json"
    monkeypatch.setenv(workflow.DM05_RUNTIME_MANIFEST_ENV, str(manifest))
    manifest.write_text(json.dumps(workflow.DM05_IMPLEMENTATION))

    assert workflow._read_dm05_runtime_manifest() == workflow.DM05_IMPLEMENTATION

    manifest.write_text(json.dumps({"repository": "unreviewed/example"}))
    with pytest.raises(RuntimeError, match="does not match"):
        workflow._read_dm05_runtime_manifest()


def test_checkpoint_download_uses_role_specific_published_contract(
    tmp_path, monkeypatch
):
    baseline = _checkpoint(tmp_path / "published-baseline", "baseline")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_: str(baseline)),
    )

    assert workflow._download_checkpoint("baseline", tmp_path) == baseline

    config_path = baseline / "config.json"
    config = json.loads(config_path.read_text())
    config["output_features"]["action"]["shape"] = [7]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="baseline DM05 checkpoint"):
        workflow._download_checkpoint("baseline", tmp_path)


def test_generic_predecessor_is_rejected_without_an_action_state_adapter(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", "baseline")
    candidate = _checkpoint(tmp_path / "candidate", "candidate")

    with pytest.raises(ValueError, match="not LIBERO-compatible"):
        workflow._require_libero_checkpoint_contract("baseline", baseline)
    workflow._require_libero_checkpoint_contract("candidate", candidate)


def test_libero_config_uses_installed_assets_without_interactive_setup(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "site-packages" / "libero"
    benchmark_root = package_root / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (benchmark_root / name).mkdir(parents=True)
    monkeypatch.setattr(
        workflow.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(
            submodule_search_locations=[str(package_root)]
        ),
    )

    config_root = workflow._write_noninteractive_libero_config(tmp_path / "run")

    config = json.loads((config_root / "config.yaml").read_text())
    assert config["benchmark_root"] == str(benchmark_root)
    assert config["assets"] == str(benchmark_root / "assets")
    assert Path(config["datasets"]).is_dir()

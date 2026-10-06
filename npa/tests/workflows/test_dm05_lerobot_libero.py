"""Contract tests for the DM05 LeRobot checkpoint comparison workflow."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.readiness import load_readiness_record
from npa.workflows import dm05_lerobot_libero as workflow
from npa.workflows import dm05_opendm_libero_baseline as baseline_workflow


WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "workflows/testing/dm05-lerobot-libero-comparison.yaml"
)


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


def _opendm_checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "dm05", "chunk_size": 50, "action_dim": 32})
    )
    (path / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"mean": [0.0] * 8, "std": [1.0] * 8},
                    "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
                }
            }
        )
    )
    return path


def _opendm_results(suite: str) -> dict[str, object]:
    tasks = []
    for task_id in range(10):
        successes = [True] * 5
        if task_id == 0:
            successes[-1] = False
        tasks.append(
            {
                "task_id": str(task_id),
                "episode_results": [
                    {"episode": episode, "success": success}
                    for episode, success in enumerate(successes)
                ],
            }
        )
    return {
        "total_tasks": 10,
        "total_episodes": 50,
        "successful_episodes": 49,
        "success_rate": 0.98,
        "task_results": tasks,
        "suite": suite,
    }


@pytest.mark.parametrize("protocol_seed", [7, 11])
def test_five_stage_contract_preserves_action_controller_boundary_and_real_artifacts(
    tmp_path, monkeypatch, protocol_seed
):
    pytest.importorskip("rerun")
    native_run = workflow.subprocess.run
    checkpoints = {"candidate": _checkpoint(tmp_path / "candidate", "candidate")}
    monkeypatch.setattr(
        workflow, "_download_checkpoint", lambda role, _: checkpoints[role]
    )
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
        assert f"--seed={protocol_seed}" in command
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
    checkpoint = _opendm_checkpoint(tmp_path / "opendm-checkpoint")
    monkeypatch.setattr(
        baseline_workflow,
        "_read_runtime_manifest",
        lambda: baseline_workflow.OPENDM_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        baseline_workflow,
        "_runtime_paths",
        lambda: (
            tmp_path,
            tmp_path,
            tmp_path / "opendm-python",
            tmp_path / "dexbotic-python",
        ),
    )
    monkeypatch.setattr(baseline_workflow, "_download_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(baseline_workflow, "_start_server", lambda *args: object())
    monkeypatch.setattr(baseline_workflow, "_wait_for_server", lambda _: None)
    monkeypatch.setattr(baseline_workflow, "_stop_server", lambda _: None)

    def opendm_suite(_, __, suite, output, ___, seed, ____):
        assert seed == protocol_seed
        _native_eval(output, "baseline")
        video = output / "videos" / "episode.mp4"
        for index in range(1, 50):
            shutil.copy2(video, output / "videos" / f"episode-{index}.mp4")
        result = _opendm_results(suite)
        (output / "results.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(baseline_workflow, "_run_suite", opendm_suite)
    prepared, baseline, candidate, metrics, report = (
        tmp_path / name
        for name in ("prepared", "baseline-out", "candidate-out", "metrics", "report")
    )
    assert (
        workflow.main(
            ["prepare", "--output-path", str(prepared), "--seed", str(protocol_seed)]
        )
        == 0
    )
    assert (
        baseline_workflow.main(
            [
                "rollout",
                "--input-path",
                str(prepared),
                "--output-path",
                str(baseline),
                "--server-device",
                "0",
                "--evaluator-device",
                "1",
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
    baseline_rollout = json.loads((baseline / "rollout.json").read_text())
    candidate_rollout = json.loads((candidate / "rollout.json").read_text())
    assert baseline_rollout["opendm_runtime"] == baseline_workflow.OPENDM_IMPLEMENTATION
    assert baseline_rollout["native_checkpoint_load_verified"] is True
    assert len(baseline_rollout["videos"]) == 200
    assert candidate_rollout["dm05_runtime"] == runtime
    assert candidate_rollout["native_checkpoint_load_verified"] is True


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
    assert (
        "'/opt/lerobot/dm05-source[training,libero,dm05]' 'hf-libero==0.1.4'"
        in instructions
    )
    assert "Lifelong-Robot-Learning/LIBERO.git" in instructions
    assert "checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01" in instructions
    assert (
        "NPA_DM05_LIBERO_BENCHMARK_ROOT=/opt/lerobot/libero-benchmark/libero/libero"
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
    candidate = _checkpoint(tmp_path / "published-candidate", "candidate")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_: str(candidate)),
    )

    assert workflow._download_checkpoint("candidate", tmp_path) == candidate

    config_path = candidate / "config.json"
    config = json.loads(config_path.read_text())
    config["output_features"]["action"]["shape"] = [6]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="candidate DM05 checkpoint"):
        workflow._download_checkpoint("candidate", tmp_path)


def test_generic_predecessor_is_rejected_without_an_action_state_adapter(tmp_path):
    baseline = _checkpoint(tmp_path / "baseline", "documented_predecessor")
    candidate = _checkpoint(tmp_path / "candidate", "candidate")

    with pytest.raises(ValueError, match="not LIBERO-compatible"):
        workflow._require_libero_checkpoint_contract("documented_predecessor", baseline)
    workflow._require_libero_checkpoint_contract("candidate", candidate)


def test_opendm_baseline_invokes_official_overrides_and_rejects_wrong_norm_shape(
    tmp_path,
):
    command = baseline_workflow._server_command(tmp_path / "checkpoint")

    assert command[:6] == [
        "bash",
        "script/dm05_launcher.sh",
        "--exp",
        "playground/dm05_libero.py",
        "--task",
        "inference",
    ]
    assert "--model-config.chunk-size" in command
    assert "--inference-config.output-action-dim" in command
    protocol = workflow._protocol(seed=11, episodes_per_task=5)
    baseline_workflow._require_protocol(protocol)
    evaluator = baseline_workflow._evaluator_command(
        "libero_goal",
        tmp_path / "native",
        protocol["episodes_per_task"],
        protocol["seed"],
    )
    overrides = {
        evaluator[index + 1]: evaluator[index + 2]
        for index, value in enumerate(evaluator)
        if value == "--set"
    }
    assert overrides == {
        "benchmark": "libero_goal",
        "base_url": "http://127.0.0.1:7891",
        "output_dir": str(tmp_path / "native"),
        "num_trails_per_task": "5",
        "seed": "11",
    }
    checkpoint = _opendm_checkpoint(tmp_path / "checkpoint-contract")
    assert (
        baseline_workflow._require_checkpoint_contract(checkpoint)[
            "effective_chunk_size"
        ]
        == 10
    )
    normalization = json.loads((checkpoint / "norm_stats.json").read_text())
    normalization["norm_stats"]["action"]["std"] = [1.0] * 8
    (checkpoint / "norm_stats.json").write_text(json.dumps(normalization))
    with pytest.raises(baseline_workflow.OpenDMBaselineError, match="vectors disagree"):
        baseline_workflow._require_checkpoint_contract(checkpoint)


@pytest.mark.parametrize("invalid_seed", [True, -1, "11", None, 11.0])
def test_opendm_baseline_rejects_an_invalid_sealed_seed(invalid_seed):
    protocol = workflow._protocol(seed=7, episodes_per_task=5)
    protocol["seed"] = invalid_seed

    with pytest.raises(baseline_workflow.OpenDMBaselineError, match="protocol seed"):
        baseline_workflow._require_protocol(protocol)


def test_opendm_private_runtime_manifest_and_bootstrap_are_exactly_bound():
    docker_root = Path(__file__).resolve().parents[2] / "docker/workbench/lerobot"
    manifest = json.loads(
        (docker_root / "dm05-opendm-baseline-runtime-manifest.json").read_text()
    )
    instructions = (
        docker_root / "Dockerfile.dm05-opendm-baseline-validation"
    ).read_text(encoding="utf-8")

    assert manifest == baseline_workflow.OPENDM_IMPLEMENTATION
    assert 'validation-disposition="operator-private-no-publication"' in instructions
    assert "checkout --detach 7d52f1591437332cb0157be3303c1c46da811344" in instructions
    assert "checkout --detach 789b87f50d9fadc7663d2e8bac057941221aab81" in instructions
    assert "checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01" in instructions
    assert "apt-get install" not in instructions
    assert "virtualenv==20.26.6" in instructions
    assert (
        "python3.10 -m virtualenv --no-download --python=/usr/bin/python3.10 "
        "/opt/opendm-venv"
    ) in instructions
    assert (
        "python3.10 -m virtualenv --no-download --python=/usr/bin/python3.10 "
        "/opt/libero-venv"
    ) in instructions
    bootstrap_block = instructions[
        instructions.index(
            "RUN python3.10 -m pip install --no-cache-dir 'virtualenv==20.26.6'"
        ) : instructions.index("\n\nUSER ubuntu")
    ]
    opendm_dependencies_block = instructions[
        instructions.index(
            "RUN /opt/opendm-venv/bin/python --version"
        ) : instructions.index("\n\n# Dexbotic's upstream local LIBERO guide")
    ]
    libero_dependencies_block = instructions[
        instructions.index(
            "RUN /opt/libero-venv/bin/python --version"
        ) : instructions.index("\n\nCOPY --chown=ubuntu:ubuntu")
    ]
    for environment, dependency_block in (
        ("/opt/opendm-venv", opendm_dependencies_block),
        ("/opt/libero-venv", libero_dependencies_block),
    ):
        assert f"{environment}/bin/python --version" in dependency_block
        assert f"{environment}/bin/python -m pip install" in dependency_block
        assert "python3.10 -m venv" not in dependency_block
        assert instructions.index(bootstrap_block) < instructions.index(
            dependency_block
        )
    assert "python3.10 -m venv /opt/opendm-venv" not in instructions
    assert "python3.10 -m venv /opt/libero-venv" not in instructions
    bddl_install = libero_dependencies_block.index("'bddl==1.0.1'")
    future_install = libero_dependencies_block.index("'future==1.0.0'")
    matplotlib_install = libero_dependencies_block.index("'matplotlib==3.5.3'")
    editable_libero_install = libero_dependencies_block.index(
        "-e /opt/dexbotic-benchmark/libero"
    )
    assert bddl_install < future_install < matplotlib_install < editable_libero_install
    assert (
        "LIBERO_CONFIG_PATH=/opt/dexbotic-benchmark/libero/libero/libero "
        "PYTHONPATH=/opt/dexbotic-benchmark:/opt/dexbotic-benchmark/libero "
        "/opt/libero-venv/bin/python"
    ) in instructions
    parent_jwt_sanitizer = instructions[
        instructions.index(
            "# The pinned parent includes a static, sample-media JWT URL"
        ) : instructions.index(
            "\n\nCOPY --chown=ubuntu:ubuntu",
            instructions.index(
                "# The pinned parent includes a static, sample-media JWT URL"
            ),
        )
    ]
    assert "USER root" in parent_jwt_sanitizer
    assert (
        "/opt/lerobot/venv/lib/python3.12/site-packages/skimage/data/_fetchers.py"
        in parent_jwt_sanitizer
    )
    assert "grep -Eoc" in parent_jwt_sanitizer
    assert '" -eq 1' in parent_jwt_sanitizer
    assert "sed -Ei" in parent_jwt_sanitizer
    assert "?token=redacted-static-jwt" in parent_jwt_sanitizer
    assert "python3.10 -m py_compile" in parent_jwt_sanitizer
    assert parent_jwt_sanitizer.rstrip().endswith("USER ubuntu")
    assert (
        "benchmark_root: /opt/dexbotic-benchmark/libero/libero/libero" in instructions
    )
    assert "/opt/opendm-venv/bin/python -m pip uninstall -y wandb" in instructions
    assert (
        "NPA_DM05_OPENDM_RUNTIME_MANIFEST=/opt/opendm/dm05-libero-baseline-runtime.json"
        in instructions
    )


def test_opendm_parent_jwt_sanitizer_executes_the_dockerfile_ere(tmp_path):
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/lerobot/Dockerfile.dm05-opendm-baseline-validation"
    )
    instructions = dockerfile.read_text(encoding="utf-8")
    fetcher = tmp_path / "_fetchers.py"
    synthetic_token = "eyJmaXh0dXJlIjoxfQ.eyJub25zZWNyZXQiOnRydWV9.signature"
    unrelated_before = "before = 'leave-this-content-alone'\n"
    unrelated_after = "after = 'leave-this-content-alone-too'\n"
    fetcher.write_text(
        f"{unrelated_before}sample_url = '?token={synthetic_token}'\n{unrelated_after}",
        encoding="utf-8",
    )
    grep_pattern = r"\?token=eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    sed_expression = (
        r"s#(\?token=)eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
        r"#\1redacted-static-jwt#"
    )
    image_fetcher = (
        "/opt/lerobot/venv/lib/python3.12/site-packages/skimage/data/_fetchers.py"
    )
    sanitizer_start = f"RUN fetcher={image_fetcher}"
    sanitizer_end = "\nUSER ubuntu"
    sanitizer = instructions[
        instructions.index(sanitizer_start) + len("RUN ") : instructions.index(
            sanitizer_end, instructions.index(sanitizer_start)
        )
    ].replace(f"fetcher={image_fetcher}", 'fetcher="$fetcher"', 1)

    assert grep_pattern in instructions
    assert sed_expression in instructions
    completed = subprocess.run(
        ["sh", "-ceu", sanitizer, "sh"],
        capture_output=True,
        check=False,
        env={**os.environ, "fetcher": str(fetcher)},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    sanitized = fetcher.read_text(encoding="utf-8")
    assert "?token=redacted-static-jwt" in sanitized
    assert synthetic_token not in sanitized
    assert unrelated_before in sanitized
    assert unrelated_after in sanitized


def test_opendm_evaluator_environment_supplies_pinned_libero_config_without_stdin(
    tmp_path, monkeypatch
):
    dexbotic_root = tmp_path / "dexbotic-benchmark"
    config = dexbotic_root / "libero" / "libero" / "libero" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("benchmark_root: pinned\n", encoding="utf-8")
    monkeypatch.setenv(baseline_workflow.LIBERO_CONFIG_PATH_ENV, "/stale/config")

    environment = baseline_workflow._evaluator_environment(
        Path("/opt/libero-venv/bin/python"), dexbotic_root
    )

    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(dexbotic_root),
        str(dexbotic_root / "libero"),
    ]
    assert environment[baseline_workflow.LIBERO_CONFIG_PATH_ENV] == str(config.parent)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from pathlib import Path; "
                "config = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'; "
                "assert config.read_text(encoding='utf-8') == 'benchmark_root: pinned\\n'; "
                "print('pinned-config-selected')"
            ),
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "pinned-config-selected\n"


def test_opendm_evaluator_environment_rejects_missing_pinned_libero_config(tmp_path):
    with pytest.raises(
        baseline_workflow.OpenDMBaselineError, match="configuration is missing"
    ):
        baseline_workflow._evaluator_environment(
            Path("/opt/libero-venv/bin/python"), tmp_path / "dexbotic-benchmark"
        )


def test_opendm_server_start_does_not_require_evaluator_libero_config(
    tmp_path, monkeypatch
):
    opendm_root = tmp_path / "opendm"
    opendm_root.mkdir()
    captured: dict[str, object] = {}
    process = object()
    monkeypatch.setenv(baseline_workflow.LIBERO_CONFIG_PATH_ENV, "/stale/config")

    def start(command, *, cwd, env):
        captured.update({"command": command, "cwd": cwd, "env": env})
        return process

    monkeypatch.setattr(baseline_workflow.subprocess, "Popen", start)

    assert (
        baseline_workflow._start_server(
            opendm_root,
            Path("/opt/opendm-venv/bin/python"),
            tmp_path / "checkpoint",
            "0",
        )
        is process
    )
    assert captured["cwd"] == opendm_root
    assert baseline_workflow.LIBERO_CONFIG_PATH_ENV not in captured["env"]


def test_libero_config_uses_installed_assets_without_interactive_setup(
    tmp_path, monkeypatch
):
    package_root = tmp_path / "site-packages" / "libero"
    packaged_root = package_root / "libero"
    benchmark_root = tmp_path / "pinned-libero" / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (packaged_root / name).mkdir(parents=True)
        (benchmark_root / name).mkdir(parents=True)
    monkeypatch.setattr(
        workflow.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(
            submodule_search_locations=[str(package_root)]
        ),
    )
    monkeypatch.setenv(workflow.DM05_LIBERO_BENCHMARK_ROOT_ENV, str(benchmark_root))

    config_root = workflow._write_noninteractive_libero_config(tmp_path / "run")

    config = json.loads((config_root / "config.yaml").read_text())
    assert config["benchmark_root"] == str(benchmark_root)
    assert config["assets"] == str(benchmark_root / "assets")
    assert config["bddl_files"] == str(benchmark_root / "bddl_files")
    assert config["init_states"] == str(benchmark_root / "init_files")
    assert Path(config["datasets"]).is_dir()


def test_readiness_record_is_bound_to_the_comparison_workflow_bytes():
    readiness = load_readiness_record(WORKFLOW.with_suffix(".readiness.json"))

    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"

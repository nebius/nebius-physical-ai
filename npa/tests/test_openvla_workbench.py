"""Native, no-GPU contract tests for the OpenVLA-OFT LIBERO pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from npa.deploy import images
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workflows.byof import openvla_pipeline as pipe


def _toolref_stage_argv(name: str) -> list[str]:
    entry = TOOL_CATALOG[name]

    def replace(match: re.Match[str]) -> str:
        values = {
            "config.task_suite": "libero_spatial",
            "config.batch_size": "8",
            "config.max_steps": "150005",
            "config.learning_rate": "0.0005",
            "config.lora_rank": "32",
            "config.processes": "8",
            "config.trials_per_task": "50",
            "config.seed": "7",
        }
        return values.get(match.group(1), "DUMMY")

    return [re.sub(r"{{(.*?)}}", replace, item) for item in entry.argv_template]


def _write_rollout_bundle(root: Path, *, successes: int = 3, episodes: int = 5) -> Path:
    source = root / "rollout-source"
    source.mkdir()
    (source / "rollout.json").write_text(
        json.dumps(
            {
                "schema": pipe.ROLLOUT_SCHEMA,
                "status": "succeeded",
                "task_suite": "libero_spatial",
                "result": {
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": successes / episodes,
                },
            }
        )
    )
    report = pipe.publish_bundle(
        source,
        str(root / "rollout-output"),
        {"schema": pipe.ROLLOUT_SCHEMA, "status": "succeeded"},
        "rollout-manifest.json",
    )
    assert report["status"] == "succeeded"
    return root / "rollout-output" / "rollout-manifest.json"


def test_prepare_reads_actual_rlds_file_and_publishes_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "episode-000.tfrecord").write_bytes(b"not-a-placeholder-rlds-record")
    (dataset / "dataset_statistics.json").write_text('{"action": {"mean": [0]}}')
    output = tmp_path / "prepared"
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(pipe, "bootstrap_runtime", lambda root: runtime)
    monkeypatch.setattr(
        pipe,
        "_verify_dlimp_rlds_read",
        lambda python, records: {
            "trajectories_read": 1,
            "feature_keys": ["action", "observation"],
        },
    )
    report = pipe.prepare(
        pipe.PrepareConfig(
            str(dataset),
            str(output),
            "libero_spatial_no_noops",
            "libero_spatial",
            str(runtime),
        )
    )
    assert report["schema"] == pipe.PREPARE_SCHEMA
    normalization = json.loads((output / "normalization.json").read_text())
    assert normalization["proprio_dim"] == 8
    assert normalization["action_mode"] == "continuous_l1"
    assert normalization["rlds_files"][0]["sha256"]
    assert normalization["dlimp_rlds_probe"] == {
        "trajectories_read": 1,
        "feature_keys": ["action", "observation"],
    }


def test_stock_decoder_bundle_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stock"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"stock")
    with pytest.raises(pipe.OpenVLAPipelineError, match="stock OpenVLA"):
        pipe._validate_oft_components(checkpoint)


def test_finetune_launcher_uses_upstream_eight_process_torchrun(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    (runtime / "venv" / "bin").mkdir(parents=True)
    (runtime / "source" / "vla-scripts").mkdir(parents=True)
    (runtime / "venv" / "bin" / "torchrun").touch()
    (runtime / "source" / pipe.UPSTREAM_FINETUNE_SCRIPT).touch()
    command = pipe._upstream_torchrun_command(
        runtime, pipe.UPSTREAM_FINETUNE_SCRIPT, 8, ["--vla_path", "/model"]
    )
    assert command[1:4] == ["--standalone", "--nnodes=1", "--nproc_per_node=8"]
    assert command[4].endswith(pipe.UPSTREAM_FINETUNE_SCRIPT)


def test_finetune_arguments_match_the_documented_decay_flag(tmp_path: Path) -> None:
    """The upstream decay flag accepts one value before the next flag."""
    arguments = pipe._finetune_arguments(
        pipe.TrainConfig("prepared.json", "output", "runtime"),
        tmp_path / "model",
        tmp_path / "rlds",
        "libero_spatial_no_noops",
        tmp_path / "results",
    )
    decay_index = arguments.index("--num_steps_before_decay")
    assert arguments[decay_index + 1] == "100000"
    assert arguments[decay_index + 2] == "--max_steps"
    assert arguments[arguments.index("--dataset_name") + 1] == "libero_spatial_no_noops"


def test_licensed_dlimp_override_is_one_line_and_hash_bound(tmp_path: Path) -> None:
    """OFT preserves its fork's sole behavior delta without fetching that fork."""
    dlimp = tmp_path / "dlimp"
    target = dlimp / pipe.DLIMP_DETERMINISTIC_PATH
    target.parent.mkdir(parents=True)
    target.write_text(
        "options.autotune.enabled = True\noptions.deterministic = False\n",
        encoding="utf-8",
    )
    result = pipe._apply_dlimp_deterministic_override(dlimp)
    assert (
        target.read_text(encoding="utf-8").splitlines()[-1]
        == "options.deterministic = True"
    )
    assert result["path"] == "dlimp/dataset.py"
    assert result["before_sha256"] != result["after_sha256"]


def test_runtime_install_graph_cannot_follow_the_unlicensed_dlimp_fork(
    tmp_path: Path,
) -> None:
    commands = pipe._runtime_install_commands(
        tmp_path / "python",
        tmp_path / "source",
        tmp_path / "transformers",
        tmp_path / "dlimp",
        tmp_path / "libero",
    )
    editable = [command for command in commands if "-e" in command]
    assert all("--no-deps" in command for command in editable[:3])
    assert all("dlimp_openvla" not in " ".join(command) for command in commands)
    cuda_install = commands[1]
    assert "--index-url" in cuda_install
    assert pipe.PYTORCH_CUDA_INDEX_URL in cuda_install
    assert set(pipe.PYTORCH_CUDA_REQUIREMENTS).issubset(cuda_install)
    assert all(
        requirement not in pipe.OFT_PYPI_DIRECT_DEPENDENCIES
        for requirement in pipe.PYTORCH_CUDA_REQUIREMENTS
    )
    assert str(tmp_path / "dlimp") in editable[1]
    assert pipe.DLIMP_REPOSITORY == "https://github.com/kvablack/dlimp.git"
    assert re.fullmatch(r"[0-9a-f]{40}", pipe.DLIMP_REVISION)
    assert "tensorflow-metadata==1.14.0" in pipe.OFT_PYPI_DIRECT_DEPENDENCIES
    assert pipe.TENSORFLOW_NUMPY_REQUIREMENT in pipe.OFT_PYPI_DIRECT_DEPENDENCIES
    # LIBERO's requirements are resolved last and previously upgraded NumPy 2,
    # which makes TensorFlow 2.15 fail before dlimp can read RLDS.  Keep the
    # compatible pin in that exact resolver call rather than trusting its
    # earlier installation to survive.
    assert commands[-1][-1] == pipe.TENSORFLOW_NUMPY_REQUIREMENT


def test_neutral_image_is_private_validation_quarantined() -> None:
    """A lawful bootstrap remains ineligible for public validation or release."""
    assert images.CONTAINER_IMAGE_NAMES["openvla-oft"] == "npa-openvla-oft"
    assert "openvla-oft" in images.NEUTRAL_UNBUILT_CANDIDATE_TOOLS
    assert images.supported_tool_version("openvla-oft").endswith("-unbuilt")
    assert not images.is_publicly_redistributable("openvla-oft")
    assert "openvla-oft" in images.PUBLICATION_QUARANTINE_TOOLS


def test_neutral_image_preserves_nonroot_access_to_its_notices() -> None:
    """Runtime attribution stays readable under Kubernetes' numeric non-root user."""
    dockerfile = (
        Path(__file__).resolve().parents[1] / "docker/workbench/openvla-oft/Dockerfile"
    ).read_text(encoding="utf-8")
    assert (
        "install -d -m 0755 -o root -g root /usr/share/doc/npa-openvla-oft"
        in dockerfile
    )
    assert "\nUSER 1000\n" in dockerfile
    assert 'CMD ["sleep", "infinity"]' in dockerfile


def test_workflow_resources_declare_the_skypilot_task_container() -> None:
    """Every native stage gets SkyPilot's named non-root Kubernetes task container."""
    workflow = (
        Path(__file__).resolve().parents[2]
        / "workflows/testing/openvla-oft-libero.yaml"
    )
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    for resource_name, resource in document["resources"].items():
        pod_spec = resource["kubernetes"]["pod_config"]["spec"]
        assert pod_spec["automountServiceAccountToken"] is False, resource_name
        container = pod_spec["containers"][0]
        assert container["name"] == "ray-node", resource_name
        # Dropping every capability prevents sudo from changing group during
        # SkyPilot's required non-root bootstrap.
        assert "allowPrivilegeEscalation" not in container["securityContext"]
        assert container["securityContext"] == {
            "runAsNonRoot": True,
            "privileged": False,
        }, resource_name


def test_private_build_contract_refuses_official_public_pushes() -> None:
    build_script = (
        Path(__file__).resolve().parents[1] / "docker/workbench/openvla-oft/build.sh"
    ).read_text(encoding="utf-8")
    assert "--registry PRIVATE_HOST/PATH --push" in build_script
    assert "--push requires --registry" in build_script
    assert "cannot use the official public registry" in build_script
    assert "--provenance=mode=max --sbom=true" in build_script


def test_immutable_model_cache_requires_matching_ready_identity(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    identity = hashlib.sha256(
        f"{pipe.DEFAULT_MODEL_ID}@{pipe.MODEL_REVISION}".encode()
    ).hexdigest()
    snapshot = runtime / "models" / identity
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "ready.json").write_text(
        json.dumps(
            {
                "repo_id": pipe.DEFAULT_MODEL_ID,
                "revision": pipe.MODEL_REVISION,
                "status": "ready",
            }
        )
    )
    assert (
        pipe._materialize_model_snapshot(
            runtime, pipe.DEFAULT_MODEL_ID, pipe.MODEL_REVISION
        )
        == snapshot
    )


def test_runtime_identity_requires_resolved_dependency_provenance(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "ready.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "dependency_inventory_sha256": "a" * 64,
            }
        )
    )
    assert pipe._runtime_identity(runtime)["status"] == "ready"
    (runtime / "ready.json").write_text('{"status": "ready"}')
    with pytest.raises(pipe.OpenVLAPipelineError, match="dependency provenance"):
        pipe._runtime_identity(runtime)


def test_training_lora_rank_is_bound_into_rollout_provenance() -> None:
    """A rollout must preserve the rank chosen by its source training bundle."""
    assert pipe._training_lora_rank({"algorithm": {"lora_rank": 16}}) == 16
    with pytest.raises(pipe.OpenVLAPipelineError, match="invalid LoRA"):
        pipe._training_lora_rank({"algorithm": {"lora_rank": 0}})


def test_train_config_rejects_moving_model_reference() -> None:
    with pytest.raises(pipe.OpenVLAPipelineError, match="immutable 40-character"):
        pipe.TrainConfig(
            "prepared.json", "output", "runtime", model_revision="main"
        ).validate()


def test_parse_rollout_log_requires_consistent_measured_counts(tmp_path: Path) -> None:
    log = tmp_path / "eval.txt"
    log.write_text(
        "Total episodes: 5\nTotal successes: 3\nOverall success rate: 0.6000 (60.0%)\n"
    )
    assert pipe._parse_rollout_log(log) == {
        "episodes": 5,
        "successes": 3,
        "success_rate": 0.6,
    }
    log.write_text(
        "Total episodes: 5\nTotal successes: 4\nOverall success rate: 0.6000 (60.0%)\n"
    )
    with pytest.raises(pipe.OpenVLAPipelineError, match="disagrees"):
        pipe._parse_rollout_log(log)


def test_evaluate_then_visualize_preserves_real_rollout_metric(tmp_path: Path) -> None:
    rollout_manifest = _write_rollout_bundle(tmp_path)
    evaluated = tmp_path / "evaluated"
    evaluation = pipe.evaluate(
        pipe.EvaluateConfig(str(rollout_manifest), str(evaluated))
    )
    assert evaluation["schema"] == pipe.EVALUATION_SCHEMA
    metrics = json.loads((evaluated / "metrics.json").read_text())
    assert metrics["success_rate"] == pytest.approx(0.6)
    compared = tmp_path / "compared"
    comparison = pipe.visualize(
        pipe.VisualizeConfig(str(evaluated / "evaluation.json"), str(compared))
    )
    assert comparison["schema"] == pipe.COMPARISON_SCHEMA
    assert "3/5" in (compared / "comparison.svg").read_text()
    assert "closed_loop_success_rate" in (compared / "comparison.csv").read_text()


@pytest.mark.parametrize("suite", sorted(pipe.OFFICIAL_SUITE_CHECKPOINTS))
def test_official_suite_contract_has_immutable_component_identity(suite: str) -> None:
    entry = pipe.OFFICIAL_SUITE_CHECKPOINTS[suite]
    assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
    assert entry["component_step"].isdigit()
    assert entry["repo_id"].startswith("moojink/openvla-7b-oft-finetuned-")
    assert (
        entry["action_head"] == f"action_head--{entry['component_step']}_checkpoint.pt"
    )
    assert entry["proprio_projector"] == (
        f"proprio_projector--{entry['component_step']}_checkpoint.pt"
    )
    assert entry["lora_adapter"] == "lora_adapter"


def test_pipeline_main_dispatches_prepare(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The module entrypoint dispatches one parsed command without mutation bugs."""
    monkeypatch.setattr(
        pipe, "prepare", lambda config: {"dataset": config.dataset_name}
    )
    result = pipe.main(
        [
            "prepare",
            "--dataset-uri",
            "s3://bucket/rlds/",
            "--dataset-name",
            "libero_spatial_no_noops",
            "--task-suite",
            "libero_spatial",
            "--runtime-root",
            "/cache/openvla-oft",
            "--output-uri",
            "s3://bucket/prepare/",
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["dataset"] == "libero_spatial_no_noops"


@pytest.mark.parametrize(
    "tool_name",
    [
        "workbench.openvla.prepare",
        "workbench.openvla.train",
        "workbench.openvla.rollout",
        "workbench.openvla.evaluate",
        "workbench.openvla.visualize",
    ],
)
def test_toolref_argv_parses_against_native_pipeline(tool_name: str) -> None:
    argv = _toolref_stage_argv(tool_name)
    assert argv[:3] == ["python3", "-m", "npa.workflows.byof.openvla_pipeline"]
    pipe.build_parser().parse_args(argv[3:])

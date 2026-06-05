from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.orchestration.skypilot.gpu_catalog import NebiusGpuCatalog, NebiusGpuResolution
from npa.workflows.sim_to_real import (
    DEFAULT_GPU_FAILOVER,
    DEFAULT_GPU_TYPE,
    FeedbackResult,
    SimToRealConfig,
    Tier,
    accelerator_candidates,
    artifact_uris,
    build_config_from_env,
    default_policy_image,
    default_s3_prefix,
    feedback_to_training_signal,
    generate_raw_envs,
    outer_loop_decision,
    parse_feedback_result,
    run_real_lerobot_loop,
    seeded_train_heldout_split,
)
from npa.workbench.lerobot.policy_container import (
    CheckpointValidationResult,
    LeRobotEvalResult,
    LeRobotImportResult,
    LeRobotTrainingResult,
    parse_feedback_batch,
    run_feedback_training_step,
)
from npa.workflows.lerobot_dataset import seeded_episode_split, summarize_lerobot_dataset


ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sim-to-real-pipeline.yaml"
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_sim_to_real_pipeline.py"
QUICKSTART_PATH = ROOT / "npa" / "scripts" / "run_sim_to_real_quickstart.py"


def _nebius_catalog() -> NebiusGpuCatalog:
    return NebiusGpuCatalog(
        {
            "H100": frozenset({1, 8}),
            "H200": frozenset({1, 8}),
            "B200": frozenset({8}),
            "L40S": frozenset({1, 2, 4}),
        }
    )


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_sim_to_real_pipeline", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_quickstart_module():
    spec = importlib.util.spec_from_file_location("run_sim_to_real_quickstart", QUICKSTART_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _docs() -> list[dict]:
    return [doc for doc in yaml.safe_load_all(YAML_PATH.read_text(encoding="utf-8")) if doc is not None]


def _write_lerobot_fixture(root: Path) -> Path:
    raw = root / "raw"
    for episode_index in range(2):
        episode = raw / f"episode_{episode_index:06d}"
        episode.mkdir(parents=True)
        state = np.zeros((3, G1_STATE_DIM), dtype=np.float32)
        state[:, 0] = np.linspace(0.0, 1.0 + episode_index, 3, dtype=np.float32)
        actions = state + 0.1
        np.save(episode / "state.npy", state)
        np.save(episode / "actions.npy", actions)

    dataset = convert(raw, root / "lerobot", fps=10, task="sim-to-real unit fixture")
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"]["observation.images.ego_view"] = {
        "dtype": "video",
        "shape": [4, 4, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 10.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return dataset


def test_seeded_raw_env_split_is_deterministic_80_20() -> None:
    envs = generate_raw_envs(count=10, seed=7)
    train, heldout = seeded_train_heldout_split(envs, train_fraction=0.8, seed=7)
    train_again, heldout_again = seeded_train_heldout_split(envs, train_fraction=0.8, seed=7)

    assert len(train) == 8
    assert len(heldout) == 2
    assert [env.env_id for env in train] == [env.env_id for env in train_again]
    assert [env.env_id for env in heldout] == [env.env_id for env in heldout_again]


def test_feedback_parse_guard_and_training_signal() -> None:
    feedback = parse_feedback_result(
        {
            "success": True,
            "score": 0.91,
            "rationale": "Task completed.",
            "critique": "Stable rollout.",
        }
    )
    signal = feedback_to_training_signal(feedback)
    decision = outer_loop_decision(feedback.score, 0.8, "s3://bucket/run/checkpoint/")

    assert feedback.success is True
    assert feedback.score == 0.91
    assert signal["scalar_reward"] == 0.91
    assert signal["natural_language_critique"] == "Stable rollout."
    assert decision["decision"] == "promote_checkpoint"


def test_real_loop_trains_evals_and_records_measured_trend(monkeypatch, tmp_path: Path) -> None:
    from npa.workflows import eval_backends as eval_backends_module
    from npa.workflows import sim_to_real as sim_to_real_module

    dataset = _write_lerobot_fixture(tmp_path / "fixture")
    base_summary = summarize_lerobot_dataset(
        dataset,
        source_uri=str(dataset),
        repo_id="lerobot/pusht",
        revision="test",
    )
    summary = replace(base_summary, loaded_with_lerobot_dataset=True, lerobot_dataset_error="")
    checkpoint = tmp_path / "policy-training" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"real-weights")
    eval_scores = iter([0.25, 0.62])

    def fake_train(**kwargs):
        return LeRobotTrainingResult(
            status="success",
            command=["lerobot-train", f"--steps={kwargs['steps']}"],
            output_dir=str(kwargs["output_dir"]),
            checkpoint_path=str(checkpoint),
            steps=kwargs["steps"],
            resume=kwargs["resume"],
            log_path=str(kwargs["log_path"]),
            duration_seconds=1.0,
            exit_code=0,
            checkpoint_validation={
                "status": "loadable",
                "checkpoint_path": str(checkpoint),
                "weight_file": str(checkpoint / "model.safetensors"),
                "tensor_count": 1,
                "parameter_count": 1,
                "bytes": 12,
            },
        )

    def fake_eval(**kwargs):
        score = next(eval_scores)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        eval_info = output_dir / "eval_info.json"
        eval_info.write_text(
            json.dumps({"overall": {"pc_success": score, "avg_sum_reward": score, "n_episodes": 2}}),
            encoding="utf-8",
        )
        return LeRobotEvalResult(
            status="success",
            backend=kwargs["env_type"],
            command=["lerobot-eval"],
            output_dir=str(output_dir),
            eval_info_path=str(eval_info),
            score=score,
            metric_name="pc_success",
            pc_success=score,
            avg_sum_reward=score,
            avg_max_reward=score,
            n_episodes=2,
            log_path=str(kwargs["log_path"]),
            duration_seconds=1.0,
            exit_code=0,
            raw_metrics={"overall": {"pc_success": score}},
        )

    monkeypatch.setattr(
        sim_to_real_module,
        "assert_lerobot_importable",
        lambda: LeRobotImportResult("ok", "0.5.1", "/tmp/lerobot", "lerobot.datasets.lerobot_dataset.LeRobotDataset"),
    )
    monkeypatch.setattr(sim_to_real_module, "summarize_lerobot_dataset", lambda *args, **kwargs: summary)
    monkeypatch.setattr(sim_to_real_module, "run_lerobot_training", fake_train)
    monkeypatch.setattr(eval_backends_module, "run_lerobot_eval", fake_eval)
    monkeypatch.setattr(
        sim_to_real_module,
        "validate_lerobot_checkpoint",
        lambda path: CheckpointValidationResult(
            "loadable",
            str(checkpoint),
            str(checkpoint / "model.safetensors"),
            1,
            1,
            12,
        ),
    )
    config = SimToRealConfig(
        run_id="s2r-test",
        output_dir=tmp_path,
        input_data_uri=str(dataset),
        env_count=10,
        train_steps=10,
        train_step_budget=20,
        max_training_iterations=2,
        eval_episodes=2,
        threshold=0.6,
        feedback_source="sim-env",
        feedback_type="scalar",
        eval_backend="state-success",
    )

    report = run_real_lerobot_loop(config)
    components = {component.name: component for component in report.components}

    assert report.feedback.score == 0.62
    assert report.outer_loop["trend"] == [0.25, 0.62]
    assert report.outer_loop["decision"] == "promote_checkpoint"
    assert components["lerobot_runtime_import"].tier == Tier.WORKS
    assert components["real_lerobot_dataset"].tier == Tier.WORKS
    assert components["lerobot_episode_split"].tier == Tier.WORKS
    assert components["state_success_eval"].tier == Tier.WORKS
    assert components["sim_env_feedback"].tier == Tier.WORKS
    assert components["real_training"].tier == Tier.WORKS
    assert components["real_rollout_eval"].tier == Tier.WORKS
    assert components["feedback_training_loop"].tier == Tier.WORKS
    assert report.training_signal["source"] == "sim-env"
    assert report.training_signal["feedback_type"] == "scalar"
    assert report.training_signal["score"] == 0.62
    assert (tmp_path / "lerobot-dataset-summary.json").exists()
    assert (tmp_path / "s2r-test.rrd").exists()
    assert (tmp_path / "sim-to-real-report.json").exists()


def test_artifact_layout_is_run_scoped_and_generic() -> None:
    config = SimToRealConfig(
        run_id="run-1",
        s3_bucket="bucket",
        s3_prefix=default_s3_prefix("run-1"),
        input_data_uri="s3://bucket/input/",
    )

    paths = artifact_uris(config)

    assert paths["root"] == "s3://bucket/sim-to-real/run-1/"
    assert paths["input_data"] == "s3://bucket/input/"
    assert paths["checkpoint"] == "s3://bucket/sim-to-real/run-1/checkpoints/policy/"
    assert paths["rrd"].endswith("/viz/run-1.rrd")


def test_yaml_exposes_parameterized_spine_and_feedback_contract() -> None:
    docs = _docs()

    assert len(docs) == 1
    task = docs[0]
    assert task["name"] == "sim-to-real-pipeline"
    assert task["resources"]["accelerators"] == ["H100:1", "H200:1", "L40S:1"]
    assert task["envs"]["S3_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    assert task["envs"]["NEBIUS_S3_ENDPOINT"] == "https://storage.eu-north1.nebius.cloud"
    assert task["envs"]["S3_BUCKET"] == "${S3_BUCKET}"
    assert task["envs"]["NPA_S3_BUCKET"] == "${S3_BUCKET}"
    assert task["envs"]["POLICY_IMAGE"] == "npa-lerobot-policy:0.1.0"
    assert task["envs"]["LEROBOT_DATASET_REPO_ID"] == "lerobot/pusht"
    assert task["envs"]["EVAL_BACKEND"] == "state-success"
    assert task["envs"]["FEEDBACK_SOURCE"] == "sim-env"
    assert task["envs"]["FEEDBACK_TYPE"] == "scalar"
    assert task["envs"]["NPA_GPU_TYPE"] == "H100:1"
    assert task["envs"]["NPA_GPU_FAILOVER"] == "H200:1,L40S:1"
    assert task["envs"]["BYO_FEEDBACK_MODE"] == "provided-rollout"
    assert task["envs"]["TRAIN_STEPS"] == "2000"
    assert task["envs"]["MAX_TRAINING_ITERATIONS"] == "3"
    assert "npa.workflows.sim_to_real real-loop" in task["run"]
    assert "--feedback-type" in task["run"]
    assert "--gpu-failover" in task["run"]
    assert "--attempt-s3-roundtrip" in task["run"]


def test_runner_renders_policy_image_and_vlm_eval_settings() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        policy_image="cr.example/npa-lerobot:custom",
        vlm_eval_backend="stub",
        vlm_eval_score=0.9,
    )
    task_env = docs[0]["envs"]

    assert task_env["NPA_SIM_TO_REAL_RUN_ID"] == "s2r-render"
    assert task_env["S3_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    assert task_env["S3_BUCKET"] == "bucket"
    assert task_env["NPA_S3_BUCKET"] == "bucket"
    assert task_env["POLICY_IMAGE"] == "cr.example/npa-lerobot:custom"
    assert task_env["LEROBOT_DATASET_REPO_ID"] == "lerobot/pusht"
    assert task_env["INPUT_DATA_URI"] == "s3://bucket/datasets/lerobot-pusht/"
    assert task_env["CHECKPOINT_URI"] == "s3://bucket/sim-to-real/s2r-render/checkpoints/policy/"
    assert task_env["EVAL_BACKEND"] == "state-success"
    assert task_env["FEEDBACK_SOURCE"] == "sim-env"
    assert task_env["FEEDBACK_TYPE"] == "scalar"
    assert task_env["NPA_GPU_TYPE"] == DEFAULT_GPU_TYPE
    assert task_env["NPA_GPU_FAILOVER"] == DEFAULT_GPU_FAILOVER
    assert task_env["GPU"] == "H100:1,H200:1,L40S:1"
    assert task_env["VLM_EVAL_BACKEND"] == "stub"
    assert task_env["VLM_EVAL_SCORE"] == "0.9"
    assert "image_id" not in docs[0]["resources"]


def test_runner_renders_ordered_gpu_failover_resources() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        gpu="H100:1",
        gpu_failover="H200:1,L40S:1",
    )

    assert docs[0]["resources"]["accelerators"] == ["H100:1", "H200:1", "L40S:1"]
    assert docs[0]["envs"]["NPA_GPU_TYPE"] == "H100:1"
    assert docs[0]["envs"]["NPA_GPU_FAILOVER"] == "H200:1,L40S:1"


def test_runner_can_render_nebius_task_cloud_fallback() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        task_cloud="nebius",
        gpu="H100:1",
        gpu_failover="H200:1,A100:1,L40S:1,RTX6000:1",
        gpu_catalog=_nebius_catalog(),
    )

    assert docs[0]["resources"]["cloud"] == "nebius"
    assert docs[0]["resources"]["region"] == "eu-north1"
    assert docs[0]["resources"]["accelerators"] == ["H100:1", "H200:1", "L40S:1"]
    assert docs[0]["resources"]["cpus"] == "16+"
    assert docs[0]["resources"]["memory"] == "64+"
    assert "image_id" not in docs[0]["resources"]


def test_sdk_env_config_reads_gpu_eval_and_feedback_knobs(monkeypatch) -> None:
    monkeypatch.setenv("NPA_GPU_TYPE", "B200:8")
    monkeypatch.setenv("NPA_GPU_FAILOVER", "L40S,H200")
    monkeypatch.setenv("EVAL_BACKEND", "heldout-metrics")
    monkeypatch.setenv("FEEDBACK_SOURCE", "sim-env")
    monkeypatch.setenv("FEEDBACK_TYPE", "pass-fail")
    monkeypatch.setenv("BYO_FEEDBACK_MODE", "self-rollout")

    config = build_config_from_env(run_id="sdk-env")

    assert config.gpu == "B200:8"
    assert config.gpu_failover == "L40S,H200"
    assert accelerator_candidates(config.gpu, config.gpu_failover) == ["B200:8", "L40S:1", "H200:1"]
    assert config.eval_backend == "heldout-metrics"
    assert config.feedback_source == "sim-env"
    assert config.feedback_type == "pass-fail"
    assert config.byo_feedback_mode == "self-rollout"


def test_runner_passes_controller_backend_to_submit(monkeypatch, tmp_path: Path, capsys) -> None:
    wrapper = _load_wrapper_module()
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    captured = {}

    def fake_submit_workflow(yaml_path, run_id, **kwargs):
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        captured["docs"] = [
            doc for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8")) if doc is not None
        ]
        return wrapper.WorkflowResult(
            status="SUBMITTED",
            job_id="42",
            returncode=0,
            log_paths={"config": str(tmp_path / "config.yaml")},
        )

    def fake_workflow_status(job_id, **kwargs):
        return wrapper.WorkflowResult(status="SUCCEEDED", job_id=job_id, returncode=0)

    monkeypatch.setattr(wrapper, "submit_workflow", fake_submit_workflow)
    monkeypatch.setattr(wrapper, "workflow_status", fake_workflow_status)
    monkeypatch.setattr(
        wrapper,
        "resolve_nebius_gpu_preferences",
        lambda gpu, gpu_failover, **kwargs: NebiusGpuResolution(
            selected="H100:1",
            accelerators=("H100:1", "H200:1", "L40S:1"),
            rejected=(),
            catalog=_nebius_catalog(),
        ),
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    rc = wrapper.main(
        [
            "--yaml",
            str(YAML_PATH),
            "--run-id",
            "s2r-submit",
            "--controller-backend",
            "nebius",
            "--task-cloud",
            "nebius",
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
        ]
    )

    assert rc == 0
    capsys.readouterr()
    assert captured["run_id"] == "s2r-submit"
    assert captured["kwargs"]["controller_backend"] == "nebius"
    assert captured["kwargs"]["secret_envs"] == ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    assert captured["docs"][0]["envs"]["NPA_SIM_TO_REAL_RUN_ID"] == "s2r-submit"
    assert captured["docs"][0]["resources"]["cloud"] == "nebius"
    assert "image_id" not in captured["docs"][0]["resources"]


def test_quickstart_uses_credentials_h100_defaults_and_tears_down(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    quickstart = _load_quickstart_module()
    run_id = "s2r-quickstart-test-20260604T000000Z"
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "\n".join(
            [
                "storage:",
                "  aws_access_key_id: quickstart-access-key",
                "  aws_secret_access_key: quickstart-secret-key",
                "  endpoint_url: https://storage.eu-north1.nebius.cloud",
                "  bucket: s3://quickstart-bucket/experiments/",
                "",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeTeardown:
        instances: list["FakeTeardown"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.marked: list[Path | None] = []
            self.teardown_calls = 0
            self.instances.append(self)

        def mark_launched(self, *, config_path=None):
            self.marked.append(config_path)

        def teardown(self):
            self.teardown_calls += 1
            return quickstart.CleanupResult(resources_removed=["cluster"])

    class FakeStorageClient:
        @classmethod
        def from_environment(cls, **kwargs):
            captured["storage_client"] = kwargs
            return cls()

        def download_path(self, uri, local_path):
            captured["download_uri"] = uri
            Path(local_path).write_text(
                json.dumps(
                    {
                        "outer_loop": {
                            "score": 0.5,
                            "decision": "retrain",
                            "trend": [0.5],
                        },
                        "feedback": {"score": 0.5},
                    }
                ),
                encoding="utf-8",
            )

    def fake_render_workflow(yaml_path, **kwargs):
        captured["render"] = kwargs
        return [
            {
                "name": "sim-to-real-pipeline",
                "resources": {"cloud": kwargs["task_cloud"], "accelerators": kwargs["gpu"]},
                "envs": {"NPA_SIM_TO_REAL_RUN_ID": kwargs["run_id"]},
                "run": "true",
            }
        ]

    def fake_run(cmd, **kwargs):
        captured["launch_cmd"] = cmd
        captured["launch_env"] = kwargs["env"]
        yaml_path = Path(cmd[-1])
        captured["submitted_yaml"] = [
            doc for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8")) if doc is not None
        ]
        return subprocess.CompletedProcess(cmd, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(quickstart, "SignalTeardown", FakeTeardown)
    monkeypatch.setattr(quickstart, "install_teardown_signal_handlers", lambda teardown: {})
    monkeypatch.setattr(quickstart, "restore_signal_handlers", lambda handlers: None)
    monkeypatch.setattr(quickstart, "StorageClient", FakeStorageClient)
    monkeypatch.setattr(quickstart, "render_workflow", fake_render_workflow)
    monkeypatch.setattr(quickstart.subprocess, "run", fake_run)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.delenv("NPA_S3_BUCKET", raising=False)

    rc = quickstart.main(
        [
            "--run-id",
            run_id,
            "--credential-path",
            str(credentials),
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
            "--source-ref",
            "quickstart-test-branch",
            "--teardown-poll-interval",
            "0",
            "--output-json",
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    render = captured["render"]
    assert render["bucket"] == "quickstart-bucket"
    assert render["s3_prefix"] == f"experiments/sim-to-real/{run_id}"
    assert render["gpu"] == "H100:1"
    assert render["gpu_failover"] == ""
    assert render["train_steps"] == 20
    assert render["train_step_budget"] == 20
    assert render["max_training_iterations"] == 1
    assert render["eval_episodes"] == 1
    assert render["task_cloud"] == "nebius"
    assert captured["launch_cmd"][:6] == [
        str(sky_bin),
        "launch",
        "--cluster",
        quickstart.run_tag(run_id),
        "--name",
        quickstart.run_tag(run_id),
    ]
    assert "--secret" in captured["launch_cmd"]
    assert "AWS_ACCESS_KEY_ID" in captured["launch_cmd"]
    assert "AWS_SECRET_ACCESS_KEY" in captured["launch_cmd"]
    assert captured["submitted_yaml"][0]["name"] == quickstart.run_tag(run_id)
    assert "workdir" not in captured["submitted_yaml"][0]
    assert captured["submitted_yaml"][0]["envs"]["NPA_SIM_TO_REAL_RUN_ID"] == run_id
    assert captured["submitted_yaml"][0]["envs"]["NPA_SOURCE_REPO"] == quickstart.DEFAULT_SOURCE_REPO
    assert captured["submitted_yaml"][0]["envs"]["NPA_SOURCE_REF"] == "quickstart-test-branch"
    assert FakeTeardown.instances[0].marked == [None]
    assert FakeTeardown.instances[0].teardown_calls == 1
    assert output["metric"]["value"] == 0.5
    assert output["artifacts"]["checkpoint"] == f"s3://quickstart-bucket/experiments/sim-to-real/{run_id}/checkpoints/policy/"
    assert output["teardown"]["cluster_absent"] is True


def test_quickstart_splits_bucket_uri_prefix() -> None:
    quickstart = _load_quickstart_module()

    assert quickstart._split_bucket_and_prefix("s3://bucket/path/to/runs/") == ("bucket", "path/to/runs")
    assert quickstart._split_bucket_and_prefix("bucket/path") == ("bucket", "path")
    assert quickstart._join_s3_prefix("path/to/runs", "sim-to-real/run") == "path/to/runs/sim-to-real/run"


def test_sdk_module_exposes_local_smoke(tmp_path: Path) -> None:
    from npa.sdk.workbench import sim_to_real

    assert sim_to_real.FeedbackType.CRITIQUE.value == "critique"

    dataset = _write_lerobot_fixture(tmp_path / "fixture")
    report = sim_to_real.local_smoke(
        run_id="sdk-s2r",
        output_dir=tmp_path,
        input_data_uri=str(dataset),
        feedback_source="vlm",
        vlm_eval_backend="stub",
        vlm_eval_score=0.7,
    )

    assert report.feedback.score == 0.7
    assert json.loads((tmp_path / "sim-to-real-report.json").read_text())["run_id"] == "sdk-s2r"


def test_feedback_result_dataclass_is_public() -> None:
    result = FeedbackResult(success=False, score=0.2, rationale="Needs another loop.")

    assert result.source == "sim-env"


def test_default_policy_image_uses_byo_policy_container() -> None:
    assert default_policy_image() == "npa-lerobot-policy:0.1.0"
    assert default_policy_image(registry="cr.example").endswith("/npa-lerobot-policy:0.1.0")


def test_lerobot_episode_split_covers_all_real_episode_ids() -> None:
    train, heldout = seeded_episode_split(list(range(10)), train_fraction=0.8, seed=7)

    assert len(train) == 8
    assert len(heldout) == 2
    assert sorted(train + heldout) == list(range(10))


def test_feedback_policy_hook_updates_adapter_checkpoint(tmp_path: Path) -> None:
    feedback = parse_feedback_batch(
        {"feedback": [{"success": True, "score": 0.8, "rationale": "Stable rollout."}]}
    )
    result = run_feedback_training_step(feedback, output_dir=tmp_path)

    assert result.status == "updated"
    assert result.steps == 1
    assert result.weight_after != result.weight_before
    assert Path(result.checkpoint_path).exists()

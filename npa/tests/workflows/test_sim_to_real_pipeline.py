from __future__ import annotations

import importlib.util
import json
import stat
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.workflows.sim_to_real import (
    FeedbackResult,
    SimToRealConfig,
    Tier,
    artifact_uris,
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


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_sim_to_real_pipeline", WRAPPER_PATH)
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
    monkeypatch.setattr(sim_to_real_module, "run_lerobot_eval", fake_eval)
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
        feedback_source="rollout",
        eval_backend="pusht",
    )

    report = run_real_lerobot_loop(config)
    components = {component.name: component for component in report.components}

    assert report.feedback.score == 0.62
    assert report.outer_loop["trend"] == [0.25, 0.62]
    assert report.outer_loop["decision"] == "promote_checkpoint"
    assert components["lerobot_runtime_import"].tier == Tier.WORKS
    assert components["real_lerobot_dataset"].tier == Tier.WORKS
    assert components["lerobot_episode_split"].tier == Tier.WORKS
    assert components["real_training"].tier == Tier.WORKS
    assert components["real_rollout_eval"].tier == Tier.WORKS
    assert components["feedback_training_loop"].tier == Tier.WORKS
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
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["envs"]["S3_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    assert task["envs"]["NEBIUS_S3_ENDPOINT"] == "https://storage.eu-north1.nebius.cloud"
    assert task["envs"]["S3_BUCKET"] == "${S3_BUCKET}"
    assert task["envs"]["NPA_S3_BUCKET"] == "${S3_BUCKET}"
    assert task["envs"]["POLICY_IMAGE"] == "npa-lerobot-policy:0.1.0"
    assert task["envs"]["LEROBOT_DATASET_REPO_ID"] == "lerobot/pusht"
    assert task["envs"]["FEEDBACK_SOURCE"] == "rollout"
    assert task["envs"]["EVAL_BACKEND"] == "pusht"
    assert task["envs"]["TRAIN_STEPS"] == "2000"
    assert task["envs"]["MAX_TRAINING_ITERATIONS"] == "3"
    assert "npa.workflows.sim_to_real real-loop" in task["run"]
    assert "--attempt-s3-roundtrip" in task["run"]


def test_runner_renders_policy_image_and_vlm_eval_settings() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        policy_image="cr.example/npa-lerobot:custom",
        feedback_source="rollout",
        eval_backend="pusht",
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
    assert task_env["FEEDBACK_SOURCE"] == "rollout"
    assert task_env["EVAL_BACKEND"] == "pusht"
    assert "VLM_EVAL_SCORE" not in task_env
    assert "image_id" not in docs[0]["resources"]


def test_runner_renders_ordered_gpu_failover_resources() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        gpu="H100:1,H200:1,A100:1",
    )

    assert docs[0]["resources"]["accelerators"] == ["H100:1", "H200:1", "A100:1"]


def test_runner_can_render_nebius_task_cloud_fallback() -> None:
    wrapper = _load_wrapper_module()
    docs = wrapper.render_workflow(
        YAML_PATH,
        run_id="s2r-render",
        bucket="bucket",
        task_cloud="nebius",
        gpu="H100:1,H200:1,A100:1",
    )

    assert docs[0]["resources"]["cloud"] == "nebius"
    assert docs[0]["resources"]["region"] == "eu-north1"
    assert docs[0]["resources"]["accelerators"] == ["H100:1", "H200:1", "A100:1"]
    assert docs[0]["resources"]["cpus"] == "16+"
    assert docs[0]["resources"]["memory"] == "64+"
    assert "image_id" not in docs[0]["resources"]


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


def test_sdk_module_exposes_local_smoke(tmp_path: Path) -> None:
    from npa.sdk.workbench import sim_to_real

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

    assert result.source == "rollout"


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

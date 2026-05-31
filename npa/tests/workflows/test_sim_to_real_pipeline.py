from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

from npa.workflows.sim_to_real import (
    FeedbackResult,
    SimToRealConfig,
    Tier,
    artifact_uris,
    default_s3_prefix,
    feedback_to_training_signal,
    generate_raw_envs,
    outer_loop_decision,
    parse_feedback_result,
    run_structural_spine,
    seeded_train_heldout_split,
)


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


def test_structural_spine_uses_existing_vlm_eval_stub(tmp_path: Path) -> None:
    config = SimToRealConfig(
        run_id="s2r-test",
        output_dir=tmp_path,
        env_count=10,
        threshold=0.5,
        vlm_eval_backend="stub",
        vlm_eval_model="vlm-eval-stub",
        vlm_eval_score=0.82,
    )

    report = run_structural_spine(config)
    components = {component.name: component for component in report.components}

    assert report.feedback.score == 0.82
    assert components["vlm_feedback"].tier == Tier.PARTIAL
    assert components["genesis_env_split"].tier == Tier.PARTIAL
    assert components["inner_feedback_training_loop"].tier == Tier.SEAM
    assert (tmp_path / "feedback" / "vlm-eval" / "vlm_eval_stub.json").exists()
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

    assert docs[0] == {"name": "sim-to-real-pipeline", "execution": "serial"}
    task = docs[1]
    assert task["name"] == "s2r-controller-spine"
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["envs"]["NEBIUS_S3_ENDPOINT"] == "https://storage.eu-north1.nebius.cloud"
    assert task["envs"]["POLICY_IMAGE"].startswith("cr.eu-north1.nebius.cloud/")
    assert task["envs"]["FEEDBACK_SOURCE"] == "vlm"
    assert "npa.workflows.sim_to_real local-smoke" in task["run"]
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
    task_env = docs[1]["envs"]

    assert task_env["NPA_SIM_TO_REAL_RUN_ID"] == "s2r-render"
    assert task_env["NPA_S3_BUCKET"] == "bucket"
    assert task_env["POLICY_IMAGE"] == "cr.example/npa-lerobot:custom"
    assert task_env["CHECKPOINT_URI"] == "s3://bucket/sim-to-real/s2r-render/checkpoints/policy/"
    assert task_env["VLM_EVAL_BACKEND"] == "stub"
    assert task_env["VLM_EVAL_SCORE"] == "0.9"


def test_sdk_module_exposes_local_smoke(tmp_path: Path) -> None:
    from npa.sdk.workbench import sim_to_real

    report = sim_to_real.local_smoke(
        run_id="sdk-s2r",
        output_dir=tmp_path,
        vlm_eval_backend="stub",
        vlm_eval_score=0.7,
    )

    assert report.feedback.score == 0.7
    assert json.loads((tmp_path / "sim-to-real-report.json").read_text())["run_id"] == "sdk-s2r"


def test_feedback_result_dataclass_is_public() -> None:
    result = FeedbackResult(success=False, score=0.2, rationale="Needs another loop.")

    assert result.source == "vlm"

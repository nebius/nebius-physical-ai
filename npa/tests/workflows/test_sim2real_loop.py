from __future__ import annotations

import json
from pathlib import Path

import yaml

from npa.sdk.workbench import sim2real
from npa.workbench.lerobot.policy_container import parse_vlm_signal_batch, run_vlm_signal_training_step
from npa.workflows.sim2real_loop import (
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
    Sim2RealLoopConfig,
    convert_vlm_eval_to_rl_signal,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_full_loop,
)


ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = ROOT / "npa" / "workflows" / "workbench" / "sim2real" / "runbook.yaml"


def test_vlm_eval_signal_converter_and_trainer_update_close_loop(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="sim2real-unit",
        output_dir=tmp_path,
        threshold=0.75,
        rollout_count=1,
        steps_per_rollout=3,
    )
    rollout = generate_action_rollouts(
        tmp_path / "actions",
        count=1,
        steps_per_rollout=3,
        seed=7,
        quality=0.4,
    )[0]

    evaluation = evaluate_rollout_with_vlm(rollout, output_dir=tmp_path / "vlm_eval", config=config)
    signal = convert_vlm_eval_to_rl_signal(evaluation)
    parsed = parse_vlm_signal_batch(signal)
    update = run_vlm_signal_training_step(parsed, output_dir=tmp_path / "update")
    control = run_vlm_signal_training_step(parsed, output_dir=tmp_path / "control", control=True)

    assert evaluation["schema"] == SCHEMA_VLM_EVAL
    assert signal["schema"] == SCHEMA_RL_SIGNAL
    assert signal["per_step"][0]["target"]["nl_correction"]
    assert update.policy_delta_l2 > control.policy_delta_l2
    assert Path(update.checkpoint_path).exists()


def test_full_loop_writes_stage_artifacts_and_candidate(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="sim2real-full-unit",
        output_dir=tmp_path,
        threshold=0.45,
        inner_iterations=2,
        outer_iterations=1,
        rollout_count=2,
        steps_per_rollout=3,
        heldout_env_count=4,
    )

    report = run_full_loop(config)
    decision = report["outer_loop"]["latest_decision"]
    reward_trend = report["inner_loop"]["reward_trend"]

    assert report["schema"] == "npa.sim2real.e2e_report.v1"
    assert reward_trend[-1] >= reward_trend[0]
    assert decision["decision"] == "promote_checkpoint"
    assert (tmp_path / "vlm_eval" / "train").exists()
    assert (tmp_path / "training_signal" / "train").exists()
    assert (tmp_path / "inner_loop" / "outer-01" / "evidence.json").exists()
    assert json.loads((tmp_path / "eval" / "heldout" / "report.json").read_text())["success_rate"] >= 0.45
    assert (tmp_path / "checkpoints" / "candidate" / "candidate.json").exists()
    assert (tmp_path / "reports" / "sim2real-report.json").exists()


def test_sdk_exposes_sim2real_run(tmp_path: Path) -> None:
    report = sim2real.run(
        run_id="sim2real-sdk-unit",
        output_dir=tmp_path,
        threshold=0.45,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
    )

    assert report["run_id"] == "sim2real-sdk-unit"
    assert "vlm_image" in report["byo_seams"]


def test_raw_runbook_invokes_full_loop_and_exposes_byo_envs() -> None:
    docs = [doc for doc in yaml.safe_load_all(RUNBOOK.read_text(encoding="utf-8")) if doc is not None]

    assert len(docs) == 1
    task = docs[0]
    assert task["name"] == "sim2real-full-loop"
    assert task["envs"]["VLM_IMAGE"] == "${VLM_IMAGE}"
    assert task["envs"]["TRAINER_IMAGE"] == "${TRAINER_IMAGE}"
    assert task["envs"]["EVAL_IMAGE"] == "${EVAL_IMAGE}"
    assert "npa.workflows.sim2real_loop full-loop" in task["run"]
    assert "--byo-signal-converter" in task["run"]

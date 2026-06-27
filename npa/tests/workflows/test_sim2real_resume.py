"""Outer-loop RESUME wiring (engine + runner).

Stage 11B "send back for more RL" only compounds if each outer iteration
CONTINUES the same policy instead of retraining from scratch. These tests pin
the seam that threads the prior iteration's checkpoint URI down to the trainer
command's environment, and the runner state that carries it across iterations.
"""

from __future__ import annotations

from pathlib import Path

import npa.workflows.sim2real.engine as engine
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.runner import Sim2RealWorkflow
from npa.workflows.sim2real.state import WorkflowState


def _update(checkpoint_path: str):
    """A minimal VlmSignalUpdateResult-shaped object the engine consumes."""
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    return VlmSignalUpdateResult.from_dict(
        {
            "schema": "npa.lerobot.vlm_signal_adapter.v1",
            "status": "success",
            "reward_head_before": 0.0,
            "reward_head_after": 0.1,
            "policy_output_before": [0.0],
            "policy_output_after": [0.2],
            "policy_delta_l2": 0.3,
            "checkpoint_path": checkpoint_path,
        }
    )


def test_trainer_command_env_carries_tag_and_resume_uri(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, *, cwd, env, component, timeout_s=7200):
        captured["env"] = dict(env)
        return {"mode": "command", "component": component, "returncode": 0}

    monkeypatch.setattr(engine, "_run_component_command", fake_run)
    monkeypatch.setattr(
        engine, "_read_component_json", lambda *a, **k: _update("s3://ck/new.pt").to_dict()
    )

    config = Sim2RealLoopConfig(run_id="r", byo_trainer_command="python trainer.py")
    result = engine._run_trainer_via_command(
        tmp_path / "signals.json",
        config=config,
        output_dir=tmp_path / "trainer",
        initial_reward_head=0.0,
        initial_action_bias=0.0,
        resume_checkpoint_uri="s3://ck/prior.pt",
        outer_iteration=2,
        iteration=1,
    )
    env = captured["env"]
    assert env["NPA_SIM2REAL_TRAINER_TAG"] == "outer-02-iter-01"
    assert env["NPA_SIM2REAL_RESUME_CHECKPOINT_URI"] == "s3://ck/prior.pt"
    assert result.checkpoint_path == "s3://ck/new.pt"


def test_trainer_command_env_omits_resume_when_none(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        engine,
        "_run_component_command",
        lambda command, *, cwd, env, component, timeout_s=7200: captured.update(env=dict(env))
        or {"returncode": 0},
    )
    monkeypatch.setattr(
        engine, "_read_component_json", lambda *a, **k: _update("s3://ck/new.pt").to_dict()
    )
    config = Sim2RealLoopConfig(run_id="r", byo_trainer_command="python trainer.py")
    engine._run_trainer_via_command(
        tmp_path / "signals.json",
        config=config,
        output_dir=tmp_path / "trainer",
        initial_reward_head=0.0,
        initial_action_bias=0.0,
    )
    # Tag always present (path uniqueness); resume only when an upstream checkpoint exists.
    assert captured["env"]["NPA_SIM2REAL_TRAINER_TAG"] == "outer-01-iter-01"
    assert "NPA_SIM2REAL_RESUME_CHECKPOINT_URI" not in captured["env"]


def test_inner_loop_compounds_checkpoint_across_iterations(tmp_path, monkeypatch):
    """Inner iter 2 resumes from inner iter 1's checkpoint; evidence surfaces the last."""
    seen_resume: list[str] = []

    def fake_trainer(signal_batch_path, *, config, output_dir, initial_reward_head,
                     initial_action_bias, train_envs_dir=None, resume_checkpoint_uri="",
                     outer_iteration=1, iteration=1):
        seen_resume.append(resume_checkpoint_uri)
        return _update(f"s3://ck/outer-{outer_iteration:02d}-iter-{iteration:02d}.pt")

    # Stub the expensive rollout/VLM/signal sub-steps so we exercise only the trainer wiring.
    monkeypatch.setattr(engine, "_run_trainer_via_command", fake_trainer)
    import npa.workflows.sim2real_stages as stages

    monkeypatch.setattr(
        stages, "run_policy_rollouts",
        lambda *a, **k: [{"rollout_id": "rollout-0000", "frames_dir": str(tmp_path)}],
    )
    monkeypatch.setattr(
        engine, "evaluate_rollout_with_vlm",
        lambda rollout, *, output_dir, config: {"rollout_id": rollout["rollout_id"], "score": 0.5},
    )
    monkeypatch.setattr(
        engine, "_convert_eval_to_signal",
        lambda evaluation, *, config, output_dir: {
            "schema": "x", "rollout_id": evaluation["rollout_id"],
            "mean_reward": 0.5, "score": 0.5, "advantages": [0.1],
            "per_step": [{"reward": 0.5}, {"reward": 0.7}],
        },
    )
    monkeypatch.setattr(
        engine, "_signal_training_imports",
        lambda: (lambda batch: batch, lambda *a, **k: _update("")),
    )

    config = Sim2RealLoopConfig(
        run_id="r", output_dir=tmp_path, byo_trainer_command="python trainer.py",
        inner_iterations=2,
    )
    evidence = engine.run_inner_loop(
        config, local_dir=tmp_path, initial_quality=0.4, outer_iteration=3,
        resume_checkpoint_uri="s3://ck/prior-outer.pt",
    )
    # iter1 resumes from the prior outer checkpoint; iter2 from iter1's fresh checkpoint.
    assert seen_resume[0] == "s3://ck/prior-outer.pt"
    assert seen_resume[1] == "s3://ck/outer-03-iter-01.pt"
    # evidence exposes the latest checkpoint for the next outer iteration.
    assert evidence["final_checkpoint_uri"] == "s3://ck/outer-03-iter-02.pt"
    assert evidence["resumed_from_checkpoint_uri"] == "s3://ck/prior-outer.pt"


def test_runner_carries_checkpoint_across_outer_iterations(tmp_path, monkeypatch):
    """state.last_checkpoint_uri feeds outer N+1 with outer N's produced checkpoint."""
    calls: list[tuple[int, str]] = []

    def fake_outer(config, *, local_dir, outer_iteration, initial_quality,
                   resume_checkpoint_uri=""):
        calls.append((outer_iteration, resume_checkpoint_uri))
        produced = f"s3://ck/outer-{outer_iteration:02d}.pt"
        return {
            "outer_iteration": outer_iteration,
            "inner": {"final_checkpoint_uri": produced},
            "heldout_report": {"success_rate": 0.0},
            "decision": {"decision": "loop_back_to_inner_loop"},
            "checkpoint_uri": produced,
            "history_entry": {"outer_iteration": outer_iteration},
            "next_quality": 0.5,
        }

    monkeypatch.setattr(engine, "run_single_outer_iteration", fake_outer)
    monkeypatch.setattr(engine, "sync_workflow_state_to_s3", lambda *a, **k: None)

    config = Sim2RealLoopConfig(run_id="r", output_dir=tmp_path, outer_iterations=3)
    workflow = Sim2RealWorkflow(config)
    # Seed initial persisted state (as run_preamble would).
    WorkflowState(run_id="r", local_artifact_dir=tmp_path, current_quality=0.4).save()

    workflow.run_outer_iteration(outer_iteration=1)
    workflow.run_outer_iteration(outer_iteration=2)
    workflow.run_outer_iteration(outer_iteration=3)

    assert calls[0] == (1, "")  # first iteration starts fresh
    assert calls[1] == (2, "s3://ck/outer-01.pt")  # resumes outer 1's policy
    assert calls[2] == (3, "s3://ck/outer-02.pt")  # resumes outer 2's policy
    # Final state persists the latest checkpoint.
    assert WorkflowState.load(tmp_path).last_checkpoint_uri == "s3://ck/outer-03.pt"

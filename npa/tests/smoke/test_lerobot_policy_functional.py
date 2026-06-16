"""Unit tests for the lerobot-policy golden eval smoke (no GPU)."""

from __future__ import annotations

from npa.workbench.lerobot.policy_container import build_lerobot_eval_command


def test_build_lerobot_eval_command_uses_pretrained_path_for_local_checkpoint() -> None:
    command = build_lerobot_eval_command(
        checkpoint_path="/tmp/checkpoints/last/pretrained_model",
        output_dir="/tmp/eval",
        env_type="pusht",
        episodes=1,
    )
    joined = " ".join(command)
    assert "--policy.type=act" in joined
    assert "--policy.pretrained_path=/tmp/checkpoints/last/pretrained_model" in joined
    assert "--policy.path=" not in joined
    assert "--env.type=pusht" in joined
    assert "--eval.n_episodes=1" in joined

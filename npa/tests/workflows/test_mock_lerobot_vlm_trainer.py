"""Mock customer trainer (seam/contract proof).

Proves an arbitrary customer container that is NOT the Isaac trainer can satisfy
the byo_trainer_command contract and that its weight update genuinely responds to
the VLM signal.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult
from npa.workflows.sim2real import mock_lerobot_vlm_trainer as mock


def _signal(rewards_and_tags, rollout_id="rollout-0000"):
    """Build an npa.sim2real.rl_signal.v1 batch from (reward, advantage, target)."""
    per_step = []
    for i, (reward, advantage, target) in enumerate(rewards_and_tags):
        per_step.append(
            {
                "step": i,
                "reward": reward,
                "advantage": advantage,
                "target": {"action_delta": target},
                "error_tags": ["ok"],
            }
        )
    return {"signals": [{"rollout_id": rollout_id, "per_step": per_step}]}


def _write(tmp_path, payload):
    p = tmp_path / "signals.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_output_satisfies_vlm_signal_update_contract(tmp_path, monkeypatch):
    sig = _write(tmp_path, _signal([(0.8, 0.5, [0.12, 0.0, 0.04]),
                                     (-0.2, -0.5, [-0.1, 0.02, 0.0])]))
    out = tmp_path / "update.json"
    monkeypatch.setenv("NPA_SIM2REAL_SIGNAL_JSON", str(sig))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out))
    monkeypatch.delenv("NPA_SIM2REAL_S3_BUCKET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_BUCKET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_RESUME_CHECKPOINT_URI", raising=False)

    rc = mock.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    parsed = VlmSignalUpdateResult.from_dict(payload)  # must not raise
    assert parsed.backend == mock.BACKEND
    assert parsed.policy_output_after  # non-empty
    assert parsed.checkpoint_path  # a real checkpoint path was written
    # the checkpoint file actually exists on disk
    assert (tmp_path / "mock_policy.npz").exists()


def test_update_responds_to_vlm_signal_strength(tmp_path):
    """A more informative critique (larger advantage spread) -> larger policy delta."""
    informative = mock.summarize_signal_batch(
        _signal([(0.9, 0.9, [0.12, 0.0, 0.04]), (-0.9, -0.9, [0.12, 0.0, 0.04])])
    )
    degenerate = mock.summarize_signal_batch(
        # uniform reward -> advantage ~ 0 -> the VLM is not steering the policy
        _signal([(0.1, 0.0, [0.12, 0.0, 0.04]), (0.1, 0.0, [0.12, 0.0, 0.04])])
    )
    assert informative["signal_strength"] > degenerate["signal_strength"]

    def delta(summary):
        policy = mock.MlpPolicy(summary["action_dim"])
        return mock.update_policy(
            policy, summary["weighted_target"], summary["signal_strength"],
            learning_rate=0.5,
        )["policy_delta_l2"]

    assert delta(informative) > delta(degenerate)
    # degenerate critique barely moves the policy
    assert delta(degenerate) == pytest.approx(0.0, abs=1e-6)


def test_update_moves_policy_toward_corrective_target(tmp_path):
    summary = mock.summarize_signal_batch(
        _signal([(0.9, 0.8, [0.3, -0.2, 0.1]), (0.7, 0.6, [0.3, -0.2, 0.1])])
    )
    policy = mock.MlpPolicy(summary["action_dim"])
    res = mock.update_policy(
        policy, summary["weighted_target"], summary["signal_strength"], learning_rate=0.5
    )
    # a real optimization step occurred: loss toward the VLM target decreased and
    # the policy output actually changed.
    assert res["loss_after"] < res["loss_before"]
    assert res["policy_output_after"] != res["policy_output_before"]
    assert res["policy_delta_l2"] > 0.0


def test_deterministic_same_input_same_output(tmp_path, monkeypatch):
    sig = _write(tmp_path, _signal([(0.8, 0.5, [0.12, 0.0, 0.04])]))
    monkeypatch.delenv("NPA_SIM2REAL_S3_BUCKET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_BUCKET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_RESUME_CHECKPOINT_URI", raising=False)
    r1 = mock.run_training(str(sig), run_id="r")
    r2 = mock.run_training(str(sig), run_id="r")
    assert r1["policy_delta_l2"] == r2["policy_delta_l2"]
    assert r1["policy_output_after"] == r2["policy_output_after"]


def test_resume_continues_from_saved_checkpoint(tmp_path):
    policy = mock.MlpPolicy(3, seed=1)
    # perturb so it differs from a fresh seed
    policy.W2 += 0.5
    ckpt = mock.save_checkpoint(policy, tmp_path / "p.npz")
    loaded = mock.load_checkpoint(ckpt)
    assert np.allclose(loaded.W2, policy.W2)
    assert loaded.action_dim == 3


def test_empty_signal_is_safe(tmp_path, monkeypatch):
    sig = _write(tmp_path, {"signals": []})
    out = tmp_path / "update.json"
    monkeypatch.setenv("NPA_SIM2REAL_SIGNAL_JSON", str(sig))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out))
    monkeypatch.delenv("NPA_SIM2REAL_S3_BUCKET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_BUCKET", raising=False)
    rc = mock.main()
    assert rc == 0
    parsed = VlmSignalUpdateResult.from_dict(json.loads(out.read_text()))
    assert parsed.policy_delta_l2 == pytest.approx(0.0, abs=1e-6)  # nothing to learn

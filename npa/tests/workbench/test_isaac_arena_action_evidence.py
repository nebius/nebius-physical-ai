"""Executed-action evidence is measured, sanitized, and hash-bound."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.isaac_arena.action_evidence import (
    ACTION_EVIDENCE_FILENAME,
    action_sequence_evidence,
    configure_action_evidence,
    finalize_action_evidence,
    read_action_evidence,
    record_executed_action,
    validate_action_evidence,
    validate_prepared_action_sequence,
)
from npa.workbench.isaac_arena.errors import IsaacArenaError


def _actions(steps: int = 4) -> np.ndarray:
    return np.arange(steps * 3, dtype=np.float32).reshape(steps, 3) / 10


def test_policy_boundary_journal_matches_prepared_replay_commitment(tmp_path) -> None:
    actions = _actions()
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "replay")
    for index, action in enumerate(actions, 1):
        record_executed_action(env, action[None, :], index)
    finalize_action_evidence(env)

    observed = read_action_evidence(tmp_path, policy_type="replay", expected_steps=4)
    prepared = action_sequence_evidence(actions)
    assert observed["sequence_sha256"] == prepared["sequence_sha256"]
    assert observed["action_step_sha256"] == prepared["action_step_sha256"]
    assert observed["nonzero_actions"] is True
    assert observed["varied_actions"] is True
    assert observed["trailing_held_action_steps"] == 1
    assert observed["trailing_held_action_fraction"] == 0.25
    assert observed["held_action_delta_abs_max_tolerance"] == 1e-6
    assert observed["interstep_delta_abs_max"] == pytest.approx([0.3, 0.3, 0.3])
    assert observed["raw_actions_retained"] is False
    assert len(observed["file"]["sha256"]) == 64
    content = (tmp_path / ACTION_EVIDENCE_FILENAME).read_text()
    assert "0.100000" not in content


def test_failed_env_step_cannot_be_recorded_as_executed(tmp_path) -> None:
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "rsl_rl")
    finalize_action_evidence(env)
    record = json.loads((tmp_path / ACTION_EVIDENCE_FILENAME).read_text())
    assert record["executed_steps"] == 0
    with pytest.raises(IsaacArenaError, match="nonzero varied actions"):
        validate_action_evidence(
            record,
            policy_type="rsl_rl",
            expected_steps=0,
            require_nonzero_varied=True,
        )


@pytest.mark.parametrize("step", [0, 2, True])
def test_policy_boundary_requires_exact_contiguous_steps(tmp_path, step) -> None:
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "replay")
    with pytest.raises(IsaacArenaError, match="not contiguous"):
        record_executed_action(env, np.ones((1, 3)), step)


@pytest.mark.parametrize(
    "action",
    [
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray(["private"]),
        np.asarray([True]),
    ],
)
def test_nonfinite_or_nonnumeric_policy_actions_are_rejected(tmp_path, action) -> None:
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "replay")
    with pytest.raises(IsaacArenaError, match="finite numeric"):
        record_executed_action(env, action, 1)


def test_tampered_sequence_chain_is_rejected(tmp_path) -> None:
    actions = _actions()
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "replay")
    for index, action in enumerate(actions, 1):
        record_executed_action(env, action, index)
    finalize_action_evidence(env)
    path = tmp_path / ACTION_EVIDENCE_FILENAME
    payload = json.loads(path.read_text())
    payload["action_step_sha256"][1] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(IsaacArenaError, match="nonzero varied actions"):
        read_action_evidence(tmp_path, policy_type="replay", expected_steps=4)


def test_tampered_trailing_action_summary_is_rejected(tmp_path) -> None:
    actions = _actions()
    env = SimpleNamespace()
    configure_action_evidence(env, tmp_path, "replay")
    for index, action in enumerate(actions, 1):
        record_executed_action(env, action, index)
    finalize_action_evidence(env)
    path = tmp_path / ACTION_EVIDENCE_FILENAME
    payload = json.loads(path.read_text())
    payload["trailing_held_action_steps"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(IsaacArenaError, match="nonzero varied actions"):
        read_action_evidence(tmp_path, policy_type="replay", expected_steps=4)


def test_near_identical_jitter_is_measured_as_a_held_tail() -> None:
    actions = np.arange(30, dtype=np.float64).reshape(10, 3)
    actions[4:] = actions[3] + (np.arange(6, dtype=np.float64)[:, None] + 1) * 1e-9
    record = action_sequence_evidence(actions)
    assert record["distinct_action_steps"] == 10
    assert record["trailing_held_action_steps"] == 7
    assert record["trailing_held_action_fraction"] == 0.7
    assert max(record["interstep_delta_abs_max"][3:]) < 1e-6


def test_repeated_action_hashes_cannot_claim_nonzero_deltas() -> None:
    actions = _actions(8)
    actions[4:] = actions[3]
    record = action_sequence_evidence(actions)
    record["interstep_delta_abs_max"] = [3.0] * 7
    record["action_step_delta_abs_max"] = 3.0
    record["trailing_held_action_steps"] = 1
    record["trailing_held_action_fraction"] = 0.125
    with pytest.raises(IsaacArenaError, match="replay prepared actions"):
        validate_prepared_action_sequence(
            record,
            expected_steps=8,
            maximum_trailing_held_fraction=0.25,
        )

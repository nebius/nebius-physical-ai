"""Verify the selected-RLC transition-refresh execution boundaries."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import signal
import sys
import types
from unittest import mock

import numpy as np

from npa.workflows.behavior_challenge import rlc_transition


@dataclasses.dataclass(frozen=True)
class NativeConfig:
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2


class NativePolicy:
    def __init__(self) -> None:
        self.config = NativeConfig()
        self.last_actions = None
        self.action_index = 0
        self.next_initial_actions = None
        self.prediction_history = []
        self.current_stage = 0
        self.calls = 0
        self.queue_empty_at_act = []
        self.advance_on_call: int | None = None

    def reset(self) -> None:
        self.last_actions = None
        self.action_index = 0
        self.next_initial_actions = None
        self.prediction_history.clear()
        self.current_stage = 0

    def act(self, observation):
        del observation
        self.calls += 1
        self.queue_empty_at_act.append(self.last_actions is None)
        action = np.full(23, self.calls, dtype=np.float32)
        self.last_actions = np.ones((20, 23), dtype=np.float32)
        self.action_index = 1
        self.next_initial_actions = np.ones((4, 23), dtype=np.float32)
        if self.advance_on_call == self.calls:
            self.current_stage += 1
        return action


def _observation(width: float) -> dict[str, np.ndarray]:
    state = np.zeros(61, dtype=np.float32)
    state[rlc_transition.LEFT_GRIPPER_SLICE] = width / 2
    state[rlc_transition.RIGHT_GRIPPER_SLICE] = width / 2
    return {"robot_r1::proprio": state}


def test_native_default_returns_exact_policy_without_mutation():
    policy = NativePolicy()
    before = dataclasses.asdict(policy.config)

    configured = rlc_transition.configure_selected_execution(policy, "native")

    assert configured is policy
    assert dataclasses.asdict(policy.config) == before
    assert configured.act(_observation(0.05)).shape == (23,)


def test_gripper_change_clears_queue_before_current_action():
    native = NativePolicy()
    policy = rlc_transition.configure_selected_execution(native, "transition-refresh")

    for width in (0.1, 0.1, 0.0, 0.0):
        policy.act(_observation(width))

    assert native.queue_empty_at_act == [True, False, False, True]
    assert policy.telemetry()["gripper_transitions"] == 1
    assert policy.telemetry()["replans_requested"] == 1


def test_gripper_stability_boundaries_do_not_cross_neutral_or_reset():
    native = NativePolicy()
    policy = rlc_transition.configure_selected_execution(native, "transition-refresh")

    for width in (0.1, 0.05, 0.1, 0.1):
        policy.act(_observation(width))

    assert policy.telemetry()["gripper_transitions"] == 0
    assert policy.telemetry()["replans_requested"] == 0
    policy.reset()
    for width in (0.0, 0.0):
        policy.act(_observation(width))

    assert policy.telemetry()["episode_ordinal"] == 1
    assert policy.telemetry()["gripper_transitions"] == 0
    assert policy.telemetry()["replans_requested"] == 0


def test_invalid_proprioception_is_rejected_before_upstream_action():
    native = NativePolicy()
    policy = rlc_transition.configure_selected_execution(native, "transition-refresh")
    invalid = (
        np.zeros(60, dtype=np.float32),
        np.full(61, np.nan, dtype=np.float32),
    )

    for state in invalid:
        with np.testing.assert_raises_regex(ValueError, "finite 61-element"):
            policy.act({"robot_r1::proprio": state})

    assert native.calls == 0


def test_accepted_stage_change_returns_current_action_then_clears_next_queue():
    native = NativePolicy()
    native.advance_on_call = 1
    policy = rlc_transition.configure_selected_execution(native, "transition-refresh")

    action = policy.act(_observation(0.05))

    assert action.tolist() == [1.0] * 23
    assert dataclasses.astuple(native.config) == (26, 4, 20, 3, 2)
    assert native.prediction_history.maxlen == 3
    assert native.current_stage == 1
    assert native.last_actions is None
    assert native.next_initial_actions is None
    assert native.action_index == 0
    assert policy.telemetry()["stage_transitions"] == 1


def test_reset_and_final_flush_emit_each_nonempty_episode_once(caplog):
    policy = rlc_transition.configure_selected_execution(
        NativePolicy(), "transition-refresh"
    )
    with caplog.at_level("INFO", logger=rlc_transition.__name__):
        policy.act(_observation(0.05))
        policy.reset()
        policy.reset()
        policy.act(_observation(0.05))
        policy.finalize_telemetry()
        policy.finalize_telemetry()

    summaries = [
        json.loads(record.message.split("RLC_VARIANT_SUMMARY ", 1)[1])
        for record in caplog.records
        if "RLC_VARIANT_SUMMARY " in record.message
    ]
    assert [row["episode_ordinal"] for row in summaries] == [0, 1]
    assert all(row["observations"] == 1 for row in summaries)


def test_graceful_server_termination_flushes_final_episode_once(monkeypatch):
    path = Path(rlc_transition.__file__).with_name("rlc_selected_server.py")
    selected = types.ModuleType("rlc_selected")
    selected.load_selected_policy = mock.Mock()
    selected.load_validated_correlation = mock.Mock()
    monkeypatch.setitem(sys.modules, "rlc_selected", selected)
    spec = importlib.util.spec_from_file_location("transition_server_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    policy = mock.Mock()
    handlers = {}
    monkeypatch.setattr(
        server.signal,
        "signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )

    server._install_termination_handlers(policy)
    with np.testing.assert_raises(SystemExit):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    policy.finalize_telemetry.assert_called_once_with()


def test_selected_server_accepts_final_backtrack_and_defaults_native(monkeypatch):
    path = Path(rlc_transition.__file__).with_name("rlc_selected_server.py")
    selected = types.ModuleType("rlc_selected")
    selected.load_selected_policy = mock.Mock()
    selected.load_validated_correlation = mock.Mock()
    monkeypatch.setitem(sys.modules, "rlc_selected", selected)
    spec = importlib.util.spec_from_file_location("selected_server_parser_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    common = [
        "--source-root",
        "/source",
        "--adapter-root",
        "/adapter",
        "--checkpoint",
        "/checkpoint",
        "--selected-export-receipt",
        "/export.json",
        "--correlation-manifest",
        "/manifest.json",
        "--validation-receipt",
        "/validation.json",
        "--task-id",
        "1",
        "--port",
        "8000",
    ]

    assert server.parser().parse_args(common).execution_variant == "native"
    parsed = server.parser().parse_args(
        [*common, "--execution-variant", "final-stage-backtrack"]
    )
    assert parsed.execution_variant == "final-stage-backtrack"


def test_transition_provenance_binds_frozen_private_source_and_config():
    assert dict(rlc_transition.TRANSITION_REFRESH_PROVENANCE) == {
        "experiment_source_sha256": (
            "7efdb10f245a380addc84b9d58a922cdaa0572a76bab01980d4b77fb5dab6c32"
        ),
        "experiment_config_sha256": (
            "a4b5ce6278f57a1aaad04549b0cceabf6fc274a5b8fdad53a13d688bb45c05dd"
        ),
        "config_schema": "npa.behavior.rlc-test-time-variant.v1",
        "config_name": "transition_refresh",
    }


def test_invalid_variant_and_nonpristine_policy_fail_closed():
    with np.testing.assert_raises_regex(ValueError, "Unsupported"):
        rlc_transition.configure_selected_execution(NativePolicy(), "other")
    policy = NativePolicy()
    policy.last_actions = np.zeros((1, 23), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "pristine"):
        rlc_transition.configure_selected_execution(policy, "transition-refresh")

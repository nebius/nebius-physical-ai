"""Prove phase diagnostics expose blocked boundaries without leaking or changing work."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from npa.workbench.isaac_arena import simulator_phases as phases


def _rows(directory: Path, rank: int = 0) -> list[dict]:
    path = directory / f"simulator-phases-rank{rank}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_begin_is_readable_while_native_operation_has_not_returned(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def operation():
        with phases.phase_scope(env, "env_step", 5):
            entered.set()
            release.wait()

    thread = threading.Thread(target=operation)
    thread.start()
    entered.wait()
    try:
        assert [(row["phase"], row["event"]) for row in _rows(tmp_path)] == [
            ("env_step", "begin")
        ]
    finally:
        release.set()
        thread.join()
    assert [row["event"] for row in _rows(tmp_path)] == ["begin", "end"]


def test_native_exception_is_identical_and_its_contents_never_enter_journal(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path, rank=2)
    error = RuntimeError("private replay payload and private input location")
    with pytest.raises(RuntimeError) as caught:
        with phases.phase_scope(env, "policy_action", 5):
            raise error
    assert caught.value is error
    rows = _rows(tmp_path, 2)
    assert [row["event"] for row in rows] == ["begin", "failed"]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert all(row["rank"] == 2 and row["action_step"] == 5 for row in rows)
    assert rows[1]["monotonic_ns"] >= rows[0]["monotonic_ns"]
    assert "private" not in json.dumps(rows)
    assert (tmp_path / "simulator-phases-rank2.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fail_operation", [False, True])
def test_unwritable_journal_never_changes_result_or_masks_exception(tmp_path, fail_operation):
    target = tmp_path / "unrelated.txt"
    target.write_text("preserve")
    (tmp_path / "simulator-phases-rank0.jsonl").symlink_to(target)
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    error = RuntimeError("native error")
    result = object()

    def operation():
        with phases.phase_scope(env, "env_step", 1):
            if fail_operation:
                raise error
            return result

    if fail_operation:
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is error
    else:
        assert operation() is result
    assert env._npa_phase_journal.write_failed
    assert target.read_text() == "preserve"


def _pink_environment(controller):
    term_type = type("PinkIKAction", (), {})
    term_type.__module__ = "isaaclab.envs.mdp.actions.pink_task_space_actions"
    term = term_type()
    term._ik_controllers = [controller]
    return SimpleNamespace(action_manager=SimpleNamespace(
        active_terms=["arms"], get_term=lambda name: term,
    ))


def test_pink_wrapper_preserves_each_call_arguments_result_and_exception(tmp_path):
    calls = []
    result, joint_state = object(), object()
    error = RuntimeError("private solver details")

    class Controller:
        def compute(self, positions, dt, *, fail=False):
            calls.append((positions, dt, fail))
            if fail:
                raise error
            return result

    Controller.__module__ = "isaaclab.controllers.pink_ik.pink_ik"
    controller = Controller()
    other = Controller()
    env = _pink_environment(controller)
    phases.configure_phase_journal(env, tmp_path)
    phases.configure_phase_journal(env, tmp_path)
    with phases.phase_scope(env, "env_step", 5):
        assert controller.compute(joint_state, 0.005) is result
        with pytest.raises(RuntimeError) as caught:
            controller.compute(joint_state, 0.005, fail=True)
    assert caught.value is error
    assert calls == [(joint_state, 0.005, False), (joint_state, 0.005, True)]
    assert "compute" not in other.__dict__
    rows = _rows(tmp_path)
    assert [row["event"] for row in rows if row["phase"] == "pink_ik"] == [
        "begin", "end", "begin", "failed"
    ]
    assert all(row["action_step"] == 5 for row in rows)
    assert "private" not in json.dumps(rows)


def test_other_controller_contracts_are_not_wrapped(tmp_path):
    controller = SimpleNamespace(compute=lambda *args: None)
    original = controller.compute
    env = _pink_environment(controller)
    phases.configure_phase_journal(env, tmp_path)
    assert controller.compute is original


@pytest.mark.parametrize("field,value", [
    ("phase", "private input location"), ("action_step", True), ("render_call", -1),
    ("render_call", 0),
])
def test_phase_rejects_arbitrary_text_and_invalid_counters(tmp_path, field, value):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    arguments = {"phase": "capture", "action_step": 0, "render_call": 1}
    arguments[field] = value
    with pytest.raises(ValueError, match="invalid simulator"):
        with phases.phase_scope(env, **arguments):
            pytest.fail("invalid diagnostic arguments reached operation")
    assert not list(tmp_path.iterdir())


def test_readiness_records_only_observed_boolean_or_null_flags(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    phases.record_readiness(
        env, 1, stage_ready=False, annotator_ready=None, nonblack_rgb=None,
    )
    row = _rows(tmp_path)[0]
    assert row["stage_ready"] is False and row["annotator_ready"] is None
    with pytest.raises(ValueError, match="invalid simulator readiness"):
        phases.record_readiness(env, 2, private_payload="never publish")
    assert len(_rows(tmp_path)) == 1


@pytest.mark.parametrize("phase,event,fields", [
    ("private phase", "begin", {}),
    ("env_step", "private error", {}),
    ("env_step", "failed", {"exception": "private exception content"}),
    ("env_step", "begin", {"render_call": "private value"}),
    ("capture_readiness", "observed", {
        "render_call": 1, "stage_ready": True, "annotator_ready": True,
        "nonblack_rgb": "private image contents",
    }),
    ("capture_readiness", "observed", {
        "render_call": 1, "stage_ready": True, "annotator_ready": True,
        "nonblack_rgb": True, "asset_path": "private location",
    }),
])
def test_write_sink_rejects_unapproved_fields_even_from_internal_callers(
    tmp_path, phase, event, fields
):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    with pytest.raises(ValueError) as caught:
        env._npa_phase_journal.emit(phase, event, 5, **fields)
    assert "private" not in str(caught.value)
    assert not list(tmp_path.iterdir())

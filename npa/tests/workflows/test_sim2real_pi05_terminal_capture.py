"""Require new terminal observations and real controls for complete target windows."""

from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import pi05_isaac
from npa.workflows.sim2real.pi05_contract import ACTION_HORIZON


def _install_terminal_simulator(monkeypatch, context, *, record=True):
    clock = SimpleNamespace(step=0, captured=None)

    def command(_context, _goal, intent):
        assert intent == 0.0
        return clock.step, 0.0, 0.0

    def observe(_native, _joints, _buffers):
        clock.captured = clock.step
        return {"wrist_frame": clock.step, "exterior_frame": clock.step}

    def control(_context, target, phase):
        assert phase == "retreat" and clock.captured == target == clock.step
        before = clock.step
        clock.step += 1
        return before, clock.step

    def store(_context, observation, intent, phase, before, after):
        assert observation["wrist_frame"] == observation["exterior_frame"] == before
        assert after == before + 1 and intent == 0.0 and phase == "retreat"
        if record:
            context["buffers"]["actions"].append(before)
            context["buffers"]["observations"].append(observation)

    monkeypatch.setattr(pi05_isaac, "_expert_command", command)
    monkeypatch.setattr(pi05_isaac, "_causal_observation", observe)
    monkeypatch.setattr(pi05_isaac, "_step_command", control)
    monkeypatch.setattr(pi05_isaac, "_record_step", store)
    return clock


def _context():
    return {
        "native": object(),
        "joint_indices": list(range(7)),
        "buffers": {"actions": ["original"] * 30, "observations": []},
    }


def test_terminal_capture_uses_new_causal_images_and_commands(monkeypatch):
    context = _context()
    clock = _install_terminal_simulator(monkeypatch, context)
    pi05_isaac._capture_terminal_retreat(context, object())
    assert clock.step == ACTION_HORIZON
    observations = context["buffers"]["observations"]
    assert [row["wrist_frame"] for row in observations] == list(range(ACTION_HORIZON))
    receipt = context["terminal_retreat_capture"]
    assert receipt["synthetic_target_padding"] is False
    assert (
        receipt["last_complete_target_input_step"]
        > receipt["last_original_waypoint_step"]
    )


def test_terminal_capture_rejects_missing_camera_control_rows(monkeypatch):
    context = _context()
    _install_terminal_simulator(monkeypatch, context, record=False)
    with pytest.raises(RuntimeError, match="do not match actual commands"):
        pi05_isaac._capture_terminal_retreat(context, object())
    assert "terminal_retreat_capture" not in context

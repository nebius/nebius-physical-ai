"""Reject misleading warehouse footage when physical state or render clocks fail."""

from fractions import Fraction
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _module(name):
    path = (
        Path(__file__).parents[2] / "examples/antioch-warehouse-spot/src" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"warehouse_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _states():
    return [
        {
            "position": [0.0, 0.0, 0.6],
            "orientation": [1, 0, 0, 0],
            "joints": [0.0, 0.0],
        },
        {
            "position": [1.5, 0.0, 0.6],
            "orientation": [1, 0, 0, 0],
            "joints": [0.4, -0.3],
        },
    ]


def test_capture_rejects_provider_override_to_root(monkeypatch):
    evidence = _module("evidence")
    monkeypatch.setattr(evidence.os, "geteuid", lambda: 0)
    monkeypatch.setattr(evidence.os, "getegid", lambda: 0)
    with pytest.raises(RuntimeError, match="non-root runtime user"):
        evidence._runtime_identity()


def test_capture_reports_effective_runtime_identity(monkeypatch):
    evidence = _module("evidence")
    monkeypatch.setattr(evidence.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(evidence.os, "getegid", lambda: 1000)
    assert evidence._runtime_identity() == {"runtime_uid": 1000, "runtime_gid": 1000}


def test_motion_summary_uses_observed_displacement_and_articulation():
    result = _module("evidence")._summarize(_states(), 50, 50)
    assert result["distance_m"] == pytest.approx(1.5)
    assert result["displacement_m"] == pytest.approx(1.5)
    assert result["maximum_joint_range_rad"] == pytest.approx(0.4)
    assert result["minimum_body_up_z"] == pytest.approx(1.0)
    assert result["finite_state"] is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_orientation_cannot_pass_physical_evidence(bad):
    samples = _states()
    samples[1]["orientation"][0] = bad
    assert _module("evidence")._summarize(samples, 50, 50)["finite_state"] is False


def test_overturned_robot_is_not_classified_as_upright_from_height():
    samples = _states()
    samples[1]["orientation"] = [0, 1, 0, 0]
    result = _module("evidence")._summarize(samples, 50, 50)
    assert result["minimum_base_height_m"] > 0.35
    assert result["minimum_body_up_z"] < 0.7


def test_repeated_or_regressing_camera_clock_does_not_create_frames():
    camera = _module("camera")
    recording = camera._Recording.__new__(camera._Recording)
    accepted = []
    recording.writer = SimpleNamespace(send=accepted.append)
    recording.frames = 0
    recording.prior_clock = None
    recording.timestamps = []
    recording.pixel_variances = []
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    assert recording.append((rgb, Fraction(1, 25)), 0.04)
    assert not recording.append((rgb, Fraction(1, 25)), 0.08)
    assert not recording.append((rgb, Fraction(0, 25)), 0.12)
    assert not recording.append(None, 0.16)
    assert recording.append((rgb, Fraction(2, 25)), 0.20)
    assert recording.frames == 2
    assert len(accepted) == 2


@pytest.mark.parametrize("elapsed", [0.032, 0.0, 0.08])
def test_native_step_rejects_clamped_frozen_or_skipped_camera_intervals(elapsed):
    scene = _module("scene")

    class Simulation(scene._Simulation):
        @property
        def current_time(self):
            return self.clock

    simulation = Simulation.__new__(Simulation)
    simulation.clock = 0.0
    simulation.app = SimpleNamespace(
        update=lambda: setattr(simulation, "clock", elapsed)
    )
    with pytest.raises(RuntimeError, match="one 25 fps camera frame"):
        simulation.step()
    simulation.clock = 0.0
    simulation.app = SimpleNamespace(update=lambda: setattr(simulation, "clock", 0.04))
    simulation.step()

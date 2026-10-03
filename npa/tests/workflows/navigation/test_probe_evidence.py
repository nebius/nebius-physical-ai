"""Preserve exact repeatability evidence before rejecting a native isolation probe."""

import json

import numpy as np
import pytest

from npa.workflows.navigation import measure
from npa.workflows.navigation.probe_evidence import save_trace, trace_difference


def _trace(delta=0.0, contact=0.0):
    rows = []
    for step in range(2):
        state = {
            "position_m": np.zeros((2, 3)),
            "peer_contact": np.zeros(2),
            "obstacle_contact": np.array([contact * step, 0.0]),
            "physical_failure": np.zeros(2),
        }
        state["position_m"][0, 1] = delta * step
        rows.append({"state": state, "observations": {"policy": np.zeros((2, 4))}})
    return rows


def test_probe_trace_keeps_all_states_and_only_focal_observations(tmp_path):
    trace = _trace()
    save_trace(tmp_path, "solo", trace)
    with np.load(tmp_path / "probe-solo.npz", allow_pickle=False) as values:
        assert values["0/state/position_m"].shape == (2, 3)
        assert values["0/state/peer_contact"].shape == (2,)
        assert values["0/observations/policy"].shape == (1, 4)
        assert np.array_equal(
            values["1/state/position_m"], trace[1]["state"]["position_m"]
        )


def test_probe_difference_is_exact_and_json_serializable():
    report = trace_difference(_trace(), _trace(0.02))
    maximum = json.loads(json.dumps(report))["maximum"]
    assert maximum == {
        "step": 1,
        "group": "state",
        "stream": "position_m",
        "component": [1],
        "absolute_delta": 0.02,
        "baseline": 0.0,
        "changed": 0.02,
    }


def test_repeated_reset_failure_retains_both_comparisons(recipe, tmp_path, monkeypatch):
    traces = iter((_trace(0.02), _trace(0.1), _trace(contact=2.0)))
    monkeypatch.setattr(measure, "_probe_trace", lambda *args: next(traces))
    with pytest.raises(ValueError, match="repeated reset.*position_m step=1"):
        measure._probe_controls(
            None, None, None, recipe, [recipe.probe.free] * 2, _trace(), tmp_path
        )
    report = json.loads((tmp_path / "isolation-comparisons.json").read_text())
    assert report["repeatability"]["maximum"]["absolute_delta"] == 0.02
    assert report["peer_isolation"]["maximum"]["absolute_delta"] == 0.1
    assert {path.name for path in tmp_path.glob("*.npz")} == {
        "probe-repeat.npz",
        "probe-overlap.npz",
        "probe-obstacle.npz",
    }

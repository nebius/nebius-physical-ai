"""Unit tests for the pure helpers of the generated-physics task injector.

``register`` itself imports Isaac-Lab (GPU-only) and is verified by an
on-cluster probe, not here.
"""

from __future__ import annotations

from npa.workflows.sim2real import isaac_physics_task as pt


def test_clamp_parses_and_bounds():
    assert pt.clamp("0.7", 0.1, 2.0, 1.0) == 0.7
    assert pt.clamp("5.0", 0.1, 2.0, 1.0) == 2.0   # above max -> max
    assert pt.clamp("0.0", 0.1, 2.0, 1.0) == 0.1   # below min -> min
    assert pt.clamp("garbage", 0.1, 2.0, 1.0) == 1.0  # unparseable -> default
    assert pt.clamp(None, 0.1, 2.0, 1.0) == 1.0
    assert pt.clamp(float("nan"), 0.1, 2.0, 1.0) == 1.0


def test_physics_params_none_when_unset():
    assert pt.physics_params_from_env({}) is None
    assert pt.physics_params_from_env({"NPA_GEN_FRICTION": "", "NPA_GEN_MASS_SCALE": ""}) is None


def test_physics_params_clamped_from_env():
    p = pt.physics_params_from_env({"NPA_GEN_FRICTION": "0.717", "NPA_GEN_MASS_SCALE": "0.969"})
    assert p == {"friction": 0.717, "mass_scale": 0.969}


def test_physics_params_one_field_present_uses_default_for_other():
    p = pt.physics_params_from_env({"NPA_GEN_FRICTION": "1.4"})
    assert p["friction"] == 1.4
    assert p["mass_scale"] == 1.0  # missing -> default, still active


def test_physics_params_garbage_is_bounded_not_crash():
    p = pt.physics_params_from_env({"NPA_GEN_FRICTION": "-9", "NPA_GEN_MASS_SCALE": "999"})
    assert p["friction"] == pt.FRICTION_MIN
    assert p["mass_scale"] == pt.MASS_SCALE_MAX

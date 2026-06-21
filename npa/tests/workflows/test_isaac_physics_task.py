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


def test_module_source_is_self_contained():
    src = pt.module_source()
    # Shipped into the Isaac container, so it must carry the helpers + register.
    assert "def physics_params_from_env" in src
    assert "def register(" in src
    assert "randomize_rigid_body_material" in src


def test_train_wrapper_enforces_boot_before_isaac_imports():
    s = pt.TRAIN_WRAPPER_SCRIPT
    # AppLauncher boot MUST precede any isaaclab/isaac_physics_task import — the
    # whole point of the wrapper (pre-boot isaaclab import pulls pxr and dies).
    boot = s.index("AppLauncher(headless=True).app")
    assert boot < s.index("import isaaclab_tasks")
    assert boot < s.index("import isaac_physics_task")
    assert s.index("import isaaclab_tasks") < s.index("physmod.register")
    # trains via the rsl_rl runner and emits the done/ckpt markers
    assert "OnPolicyRunner" in s and "runner.learn" in s
    assert "PHYS_TRAIN_DONE" in s

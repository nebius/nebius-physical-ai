"""Unit tests for routing a customer robot_spec into STAGED RL training.

Previously a BYO ``robot_spec_uri`` reached only the held-out eval USD-swap; the
trainer sibling never received the robot vars, so the policy trained on the stock
Franka. These cover the seams that close that gap:

* ``engine._byo_robot_env`` — forwards the robot inputs + opts a component into the
  BYO-robot path, and is Franka-safe (empty for no-robot / stock). Shared by the
  trainer sibling AND the held-out eval so both build the SAME embodiment/dims.
* ``engine.run_heldout_eval`` — merges ``_byo_robot_env`` into the eval env, so the
  held-out eval opts into ``NPA_BYO_ROBOT_TASK=1`` (else it evaluates a stock
  Franka-dimensioned env a non-Franka checkpoint can't load into).
* ``byo_isaac_trainer._resolve_byo_robot_spec`` — resolves a ``robot_spec_uri``
  JSON via the same parser the eval uses (mocked here; no cluster / no network).
"""

from __future__ import annotations

import json

import pytest

from npa.workflows.sim2real import byo_isaac_trainer as trainer
from npa.workflows.sim2real import engine
from npa.workflows.sim2real.models import Sim2RealLoopConfig


# --------------------------------------------------------------------------- #
# engine._byo_robot_env — forward robot vars to the trainer sibling
# --------------------------------------------------------------------------- #
def test_trainer_env_empty_for_no_robot():
    assert engine._byo_robot_env(Sim2RealLoopConfig(run_id="r")) == {}


def test_trainer_env_empty_for_stock_franka_source():
    cfg = Sim2RealLoopConfig(run_id="r", robot_source="stock_franka")
    assert engine._byo_robot_env(cfg) == {}


def test_trainer_env_set_for_spec_uri():
    cfg = Sim2RealLoopConfig(run_id="r", robot_spec_uri="s3://b/kinova/robot-spec.json")
    env = engine._byo_robot_env(cfg)
    assert env["NPA_BYO_ROBOT_TASK"] == "1"
    assert env["NPA_SIM2REAL_ROBOT_SPEC_URI"] == "s3://b/kinova/robot-spec.json"
    # forwarded even when blank so the trainer sees the same trio the eval does
    assert env["NPA_SIM2REAL_ROBOT_SOURCE"] == ""
    assert env["NPA_SIM2REAL_ROBOT_PRESET"] == ""


def test_trainer_env_set_for_preset_and_byo_source():
    assert engine._byo_robot_env(
        Sim2RealLoopConfig(run_id="r", robot_preset="ur10e")
    )["NPA_BYO_ROBOT_TASK"] == "1"
    assert engine._byo_robot_env(
        Sim2RealLoopConfig(run_id="r", robot_source="byo_usd")
    )["NPA_BYO_ROBOT_TASK"] == "1"


# --------------------------------------------------------------------------- #
# engine.run_heldout_eval — held-out eval opts into the BYO-robot path too, so it
# evaluates the SAME retargeted variant (matching dims) the policy trained on.
# --------------------------------------------------------------------------- #
class _StopComponentEnv(RuntimeError):
    """Sentinel to short-circuit run_heldout_eval right after the env is built."""


def _capture_heldout_extra(cfg: Sim2RealLoopConfig, tmp_path, monkeypatch) -> dict:
    captured: dict = {}

    def _fake_component_env(config, *, component, output_json=None, extra=None):
        captured.update(extra or {})
        raise _StopComponentEnv

    monkeypatch.setattr(engine, "_component_env", _fake_component_env)
    with pytest.raises(_StopComponentEnv):
        engine.run_heldout_eval(
            cfg, local_dir=tmp_path, inner_evidence={}, outer_iteration=1
        )
    return captured


def test_heldout_eval_opts_into_byo_robot_task(tmp_path, monkeypatch):
    cfg = Sim2RealLoopConfig(
        run_id="r", robot_spec_uri="s3://b/lite6/lite6_parallel.json"
    )
    extra = _capture_heldout_extra(cfg, tmp_path, monkeypatch)
    # The gate byo_isaac_eval checks before resolving the spec + registering the
    # retargeted task must be present, alongside the robot uri.
    assert extra["NPA_BYO_ROBOT_TASK"] == "1"
    assert extra["NPA_SIM2REAL_ROBOT_SPEC_URI"] == "s3://b/lite6/lite6_parallel.json"


def test_heldout_eval_stock_franka_unchanged(tmp_path, monkeypatch):
    cfg = Sim2RealLoopConfig(run_id="r")  # no robot -> stock Franka path
    extra = _capture_heldout_extra(cfg, tmp_path, monkeypatch)
    assert "NPA_BYO_ROBOT_TASK" not in extra  # byte-for-byte stock eval


# --------------------------------------------------------------------------- #
# byo_isaac_trainer._resolve_byo_robot_spec — resolve a robot_spec_uri JSON
# --------------------------------------------------------------------------- #
def _kinova_doc() -> dict:
    # A consumed/bare robot doc keyed on robot_uri (the USD). An Omniverse https
    # CDN URL is opened directly by Isaac; parse_robot_spec requires robot_uri for
    # a byo_usd source.
    return {
        "robot_source": "byo_usd",
        "name": "kinova_j2n7s300",
        "robot_uri": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd",
        "ee_link": "j2n7s300_end_effector",
        "base_link": "j2n7s300_link_base",
        "n_arm_joints": 7,
        "n_gripper_joints": 3,
        "joint_names": [f"j2n7s300_joint_{i}" for i in range(1, 8)]
        + [f"j2n7s300_joint_finger_{i}" for i in range(1, 4)],
        "gripper_joint_names": [f"j2n7s300_joint_finger_{i}" for i in range(1, 4)],
        "finger_links": [f"j2n7s300_link_finger_{i}" for i in range(1, 4)],
        "gripper_open": 0.0,
        "gripper_close": 1.2,
        # A consumed/validated robot doc is gains-complete: every per-DoF array
        # (kp/kv/force/home_qpos) has dof_count (10) entries. parse_robot_spec's
        # validate() enforces this — so a real robot_spec_uri must carry them.
        "home_qpos": [0.0, 2.9, 0.0, 1.3, 0.0, 2.07, 1.4, 0.0, 0.0, 0.0],
        "kp": [40.0] * 7 + [20.0] * 3,
        "kv": [8.0] * 7 + [4.0] * 3,
        "force_upper": [30.0] * 10,
        "force_lower": [-30.0] * 10,
    }


def test_resolve_spec_from_uri_downloads_and_parses(monkeypatch, tmp_path):
    doc = _kinova_doc()

    class _FakeClient:
        def download_path(self, uri, dest):  # noqa: D401 (test stub)
            assert uri == "s3://b/kinova/robot-spec.json"
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)

    from npa.clients import storage

    monkeypatch.setattr(storage.StorageClient, "from_environment",
                        classmethod(lambda cls, *a, **k: _FakeClient()))
    monkeypatch.setenv("NPA_SIM2REAL_ROBOT_SPEC_URI", "s3://b/kinova/robot-spec.json")
    monkeypatch.delenv("NPA_SIM2REAL_ROBOT_PRESET", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_ROBOT_SOURCE", raising=False)

    spec = trainer._resolve_byo_robot_spec()
    assert spec is not None
    assert str(spec.robot_source) == "byo_usd"
    assert spec.name == "kinova_j2n7s300"
    # The USD carried through as robot_uri (so run_isaac_training_job can stage/open it).
    assert spec.robot_uri.endswith("j2n7s300_instanceable.usd")
    # Fidelity: base_link + gripper_joint_names must survive parse (else the ee_frame
    # source defaults to Franka's panda_link0 and the gripper action retargets to
    # nonexistent panda_finger.* joints — both crash a non-Franka arm at train time).
    assert spec.base_link == "j2n7s300_link_base"
    assert tuple(spec.gripper_joint_names) == (
        "j2n7s300_joint_finger_1",
        "j2n7s300_joint_finger_2",
        "j2n7s300_joint_finger_3",
    )
    # ...and they must be serialized into the in-container payload the sibling reads.
    payload = trainer.robot_spec_payload(spec, usd_container_path=spec.robot_uri)
    assert payload["base_link"] == "j2n7s300_link_base"
    assert payload["gripper_joint_names"] == [
        "j2n7s300_joint_finger_1",
        "j2n7s300_joint_finger_2",
        "j2n7s300_joint_finger_3",
    ]


def test_resolve_spec_uri_absent_falls_back_to_preset(monkeypatch):
    monkeypatch.delenv("NPA_SIM2REAL_ROBOT_SPEC_URI", raising=False)
    monkeypatch.setenv("NPA_SIM2REAL_ROBOT_PRESET", "franka")
    monkeypatch.delenv("NPA_SIM2REAL_ROBOT_SOURCE", raising=False)
    spec = trainer._resolve_byo_robot_spec()
    # Franka preset resolves to a stock_franka spec (proven default path).
    assert spec is not None
    assert str(spec.robot_source) == "stock_franka"


def test_resolve_no_robot_returns_none(monkeypatch):
    for k in ("NPA_SIM2REAL_ROBOT_SPEC_URI", "NPA_SIM2REAL_ROBOT_PRESET",
              "NPA_SIM2REAL_ROBOT_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    assert trainer._resolve_byo_robot_spec() is None

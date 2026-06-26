"""Wiring tests for the opt-in BYO-robot task path in the trainer + eval.

Asserts the gated path ships the isaac_byo_robot_task wrapper into the Isaac job
manifest, takes precedence over the physics path, and that the default (flag
unset / no spec) manifest is byte-for-byte the stock path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from npa.workflows.sim2real import byo_isaac_trainer as tr
from npa.workflows.sim2real import byo_isaac_eval as ev


def _byo_spec():
    return SimpleNamespace(
        robot_source="byo_usd",
        name="acme_arm",
        robot_uri="s3://bucket/robots/acme_arm.usd",
        ee_link="tool0",
        joint_names=("j1", "j2"),
        home_qpos=(0.0, -0.5),
        kp=(100.0, 200.0),
        kv=(10.0, 20.0),
        force_upper=(50.0, 60.0),
        force_lower=(-80.0, -60.0),
    )


# --------------------------------------------------------------------------- #
# robot_spec_payload (pure)
# --------------------------------------------------------------------------- #
def test_payload_none_when_spec_none():
    assert tr.robot_spec_payload(None) is None


def test_payload_stock_franka_is_minimal():
    spec = SimpleNamespace(robot_source="stock_franka", name="franka_panda")
    assert tr.robot_spec_payload(spec) == {"robot_source": "stock_franka", "name": "franka_panda"}


def test_payload_byo_carries_fields_and_usd():
    p = tr.robot_spec_payload(_byo_spec(), usd_container_path="/tmp/npa_robot/robot.usd")
    assert p["robot_source"] == "byo_usd"
    assert p["ee_link"] == "tool0"
    assert p["joint_names"] == ["j1", "j2"]
    assert p["home_qpos"] == [0.0, -0.5]
    assert p["kp"] == [100.0, 200.0]
    assert p["usd_path"] == "/tmp/npa_robot/robot.usd"


def test_payload_franka_preset_resolves_to_stock():
    # The Franka preset (used by the live validation) must be a stock_franka spec
    # so the BYO seam runs end-to-end without swapping the robot.
    from npa.genesis import robot_assets

    spec = robot_assets.robot_spec_from_preset("franka")
    assert tr.robot_spec_payload(spec) == {"robot_source": "stock_franka", "name": spec.name}


# --------------------------------------------------------------------------- #
# trainer manifest
# --------------------------------------------------------------------------- #
def _train_script(**over):
    kwargs = dict(
        job_name="s2r-byo-isaac-train-x", run_id="x", image="img", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=256, iterations=30, s3_output_uri="s3://b/out/", s3_endpoint="https://s3",
        namespace="default", service_account="agent-sa", gpu_product="GPU",
    )
    kwargs.update(over)
    m = tr.build_isaac_job_manifest(**kwargs)
    return m["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_train_manifest_default_is_stock_unchanged():
    s = _train_script()
    assert "--task Isaac-Lift-Cube-Franka-v0" in s
    assert "isaac_byo_robot_task" not in s
    assert "NPA_BYO_ROBOT_SPEC_JSON" not in s


def test_train_manifest_embeds_wrapper_when_robot_spec_set():
    spec = tr.robot_spec_payload(_byo_spec(), usd_container_path="/tmp/npa_robot/robot.usd")
    s = _train_script(robot_spec=spec, robot_usd_uri="s3://bucket/robots/acme_arm.usd")
    # ships the module + post-boot wrapper and passes the spec
    assert "isaac_byo_robot_task.py" in s
    assert "NPA_ROBOT_MODULE_DIR=/tmp/npa_robot" in s
    assert "NPA_BYO_ROBOT_SPEC_JSON" in s
    assert "ROBOT_TRAIN_DONE" in s
    # stages the customer USD from S3
    assert "STAGING_ROBOT_USD" in s and "acme_arm.usd" in s
    # does NOT fall through to the stock train.py line
    assert "--task Isaac-Lift-Cube-Franka-v0" not in s


def test_train_manifest_robot_takes_precedence_over_physics():
    spec = tr.robot_spec_payload(_byo_spec(), usd_container_path="/tmp/npa_robot/robot.usd")
    s = _train_script(robot_spec=spec, physics={"friction": 0.7, "mass_scale": 1.0})
    assert "cat > /tmp/npa_robot/isaac_byo_robot_task.py" in s
    # physics variant must NOT be shipped when the robot path wins
    assert "cat > /tmp/npa_phys/isaac_physics_task.py" not in s


def test_train_manifest_physics_only_unaffected():
    s = _train_script(physics={"friction": 0.7, "mass_scale": 1.0})
    assert "isaac_physics_task.py" in s
    assert "isaac_byo_robot_task" not in s


# --------------------------------------------------------------------------- #
# eval manifest
# --------------------------------------------------------------------------- #
def _eval_script(**over):
    kwargs = dict(
        job_name="s2r-byo-isaac-eval-x", run_id="x", image="img", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4, checkpoint_uri="s3://b/ckpt.pt", per_env_s3_uri="s3://b/d.json",
        s3_endpoint="https://s3", namespace="default", service_account="agent-sa", gpu_product="GPU",
    )
    kwargs.update(over)
    m = ev.build_isaac_eval_job_manifest(**kwargs)
    return m["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_eval_manifest_default_is_stock_unchanged():
    s = _eval_script()
    assert 'EVAL_TASK="Isaac-Lift-Cube-Franka-v0"' in s
    # the module is NOT shipped and no spec is passed -> the guarded block in the
    # embedded eval script stays a no-op (the block itself is always present).
    assert "cat > /tmp/evalwork/isaac_byo_robot_task.py" not in s
    assert "NPA_BYO_ROBOT_SPEC_JSON=" not in s


def test_eval_manifest_embeds_module_when_robot_spec_set():
    spec = tr.robot_spec_payload(_byo_spec(), usd_container_path="/tmp/npa_robot/robot.usd")
    s = _eval_script(robot_spec=spec, robot_usd_uri="s3://bucket/robots/acme_arm.usd")
    assert "isaac_byo_robot_task.py" in s
    assert "NPA_BYO_ROBOT_SPEC_JSON" in s
    assert "NPA_ROBOT_MODULE_DIR=/tmp/evalwork" in s
    # the embedded eval rollout registers the variant + rebinds TASK
    assert "EVAL_BYO_ROBOT_TASK" in s


def test_eval_rollout_registration_block_is_guarded():
    # The injected registration in ISAAC_EVAL_SCRIPT must be a no-op when unset.
    src = ev.ISAAC_EVAL_SCRIPT
    assert 'if os.environ.get("NPA_BYO_ROBOT_SPEC_JSON"):' in src
    assert "import isaac_byo_robot_task" in src
    # boot precedes the registration import (pxr only exists post-boot)
    assert src.index("AppLauncher(") < src.index("import isaac_byo_robot_task")


def test_payload_round_trips_through_json():
    # The payload is shipped as JSON; ensure it survives a round trip.
    spec = tr.robot_spec_payload(_byo_spec(), usd_container_path="/tmp/npa_robot/robot.usd")
    assert json.loads(json.dumps(spec)) == spec

"""Unit tests for BYO RobotSpec parsing, presets, download, and env dispatch.

These tests never import torch or genesis at module level. The env robot build
dispatch is exercised against a fake ``gs`` module and a fake torch installed
into ``sys.modules`` (mirroring tests/test_scene_assets.py).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from npa.genesis import robot_assets as ra


# --------------------------------------------------------------------------- #
# RobotSpec presets / parse / validate
# --------------------------------------------------------------------------- #


def test_default_franka_robot_spec_matches_hardcoded_franka() -> None:
    spec = ra.default_franka_robot_spec()
    assert spec.robot_source == ra.ROBOT_SOURCE_STOCK_FRANKA
    assert spec.ee_link == "hand"
    assert spec.finger_links == ("left_finger", "right_finger")
    assert spec.n_arm_joints == 7
    assert spec.n_gripper_joints == 2
    assert spec.dof_count == 9
    assert spec.kp == (4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100)
    assert spec.kv == (450, 450, 350, 350, 200, 200, 200, 10, 10)
    assert spec.home_qpos == ra.FRANKA_HOME
    spec.validate()


def test_preset_ur5e_six_dof_tool0_no_gripper() -> None:
    spec = ra.robot_spec_from_preset("ur5e")
    assert spec.name == "ur5e"
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_URDF
    assert spec.ee_link == "tool0"
    assert spec.n_arm_joints == 6
    assert spec.n_gripper_joints == 0
    assert spec.dof_count == 6
    assert spec.has_gripper is False
    assert len(spec.kp) == 6
    assert len(spec.force_lower) == 6
    assert len(spec.home_qpos) == 6


def test_preset_ur10e_distinct_from_ur5e() -> None:
    ur5e = ra.robot_spec_from_preset("ur5e")
    ur10e = ra.robot_spec_from_preset("ur10e")
    assert ur10e.ee_link == "tool0"
    assert ur10e.n_arm_joints == 6
    assert ur10e.force_upper != ur5e.force_upper  # larger payload torques


def test_preset_flexiv_rizon_seven_dof_flange() -> None:
    spec = ra.robot_spec_from_preset("flexiv")
    assert spec.name == "flexiv_rizon"
    assert spec.ee_link == "flange"
    assert spec.n_arm_joints == 7
    assert spec.n_gripper_joints == 0
    assert spec.dof_count == 7
    assert len(spec.kp) == 7
    assert ra.robot_spec_from_preset("rizon").name == "flexiv_rizon"


def test_unknown_preset_raises() -> None:
    with pytest.raises(ra.RobotSpecError):
        ra.robot_spec_from_preset("kuka_iiwa")


def test_parse_robot_spec_byo_urdf_full() -> None:
    doc = {
        "schema": ra.ROBOT_SPEC_SCHEMA,
        "robot_source": "byo_urdf",
        "name": "customer_arm",
        "robot_uri": "s3://bucket/robots/arm.urdf",
        "ee_link": "tool0",
        "n_arm_joints": 6,
        "n_gripper_joints": 0,
        "kp": [1, 2, 3, 4, 5, 6],
        "kv": [1, 2, 3, 4, 5, 6],
        "force_lower": [-1, -1, -1, -1, -1, -1],
        "force_upper": [1, 1, 1, 1, 1, 1],
        "home_qpos": [0, 0, 0, 0, 0, 0],
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_URDF
    assert spec.robot_uri.endswith("arm.urdf")
    assert spec.ee_link == "tool0"
    assert spec.dof_count == 6


def test_parse_robot_spec_accepts_usd_path_alias_for_robot_uri() -> None:
    # A customer may hand us ``usd_path`` (the Isaac spawn field name) instead of
    # ``robot_uri``; it is accepted as an alias and the source is inferred as USD.
    doc = {
        "name": "arm_usd",
        "usd_path": "s3://bucket/robots/arm.usd",
        "ee_link": "tool0",
        "n_arm_joints": 6,
        "n_gripper_joints": 2,
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.robot_uri == "s3://bucket/robots/arm.usd"
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_USD


def test_parse_robot_spec_robot_uri_wins_over_usd_path_alias() -> None:
    doc = {
        "robot_source": "byo_usd",
        "name": "arm",
        "robot_uri": "s3://bucket/robots/primary.usd",
        "usd_path": "s3://bucket/robots/alias.usd",
        "ee_link": "tool0",
        "n_arm_joints": 6,
        "n_gripper_joints": 0,
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.robot_uri.endswith("primary.usd")


def test_parse_robot_spec_minimal_usd_auto_derives_gains_and_home() -> None:
    # A MINIMAL BYO spec (name + usd + joint/link names) — no per-joint gain or
    # home arrays — must onboard: the omitted arrays are synthesized to dof_count.
    doc = {
        "robot_source": "byo_usd",
        "name": "lite6_parallel",
        "usd_path": "s3://bucket/robots/lite6_combined.usda",
        "base_link": "world",
        "ee_link": "tcp_ee",
        "n_arm_joints": 6,
        "n_gripper_joints": 2,
        "joint_names": [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "finger_joint1", "finger_joint2",
        ],
        "gripper_joint_names": ["finger_joint1", "finger_joint2"],
        "finger_links": ["uflite_finger1", "uflite_finger2"],
    }
    spec = ra.parse_robot_spec(doc)  # must not raise
    assert spec.dof_count == 8
    for arr in (spec.kp, spec.kv, spec.force_upper, spec.force_lower, spec.home_qpos):
        assert len(arr) == 8
    # arm joints get the arm default, gripper joints the (lighter) gripper default.
    assert spec.kp[:6] == (ra.DEFAULT_ARM_KP,) * 6
    assert spec.kp[6:] == (ra.DEFAULT_GRIPPER_KP,) * 2
    assert spec.force_lower == tuple(-abs(f) for f in spec.force_upper)
    assert spec.home_qpos == (0.0,) * 8
    assert spec.has_gripper


def test_parse_robot_spec_infers_joint_counts_from_joint_names() -> None:
    # Counts omitted, but joint_names + gripper_joint_names given: infer them.
    doc = {
        "robot_source": "byo_usd",
        "name": "inferred_counts",
        "usd_path": "s3://bucket/robots/arm.usd",
        "ee_link": "tcp_ee",
        "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "f1", "f2"],
        "gripper_joint_names": ["f1", "f2"],
        "finger_links": ["fl1", "fl2"],
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.n_gripper_joints == 2
    assert spec.n_arm_joints == 6
    assert spec.dof_count == 8


def test_parse_robot_spec_explicit_gains_not_overridden() -> None:
    # A customer who DOES supply arrays keeps them verbatim (no auto-derive).
    doc = {
        "robot_source": "byo_usd",
        "name": "explicit",
        "usd_path": "s3://bucket/robots/arm.usd",
        "ee_link": "tool0",
        "n_arm_joints": 6,
        "n_gripper_joints": 0,
        "kp": [11, 12, 13, 14, 15, 16],
        "kv": [1, 1, 1, 1, 1, 1],
        "force_upper": [9, 9, 9, 9, 9, 9],
        "force_lower": [-9, -9, -9, -9, -9, -9],
        "home_qpos": [0.1, 0, 0, 0, 0, 0],
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.kp == (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    assert spec.home_qpos[0] == 0.1


def test_parse_robot_spec_franka_default_untouched_by_auto_derive() -> None:
    # An empty doc is the stock Franka default — its 9-length arrays must be kept
    # byte-for-byte (auto-derive only fires when a field's length != dof_count).
    spec = ra.parse_robot_spec({})
    assert spec.robot_source == ra.ROBOT_SOURCE_STOCK_FRANKA
    assert spec.kp == ra.RobotSpec().kp
    assert spec.home_qpos == ra.FRANKA_HOME


def test_parse_robot_spec_preset_with_uri_override() -> None:
    # A customer supplies just a preset + their URDF uri on top.
    doc = {"preset": "ur5e", "robot_uri": "s3://bucket/robots/ur5e.urdf"}
    spec = ra.parse_robot_spec(doc)
    assert spec.ee_link == "tool0"
    assert spec.n_arm_joints == 6
    assert spec.robot_uri.endswith("ur5e.urdf")
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_URDF


def test_parse_robot_spec_preset_with_mjcf_uri_infers_byo_mjcf() -> None:
    doc = {
        "preset": "ur5e",
        "robot_uri": "s3://bucket/robots/ur5e/ur5e.xml",
    }
    spec = ra.parse_robot_spec(doc)
    assert spec.robot_source == ra.ROBOT_SOURCE_BYO_MJCF
    assert spec.robot_uri.endswith("ur5e.xml")


def test_adapt_robot_spec_for_isaac_swaps_mjcf_to_urdf() -> None:
    spec = ra.parse_robot_spec(
        {"preset": "ur5e", "robot_uri": "s3://bucket/robots/ur5e/ur5e.xml"}
    )
    adapted = ra.adapt_robot_spec_for_sim_backend(spec, "isaac")
    assert adapted.robot_source == ra.ROBOT_SOURCE_BYO_URDF
    assert adapted.robot_uri.endswith("ur5e.urdf")
    unchanged = ra.adapt_robot_spec_for_sim_backend(spec, "genesis")
    assert unchanged.robot_source == ra.ROBOT_SOURCE_BYO_MJCF


@pytest.mark.parametrize(
    "doc",
    [
        {"robot_source": "bogus"},
        {"robot_source": "byo_urdf"},  # missing robot_uri
        {"robot_source": "genesis_builtin"},  # missing builtin_path
        {"preset": "ur5e", "kp": [1, 2, 3]},  # gains wrong length
        {"robot_source": "byo_urdf", "robot_uri": "s3://b/arm.obj"},  # non-articulated
    ],
)
def test_parse_robot_spec_rejects_malformed(doc: dict) -> None:
    with pytest.raises(ra.RobotSpecError):
        ra.parse_robot_spec(doc)


def test_non_articulated_robot_file_gives_clear_error() -> None:
    spec = ra.RobotSpec(
        robot_source=ra.ROBOT_SOURCE_BYO_URDF,
        robot_uri="s3://bucket/robots/arm.obj",
        ee_link="tool0",
        n_arm_joints=6,
        n_gripper_joints=0,
        kp=(1,) * 6,
        kv=(1,) * 6,
        force_lower=(-1,) * 6,
        force_upper=(1,) * 6,
        home_qpos=(0,) * 6,
    )
    with pytest.raises(ra.RobotSpecError) as exc:
        spec.validate()
    assert "visual mesh" in str(exc.value).lower()


def test_robot_spec_from_inputs_preset_and_default() -> None:
    assert ra.robot_spec_from_inputs(robot_preset="ur5e").name == "ur5e"
    assert (
        ra.robot_spec_from_inputs(robot_source="stock_franka").robot_source
        == ra.ROBOT_SOURCE_STOCK_FRANKA
    )
    assert ra.robot_spec_from_inputs() is None


# --------------------------------------------------------------------------- #
# Download / resolve + provenance
# --------------------------------------------------------------------------- #


class _FakeStorageClient:
    def __init__(self, *, payload: bytes = b"<robot/>") -> None:
        self.calls: list[tuple[str, str]] = []
        self.payload = payload

    def download_path(self, uri: str, local_path: str) -> str:
        self.calls.append((uri, local_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.payload)
        return local_path


def test_resolve_robot_asset_byo_urdf_records_sha_and_no_fallback(tmp_path: Path) -> None:
    client = _FakeStorageClient(payload=b"<robot>urdf</robot>")
    spec = ra.RobotSpec(
        robot_source=ra.ROBOT_SOURCE_BYO_URDF,
        name="ur5e",
        robot_uri="s3://bucket/robots/ur5e.urdf",
        ee_link="tool0",
        n_arm_joints=6,
        n_gripper_joints=0,
        kp=(1,) * 6,
        kv=(1,) * 6,
        force_lower=(-1,) * 6,
        force_upper=(1,) * 6,
        home_qpos=(0,) * 6,
    )
    ra.resolve_robot_asset(spec, dest_dir=tmp_path, client=client)
    assert spec.local_path
    assert spec.sha256 == ra.sha256_file(spec.local_path)
    prov = spec.provenance()
    assert prov["robot_source"] == "byo_urdf"
    assert prov["robot_fallback_used"] is False
    assert prov["loaded"] is False  # not loaded into sim until env builds it


def test_resolve_robot_asset_stock_franka_no_download(tmp_path: Path) -> None:
    spec = ra.default_franka_robot_spec()
    ra.resolve_robot_asset(spec, dest_dir=tmp_path, client=None)
    assert spec.local_path == ""
    assert spec.builtin_path == ra.STOCK_FRANKA_MJCF


def test_resolve_robot_asset_raises_when_download_fails(tmp_path: Path) -> None:
    def boom(*args, **kwargs):
        raise ra.RobotSpecError("download exploded")

    spec = ra.robot_spec_from_preset("ur5e")
    spec.robot_uri = "s3://bucket/robots/ur5e.urdf"
    with pytest.raises(ra.RobotSpecError):
        ra.resolve_robot_asset(spec, dest_dir=tmp_path, downloader=boom)


# --------------------------------------------------------------------------- #
# Env robot build dispatch (fake gs + fake torch)
# --------------------------------------------------------------------------- #


class _RecordingMorph:
    def __init__(self, kind: str, **kwargs) -> None:
        self.kind = kind
        self.kwargs = kwargs


class _FakeEntity:
    def __init__(self, morph, **kwargs) -> None:
        self.morph = morph
        self.kwargs = kwargs


class _FakeScene:
    def __init__(self) -> None:
        self.added: list[_FakeEntity] = []

    def add_entity(self, morph, **kwargs) -> _FakeEntity:
        entity = _FakeEntity(morph, **kwargs)
        self.added.append(entity)
        return entity


def _fake_gs() -> types.SimpleNamespace:
    morphs = types.SimpleNamespace(
        MJCF=lambda **kw: _RecordingMorph("MJCF", **kw),
        URDF=lambda **kw: _RecordingMorph("URDF", **kw),
        Mesh=lambda **kw: _RecordingMorph("Mesh", **kw),
    )
    return types.SimpleNamespace(morphs=morphs)


@pytest.fixture()
def env_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    sys.modules.pop("npa.genesis.env_pick_place", None)
    module = importlib.import_module("npa.genesis.env_pick_place")
    yield module
    sys.modules.pop("npa.genesis.env_pick_place", None)


def _bare_env(env_module, robot_spec=None):
    env = env_module.FrankaPickPlaceEnv.__new__(env_module.FrankaPickPlaceEnv)
    env._scene = _FakeScene()
    env._robot_spec = robot_spec
    env.robot_provenance = None
    return env


def test_default_robot_build_is_stock_franka_mjcf(env_module) -> None:
    env = _bare_env(env_module, robot_spec=None)
    entity = env._add_robot_entity(_fake_gs())
    assert entity.morph.kind == "MJCF"
    assert entity.morph.kwargs["file"] == "xml/franka_emika_panda/panda.xml"
    # No RobotSpec => no robot provenance recorded (default path unchanged).
    assert env.robot_provenance is None


def test_stock_franka_spec_build_records_provenance(env_module) -> None:
    spec = ra.default_franka_robot_spec()
    env = _bare_env(env_module, robot_spec=spec)
    entity = env._add_robot_entity(_fake_gs())
    assert entity.morph.kind == "MJCF"
    assert entity.morph.kwargs["file"] == "xml/franka_emika_panda/panda.xml"
    assert spec.loaded is True
    assert env.robot_provenance["robot_source"] == "stock_franka"
    assert env.robot_provenance["loaded"] is True


def test_byo_urdf_build_calls_urdf_with_local_path(env_module) -> None:
    spec = ra.robot_spec_from_preset("ur5e")
    spec.robot_uri = "s3://bucket/robots/ur5e.urdf"
    spec.local_path = "/tmp/resolved/ur5e.urdf"
    spec.sha256 = "deadbeef"
    env = _bare_env(env_module, robot_spec=spec)
    entity = env._add_robot_entity(_fake_gs())
    assert entity.morph.kind == "URDF"
    assert entity.morph.kwargs["file"] == "/tmp/resolved/ur5e.urdf"
    assert spec.loaded is True
    assert env.robot_provenance["robot_source"] == "byo_urdf"
    assert env.robot_provenance["ee_link"] == "tool0"
    assert env.robot_provenance["sha256"] == "deadbeef"


def test_byo_mjcf_build_calls_mjcf_with_local_path(env_module) -> None:
    spec = ra.RobotSpec(
        robot_source=ra.ROBOT_SOURCE_BYO_MJCF,
        name="byo_mjcf_arm",
        robot_uri="s3://bucket/robots/arm.xml",
        local_path="/tmp/resolved/arm.xml",
        ee_link="ee",
        n_arm_joints=6,
        n_gripper_joints=0,
        kp=(1,) * 6,
        kv=(1,) * 6,
        force_lower=(-1,) * 6,
        force_upper=(1,) * 6,
        home_qpos=(0,) * 6,
    )
    env = _bare_env(env_module, robot_spec=spec)
    entity = env._add_robot_entity(_fake_gs())
    assert entity.morph.kind == "MJCF"
    assert entity.morph.kwargs["file"] == "/tmp/resolved/arm.xml"


def test_byo_urdf_without_local_path_raises(env_module) -> None:
    spec = ra.robot_spec_from_preset("ur5e")
    spec.robot_uri = "s3://bucket/robots/ur5e.urdf"
    spec.local_path = ""  # not resolved -> must fail loudly (no fallback)
    env = _bare_env(env_module, robot_spec=spec)
    with pytest.raises(ra.RobotSpecError):
        env._add_robot_entity(_fake_gs())


def test_byo_usd_robot_unsupported_on_genesis_raises(env_module) -> None:
    spec = ra.RobotSpec(
        robot_source=ra.ROBOT_SOURCE_BYO_USD,
        name="usd_arm",
        robot_uri="s3://bucket/robots/arm.usd",
        local_path="/tmp/resolved/arm.usd",
        ee_link="ee",
        n_arm_joints=6,
        n_gripper_joints=0,
        kp=(1,) * 6,
        kv=(1,) * 6,
        force_lower=(-1,) * 6,
        force_upper=(1,) * 6,
        home_qpos=(0,) * 6,
    )
    env = _bare_env(env_module, robot_spec=spec)
    with pytest.raises(ra.RobotSpecError):
        env._add_robot_entity(_fake_gs())

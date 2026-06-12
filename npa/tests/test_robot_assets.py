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


def test_parse_robot_spec_preset_with_uri_override() -> None:
    # A customer supplies just a preset + their URDF uri on top.
    doc = {"preset": "ur5e", "robot_uri": "s3://bucket/robots/ur5e.urdf"}
    spec = ra.parse_robot_spec(doc)
    assert spec.ee_link == "tool0"
    assert spec.n_arm_joints == 6
    assert spec.robot_uri.endswith("ur5e.urdf")


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

"""Reject XML entities before SONIC hands scene files to the simulator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from defusedxml.common import EntitiesForbidden
import pytest


def _model_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    source = (
        Path(__file__).resolve().parents[2] / "docker/workbench/sonic/mujoco_eval.py"
    )
    calls = []
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
    simulator = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: calls.append(path) or model)
    )
    spec = importlib.util.spec_from_file_location("sonic_xml_security_adapter", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "mujoco", simulator)
        imports.setitem(sys.modules, "torch", SimpleNamespace())
        spec.loader.exec_module(module)
    monkeypatch.setenv("SONIC_HOME", str(tmp_path))
    return module._load_model, calls, model


@pytest.mark.parametrize("target", ["scene", "robot"])
def test_entities_are_denied_in_scene_and_included_robot(monkeypatch, tmp_path, target):
    load, calls, _ = _model_loader(monkeypatch, tmp_path)
    scene = '<mujoco><include file="robot.xml"/></mujoco>'
    robot = '<mujoco model="&entity;"><worldbody/></mujoco>'
    declaration = '<!DOCTYPE mujoco [<!ENTITY entity "expanded">]>'
    if target == "scene":
        scene = declaration + scene.replace("<mujoco>", '<mujoco model="&entity;">')
        robot = "<mujoco><worldbody/></mujoco>"
    else:
        robot = declaration + robot
    (tmp_path / "scene.xml").write_text(scene)
    (tmp_path / "robot.xml").write_text(robot)
    with pytest.raises(EntitiesForbidden):
        load({"ROBOT_SCENE": "scene.xml"})
    assert calls == []


def test_ordinary_scene_still_reaches_simulator(monkeypatch, tmp_path):
    load, calls, model = _model_loader(monkeypatch, tmp_path)
    (tmp_path / "scene.xml").write_text('<mujoco><include file="robot.xml"/></mujoco>')
    (tmp_path / "robot.xml").write_text("<mujoco><worldbody/></mujoco>")
    assert load({"ROBOT_SCENE": "scene.xml", "SIMULATE_DT": 0.005}) is model
    assert calls == [str(tmp_path / "scene.xml")]
    assert model.opt.timestep == 0.005

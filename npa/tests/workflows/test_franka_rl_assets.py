"""Verify exact procedural asset bytes, real USD collisions, and explicit asset selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.franka_rl_assets import ASSET_NAMES, configure_assets, write_assets


def test_assets_are_deterministic_and_include_non_cube_contact_geometry(tmp_path):
    first = write_assets(tmp_path / "first", "spool")
    second = write_assets(tmp_path / "second", "spool")
    assert first == second
    for name in ASSET_NAMES:
        assert (
            "PhysicsRigidBodyAPI"
            in (tmp_path / "first/assets" / f"{name}.usda").read_text()
        )
    assert "Sector5" in (tmp_path / "first/assets/hex_nut.usda").read_text()


@pytest.mark.parametrize(
    "name,collisions", [("spool", 3), ("hex_nut", 6), ("bottle", 3), ("fixture", 6)]
)
def test_actual_usd_parser_reads_asset_and_collision_shapes(tmp_path, name, collisions):
    usd = pytest.importorskip("pxr.Usd")
    physics = pytest.importorskip("pxr.UsdPhysics")
    write_assets(tmp_path, "spool")
    stage = usd.Stage.Open(str(tmp_path / f"assets/{name}.usda"))
    assert stage.GetDefaultPrim().GetName() == "Asset"
    assert (
        sum(prim.HasAPI(physics.CollisionAPI) for prim in stage.Traverse())
        == collisions
    )
    assert not stage.GetRootLayer().subLayerPaths


def test_unknown_asset_fails_before_any_artifact_is_created(tmp_path):
    with pytest.raises(ValueError, match="Unknown Franka asset"):
        write_assets(tmp_path, "unknown")
    assert not list(tmp_path.iterdir())


def test_asset_mutation_refuses_simulation_instead_of_falling_back(
    tmp_path, monkeypatch
):
    import sys

    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(sim=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "isaaclab.assets", SimpleNamespace(AssetBaseCfg=object)
    )
    manifest = write_assets(tmp_path, "hex_nut")
    Path(tmp_path / "assets/hex_nut.usda").write_text("changed")
    with pytest.raises(ValueError, match="asset bytes changed"):
        configure_assets(SimpleNamespace(), manifest, tmp_path)


def test_replacing_geometry_preserves_task_solver_without_cube_scale(
    tmp_path, monkeypatch
):
    import sys

    class Asset(SimpleNamespace):
        InitialStateCfg = SimpleNamespace

    sim = SimpleNamespace(
        UsdFileCfg=SimpleNamespace,
        RigidBodyPropertiesCfg=SimpleNamespace,
        MassPropertiesCfg=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(sim=sim))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(
        sys.modules, "isaaclab.assets", SimpleNamespace(AssetBaseCfg=Asset)
    )
    rigid = SimpleNamespace(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=5,
        max_linear_velocity=1000,
    )
    original = SimpleNamespace(rigid_props=rigid, scale=(0.8, 0.8, 0.8))
    obj = SimpleNamespace(spawn=original, init_state=SimpleNamespace())
    config = SimpleNamespace(scene=SimpleNamespace(object=obj))
    manifest = write_assets(tmp_path, "spool")
    configure_assets(config, manifest, tmp_path)
    assert obj.spawn.rigid_props == rigid and obj.spawn.rigid_props is not rigid
    assert not hasattr(obj.spawn, "scale")
    assert obj.spawn.mass_props.mass == manifest["nominal_mass_kg"]
    obj.spawn.rigid_props.solver_position_iteration_count = 64
    assert original.rigid_props.solver_position_iteration_count == 16


@pytest.mark.parametrize("target", ASSET_NAMES)
def test_sealed_distractors_rest_on_authored_tray_surface(tmp_path, target):
    usd = pytest.importorskip("pxr.Usd")
    manifest = write_assets(tmp_path, target)
    for name, position in manifest["distractor_positions_m"].items():
        stage = usd.Stage.Open(str(tmp_path / f"assets/{name}.usda"))
        lower = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Cylinder":
                z = prim.GetAttribute("xformOp:translate").Get()[2]
                lower.append(z - prim.GetAttribute("height").Get() / 2)
            elif prim.GetTypeName() == "Mesh":
                lower.extend(point[2] for point in prim.GetAttribute("points").Get())
        tray = usd.Stage.Open(str(tmp_path / "assets/fixture.usda"))
        base = tray.GetPrimAtPath("/Asset/TrayBase")
        top = (
            base.GetAttribute("xformOp:translate").Get()[2]
            + base.GetAttribute("xformOp:scale").Get()[2] / 2
        )
        assert position[2] + min(lower) == pytest.approx(top)

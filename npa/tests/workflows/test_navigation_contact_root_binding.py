"""Bind native articulation pose indices to real USD root APIs without suffix guesses."""

from types import SimpleNamespace

import pytest
from pxr import Usd, UsdPhysics

from npa.workflows.navigation.contact_surface import _root_indices


def _container(robot):
    return f"/World/envs/env_{robot}/Robot"


def _environment(paths, api_paths, containers=(0, 1)):
    stage = Usd.Stage.CreateInMemory()
    for robot in containers:
        stage.DefinePrim(_container(robot), "Xform")
    for path in api_paths:
        UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim(path, "Xform"))
    return SimpleNamespace(
        num_envs=2,
        sim=SimpleNamespace(stage=stage),
        scene={"robot": SimpleNamespace(root_view=SimpleNamespace(prim_paths=paths))},
    )


@pytest.mark.parametrize("suffix", ["", "/floating_link", "/assembly/floating_link"])
def test_resolved_root_api_maps_reordered_native_view_without_stage_mutation(suffix):
    paths = [_container(robot) + suffix for robot in (1, 0)]
    env = _environment(paths, paths)
    before = env.sim.stage.GetRootLayer().ExportToString()
    assert _root_indices(env) == {1: 0, 0: 1}
    assert env.sim.stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    "paths,match",
    [
        ([_container(0)], "exactly once"),
        ([_container(0)] * 2, "exactly once"),
        ([_container(0), _container(2)], "population"),
        ([_container(0), "/World/envs/env_1/OtherRobot"], "native robot path"),
        ([_container(0), "/World/envs/env_01/Robot"], "resolved articulation"),
    ],
)
def test_native_roots_reject_missing_duplicate_or_foreign_population(paths, match):
    env = _environment(paths, [_container(0), _container(1)])
    with pytest.raises(ValueError, match=match):
        _root_indices(env)


@pytest.mark.parametrize("wrong_suffix", ["", "/unrelated_body"])
def test_native_container_or_wrong_body_cannot_replace_nested_articulation(
    wrong_suffix,
):
    actual = [_container(robot) + "/floating_link" for robot in range(2)]
    env = _environment([actual[0], _container(1) + wrong_suffix], actual)
    env.sim.stage.DefinePrim(_container(1) + "/unrelated_body", "Xform")
    with pytest.raises(ValueError, match="resolved articulation"):
        _root_indices(env)


def test_nested_native_path_without_articulation_api_is_rejected():
    paths = [_container(robot) + "/floating_link" for robot in range(2)]
    env = _environment(paths, paths[:1])
    env.sim.stage.DefinePrim(paths[1], "Xform")
    with pytest.raises(ValueError, match="exactly one ArticulationRootAPI"):
        _root_indices(env)


def test_ambiguous_api_prims_are_rejected_even_when_native_path_has_api():
    paths = [_container(robot) + "/floating_link" for robot in range(2)]
    env = _environment(paths, [*paths, _container(1) + "/second_root"])
    with pytest.raises(ValueError, match="exactly one ArticulationRootAPI"):
        _root_indices(env)


def test_absent_canonical_container_is_rejected():
    env = _environment([_container(0), _container(1)], [_container(0)], containers=(0,))
    with pytest.raises(ValueError, match="container is absent"):
        _root_indices(env)


def test_instance_proxy_api_is_excluded_like_native_lab_resolution():
    paths = [_container(robot) + "/floating_link" for robot in range(2)]
    env = _environment(paths, paths)
    stage = env.sim.stage
    stage.DefinePrim("/Prototype", "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim("/Prototype/hidden", "Xform"))
    proxy = stage.DefinePrim(_container(1) + "/instance", "Xform")
    proxy.GetReferences().AddInternalReference("/Prototype")
    proxy.SetInstanceable(True)
    assert stage.GetPrimAtPath(str(proxy.GetPath()) + "/hidden").IsInstanceProxy()
    assert _root_indices(env) == {0: 0, 1: 1}

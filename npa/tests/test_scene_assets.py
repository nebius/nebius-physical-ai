"""Unit tests for BYO SceneSpec parsing, asset download, and env build dispatch.

These tests never import torch or genesis at module level. The env build
dispatch is exercised against a fake ``gs`` module and a fake torch installed
into ``sys.modules`` (mirroring tests/test_genesis.py).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from npa.genesis import scene_assets as sa


# --------------------------------------------------------------------------- #
# SceneSpec parse / validate
# --------------------------------------------------------------------------- #


def test_parse_scene_spec_byo_mesh_object() -> None:
    doc = {
        "schema": sa.SCENE_SPEC_SCHEMA,
        "goal_pos": [0.5, 0.3, 0.04],
        "goal_threshold": 0.05,
        "objects": [
            {
                "name": "widget",
                "asset_source": "byo_mesh",
                "role": "manipuland",
                "uri": "s3://bucket/sim2real-assets/run/object.obj",
                "scale": 1.5,
                "pos": [0.5, 0.0, 0.05],
                "color": [0.1, 0.8, 0.2],
                "mass": 0.3,
                "friction": 0.9,
            }
        ],
    }
    spec = sa.parse_scene_spec(doc, source_uri="s3://bucket/spec.json")

    assert len(spec.objects) == 1
    obj = spec.manipuland()
    assert obj.asset_source == sa.ASSET_SOURCE_BYO_MESH
    assert obj.uri.endswith("object.obj")
    assert obj.scale == 1.5
    assert obj.pos == (0.5, 0.0, 0.05)
    assert obj.mass == 0.3
    assert obj.friction == 0.9
    assert spec.goal_pos == (0.5, 0.3, 0.04)
    assert spec.source_uri == "s3://bucket/spec.json"


def test_parse_scene_spec_supports_multiple_objects_and_target() -> None:
    doc = {
        "objects": [
            {"name": "cube", "asset_source": "primitive", "primitive": "box"},
            {
                "name": "table",
                "asset_source": "genesis_builtin",
                "role": "static",
                "builtin_path": "meshes/table.obj",
            },
        ],
        "goal_pos": [0.4, 0.2, 0.04],
    }
    spec = sa.parse_scene_spec(doc)

    assert [o.name for o in spec.objects] == ["cube", "table"]
    table = spec.objects[1]
    assert table.asset_source == sa.ASSET_SOURCE_GENESIS_BUILTIN
    assert table.role == sa.ROLE_STATIC
    assert table.fixed is True  # static defaults to fixed
    assert spec.manipuland().name == "cube"


@pytest.mark.parametrize(
    "doc",
    [
        {},  # no objects
        {"objects": []},  # empty objects
        {"objects": [{"name": "x", "asset_source": "bogus"}]},  # bad source
        {"objects": [{"name": "x", "asset_source": "byo_mesh"}]},  # missing uri
        {"objects": [{"name": "x", "asset_source": "genesis_builtin"}]},  # missing path
        {"objects": [{"name": "x", "asset_source": "primitive", "primitive": "cone"}]},
        {
            "objects": [
                {"name": "t", "asset_source": "primitive", "role": "target"}
            ]
        },  # no manipuland
        {
            "objects": [{"name": "x", "asset_source": "primitive"}],
            "goal_pos": [0.1, 0.2],
        },  # bad goal_pos
    ],
)
def test_parse_scene_spec_rejects_malformed(doc: dict) -> None:
    with pytest.raises(sa.SceneSpecError):
        sa.parse_scene_spec(doc)


def test_synthesize_scene_spec_from_mesh_uri() -> None:
    spec = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/object.obj")
    obj = spec.manipuland()
    assert obj.asset_source == sa.ASSET_SOURCE_BYO_MESH
    assert obj.uri == "s3://bucket/run/object.obj"
    assert spec.source_uri == "s3://bucket/run/object.obj"


def test_synthesize_scene_spec_requires_uri() -> None:
    with pytest.raises(sa.SceneSpecError):
        sa.synthesize_scene_spec()


def test_default_scene_spec_matches_red_box_primitive() -> None:
    spec = sa.default_scene_spec()
    obj = spec.manipuland()
    assert obj.asset_source == sa.ASSET_SOURCE_PRIMITIVE
    assert obj.primitive == sa.PRIMITIVE_BOX
    assert obj.size == (0.04, 0.04, 0.04)
    assert obj.pos == (0.5, 0.0, 0.04)
    assert obj.color == (1.0, 0.0, 0.0)
    assert spec.goal_pos == (0.5, 0.3, 0.04)


# --------------------------------------------------------------------------- #
# Asset download + resolve + provenance
# --------------------------------------------------------------------------- #


class _FakeStorageClient:
    """Records download_path calls and writes a fake mesh file."""

    def __init__(self, *, payload: bytes = b"OBJ-DATA") -> None:
        self.calls: list[tuple[str, str]] = []
        self.payload = payload

    def download_path(self, uri: str, local_path: str) -> str:
        self.calls.append((uri, local_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.payload)
        return local_path


def test_download_asset_s3_uses_client_and_validates(tmp_path: Path) -> None:
    client = _FakeStorageClient()
    dest = sa.download_asset(
        "s3://bucket/run/object.obj", tmp_path / "obj", client=client
    )
    assert dest.is_file()
    assert dest.name == "object.obj"
    assert dest.read_bytes() == b"OBJ-DATA"
    assert client.calls[0][0] == "s3://bucket/run/object.obj"


def test_download_asset_local_path(tmp_path: Path) -> None:
    src = tmp_path / "src.obj"
    src.write_bytes(b"v 0 0 0\n")
    dest = sa.download_asset(str(src), tmp_path / "out")
    assert dest.read_bytes() == b"v 0 0 0\n"


def test_download_asset_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.obj"
    empty.write_bytes(b"")
    with pytest.raises(sa.SceneSpecError):
        sa.download_asset(str(empty), tmp_path / "out")


def test_resolve_scene_assets_byo_records_sha_source_and_no_fallback(
    tmp_path: Path,
) -> None:
    client = _FakeStorageClient(payload=b"MESH-BYTES")
    spec = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/object.obj")
    sa.resolve_scene_assets(spec, dest_dir=tmp_path, client=client)

    obj = spec.manipuland()
    assert obj.local_path
    assert obj.sha256 == sa.sha256_file(obj.local_path)
    block = spec.provenance_block()
    assert block["asset_fallback_used"] is False
    assert block["objects"][0]["asset_source"] == "byo_mesh"
    assert block["objects"][0]["sha256"] == obj.sha256
    # Not yet loaded into the sim until the env builds it.
    assert block["objects"][0]["loaded"] is False


def test_resolve_scene_assets_genesis_builtin_uses_resolver(tmp_path: Path) -> None:
    builtin_file = tmp_path / "builtin" / "duck.obj"
    builtin_file.parent.mkdir(parents=True)
    builtin_file.write_bytes(b"BUILTIN-MESH")

    doc = {
        "objects": [
            {
                "name": "duck",
                "asset_source": "genesis_builtin",
                "builtin_path": "meshes/duck.obj",
            }
        ]
    }
    spec = sa.parse_scene_spec(doc)
    sa.resolve_scene_assets(
        spec,
        dest_dir=tmp_path,
        builtin_resolver=lambda path: builtin_file,
    )
    obj = spec.manipuland()
    assert obj.asset_source == "genesis_builtin"
    assert obj.local_path == str(builtin_file)
    assert obj.sha256 == sa.sha256_file(builtin_file)


def test_resolve_scene_assets_raises_when_download_fails(tmp_path: Path) -> None:
    def boom(*args, **kwargs):
        raise sa.SceneSpecError("download exploded")

    spec = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/object.obj")
    with pytest.raises(sa.SceneSpecError):
        sa.resolve_scene_assets(spec, dest_dir=tmp_path, downloader=boom)


# --------------------------------------------------------------------------- #
# Env build dispatch (fake gs + fake torch)
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
        Mesh=lambda **kw: _RecordingMorph("Mesh", **kw),
        Box=lambda **kw: _RecordingMorph("Box", **kw),
        Sphere=lambda **kw: _RecordingMorph("Sphere", **kw),
        Cylinder=lambda **kw: _RecordingMorph("Cylinder", **kw),
    )
    surfaces = types.SimpleNamespace(Rough=lambda **kw: ("Rough", kw))
    materials = types.SimpleNamespace(Rigid=lambda **kw: ("Rigid", kw))
    return types.SimpleNamespace(morphs=morphs, surfaces=surfaces, materials=materials)


@pytest.fixture()
def env_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    sys.modules.pop("npa.genesis.env_pick_place", None)
    module = importlib.import_module("npa.genesis.env_pick_place")
    yield module
    sys.modules.pop("npa.genesis.env_pick_place", None)


def _bare_env(env_module):
    env = env_module.FrankaPickPlaceEnv.__new__(env_module.FrankaPickPlaceEnv)
    env._scene = _FakeScene()
    env.scene_provenance = None
    return env


def test_build_dispatch_byo_mesh_calls_mesh_with_local_path(env_module) -> None:
    env = _bare_env(env_module)
    obj = sa.ObjectSpec(
        name="widget",
        asset_source=sa.ASSET_SOURCE_BYO_MESH,
        uri="s3://bucket/object.obj",
        local_path="/tmp/resolved/object.obj",
        scale=2.0,
        pos=(0.5, 0.0, 0.05),
    )
    entity = env._add_object_entity(_fake_gs(), obj)

    assert entity.morph.kind == "Mesh"
    assert entity.morph.kwargs["file"] == "/tmp/resolved/object.obj"
    assert entity.morph.kwargs["scale"] == 2.0
    assert obj.loaded is True


def test_build_dispatch_genesis_builtin_calls_mesh(env_module) -> None:
    env = _bare_env(env_module)
    obj = sa.ObjectSpec(
        name="duck",
        asset_source=sa.ASSET_SOURCE_GENESIS_BUILTIN,
        builtin_path="meshes/duck.obj",
        local_path="/opt/genesis/assets/meshes/duck.obj",
    )
    entity = env._add_object_entity(_fake_gs(), obj)
    assert entity.morph.kind == "Mesh"
    assert entity.morph.kwargs["file"].endswith("duck.obj")
    assert obj.loaded is True


def test_build_dispatch_primitive_box_calls_box(env_module) -> None:
    env = _bare_env(env_module)
    obj = sa.ObjectSpec(
        name="cube",
        asset_source=sa.ASSET_SOURCE_PRIMITIVE,
        primitive=sa.PRIMITIVE_BOX,
        size=(0.04, 0.04, 0.04),
        pos=(0.5, 0.0, 0.04),
    )
    entity = env._add_object_entity(_fake_gs(), obj)
    assert entity.morph.kind == "Box"
    assert entity.morph.kwargs["size"] == (0.04, 0.04, 0.04)
    assert obj.loaded is True


def test_build_dispatch_primitive_sphere_calls_sphere(env_module) -> None:
    env = _bare_env(env_module)
    obj = sa.ObjectSpec(
        name="ball",
        asset_source=sa.ASSET_SOURCE_PRIMITIVE,
        primitive=sa.PRIMITIVE_SPHERE,
        radius=0.03,
    )
    entity = env._add_object_entity(_fake_gs(), obj)
    assert entity.morph.kind == "Sphere"
    assert entity.morph.kwargs["radius"] == 0.03


def test_build_dispatch_mesh_without_local_path_raises(env_module) -> None:
    env = _bare_env(env_module)
    obj = sa.ObjectSpec(
        name="widget",
        asset_source=sa.ASSET_SOURCE_BYO_MESH,
        uri="s3://bucket/object.obj",
        local_path="",  # not resolved -> must fail loudly
    )
    with pytest.raises(sa.SceneSpecError):
        env._add_object_entity(_fake_gs(), obj)


def test_build_scene_objects_records_provenance_and_manipuland(env_module) -> None:
    env = _bare_env(env_module)
    spec = sa.SceneSpec(
        objects=[
            sa.ObjectSpec(
                name="widget",
                asset_source=sa.ASSET_SOURCE_BYO_MESH,
                role=sa.ROLE_MANIPULAND,
                uri="s3://bucket/object.obj",
                local_path="/tmp/object.obj",
                sha256="abc123",
            ),
            sa.ObjectSpec(
                name="table",
                asset_source=sa.ASSET_SOURCE_PRIMITIVE,
                role=sa.ROLE_STATIC,
                primitive=sa.PRIMITIVE_BOX,
            ),
        ]
    )
    manip = env._build_scene_objects(_fake_gs(), spec)

    assert manip.morph.kind == "Mesh"
    assert env.scene_provenance["asset_fallback_used"] is False
    objs = {o["name"]: o for o in env.scene_provenance["objects"]}
    assert objs["widget"]["loaded"] is True
    assert objs["widget"]["sha256"] == "abc123"
    assert objs["table"]["loaded"] is True


def test_apply_scene_spec_overrides_cube_and_target(env_module) -> None:
    env = env_module.FrankaPickPlaceEnv.__new__(env_module.FrankaPickPlaceEnv)
    env.cfg = env_module.EnvConfig()
    env._manip_quat = (1.0, 0.0, 0.0, 0.0)
    spec = sa.SceneSpec(
        objects=[
            sa.ObjectSpec(
                name="cube",
                asset_source=sa.ASSET_SOURCE_PRIMITIVE,
                primitive=sa.PRIMITIVE_BOX,
                size=(0.06, 0.06, 0.06),
                pos=(0.55, 0.01, 0.06),
            )
        ],
        goal_pos=(0.4, 0.25, 0.04),
        goal_threshold=0.07,
    )
    env._apply_scene_spec(spec)
    assert env.cfg.cube_init_pos == (0.55, 0.01, 0.06)
    assert env.cfg.cube_size == 0.06
    assert env.cfg.target_pos == (0.4, 0.25, 0.04)
    assert env.cfg.target_threshold == 0.07

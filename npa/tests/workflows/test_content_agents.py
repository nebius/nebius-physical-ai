from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from npa.workflows import content_agents as ca


_UNAUTHORED = object()


class _FakeAttr:
    def __init__(self, value: object = _UNAUTHORED, *, fallback: object = 0.0):
        self._authored = value is not _UNAUTHORED
        self._value = fallback if not self._authored else value

    def HasAuthoredValueOpinion(self) -> bool:
        return self._authored

    def Get(self) -> object:
        return self._value


class _FakeRelationship:
    def GetName(self) -> str:
        return "material:binding"

    def HasAuthoredTargets(self) -> bool:
        return True


class _FakePrim:
    def __init__(self, path: str, apis: set[type], attributes: dict[str, _FakeAttr]):
        self.path = path
        self.apis = apis
        self.attributes = attributes

    def GetPath(self) -> str:
        return self.path

    def HasAPI(self, api: type) -> bool:
        return api in self.apis

    def GetRelationships(self) -> list[_FakeRelationship]:
        return [_FakeRelationship()]


class _RigidBodyAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetRigidBodyEnabledAttr(self) -> _FakeAttr:
        return _FakeAttr(True)


class _CollisionAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetCollisionEnabledAttr(self) -> _FakeAttr:
        return _FakeAttr(True)


class _MassAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetMassAttr(self) -> _FakeAttr:
        return self.prim.attributes["mass"]

    def GetDensityAttr(self) -> _FakeAttr:
        return self.prim.attributes["density"]


class _MaterialAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetStaticFrictionAttr(self) -> _FakeAttr:
        return self.prim.attributes["static_friction"]

    def GetDynamicFrictionAttr(self) -> _FakeAttr:
        return self.prim.attributes["dynamic_friction"]

    def GetRestitutionAttr(self) -> _FakeAttr:
        return self.prim.attributes["restitution"]


_FAKE_USD_PHYSICS = SimpleNamespace(
    RigidBodyAPI=_RigidBodyAPI,
    CollisionAPI=_CollisionAPI,
    MassAPI=_MassAPI,
    MaterialAPI=_MaterialAPI,
)


def _fake_pxr(
    *,
    mass: object = _UNAUTHORED,
    density: object = _UNAUTHORED,
    static_friction: object = _UNAUTHORED,
    dynamic_friction: object = _UNAUTHORED,
    restitution: object = _UNAUTHORED,
) -> tuple[object, ...]:
    prims = [
        _FakePrim(
            "/World/RigidObject",
            {_RigidBodyAPI, _CollisionAPI, _MassAPI},
            {
                "mass": _FakeAttr(mass),
                "density": _FakeAttr(density),
            },
        ),
        _FakePrim(
            "/World/PhysicsMaterial",
            {_MaterialAPI},
            {
                "static_friction": _FakeAttr(static_friction),
                "dynamic_friction": _FakeAttr(dynamic_friction),
                "restitution": _FakeAttr(restitution),
            },
        ),
    ]

    class Stage:
        @staticmethod
        def Open(_path: str) -> object:
            return SimpleNamespace(Traverse=lambda: iter(prims))

    return (None, None, SimpleNamespace(Stage=Stage), None, _FAKE_USD_PHYSICS, None, None)


def _inspection(
    *,
    mass: float | None = None,
    density: float | None = None,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
) -> dict[str, object]:
    mass_property = "mass" if mass is not None else "density"
    mass_value = mass if mass is not None else density
    friction_property = (
        "static_friction" if static_friction is not None else "dynamic_friction"
    )
    friction_value = (
        static_friction if static_friction is not None else dynamic_friction
    )
    return {
        "checks": {
            "rigid_body": True,
            "collision": True,
            "mass_or_density": True,
            "physics_material": True,
            "friction": True,
            "visual_material_binding": True,
        },
        "mass_properties": [
            {
                "prim": "/World/RigidObject",
                "mass": mass,
                "density": density,
                "authored_property": mass_property,
                "authored_value": mass_value,
            }
        ],
        "physics_materials": [
            {
                "prim": "/World/PhysicsMaterial",
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "restitution": None,
                "authored_property": friction_property,
                "authored_value": friction_value,
            }
        ],
    }


def _package_scene_spec(
    monkeypatch: pytest.MonkeyPatch,
    inspection: dict[str, object],
) -> dict[str, object]:
    uploaded_json: dict[str, dict[str, object]] = {}

    def download(_uri: str, destination: Path) -> Path:
        destination.write_text("#usda 1.0\n", encoding="utf-8")
        return destination

    def upload(path: Path, uri: str) -> str:
        if path.suffix == ".json":
            uploaded_json[path.name] = json.loads(path.read_text(encoding="utf-8"))
        return uri

    class Stage:
        @staticmethod
        def Open(_path: str) -> object:
            return object()

    class UsdUtils:
        @staticmethod
        def CreateNewUsdzPackage(source: str, destination: str) -> bool:
            Path(destination).write_bytes(Path(source).read_bytes())
            return True

    monkeypatch.setattr(ca, "_download", download)
    monkeypatch.setattr(ca, "_upload", upload)
    monkeypatch.setattr(ca, "inspect_physics", lambda _path: inspection)
    monkeypatch.setattr(
        ca,
        "_pxr",
        lambda: (None, None, SimpleNamespace(Stage=Stage), None, None, None, UsdUtils),
    )
    ca.package_stage(run_uri="s3://bucket/content-agents/run")
    return uploaded_json["scene_spec.json"]


def test_upstream_failure_preserves_private_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[tuple[Path, str]] = []

    monkeypatch.setattr(
        ca.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["material-agent"], 17
        ),
    )
    monkeypatch.setattr(
        ca,
        "_upload",
        lambda path, uri: uploaded.append((path, uri)) or uri,
    )

    with pytest.raises(ca.ContentAgentsError, match="private stage log was preserved"):
        ca._run(
            ["material-agent", "run"],
            cwd=tmp_path,
            log_path=tmp_path / "material-agent.log",
            failure_log_uri="s3://private/run/material-agent.failed.log",
        )

    assert uploaded == [
        (tmp_path / "material-agent.log", "s3://private/run/material-agent.failed.log")
    ]


def test_upstream_failure_remains_primary_when_log_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ca.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["physics-agent"], 23
        ),
    )

    def fail_upload(_path: Path, _uri: str) -> str:
        raise RuntimeError("private storage unavailable")

    monkeypatch.setattr(ca, "_upload", fail_upload)

    with pytest.raises(
        ca.ContentAgentsError,
        match="physics-agent failed with exit code 23; the private stage log could not be preserved",
    ):
        ca._run(
            ["physics-agent", "run"],
            cwd=tmp_path,
            log_path=tmp_path / "physics-agent.log",
            failure_log_uri="s3://private/run/physics-agent.failed.log",
        )


ROOT = Path(__file__).resolve().parents[3]
SPEC = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "content-agents-rigid-object.yaml"
)


def test_upstream_selection_is_immutable_and_antioch_is_review_only() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", ca.CONTENT_AGENTS_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", ca.ANTIOCH_REVISION)
    assert ca.CONTENT_AGENTS_VERSION == "0.5.2"
    assert "NVIDIA-Omniverse/content-agents" in ca.CONTENT_AGENTS_REPOSITORY
    assert "antioch-robotics/antioch-content-agents" in ca.ANTIOCH_REPOSITORY


def test_material_config_uses_real_upstream_ovrtx_and_runtime_key(
    tmp_path: Path,
) -> None:
    config = ca.material_config(
        input_usd=tmp_path / "input.usda",
        output_usd=tmp_path / "output.usda",
        work_dir_name=".work",
        model=ca.DEFAULT_MODEL,
        base_url=ca.DEFAULT_BASE_URL,
    )

    assert config["steps"]["optimize_usd"] == {"enabled": False}
    assert config["steps"]["validate_input"]["on_failure"] == "block"
    assert config["steps"]["validate_output"]["on_failure"] == "block"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["render"]["backend"] == "ovrtx"
    vlm = config["steps"]["predict"]["vlm"]
    assert vlm == {
        "backend": "openai",
        "model": ca.DEFAULT_MODEL,
        "base_url": ca.DEFAULT_BASE_URL,
        "api_key_env": "${NEBIUS_TOKEN_FACTORY_KEY}",
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    assert "api_key" not in json.dumps(config).lower().replace("api_key_env", "")
    assert config["steps"]["apply"]["fail_on_unknown_material"] is True
    assert config["materials"]["path"] == "materials.yaml"


def test_physics_config_requires_real_vlm_and_authors_the_rigid_contract(
    tmp_path: Path,
) -> None:
    config = ca.physics_config(
        input_usd=tmp_path / "input.usda",
        output_usd=tmp_path / "output.usda",
        work_dir_name=".work",
        model=ca.DEFAULT_MODEL,
        base_url=ca.DEFAULT_BASE_URL,
    )

    assert config["steps"]["identify_asset"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "ovrtx"
    assert config["steps"]["predict"]["enabled"] is True
    assert config["steps"]["predict"]["allow_empty_predictions"] is False
    prompts = config["steps"]["build_dataset_prepare_dataset"]["prompts"]
    assert prompts["system"] == ca.PHYSICS_SYSTEM_PROMPT
    assert "strict JSON" in prompts["system"]
    assert "never use Markdown fences, comments" in prompts["system"]
    assert prompts["user"] == ca.PHYSICS_USER_PROMPT
    apply = config["steps"]["apply_physics"]
    assert apply["enabled"] is True
    assert apply["collision_approx"] == "convexHull"
    assert apply["mass_scale_policy"] == "warn"
    assert apply["allow_empty_predictions"] is False


@pytest.mark.parametrize(
    ("static_friction", "dynamic_friction", "selected_property", "selected_value"),
    [
        (0.61, _UNAUTHORED, "static_friction", 0.61),
        (_UNAUTHORED, 0.47, "dynamic_friction", 0.47),
        (0.61, 0.47, "static_friction", 0.61),
    ],
)
def test_inspect_physics_accepts_supported_authored_friction_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    static_friction: object,
    dynamic_friction: object,
    selected_property: str,
    selected_value: float,
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        ca,
        "_pxr",
        lambda: _fake_pxr(
            density=2700.0,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        ),
    )

    inspection = ca.inspect_physics(usd)

    material = inspection["physics_materials"][0]
    assert inspection["checks"]["friction"] is True
    assert material["authored_property"] == selected_property
    assert material["authored_value"] == selected_value


@pytest.mark.parametrize(
    ("mass", "density", "selected_property", "selected_value"),
    [
        (2.7, _UNAUTHORED, "mass", 2.7),
        (_UNAUTHORED, 2700.0, "density", 2700.0),
    ],
)
def test_inspect_physics_selects_authored_mass_or_density(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mass: object,
    density: object,
    selected_property: str,
    selected_value: float,
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        ca,
        "_pxr",
        lambda: _fake_pxr(
            mass=mass,
            density=density,
            static_friction=0.61,
        ),
    )

    inspection = ca.inspect_physics(usd)

    mass_property = inspection["mass_properties"][0]
    assert mass_property["authored_property"] == selected_property
    assert mass_property["authored_value"] == selected_value


def test_inspect_physics_rejects_material_api_without_authored_friction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        ca,
        "_pxr",
        lambda: _fake_pxr(density=2700.0, restitution=0.2),
    )

    with pytest.raises(ca.ContentAgentsError, match=r"missing: friction"):
        ca.inspect_physics(usd)


@pytest.mark.parametrize("friction", ["0.5", math.inf, 0.09, 2.01, True])
def test_inspect_physics_rejects_invalid_authored_friction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, friction: object
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        ca,
        "_pxr",
        lambda: _fake_pxr(density=2700.0, static_friction=friction),
    )

    with pytest.raises(ca.ContentAgentsError, match=r"static friction"):
        ca.inspect_physics(usd)


def test_package_stage_preserves_density_only_without_null_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_spec = _package_scene_spec(
        monkeypatch,
        _inspection(density=2700.0, dynamic_friction=0.47),
    )

    object_spec = scene_spec["objects"][0]
    assert object_spec["density"] == 2700.0
    assert "mass" not in object_spec
    assert object_spec["friction"] == 0.47
    assert object_spec["friction_source"] == "dynamic_friction"
    assert object_spec["dynamic_friction"] == 0.47


def test_package_stage_preserves_explicit_mass_and_static_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_spec = _package_scene_spec(
        monkeypatch,
        _inspection(mass=2.7, static_friction=0.61, dynamic_friction=0.47),
    )

    object_spec = scene_spec["objects"][0]
    assert object_spec["mass"] == 2.7
    assert "density" not in object_spec
    assert object_spec["friction"] == 0.61
    assert object_spec["friction_source"] == "static_friction"
    assert object_spec["static_friction"] == 0.61
    assert object_spec["dynamic_friction"] == 0.47


def test_generated_material_library_is_portable_and_asset_free(tmp_path: Path) -> None:
    manifest_path = ca._material_library(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    library = tmp_path / manifest["library_path"]

    assert library.is_file()
    assert manifest["entries"] == [
        {
            "name": "Aluminum",
            "description": (
                "Smooth silver-gray structural aluminum with metallic reflections "
                "and a moderately polished finish"
            ),
            "binding": "/World/Looks/Aluminum",
        }
    ]
    text = library.read_text(encoding="utf-8")
    assert "UsdPreviewSurface" in text
    assert "asset inputs:" not in text
    assert not list(tmp_path.rglob("*.mdl"))
    assert not list(tmp_path.rglob("*.png"))


def test_s3_contract_fails_closed() -> None:
    assert ca._s3_join("s3://bucket/run", "/physics/", "asset.usda") == (
        "s3://bucket/run/physics/asset.usda"
    )
    with pytest.raises(ca.ContentAgentsError, match="must use s3"):
        ca._s3_join("https://example.invalid/run", "asset.usda")


def test_live_stages_refuse_missing_model_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    with pytest.raises(ca.ContentAgentsError, match="NEBIUS_TOKEN_FACTORY_KEY"):
        ca.materials_stage(run_uri="s3://bucket/run", model="m", base_url="https://v1/")
    with pytest.raises(ca.ContentAgentsError, match="NEBIUS_TOKEN_FACTORY_KEY"):
        ca.physics_stage(run_uri="s3://bucket/run", model="m", base_url="https://v1/")


def test_workflow_routes_only_render_stages_to_rtx_and_never_b200() -> None:
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    resources = payload["resources"]
    assert resources["rtx-render"]["accelerators"] == "RTXPRO6000:1"
    assert all(
        "B200" not in str(resource.get("accelerators", ""))
        for resource in resources.values()
    )
    assert payload["states"]["acquire"]["resources"] == "cpu"
    assert payload["states"]["package"]["resources"] == "cpu"
    for name in ("materials", "physics", "validate"):
        assert payload["states"][name]["resources"] == "rtx-render"


def test_workflow_uses_every_real_content_agent_toolref_once() -> None:
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert [state["toolRef"] for state in payload["states"].values()] == [
        "workbench.content_agents.acquire",
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
        "workbench.content_agents.package",
    ]
    assert "tool://" not in payload["config"]["runtime_image"]
    assert "@sha256:" in payload["config"]["runtime_image"]


def test_capability_smoke_calls_real_upstream_authoring_and_validation() -> None:
    source = Path(ca.__file__).read_text(encoding="utf-8")
    assert "from physics_agent.functions.apply_physics import apply_physics" in source
    assert '"validation-agent",\n            "validate"' in source
    assert '"--render-backend",\n            "ovrtx"' in source
    assert '"render_valid"' in source
    assert '"physics_sane"' in source
    assert "echo" not in source.lower()

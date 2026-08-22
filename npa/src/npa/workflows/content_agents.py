"""NVIDIA Content Agents stages for rigid-ready USD object assets.

This module is a Tier-1 workflow adapter, not a second implementation of the
vendor pipeline.  The material and physics stages invoke the real
``material-agent`` and ``physics-agent`` console scripts from the immutable
NVIDIA checkout baked into the restricted image.  Validation invokes the real
``validation-agent`` profiles.  NPA owns only S3 hand-off, provenance, and the
narrow Isaac rigid-object adapter contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

import yaml

from npa.clients.storage import StorageClient


CONTENT_AGENTS_REPOSITORY = "https://github.com/NVIDIA-Omniverse/content-agents.git"
CONTENT_AGENTS_REVISION = "36dbf3f274f8e256637230a05a085853f65cc175"
CONTENT_AGENTS_VERSION = "0.5.2"
ANTIOCH_REPOSITORY = "https://github.com/antioch-robotics/antioch-content-agents.git"
ANTIOCH_REVISION = "9611fc17a899dee1a2fbf4837cce300019ad7210"
OVRTX_VERSION = "0.3.0.312915"

STAGE_REPORT_SCHEMA = "npa.content_agents.stage_report.v1"
RIGID_ASSET_SCHEMA = "npa.content_agents.isaac_rigid_asset.v1"
PROVENANCE_SCHEMA = "npa.content_agents.provenance.v1"
SCENE_SPEC_SCHEMA = "npa.sim2real.manip_scene_spec.v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
SUPPORTED_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
FRICTION_MIN = 0.1
FRICTION_MAX = 2.0
PHYSICS_SYSTEM_PROMPT = """\
Identify the rendered component and estimate simulation physics from its geometry,
appearance, material, and role in the complete asset. Use SI units. Compute mass as
density times the component volume times a realistic solid-fill factor, and check the
arithmetic before responding.

Return exactly two XML-style blocks and no other text:
<reasoning>one concise sentence</reasoning>
<answer>one JSON object</answer>

The answer object must contain asset_type, component_type, component_name, material,
physical_properties, confidence, and reasoning. physical_properties must contain
density, estimated_mass_kg, static_friction, dynamic_friction, and restitution. Every
physical property must be a finite JSON number. The answer block must be strict JSON:
never use Markdown fences, comments, equations, units in numeric fields, NaN, trailing
commas, or placeholders.
"""
PHYSICS_USER_PROMPT = (
    "Classify this component and provide the complete strict-JSON physics record."
)


class ContentAgentsError(RuntimeError):
    """Raised when a stage cannot prove its artifact contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContentAgentsError(f"{label} is missing or empty: {path.name}")
    return path


def _s3_join(root: str, *parts: str) -> str:
    root = str(root or "").strip().rstrip("/")
    if not root.startswith("s3://"):
        raise ContentAgentsError("run URI must use s3:// object storage")
    clean = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return "/".join((root, *clean))


def _storage() -> StorageClient:
    return StorageClient.from_environment()


def _download(uri: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("s3://"):
        _storage().download_path(uri, str(destination))
    elif uri.startswith("file://"):
        shutil.copy2(Path(uri.removeprefix("file://")), destination)
    else:
        shutil.copy2(Path(uri), destination)
    return _require_file(destination, "downloaded artifact")


def _upload(path: Path, uri: str) -> str:
    _require_file(path, "upload source")
    return _storage().upload_file(str(path), uri)


def _upload_tree(directory: Path, uri: str) -> str:
    if not directory.is_dir() or not any(
        path.is_file() for path in directory.rglob("*")
    ):
        raise ContentAgentsError(
            f"artifact directory is missing or empty: {directory.name}"
        )
    return _storage().upload_directory(str(directory), uri)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    failure_log_uri: str = "",
) -> None:
    """Run one real upstream entrypoint without leaking environment values."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        log_preserved = False
        if failure_log_uri:
            try:
                _upload(log_path, failure_log_uri)
                log_preserved = True
            except Exception:  # preserve the upstream failure as the root cause
                log_preserved = False
        raise ContentAgentsError(
            f"{Path(command[0]).name} failed with exit code {completed.returncode}; "
            + (
                "the private stage log was preserved"
                if log_preserved
                else "the private stage log could not be preserved"
            )
        )


def _pxr() -> tuple[Any, ...]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    except ImportError as exc:  # pragma: no cover - available in the runtime image
        raise ContentAgentsError(
            "OpenUSD is unavailable; run this stage in npa-content-agents"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


def _generate_fixture(path: Path) -> None:
    """Create a redistributable one-object USD used by the real live smoke."""

    Gf, _Sdf, Usd, UsdGeom, _UsdPhysics, UsdShade, _UsdUtils = _pxr()
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/World/RigidObject")
    cube = UsdGeom.Cube.Define(stage, "/World/RigidObject/AluminumCube")
    cube.CreateSizeAttr(0.1)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/InputNeutral")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/InputNeutral/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.35, 0.37, 0.40)
    )
    shader.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateOutput("surface", _Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    assert body.GetPrim().IsValid() and looks.GetPrim().IsValid()
    if not stage.GetRootLayer().Save():
        raise ContentAgentsError("failed to save generated USD fixture")


def _validate_open_stage(path: Path) -> dict[str, Any]:
    _require_file(path, "USD artifact")
    _Gf, _Sdf, Usd, UsdGeom, _UsdPhysics, _UsdShade, _UsdUtils = _pxr()
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ContentAgentsError(f"OpenUSD could not read {path.name}")
    default = stage.GetDefaultPrim()
    if not default or not default.IsValid() or not default.IsA(UsdGeom.Xformable):
        raise ContentAgentsError("USD must have an Xformable default prim")
    prim_count = sum(1 for _ in stage.Traverse())
    if prim_count < 2:
        raise ContentAgentsError("USD stage has no usable object geometry")
    return {
        "default_prim": str(default.GetPath()),
        "prim_count": prim_count,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    }


def _material_library(directory: Path) -> Path:
    """Write a tiny generated PreviewSurface library; no sample assets are baked."""

    usd = directory / "npa_materials.usda"
    usd.write_text(
        """#usda 1.0
(defaultPrim = \"World\")
def Scope \"World\" {
    def Scope \"Looks\" {
        def Material \"Aluminum\" {
            token outputs:surface.connect = </World/Looks/Aluminum/PreviewSurface.outputs:surface>
            def Shader \"PreviewSurface\" {
                uniform token info:id = \"UsdPreviewSurface\"
                color3f inputs:diffuseColor = (0.32, 0.35, 0.38)
                float inputs:metallic = 1
                float inputs:roughness = 0.28
                token outputs:surface
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    manifest = directory / "materials.yaml"
    _write_yaml(
        manifest,
        {
            "library_path": usd.name,
            "entries": [
                {
                    "name": "Aluminum",
                    "description": (
                        "Smooth silver-gray structural aluminum with metallic reflections "
                        "and a moderately polished finish"
                    ),
                    "binding": "/World/Looks/Aluminum",
                }
            ],
        },
    )
    return manifest


def material_config(
    *, input_usd: Path, output_usd: Path, work_dir_name: str, model: str, base_url: str
) -> dict[str, Any]:
    """Return the config consumed by the upstream Material Agent CLI."""

    return {
        "project": {
            "name": "npa_rigid_object_materials",
            "session_id": "npa-materials",
            "working_dir": work_dir_name,
            "description": "Assign the generated aluminum material to a rigid object.",
        },
        "input": {"usd_path": str(input_usd)},
        "output": {
            "usd_path": str(output_usd),
            "layer_only": False,
            "flatten_output": True,
            "material_profile": "auto",
        },
        "materials": {"path": "materials.yaml"},
        "steps": {
            # NVIDIA Content Agents v0.5.2 accepts warn/block/fix here.  Use
            # block for a fail-closed release pipeline; "fail" is not a valid
            # upstream mode and is rejected before the validator runs.
            "validate_input": {"enabled": True, "on_failure": "block"},
            "optimize_usd": {"enabled": False},
            "build_dataset_usd": {
                "enabled": True,
                "renderer": {
                    "backend": "ovrtx",
                    "image_width": 512,
                    "image_height": 512,
                    "camera_view_type": "corner",
                    "rendering_modes": {
                        "composition": {
                            "margin": 2.5,
                            "cameras": ["+x+y+z", "-x-y+z"],
                            "camera_focus_mode": "stage",
                            "skip_occluded_images": False,
                            "use_original_materials": True,
                        },
                        "prim_only_original": {
                            "margin": 1.3,
                            "cameras": ["+x+y+z", "-x-y+z"],
                            "camera_focus_mode": "prim",
                            "use_original_materials": True,
                        },
                    },
                    "should_highlight_prim": False,
                    "should_assign_random_colors": True,
                },
                "prim_filters": {
                    "types": ["UsdGeom.Cube", "UsdGeom.Mesh"],
                    "skip_instances": True,
                    "skip_prototypes": False,
                    "skip_invisible": True,
                },
                "extract_material_bindings": True,
                "extract_hierarchy": True,
                "extract_metadata": True,
                "skip_existing": False,
                "batch_size": 1,
                "num_workers": 1,
            },
            "build_dataset_prepare_dataset": {
                "enabled": True,
                "include_ground_truth": False,
                "include_prim_path_context": True,
                "include_geometric_context": True,
                "prompts": {
                    "vlm_system": (
                        "Select exactly one supplied material for every rendered object part. "
                        "The target is a reusable rigid aluminum object. Available materials: "
                        "{materials_list}. Respond as <reasoning>brief</reasoning>"
                        '<answer>{{"material": "material name"}}</answer>.'
                    ),
                    "vlm_user": "Assign the closest simulation-ready material.",
                    "vlm_image_prompts": {
                        "composition": "The complete object with the target part in context.",
                        "prim_only_original": "The isolated target part.",
                    },
                },
            },
            "predict": {
                "enabled": True,
                "prediction_batch_size": 1,
                "vlm": {
                    "backend": "openai",
                    "model": model,
                    "base_url": base_url,
                    "api_key_env": "${NEBIUS_TOKEN_FACTORY_KEY}",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "max_workers": 1,
                "allow_empty_predictions": False,
            },
            "validate_predictions": {"enabled": True},
            "harmonize_predictions": {"enabled": False},
            "apply": {
                "enabled": True,
                "layer_only": False,
                "flatten_output": True,
                "allow_empty_predictions": False,
                "fail_on_unknown_material": True,
            },
            "validate_output": {"enabled": True, "on_failure": "block"},
            "render": {
                "enabled": True,
                "backend": "ovrtx",
                "image_width": 512,
                "image_height": 512,
                "camera_corners": ["+x+y+z", "-x-y+z"],
                "camera_margin": 1.3,
                "background_color": [1.0, 1.0, 1.0],
                "flatten_before_render": False,
                "max_workers": 1,
            },
        },
        "advanced": {"keep_temp_files": True, "log_level": "INFO"},
    }


def physics_config(
    *, input_usd: Path, output_usd: Path, work_dir_name: str, model: str, base_url: str
) -> dict[str, Any]:
    """Return the config consumed by the upstream Physics Agent CLI."""

    vlm = {
        "backend": "openai",
        "model": model,
        "base_url": base_url,
        "api_key_env": "${NEBIUS_TOKEN_FACTORY_KEY}",
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    return {
        "project": {
            "name": "npa_rigid_object_physics",
            "session_id": "npa-physics",
            "working_dir": work_dir_name,
            "description": "Author rigid-body, collider, mass, and friction properties.",
        },
        "input": {"usd_path": str(input_usd)},
        "steps": {
            "optimize_usd": {"enabled": False},
            "identify_asset": {
                "enabled": True,
                "renderer": {
                    "backend": "ovrtx",
                    "image_width": 512,
                    "image_height": 512,
                    "cameras": ["+x+y+z", "-x-y+z"],
                },
                "vlm": vlm,
            },
            "build_dataset_usd": {
                "enabled": True,
                "renderer": {
                    "backend": "ovrtx",
                    "image_width": 512,
                    "image_height": 512,
                    "camera_view_type": "corner",
                    "rendering_modes": {
                        "composition": {
                            "margin": 2.5,
                            "cameras": ["+x+y+z", "-x-y+z"],
                            "camera_focus_mode": "stage",
                            "skip_occluded_images": False,
                            "use_original_materials": True,
                        },
                        "prim_only_original": {
                            "margin": 1.3,
                            "cameras": ["+x+y+z", "-x-y+z"],
                            "camera_focus_mode": "prim",
                            "use_original_materials": True,
                        },
                    },
                    "should_highlight_prim": False,
                    "should_assign_random_colors": True,
                },
                "prim_filters": {
                    "types": ["UsdGeom.Cube", "UsdGeom.Mesh"],
                    "skip_instances": True,
                    "skip_prototypes": False,
                },
                "extract_hierarchy": True,
                "extract_metadata": True,
                "skip_existing": False,
                "batch_size": 1,
                "num_workers": 1,
            },
            "build_dataset_prepare_dataset": {
                "enabled": True,
                "include_prim_path_context": True,
                "include_geometric_context": True,
                "prompts": {
                    "system": PHYSICS_SYSTEM_PROMPT,
                    "user": PHYSICS_USER_PROMPT,
                    "vlm_image_prompts": {
                        "composition": (
                            "The complete asset gives the target component's context."
                        ),
                        "prim_only_original": (
                            "The isolated target component retains its original appearance."
                        ),
                    },
                },
            },
            "predict": {
                "enabled": True,
                "vlm": vlm,
                "max_workers": 1,
                "output_key": "classification",
                "allow_empty_predictions": False,
            },
            "apply_physics": {
                "enabled": True,
                "output_usd_path": str(output_usd),
                "collision_approx": "convexHull",
                "mass_scale_policy": "warn",
                "allow_empty_predictions": False,
            },
        },
        "advanced": {"keep_temp_files": True, "log_level": "INFO"},
    }


def inspect_physics(path: Path) -> dict[str, Any]:
    """Fail closed unless a USD is a narrow, rigid-ready Isaac object asset."""

    _Gf, _Sdf, Usd, _UsdGeom, UsdPhysics, UsdShade, _UsdUtils = _pxr()
    stage = Usd.Stage.Open(str(_require_file(path, "physics USD")))
    if not stage:
        raise ContentAgentsError("OpenUSD could not reopen the physics USD")
    rigid: list[str] = []
    colliders: list[str] = []
    masses: list[dict[str, Any]] = []
    physics_material_prims: list[str] = []
    physics_materials: list[dict[str, Any]] = []
    visual_bindings: list[str] = []

    def authored_value(attribute: Any) -> Any:
        return attribute.Get() if attribute.HasAuthoredValueOpinion() else None

    def finite_number(
        value: Any,
        *,
        label: str,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ContentAgentsError(f"{label} must be an authored numeric value")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ContentAgentsError(f"{label} must be finite")
        if parsed < minimum or (maximum is not None and parsed > maximum):
            expected = (
                f"between {minimum} and {maximum}"
                if maximum is not None
                else f"at least {minimum}"
            )
            raise ContentAgentsError(f"{label} must be {expected}, got {parsed}")
        return parsed

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is not False:
                rigid.append(prim_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                colliders.append(prim_path)
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass = authored_value(api.GetMassAttr())
            density = authored_value(api.GetDensityAttr())
            parsed_mass = (
                finite_number(mass, label=f"{prim_path} mass", minimum=0.0)
                if mass is not None
                else None
            )
            parsed_density = (
                finite_number(
                    density, label=f"{prim_path} density", minimum=0.0
                )
                if density is not None
                else None
            )
            if parsed_mass == 0.0:
                raise ContentAgentsError(f"{prim_path} mass must be greater than zero")
            if parsed_density == 0.0:
                raise ContentAgentsError(
                    f"{prim_path} density must be greater than zero"
                )
            if parsed_mass is not None or parsed_density is not None:
                authored_property = "mass" if parsed_mass is not None else "density"
                authored_value_number = (
                    parsed_mass if parsed_mass is not None else parsed_density
                )
                masses.append(
                    {
                        "prim": prim_path,
                        "mass": parsed_mass,
                        "density": parsed_density,
                        "authored_property": authored_property,
                        "authored_value": authored_value_number,
                    }
                )
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            physics_material_prims.append(prim_path)
            api = UsdPhysics.MaterialAPI(prim)
            static_friction = authored_value(api.GetStaticFrictionAttr())
            dynamic_friction = authored_value(api.GetDynamicFrictionAttr())
            parsed_static = (
                finite_number(
                    static_friction,
                    label=f"{prim_path} static friction",
                    minimum=FRICTION_MIN,
                    maximum=FRICTION_MAX,
                )
                if static_friction is not None
                else None
            )
            parsed_dynamic = (
                finite_number(
                    dynamic_friction,
                    label=f"{prim_path} dynamic friction",
                    minimum=FRICTION_MIN,
                    maximum=FRICTION_MAX,
                )
                if dynamic_friction is not None
                else None
            )
            if parsed_static is not None or parsed_dynamic is not None:
                authored_property = (
                    "static_friction"
                    if parsed_static is not None
                    else "dynamic_friction"
                )
                authored_value_number = (
                    parsed_static if parsed_static is not None else parsed_dynamic
                )
                physics_materials.append(
                    {
                        "prim": prim_path,
                        "static_friction": parsed_static,
                        "dynamic_friction": parsed_dynamic,
                        "restitution": authored_value(api.GetRestitutionAttr()),
                        "authored_property": authored_property,
                        "authored_value": authored_value_number,
                    }
                )
        for rel in prim.GetRelationships():
            if (
                rel.GetName().startswith("material:binding")
                and rel.HasAuthoredTargets()
            ):
                visual_bindings.append(prim_path)
                break
    checks = {
        "rigid_body": bool(rigid),
        "collision": bool(colliders),
        "mass_or_density": bool(masses),
        "physics_material": bool(physics_material_prims),
        "friction": bool(physics_materials),
        "visual_material_binding": bool(visual_bindings),
    }
    if not all(checks.values()):
        missing = ", ".join(name for name, ok in checks.items() if not ok)
        raise ContentAgentsError(f"physics USD is not rigid-ready; missing: {missing}")
    return {
        "checks": checks,
        "rigid_body_prims": rigid,
        "collision_prims": colliders,
        "mass_properties": masses,
        "physics_material_prims": physics_material_prims,
        "physics_materials": physics_materials,
        "visual_material_binding_prims": sorted(set(visual_bindings)),
    }


def _stage_report(
    *, stage: str, input_path: Path, output_path: Path, details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": STAGE_REPORT_SCHEMA,
        "stage": stage,
        "status": "completed",
        "upstream": {
            "repository": CONTENT_AGENTS_REPOSITORY,
            "version": CONTENT_AGENTS_VERSION,
            "revision": CONTENT_AGENTS_REVISION,
        },
        "input": {"name": input_path.name, "sha256": _sha256(input_path)},
        "output": {
            "name": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "details": details,
    }


def acquire_stage(*, source_uri: str, run_uri: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-acquire-") as raw:
        work = Path(raw)
        source = work / "source.usda"
        if source_uri == "generated://rigid-cube":
            _generate_fixture(source)
        else:
            suffix = Path(source_uri.split("?", 1)[0]).suffix.lower()
            if suffix not in SUPPORTED_USD_SUFFIXES:
                raise ContentAgentsError(
                    "accepted inputs are self-contained USD/USDZ assets; conversion "
                    "dependencies are not part of the proven object workflow"
                )
            source = work / f"source{suffix}"
            _download(source_uri, source)
        metadata = _validate_open_stage(source)
        output = work / "asset.usda"
        _Gf, _Sdf, Usd, _UsdGeom, _UsdPhysics, _UsdShade, _UsdUtils = _pxr()
        stage = Usd.Stage.Open(str(source))
        if not stage or not stage.Flatten().Export(str(output)):
            raise ContentAgentsError("failed to export normalized USDA")
        _validate_open_stage(output)
        report = _stage_report(
            stage="acquire", input_path=source, output_path=output, details=metadata
        )
        report_path = work / "report.json"
        _write_json(report_path, report)
        _upload(output, _s3_join(run_uri, "acquire", output.name))
        _upload(report_path, _s3_join(run_uri, "acquire", report_path.name))
        return report


def materials_stage(*, run_uri: str, model: str, base_url: str) -> dict[str, Any]:
    if not os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip():
        raise ContentAgentsError(
            "NEBIUS_TOKEN_FACTORY_KEY is required for real VLM inference"
        )
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-materials-") as raw:
        work = Path(raw)
        input_usd = _download(
            _s3_join(run_uri, "acquire", "asset.usda"), work / "input.usda"
        )
        output_usd = work / "asset_material.usda"
        _material_library(work)
        config_path = work / "material-config.yaml"
        config = material_config(
            input_usd=input_usd,
            output_usd=output_usd,
            work_dir_name=".material-work",
            model=model,
            base_url=base_url,
        )
        _write_yaml(config_path, config)
        log_path = work / "material-agent.log"
        _run(
            [
                "material-agent",
                "run",
                str(config_path),
                "--clean",
                "--log-file",
                str(log_path),
            ],
            cwd=work,
            log_path=log_path,
            failure_log_uri=_s3_join(
                run_uri, "materials", "material-agent.failed.log"
            ),
        )
        _require_file(output_usd, "Material Agent output")
        metadata = _validate_open_stage(output_usd)
        rendered = [p for p in work.rglob("*.png") if p.stat().st_size > 0]
        if not rendered:
            raise ContentAgentsError("Material Agent produced no OVRTX render evidence")
        report = _stage_report(
            stage="materials",
            input_path=input_usd,
            output_path=output_usd,
            details={
                **metadata,
                "entrypoint": "material-agent run",
                "model": model,
                "renderer": "ovrtx",
                "render_count": len(rendered),
            },
        )
        report_path = work / "report.json"
        _write_json(report_path, report)
        evidence = work / "render-evidence"
        evidence.mkdir()
        for index, path in enumerate(rendered):
            shutil.copy2(path, evidence / f"material-{index:03d}.png")
        _upload(output_usd, _s3_join(run_uri, "materials", output_usd.name))
        _upload(report_path, _s3_join(run_uri, "materials", report_path.name))
        _upload(log_path, _s3_join(run_uri, "materials", "material-agent.log"))
        _upload_tree(evidence, _s3_join(run_uri, "renders", "materials"))
        return report


def physics_stage(*, run_uri: str, model: str, base_url: str) -> dict[str, Any]:
    if not os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip():
        raise ContentAgentsError(
            "NEBIUS_TOKEN_FACTORY_KEY is required for real VLM inference"
        )
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-physics-") as raw:
        work = Path(raw)
        input_usd = _download(
            _s3_join(run_uri, "materials", "asset_material.usda"), work / "input.usda"
        )
        output_usd = work / "asset_physics.usda"
        config_path = work / "physics-config.yaml"
        _write_yaml(
            config_path,
            physics_config(
                input_usd=input_usd,
                output_usd=output_usd,
                work_dir_name=".physics-work",
                model=model,
                base_url=base_url,
            ),
        )
        log_path = work / "physics-agent.log"
        _run(
            [
                "physics-agent",
                "run",
                str(config_path),
                "--clean",
                "--log-file",
                str(log_path),
            ],
            cwd=work,
            log_path=log_path,
            failure_log_uri=_s3_join(run_uri, "physics", "physics-agent.failed.log"),
        )
        _require_file(output_usd, "Physics Agent output")
        inspection = inspect_physics(output_usd)
        rendered = [p for p in work.rglob("*.png") if p.stat().st_size > 0]
        if not rendered:
            raise ContentAgentsError("Physics Agent produced no OVRTX render evidence")
        report = _stage_report(
            stage="physics",
            input_path=input_usd,
            output_path=output_usd,
            details={
                "entrypoint": "physics-agent run",
                "model": model,
                "renderer": "ovrtx",
                "render_count": len(rendered),
                "physics": inspection,
            },
        )
        report_path = work / "report.json"
        _write_json(report_path, report)
        evidence = work / "render-evidence"
        evidence.mkdir()
        for index, path in enumerate(rendered):
            shutil.copy2(path, evidence / f"physics-{index:03d}.png")
        _upload(output_usd, _s3_join(run_uri, "physics", output_usd.name))
        _upload(report_path, _s3_join(run_uri, "physics", report_path.name))
        _upload(log_path, _s3_join(run_uri, "physics", "physics-agent.log"))
        _upload_tree(evidence, _s3_join(run_uri, "renders", "physics"))
        return report


def validate_stage(*, run_uri: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-validate-") as raw:
        work = Path(raw)
        input_usd = _download(
            _s3_join(run_uri, "physics", "asset_physics.usda"),
            work / "asset_physics.usda",
        )
        validation_dir = work / "upstream-validation"
        log_path = work / "validation-agent.log"
        _run(
            [
                "validation-agent",
                "validate",
                str(input_usd),
                "--task",
                "Release-gate this asset as one renderable rigid Isaac object.",
                "--template",
                "render_valid",
                "--template",
                "physics_sane",
                "--render-backend",
                "ovrtx",
                "--render-view",
                "corner",
                "--image-width",
                "512",
                "--image-height",
                "512",
                "--output-dir",
                str(validation_dir),
                "--fail-on-warn",
                "--format",
                "json",
            ],
            cwd=work,
            log_path=log_path,
            failure_log_uri=_s3_join(
                run_uri, "validation", "validation-agent.failed.log"
            ),
        )
        upstream_result = validation_dir / "validation_result.json"
        _require_file(upstream_result, "Validation Agent result")
        validation_payload = json.loads(upstream_result.read_text(encoding="utf-8"))
        if validation_payload.get("verdict") != "pass":
            raise ContentAgentsError(
                f"Validation Agent verdict was {validation_payload.get('verdict')!r}"
            )
        inspection = inspect_physics(input_usd)
        rendered = [p for p in validation_dir.rglob("*.png") if p.stat().st_size > 0]
        if not rendered:
            raise ContentAgentsError(
                "Validation Agent produced no OVRTX render evidence"
            )
        report = _stage_report(
            stage="validate",
            input_path=input_usd,
            output_path=upstream_result,
            details={
                "entrypoint": "validation-agent validate",
                "profiles": ["render_valid", "physics_sane"],
                "verdict": "pass",
                "render_count": len(rendered),
                "physics": inspection,
            },
        )
        report_path = work / "report.json"
        _write_json(report_path, report)
        _upload_tree(validation_dir, _s3_join(run_uri, "validation", "upstream"))
        _upload(report_path, _s3_join(run_uri, "validation", report_path.name))
        _upload(log_path, _s3_join(run_uri, "validation", "validation-agent.log"))
        return report


def package_stage(*, run_uri: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-content-agents-package-") as raw:
        work = Path(raw)
        input_usd = _download(
            _s3_join(run_uri, "physics", "asset_physics.usda"),
            work / "asset_simready.usda",
        )
        inspection = inspect_physics(input_usd)
        usdz = work / "asset_simready.usdz"
        _Gf, _Sdf, Usd, _UsdGeom, _UsdPhysics, _UsdShade, UsdUtils = _pxr()
        if not UsdUtils.CreateNewUsdzPackage(str(input_usd), str(usdz)):
            raise ContentAgentsError("OpenUSD failed to create the self-contained USDZ")
        _require_file(usdz, "USDZ package")
        if not Usd.Stage.Open(str(usdz)):
            raise ContentAgentsError("OpenUSD could not reopen the packaged USDZ")
        inspect_physics(usdz)

        asset_uri = _s3_join(run_uri, "package", usdz.name)
        usd_uri = _s3_join(run_uri, "package", input_usd.name)
        scene_spec_uri = _s3_join(run_uri, "package", "scene_spec.json")
        first_mass = inspection["mass_properties"][0]
        first_material = inspection["physics_materials"][0]
        mass_property = first_mass.get("authored_property")
        mass_value = first_mass.get("authored_value")
        if mass_property not in {"mass", "density"} or mass_value is None:
            raise ContentAgentsError(
                "physics inspection did not select an authored mass or density"
            )
        friction_property = first_material.get("authored_property")
        friction_value = first_material.get("authored_value")
        if (
            friction_property not in {"static_friction", "dynamic_friction"}
            or friction_value is None
        ):
            raise ContentAgentsError(
                "physics inspection did not select an authored friction coefficient"
            )
        object_spec = {
            "name": "content_agents_rigid_object",
            "asset_source": "byo_mesh",
            "role": "manipuland",
            "uri": asset_uri,
            "scale": 1.0,
            "pos": [0.5, 0.0, 0.05],
            "color": [0.32, 0.35, 0.38],
            mass_property: mass_value,
            "friction": friction_value,
            "friction_source": friction_property,
            "fixed": False,
        }
        for coefficient in ("static_friction", "dynamic_friction"):
            value = first_material.get(coefficient)
            if value is not None:
                object_spec[coefficient] = value
        scene_spec = {
            "schema": SCENE_SPEC_SCHEMA,
            "source_uri": asset_uri,
            "goal_pos": [0.5, 0.3, 0.04],
            "goal_threshold": 0.05,
            "objects": [object_spec],
            "consumer_contract": {
                "accepted": "isaac_lab_rigid_object",
                "genesis_direct_usd": False,
                "arbitrary_scene": False,
                "articulated_robot": False,
            },
        }
        scene_path = work / "scene_spec.json"
        _write_json(scene_path, scene_spec)
        adapter = {
            "schema": RIGID_ASSET_SCHEMA,
            "status": "accepted",
            "consumer": "npa.sim2real.stage_02.isaac_rigid_object",
            "asset_uri": asset_uri,
            "source_layer_uri": usd_uri,
            "scene_spec_uri": scene_spec_uri,
            "asset_sha256": _sha256(usdz),
            "asset_size_bytes": usdz.stat().st_size,
            "physics": inspection,
            "limitations": [
                "one rigid object only",
                "Isaac USD consumer proven; arbitrary scenes and robots are not claimed",
                "Genesis has no direct USD handoff in this contract",
                "joint and texture generation stages are not enabled",
            ],
        }
        adapter_path = work / "sim_asset_manifest.json"
        _write_json(adapter_path, adapter)
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "source": {
                "repository": CONTENT_AGENTS_REPOSITORY,
                "revision": CONTENT_AGENTS_REVISION,
                "version": CONTENT_AGENTS_VERSION,
            },
            "antioch_evaluation": {
                "repository": ANTIOCH_REPOSITORY,
                "revision": ANTIOCH_REVISION,
                "ported_patches": [],
                "rationale": (
                    "v0.5.2 supersedes the fork baseline; its safe texture and endpoint "
                    "handling are upstream, while fork-specific internal model defaults "
                    "and obsolete material-copy code are intentionally excluded"
                ),
            },
            "runtime": {
                "redistribution": "restricted",
                "ovrtx_version": OVRTX_VERSION,
                "ovphysx": "not installed",
                "scene_optimizer_core": "not installed",
                "model_weights": "not baked; hosted Token Factory inference",
            },
            "artifacts": {
                "usda": {"uri": usd_uri, "sha256": _sha256(input_usd)},
                "usdz": {"uri": asset_uri, "sha256": _sha256(usdz)},
                "scene_spec": {"uri": scene_spec_uri, "sha256": _sha256(scene_path)},
            },
        }
        provenance_path = work / "provenance.json"
        _write_json(provenance_path, provenance)
        final = {
            "schema": "npa.content_agents.final.v1",
            "status": "completed",
            "asset_manifest_uri": _s3_join(run_uri, "package", adapter_path.name),
            "asset_uri": asset_uri,
            "scene_spec_uri": scene_spec_uri,
            "provenance_uri": _s3_join(run_uri, "reports", provenance_path.name),
            "validation_uri": _s3_join(
                run_uri, "validation", "upstream", "validation_result.json"
            ),
        }
        final_path = work / "final.json"
        _write_json(final_path, final)

        for path, uri in (
            (input_usd, usd_uri),
            (usdz, asset_uri),
            (scene_path, scene_spec_uri),
            (adapter_path, _s3_join(run_uri, "package", adapter_path.name)),
            (provenance_path, _s3_join(run_uri, "reports", provenance_path.name)),
            (final_path, _s3_join(run_uri, "reports", final_path.name)),
        ):
            _upload(path, uri)
        return final


def inspect_runtime() -> dict[str, Any]:
    """Prove the restricted image carries the selected source and isolated OVRTX."""

    import importlib.metadata

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "world-understanding",
            "material-agent",
            "physics-agent",
            "validation-agent",
        )
    }
    if set(versions.values()) != {CONTENT_AGENTS_VERSION}:
        raise ContentAgentsError(
            f"installed Content Agents packages differ from {CONTENT_AGENTS_VERSION}: "
            f"{versions}"
        )
    ovrtx_venv = Path(
        os.environ.get("WU_OVRTX_VENV_DIR", "/opt/content-agents/.ovrtx_venv")
    )
    _require_file(ovrtx_venv / "bin/python", "isolated OVRTX interpreter")
    ovrtx_probe = subprocess.run(
        [
            str(ovrtx_venv / "bin/python"),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('ovrtx'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ovrtx_probe.returncode or ovrtx_probe.stdout.strip() != OVRTX_VERSION:
        raise ContentAgentsError(
            "isolated OVRTX runtime differs from the reviewed lock"
        )
    try:
        importlib.metadata.version("ovphysx")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ContentAgentsError(
            "OvPhysX must not be installed in the accepted object workflow"
        )
    payload = {
        "schema": "npa.content_agents.runtime.v1",
        "status": "ready",
        "content_agents_revision": os.environ.get(
            "NPA_CONTENT_AGENTS_REVISION", CONTENT_AGENTS_REVISION
        ),
        "packages": versions,
        "ovrtx": {"version": ovrtx_probe.stdout.strip(), "isolated_venv": True},
        "scene_optimizer_core": False,
        "ovphysx": False,
    }
    if payload["content_agents_revision"] != CONTENT_AGENTS_REVISION:
        raise ContentAgentsError(
            "runtime source revision differs from the workflow contract"
        )
    return payload


def local_smoke(*, output_dir: Path) -> dict[str, Any]:
    """Exercise real upstream physics authoring plus OVRTX validation on one GPU."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "fixture.usda"
    _generate_fixture(source)
    predictions = output_dir / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "/World/RigidObject/AluminumCube",
                "classification": {
                    "asset_type": "rigid cube",
                    "component_type": "structural",
                    "component_name": "aluminum cube",
                    "material": "aluminum",
                    "physical_properties": {
                        "density": 2700.0,
                        "estimated_mass_kg": 2.7,
                        "static_friction": 0.61,
                        "dynamic_friction": 0.47,
                        "restitution": 0.2,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = output_dir / "fixture_physics.usda"
    from physics_agent.functions.apply_physics import apply_physics

    apply_physics(str(source), str(predictions), str(output), mass_scale_policy="warn")
    inspection = inspect_physics(output)
    validation_dir = output_dir / "validation"
    _run(
        [
            "validation-agent",
            "validate",
            str(output),
            "--task",
            "Validate this rigid object.",
            "--template",
            "render_valid",
            "--template",
            "physics_sane",
            "--render-backend",
            "ovrtx",
            "--render-view",
            "corner",
            "--output-dir",
            str(validation_dir),
            "--fail-on-warn",
            "--format",
            "json",
        ],
        cwd=output_dir,
        log_path=output_dir / "validation-agent.log",
    )
    result = json.loads(
        _require_file(
            validation_dir / "validation_result.json", "smoke validation result"
        ).read_text(encoding="utf-8")
    )
    if result.get("verdict") != "pass":
        raise ContentAgentsError("functional smoke validation did not pass")
    payload = {
        "schema": "npa.content_agents.functional_smoke.v1",
        "status": "passed",
        "physics": inspection,
        "validation_verdict": result["verdict"],
        "render_count": len(
            [p for p in validation_dir.rglob("*.png") if p.stat().st_size]
        ),
        "artifact": {"path": str(output), "sha256": _sha256(output)},
    }
    if payload["render_count"] < 1:
        raise ContentAgentsError("functional smoke produced no rendered image")
    _write_json(output_dir / "smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m npa.workflows.content_agents",
        description="Run real NVIDIA Content Agents stages for a rigid-ready USD object.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser(
        "acquire", help="Acquire or generate and normalize a source USD"
    )
    acquire.add_argument("--source-uri", required=True)
    acquire.add_argument("--run-uri", required=True)
    for name in ("materials", "physics"):
        stage = sub.add_parser(name, help=f"Run the real upstream {name} pipeline")
        stage.add_argument("--run-uri", required=True)
        stage.add_argument("--model", default=DEFAULT_MODEL)
        stage.add_argument("--base-url", default=DEFAULT_BASE_URL)
    validate = sub.add_parser(
        "validate", help="Run upstream render_valid and physics_sane profiles"
    )
    validate.add_argument("--run-uri", required=True)
    package = sub.add_parser(
        "package", help="Package USD/USDZ and the Isaac Stage-2 adapter"
    )
    package.add_argument("--run-uri", required=True)
    sub.add_parser("inspect-runtime", help="Inspect immutable source/runtime pins")
    smoke = sub.add_parser(
        "local-smoke", help="Run real physics authoring and OVRTX validation"
    )
    smoke.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "acquire":
        result = acquire_stage(source_uri=args.source_uri, run_uri=args.run_uri)
    elif args.command == "materials":
        result = materials_stage(
            run_uri=args.run_uri, model=args.model, base_url=args.base_url
        )
    elif args.command == "physics":
        result = physics_stage(
            run_uri=args.run_uri, model=args.model, base_url=args.base_url
        )
    elif args.command == "validate":
        result = validate_stage(run_uri=args.run_uri)
    elif args.command == "package":
        result = package_stage(run_uri=args.run_uri)
    elif args.command == "inspect-runtime":
        result = inspect_runtime()
    elif args.command == "local-smoke":
        result = local_smoke(output_dir=args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set
        raise ContentAgentsError(f"unsupported command: {args.command}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

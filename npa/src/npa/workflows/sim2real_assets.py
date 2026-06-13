"""Stage 2 sim assets: stock defaults and BYO SceneSpec / RobotSpec materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from npa.workflows.sim2real_loop import Sim2RealLoopConfig

STOCK_SCENE_SCHEMA = "npa.sim2real.stock_scene_spec.v1"
STOCK_ROBOT_SCHEMA = "npa.sim2real.stock_robot_spec.v1"
CONSUMED_SCENE_SCHEMA = "npa.sim2real.consumed_scene_spec.v1"
CONSUMED_ROBOT_SCHEMA = "npa.sim2real.consumed_robot_spec.v1"

DEFAULT_CAMERA_STOCK = {
    "workspace": {
        "placement": "stock_overhead",
        "resolution": [640, 480],
        "dtype": "uint8",
    },
    "wrist": {
        "placement": "stock_ee_mounted",
        "resolution": [640, 480],
        "dtype": "uint8",
    },
}


@dataclass(frozen=True)
class AssetsStageResult:
    """Artifacts produced by Stage 2."""

    scene_spec_uri: str
    robot_spec_uri: str
    consumed_scene_path: str
    consumed_robot_path: str
    stage_record: dict[str, Any]
    component: dict[str, Any]


class Sim2RealAssetsError(RuntimeError):
    """Raised when Stage 2 asset materialization fails."""


def run_assets_stage(config: Sim2RealLoopConfig, local_dir: Path) -> AssetsStageResult:
    """Materialize stock or BYO scene + robot specs for downstream envgen and eval."""

    from npa.genesis import robot_assets, scene_assets
    from npa.workflows.sim2real_loop import (
        SIM_BACKEND_ISAAC,
        _consume_stage_assets,
        _storage_client,
        _write_stage,
    )

    stage_dir = local_dir / "stage_02_assets"
    stage_dir.mkdir(parents=True, exist_ok=True)
    preset = (config.robot_preset or "franka").strip().lower()
    sim_backend = (config.sim_backend or SIM_BACKEND_ISAAC).strip().lower()

    if (config.scene_spec_uri or config.assets_uri).strip():
        consumed = _consume_stage_assets(config, local_dir)
        scene_doc = consumed["scene"].to_dict()
        scene_status = "consumed_byo"
        scene_name = "BYO mesh / SceneSpec"
        consumed_scene_path = consumed["consumed_spec_path"]
    else:
        if sim_backend == SIM_BACKEND_ISAAC:
            scene = scene_assets.default_isaac_stock_scene_spec()
        else:
            scene = scene_assets.default_scene_spec()
        scene_doc = scene.to_dict()
        scene_status = "stock_tabletop"
        scene_name = "Stock tabletop manipuland (table + object)"

    robot_spec_uri_input = (config.robot_spec_uri or "").strip()
    robot_source = (config.robot_source or "").strip().lower()
    if robot_spec_uri_input:
        client = _storage_client(config)
        spec_local = stage_dir / "robot-spec-input.json"
        client.download_path(robot_spec_uri_input, str(spec_local))
        robot_doc = json.loads(spec_local.read_text(encoding="utf-8"))
        robot_status = "consumed_byo"
        robot_name = "BYO robot spec"
    else:
        try:
            robot = robot_assets.robot_spec_from_preset(preset)
        except robot_assets.RobotSpecError:
            robot = robot_assets.default_franka_robot_spec()
        robot_doc = {
            "schema": robot_assets.ROBOT_SPEC_SCHEMA,
            "preset": preset,
            "robot_source": robot.robot_source,
            "name": robot.name,
            "ee_link": robot.ee_link,
            "n_arm_joints": robot.n_arm_joints,
            "n_gripper_joints": robot.n_gripper_joints,
            "isaac_robot_hint": robot.isaac_robot_hint,
            "robot_uri": config.robot_source if robot.is_byo() else "",
            "status": "stock_preset" if robot.is_stock_franka() else "preset_pending_urdf",
        }
        robot_status = "stock_franka" if robot.is_stock_franka() else "preset_pending_urdf"
        robot_name = f"Stock robot preset ({preset})"

    consumed_scene = {
        "schema": CONSUMED_SCENE_SCHEMA,
        "stage": 2,
        "name": scene_name,
        "status": scene_status,
        "sim_backend": sim_backend,
        "assets_uri": config.assets_uri,
        "scene_spec_uri": config.scene_spec_uri,
        "scene_spec": scene_doc,
        "cameras": DEFAULT_CAMERA_STOCK,
        "next_action": "CONTINUE",
    }
    consumed_robot = {
        "schema": CONSUMED_ROBOT_SCHEMA,
        "stage": 2,
        "name": robot_name,
        "status": robot_status,
        "robot_preset": preset,
        "robot_spec_uri": robot_spec_uri_input,
        "robot_source": robot_source,
        "robot_spec": robot_doc,
        "next_action": "CONTINUE",
    }

    scene_path = stage_dir / "consumed_scene_spec.json"
    robot_path = stage_dir / "consumed_robot_spec.json"
    _write_json(scene_path, consumed_scene)
    _write_json(robot_path, consumed_robot)

    scene_spec_uri = str(scene_path)
    robot_spec_uri = str(robot_path)
    if config.s3_bucket and config.s3_endpoint.strip():
        client = _storage_client(config)
        root = _artifact_root(config)
        scene_spec_uri = client.upload_file(
            str(scene_path), f"{root}/stage_02_assets/consumed_scene_spec.json"
        )
        robot_spec_uri = client.upload_file(
            str(robot_path), f"{root}/stage_02_assets/consumed_robot_spec.json"
        )

    stage_record = _write_stage(
        local_dir,
        2,
        "assets",
        {
            **consumed_scene,
            "robot_spec_uri": robot_spec_uri,
            "consumed_robot_spec": str(robot_path),
        },
        filename="assets_manifest.json",
    )
    component = {
        "name": "stage_02_assets",
        "tier": "WORKS",
        "evidence": (
            f"Materialized {scene_status} scene and {robot_status} robot specs "
            "with stock camera placements for envgen and held-out eval."
        ),
        "artifacts": {
            "scene_spec": scene_spec_uri,
            "robot_spec": robot_spec_uri,
        },
    }
    return AssetsStageResult(
        scene_spec_uri=scene_spec_uri,
        robot_spec_uri=robot_spec_uri,
        consumed_scene_path=str(scene_path),
        consumed_robot_path=str(robot_path),
        stage_record=stage_record,
        component=component,
    )


def build_envgen_scene_spec(
    config: Sim2RealLoopConfig,
    *,
    scene_spec_uri: str,
    robot_spec_uri: str,
    augmented_frames_uri: str,
) -> Any:
    """Build the envgen SceneSpec from Stage 2 outputs."""

    from npa.workflows.sim2real_envgen import SceneSpec, build_scene_spec
    from npa.workflows.sim2real_loop import SIM_BACKEND_ISAAC

    base = build_scene_spec(
        byo_mesh_uri=config.assets_uri,
        augmented_frames_uri=augmented_frames_uri,
        notes=(
            f"robot_preset={config.robot_preset or 'franka'}",
            f"sim_backend={config.sim_backend}",
        ),
    )
    return SceneSpec(
        schema=base.schema,
        simready_catalog=base.simready_catalog,
        byo_mesh_uri=base.byo_mesh_uri,
        augmented_frames_uri=base.augmented_frames_uri,
        camera_names=base.camera_names,
        physics_profile=(
            "isaac-lift-franka"
            if (config.sim_backend or "").lower() == "isaac"
            else base.physics_profile
        ),
        notes=base.notes,
        scene_spec_uri=scene_spec_uri,
        robot_spec_uri=robot_spec_uri,
        robot_preset=config.robot_preset or "franka",
        sim_backend=config.sim_backend or SIM_BACKEND_ISAAC,
        cameras=DEFAULT_CAMERA_STOCK,
    )


def _artifact_root(config: Sim2RealLoopConfig) -> str:
    prefix = (config.s3_prefix or "sim2real-b").strip("/")
    return f"s3://{config.s3_bucket}/{prefix}/{config.run_id}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Bind a calibrated reconstructed surface to the existing measured-state lift task."""

from __future__ import annotations

from pathlib import Path
import re

from npa.workflows.lerobot_transfer_data import file_sha256


def validate_binding(binding: dict, root: Path) -> None:
    """Require the exact portable scene bytes named by a sealed task recipe.

    Args:
        binding: Scene identity and explicit robot/object assumptions.
        root: Directory containing the recipe and self-contained scene.
    Returns:
        None.
    Raises:
        ValueError: Scene identity or its task assumptions are unsupported.
        OSError: The scene cannot be read.
    """
    fixed = {
        "schema": "npa.lyra-scene-binding.v1",
        "asset": "scene.usdc",
        "robot": "fixed-base Franka; operator-calibrated mounting pose",
        "object": "inserted 5 cm rigid cube; mass and friction assigned explicitly",
        "visual": "colored mesh fused from Lyra predicted depth and source RGB",
        "collision": "same reconstructed triangles; unseen regions remain empty",
    }
    if not isinstance(binding, dict) or set(binding) != {
        *fixed,
        "asset_sha256",
        "geometry_sha256",
    }:
        raise ValueError("Unsupported scene binding fields")
    if any(binding[key] != value for key, value in fixed.items()):
        raise ValueError(
            "Scene binding must state its robot, object and geometry assumptions"
        )
    for name in ("asset_sha256", "geometry_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", str(binding[name])):
            raise ValueError("Scene binding requires full SHA-256 identities")
    asset = root / binding["asset"]
    if asset.is_symlink() or file_sha256(asset) != binding["asset_sha256"]:
        raise ValueError("Imported scene differs from the sealed task asset")


def configure_imported_scene(config, binding: dict, root: Path) -> None:
    """Replace studio surfaces with the exact calibrated collision scene.

    Args:
        config: Native task configuration before scene construction.
        binding: Validated scene identity.
        root: Local materialized scene directory.
    Returns:
        None.
    Raises:
        ValueError: The scene does not match its sealed identity.
        ImportError: Native Isaac APIs are unavailable.
    """
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import TiledCameraCfg

    validate_binding(binding, root)
    config.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReconstructedScene",
        spawn=sim.UsdFileCfg(usd_path=str((root / binding["asset"]).resolve())),
    )
    config.scene.plane = None
    config.scene.npa_wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/NpaWristCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.04, 0.0, 0.025), rot=(0.0, 1.0, 0.0, 0.0), convention="opengl"
        ),
        data_types=["rgb"],
        width=320,
        height=240,
        update_period=0.0,
        spawn=sim.PinholeCameraCfg(
            focal_length=18, horizontal_aperture=24, clipping_range=(0.005, 20)
        ),
    )

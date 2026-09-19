"""Author reproducible industrial manipulands and fixtures as self-contained USD assets."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from npa.workflows.lerobot_transfer_data import file_sha256

ASSET_NAMES = ("spool", "hex_nut", "bottle")
ASSET_DESCRIPTIONS = {
    "spool": "orange spool with two wide flanges and a narrow waist",
    "hex_nut": "brass hexagonal nut with an open central hole",
    "bottle": "blue bottle-shaped part with a narrow neck",
}
_COLORS = {
    "spool": (0.9, 0.28, 0.04),
    "hex_nut": (0.65, 0.45, 0.12),
    "bottle": (0.05, 0.35, 0.8),
}
_LOWER_BOUNDS = {"spool": -0.026, "hex_nut": -0.018, "bottle": -0.038}
_TRAY_TOP = 0.012
_HEADER = '#usda 1.0\n(defaultPrim = "Asset"; metersPerUnit = 1; upAxis = "Z")\n'


def _cylinder(name: str, radius: float, height: float, z: float, color: tuple) -> str:
    return f'''def Cylinder "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double radius = {radius}
        double height = {height}
        uniform token axis = "Z"
        double3 xformOp:translate = (0, 0, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [{color}]
    }}'''


def _nut_sector(index: int) -> str:
    angles = [index * math.pi / 3, (index + 1) * math.pi / 3]
    ring = [
        (radius * math.cos(angle), radius * math.sin(angle))
        for radius, angle in (
            (0.027, angles[0]),
            (0.027, angles[1]),
            (0.012, angles[1]),
            (0.012, angles[0]),
        )
    ]
    points = [(round(x, 8), round(y, 8), z) for z in (-0.018, 0.018) for x, y in ring]
    return f"""def Mesh "Sector{index}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {{
        point3f[] points = {points}
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [3,2,1,0,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
        uniform token subdivisionScheme = "none"
        token physics:approximation = "convexHull"
        color3f[] primvars:displayColor = [{_COLORS["hex_nut"]}]
    }}"""


def _manipuland(name: str) -> str:
    if name == "hex_nut":
        shapes = [_nut_sector(index) for index in range(6)]
    else:
        sections = ((0.016, 0.036, 0.0), (0.027, 0.008, -0.022), (0.027, 0.008, 0.022))
        if name == "bottle":
            sections = (
                (0.022, 0.056, -0.01),
                (0.016, 0.012, 0.024),
                (0.011, 0.02, 0.04),
            )
        shapes = [
            _cylinder(f"Section{index}", *section, _COLORS[name])
            for index, section in enumerate(sections)
        ]
    body = "\n".join(shapes)
    return (
        _HEADER
        + f"""def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = 0.08
        {body}
    }}\n"""
    )


def _box(name: str, size: tuple, position: tuple, color: tuple) -> str:
    return f'''def Cube "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double size = 1
        double3 xformOp:scale = {size}
        double3 xformOp:translate = {position}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [{color}]
    }}'''


def _fixture() -> str:
    color = (0.15, 0.2, 0.24)
    pieces = [_box("TrayBase", (0.18, 0.18, 0.012), (0, 0, 0.006), color)]
    for index, (size, position) in enumerate(
        [
            ((0.18, 0.008, 0.04), (0, -0.086, 0.026)),
            ((0.18, 0.008, 0.04), (0, 0.086, 0.026)),
            ((0.008, 0.18, 0.04), (-0.086, 0, 0.026)),
            ((0.008, 0.18, 0.04), (0.086, 0, 0.026)),
        ]
    ):
        pieces.append(_box(f"TrayWall{index}", size, position, color))
    pieces.append(_cylinder("LocatingPeg", 0.008, 0.06, 0.042, (0.65, 0.67, 0.7)))
    return _HEADER + 'def Xform "Asset" {\n' + "\n".join(pieces) + "\n}\n"


def _distractor_positions(target: str) -> dict[str, list[float]]:
    return {
        name: [0.69 + index * 0.065, 0.38, _TRAY_TOP - _LOWER_BOUNDS[name]]
        for index, name in enumerate(
            candidate for candidate in ASSET_NAMES if candidate != target
        )
    }


def write_assets(output: Path, target: str) -> dict:
    """Write original collision-bearing parts and seal their exact bytes.

    Args:
        output: Prepared stage directory.
        target: Manipuland selected for this experiment.
    Returns:
        Portable asset manifest with geometry hashes and provenance.
    Raises:
        ValueError: The target is unsupported.
        OSError: An asset cannot be written.
    """
    if target not in ASSET_NAMES:
        raise ValueError(f"Unknown Franka asset: {target}")
    directory = output / "assets"
    directory.mkdir(parents=True)
    contents = {name: _manipuland(name) for name in ASSET_NAMES}
    contents["fixture"] = _fixture()
    for name, content in contents.items():
        (directory / f"{name}.usda").write_text(content)
    return {
        "schema": "npa.franka-rl.assets.v1",
        "target": target,
        "description": ASSET_DESCRIPTIONS[target],
        "source": "npa_original_procedural_usd",
        "license": "Apache-2.0",
        "units": "metres",
        "nominal_mass_kg": 0.08,
        "fixture_position_m": [0.72, 0.34, 0.0],
        "distractor_positions_m": _distractor_positions(target),
        "files": {
            f"assets/{name}.usda": file_sha256(directory / f"{name}.usda")
            for name in contents
        },
    }


def configure_assets(config, manifest: dict, root: Path) -> None:
    """Install the sealed target, collision-bearing fixture, and distinct distractors.

    Args:
        config: Isaac environment configuration to update.
        manifest: Prepared asset manifest.
        root: Materialized directory containing the sealed asset bytes.
    Returns:
        None.
    Raises:
        ValueError: Asset bytes differ from the sealed manifest.
        ImportError: Isaac is unavailable.
    """
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    for relative, digest in manifest["files"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError("Franka simulation asset bytes changed after preparation")
    target = manifest["target"]
    # Asset geometry must not discard the task's contact solver configuration.
    rigid_props = deepcopy(config.scene.object.spawn.rigid_props)
    config.scene.object.spawn = sim.UsdFileCfg(
        usd_path=str(root / f"assets/{target}.usda"),
        rigid_props=rigid_props,
        mass_props=sim.MassPropertiesCfg(mass=manifest["nominal_mass_kg"]),
    )
    config.scene.object.init_state.pos = (0.5, 0.0, 0.06)
    config.scene.npa_fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        spawn=sim.UsdFileCfg(usd_path=str(root / "assets/fixture.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(manifest["fixture_position_m"])
        ),
    )
    for index, (name, position) in enumerate(
        manifest["distractor_positions_m"].items()
    ):
        setattr(
            config.scene,
            f"npa_distractor_{index}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Distractor{index}",
                spawn=sim.UsdFileCfg(
                    usd_path=str(root / f"assets/{name}.usda"),
                    rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position)),
            ),
        )


def asset_physics_evidence(native) -> dict:
    """Verify object solver properties on the composed simulation prim.

    Args:
        native: Initialized Isaac environment after object spawning.
    Returns:
        Composed USD properties and requested task settings; not solver-internal readback.
    Raises:
        ValueError: The object has no unique rigid body or its properties differ.
    """
    from isaaclab.sim.utils.queries import find_first_matching_prim
    from pxr import Usd, UsdPhysics

    config = native.scene["object"].cfg.spawn.rigid_props
    root = find_first_matching_prim(native.scene["object"].cfg.prim_path)
    bodies = (
        [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if root
        else []
    )
    if len(bodies) != 1:
        raise ValueError("Manipuland must contain exactly one composed rigid body")
    fields = {
        "solver_position_iteration_count": "solverPositionIterationCount",
        "solver_velocity_iteration_count": "solverVelocityIterationCount",
        "max_depenetration_velocity": "maxDepenetrationVelocity",
        "max_linear_velocity": "maxLinearVelocity",
        "max_angular_velocity": "maxAngularVelocity",
    }
    values = {
        name: bodies[0].GetAttribute("physxRigidBody:" + attribute).Get()
        for name, attribute in fields.items()
    }
    requested = {name: getattr(config, name) for name in fields}
    if any(
        value is not None
        and (
            values[name] is None or not math.isclose(values[name], value, rel_tol=1e-6)
        )
        for name, value in requested.items()
    ):
        raise ValueError(
            "Composed manipuland solver properties differ from the preserved task settings"
        )
    return {
        "source": "composed USD schema attributes",
        "solver_internal_readback": False,
        "requested": requested,
        "composed": values,
    }

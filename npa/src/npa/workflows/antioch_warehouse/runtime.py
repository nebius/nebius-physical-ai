"""Execute one native Isaac batch without an Antioch session or notebook dependency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from ._scene import _WarehouseCell

VIEWS = {
    "aisle": ((8.5, -10.5, 4.5), (-1.1, 0.2, 1.5)),
    "overview": ((6.3, -7.4, 5.8), (-0.7, -0.25, 1.0)),
    "packing": ((4.0, -0.9, 2.9), (0.0, -1.8, 1.2)),
}
FIDELITY = {
    "cartons": "dynamic 2.5 kg PhysX bodies",
    "conveyor": "contact surface velocity, 0.45 m/s in world space",
    "gantry": "scripted kinematic Cartesian axes; no policy or robot IK",
    "grasp": "ideal fixed joint; no suction or seal model",
    "trigger": "carton position threshold; no optical sensor",
    "guards": "layout geometry; no safety certification",
    "scope": "one six-carton batch; no automatic episode reset",
}


def _measure(cell: _WarehouseCell) -> dict:
    from isaacsim.core.simulation_manager import SimulationManager

    prims = list(cell.stage.Traverse())
    return {
        "schema": "npa.antioch-warehouse.measurements.v1",
        "engine": "isaac-sim-physx",
        "authoring_origin": "local-antioch-warehouse",
        "sim_s": cell.t,
        "phase": cell.phase,
        "stage_prims": len(prims),
        "rigid_bodies": sum(p.HasAPI(cell.Physics.RigidBodyAPI) for p in prims),
        "physics_backend": str(SimulationManager.get_active_physics_engine()),
        "physics_dt": cell.world.get_physics_dt(),
        "completed_batches": cell.completed_batches,
        "total_placements": cell.total_placements,
        "max_attachment_error_m": cell.max_grip_error,
        "accumulation_speed_m_s": cell.hold_speed,
        "max_conveyor_displacement_m": cell.max_belt_dx,
        "settled": cell.measured_settled,
        "assets": cell.asset_records,
        "fidelity": FIDELITY,
    }


def _save(cell: _WarehouseCell, directory: Path, run_id: str) -> None:
    measured = _measure(cell)
    measured["run_id"] = run_id
    measured["python_version"] = sys.version.split()[0]
    measured["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    for name, data in (
        ("validation", measured),
        ("events", cell.events),
        ("trajectory", cell.samples),
    ):
        (directory / f"{name}.json").write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n"
        )


def _capture(application, directory: Path) -> None:
    import omni.kit.renderer_capture
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac runtime has no render viewport")
    viewport.fill_frame = False
    viewport.set_texture_resolution((1280, 720))
    for name, (eye, target) in VIEWS.items():
        set_camera_view(eye=eye, target=target, camera_prim_path=viewport.camera_path)
        for _ in range(30):
            application.update()
        capture = capture_viewport_to_file(viewport, str(directory / f"{name}.png"))
        completion = asyncio.ensure_future(capture.wait_for_result())
        while not completion.done():
            application.update()
        completion.result()
        # Viewport completion signals readback; the PNG writer has its own queue.
        writer = omni.kit.renderer_capture.acquire_renderer_capture_interface()
        writer.wait_async_capture()


def simulate(directory: Path, run_id: str):
    """Run a six-carton batch and return its still-open application after capture.

    Args:
        directory: Fresh artifact directory, including generated sign textures.
        run_id: Workflow identity recorded with the measurements.
    Returns:
        Isaac SimulationApp, to close only after evidence publication succeeds.
    Raises:
        RuntimeError: Kit stops or the simulation callback fails before completion.
        Exception: Native simulator, runtime asset, or capture failure.
    """
    from isaacsim import SimulationApp

    application = SimulationApp({"headless": True, "width": 1280, "height": 720})
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
    cell = _WarehouseCell(world, directory)
    cell.build()
    world.reset()
    _step_batch(application, cell)
    world.pause()
    _save(cell, directory, run_id)
    _capture(application, directory)
    return application


def _step_batch(application, cell: _WarehouseCell) -> None:
    errors = []

    def control(dt):
        try:
            cell.physics_tick(dt)
            cell.animate()
        except Exception as error:
            errors.append(error)

    cell.world.add_physics_callback("warehouse_control", control)
    while cell.completed_batches == 0:
        if not application.is_running():
            raise RuntimeError("Isaac stopped before completing the warehouse batch")
        cell.world.step(render=True)
        if errors:
            raise RuntimeError("Warehouse physics callback failed") from errors[0]
    cell.world.remove_physics_callback("warehouse_control")
    cell.record_sample()

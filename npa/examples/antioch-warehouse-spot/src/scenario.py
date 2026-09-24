"""Run a physics-driven Spot warehouse patrol with native RTX evidence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import antioch

from camera import _Camera, _Recording
from evidence import _publish
from scene import _build_scene, _PolicyStepper, _read_state, _Simulation


def _warm_up(world, stepper, camera, position) -> None:
    camera.aim(position, 0.0, "overview")
    for index in range(75):
        world.step()
        stepper.verify()
        if index in {0, 24, 74}:
            position, orientation, _ = _read_state(stepper.controller)
            print(
                "Warmup state",
                index,
                position.tolist(),
                orientation.tolist(),
                flush=True,
            )
    print("Warehouse policy and camera warmup complete", flush=True)


def _capture(world, controller, stepper, camera, recording, seconds, speed, view):
    samples = []
    stepper.set_velocity(speed)
    for index in range(round(seconds * 25)):
        position, _, _ = _read_state(controller)
        camera.aim(position, index / 25, view)
        world.step()
        stepper.verify()
        position, orientation, joints = _read_state(controller)
        sim_seconds = float(world.current_time)
        recording.append(camera.read(), sim_seconds)
        samples.append(
            {
                "time": sim_seconds,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "joints": joints.tolist(),
            }
        )
        if (index + 1) % 25 == 0:
            print(f"Recorded {index + 1} native camera steps", flush=True)
    return samples


def _run_patrol(run, seconds, speed, width, start_x, start_y, view):
    import numpy as np
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

    world = _Simulation()
    controller = _build_scene(start_x, start_y)
    print("Warehouse scene and Spot policy loaded", flush=True)
    camera = _Camera(world.stage, width)
    stepper = _PolicyStepper(controller)
    callback = SimulationManager.register_callback(
        stepper, IsaacEvents.POST_PHYSICS_STEP
    )
    world.start()
    print("Warehouse physics initialized", flush=True)
    _warm_up(world, stepper, camera, np.array([start_x, start_y, 0.8]))
    folder = Path(tempfile.mkdtemp(prefix="warehouse-spot-"))
    recording = _Recording(folder / "warehouse-spot.mp4", camera)
    try:
        samples = _capture(
            world, controller, stepper, camera, recording, seconds, speed, view
        )
    finally:
        recording.close()
        SimulationManager.deregister_callback(callback)
    _publish(run, folder, samples, recording, round(seconds * 25), stepper, speed > 0)


@antioch.scenario(
    name="warehouse_spot_patrol",
    tags=["warehouse", "quadruped", "film"],
    capture=False,
    config=antioch.SimulationConfig(
        physics_dt=1 / 500, render_dt=1 / 25, renderer_quality="quality", stream=False
    ),
)
def warehouse_spot_patrol(
    run: antioch.ScenarioRun,
    seconds: float = antioch.param(
        8.0, ge=0.04, description="Recorded simulation duration"
    ),
    speed: float = antioch.param(
        0.8, ge=0, le=1.5, description="Forward velocity command in m/s"
    ),
    width: int = antioch.param(
        1920, description="RGB video width; height is 9/16 of width"
    ),
    start_x: float = -3.0,
    start_y: float = 0.0,
    view: str = "tracking",
) -> None:
    """Capture Spot walking through a warehouse using its learned locomotion policy.

    Args:
        run: Antioch's result and artifact recorder.
        seconds: Recorded simulation duration after policy warmup.
        speed: Commanded forward velocity in metres per second.
        width: Native camera width, divisible by 32 for even 16:9 video.
        start_x: Initial warehouse x coordinate in metres.
        start_y: Initial warehouse y coordinate in metres.
        view: Tracking view or an overview for checking scene placement.
    Returns:
        None.
    Raises:
        ValueError: Camera width or view is unsupported.
        RuntimeError: A simulator, policy, or camera operation fails.
    """
    if width < 320 or width % 32 or view not in {"tracking", "overview"}:
        raise ValueError(
            "Use width >= 320 divisible by 32 and tracking or overview view"
        )
    _run_patrol(run, seconds, speed, width, start_x, start_y, view)

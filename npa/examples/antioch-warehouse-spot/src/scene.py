"""Load the native warehouse and advance Spot through its learned locomotion policy."""

from __future__ import annotations

import numpy as np


class _Simulation:
    """Use the native application lifecycle required by the experimental policy."""

    def __init__(self):
        import antioch
        import omni.timeline

        self.app = antioch.application()
        self.stage = antioch.stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()

    @property
    def current_time(self) -> float:
        from isaacsim.core.simulation_manager import SimulationManager

        return SimulationManager.get_simulation_time()

    def start(self) -> None:
        self.timeline.play()
        self.app.update()

    def step(self) -> None:
        before = self.current_time
        self.app.update()
        if not np.isclose(self.current_time - before, 1 / 25, atol=1e-5):
            raise RuntimeError("Native physics did not advance one 25 fps camera frame")


def _build_scene(start_x: float, start_y: float):
    import carb
    from isaacsim.core.experimental.utils.stage import define_prim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.storage.native import get_assets_root_path

    enable_extension("isaacsim.robot.policy.examples")
    from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac asset root is unavailable")
    warehouse = define_prim("/World/Warehouse", "Xform")
    warehouse.GetReferences().AddReference(
        root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    )
    define_prim("/World/PhysicsScene", "PhysicsScene")
    # PhysX otherwise clamps a 40 ms frame to its default 30 Hz minimum, dropping
    # substeps and causing the multi-tick camera to repeat producer timestamps.
    carb.settings.get_settings().set_float("/persistent/simulation/minFrameRate", 25.0)
    RenderingManager.set_dt(8.0 / 200.0)
    SimulationManager.set_physics_sim_device("cpu")
    SimulationManager.set_physics_dt(1.0 / 200.0)
    controller = SpotFlatTerrainPolicy(
        prim_path="/World/Spot",
        position=[start_x, start_y, 0.8],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    # The policy asset uses its own training timestep and decimation. The
    # standalone example's fixed 200 Hz changes the loaded policy's control rate.
    SimulationManager.set_physics_dt(controller._dt)
    return controller


def _read_state(controller) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position, orientation = controller.robot.get_world_poses()
    return (
        position.numpy()[0].astype(float),
        orientation.numpy()[0].astype(float),
        controller.robot.get_dof_positions().numpy()[0].astype(float),
    )


class _PolicyStepper:
    def __init__(self, controller):
        from isaacsim.core.deprecation_manager import import_module

        self.controller = controller
        self.torch = import_module("torch")
        self.torch.set_num_threads(1)
        self.command = self.torch.zeros(3, device="cpu")
        self.initialized = False
        self.steps = 0
        self.error = None

    def __call__(self, step_size: float, context: object) -> None:
        # Isaac callbacks can otherwise swallow a policy failure and keep rendering.
        if self.error is not None:
            return
        try:
            if not self.initialized:
                self.controller.initialize()
                self.initialized = True
                print(
                    "Spot policy initialized",
                    {
                        "dt": self.controller._dt,
                        "decimation": self.controller._decimation,
                    },
                    flush=True,
                )
            else:
                self.controller.forward(step_size, self.command)
                self.steps += 1
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            self.error = exc

    def set_velocity(self, forward: float, yaw_rate: float = 0.0) -> None:
        self.command = self.torch.tensor([forward, 0.0, yaw_rate], device="cpu")

    def verify(self) -> None:
        if self.error is not None:
            raise RuntimeError("Spot locomotion callback failed") from self.error

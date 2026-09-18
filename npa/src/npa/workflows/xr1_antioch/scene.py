"""Build and control two physical Franka arms with a shared XR1 Cartesian action contract."""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")


def _enable_robot_extensions() -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path("/isaac-sim/extsDeprecated")
    if not manager.set_extension_enabled_immediate("isaacsim.robot.manipulators.examples", True):
        raise RuntimeError("Isaac Sim's Franka motion-control extension could not load")


def _servo_class():
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController

    class RecordingServo(RMPFlowController):
        def forward(self, target_end_effector_position, target_end_effector_orientation=None):
            from isaacsim.core.utils.rotations import quat_to_rot_matrix

            self.target_position = np.asarray(target_end_effector_position).copy()
            if target_end_effector_orientation is not None:
                self.target_rotation = quat_to_rot_matrix(target_end_effector_orientation)
            return super().forward(target_end_effector_position, target_end_effector_orientation)

    return RecordingServo


class _Cell:
    def __init__(self, seed: int):
        from isaacsim.core.api import World
        import omni.usd
        from pxr import UsdLux

        _enable_robot_extensions()
        self.random = np.random.default_rng(seed)
        self.world = World(stage_units_in_meters=1, physics_dt=1 / 60, rendering_dt=1 / 60)
        self.world.scene.add_default_ground_plane()
        self.stage = omni.usd.get_context().get_stage()
        light = UsdLux.DomeLight.Define(self.stage, "/World/Light")
        light.CreateIntensityAttr(float(self.random.uniform(900, 1500)))
        self.robots, self.cubes, self.servos, self.experts, self.targets = [], [], [], [], []
        for index, side in enumerate(SIDES):
            self._add_arm(index, side)
        self.world.reset()
        self._initialize_controllers()

    def _add_arm(self, index: int, side: str) -> None:
        from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
        from isaacsim.robot.manipulators.examples.franka import Franka

        y = .4 if side == "left" else -.4
        robot = self.world.scene.add(Franka(
            f"/World/{side}", name=side, position=np.array([0., y, 0.]),
            end_effector_prim_name="panda_hand", gripper_open_position=np.array([.04, .04]),
        ))
        position = np.array([self.random.uniform(.40, .52), y + self.random.uniform(-.13, -.04), .025])
        target = np.array([self.random.uniform(.40, .52), y + self.random.uniform(.07, .18), .025])
        cube = self.world.scene.add(DynamicCuboid(
            f"/World/{side}_cube", name=f"{side}_cube", position=position,
            scale=np.array([.04, .04, .04]), mass=float(self.random.uniform(.04, .065)),
            color=np.array([.85, .12, .08]) if index == 0 else np.array([.08, .2, .9]),
        ))
        self.world.scene.add(VisualCuboid(
            f"/World/{side}_target", name=f"{side}_target", position=target * [1, 1, 0] + [0, 0, .001],
            scale=np.array([.10, .10, .002]), color=np.array([.15, .75, .25]),
        ))
        self.robots.append(robot)
        self.cubes.append(cube)
        self.targets.append(target)

    def _initialize_controllers(self) -> None:
        from isaacsim.robot.manipulators.controllers import PickPlaceController

        servo_type = _servo_class()
        for index, robot in enumerate(self.robots):
            robot.gripper.set_joint_positions(np.array([.04, .04]))
            robot.apply_action(robot.gripper.forward("open"))
            servo = servo_type(f"servo{index}", robot_articulation=robot, physics_dt=1 / 60)
            position, rotation = servo.rmp_flow.get_end_effector_pose(robot.get_joint_positions()[:7])
            servo.target_position, servo.target_rotation = position, rotation
            self.servos.append(servo)
            self.experts.append(PickPlaceController(
                name=f"expert{index}", cspace_controller=servo, gripper=robot.gripper,
                end_effector_initial_height=.30,
                events_dt=[.008, .005, 1., .1, .05, .05, .0025, 1., .008, .08],
            ))

    def expert_step(self) -> None:
        for robot, cube, expert, target in zip(self.robots, self.cubes, self.experts, self.targets):
            action = expert.forward(
                picking_position=cube.get_world_pose()[0], placing_position=target,
                current_joint_positions=robot.get_joint_positions(),
            )
            robot.apply_action(action)
        self.world.step(render=True)

    def state(self) -> dict:
        result = {"waist_pos": [0.]}
        for side, robot, servo in zip(SIDES, self.robots, self.servos):
            joints = robot.get_joint_positions()
            position, rotation = servo.rmp_flow.get_end_effector_pose(joints[:7])
            result.update({
                f"{side}_ee_pos": np.asarray(position).reshape(3).tolist(),
                f"{side}_ee_rotm": rotation.reshape(9).tolist(),
                f"{side}_arm_joint": joints[:7].tolist(),
                f"{side}_gripper_pos": [float(joints[7] + joints[8])],
            })
        return result

    def action_targets(self) -> dict:
        result = {"waist_pos": [0.], "base_vel": [0., 0., 0.]}
        for side, robot, servo in zip(SIDES, self.robots, self.servos):
            applied = robot.get_applied_action().joint_positions
            result.update({
                f"{side}_ee_pos": servo.target_position.reshape(3).tolist(),
                f"{side}_ee_rotm": servo.target_rotation.reshape(9).tolist(),
                f"{side}_gripper_pos": [float(applied[7] + applied[8])],
            })
        return result

    def policy_step(self, targets: dict) -> None:
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        from isaacsim.core.utils.types import ArticulationAction

        for side, robot, servo in zip(SIDES, self.robots, self.servos):
            position = np.asarray(targets[f"{side}_ee_pos"])
            rotation = np.asarray(targets[f"{side}_ee_rotm"]).reshape(3, 3)
            aperture = float(targets[f"{side}_gripper_pos"][0])
            action = servo.forward(position, rot_matrix_to_quat(rotation))
            robot.apply_action(action)
            robot.apply_action(ArticulationAction(
                joint_positions=np.full(2, np.clip(aperture / 2, 0, .04)),
                joint_indices=np.array([7, 8]),
            ))
        self.world.step(render=True)

    def measures(self) -> dict:
        return {
            "positions": [cube.get_world_pose()[0].tolist() for cube in self.cubes],
            "velocities": [cube.get_linear_velocity().tolist() for cube in self.cubes],
            "targets": [target.tolist() for target in self.targets],
            "gripper_apertures": [float(sum(robot.get_joint_positions()[7:9])) for robot in self.robots],
        }

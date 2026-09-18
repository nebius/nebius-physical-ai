"""Capture synchronized ego and wrist camera images from the simulated robot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _overview_camera(random):
    from isaacsim.sensors.camera import Camera
    from pxr import Gf

    eye = Gf.Vec3d(1.5 + random.uniform(-.05, .05), random.uniform(-.06, .06), 1.3)
    view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(.35, 0, .1), Gf.Vec3d(0, 0, 1)).GetInverse()
    quaternion = view.ExtractRotationQuat()
    camera = Camera("/World/ego_camera", position=np.array(eye), resolution=(384, 288))
    camera.initialize()
    camera.set_world_pose(
        position=np.array(eye), camera_axes="usd",
        orientation=np.array([quaternion.GetReal(), *quaternion.GetImaginary()]),
    )
    camera.set_focal_length(.018)
    camera.set_horizontal_aperture(.020955)
    camera.set_clipping_range(.005, 100.)
    return camera


def _wrist_camera(side: str):
    from isaacsim.sensors.camera import Camera

    camera = Camera(f"/World/{side}_wrist_camera", resolution=(384, 288))
    camera.initialize()
    camera.set_focal_length(.010)
    camera.set_horizontal_aperture(.020955)
    camera.set_clipping_range(.005, 100.)
    return camera


class _Cameras:
    def __init__(self, random, output: Path):
        import cv2

        self.cameras = {"ego": _overview_camera(random)}
        self.cameras.update({f"wrist_{side}": _wrist_camera(side) for side in ("left", "right")})
        self.output = output
        self.frames = 0
        for name, camera in self.cameras.items():
            fov = np.degrees(2 * np.arctan(camera.get_horizontal_aperture() / (2 * camera.get_focal_length())))
            if not 50 < fov < 110:
                raise ValueError(f"{name} field of view is inconsistent with the metric lens contract")
        self.writers = {
            name: cv2.VideoWriter(str(output / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (384, 288))
            for name in self.cameras
        }
        if not all(writer.isOpened() for writer in self.writers.values()):
            raise RuntimeError("Could not initialize all three synchronized video encoders")

    def follow_wrists(self, robots) -> None:
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        from pxr import Gf

        for side, robot in zip(("left", "right"), robots):
            position, quaternion = robot.end_effector.get_world_pose()
            rotation = quat_to_rot_matrix(quaternion)
            eye = np.asarray(position) + rotation @ np.array([.10, -.06, .045])
            target = np.asarray(position) + rotation @ np.array([-.35, 0., .65])
            up = rotation @ np.array([1., 0., 0.])
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
            orientation = view.ExtractRotationQuat()
            self.cameras[f"wrist_{side}"].set_world_pose(
                position=eye, orientation=np.array([orientation.GetReal(), *orientation.GetImaginary()]),
                camera_axes="usd",
            )

    def capture(self) -> dict:
        import cv2

        images = {}
        for name, camera in self.cameras.items():
            pixels = np.asarray(camera.get_rgba())
            if pixels.shape != (288, 384, 4) or pixels.dtype != np.uint8:
                raise RuntimeError(f"Missing uint8 RGB camera observation: {name}")
            images[name] = pixels[:, :, :3].copy()
            if self.frames == 0 and np.std(images[name]) < 5:
                raise ValueError(f"Camera {name} has no usable visual observation at reset")
            self.writers[name].write(cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR))
            if self.frames == 0:
                cv2.imwrite(str(self.output / f"{name}-first.png"), cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR))
        self.frames += 1
        return images

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()

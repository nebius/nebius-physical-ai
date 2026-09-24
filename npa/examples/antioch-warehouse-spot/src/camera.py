"""Record native RTX frames with producer timestamps and a tracking camera."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np


def _look_at(stage, path: str, eye, target) -> None:
    from pxr import Gf, UsdGeom

    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = Gf.Matrix4d(1.0)
    for index, vector in enumerate([right, up, -forward]):
        matrix.SetRow3(index, Gf.Vec3d(*vector))
    matrix.SetTranslateOnly(Gf.Vec3d(*eye))
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(matrix)


class _Camera:
    def __init__(self, stage, width: int):
        import omni.replicator.core as rep
        import warp as wp
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from pxr import UsdGeom

        self.path = "/World/FilmCamera"
        self.stage = stage
        self.width = width
        self.height = width * 9 // 16
        self.authoring = RtxCamera(self.path, tick_rate=25)
        self.sensor = CameraSensor(
            self.authoring, resolution=(self.height, width), annotators=["rgb"]
        )
        self.buffer = wp.empty((self.height, width, 3), dtype=wp.uint8, device="cpu")
        self.clock = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
        self.clock.attach([str(self.sensor.render_product.GetPrim().GetPath())])
        camera = UsdGeom.Camera(stage.GetPrimAtPath(self.path))
        camera.CreateFocalLengthAttr().Set(0.30)
        camera.CreateHorizontalApertureAttr().Set(0.36)
        camera.CreateVerticalApertureAttr().Set(0.36 * 9 / 16)
        camera.CreateClippingRangeAttr().Set((0.05, 2000.0))

    def aim(self, position: np.ndarray, seconds: float, view: str) -> None:
        if view == "overview":
            eye = position + np.array([-5.0, -6.0, 4.0])
            target = position + np.array([1.0, 0.0, 0.0])
        else:
            angle = -2.2 + 0.045 * seconds
            eye = position + np.array([3.6 * np.cos(angle), 3.6 * np.sin(angle), 1.6])
            target = position + np.array([0.35, 0.0, 0.1])
        _look_at(self.stage, self.path, eye, target)

    def read(self):
        data, _ = self.sensor.get_data("rgb", out=self.buffer)
        clock = self.clock.get_data()
        if data is None or not isinstance(clock, dict):
            return None
        numerator = clock.get("referenceTimeNumerator")
        denominator = clock.get("referenceTimeDenominator")
        if numerator is None or not denominator:
            return None
        pixels = np.ascontiguousarray(data.numpy()).copy()
        return pixels, Fraction(int(numerator), int(denominator))


class _Recording:
    def __init__(self, path: Path, camera: _Camera):
        import imageio_ffmpeg

        self.writer = imageio_ffmpeg.write_frames(
            str(path),
            (camera.width, camera.height),
            fps=25,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "17", "-preset", "fast", "-movflags", "+faststart"],
        )
        self.writer.send(None)
        self.frames = 0
        self.prior_clock = None
        self.timestamps = []
        self.pixel_variances = []

    def append(self, sample, sim_seconds: float) -> bool:
        if sample is None:
            return False
        pixels, clock = sample
        if self.prior_clock is not None and clock <= self.prior_clock:
            return False
        self.writer.send(pixels)
        self.prior_clock = clock
        self.frames += 1
        self.timestamps.append([sim_seconds, float(clock)])
        self.pixel_variances.append(float(pixels.std()))
        return True

    def close(self) -> None:
        self.writer.close()

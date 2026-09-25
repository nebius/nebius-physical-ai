"""Capture real Isaac renderer frames with measured goal and contact annotations."""

from contextlib import contextmanager
import shutil
import subprocess

import numpy as np

from npa.workflows.navigation.artifacts import write_json
from npa.workflows.navigation.render_evidence import (
    capture_settings,
    frozen_physics,
    renderer_evidence,
)


@contextmanager
def recording(env, recipe, output):
    """Record the focal robot in the public reference's actual held-out rollout.

    Args:
        env: Real native Isaac environment.
        recipe: Sealed experiment recipe.
        output: Evaluation artifact directory.
    Returns:
        Frame callback; external BYOF tasks retain their own visualization path.
    Raises:
        RuntimeError: The native renderer returns invalid or missing pixels.
    """
    if recipe.adapter_module != "npa.workflows.navigation.reference":
        yield lambda state, step: None
        return
    with capture_settings():
        capture = _Capture(env, recipe, output)
        try:
            with frozen_physics(capture.env):
                capture.setup()
            yield capture.frame
        finally:
            capture.close()


class _Capture:
    def __init__(self, env, recipe, output):
        self.env, self.recipe, self.output = env.unwrapped, recipe, output
        self.frames = output / "rendered-rollout"
        self.frames.mkdir()
        self.rows = []
        self.hidden = []
        self.annotators = {}
        self.product = None

    def setup(self):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        for path in self.env.scene.env_prim_paths[1:]:
            imageable = UsdGeom.Imageable(self.env.sim.stage.GetPrimAtPath(path))
            attribute = imageable.GetVisibilityAttr()
            self.hidden.append((attribute, attribute.Get()))
            imageable.MakeInvisible()
        self.camera = UsdGeom.Camera.Define(self.env.sim.stage, "/World/NpaProofCamera")
        self.camera.CreateFocalLengthAttr(18.0)
        self.camera.CreateClippingRangeAttr((0.1, 100.0))
        self.transform = UsdGeom.Xformable(self.camera).AddTransformOp()
        self.camera.CreateHorizontalApertureAttr(20.955)
        self.camera.CreateVerticalApertureAttr(15.71625)
        self.camera.CreateHorizontalApertureOffsetAttr(0.0)
        self.camera.CreateVerticalApertureOffsetAttr(0.0)
        self.product = rep.create.render_product(str(self.camera.GetPath()), (640, 480))
        for name in ("rgb", "CameraParams", "ReferenceTime"):
            annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            self.annotators[name] = annotator
            annotator.attach([self.product])

    def frame(self, state, step):
        import omni.replicator.core as rep
        from pxr import Gf

        position = state["position_m"][0]
        target = Gf.Vec3d(*position)
        eye = target + Gf.Vec3d(-3.0, -3.0, 3.0)
        self.transform.Set(
            Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        with frozen_physics(self.env) as native:
            # PhysX's manual step does not flush its current transforms to Fabric.
            self.env.sim.forward()
            for _ in range(2):
                rep.orchestrator.step(
                    delta_time=0.0, pause_timeline=False, wait_for_render=True
                )
            evidence = renderer_evidence(
                self.annotators, self.camera, self.transform.Get(), native
            )
            pixels = np.asarray(self.annotators["rgb"].get_data()).copy()
            _validate_pixels(pixels)
        row = self._row(state, step)
        row["render_evidence"] = evidence
        self.rows.append(row)
        self._write_frame(pixels[..., :3], row)

    def _row(self, state, step):
        position = state["position_m"][0]
        return {
            "step": step,
            "simulation_seconds": step * self.env.step_dt,
            "position_m": position.tolist(),
            "goal_m": state["goal_m"][0].tolist(),
            "goal_distance_m": float(np.linalg.norm(position[:2] - state["goal_m"][0])),
            "obstacle_contact_n": float(state["obstacle_contact"][0]),
            "peer_contact_n": float(state["peer_contact"][0]),
            "physical_failure": bool(state["physical_failure"][0]),
            "upright_cosine": float(state["upright_cosine"][0]),
            "ground_clearance_m": float(state["ground_clearance_m"][0]),
        }

    def _write_frame(self, pixels, row):
        from PIL import Image, ImageDraw

        frame = Image.fromarray(pixels.copy())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 640, 44), fill=(0, 0, 0))
        label = f"Real Isaac rollout | t={row['simulation_seconds']:.1f}s | goal distance={row['goal_distance_m']:.2f} m"
        draw.text((8, 6), label, fill=(255, 255, 255))
        draw.text(
            (8, 24),
            f"Obstacle={row['obstacle_contact_n']:.2f} N | Peer={row['peer_contact_n']:.2f} N | Failed={row['physical_failure']} | population={self.env.num_envs}",
            fill=(255, 255, 255),
        )
        frame.save(self.frames / f"{row['step']:06d}.png")

    def close(self):
        for annotator in self.annotators.values():
            annotator.detach([self.product])
        if self.product is not None:
            self.product.destroy()
        for attribute, value in self.hidden:
            attribute.Set(value)
        write_json(
            self.frames / "frames.json",
            {
                "renderer": "isaac-replicator-rgb",
                "robot_index": 0,
                "peers_hidden_for_visualization_only": True,
                "render_clock": "native_physx_fabric",
                "native_time_origin_seconds": (
                    self.rows[0]["render_evidence"]["native_clocks"]["physics_seconds"]
                    if self.rows
                    else None
                ),
                "frames": self.rows,
            },
        )
        self._encode_video()

    def _encode_video(self):
        encoder = shutil.which("ffmpeg")
        if encoder and self.rows:
            subprocess.run(
                [
                    encoder,
                    "-y",
                    "-framerate",
                    str(1 / self.env.step_dt),
                    "-i",
                    str(self.frames / "%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(self.output / "rollout.mp4"),
                ],
                check=True,
                capture_output=True,
            )


def _validate_pixels(pixels):
    if (
        pixels.shape != (480, 640, 4)
        or pixels.dtype != np.uint8
        or not pixels[..., :3].any()
    ):
        raise RuntimeError("native rollout renderer returned an invalid RGB frame")

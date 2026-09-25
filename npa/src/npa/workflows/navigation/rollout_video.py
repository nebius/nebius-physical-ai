"""Capture real Isaac renderer frames with measured goal and contact annotations."""

from contextlib import contextmanager
import shutil
import subprocess

import numpy as np

from npa.workflows.navigation.artifacts import write_json


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
    capture = _Capture(env, recipe, output)
    try:
        yield capture.frame
    finally:
        capture.close()


class _Capture:
    def __init__(self, env, recipe, output):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.env, self.recipe, self.output = env.unwrapped, recipe, output
        self.frames = output / "rendered-rollout"
        self.frames.mkdir()
        self.rows = []
        self.hidden = []
        for path in self.env.scene.env_prim_paths[1:]:
            imageable = UsdGeom.Imageable(self.env.sim.stage.GetPrimAtPath(path))
            attribute = imageable.GetVisibilityAttr()
            self.hidden.append((attribute, attribute.Get()))
            imageable.MakeInvisible()
        self.camera = UsdGeom.Camera.Define(self.env.sim.stage, "/World/NpaProofCamera")
        self.camera.CreateFocalLengthAttr(18.0)
        self.camera.CreateClippingRangeAttr((0.1, 100.0))
        self.transform = UsdGeom.Xformable(self.camera).AddTransformOp()
        rep.orchestrator.set_capture_on_play(False)
        self.product = rep.create.render_product(str(self.camera.GetPath()), (640, 480))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.product])

    def frame(self, state, step):
        import omni.replicator.core as rep
        from pxr import Gf

        position = state["position_m"][0]
        target = Gf.Vec3d(*position)
        eye = target + Gf.Vec3d(-3.0, -3.0, 3.0)
        self.transform.Set(
            Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        before = self.env.scene["robot"].data.root_pos_w.torch.clone()
        rep.orchestrator.step(
            delta_time=0.0, pause_timeline=False, wait_for_render=True
        )
        if not before.equal(self.env.scene["robot"].data.root_pos_w.torch):
            raise RuntimeError("recording unexpectedly advanced native robot physics")
        pixels = np.asarray(self.annotator.get_data())
        if (
            pixels.shape != (480, 640, 4)
            or pixels.dtype != np.uint8
            or not pixels[..., :3].any()
        ):
            raise RuntimeError("native rollout renderer returned an invalid RGB frame")
        row = {
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
        self.rows.append(row)
        self._write_frame(pixels[..., :3], row)

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
        self.annotator.detach([self.product])
        self.product.destroy()
        for attribute, value in self.hidden:
            attribute.Set(value)
        write_json(
            self.frames / "frames.json",
            {
                "renderer": "isaac-replicator-rgb",
                "robot_index": 0,
                "peers_hidden_for_visualization_only": True,
                "frames": self.rows,
            },
        )
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

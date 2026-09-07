"""Qualify the real MuJoCo checkpoint adapter inside its CUDA image.

Run this file with the image interpreter and explicit live inputs. It uses only
the standard library at collection time so ordinary CPU test runs can skip it.
The rollout is the production checkpoint-statistics adapter, not learned SONIC
policy inference or a locomotion benchmark. Rendered geometry is reported as-is.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shlex
import tempfile
import unittest
from unittest.mock import patch


class _UnsafeCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # The marker is confined to this test's private temporary directory.
        return os.system, (f"touch {shlex.quote(str(self.marker))}",)


@unittest.skipUnless(
    os.environ.get("NPA_INTEGRATION_E2E") == "1"
    and os.environ.get("NPA_SONIC_E2E_CHECKPOINT")
    and os.environ.get("NPA_SONIC_E2E_OUTPUT"),
    "requires an explicitly configured real checkpoint and CUDA image",
)
class SonicMuJoCoSecurityLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path("/opt/npa/docker/workbench/sonic/mujoco_eval.py")
        spec = importlib.util.spec_from_file_location("sonic_mujoco_live", source)
        if spec is None or spec.loader is None:
            raise RuntimeError("production checkpoint adapter is unavailable")
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)
        cls.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        cls.output = Path(os.environ["NPA_SONIC_E2E_OUTPUT"])
        cls.output.mkdir(parents=True, exist_ok=True)

    def test_unsafe_checkpoint_is_rejected_before_side_effect(self):
        import torch

        with tempfile.TemporaryDirectory(prefix="npa-checkpoint-regression-") as tmp:
            marker = Path(tmp) / "unsafe-load-marker"
            checkpoint = Path(tmp) / "untrusted.pt"
            torch.save({"policy_state_dict": _UnsafeCheckpoint(marker)}, checkpoint)
            with self.assertRaises(pickle.UnpicklingError):
                self.adapter._load_checkpoint(checkpoint)
            self.assertFalse(marker.exists())

    def test_real_checkpoint_cuda_dynamics_and_rendered_artifacts(self):
        import mujoco
        import numpy as np
        import torch

        checkpoint_path = Path(os.environ["NPA_SONIC_E2E_CHECKPOINT"])
        checkpoint = self.adapter._load_checkpoint(checkpoint_path)
        stats = self.adapter._checkpoint_tensor_stats(checkpoint)
        self.assertGreater(stats["parameter_count"], 0)
        self.assertTrue(torch.cuda.is_available(), "real CUDA device is required")
        state = checkpoint.get("policy_state_dict") or checkpoint["actor_model_state_dict"]
        matrix = next(value for value in state.values()
                      if torch.is_tensor(value) and value.ndim == 2 and min(value.shape) > 1)
        weight = matrix.detach().to(device="cuda", dtype=torch.float32).requires_grad_()
        inputs = torch.randn(16, weight.shape[1], device="cuda", requires_grad=True)
        result = torch.nn.functional.linear(inputs, weight)
        loss = result.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool(torch.isfinite(weight.grad).all()))
        self.assertGreater(float(weight.grad.norm()), 0)

        metrics_path = self.output / "mujoco_eval_metrics.json"
        with patch.dict(os.environ, {
            "SONIC_EVAL_CHECKPOINT_PATH": str(checkpoint_path),
            "SONIC_MUJOCO_METRICS_PATH": str(metrics_path),
            "SONIC_MUJOCO_STEPS": "128",
            "SONIC_MUJOCO_EPISODES": "1",
        }):
            self.assertEqual(self.adapter.main(), 0)
        metrics = json.loads(metrics_path.read_text())
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["mode"], "checkpoint-adapter-rollout")
        self.assertTrue(all(row["finite"] for row in metrics["episodes"]))

        config = self.adapter._load_yaml(Path(self.adapter.DEFAULT_CONFIG))
        model = self.adapter._load_model(config)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        frames, qpos, body_positions, timestamps = [], [], [], []
        renderer = mujoco.Renderer(model, height=240, width=320)
        try:
            for step in range(128):
                data.ctrl[:] = self.adapter._control_from_checkpoint(stats["sample"], model.nu, step, 0)
                mujoco.mj_step(model, data)
                self.assertTrue(bool(np.isfinite(data.qpos).all()))
                if step % 8 == 0:
                    renderer.update_scene(data)
                    frames.append(renderer.render().copy())
                    qpos.append(data.qpos.copy())
                    body_positions.append(data.xpos.copy())
                    timestamps.append(float(data.time))
        finally:
            renderer.close()
        frames = np.asarray(frames)
        self.assertEqual(frames.shape, (16, 240, 320, 3))
        self.assertGreater(float(frames.std()), 0)
        self.assertGreater(float(np.linalg.norm(qpos[-1] - qpos[0])), 0)
        self.assertTrue(bool(np.all(np.diff(timestamps) > 0)))
        artifact = self.output / "mujoco_rollout.npz"
        np.savez_compressed(artifact, frames=frames, qpos=np.asarray(qpos),
                            body_positions=np.asarray(body_positions), time=np.asarray(timestamps))
        with checkpoint_path.open("rb") as handle:
            checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        report = {
            "status": "completed", "adapter_source_sha256": self.source_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "torch_version": torch.__version__, "mujoco_version": mujoco.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "checkpoint_parameter_count": stats["parameter_count"],
            "actual_weight_matrix_shape": list(weight.shape),
            "cuda_forward_backward_loss": float(loss.detach()),
            "cuda_weight_gradient_norm": float(weight.grad.norm()),
            "rendered_frames": len(frames), "physics_steps": 128,
            "geometry_mode": self.adapter.GEOMETRY_MODE,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "scope": "safe checkpoint loading, real CUDA linear forward/backward, "
                     "production checkpoint-statistics MuJoCo adapter and EGL rendering",
            "limitations": ["No learned policy inference or locomotion benchmark is claimed",
                            "Geometry may use the production primitive collision proxies"],
        }
        (self.output / "security_validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)

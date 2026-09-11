"""Opt-in real MuJoCo dynamics coverage for SONIC workload image selection.

The operator first preflights and starts an owned GPU container, then supplies
NPA_SONIC_LIVE_RUNTIME_CONFIG pointing to an owner-only JSON file with context,
namespace, pod, owner (matching the pod's npa-live-owner label), gpu_target,
optional image_variant, and evidence_dir. The container must use the current
manifest's selected digest.
The full upstream training checkpoint needs a separate checkpoint_loader_python
with the pinned SONIC source's training dependencies; this strips trainer
metadata losslessly before the unchanged MuJoCo adapter reads the weights.
The operator retains responsibility for deleting this exact container afterward.
No cluster, shared runtime, storage or credentials are modified by this test.

This exercises the shipped checkpoint adapter, whose controls derive from
checkpoint tensors. It is physics evidence, not learned-policy or mesh-fidelity
validation, and does not certify a new GPU/image combination.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from npa.deploy.images import container_image_for_tool, sonic_image_entry
from npa.workbench.sonic.routing import MUJOCO_EVAL


pytestmark = pytest.mark.e2e
_MODEL_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
_CHECKPOINT_SHA256 = "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
_EPISODES = 8
_STEPS = 4096
_WORKLOAD = r'''
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.request

import mujoco
import numpy as np
import torch

out = Path("/tmp/npa-sonic-live-evidence")
out.mkdir(mode=0o700, exist_ok=True)
original_checkpoint = out / "original-checkpoint.pt"
checkpoint = out / "checkpoint.pt"
url = "https://huggingface.co/nvidia/GEAR-SONIC/resolve/" + os.environ["MODEL_REVISION"] + "/sonic_release/last.pt"
with urllib.request.urlopen(url) as source, original_checkpoint.open("wb") as target:
    while block := source.read(1024 * 1024):
        target.write(block)
with original_checkpoint.open("rb") as source:
    original_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
assert original_sha256 == os.environ["CHECKPOINT_SHA256"]
prepare = r"""
from importlib.metadata import version
import json
from pathlib import Path
import sys
import torch
from trl.experimental.ppo.ppo_trainer import OnlineTrainerState
import trl.trainer.utils
# This upstream class moved in the pinned trl release. Restore its historical
# pickle import path using the real class, without replacing any model tensors.
trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
source, target = map(Path, sys.argv[1:])
original = torch.load(source, map_location="cpu", weights_only=False)
state = original.get("policy_state_dict") or original.get("actor_model_state_dict")
assert state and all(torch.is_tensor(value) for value in state.values())
torch.save({"policy_state_dict": state}, target)
restored = torch.load(target, map_location="cpu", weights_only=True)["policy_state_dict"]
assert state.keys() == restored.keys()
assert all(torch.equal(value, restored[key]) for key, value in state.items())
(target.parent / "checkpoint-preparation.json").write_text(json.dumps({
    "lossless_state_dict": True,
    "tensor_count": len(state),
    "parameter_count": sum(value.numel() for value in state.values()),
    "trainer_state_class": OnlineTrainerState.__module__ + "." + OnlineTrainerState.__name__,
    "loader_versions": {name: version(name) for name in ("torch", "trl", "transformers", "accelerate")},
}, indent=2))
"""
subprocess.run([os.environ["CHECKPOINT_LOADER_PYTHON"], "-c", prepare, str(original_checkpoint), str(checkpoint)], check=True)
assert checkpoint.stat().st_size > 1000000
assert torch.cuda.is_available(), "GPU is required for this live acceptance"
matrix = torch.arange(1024 * 1024, device="cuda", dtype=torch.float32).reshape(1024, 1024) / 1048576
product = matrix @ matrix.T
assert bool(torch.isfinite(product).all())
torch.cuda.synchronize()

os.environ.update({
    "SONIC_EVAL_CHECKPOINT_PATH": str(checkpoint),
    "SONIC_MUJOCO_METRICS_PATH": str(out / "mujoco_eval_metrics.json"),
    "SONIC_MUJOCO_EPISODES": os.environ["LIVE_EPISODES"],
    "SONIC_MUJOCO_STEPS": os.environ["LIVE_STEPS"],
})
spec = importlib.util.spec_from_file_location("sonic_live_adapter", "/opt/npa/docker/workbench/sonic/mujoco_eval.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
states, velocities, controls = [], [], []
original_step = mujoco.mj_step
def observed_step(model, data, *args, **kwargs):
    result = original_step(model, data, *args, **kwargs)
    states.append(data.qpos.copy())
    velocities.append(data.qvel.copy())
    controls.append(data.ctrl.copy())
    return result
mujoco.mj_step = observed_step
try:
    assert adapter.main() == 0
finally:
    mujoco.mj_step = original_step
np.savez_compressed(out / "dynamics.npz", qpos=np.stack(states), qvel=np.stack(velocities), ctrl=np.stack(controls))
with checkpoint.open("rb") as checkpoint_file:
    checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
proof = {
    "checkpoint_sha256": checkpoint_sha256,
    "upstream_checkpoint_sha256": original_sha256,
    "checkpoint_bytes": checkpoint.stat().st_size,
    "model_revision": os.environ["MODEL_REVISION"],
    "observed_physics_steps": len(states),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_kernel_finite": bool(torch.isfinite(product).all()),
    "learned_policy_inference": False,
}
(out / "live-proof.json").write_text(json.dumps(proof, indent=2))
checkpoint.unlink()
original_checkpoint.unlink()
'''


def test_manifest_selected_mujoco_checkpoint_dynamics() -> None:
    if os.environ.get("NPA_SONIC_IMAGE_WORKLOAD_LIVE") != "1":
        pytest.skip("set NPA_SONIC_IMAGE_WORKLOAD_LIVE=1 after GPU/access preflight")
    config_path = Path(os.environ["NPA_SONIC_LIVE_RUNTIME_CONFIG"])
    assert config_path.stat().st_mode & 0o077 == 0, "runtime config must be owner-only"
    config = json.loads(config_path.read_text())
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0, "evidence must be owner-only"
    kubectl = ["kubectl", "--context", config["context"], "-n", config["namespace"]]

    def run(args: list[str], *, input_text: str | None = None) -> str:
        result = subprocess.run(
            [*kubectl, *args], input=input_text, capture_output=True, text=True, check=False
        )
        if result.returncode:
            (evidence / "runtime-failure.json").write_text(json.dumps({
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }, indent=2))
            pytest.fail("live kubectl operation failed; inspect private runtime evidence", pytrace=False)
        return result.stdout

    pod = json.loads(run(["get", "pod", config["pod"], "-o", "json"]))
    assert pod["metadata"]["labels"].get("npa-live-owner") == config["owner"]
    assert pod["status"]["phase"] == "Running"
    selection = {
        "gpu_target": config["gpu_target"],
        "image_variant": config.get("image_variant"),
        "workload": MUJOCO_EVAL,
    }
    image = container_image_for_tool("sonic", **selection)
    entry = sonic_image_entry(**selection)
    digest = entry["digest"]
    assert digest in pod["status"]["containerStatuses"][0]["imageID"]
    assert pod["spec"]["containers"][0]["image"].split("@")[0] == image.rsplit(":", 1)[0]
    command = [
        "exec", "-i", config["pod"], "--", "env",
        f"MODEL_REVISION={_MODEL_REVISION}", f"LIVE_EPISODES={_EPISODES}", f"LIVE_STEPS={_STEPS}",
        f"CHECKPOINT_SHA256={_CHECKPOINT_SHA256}",
        f"CHECKPOINT_LOADER_PYTHON={config['checkpoint_loader_python']}",
        "/opt/npa/venv/bin/python", "-",
    ]
    run(command, input_text=_WORKLOAD)
    run(["cp", f"{config['pod']}:/tmp/npa-sonic-live-evidence/.", str(evidence)])
    metrics = json.loads((evidence / "mujoco_eval_metrics.json").read_text())
    proof = json.loads((evidence / "live-proof.json").read_text())
    preparation = json.loads((evidence / "checkpoint-preparation.json").read_text())
    assert preparation["lossless_state_dict"] is True
    assert proof["upstream_checkpoint_sha256"] == _CHECKPOINT_SHA256
    assert metrics["status"] == "completed"
    assert metrics["mode"] == "checkpoint-adapter-rollout"
    assert metrics["eval"]["episodes"] == _EPISODES
    assert metrics["eval"]["steps_per_episode"] == _STEPS
    assert metrics["metrics"]["finite_rate"] == 1.0
    assert all(metrics["mujoco"][key] > 0 for key in ("nq", "nv", "nu", "nbody"))
    assert metrics["checkpoint"]["parameter_count"] > 1000000
    assert proof["observed_physics_steps"] == _EPISODES * _STEPS
    assert proof["cuda_kernel_finite"] is True
    assert proof["learned_policy_inference"] is False
    assert (evidence / "dynamics.npz").stat().st_size > 100000
    with np.load(evidence / "dynamics.npz") as dynamics:
        for key, dimension in (("qpos", "nq"), ("qvel", "nv"), ("ctrl", "nu")):
            assert dynamics[key].shape == (_EPISODES * _STEPS, metrics["mujoco"][dimension])
            assert np.isfinite(dynamics[key]).all()
            assert np.ptp(dynamics[key], axis=0).max() > 0
    proof["resolved_image"] = image
    proof["runtime_digest"] = digest
    proof["gpu_target"] = config["gpu_target"]
    with (evidence / "dynamics.npz").open("rb") as dynamics_file:
        proof["dynamics_sha256"] = hashlib.file_digest(dynamics_file, "sha256").hexdigest()
    (evidence / "live-proof.json").write_text(json.dumps(proof, indent=2) + "\n")

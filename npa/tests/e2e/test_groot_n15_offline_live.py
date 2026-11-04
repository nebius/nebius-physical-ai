"""Opt-in real N1.5 CUDA training with operator-staged data and model weights."""

import json
import hashlib
import math
import os
from pathlib import Path
import subprocess

import pytest

from npa.workbench.groot.offline_feedback import prepare_feedback


@pytest.mark.e2e
@pytest.mark.gpu
def test_groot_n15_offline_native_training(tmp_path):
    config_path = os.environ.get("NPA_GROOT_N15_OFFLINE_LIVE_CONFIG")
    if not config_path:
        pytest.skip(
            "native N1.5 CUDA runtime and staged dataset/model config are not supplied"
        )
    config = json.loads(Path(config_path).read_text())
    manifest_path = tmp_path / "weights.json"
    manifest = prepare_feedback(
        Path(config["dataset_root"]),
        Path(config["feedback_path"]),
        manifest_path,
        action_horizon=config["action_horizon"],
        mode=config["mode"],
        temperature=config.get("temperature", 1.0),
    )
    output = tmp_path / "trained"
    command = [
        config["python"],
        "-m",
        "npa.workbench.groot.offline",
        "train",
        "--manifest-path",
        str(manifest_path),
        "--output-dir",
        str(output),
    ]
    for key in (
        "dataset_root",
        "groot_source",
        "base_model",
        "data_config",
        "embodiment_tag",
        "max_steps",
        "batch_size",
    ):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    completed = subprocess.run(
        command, env=env, check=True, text=True, capture_output=True
    )
    receipt = json.loads(completed.stdout)
    assert receipt == json.loads((output / "offline-training.json").read_text())
    assert receipt["optimizer_steps"] == config["max_steps"]
    assert math.isfinite(receipt["training_loss"])
    assert receipt["optimizer_loss_trace"]
    assert receipt["optimizer_loss_trace"][-1]["step"] == config["max_steps"]
    assert all(
        math.isfinite(point["loss"]) for point in receipt["optimizer_loss_trace"]
    )
    assert math.isfinite(receipt["held_out_eval_loss"])
    assert receipt["visible_cuda_devices"] == 1
    assert receipt["checkpoint_reload"]["verified"] is True
    assert (
        receipt["checkpoint_reload"]["checkpoint_sha256"]
        == receipt["candidate_checkpoint_sha256"]
    )
    assert receipt["candidate_checkpoint_sha256"] != receipt["base_checkpoint_sha256"]
    assert receipt["terminal_mask_evidence"]["valid_action_horizon_steps"] == 1
    assert receipt["terminal_mask_evidence"]["masked_action_horizon_steps"] == (
        config["action_horizon"] - 1
    )
    assert receipt["terminal_mask_evidence"]["decoded_observation_shapes"]
    assert receipt["dataset_sha256"] == manifest.dataset_sha256
    assert receipt["robot_evaluation_performed"] is False
    before = _action_decoder_weights(Path(config["base_model"]))
    after = _action_decoder_weights(output)
    assert before.keys() == after.keys()
    assert any(before[key] != after[key] for key in before)


def _action_decoder_weights(checkpoint):
    import torch
    from safetensors import safe_open

    result = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith("action_head.action_decoder."):
                    tensor = tensors.get_tensor(key).contiguous().view(torch.uint8)
                    result[key] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    assert result, "native safetensors action decoder weights are required"
    return result

"""Reload the portable trained policy offline and infer an action from a public demonstration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .public_vla_data import write_json
from .public_vla_export import file_sha256


def main() -> None:
    """Verify exported dependencies and CUDA inference without the original model cache.

    Args:
        None; --input-path is the private run directory.
    Returns:
        None; writes export-verification.json in the private run directory.
    Raises:
        ValueError: Exported bytes or inferred actions fail validation.
        RuntimeError: Native policy loading or CUDA inference fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    root = parser.parse_args().input_path.resolve()
    bundle = root / "exported-policy"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or file_sha256(path) != checksum:
            raise ValueError("exported artifact checksum or containment failed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ["HF_HOME"] = str(root / "export-verification-cache")
    os.chdir(bundle)
    action = _infer_public_observation(root)
    action["parameter_change"] = _weight_change(root)
    write_json(root / "export-verification.json", action)


def _weight_change(root):
    import torch
    from safetensors import safe_open

    name = "model.state_proj.weight"
    with safe_open(
        root / "inputs/model/model.safetensors", framework="pt", device="cpu"
    ) as source:
        initial = source.get_tensor(name).float()
    with safe_open("policy/model.safetensors", framework="pt", device="cpu") as trained:
        delta = (trained.get_tensor(name).float() - initial).abs()
    maximum = delta.max().item()
    if not torch.isfinite(delta).all() or maximum <= 0:
        raise ValueError("native training did not change the selected state projection")
    return {
        "parameter": name,
        "maximum_absolute_change": maximum,
        "changed_elements": int((delta > 0).sum().item()),
    }


def _infer_public_observation(root):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    plan = json.loads((root / "plans/specialist.json").read_text())
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=root / "inputs/dataset",
        episodes=[plan["episodes"][0]],
        video_backend="torchcodec",
    )
    policy = SmolVLAPolicy.from_pretrained("policy").to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path="policy",
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    observation = preprocessor(dataset[0])
    with torch.inference_mode():
        action = postprocessor(policy.select_action(observation))
    if tuple(action.shape) != (1, 7) or not torch.isfinite(action).all():
        raise ValueError("portable policy did not produce a finite seven-axis action")
    return {
        "offline_reload": True,
        "cuda_inference": True,
        "observation_source": "public LeRobot demonstration",
        "action_shape": list(action.shape),
        "finite_action": True,
        "weights_sha256": file_sha256(Path("policy/model.safetensors")),
    }


if __name__ == "__main__":
    main()

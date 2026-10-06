"""Validate installed dependencies, native default decoding and ACT on CPU."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


def _dependencies() -> None:
    import npa
    from npa.server.app import app

    assert Path(npa.__file__) == Path("/opt/npa/src/npa/__init__.py")
    assert app is not None
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    torch.set_num_threads(2)
    torch.manual_seed(0)


def _dataset_features() -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (8,), "names": None},
    }
    for camera in ("workspace", "wrist"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _native_dataset(root: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_codec

    dataset = LeRobotDataset.create(
        "npa/native-cpu-smoke",
        fps=10,
        root=root / "dataset",
        features=_dataset_features(),
        vcodec="h264",
        encoder_threads=1,
    )
    for index in range(4):
        image = np.full((64, 64, 3), index * 60, dtype=np.uint8)
        dataset.add_frame(
            {
                "observation.state": np.arange(7, dtype=np.float32) / 7,
                "action": np.arange(8, dtype=np.float32) / 8,
                "observation.images.workspace": image,
                "observation.images.wrist": image[:, ::-1].copy(),
                "task": "offline camera and ACT validation",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    loaded = LeRobotDataset("npa/native-cpu-smoke", root=root / "dataset")
    assert len(loaded) == 4
    for index in (0, 3):
        sample = loaded[index]
        for camera in ("workspace", "wrist"):
            image = sample[f"observation.images.{camera}"]
            assert image.shape == (3, 64, 64) and image.dtype == torch.float32
            assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        assert sample["frame_index"].item() == index
        assert abs(sample["timestamp"].item() - index / 10) < 1e-6
    return {"frames": len(loaded), "backend": get_safe_default_codec()}


def _act_configuration():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=1,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.workspace": PolicyFeature(
                FeatureType.VISUAL, (3, 64, 64)
            ),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )


def _training_batch(configuration) -> dict:
    batch = {
        name: torch.rand(2, *feature.shape)
        for name, feature in configuration.input_features.items()
    }
    batch["action"] = torch.rand(2, 2, 8)
    batch["action_is_pad"] = torch.zeros(2, 2, dtype=torch.bool)
    return batch


def _optimizer_step(policy, batch) -> tuple[float, int]:
    original = {
        name: value.detach().clone() for name, value in policy.named_parameters()
    }
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loss, _ = policy(batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in policy.parameters()
    )
    optimizer.step()
    changed = sum(
        not torch.equal(original[name], value.detach())
        for name, value in policy.named_parameters()
    )
    assert changed > 0
    return loss.item(), changed


def _checkpoint_round_trip(policy, checkpoint, batch):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy.save_pretrained(checkpoint)
    loaded = ACTPolicy.from_pretrained(checkpoint).eval()
    assert set(policy.state_dict()) == set(loaded.state_dict())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    policy.eval()
    observation = {
        name: value
        for name, value in batch.items()
        if name in policy.config.input_features
    }
    policy.reset()
    loaded.reset()
    with torch.inference_mode():
        expected = policy.select_action(observation)
        actual = loaded.select_action(observation)
    assert actual.shape == (2, 8) and torch.isfinite(actual).all()
    torch.testing.assert_close(expected, actual)
    return loaded, observation


def _processor_round_trip(configuration, policy, checkpoint, observation) -> None:
    from lerobot.policies.factory import make_pre_post_processors

    features = {**configuration.input_features, **configuration.output_features}
    stats = {
        name: {"mean": torch.zeros(feature.shape), "std": torch.ones(feature.shape)}
        for name, feature in features.items()
    }
    pre, post = make_pre_post_processors(configuration, dataset_stats=stats)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    loaded_pre, loaded_post = make_pre_post_processors(
        configuration, pretrained_path=str(checkpoint)
    )
    policy.reset()
    with torch.inference_mode():
        action = loaded_post(
            policy.select_action(
                loaded_pre({name: value[0] for name, value in observation.items()})
            )
        )
    assert action.shape == (1, 8) and torch.isfinite(action).all()


def _native_act(root: Path) -> dict:
    from lerobot.policies.act.modeling_act import ACTPolicy

    configuration = _act_configuration()
    policy = ACTPolicy(configuration).train()
    batch = _training_batch(configuration)
    loss, changed = _optimizer_step(policy, batch)
    checkpoint = root / "pretrained_model"
    loaded, observation = _checkpoint_round_trip(policy, checkpoint, batch)
    _processor_round_trip(configuration, loaded, checkpoint, observation)
    return {
        "loss": loss,
        "changed_parameters": changed,
        "checkpoint_processors": "passed",
    }


def _diffusion_configuration():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    return DiffusionConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (4,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        num_train_timesteps=4,
        num_inference_steps=4,
    )


def _native_diffusion(root: Path) -> dict:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    configuration = _diffusion_configuration()
    policy = DiffusionPolicy(configuration).train()
    batch = {
        name: torch.rand(2, 2, *feature.shape)
        for name, feature in configuration.input_features.items()
    }
    batch.update(
        action=torch.rand(2, 8, 4), action_is_pad=torch.zeros(2, 8, dtype=torch.bool)
    )
    loss, changed = _optimizer_step(policy, batch)
    checkpoint = root / "diffusion_pretrained_model"
    policy.save_pretrained(checkpoint)
    loaded = DiffusionPolicy.from_pretrained(checkpoint).eval()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    observation = {
        name: value[:, 0]
        for name, value in batch.items()
        if name in configuration.input_features
    }
    policy.eval()
    policy.reset()
    loaded.reset()
    with torch.inference_mode():
        torch.manual_seed(0)
        expected = policy.select_action(observation)
        torch.manual_seed(0)
        actual = loaded.select_action(observation)
    assert actual.shape == (2, 4) and torch.isfinite(actual).all()
    torch.testing.assert_close(expected, actual)
    return {"loss": loss, "changed_parameters": changed, "checkpoint_equal": True}


def _main() -> None:
    _dependencies()
    with tempfile.TemporaryDirectory(prefix="npa-lerobot-native-cpu-") as directory:
        root = Path(directory)
        receipt = {
            "version": metadata.version("lerobot"),
            "default_camera_dataset": _native_dataset(root),
            "act": _native_act(root),
            "diffusion": _native_diffusion(root),
            "gpu_acceptance": False,
        }
    print("LEROBOT_NATIVE_CPU_OK " + json.dumps(receipt))


if __name__ == "__main__":
    _main()

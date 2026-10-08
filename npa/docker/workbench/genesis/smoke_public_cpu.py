"""Exercise the public Genesis image's real ACT and PyAV dataset dependencies."""

from __future__ import annotations

from importlib import metadata
from importlib.util import find_spec
import builtins
import os
from pathlib import Path
import socket
import tempfile

import numpy as np
import torch


EXCLUDED = {"tetgen", "diffusers", "wandb", "torchcodec", "transformers", "peft"}


def _require_offline_runtime() -> None:
    for name in EXCLUDED:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"excluded public runtime distribution: {name}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in EXCLUDED - {"tetgen"}:
        assert find_spec(name) is None, f"excluded import is available: {name}"
    original_import = builtins.__import__

    def refuse_excluded(name, *arguments, **options):
        if name.split(".")[0] in EXCLUDED - {"tetgen"}:
            raise ModuleNotFoundError(f"excluded public runtime import: {name}")
        return original_import(name, *arguments, **options)

    builtins.__import__ = refuse_excluded

    def refuse_network(*_args, **_kwargs):
        raise RuntimeError("the public CPU smoke must not fetch external payloads")

    socket.socket.connect = refuse_network


def _native_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_codec
    from npa.adapter.sim_to_lerobot import convert

    assert get_safe_default_codec() == "pyav"
    episode = root / "raw" / "episode_0000"
    episode.mkdir(parents=True)
    frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    for index in range(4):
        frames[index] = index * 60
    np.save(episode / "obs_workspace.npy", frames)
    np.save(episode / "obs_wrist.npy", frames[:, :, ::-1].copy())
    np.save(episode / "state.npy", np.arange(28, dtype=np.float32).reshape(4, 7) / 28)
    np.save(episode / "actions.npy", np.arange(32, dtype=np.float32).reshape(4, 8) / 32)
    dataset_root = convert(root / "raw", root / "dataset", fps=10)
    dataset = LeRobotDataset("npa/genesis-cpu-smoke", root=dataset_root)
    assert dataset.video_backend == "pyav"
    assert len(dataset) == 4
    for index in (0, 3):
        sample = dataset[index]
        for camera in ("workspace", "wrist"):
            image = sample[f"observation.images.{camera}"]
            assert image.shape == (3, 64, 64) and image.dtype == torch.float32
            assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        assert torch.isfinite(sample["observation.state"]).all()
        assert torch.isfinite(sample["action"]).all()
        assert sample["frame_index"].item() == index
        assert abs(sample["timestamp"].item() - index / 10) < 1e-6
    return dataset


def _act_configuration():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.workspace": PolicyFeature(
                FeatureType.VISUAL, (3, 64, 64)
            ),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )


def _act_round_trip(root: Path, dataset) -> None:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from npa.genesis.eval_student import _load_student_policy

    torch.manual_seed(0)
    configuration = _act_configuration()
    policy = make_policy(configuration)
    assert isinstance(policy, ACTPolicy)
    policy = policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        configuration, dataset_stats=dataset.meta.stats
    )
    observation = {key: dataset[0][key] for key in configuration.input_features}
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(observation)))
    assert action.shape == (1, 8) and torch.isfinite(action).all()
    checkpoint = root / "pretrained_model"
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    loaded, loaded_preprocessor, loaded_postprocessor = _load_student_policy(checkpoint)
    assert loaded_preprocessor is not None and loaded_postprocessor is not None
    loaded.reset()
    with torch.inference_mode():
        reloaded_action = loaded_postprocessor(
            loaded.select_action(loaded_preprocessor(observation))
        )
    torch.testing.assert_close(action, reloaded_action)


def _main() -> None:
    _require_offline_runtime()
    with tempfile.TemporaryDirectory(prefix="npa-genesis-public-cpu-") as directory:
        root = Path(directory)
        _act_round_trip(root, _native_dataset(root))
    print("GENESIS_PUBLIC_CPU_OK native_pyav_dataset=passed act_round_trip=passed")


if __name__ == "__main__":
    _main()

"""Exercise the exact NPA LeRobot ACT derivative without external assets."""

from __future__ import annotations

import json
import math
import tempfile
from importlib.metadata import metadata, version
from pathlib import Path

import torch
from packaging.requirements import Requirement

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


EXPECTED_REQUIRES = {
    "gymnasium": "==0.29.1",
    "opencv-python": "<4.14.0,>=4.9.0",
    "setuptools": "<84.0.0,>=83.0.0",
    "torch": "<2.14.0,>=2.13.0",
    "torchvision": "<0.29.0,>=0.28.0",
}


def _config() -> ACTConfig:
    return ACTConfig(
        input_features={
            "observation.images.workspace": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
            "observation.images.wrist": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 64, 64)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        device="cpu",
    )


def _batch(*, state_width: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(595)
    return {
        "observation.images.workspace": torch.rand((1, 3, 64, 64), generator=generator),
        "observation.images.wrist": torch.rand((1, 3, 64, 64), generator=generator),
        "observation.state": torch.rand((1, state_width), generator=generator),
    }


def main() -> None:
    assert version("lerobot") == "0.6.1+npa1"
    requirements = [
        Requirement(raw) for raw in (metadata("lerobot").get_all("Requires-Dist") or [])
    ]
    observed = {
        requirement.name: str(requirement.specifier)
        for requirement in requirements
        if requirement.marker is None and requirement.name in EXPECTED_REQUIRES
    }
    assert observed == EXPECTED_REQUIRES, observed
    assert not any(
        requirement.name == "opencv-python-headless" for requirement in requirements
    )

    torch.manual_seed(595)
    policy = ACTPolicy(_config()).eval()
    with torch.inference_mode():
        first = policy.select_action(_batch())
    assert first.shape == (1, 7), first.shape
    assert all(math.isfinite(float(item)) for item in first.flatten())

    try:
        with torch.inference_mode():
            policy.select_action(_batch(state_width=15))
    except (RuntimeError, ValueError) as exc:
        shape_failure = type(exc).__name__
    else:
        raise AssertionError("ACT accepted a 15-wide state for a 16-wide configuration")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint"
        policy.save_pretrained(checkpoint)
        reloaded = ACTPolicy.from_pretrained(checkpoint, config=policy.config).eval()
        for expected, actual in zip(
            policy.parameters(), reloaded.parameters(), strict=True
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        policy.reset()
        reloaded.reset()
        with torch.inference_mode():
            expected_action = policy.select_action(_batch())
            actual_action = reloaded.select_action(_batch())
        torch.testing.assert_close(expected_action, actual_action, rtol=0, atol=0)

        weights = next(checkpoint.glob("*.safetensors"))
        weights.unlink()
        try:
            ACTPolicy.from_pretrained(checkpoint, config=policy.config)
        except (OSError, RuntimeError, ValueError) as exc:
            corrupt_failure = type(exc).__name__
        else:
            raise AssertionError("ACT accepted a checkpoint with missing weights")

    print(
        json.dumps(
            {
                "checkpoint_roundtrip": True,
                "corrupt_checkpoint_rejected": corrupt_failure,
                "input_shape_rejected": shape_failure,
                "lerobot": version("lerobot"),
                "output_shape": list(first.shape),
                "torch": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

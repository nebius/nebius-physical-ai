"""Execute the shipped DINO patch at its model-loading boundary."""

import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest


@pytest.fixture
def patched_encoder(tmp_path):
    source = tmp_path / "src/flexpi/models/dino_encoder.py"
    source.parent.mkdir(parents=True)
    # Preserve the pinned upstream hunk positions while omitting GPU-only imports.
    lines = ["from __future__ import annotations"] + [""] * 57
    lines += ["class DinoEncoder:", "    def __init__(self, model_name):"]
    lines += [
        "        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)"
    ]
    source.write_text("\n".join(lines) + "\n")
    patch = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/pins/dino-checkpoint.patch"
    )
    subprocess.run(
        ["patch", "--batch", "-p1", "-i", str(patch)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    timm = Mock()
    spec = importlib.util.spec_from_file_location("patched_dino_encoder", source)
    module = importlib.util.module_from_spec(spec)
    module.timm = timm
    spec.loader.exec_module(module)
    return module.DinoEncoder, timm


@pytest.mark.parametrize("checkpoint", [None, "", "  "])
def test_missing_dino_pin_refuses_before_model_or_network_loading(
    patched_encoder, monkeypatch, checkpoint
):
    encoder, timm = patched_encoder
    monkeypatch.delenv("FLEX_PI_DINO_CHECKPOINT", raising=False)
    if checkpoint is not None:
        monkeypatch.setenv("FLEX_PI_DINO_CHECKPOINT", checkpoint)
    with pytest.raises(RuntimeError, match="FLEX_PI_DINO_CHECKPOINT"):
        encoder("pinned-dino")
    timm.create_model.assert_not_called()


def test_verified_dino_path_disables_pretrained_fetch(
    patched_encoder, monkeypatch, tmp_path
):
    encoder, timm = patched_encoder
    checkpoint = tmp_path / "verified.safetensors"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setenv("FLEX_PI_DINO_CHECKPOINT", str(checkpoint))
    encoder("pinned-dino")
    timm.create_model.assert_called_once_with(
        "pinned-dino",
        pretrained=False,
        num_classes=0,
        checkpoint_path=str(checkpoint),
    )

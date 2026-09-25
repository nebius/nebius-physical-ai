"""Reject mislabeled checkpoints and invalid full-model rendering evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fa4_sdxl_validation.py"


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fa4_sdxl_validation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed", [False, True])
def test_checkpoint_bytes_must_match_pinned_manifest(
    validation, tmp_path, monkeypatch, changed
):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"checkpoint")
    manifest = {
        "model": validation.MODEL_ID,
        "revision": validation.MODEL_REVISION,
        "files": {
            weights.name: {
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "bytes": weights.stat().st_size,
            }
        },
    }
    (tmp_path / "fa4_sdxl_model.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(validation, "__file__", str(tmp_path / SCRIPT.name))
    if changed:
        weights.write_bytes(b"different!")
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            validation._verify_model(tmp_path)
    else:
        assert validation._verify_model(tmp_path) == manifest


def test_model_revision_cannot_be_relabelled(validation, tmp_path, monkeypatch):
    manifest = {"model": validation.MODEL_ID, "revision": "other-revision", "files": {}}
    (tmp_path / "fa4_sdxl_model.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(validation, "__file__", str(tmp_path / SCRIPT.name))
    with pytest.raises(RuntimeError, match="Unexpected model revision"):
        validation._verify_model(tmp_path)


@pytest.mark.parametrize("failure", ["blank", "nonfinite", "dimensions"])
def test_bad_render_does_not_become_successful_evidence(validation, failure):
    from PIL import Image

    image = Image.new("RGB", (8, 8))
    latents = torch.ones(1, 4, 2, 2)
    if failure == "nonfinite":
        latents[0, 0, 0, 0] = float("nan")
    if failure != "blank":
        image.putpixel((0, 0), (255, 255, 255))
    case = {"width": 16 if failure == "dimensions" else 8, "height": 8}
    with pytest.raises(RuntimeError):
        validation._check_outputs(image, latents, case)


@pytest.mark.parametrize("flag", ["--steps", "--repeats"])
def test_empty_experiment_is_rejected(validation, monkeypatch, flag):
    monkeypatch.setattr(
        sys, "argv", ["validate", "--model-path", ".", "--output-path", ".", flag, "0"]
    )
    with pytest.raises(SystemExit) as error:
        validation._parse_args()
    assert error.value.code == 2

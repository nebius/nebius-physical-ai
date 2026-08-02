"""Build-time verification that the npa-cosmos3 generate path is fully resolvable.

Runs inside the framework's own venv (no GPU, no weights) and walks the inference
graph up to — but not including — weight loading: flags, the torch/flash-attn
stack, the guardrail package, checkpoint-URI resolution, and setup/sample
resolution for every generation mode the workbench advertises.

This is what lets the image ship a trimmed dependency set with confidence. The
image installs upstream's ``guardrail`` extra plus the two lock-pinned members of
the ``train`` extra that the generate path actually needs, instead of the whole
``--all-extras`` closure; if a framework bump changes that, this script fails the
build rather than the first GPU run.

It also asserts no model weights were baked into the image, which is what keeps
the image redistributable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Generation is inference: upstream's own switch for making the training
# dependencies optional (it also suppresses training-only CLI args and limits
# helper downloads to tokenizer configs).
os.environ.setdefault("COSMOS_TRAINING", "0")
os.environ.setdefault("COSMOS_DEVICE", "cpu")

MODES = ("text2image", "image2image", "text2video", "image2video", "video2video")
VISION_MODES = {"image2image", "image2video", "video2video"}
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt")
WEIGHT_MIN_BYTES = 50 * 1024 * 1024

failures: list[str] = []


def step(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
        failures.append(name)
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"[ok]   {name}" + (f": {detail}" if detail else ""))


def check_flags() -> str:
    from cosmos_framework.utils import flags

    if flags.TRAINING:
        raise RuntimeError(
            "COSMOS_TRAINING must be off in this image: the generate runtime "
            "installs no training dependencies"
        )
    return f"TRAINING={flags.TRAINING} DEVICE={flags.DEVICE}"


def check_torch_stack() -> str:
    import flash_attn
    import torch

    return (
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"flash_attn={flash_attn.__version__}"
    )


def check_inference_entrypoint() -> str:
    from cosmos_framework.scripts import inference

    return inference.__name__


def check_model_module() -> str:
    from cosmos_framework.inference import model

    return model.__name__


def check_guardrail() -> str:
    from cosmos_framework.auxiliary.guardrail import common  # noqa: F401

    return "guardrail package importable (guardrails stay on by default)"


def check_checkpoint_lookup() -> str:
    from cosmos_framework.utils.checkpoint_db import get_checkpoint_uri

    checkpoint = os.environ.get("NPA_COSMOS3_CHECKPOINT", "Cosmos3-Nano")
    return f"{checkpoint} -> {get_checkpoint_uri(checkpoint)}"


def _sample_for(mode: str, root: Path) -> Path:
    payload: dict[str, object] = {
        "model_mode": mode,
        "name": f"verify-{mode}",
        "prompt": "a robot arm sorting colored blocks on a white workbench",
    }
    if mode in VISION_MODES:
        vision = root / ("source.mp4" if mode == "video2video" else "source.png")
        vision.parent.mkdir(parents=True, exist_ok=True)
        vision.write_bytes(b"\0")
        payload["vision_path"] = str(vision)
    path = root / f"{mode}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def check_mode_resolution() -> str:
    """Resolve setup args and sample overrides for every advertised mode."""

    from cosmos_framework.inference.args import OmniSetupOverrides

    resolved = []
    with tempfile.TemporaryDirectory(prefix="npa-cosmos3-verify-") as tmp:
        root = Path(tmp)
        for mode in MODES:
            sample = _sample_for(mode, root)
            setup_args = OmniSetupOverrides(
                checkpoint_path=os.environ.get("NPA_COSMOS3_CHECKPOINT", "Cosmos3-Nano"),
                output_dir=root / "out",
            ).build_setup()
            samples = setup_args.get_sample_overrides_cls().from_files([sample])
            if len(samples) != 1 or samples[0].model_mode != mode:
                raise RuntimeError(f"{mode}: unexpected sample resolution {samples!r}")
            resolved.append(f"{mode}->{setup_args.get_inference_cls().__name__}")
    return " ".join(resolved)


def check_no_baked_weights() -> str:
    roots = [Path("/opt/cosmos3"), Path("/opt/npa")]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home))
    baked = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in WEIGHT_SUFFIXES
                and path.stat().st_size > WEIGHT_MIN_BYTES
            ):
                baked.append(str(path))
    if baked:
        raise RuntimeError(f"model weights present in the image: {baked[:5]}")
    return "no weight files baked (checkpoints download at runtime)"


def main() -> int:
    step("flags", check_flags)
    step("torch + flash-attn", check_torch_stack)
    step("scripts.inference import", check_inference_entrypoint)
    step("inference.model import", check_model_module)
    step("guardrail import", check_guardrail)
    step("checkpoint db lookup", check_checkpoint_lookup)
    step("setup + sample resolution", check_mode_resolution)
    step("no baked weights", check_no_baked_weights)

    print()
    if failures:
        print(f"[FAIL] npa-cosmos3 environment verification failed: {failures}")
        return 1
    print("[PASS] npa-cosmos3 generate environment verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

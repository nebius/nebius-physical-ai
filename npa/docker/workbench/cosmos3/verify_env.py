"""Build-time verification that the npa-cosmos3 generate path is fully resolvable.

Runs inside the framework's own venv (no GPU, no weights) and walks the inference
graph up to — but not including — weight loading: flags, the torch/NATTEN
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
import re
import sys
import tempfile
from importlib import metadata
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
XET_KNOWN_BAD_PAIR = ("1.23.0", "1.5.1")
NATIVE_SOURCE_MARKER = ".npa_source_revision"

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
    import natten
    import torch
    from natten.functional import attention

    expected = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "torchcodec": "0.14.0+cu130",
        "natten": "0.21.6+cu130.torch213",
    }
    for name, version in expected.items():
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}={actual}; expected {version}")
    for name in ("flash-attn", "flash-attn-3-nv"):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"incompatible inherited attention package: {name}")
    if not callable(attention):
        raise RuntimeError("NATTEN attention is not callable")

    return (
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"natten={natten.__version__}"
    )


def check_hf_transfer_pair() -> str:
    """Reject the frozen Hugging Face pair known to corrupt gated Xet downloads."""

    hub = metadata.version("huggingface_hub")
    xet = metadata.version("hf-xet")
    if (hub, xet) == XET_KNOWN_BAD_PAIR:
        raise RuntimeError(
            "known-bad gated-download pair baked into image: "
            f"huggingface_hub={hub} hf-xet={xet} (huggingface/xet-core#895)"
        )
    return f"huggingface_hub={hub} hf-xet={xet} (Xet enabled)"


def check_inference_entrypoint() -> str:
    from cosmos_framework.scripts import inference

    return inference.__name__


def check_pinned_framework_source() -> str:
    """Bind the removed-git checkout to the exact source revision in the image."""

    root = Path(os.environ.get("COSMOS3_REPO", "/opt/cosmos3/cosmos-framework"))
    marker = root / NATIVE_SOURCE_MARKER
    revision = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"invalid or absent framework source marker: {marker}")
    return f"cosmos-framework={revision}"


def check_action_inference_contract() -> str:
    """Verify upstream's native JSON action schema and embodiment mapping."""

    from cosmos_framework.data.generator.action.domain_utils import (
        EMBODIMENT_TO_RAW_ACTION_DIM,
        get_domain_id,
    )
    from cosmos_framework.inference.action import get_action_sample_data
    from cosmos_framework.inference.args import ActionDataOverrides, ModelMode

    if EMBODIMENT_TO_RAW_ACTION_DIM.get("droid_lerobot") != 10:
        raise RuntimeError("droid_lerobot must retain its 10-channel raw action contract")
    if get_domain_id("droid_lerobot") != 8:
        raise RuntimeError("droid_lerobot must retain embodiment domain id 8")
    required = {"action_path", "domain_name", "action_chunk_size", "image_size", "view_point"}
    absent = sorted(required - set(ActionDataOverrides.model_fields))
    if absent:
        raise RuntimeError(f"native action sample JSON is missing fields: {absent}")
    if ModelMode.FORWARD_DYNAMICS.value != "forward_dynamics":
        raise RuntimeError("native forward-dynamics model mode changed")
    if not callable(get_action_sample_data):
        raise RuntimeError("native action sample loader is not callable")
    return "droid_lerobot domain=8 raw_action_dim=10 action_json_schema=present"


def check_model_module() -> str:
    from cosmos_framework.inference import model

    return model.__name__


def check_guardrail() -> str:
    from cosmos_framework.auxiliary.guardrail import common  # noqa: F401
    from cosmos_framework.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
        VideoContentSafetyFilter,
    )

    return (
        "guardrail package and generated-media safety model importable "
        f"({VideoContentSafetyFilter.__name__}; guardrails stay on by default)"
    )


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
                checkpoint_path=os.environ.get(
                    "NPA_COSMOS3_CHECKPOINT", "Cosmos3-Nano"
                ),
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
    step("Hugging Face transfer pair", check_hf_transfer_pair)
    step("pinned framework source", check_pinned_framework_source)
    step("scripts.inference import", check_inference_entrypoint)
    step("DROID action inference contract", check_action_inference_contract)
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

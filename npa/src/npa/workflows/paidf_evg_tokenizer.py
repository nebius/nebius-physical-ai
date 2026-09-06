"""Source-bound Cosmos3 tokenizer compatibility for the accepted EVG image."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.paidf_guardrails import PaidfGuardrailError, _digest_document
from npa.workflows.paidf_upstream import (
    COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
    COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
)

PIPELINE_PATH = "diffusion/models/cosmos3/pipeline_cosmos3.py"
PIPELINE_SOURCE_SHA256 = "8a70b5d446315d2f6281bfacb4c332dac17cf256f813cbd26c4e01a105d41f49"
PIPELINE_PATCHED_SHA256 = "e0cfe2d60c44900d38bcae930740944cf3a23772cefab97328c372687bb0b262"
MODEL_CONFIG_SHA256 = "355f2b4e5bad7b01f11ef6cb68ebc176f61b95c3276092ea225b1bea0e01e95c"
VENDOR_PACKAGE = Path("/usr/local/lib/python3.12/dist-packages/vllm_omni")
TOKENIZER_CALL = (
    "        self.tokenizer = AutoTokenizer.from_pretrained(\n"
    "            model_path,\n"
    '            subfolder="text_tokenizer",\n'
    "            local_files_only=local_files_only,\n"
    "        )\n"
)


def tokenizer_source_adaptation() -> dict[str, Any]:
    value = {
        "schema": "npa.paidf.evg-tokenizer-source-adaptation.v1",
        "repository": "https://github.com/vllm-project/vllm-omni",
        "license": "Apache-2.0",
        "installed_distribution": "vllm-omni 0.25.0rc2.dev62+g9c1b7504b",
        "path": "vllm_omni/" + PIPELINE_PATH,
        "original_sha256": PIPELINE_SOURCE_SHA256,
        "patched_sha256": PIPELINE_PATCHED_SHA256,
        "tokenizer_type": "qwen2",
        "tokenizer_class": "Qwen2Tokenizer",
        "model": COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
        "model_revision": COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "behavior": "explicit published tokenizer type; original prompt and token IDs preserved",
    }
    value["patch_sha256"] = _digest_document(value)
    return value


def patch_pipeline_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != PIPELINE_SOURCE_SHA256:
        raise PaidfGuardrailError("EVG tokenizer pipeline differs from the reviewed image")
    source = original.decode("utf-8")
    if source.count(TOKENIZER_CALL) != 1:
        raise PaidfGuardrailError("EVG tokenizer pipeline lacks its reviewed call")
    replacement = TOKENIZER_CALL.replace(
        "            model_path,\n",
        '            model_path,\n            tokenizer_type="qwen2",\n',
        1,
    )
    patched = source.replace(TOKENIZER_CALL, replacement, 1).encode()
    if hashlib.sha256(patched).hexdigest() != PIPELINE_PATCHED_SHA256:
        raise PaidfGuardrailError("EVG tokenizer adaptation differs from its reviewed bytes")
    return patched


def _package_files(package: Path, *, installed: bool = False) -> dict[str, bytes]:
    if any(path.is_symlink() for path in (package, *package.parents)) or not package.is_dir():
        raise PaidfGuardrailError("EVG tokenizer package is missing or redirected")
    files = {}
    for path in package.rglob("*"):
        relative = path.relative_to(package)
        if installed and ("__pycache__" in relative.parts or path.suffix == ".pyc"):
            continue
        if path.is_symlink():
            raise PaidfGuardrailError("EVG tokenizer package contains a source redirect")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PaidfGuardrailError("EVG tokenizer package contains a non-regular file")
        files[relative.as_posix()] = path.read_bytes()
    if "__init__.py" not in files or PIPELINE_PATH not in files:
        raise PaidfGuardrailError("EVG tokenizer package is incomplete")
    return files


def prepare_evg_tokenizer_overlay(
    home: Path, hub: Path, *, source_package: Path = VENDOR_PACKAGE
) -> Path:
    """Select the published tokenizer without modifying installed vendor bytes.

    Transformers 5.13 otherwise resolves a model config before its tokenizer
    config. The exact Cosmos3 snapshot has no text_tokenizer/config.json. Its
    tokenizer_config explicitly selects Qwen2Tokenizer, whose supported explicit
    type branch preserves the original files, prompt method and token IDs.
    """
    package = hub / ("models--" + COSMOS3_SUPER_IMAGE2VIDEO_MODEL.replace("/", "--"))
    snapshot = package / "snapshots" / COSMOS3_SUPER_IMAGE2VIDEO_REVISION
    config = snapshot / "text_tokenizer/tokenizer_config.json"
    try:
        if any(path.is_symlink() for path in (config.parent, *config.parent.parents)):
            raise PaidfGuardrailError("EVG tokenizer configuration has a directory redirect")
        resolved = config.resolve(strict=True)
        if not resolved.is_file() or not (
            resolved.is_relative_to(snapshot) or resolved.parent == package / "blobs"
        ):
            raise PaidfGuardrailError("EVG tokenizer configuration has an unsafe cache target")
        payload = config.read_bytes()
        configuration = json.loads(payload)
    except PaidfGuardrailError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise PaidfGuardrailError("EVG tokenizer configuration is missing or malformed") from exc
    if (
        hashlib.sha256(payload).hexdigest() != MODEL_CONFIG_SHA256
        or not isinstance(configuration, dict)
        or configuration.get("tokenizer_class") != "Qwen2Tokenizer"
    ):
        raise PaidfGuardrailError("EVG tokenizer configuration differs from its reviewed model")
    original = _package_files(source_package, installed=True)
    files = dict(original)
    files[PIPELINE_PATH] = patch_pipeline_source(files[PIPELINE_PATH])
    root = home / "npa-paidf-tokenizer-code"
    destination = root / PIPELINE_PATCHED_SHA256
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise PaidfGuardrailError("EVG tokenizer overlay contains a directory redirect")
    root.mkdir(mode=0o700, exist_ok=True)
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(dir=root))
        try:
            for relative, content in files.items():
                output = staging / "vllm_omni" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
                output.chmod(0o444)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if (
        {path.name for path in destination.iterdir()} != {"vllm_omni"}
        or _package_files(destination / "vllm_omni") != files
        or _package_files(source_package, installed=True) != original
    ):
        raise PaidfGuardrailError("EVG tokenizer overlay differs from the reviewed adaptation")
    return destination

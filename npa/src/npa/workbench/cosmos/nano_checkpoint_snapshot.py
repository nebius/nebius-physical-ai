"""Bind native Nano inference to immutable checkpoint bytes and effective configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_REPOSITORY = "nvidia/Cosmos3-Nano"
_BINDING_SCHEMA = "npa.cosmos3.nano-model-binding.v1"
_MATERIALIZATION_SCHEMA = "npa.cosmos3.nano-materialization.v1"
_SELECTION_SCHEMA = "npa.cosmos3.native-model-selection.v1"
_NANO_CONFIG = "cosmos_framework/inference/configs/model/Cosmos3-Nano.yaml"
_WAN_URI = "s3://bucket/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
_SOUND_URI = "s3://bucket/pretrained/tokenizers/audio/avae"
_SOUND_CHECKPOINT = "avae_48k_noncausal_25hz_64ch.ckpt"
_SOUND_CONFIG = "avae_48k_noncausal_25hz_64ch.json"
_SOURCES = (
    "cosmos_framework/inference/args.py",
    "cosmos_framework/inference/common/args.py",
    "cosmos_framework/inference/common/checkpoints.py",
    "cosmos_framework/inference/common/config.py",
    "cosmos_framework/inference/common/public_model_config.py",
    "cosmos_framework/inference/inference.py",
    "cosmos_framework/inference/model.py",
    "cosmos_framework/model/generator/tokenizers/audio/avae.py",
    "cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py",
    "cosmos_framework/utils/checkpoint_db.py",
    _NANO_CONFIG,
)
_PROCESSOR_FILES = (
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


class NanoCheckpointError(ValueError):
    """Reject an incomplete or inconsistent immutable native checkpoint binding.

    Args:
        message: Non-sensitive description of the violated checkpoint contract.
    Returns:
        None.
    Raises:
        None.
    """


def _revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", str(value)):
        raise NanoCheckpointError(
            "Nano checkpoint revision must be a full lowercase SHA"
        )
    return value


def _model_info(repository: str, revision: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi().model_info(repository, revision=revision, files_metadata=True)


def resolve_nano_revision() -> str:
    """Resolve Nano's public main reference to an immutable revision without weights.

    Args:
        None; Hugging Face uses the framework process's existing authentication.
    Returns:
        The actual full revision reported by Hugging Face.
    Raises:
        NanoCheckpointError: The reported revision is not immutable.
        OSError: Model metadata cannot be read.
    """
    return _revision(_model_info(_REPOSITORY, "main").sha)


def _native_checkpoint_support() -> tuple[Any, Any]:
    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import CheckpointConfig, CheckpointDirHf

    register_checkpoints()
    return CheckpointConfig, CheckpointDirHf


def _config_path() -> Path:
    from cosmos_framework.inference.common.config import CONFIG_DIR

    return Path(CONFIG_DIR) / "model/Cosmos3-Nano.yaml"


def _registered_sound(revision: str) -> Any:
    from cosmos_framework.inference.common.checkpoints import _materialize_avae_ckpt

    checkpoint_class, _ = _native_checkpoint_support()
    registered = checkpoint_class.from_uri(_SOUND_URI)
    if (
        registered.hf.repository != _REPOSITORY
        or registered.hf.subdirectory != "sound_tokenizer"
        or registered.post_download is not _materialize_avae_ckpt
    ):
        raise NanoCheckpointError("Native default sound checkpoint contract changed")
    pinned_values = registered.hf.model_dump() | {"revision": revision}
    pinned_hf = type(registered.hf).model_validate(pinned_values)
    return registered.model_copy(update={"hf": pinned_hf})


def _materialized_paths(revision: str) -> dict[str, str]:
    checkpoint_class, directory_class = _native_checkpoint_support()
    snapshot = Path(
        directory_class(repository=_REPOSITORY, revision=revision).download()
    )
    if (
        snapshot.name != revision
        or not snapshot.is_dir()
        or (snapshot / "model").exists()
    ):
        raise NanoCheckpointError(
            "Native Nano snapshot root does not match its revision"
        )
    sound = Path(_registered_sound(revision).download())
    if sound.resolve() != (snapshot / "sound_tokenizer").resolve():
        raise NanoCheckpointError(
            "Pinned default sound checkpoint selected another snapshot"
        )
    wan = checkpoint_class.from_uri(_WAN_URI)
    _revision(wan.hf.revision)
    config = _config_path().absolute()
    return {
        "snapshot_root": str(snapshot.absolute()),
        "sound_directory": str(sound),
        "wan_file": str(Path(wan.download()).absolute()),
        "config_file": str(config),
        "framework_root": str(config.parents[4]),
    }


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_hash(path: Path) -> dict[str, Any]:
    before = _file_identity(path)
    if before[2] <= 0 or not path.is_file():
        raise NanoCheckpointError(
            "A selected native checkpoint file is empty or absent"
        )
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(
        b"blob " + str(before[2]).encode() + b"\0", usedforsecurity=False
    )
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            git_digest.update(block)
    if before != _file_identity(path):
        raise NanoCheckpointError(
            "A selected native checkpoint file changed during hashing"
        )
    return {
        "sha256": digest.hexdigest(),
        "bytes": before[2],
        "git_blob": git_digest.hexdigest(),
    }


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\0" in value
        or path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise NanoCheckpointError(
            "Native checkpoint index contains an unsafe relative path"
        )
    return value


def _index(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    weights = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise NanoCheckpointError("Native checkpoint index has no weight map")
    for name, relative in weights.items():
        if not isinstance(name, str) or not name or not isinstance(relative, str):
            raise NanoCheckpointError(
                "Native checkpoint index has invalid weight entries"
            )
        _relative_path(relative)
    return weights


def _weight_paths(snapshot: Path) -> list[str]:
    weights = _index(snapshot / "model.safetensors.index.json")
    weights = {
        name: path
        for name, path in weights.items()
        if ".k_norm_und_for_gen." not in name
    }
    for name, relative in _index(
        snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json"
    ).items():
        weights.setdefault(name, "transformer/" + relative)
    selected = sorted(
        {
            path
            for path in weights.values()
            if path.startswith(("transformer/", "vision_encoder/"))
        }
    )
    if not any(path.startswith("transformer/") for path in selected):
        raise NanoCheckpointError("Native Nano checkpoint has no transformer weights")
    if "vision_encoder/model.safetensors" not in selected:
        raise NanoCheckpointError(
            "Native Nano checkpoint has no bundled vision weights"
        )
    return selected


def _snapshot_files(snapshot: Path) -> list[str]:
    processor = [name for name in _PROCESSOR_FILES if (snapshot / name).is_file()]
    if not {"preprocessor_config.json", "tokenizer_config.json"}.issubset(processor):
        raise NanoCheckpointError(
            "Native Nano checkpoint has no complete processor configuration"
        )
    if "tokenizer.json" not in processor and not {"vocab.json", "merges.txt"}.issubset(
        processor
    ):
        raise NanoCheckpointError(
            "Native Nano checkpoint has no complete tokenizer data"
        )
    required = [
        "config.json",
        "model_index.json",
        "model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "sound_tokenizer/config.json",
        "sound_tokenizer/diffusion_pytorch_model.safetensors",
    ]
    return sorted(set(required + processor + _weight_paths(snapshot)))


def _remote_files(repository: str, revision: str) -> dict[str, Any]:
    metadata = _model_info(repository, revision)
    if metadata.sha != revision:
        raise NanoCheckpointError(
            "Hugging Face resolved a different immutable revision"
        )
    files = {item.rfilename: item for item in metadata.siblings}
    if len(files) != len(metadata.siblings):
        raise NanoCheckpointError(
            "Immutable checkpoint metadata contains duplicate files"
        )
    return files


def _verified_file(path: Path, remote: Any) -> dict[str, Any]:
    hashed = _file_hash(path)
    if remote.size is not None and remote.size != hashed["bytes"]:
        raise NanoCheckpointError(
            "Native checkpoint file size differs from its immutable object"
        )
    if remote.lfs is not None:
        if remote.lfs.sha256 != hashed["sha256"]:
            raise NanoCheckpointError(
                "Native checkpoint file differs from its immutable LFS object"
            )
    elif remote.blob_id != hashed["git_blob"]:
        raise NanoCheckpointError(
            "Native checkpoint file differs from its immutable Git object"
        )
    return {"sha256": hashed["sha256"], "bytes": hashed["bytes"]}


def _snapshot_hashes(snapshot: Path, revision: str) -> dict[str, Any]:
    remote = _remote_files(_REPOSITORY, revision)
    if set(_snapshot_files(snapshot)) - remote.keys():
        raise NanoCheckpointError(
            "Selected native checkpoint file is absent from its immutable revision"
        )
    hashes = {}
    for name in sorted(remote):
        _relative_path(name)
        hashes["nano/" + name] = _verified_file(snapshot / name, remote[name])
    return hashes


def _wan_binding(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    checkpoint_class, _ = _native_checkpoint_support()
    checkpoint = checkpoint_class.from_uri(_WAN_URI)
    revision = _revision(checkpoint.hf.revision)
    filename = _relative_path(checkpoint.hf.filename)
    if Path(checkpoint.download()).resolve() != path.resolve():
        raise NanoCheckpointError(
            "Native Wan checkpoint selected a different local file"
        )
    remote = _remote_files(checkpoint.hf.repository, revision)
    if filename not in remote:
        raise NanoCheckpointError(
            "Registered native Wan file is absent from its immutable revision"
        )
    binding = {
        "repository": checkpoint.hf.repository,
        "revision": revision,
        "filename": filename,
    }
    return binding, _verified_file(path, remote[filename])


def _sound_state(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file
    from cosmos_framework.inference.common.checkpoints import _avae_block_key_to_legacy

    source = load_file(
        str(directory / "diffusion_pytorch_model.safetensors"), device="cpu"
    )
    state = _derived_sound_state(directory / _SOUND_CHECKPOINT)
    blocks = [
        int(match.group(1))
        for key in source
        if (match := re.fullmatch(r"decoder\.block\.(\d+)\..+", key))
    ]
    if not blocks:
        raise NanoCheckpointError(
            "Native sound source has no converter-compatible decoder blocks"
        )
    expected = {}
    for name, value in source.items():
        key = _avae_block_key_to_legacy(name, max(blocks) + 1)
        if key.endswith((".alpha", ".beta")) and value.ndim == 3:
            value = value.reshape(-1).contiguous()
        expected[key] = value
    return expected, state


def _derived_sound_state(path: Path) -> dict[str, Any]:
    import torch

    try:
        derived = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise NanoCheckpointError(
            "Native sound checkpoint cannot be decoded safely"
        ) from error
    state = derived.get("state_dict") if isinstance(derived, dict) else None
    if not isinstance(state, dict):
        raise NanoCheckpointError("Native sound checkpoint has no decoded state dict")
    return state


def _sound_tensor_hash(directory: Path) -> str:
    import torch

    expected, actual = _sound_state(directory)
    if expected.keys() != actual.keys():
        raise NanoCheckpointError(
            "Native sound conversion changed the selected tensor keys"
        )
    digest = hashlib.sha256()
    for name in sorted(actual):
        value = actual[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != expected[name].dtype
            or value.shape != expected[name].shape
            or not torch.equal(value, expected[name])
        ):
            raise NanoCheckpointError(
                "Native sound conversion changed a selected tensor"
            )
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _sound_conversion(directory: Path, sources: Mapping[str, str]) -> dict[str, str]:
    source_config = _file_hash(directory / "config.json")
    derived_config = _file_hash(directory / _SOUND_CONFIG)
    if source_config["sha256"] != derived_config["sha256"]:
        raise NanoCheckpointError(
            "Native sound conversion changed its selected configuration"
        )
    return {
        "native_converter_sha256": sources[
            "cosmos_framework/inference/common/checkpoints.py"
        ],
        "derived_config_sha256": derived_config["sha256"],
        "state_dict_sha256": _sound_tensor_hash(directory),
    }


def _binding(paths: Mapping[str, str], revision: str) -> dict[str, Any]:
    framework = Path(paths["framework_root"])
    sources = {
        name: _file_hash(framework / name)["sha256"] for name in sorted(_SOURCES)
    }
    if Path(paths["config_file"]).resolve() != (framework / _NANO_CONFIG).resolve():
        raise NanoCheckpointError(
            "Native Nano configuration selected another source file"
        )
    snapshot = Path(paths["snapshot_root"])
    files = _snapshot_hashes(snapshot, revision)
    wan, wan_hash = _wan_binding(Path(paths["wan_file"]))
    files["wan/" + wan["filename"]] = wan_hash
    return {
        "schema": _BINDING_SCHEMA,
        "repository": _REPOSITORY,
        "revision": revision,
        "files": dict(sorted(files.items())),
        "native_sources": sources,
        "setup_config": {"path": _NANO_CONFIG, "sha256": sources[_NANO_CONFIG]},
        "wan_vae": wan,
        "sound_conversion": _sound_conversion(Path(paths["sound_directory"]), sources),
    }


def materialize_nano_snapshot(revision: str) -> dict[str, Any]:
    """Download and verify native Nano and auxiliary bytes without loading a GPU model.

    Args:
        revision: Exact lowercase forty-character Nano revision.
    Returns:
        Portable model binding and private local paths for native preparation.
    Raises:
        NanoCheckpointError: Selected bytes, indices, or native conversion disagree.
        OSError: Source, checkpoint, or metadata access fails.
    """
    revision = _revision(revision)
    paths = _materialized_paths(revision)
    binding = _binding(paths, revision)
    return {
        "schema": _MATERIALIZATION_SCHEMA,
        "model_binding": binding,
        "local_paths": paths,
        "derived_file_hashes": {
            "sound_checkpoint": _file_hash(
                Path(paths["sound_directory"]) / _SOUND_CHECKPOINT
            )["sha256"]
        },
    }


def prepare_native_setup(setup_overrides: Any, revision: str) -> dict[str, Any]:
    """Select verified immutable Nano bytes through the existing native setup API.

    Args:
        setup_overrides: Native parsed setup overrides, before build_setup.
        revision: Exact immutable Nano revision selected for this batch.
    Returns:
        Verified preparation receipt used after ordinary native model creation.
    Raises:
        NanoCheckpointError: Immutable checkpoint preparation is inconsistent.
        OSError: Required native files cannot be read.
    """
    if setup_overrides.checkpoint_path != "Cosmos3-Nano":
        raise NanoCheckpointError(
            "Immutable Nano preparation requires the named Nano checkpoint"
        )
    preparation = materialize_nano_snapshot(revision)
    paths = preparation["local_paths"]
    setup_overrides.checkpoint_path = paths["snapshot_root"]
    setup_overrides.config_file = paths["config_file"]
    setup_overrides.experiment_overrides.append(
        "model.config.sound_tokenizer.from_checkpoint=true"
    )
    setup_overrides.use_torch_compile = False
    return preparation


def _verify_setup(setup: Any, paths: Mapping[str, str]) -> None:
    if (
        str(setup.checkpoint_type) != "hf"
        or setup.checkpoint_hf is not None
        or Path(setup.download_checkpoint()).resolve()
        != Path(paths["snapshot_root"]).resolve()
        or Path(setup.config_file).resolve() != Path(paths["config_file"]).resolve()
        or setup.use_torch_compile
    ):
        raise NanoCheckpointError(
            "Native model setup differs from its immutable preparation"
        )


def _portable_config(value: Any, paths: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable_config(item, paths) for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_portable_config(item, paths) for item in value]
    if isinstance(value, str):
        for key in ("snapshot_root", "framework_root", "wan_file"):
            path = paths[key]
            if value == path or value.startswith(path + "/"):
                return "binding://" + key + value[len(path) :]
    return value


def _effective_config(pipe: Any, paths: Mapping[str, str]) -> dict[str, Any]:
    from cosmos_framework.inference.common.config import unstructure_config

    actual = unstructure_config(pipe.model_config, invalid="error")
    sound = actual.get("sound_tokenizer") if isinstance(actual, dict) else None
    expected = str(Path(paths["sound_directory"]) / _SOUND_CHECKPOINT)
    if (
        not isinstance(sound, dict)
        or sound.get("bucket_name") != ""
        or Path(str(sound.get("avae_path", ""))).resolve() != Path(expected).resolve()
        or sound.get("from_checkpoint") is not None
    ):
        raise NanoCheckpointError(
            "Native model selected another sound checkpoint configuration"
        )
    tokenizer = actual.get("vlm_config", {}).get("tokenizer", {})
    if tokenizer.get("tokenizer_type") != paths["snapshot_root"]:
        raise NanoCheckpointError("Native model selected another processor snapshot")
    if actual.get("compile", {}).get("enabled") is not False:
        raise NanoCheckpointError(
            "Native model selected another compilation configuration"
        )
    return _portable_config(actual, paths)


def _write_receipt(payload: Mapping[str, Any], receipt_path: str | Path) -> None:
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selection_receipt(
    binding: Mapping[str, Any], effective: dict[str, Any]
) -> dict[str, Any]:
    encoded = json.dumps(
        effective, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "schema": _SELECTION_SCHEMA,
        "model_binding": dict(binding),
        "effective_config": effective,
        "effective_config_sha256": hashlib.sha256(encoded).hexdigest(),
        "native_runtime": {
            "selection_basis": "native-config-and-pinned-resolver-contract",
            "loader_open_observed": False,
            "checkpoint_type": "hf",
            "torch_compile": False,
        },
    }


def record_native_model_selection(
    setup: Any,
    pipe: Any,
    preparation: Mapping[str, Any],
    *,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Record verified selected bytes and actual configuration after native model creation.

    Args:
        setup: Resolved native setup passed unchanged to model creation.
        pipe: Successfully created native inference pipeline.
        preparation: Verified immutable checkpoint preparation receipt.
        receipt_path: Exact local destination for the atomically written receipt.
    Returns:
        Portable model selection receipt suitable for durable variant publication.
    Raises:
        NanoCheckpointError: Actual selection or reread bytes disagree with preparation.
        OSError: Required selected files or the receipt cannot be accessed.
    """
    paths = preparation["local_paths"]
    _verify_setup(setup, paths)
    binding = _binding(paths, _revision(preparation["model_binding"]["revision"]))
    if binding != preparation["model_binding"]:
        raise NanoCheckpointError(
            "Native checkpoint bytes changed during model construction"
        )
    receipt = _selection_receipt(binding, _effective_config(pipe, paths))
    _write_receipt(receipt, receipt_path)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Resolve or materialize a checkpoint in the existing native framework interpreter.

    Args:
        argv: Internal subprocess arguments; defaults to sys.argv.
    Returns:
        Zero for verified receipt delivery, or one with a safe validation failure receipt.
    Raises:
        OSError: Metadata, checkpoint, or receipt access fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("resolve", "materialize"), required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--receipt-path", required=True)
    args = parser.parse_args(argv)
    try:
        payload = _snapshot_payload(args.mode, args.revision)
    except NanoCheckpointError as error:
        _write_receipt(
            {"schema": "npa.cosmos3.nano-failure.v1", "reason": str(error)},
            args.receipt_path,
        )
        return 1
    _write_receipt(payload, args.receipt_path)
    return 0


def _snapshot_payload(mode: str, revision: str) -> dict[str, Any]:
    if mode == "resolve":
        return {
            "schema": "npa.cosmos3.nano-revision.v1",
            "repository": _REPOSITORY,
            "revision": resolve_nano_revision(),
        }
    return materialize_nano_snapshot(revision)


if __name__ == "__main__":
    raise SystemExit(main())

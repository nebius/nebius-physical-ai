"""Capture private evidence from the actual Isaac-GR00T policy-server process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import runpy
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


EVIDENCE_SCHEMA = "npa.groot_libero_x.native_policy_server_evidence.v1"
CHECKPOINT_SUFFIXES = frozenset({".bin", ".safetensors"})


def _sha256_file(path: Path) -> str:
    """Hash one regular checkpoint or source file.

    Args:
        path: Existing regular file to hash.

    Returns:
        Lower-case SHA-256 digest.

    Raises:
        RuntimeError: If the path is not a regular file.
    """

    if not path.is_file():
        raise RuntimeError("native evidence target is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_inventory(model_path: Path) -> list[dict[str, str | int]]:
    """Return digest-bound checkpoint files below the selected model directory.

    Args:
        model_path: Model directory used by the native policy server.

    Returns:
        Sorted relative checkpoint identities.

    Raises:
        RuntimeError: If no checkpoint files exist beneath the selected model path.
    """

    root = model_path.resolve()
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix in CHECKPOINT_SUFFIXES
    )
    if not files:
        raise RuntimeError("selected policy model has no checkpoint files")
    return [
        {
            "path": item.relative_to(root).as_posix(),
            "sha256": _sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    ]


def _require_checkpoint_inventory(value: Any) -> list[dict[str, str | int]]:
    """Validate an expected checkpoint inventory from the parent evaluator.

    Args:
        value: Untrusted parsed JSON value.

    Returns:
        Canonical expected checkpoint rows.

    Raises:
        RuntimeError: If a row is incomplete, unsafe, or not digest-bound.
    """

    if not isinstance(value, list) or not value:
        raise RuntimeError("native evidence input lacks checkpoint digest records")
    rows: list[dict[str, str | int]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeError("native evidence checkpoint record is not an object")
        relative = PurePosixPath(str(item.get("path") or ""))
        digest = str(item.get("sha256") or "").lower()
        size = item.get("bytes")
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix not in CHECKPOINT_SUFFIXES
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 1
        ):
            raise RuntimeError("native evidence checkpoint record is invalid")
        rows.append({"path": relative.as_posix(), "sha256": digest, "bytes": size})
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise RuntimeError("native evidence checkpoint paths are not unique")
    return sorted(rows, key=lambda row: str(row["path"]))


def _selected_tensor_device(model: Any, torch_module: Any) -> dict[str, Any]:
    """Read actual CUDA and model-tensor placement from a loaded policy model.

    Args:
        model: Loaded policy model exposing parameters and buffers.
        torch_module: Imported torch module from the policy-server interpreter.

    Returns:
        Observed CUDA and selected-model device facts.

    Raises:
        RuntimeError: If CUDA or a single CUDA model-tensor device is unavailable.
    """

    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError("native policy server did not observe CUDA availability")
    tensors = [*model.parameters(), *model.buffers()]
    devices = {str(tensor.device) for tensor in tensors}
    if not devices:
        raise RuntimeError("native policy model has no observable tensor device")
    if len(devices) != 1:
        raise RuntimeError("native policy model tensors span multiple devices")
    selected = next(iter(devices))
    device = torch_module.device(selected)
    if device.type != "cuda":
        raise RuntimeError("native policy model tensors are not on CUDA")
    index = torch_module.cuda.current_device() if device.index is None else device.index
    properties = torch_module.cuda.get_device_properties(index)
    return {
        "cuda_available": True,
        "cuda_runtime": str(torch_module.version.cuda or ""),
        "device_count": int(torch_module.cuda.device_count()),
        "selected_model_tensor_device": selected,
        "selected_device_index": int(index),
        "selected_device_name": str(properties.name),
        "selected_device_capability": [int(properties.major), int(properties.minor)],
        "visible_device_selection": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "torch_version": str(torch_module.__version__),
    }


def _module_record(module_name: str) -> dict[str, str]:
    """Capture the resolved path and content digest of one imported module.

    Args:
        module_name: Importable module that participated in native execution.

    Returns:
        Module name, resolved file path, and SHA-256 digest.

    Raises:
        RuntimeError: If the module has no regular source file.
    """

    module = importlib.import_module(module_name)
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    return {"module": module_name, "path": str(path), "sha256": _sha256_file(path)}


def _option_value(argv: list[str], option: str) -> str:
    """Read exactly one required forwarded upstream command-line option.

    Args:
        argv: Upstream server arguments after wrapper-only options were removed.
        option: Required option name.

    Returns:
        The sole non-empty option value.

    Raises:
        RuntimeError: If the option is absent, repeated, or lacks a value.
    """

    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1:
        raise RuntimeError(f"native policy server option is not unique: {option}")
    index = positions[0]
    if index + 1 >= len(argv) or not argv[index + 1].strip():
        raise RuntimeError(f"native policy server option lacks a value: {option}")
    return argv[index + 1]


def _load_evidence_input(path: Path) -> dict[str, Any]:
    """Load the parent evaluator's model provenance contract.

    Args:
        path: Private JSON file written beside the native evaluator config.

    Returns:
        Parsed evidence input with a validated model identity.

    Raises:
        RuntimeError: If the input is unreadable or incomplete.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native policy evidence input is unreadable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("native policy evidence input lacks model identity")
    model = dict(payload["model"])
    model["checkpoint_files"] = _require_checkpoint_inventory(
        model.get("checkpoint_files")
    )
    payload["model"] = model
    return payload


def _write_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one private policy-server evidence document.

    Args:
        path: Requested private evidence destination.
        payload: Fully validated evidence payload.

    Returns:
        None.

    Raises:
        RuntimeError: If the destination cannot be published atomically.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except OSError as exc:
        raise RuntimeError("native policy evidence could not be published") from exc


def _capture_policy_evidence(
    *,
    model: Any,
    torch_module: Any,
    model_path: Path,
    expected: Mapping[str, Any],
    upstream_server: Path,
) -> dict[str, Any]:
    """Build evidence only after the upstream policy constructor completed.

    Args:
        model: Actual loaded policy model from ``Gr00tPolicy``.
        torch_module: Imported torch module from the policy-server process.
        model_path: Model directory forwarded to the upstream server.
        expected: Parent evaluator's digest-bound model identity.
        upstream_server: Exact upstream entrypoint executed by this wrapper.

    Returns:
        Complete private policy-server evidence.

    Raises:
        RuntimeError: If checkpoint bytes or native device evidence conflict.
    """

    actual_checkpoints = _checkpoint_inventory(model_path)
    if actual_checkpoints != expected["model"]["checkpoint_files"]:
        raise RuntimeError("loaded policy checkpoint bytes differ from parent evidence")
    return {
        "schema": EVIDENCE_SCHEMA,
        "model_load_completed": True,
        "model": dict(expected["model"]) | {"checkpoint_files": actual_checkpoints},
        "device": _selected_tensor_device(model, torch_module),
        "executed_entrypoint": {
            "path": str(upstream_server.resolve()),
            "sha256": _sha256_file(upstream_server),
        },
        "imported_modules": [
            _module_record("gr00t"),
            _module_record("gr00t.policy.gr00t_policy"),
        ],
        "python": {"executable": sys.executable, "version": sys.version},
    }


def _run_server(args: argparse.Namespace, forwarded: list[str]) -> None:
    """Patch the loaded policy constructor, capture evidence, then run upstream.

    Args:
        args: Wrapper-only paths parsed from this process command line.
        forwarded: Unmodified arguments accepted by the upstream server.

    Returns:
        None.

    Raises:
        RuntimeError: If upstream model construction does not yield native evidence.
    """

    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    evidence_input = _load_evidence_input(Path(args.npa_evidence_input))
    model_path = Path(_option_value(forwarded, "--model-path")).resolve()
    upstream_server = Path(args.npa_upstream_server).resolve()
    original_init = Gr00tPolicy.__init__

    def instrumented_init(instance: Any, *init_args: Any, **init_kwargs: Any) -> None:
        original_init(instance, *init_args, **init_kwargs)
        _write_evidence(
            Path(args.npa_evidence_output),
            _capture_policy_evidence(
                model=instance.model,
                torch_module=torch,
                model_path=model_path,
                expected=evidence_input,
                upstream_server=upstream_server,
            ),
        )

    Gr00tPolicy.__init__ = instrumented_init
    sys.argv = [str(upstream_server), *forwarded]
    runpy.run_path(str(upstream_server), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    """Run the upstream server with mandatory loaded-policy provenance capture.

    Args:
        argv: Optional command line used by tests or direct execution.

    Returns:
        Zero after the upstream server exits.

    Raises:
        RuntimeError: If the server cannot prove its loaded model and device state.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--npa-evidence-input", required=True)
    parser.add_argument("--npa-evidence-output", required=True)
    parser.add_argument("--npa-upstream-server", required=True)
    args, forwarded = parser.parse_known_args(argv)
    _run_server(args, forwarded)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

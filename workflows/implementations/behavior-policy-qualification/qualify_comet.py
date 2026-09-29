"""Qualify one Comet profile with two fresh-process actions on frozen input bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).parent
REPO_ADAPTER_ROOT = (
    Path(__file__).parents[3] / "npa/src/npa/workflows/behavior_challenge"
)
for candidate in (SCRIPT_ROOT, REPO_ADAPTER_ROOT):
    if (candidate / "comet_policy.py").is_file():
        sys.path.insert(0, str(candidate))
        break
import comet_policy

INVENTORY_SHA256 = {
    "comet12": "2ec0c083c0cfdacdf7a7ba02fe2185f76e8d91c9aafea88469042806bd131fda",
    "comet50": "3498c1d4f1fc5dbb9bb391554d7d794085c94b0f07e817a5a60700f06c612087",
}
FETCH_SCHEMAS = {
    "comet12": "npa.behavior.private-comet-fetch.v1",
    "comet50": "npa.behavior.private-comet50-fetch.v1",
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_archive(archive: object) -> dict:
    if not isinstance(archive, dict) or set(archive) != {"sha256", "bytes", "path"}:
        raise ValueError("Private Comet archive identity is absent")
    if not _valid_sha256(archive["sha256"]):
        raise ValueError("Private Comet archive SHA-256 is malformed")
    if type(archive["bytes"]) is not int or archive["bytes"] <= 0:
        raise ValueError("Private Comet archive byte count is malformed")
    if not isinstance(archive["path"], str) or not archive["path"]:
        raise ValueError("Private Comet archive path is malformed")
    return archive


def _verify_fetch_receipt(path: Path, profile: comet_policy.CometProfile) -> dict:
    receipt = json.loads(path.read_text())
    expected = {
        "schema": FETCH_SCHEMAS[profile.kind],
        "status": "bytes_verified_not_gpu_qualified",
        "source_commit": comet_policy.SOURCE_COMMIT,
        "model_repository": comet_policy.MODEL_REPOSITORY,
        "model_revision": profile.revision,
        "inventory_sha256": INVENTORY_SHA256[profile.kind],
        "files": profile.file_count,
        "checkpoint_bytes": profile.total_bytes,
        "readback_verified": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Private Comet fetch receipt differs")
    _verify_archive(receipt.get("archive"))
    return receipt


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(comet_policy.COMET_PROFILES), required=True
    )
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fetch-receipt", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--expected-observation-sha256", required=True)
    parser.add_argument("--expected-action-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
    ):
        environment.pop(name, None)
    adapter_root = str(Path(comet_policy.__file__).parent)
    environment["PYTHONPATH"] = os.pathsep.join((str(SCRIPT_ROOT), adapter_root))
    return environment


def _run_process(args: argparse.Namespace, index: int) -> tuple[dict, float]:
    result = args.output / f"process-{index}.json"
    action = args.output / f"action-{index}.npy"
    command = [
        str(args.runtime_python),
        str(SCRIPT_ROOT / "process_action.py"),
        "--profile",
        args.profile,
        "--source-root",
        str(args.source_root),
        "--checkpoint",
        str(args.checkpoint),
        "--observation",
        str(args.observation),
        "--action",
        str(action),
        "--result",
        str(result),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command, env=_child_environment(), capture_output=True, text=True, check=False
    )
    (args.output / f"process-{index}.log").write_text(
        completed.stdout + completed.stderr
    )
    if completed.returncode != 0 or not result.is_file() or not action.is_file():
        raise RuntimeError(f"Comet qualification process {index} failed")
    return json.loads(result.read_text()), time.monotonic() - started


def _action_bytes(path: Path) -> tuple[np.ndarray, bytes]:
    action = np.load(path, allow_pickle=False)
    if action.shape != (23,) or not np.issubdtype(action.dtype, np.floating):
        raise ValueError("First action requires one floating 23-vector")
    if not np.isfinite(action).all():
        raise ValueError("First action contains a non-finite value")
    return action, action.tobytes(order="C")


def _compare_actions(args: argparse.Namespace, results: list[dict]) -> dict:
    left, left_bytes = _action_bytes(args.output / "action-1.npy")
    right, right_bytes = _action_bytes(args.output / "action-2.npy")
    if left.dtype != right.dtype or left_bytes != right_bytes:
        raise ValueError("Fresh-process first-action bytes differ")
    raw_sha256 = hashlib.sha256(left_bytes).hexdigest()
    if any(
        result.get("action", {}).get("raw_sha256") != raw_sha256 for result in results
    ):
        raise ValueError("Fresh-process action receipts differ")
    if args.expected_action_sha256 and raw_sha256 != args.expected_action_sha256:
        raise ValueError("First action differs from the wrapper-frozen expected bytes")
    return {
        "shape": [23],
        "dtype": str(left.dtype),
        "raw_sha256": raw_sha256,
        "equal_across_two_fresh_processes": True,
        "equal_to_wrapper_expected_action": bool(args.expected_action_sha256),
    }


def _receipt(args, profile, fetch, source, checkpoint, rows, elapsed) -> dict:
    return {
        "schema": "npa.behavior.comet-profile-qualification.v1",
        "status": "policy_load_and_synthetic_action_qualification_passed_not_evaluated",
        "profile": profile.kind,
        "source_commit": comet_policy.SOURCE_COMMIT,
        "source_files": source,
        "model_revision": profile.revision,
        "checkpoint_inventory": {
            "sha256": INVENTORY_SHA256[profile.kind],
            "files": len(checkpoint),
            "bytes": sum(row["size"] for row in checkpoint.values()),
        },
        "fetch_receipt_sha256": _digest(args.fetch_receipt),
        "archive": fetch["archive"],
        "observation_sha256": args.expected_observation_sha256,
        "fresh_processes": rows,
        "process_elapsed_seconds": elapsed,
        "first_action": _compare_actions(args, rows),
        "limitations": [
            "Synthetic bytes prove loader and adapter reproducibility only.",
            "No evaluator, simulator, official case, Q score, or success metric ran.",
            "This smoke does not establish compliance with the official 24 GB policy GPU limit.",
        ],
    }


def main() -> None:
    """Run qualification.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Frozen inputs or action evidence differ.
        RuntimeError: A fresh child process fails.
    """
    args = _arguments()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if not _valid_sha256(args.expected_observation_sha256):
        raise ValueError("Expected observation SHA-256 is malformed")
    if args.expected_action_sha256 and not _valid_sha256(args.expected_action_sha256):
        raise ValueError("Expected action SHA-256 is malformed")
    if _digest(args.observation) != args.expected_observation_sha256:
        raise ValueError("Synthetic observation differs from wrapper-frozen bytes")
    args.output.mkdir(parents=True)
    profile = comet_policy.get_profile(args.profile)
    fetch = _verify_fetch_receipt(args.fetch_receipt, profile)
    source = comet_policy.verify_source(args.source_root)
    checkpoint = comet_policy.verify_checkpoint(args.checkpoint, profile)
    rows, elapsed = [], []
    for index in (1, 2):
        row, duration = _run_process(args, index)
        rows.append(row)
        elapsed.append(duration)
    receipt = _receipt(args, profile, fetch, source, checkpoint, rows, elapsed)
    (args.output / "qualification-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

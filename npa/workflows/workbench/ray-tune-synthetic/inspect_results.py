"""Verify checksum-bound native Ray Tune result summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_ARTIFACTS = frozenset({"result.json", "runtime.json", "trials.json"})
SEARCH_VALUES = (0.1, 0.2, 0.3)


def verify_manifest(directory: Path) -> int:
    """Verify every manifest entry and reject unsafe or unexpected files.

    Args:
        directory: Export directory to inspect.
    Returns:
        Number of verified payload artifacts.
    Raises:
        ValueError: The directory shape, name, or digest is invalid.
        OSError: Artifact bytes cannot be read.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("result path must be a regular directory")
    paths = list(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("result artifacts must be regular files")
    manifest = directory / "SHA256SUMS"
    entries = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
    if any(len(entry) != 2 or Path(entry[1]).name != entry[1] for entry in entries):
        raise ValueError("checksum manifest contains an unsafe name")
    names = {name for _, name in entries}
    if {path.name for path in paths} != names | {"SHA256SUMS"}:
        raise ValueError("checksum manifest does not cover the exact export")
    for expected, name in entries:
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"hash mismatch for {name}")
    return len(entries)


def _load_json(directory: Path, name: str) -> Any:
    """Decode one verified JSON artifact."""
    return json.loads((directory / name).read_text())


def _validate_trials(trials: Any) -> int:
    """Validate the complete synthetic search evidence."""
    if not isinstance(trials, list) or len(trials) != len(SEARCH_VALUES):
        raise ValueError("result does not contain the complete search")
    if any(not isinstance(trial, dict) for trial in trials):
        raise ValueError("trial records must be JSON objects")
    if [trial.get("step_size") for trial in trials] != list(SEARCH_VALUES):
        raise ValueError("trial search values differ")
    for trial in trials:
        expected_loss = (trial["step_size"] - 0.2) ** 2
        loss = trial.get("loss")
        valid = (
            type(loss) in (int, float)
            and math.isfinite(loss)
            and loss == expected_loss
            and trial.get("iterations") == 3
            and type(trial.get("resumed")) is bool
            and trial.get("checkpoint_available") is True
        )
        if not valid:
            raise ValueError("trial metrics or checkpoint evidence are invalid")
    return sum(trial["resumed"] for trial in trials)


def _validate_result(result: Any, trials: list[dict[str, Any]]) -> None:
    """Require the summary to agree with the trial evidence."""
    if not isinstance(result, dict) or result.get("status") != "succeeded":
        raise ValueError("result did not complete on Ray 2.58.0")
    best = min(trials, key=lambda trial: trial["loss"])
    valid = (
        result.get("ray_version") == "2.58.0"
        and result.get("trial_count") == len(trials)
        and result.get("best_step_size") == best["step_size"]
        and result.get("best_loss") == best["loss"]
    )
    if not valid:
        raise ValueError("reported optimum differs from trial evidence")


def inspect(directory: Path) -> dict[str, int]:
    """Validate trial completeness, optimum selection, recovery, and runtime.

    Args:
        directory: Checksum-bound result directory.
    Returns:
        Counts of verified artifacts, trials, and recovered trials.
    Raises:
        ValueError: The result contradicts the declared synthetic contract.
        OSError: Artifact bytes cannot be read.
    """
    artifacts = verify_manifest(directory)
    if {path.name for path in directory.iterdir()} != EXPECTED_ARTIFACTS | {"SHA256SUMS"}:
        raise ValueError("result export contains unexpected artifacts")
    result = _load_json(directory, "result.json")
    runtime = _load_json(directory, "runtime.json")
    trials = _load_json(directory, "trials.json")
    source_path = Path(__file__).with_name("search.py")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if (
        artifacts != len(EXPECTED_ARTIFACTS)
        or not isinstance(runtime, dict)
        or runtime.get("ray") != "2.58.0"
    ):
        raise ValueError("result did not complete on Ray 2.58.0")
    if runtime.get("source_sha256") != source_digest:
        raise ValueError("result source hash differs from the inspector source")
    resumed = _validate_trials(trials)
    _validate_result(result, trials)
    return {"artifacts_verified": artifacts, "trials_verified": len(trials), "resumed_trials": resumed}


def main() -> None:
    """Inspect one result directory and print a JSON receipt.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Result verification fails.
        OSError: Artifact bytes cannot be read.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    print(json.dumps(inspect(parser.parse_args().result_dir), sort_keys=True))


if __name__ == "__main__":
    main()

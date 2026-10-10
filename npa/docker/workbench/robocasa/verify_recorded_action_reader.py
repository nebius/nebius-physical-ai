"""Verify native RoboCasa recorded-action imports and immutable reader semantics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.io_utils import write_info
from robocasa.utils import lerobot_utils


def _write_fixture(root: Path) -> np.ndarray:
    """Write a deliberately reordered action fixture for the native reader."""
    expected = np.arange(24, dtype=np.float64).reshape(2, 12)
    ordering = lerobot_utils.ACTION_KEY_ORDERING_HDF5
    modality = {}
    columns = []
    offset = 0
    for name, (start, end) in reversed(list(ordering.items())):
        width = end - start
        modality[name] = {"start": offset, "end": offset + width}
        columns.append(expected[:, start:end])
        offset += width
    raw = np.concatenate(columns, axis=1)
    (root / "meta").mkdir()
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/modality.json").write_text(json.dumps({"action": modality}))
    pd.DataFrame({"action": list(raw)}).to_parquet(
        root / "data/chunk-000/episode_000000.parquet"
    )
    return expected


def _verify_reader(root: Path, expected: np.ndarray) -> None:
    """Check real reordering plus missing-episode and unsupported-action refusals."""
    actual = lerobot_utils.get_episode_actions(root, 0)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    try:
        lerobot_utils.get_episode_actions(root, 1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("native reader accepted a missing episode")
    try:
        lerobot_utils.get_episode_actions(root, 0, abs_actions=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("native reader accepted unsupported absolute actions")


def main() -> None:
    """Exercise complete native imports, CLI parsing and recorded-action loading.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The native import or reader contract is not preserved.
        subprocess.CalledProcessError: The actual native CLI fails to import.
    """
    assert lerobot_utils.write_info is write_info
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robocasa.scripts.dataset_scripts.playback_dataset",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory(prefix="robocasa-recorded-reader-") as directory:
        root = Path(directory)
        _verify_reader(root, _write_fixture(root))
    print(json.dumps({"native_recorded_action_reader": True, "action_width": 12}))


if __name__ == "__main__":
    main()

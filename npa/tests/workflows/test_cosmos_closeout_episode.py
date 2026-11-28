import importlib.util
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"
spec = importlib.util.spec_from_file_location(
    "inspect_episode", RECIPE / "inspect_episode.py"
)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


def test_nonzero_window_and_timestamp_shift():
    rows = [
        {
            "frame_index": i,
            "timestamp": i / 20,
            "action": [0.0] * 7,
            "observation.state": [0.0] * 8,
        }
        for i in range(22)
    ]
    times, states, actions = inspector.action_window(rows[::-1], 4)
    assert (times[0], len(states), len(actions)) == (0.2, 17, 16)
    rows[10]["timestamp"] += 0.025
    with pytest.raises(ValueError, match="20 Hz frame grid"):
        inspector.action_window(rows, 4)


def test_missing_state_frame():
    with pytest.raises(ValueError, match="17 consecutive"):
        inspector.action_window([], 0)


def test_metadata_symlink_outside_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    outside = tmp_path / "other.json"
    outside.write_text("{}")
    (root / "meta/info.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside the dataset"):
        inspector.inspect(root, 0, 0, tmp_path / "result")
    assert not (tmp_path / "result").exists()

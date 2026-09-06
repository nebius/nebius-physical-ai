from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, IsaacLabLeRobotError, convert


def _write_rollouts(raw: Path) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for index, frames in enumerate((2, 3)):
        episode = raw / f"episode_{index:06d}"
        episode.mkdir(parents=True)
        state = np.arange(frames * G1_STATE_DIM, dtype=np.float32).reshape(
            frames, G1_STATE_DIM
        ) + index * 100
        action = state * -0.5 + 0.25
        np.save(episode / "state.npy", state)
        np.save(episode / "actions.npy", action)
        states.append(state)
        actions.append(action)
    (raw / "meta.json").write_text('{"task": "Synthetic input preservation control"}')
    return np.concatenate(states), np.concatenate(actions)


def _snapshot(root: Path) -> dict[str, bytes | str | None]:
    return {
        str(path.relative_to(root)): (
            str(path.readlink()) if path.is_symlink()
            else path.read_bytes() if path.is_file()
            else None
        )
        for path in root.rglob("*")
    }


@pytest.mark.parametrize(
    "layout",
    [
        "same",
        "output_ancestor",
        "source_episode",
        "new_descendant",
        "relative_alias",
        "input_symlink",
        "output_symlink",
        "symlink_parent_ancestor",
        "symlink_parent_descendant",
    ],
)
def test_overlapping_paths_preserve_all_input_and_sibling_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    raw = tmp_path / "raw"
    _write_rollouts(raw)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "keep.bin").write_bytes(b"unrelated sibling\x00\xff")
    alias = tmp_path / "alias"
    input_dir, output_dir = raw, raw
    if layout == "output_ancestor":
        output_dir = tmp_path
    elif layout == "source_episode":
        output_dir = raw / "episode_000001"
    elif layout == "new_descendant":
        output_dir = raw / "converted"
    elif layout == "relative_alias":
        monkeypatch.chdir(tmp_path)
        input_dir, output_dir = Path("raw"), Path("raw/../raw")
    elif layout == "input_symlink":
        alias.symlink_to(raw, target_is_directory=True)
        input_dir = alias
    elif layout == "output_symlink":
        alias.symlink_to(raw, target_is_directory=True)
        output_dir = alias
    elif layout == "symlink_parent_ancestor":
        alias.symlink_to(tmp_path, target_is_directory=True)
        input_dir, output_dir = alias / "raw", tmp_path
    elif layout == "symlink_parent_descendant":
        alias.symlink_to(raw, target_is_directory=True)
        output_dir = alias / "episode_000001"

    before = _snapshot(tmp_path)
    error = None
    try:
        convert(input_dir, output_dir)
    except (IsaacLabLeRobotError, OSError) as exc:
        error = exc

    # Check bytes and directory/link entries even if conversion already failed.
    assert _snapshot(tmp_path) == before
    assert isinstance(error, IsaacLabLeRobotError)
    assert "overlap" in str(error).lower()


@pytest.mark.parametrize("layout", ["new", "replace", "resolved_disjoint"])
def test_disjoint_conversion_preserves_sources_and_exact_data(
    tmp_path: Path, layout: str
) -> None:
    raw = tmp_path / "raw"
    states, actions = _write_rollouts(raw)
    # A shared name prefix must not be mistaken for directory containment.
    output_dir = tmp_path / "raw-converted"
    if layout == "replace":
        stale = output_dir / "stale" / "old.bin"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"previous output")
    input_dir = raw
    if layout == "resolved_disjoint":
        input_dir = tmp_path / "input-alias"
        input_dir.symlink_to(raw, target_is_directory=True)
        output_parent = tmp_path / "output-parent"
        output_parent.mkdir()
        output_alias = raw / "output-alias"
        output_alias.symlink_to(output_parent, target_is_directory=True)
        output_dir = output_alias / "converted"
    before = _snapshot(raw)

    result = convert(input_dir, output_dir, fps=20)

    assert result == output_dir
    assert _snapshot(raw) == before
    assert not (output_dir / "stale").exists()
    data = pq.read_table(output_dir / "data/chunk-000/file-000.parquet")
    np.testing.assert_array_equal(data["observation.state"].to_pylist(), states)
    np.testing.assert_array_equal(data["action"].to_pylist(), actions)
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1, 1]
    assert data["frame_index"].to_pylist() == [0, 1, 0, 1, 2]
    np.testing.assert_allclose(data["timestamp"].to_pylist(), [0, 0.05, 0, 0.05, 0.1])

"""Exercise the native LeRobot conversion path during the CPU image build."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from npa.adapter.sim_to_lerobot import convert


TASK = "Synthetic image-build conversion smoke"
FRAME_COUNT = 2
IMAGE_SIZE = 16


def _write_synthetic_episode(root: Path) -> None:
    """Write a tiny action-rich episode that is never task evidence."""

    episode = root / "episode_000000"
    episode.mkdir(parents=True)
    workspace = np.zeros((FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    wrist = np.zeros_like(workspace)
    workspace[0, :, :, 0] = 255
    workspace[1, :, :, 1] = 255
    wrist[0, :, :, 2] = 255
    wrist[1, :, :, :2] = 127
    np.save(episode / "obs_workspace.npy", workspace)
    np.save(episode / "obs_wrist.npy", wrist)
    np.save(episode / "state.npy", np.zeros((FRAME_COUNT, 8), dtype=np.float32))
    np.save(episode / "actions.npy", np.ones((FRAME_COUNT, 8), dtype=np.float32))
    np.save(episode / "timestamps.npy", np.array([0.0, 0.2], dtype=np.float64))


def _verify_parquet(dataset: Path) -> None:
    """Read back task indexing plus data and episode row counts."""

    tasks = pq.read_table(dataset / "meta/tasks.parquet")
    data = pq.read_table(dataset / "data/chunk-000/file-000.parquet")
    episodes = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    task_frame = tasks.to_pandas()
    assert tasks["task"].to_pylist() == [TASK]
    assert task_frame.index.tolist() == [TASK]
    assert data.num_rows == FRAME_COUNT
    assert data["task_index"].to_pylist() == [0, 0]
    assert episodes["length"].to_pylist() == [FRAME_COUNT]


def _decode_first_frame(video: Path) -> None:
    """Decode one real MP4 frame and reject empty or malformed video output."""

    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    assert len(result.stdout) == IMAGE_SIZE * IMAGE_SIZE * 3


def main() -> None:
    """Convert and read back synthetic bytes in an automatically removed directory."""

    with tempfile.TemporaryDirectory(prefix="npa-pi05-build-smoke-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        _write_synthetic_episode(source)
        dataset = convert(source, root / "dataset", fps=5, task=TASK)
        _verify_parquet(dataset)
        for camera in ("workspace", "wrist"):
            _decode_first_frame(
                dataset / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            )


if __name__ == "__main__":
    main()

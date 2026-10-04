"""Optional CPU compatibility checks against the installed pinned native FLUX source.

Run with the pinned source on PYTHONPATH; no model downloads or GPU are needed.
This exercises real LeRobot indexing and training-window decoding, not model learning.
"""

import json

import numpy as np
import pytest

from npa.workbench.flux_action.configuration import training_config, validate_dataset
from npa.workbench.flux_action.schemas import Recipe

# These imports require the optional native runtime, which ordinary NPA CI omits.
pytest.importorskip("flux_action")
torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")

index_dataset = pytest.importorskip("flux_action.data.lerobot.index").index_dataset
native_data = pytest.importorskip("flux_action.training.data")
WindowDataset, load_manifest = native_data.WindowDataset, native_data.load_manifest
TrainConfig = pytest.importorskip("flux_action.training.trainer").TrainConfig
PolicyConfig = pytest.importorskip("flux_action.config").PolicyConfig


def write_dataset(root, width):
    names = [f"joint_{i}" for i in range(width)]
    features = {
        key: {"dtype": "float32", "shape": [width], "names": names}
        for key in ("action", "observation.state")
    }
    features["observation.images.front"] = {
        "dtype": "video",
        "shape": [32, 32, 3],
        "names": ["height", "width", "channels"],
    }
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 20,
        "total_episodes": 1,
        "total_frames": 40,
        "chunks_size": 1000,
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(
        '{"task_index": 0, "task": "move the arm"}\n'
    )
    (root / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"length":40,"tasks":["move the arm"]}\n'
    )
    rows = [
        {
            "index": i,
            "frame_index": i,
            "episode_index": 0,
            "task_index": 0,
            "timestamp": np.float32(i / 20),
            "observation.state": [float(i)] * width,
            "action": [float(i + 1)] * width,
        }
        for i in range(40)
    ]
    data = root / "data/chunk-000/episode_000000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data)
    video = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(40):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i, np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return names


@pytest.mark.parametrize("width", [3, 7, 14])
def test_native_index_and_window_for_non_so101_robot(tmp_path, width):
    root = tmp_path / "dataset"
    names = write_dataset(root, width)
    channels = [{"name": name, "unit": "rad"} for name in names]
    recipe = Recipe.model_validate(
        {
            "robot": {
                "embodiment": "test_arm",
                "fps": 20,
                "actions": channels,
                "states": channels,
                "cameras": {"front": "observation.images.front"},
                "camera_layout": "single",
            },
            "training": {
                "steps": 4,
                "checkpoint_every": 4,
                "frozen_steps": 0,
                "trunk_warmup_steps": 1,
                "heads_warmup_steps": 1,
            },
        }
    )
    validate_dataset(root, recipe)
    output = tmp_path / "out"
    index_dataset(
        root,
        output / "index",
        recipe.robot.cameras,
        action_parameterization="absolute",
        absolute_action_dims=(),
        hash_files=True,
    )
    config = TrainConfig(**training_config(recipe, root, output, 1))
    policy = PolicyConfig(**config.policy)
    policy.validate_training()
    policy.validate_inference()
    manifest = load_manifest(output / "index/manifest.json")
    statistics = json.loads((output / "index/statistics.json").read_text())
    assert len(statistics["action"]["q01"]) == width
    assert statistics["action"]["parameterization"] == "absolute"
    dataset = WindowDataset(
        manifest,
        root,
        output / "index/rows.f32.npy",
        seed=42,
        epoch=0,
        rank=0,
        world_size=1,
        num_workers=1,
        windows_per_rank=1,
        frame_hw=(32, 32),
        decoder="pyav",
    )
    sample = next(iter(dataset))
    assert sample["action"].shape == (32, width)
    assert sample["state"].shape == (width,)
    assert sample["images.front"].shape == (33, 3, 32, 32)
    # Independent fixture semantics: each command is one ahead of measured state.
    assert torch.all(sample["action"][0] == sample["state"] + 1)

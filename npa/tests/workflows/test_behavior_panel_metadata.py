"""Check frame identity without decoding expensive multi-camera observations."""

# ruff: noqa: E402

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-matched-training"
)
sys.path.insert(0, str(IMPLEMENTATION))

from panel_data import PanelDataset, episode_boundaries
from stage_conditioning import FrameKey


@pytest.fixture
def panel():
    class MetadataOnlyPanel(PanelDataset):
        def __getitem__(self, index):
            raise AssertionError("Frame identity must not decode observations")

    value = MetadataOnlyPanel.__new__(MetadataOnlyPanel)
    value.task_ids = (0, 1, 22)
    value.offsets = [0, 12, 21]
    value.datasets = [
        SimpleNamespace(
            episodes=[4, 99],
            episode_data_index={"from": np.array([0, 5]), "to": np.array([5, 12])},
        ),
        SimpleNamespace(
            episodes=[205],
            episode_data_index={"from": np.array([0]), "to": np.array([9])},
        ),
        SimpleNamespace(
            episodes=[4402, 4511],
            episode_data_index={"from": np.array([0, 8]), "to": np.array([8, 11])},
        ),
    ]
    return value


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, FrameKey(0, 4, 0)),
        (4, FrameKey(0, 4, 4)),
        (5, FrameKey(0, 99, 0)),
        (11, FrameKey(0, 99, 6)),
        (12, FrameKey(1, 205, 0)),
        (20, FrameKey(1, 205, 8)),
        (21, FrameKey(22, 4402, 0)),
        (29, FrameKey(22, 4511, 0)),
        (31, FrameKey(22, 4511, 2)),
    ],
)
def test_frame_key_uses_metadata_at_task_and_episode_boundaries(panel, index, expected):
    assert panel.key_for_index(index) == expected


@pytest.mark.parametrize("index", [-1, 32, 100])
def test_frame_key_rejects_out_of_range_indices(panel, index):
    with pytest.raises(IndexError):
        panel.key_for_index(index)


def test_decoded_sample_must_match_the_metadata_index(panel):
    sample = {"task_index": np.array(22), "episode_index": np.array(4511)}
    assert panel.key_for_index(30, sample=sample) == FrameKey(22, 4511, 1)
    with pytest.raises(ValueError, match="metadata"):
        panel.key_for_index(28, sample=sample)
    with pytest.raises(ValueError, match="metadata"):
        panel.key_for_index(12, sample=sample)


def test_frame_key_rejects_a_gap_in_episode_metadata(panel):
    panel.datasets[0].episode_data_index["to"][0] = 3
    with pytest.raises(IndexError):
        panel.key_for_index(4)


@pytest.fixture
def filtered_rows():
    columns = {
        "episode_index": np.array([4, 4, 99, 99, 99]),
        "frame_index": np.array([0, 1, 0, 1, 2]),
        "task_index": np.array([1, 1, 1, 1, 1]),
        "index": np.array([10, 11, 50, 51, 52]),
    }
    metadata = {
        4: {"length": 2, "dataset_from_index": 10, "dataset_to_index": 12},
        99: {"length": 3, "dataset_from_index": 50, "dataset_to_index": 53},
    }
    return columns, metadata


def test_filtered_reader_positions_differ_from_absolute_frame_indices(filtered_rows):
    columns, metadata = filtered_rows
    bounds = episode_boundaries(columns, [4, 99], metadata, 1)
    assert bounds["from"].tolist() == [0, 2]
    assert bounds["to"].tolist() == [2, 5]


@pytest.mark.parametrize("column", ["episode_index", "frame_index", "task_index", "index"])
def test_corrupt_native_row_identity_is_rejected(filtered_rows, column):
    columns, metadata = filtered_rows
    columns[column][3] += 1
    with pytest.raises(ValueError):
        episode_boundaries(columns, [4, 99], metadata, 1)


def test_selected_episode_order_must_match_the_frozen_split(filtered_rows):
    columns, metadata = filtered_rows
    with pytest.raises(ValueError, match="order"):
        episode_boundaries(columns, [99, 4], metadata, 1)


def test_frozen_metadata_length_must_match_every_selected_frame(filtered_rows):
    columns, metadata = filtered_rows
    metadata[99]["length"] += 1
    with pytest.raises(ValueError, match="lengths"):
        episode_boundaries(columns, [4, 99], metadata, 1)


def test_absolute_episode_end_must_agree_with_its_length(filtered_rows):
    columns, metadata = filtered_rows
    metadata[99]["dataset_to_index"] += 1
    with pytest.raises(ValueError, match="absolute"):
        episode_boundaries(columns, [4, 99], metadata, 1)


def test_reader_metadata_rejects_fractional_frame_indices(filtered_rows):
    columns, metadata = filtered_rows
    columns["frame_index"] = columns["frame_index"].astype(np.float32)
    columns["frame_index"][3] = 1.5
    with pytest.raises(ValueError, match="integer"):
        episode_boundaries(columns, [4, 99], metadata, 1)

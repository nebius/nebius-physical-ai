"""Keep conversion frame storage bounded while validating the complete input first."""

import json
from pathlib import Path
import weakref

import numpy as np

from npa.adapter import sim_to_lerobot


def _episodes(root, lengths):
    for index, length in enumerate(lengths):
        episode = root / f"episode_{index:04d}"
        episode.mkdir(parents=True)
        for name in ("obs_workspace", "obs_wrist"):
            frames = np.full((length, 8, 10, 3), 100 + index, dtype=np.uint8)
            frames[:, 0, 0] += 1
            np.save(episode / f"{name}.npy", frames)
        for name in ("state", "actions"):
            np.save(episode / f"{name}.npy", np.ones((length, 2)) * index)


def test_preflight_uses_mapped_arrays_and_releases_them(tmp_path, monkeypatch):
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _episodes(raw, [3] * 12)
    load = np.load
    validated, mapped = set(), []

    def observe_load(path, *args, **kwargs):
        array = load(path, *args, **kwargs)
        if not output.exists():
            assert isinstance(array, np.memmap)
            assert not array.flags.writeable
            validated.add(Path(path))
            mapped.append(weakref.ref(array))
        return array

    def encode(*args):
        assert len(validated) == 48
        assert all(reference() is None for reference in mapped)

    monkeypatch.setattr(sim_to_lerobot.np, "load", observe_load)
    monkeypatch.setattr(sim_to_lerobot, "encode_video", encode)
    sim_to_lerobot.convert(raw, output)


def test_conversion_does_not_retain_camera_arrays_across_dataset(tmp_path, monkeypatch):
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _episodes(raw, [3] * 12)
    load = np.load
    cameras, peak = [], []

    def observe_load(path, *args, **kwargs):
        array = load(path, *args, **kwargs)
        if array.ndim == 4:
            cameras.append(weakref.ref(array))
            peak.append(sum(reference() is not None for reference in cameras))
        return array

    monkeypatch.setattr(sim_to_lerobot.np, "load", observe_load)
    monkeypatch.setattr(sim_to_lerobot, "encode_video", lambda *args: None)
    sim_to_lerobot.convert(raw, output)
    # Loading the next episode may overlap the previous one, never the corpus.
    assert max(peak) <= 4
    assert all(reference() is None for reference in cameras)


def test_merged_statistics_weight_unequal_episodes(tmp_path, monkeypatch):
    raw, output = tmp_path / "raw", tmp_path / "dataset"
    _episodes(raw, [2, 5, 3])
    monkeypatch.setattr(sim_to_lerobot, "encode_video", lambda *args: None)
    sim_to_lerobot.convert(raw, output)
    stats = json.loads((output / "meta/stats.json").read_text())
    for filename, feature in (
        ("obs_workspace", "observation.images.workspace"),
        ("obs_wrist", "observation.images.wrist"),
        ("state", "observation.state"),
        ("actions", "action"),
    ):
        values = np.concatenate(
            [np.load(path) for path in sorted(raw.glob(f"*/{filename}.npy"))]
        ).astype(np.float64)
        video = values.ndim == 4
        if video:
            values = values.reshape(-1, 3) / 255.0
        for statistic in ("min", "max", "mean", "std"):
            expected = getattr(values, statistic)(axis=0)
            np.testing.assert_allclose(
                np.asarray(stats[feature][statistic]).reshape(-1),
                expected,
                rtol=1e-10,
                atol=1e-12,
            )
        assert stats[feature]["count"] == [10]

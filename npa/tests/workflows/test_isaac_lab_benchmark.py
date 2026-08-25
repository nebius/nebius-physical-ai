from __future__ import annotations

import pytest

from npa.workflows.isaac_lab_benchmark import compare_records


def _record(generation: str, duration: float, *, reward: float = 1.0) -> dict:
    return {
        "generation": generation,
        "status": "success",
        "task": "Isaac-Cartpole-v0",
        "num_envs": 64,
        "max_iterations": 10,
        "hardware_model": "RTX PRO 6000 Blackwell",
        "duration_seconds": duration,
        "mean_reward": reward,
        "isaac_lab_version": "2.3.2.post1" if generation == "2" else "3.0.0b2.post1",
        "image_digest": "sha256:" + generation * 64,
    }


def test_compare_records_reports_median_matched_measurement() -> None:
    report = compare_records(
        [
            _record("2", 12, reward=3),
            _record("2", 10, reward=5),
            _record("3", 8, reward=6),
            _record("3", 6, reward=8),
        ]
    )
    assert report["baseline"]["median_duration_seconds"] == 11
    assert report["candidate"]["median_duration_seconds"] == 7
    assert report["measured"]["duration_reduction_percent"] == 36.364
    assert report["measured"]["duration_speedup_ratio"] == 1.571429
    assert report["measured"]["median_reward_delta"] == 3


def test_compare_records_rejects_different_hardware() -> None:
    records = [_record("2", 10), _record("3", 8)]
    records[1]["hardware_model"] = "different GPU"
    with pytest.raises(ValueError, match="not matched"):
        compare_records(records)


def test_compare_records_requires_immutable_digest() -> None:
    records = [_record("2", 10), _record("3", 8)]
    records[1]["image_digest"] = "latest"
    with pytest.raises(ValueError, match="immutable image_digest"):
        compare_records(records)

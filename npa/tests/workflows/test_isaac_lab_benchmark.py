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
        "gpu_count": 1,
        "seed": 7,
        "repetition": 1,
        "cache_state": "warm",
        "driver_version": "matched-driver",
        "runtime_version": "matched-cuda-runtime",
        "duration_seconds": duration,
        "mean_reward": reward,
        "isaac_lab_version": "2.3.2.post1" if generation == "2" else "3.0.0b2.post1",
        "image_digest": "sha256:" + generation * 64,
    }


def test_compare_records_reports_median_matched_measurement() -> None:
    records = [
        _record("2", 12, reward=3),
        _record("2", 10, reward=5),
        _record("3", 8, reward=6),
        _record("3", 6, reward=8),
    ]
    for index, item in enumerate(records):
        item["seed"] = 7 + (index % 2)
        item["repetition"] = 1 + (index % 2)
    report = compare_records(records)
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


def test_compare_records_rejects_mixed_digest_within_generation() -> None:
    records = [
        _record("2", 12),
        _record("2", 10),
        _record("3", 8),
        _record("3", 6),
    ]
    for index, item in enumerate(records):
        item["seed"] = 7 + (index % 2)
        item["repetition"] = 1 + (index % 2)
    records[1]["image_digest"] = "sha256:" + "a" * 64

    with pytest.raises(ValueError, match="generation 2 has multiple image digests"):
        compare_records(records)


def test_compare_records_allows_different_digest_between_generations() -> None:
    report = compare_records([_record("2", 10), _record("3", 8)])

    assert report["baseline"]["image_digests"] == ["sha256:" + "2" * 64]
    assert report["candidate"]["image_digests"] == ["sha256:" + "3" * 64]


def test_compare_records_rejects_malformed_digest() -> None:
    records = [_record("2", 10), _record("3", 8)]
    records[0]["image_digest"] = "sha256:" + "g" * 64

    with pytest.raises(ValueError, match="immutable image_digest"):
        compare_records(records)


def test_compare_records_rejects_failed_or_unpaired_campaign() -> None:
    records = [_record("2", 10), _record("3", 8)]
    records[1]["status"] = "failed"
    with pytest.raises(ValueError, match="only successful"):
        compare_records(records)
    records[1]["status"] = "success"
    records[1]["seed"] = 99
    with pytest.raises(ValueError, match="not paired"):
        compare_records(records)

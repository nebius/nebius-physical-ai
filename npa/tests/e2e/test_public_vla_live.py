"""Verify real GPU training and native simulator artifacts collected from a completed public VLA run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from npa.workflows.policy_training.public_vla_export import file_sha256

pytestmark = pytest.mark.e2e


@pytest.fixture
def collected():
    if not os.environ.get("NPA_PUBLIC_VLA_RESULTS"):
        pytest.skip("requires artifacts collected from the public VLA GPU pipeline")
    return Path(os.environ["NPA_PUBLIC_VLA_RESULTS"])


def test_native_training_changed_the_checkpoint(collected):
    evidence = json.loads((collected / "report/evidence.json").read_text())
    export = json.loads((collected / "exported-policy/manifest.json").read_text())
    assert evidence["training_executed"] is True
    assert evidence["runtime"]["parameters"] > 100_000_000
    assert evidence["runtime"]["trainable_parameters"] > 1_000_000
    verification = evidence["export_verification"]
    assert verification["offline_reload"] and verification["cuda_inference"]
    assert verification["finite_action"] and verification["action_shape"] == [1, 7]
    assert verification["parameter_change"]["maximum_absolute_change"] > 0
    assert verification["parameter_change"]["changed_elements"] > 0
    assert export["initial_weights_sha256"] != export["trained_weights_sha256"]
    policy = collected / "exported-policy/policy/model.safetensors"
    assert file_sha256(policy) == export["trained_weights_sha256"]
    assert (
        evidence["selection"]["checkpoint_sha256"] == export["trained_weights_sha256"]
    )
    expected_gate = (
        evidence["selection"]["validation_pass"]
        and evidence["evaluation"]["test"]["pc_success"] / 100
        >= evidence["recipe"]["minimum_success"]
    )
    assert evidence["selection"]["quality_gate_passed"] == expected_gate
    for phase in ("generalist", "specialist"):
        training = evidence["training"][phase]
        assert (
            training["steps"] * evidence["recipe"]["batch_size"] >= training["frames"]
        )
        assert training["metrics"][-1]["step"] == training["steps"]
        assert any(row["grad_norm"] > 0 for row in training["metrics"])


def test_native_evaluation_videos_decode_and_match_recorded_counts(collected):
    import av

    evidence = json.loads((collected / "report/evidence.json").read_text())
    assert evidence["evaluation_executed"] is True
    for result in evidence["evaluation"].values():
        assert result["n_episodes"] == evidence["recipe"]["evaluation_episodes"]
        assert 0 <= result["pc_success"] <= 100
        assert len(result["successes"]) == result["n_episodes"]
        assert result["pc_success"] == pytest.approx(
            100 * sum(result["successes"]) / result["n_episodes"]
        )
        assert len(result["videos"]) == min(10, result["n_episodes"])
        for filename in result["videos"]:
            with av.open(str(collected / "report" / filename)) as video:
                frame = next(video.decode(video=0))
                assert frame.width >= 256 and frame.height >= 256
                assert frame.to_ndarray().std() > 1


def test_public_report_has_no_runtime_locations(collected):
    report = collected / "report"
    for filename in ("evidence.json", "index.html"):
        text = (report / filename).read_text()
        assert "/work/run" not in text
        assert "KUBERNETES_SERVICE_HOST" not in text
        assert "SLURM_JOB_ID" not in text
    checksums = json.loads((report / "checksums.json").read_text())
    for filename, checksum in checksums.items():
        path = (report / filename).resolve()
        assert path.is_relative_to(report.resolve())
        assert file_sha256(path) == checksum

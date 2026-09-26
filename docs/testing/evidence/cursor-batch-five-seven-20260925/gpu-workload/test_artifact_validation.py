"""Exercise validator rejection paths with explicitly synthetic CPU-only fixtures."""

import copy
import json

import pytest
from artifact_validation import _loss_gradient, _oracle, _validate_artifacts
from regression_contract import (
    IMAGE,
    INITIAL,
    RECIPE,
    TORCH_VERSION,
    _dataset,
    _sha256,
    _validate_integrity,
)

BINDING = {"run_id": "unit-fixture", "source_sha": "a" * 40, "payload_sha": "b" * 64}


@pytest.fixture
def evidence():
    parameters, momentum, journal = _oracle()
    for row in journal:
        row.update(elapsed_seconds=1.0, samples_per_second=RECIPE["samples"])
    metrics = {
        **BINDING,
        "recipe": RECIPE,
        "image": IMAGE,
        "torch_version": TORCH_VERSION,
        "device_type": "cuda",
        "device_name": "synthetic-test-fixture",
        "compute_capability": [10, 0],
        "journal": journal,
        "final_train_loss": _loss_gradient(parameters, *_dataset(7, 1024))[0],
        "heldout_loss": _loss_gradient(parameters, *_dataset(8, 1024))[0],
    }
    checkpoint = {
        **BINDING,
        "recipe": RECIPE,
        "parameters": parameters,
        "momentum": momentum,
        "initial_parameters": INITIAL,
        "optimizer_step": RECIPE["steps"],
    }
    artifacts = {
        "checkpoint.pt": b"explicit-unit-fixture-not-a-torch-checkpoint",
        "metrics.json": json.dumps(metrics).encode(),
    }
    manifest = {
        "schema": "cuda-regression-manifest/v1",
        **BINDING,
        "artifacts": {
            name: {"sha256": _sha256(data), "size": len(data)}
            for name, data in artifacts.items()
        },
    }
    return manifest, metrics, checkpoint, artifacts


def test_cpu_reference_loss_and_momentum_fixture(evidence):
    manifest, metrics, checkpoint, artifacts = evidence
    _validate_integrity(manifest, artifacts)
    result = _validate_artifacts(manifest, metrics, checkpoint, BINDING)
    assert result["heldout_loss"] == pytest.approx(0.0015574999575306604, abs=1e-12)
    assert result["final_train_loss"] == pytest.approx(0.0016543993155160245, abs=1e-12)
    assert result["optimizer_steps_verified"] == 32


@pytest.mark.parametrize("name", ["checkpoint.pt", "metrics.json"])
def test_rejects_corrupt_artifact_bytes(evidence, name):
    manifest, _, _, artifacts = evidence
    artifacts[name] += b"corrupt"
    with pytest.raises(ValueError, match="integrity mismatch"):
        _validate_integrity(manifest, artifacts)


@pytest.mark.parametrize("name", ["checkpoint.pt", "metrics.json"])
def test_rejects_corrupt_expected_hash(evidence, name):
    manifest, _, _, artifacts = evidence
    manifest["artifacts"][name]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity mismatch"):
        _validate_integrity(manifest, artifacts)


@pytest.mark.parametrize(
    "change", ["loss", "step", "throughput", "nonfinite", "heldout"]
)
def test_rejects_corrupt_measurements_even_with_new_hash(evidence, change):
    manifest, metrics, checkpoint, _ = evidence
    if change == "heldout":
        metrics["heldout_loss"] += 0.1
    elif change == "step":
        metrics["journal"][1]["optimizer_step"] = 1
    elif change == "throughput":
        metrics["journal"][0]["samples_per_second"] *= 2
    else:
        metrics["journal"][0]["loss"] = float("nan") if change == "nonfinite" else 0.1
    with pytest.raises(ValueError):
        _validate_artifacts(manifest, metrics, checkpoint, BINDING)


@pytest.mark.parametrize("field", ["parameters", "momentum", "initial_parameters"])
def test_rejects_changed_checkpoint_state(evidence, field):
    manifest, metrics, checkpoint, _ = evidence
    checkpoint[field] = copy.copy(checkpoint[field])
    checkpoint[field][0] += 0.1
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_artifacts(manifest, metrics, checkpoint, BINDING)


def test_rejects_cross_run_binding(evidence):
    manifest, metrics, checkpoint, _ = evidence
    checkpoint["run_id"] = "another-unit-fixture"
    with pytest.raises(ValueError, match="binding differs"):
        _validate_artifacts(manifest, metrics, checkpoint, BINDING)


def test_rejects_cpu_device_claim(evidence):
    manifest, metrics, checkpoint, _ = evidence
    metrics["device_type"] = "cpu"
    with pytest.raises(ValueError, match="CUDA device"):
        _validate_artifacts(manifest, metrics, checkpoint, BINDING)

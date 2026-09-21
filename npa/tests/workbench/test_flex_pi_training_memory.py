"""A memory allocation candidate must preserve math and fail closed on drift."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.utils.deterministic

from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_fixture import fixture_indices
from npa.workbench.flex_pi.training_memory import (
    execution_without_memory_fill,
    memory_fill_receipt,
    observation_with_memory_fill,
)
from npa.workbench.flex_pi.training_qualification import qualify_memory_fill


@pytest.fixture
def strict_determinism(monkeypatch):
    monkeypatch.setattr("npa.workbench.flex_pi.training_memory._disabled_scopes", 0)
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.utils.deterministic.fill_uninitialized_memory = original[2]
        torch.backends.cudnn.benchmark = original[3]
        torch.backends.cudnn.deterministic = original[4]


def test_execution_scope_preserves_rng_algorithms_and_restores_after_failure(
    strict_determinism,
):
    original = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model failed"):
        with execution_without_memory_fill():
            assert not torch.utils.deterministic.fill_uninitialized_memory
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            raise ValueError("model failed")
    assert torch.equal(original, torch.get_rng_state())
    receipt = memory_fill_receipt({"npa_memory_fill": "off"})
    assert receipt["model_execution_fill"] is False
    assert receipt["restored_fill"] is True


@pytest.mark.parametrize("warn_only", [False, True])
def test_scope_cannot_silently_relax_algorithm_determinism(
    strict_determinism, warn_only
):
    torch.use_deterministic_algorithms(warn_only, warn_only=warn_only)
    with pytest.raises(RuntimeError, match="strict determinism"):
        with execution_without_memory_fill():
            pytest.fail("must reject before model execution")
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_scope_detects_policy_mutation_and_still_restores(strict_determinism):
    with pytest.raises(RuntimeError, match="changed its allocation policy"):
        with execution_without_memory_fill():
            torch.utils.deterministic.fill_uninitialized_memory = True
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_off_policy_requires_observed_execution_scopes(strict_determinism):
    with pytest.raises(RuntimeError, match="observed allocation scopes"):
        memory_fill_receipt({"npa_memory_fill": "off"})


def test_fixture_selection_uses_true_epoch_tail_without_changing_rng():
    state = torch.get_rng_state().clone()
    permutation = torch.randperm(115620, generator=torch.Generator().manual_seed(42))
    indices = fixture_indices()
    assert indices == permutation[:96].tolist() + permutation[-36:].tolist()
    assert len(set(indices)) == 132
    assert torch.equal(state, torch.get_rng_state())


def _qualification_launcher(work, *, drift=None):
    calls = []

    def launch(plan, root):
        fill = plan["execution"]["memory_fill"]
        calls.append(copy.deepcopy(plan))
        phase = Path(plan["work_directory"])
        phase.mkdir()
        config = {"model": {"width": 32}}
        if fill == "off":
            config["npa_memory_fill"] = "off"
            if drift == "model":
                config["model"]["width"] = 64
        (phase / "workload.json").write_text(json.dumps({"configuration": config}))
        return {
            "initial_model_sha256": "initial",
            "training_source_sha256": {"source": "pinned"},
            "normalization_sha256": "original",
            "qualification": {
                "gradient": "changed"
                if fill == "off" and drift == "gradient"
                else "same"
            },
            "memory_fill": {
                "model_execution_fill": fill == "on",
                "observed_disabled_scopes": 35 if fill == "off" else 0,
                "initialization_and_loader_creation_fill": True,
                "loader_worker_fill": True,
                "allocation_flag_scope": "process_global_including_pin_thread",
                "restored_fill": True,
                "strict_deterministic_algorithms": True,
            },
        }

    return launch, calls


@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_qualification_isolates_processes_and_binds_checkpoint(
    tmp_path, checkpoint
):
    launch, calls = _qualification_launcher(tmp_path)
    saved = tmp_path / "restored" if checkpoint else None
    plan = {"execution": {"mode": "train", "memory_fill": "off"}, "untouched": 1}
    original = copy.deepcopy(plan)
    root = tmp_path / "requests"
    root.mkdir()
    result = qualify_memory_fill(plan, root, tmp_path, launch, checkpoint=saved)
    assert [call["execution"]["memory_fill"] for call in calls] == ["on", "on", "off"]
    assert len({call["work_directory"] for call in calls}) == 3
    assert all(call["execution"]["mode"] == "qualify" for call in calls)
    assert all(
        call.get("resume_directory") == (str(saved) if saved else None)
        for call in calls
    )
    assert plan == original
    assert result["bitwise_equal"] and result["diagnostic_only"]
    assert result["fixture_samples"] == [96, 36]


@pytest.mark.parametrize("drift", ["gradient", "model"])
def test_any_numerical_or_workload_drift_rejects_candidate(tmp_path, drift):
    launch, _ = _qualification_launcher(tmp_path, drift=drift)
    root = tmp_path / "requests"
    root.mkdir()
    with pytest.raises(FlexPiError, match="qualification"):
        qualify_memory_fill({"execution": {}}, root, tmp_path, launch)
    rejection = json.loads((tmp_path / "fill-initial-rejection.json").read_text())
    assert rejection["passed"] is False
    assert rejection["failed_phase"] == "candidate"
    assert len(rejection["reports"]) == 3


def test_fill_candidate_requires_normalization_before_launch():
    with pytest.raises(FlexPiError, match="verified original normalization"):
        train(output_path="s3://example-bucket/run", memory_fill="off", dry_run=True)


def test_observation_restores_enclosing_candidate_scope(strict_determinism):
    with execution_without_memory_fill():
        with pytest.raises(ValueError):
            with observation_with_memory_fill():
                assert torch.utils.deterministic.fill_uninitialized_memory
                raise ValueError("failed diagnostic")
        assert not torch.utils.deterministic.fill_uninitialized_memory
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_mislabeled_candidate_cannot_claim_qualification(tmp_path):
    launcher, _ = _qualification_launcher(tmp_path)

    def mislabeled(plan, root):
        result = launcher(plan, root)
        result["memory_fill"]["model_execution_fill"] = True
        return result

    root = tmp_path / "requests"
    root.mkdir()
    with pytest.raises(FlexPiError, match="requested memory-fill policy"):
        qualify_memory_fill({"execution": {}}, root, tmp_path, mislabeled)


@pytest.mark.parametrize("corrupt", [False, True])
def test_rejection_publication_verifies_bytes_without_masking_original_failure(
    tmp_path, monkeypatch, capsys, corrupt
):
    from npa.workbench.flex_pi.training import _qualify_memory_fill

    rejection = FlexPiError("gradient mismatch")

    def fail(*args, **kwargs):
        (tmp_path / "fill-initial-rejection.json").write_text('{"passed":false}')
        raise rejection

    class Storage:
        def upload_file(self, path, destination):
            self.original = Path(path).read_bytes()

        def download_file(self, source, path):
            Path(path).write_bytes(b"different" if corrupt else self.original)

    storage = Storage()
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment", lambda: storage
    )
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training_qualification.qualify_memory_fill", fail
    )
    plan = {
        "execution": {"memory_fill": "off", "output_path": "s3://example-bucket/run"}
    }
    with pytest.raises(FlexPiError) as caught:
        _qualify_memory_fill(plan, tmp_path, tmp_path)
    assert caught.value is rejection
    proof = tmp_path / "fill-initial-rejection-publication.json"
    assert proof.exists() is not corrupt
    if not corrupt:
        assert json.loads(proof.read_text())["read_after_write_verified"]
    assert ("publication failed: FlexPiError" in capsys.readouterr().err) is corrupt

"""A memory allocation candidate must preserve math and fail closed on drift."""

import copy
import json
from pathlib import Path

import pytest

from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_qualification import qualify_memory_fill


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

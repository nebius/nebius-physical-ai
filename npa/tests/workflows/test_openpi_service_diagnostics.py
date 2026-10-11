"""Exercise private diagnostics and the actual checkpoint-proof producer/reader."""

import hashlib
import json
import runpy
from types import SimpleNamespace

import pytest

from npa.workflows.byof import openpi_selected_policy as selected
from npa.workflows.byof import openpi_service_diagnostics as diagnostics
from npa.workflows.byof import openpi_service_lifecycle as lifecycle


@pytest.fixture
def directory(tmp_path, monkeypatch):
    path = tmp_path / "diagnostics"
    monkeypatch.setattr(diagnostics, "DIAGNOSTICS_DIRECTORY", path)
    return path


def test_diagnostics_create_private_owned_directory(directory):
    assert diagnostics.diagnostics_directory() == directory
    assert directory.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_diagnostics_reject_public_existing_directory(directory, mode):
    directory.mkdir()
    directory.chmod(mode)
    with pytest.raises(RuntimeError, match="private service-owned"):
        diagnostics.diagnostics_directory()


def test_diagnostics_reject_symlink_directory(directory, tmp_path):
    destination = tmp_path / "foreign"
    destination.mkdir(mode=0o700)
    directory.symlink_to(destination, target_is_directory=True)
    with pytest.raises(RuntimeError, match="private service-owned"):
        diagnostics.diagnostics_directory()


def test_diagnostics_reject_symlink_receipt(directory, tmp_path):
    diagnostics.diagnostics_directory()
    destination = tmp_path / "foreign.json"
    destination.write_text("original")
    (directory / "npa-openpi-loaded-checkpoint.json").symlink_to(destination)
    with pytest.raises(RuntimeError, match="symlink"):
        diagnostics.diagnostic_path("npa-openpi-loaded-checkpoint.json")
    assert destination.read_text() == "original"


def test_diagnostics_reject_other_files(directory):
    with pytest.raises(ValueError, match="unknown"):
        diagnostics.diagnostic_path("credentials.json")
    assert not directory.exists()


def test_native_probe_and_selected_policy_share_private_receipt(
    directory, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = b"actual checkpoint bytes"
    (checkpoint / "params").write_bytes(payload)
    records = [
        {
            "path": "params",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {"files": records, "content_manifest_sha256": digest}
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("NPA_RESOLVED_CHECKPOINT_DIR", str(checkpoint))
    program = tmp_path / "checkpoint-probe.py"
    program.write_text(lifecycle._checkpoint_probe_program())
    runpy.run_path(str(program))
    config = SimpleNamespace(
        name="selected",
        model=SimpleNamespace(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=15,
        ),
    )
    selected._record_loaded_policy(
        checkpoint, config, {"archive_sha256": "a" * 64}, manifest, 37
    )
    receipt = json.loads((directory / "npa-openpi-loaded-checkpoint.json").read_text())
    assert receipt["checkpoint_manifest_sha256"] == digest
    assert receipt["selected_policy"]["saved_step"] == 37
    assert receipt["selected_policy"]["loaded"] is True
    assert sorted(path.name for path in checkpoint.iterdir()) == [
        "manifest.json",
        "params",
    ]

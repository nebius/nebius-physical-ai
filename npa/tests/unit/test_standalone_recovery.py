"""Verify interrupted standalone ownership recovery and unsafe-state refusal."""

from __future__ import annotations

import json

import pytest

from npa.cluster_backends import standalone_recovery as recovery
from npa.provisioning_journal import ProvisioningOperation


def _write_partial_state(install):
    sidecar = {
        "backend": "mk8s",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "region": "region-test",
        "cluster_name": "test-cluster",
        "context": "fleet-standalone-project-test-test-cluster",
        "status": "provisioning",
    }
    cluster = {
        "id": "cluster-test",
        "parent_id": "project-test",
        "name": "test-cluster",
    }
    state = {
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_cluster",
                "instances": [{"attributes": cluster}],
            }
        ]
    }
    sidecar_path = install / ".npa-fleet-env.json"
    state_path = install / "k8s-training" / "terraform.tfstate"
    state_path.parent.mkdir(parents=True)
    for path, payload in ((sidecar_path, sidecar), (state_path, state)):
        path.write_text(json.dumps(payload))
        path.chmod(0o600)
    return sidecar_path, state_path


@pytest.fixture
def partial_cluster(tmp_path, monkeypatch):
    root = tmp_path / "owned-context" / "backend-state"
    sidecar_path, state_path = _write_partial_state(
        root / "project-test" / "test-cluster"
    )
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias="test",
        project_id="project-test",
        tenant_id="tenant-test",
        region="region-test",
        backend={},
        resource_type="cluster",
        requested_name="test-cluster",
        ownership_source="test",
        resume_command="",
        destroy_argv=["npa", "cluster", "down", "--context", "owned-context"],
    )
    operation.transition("mutating")
    monkeypatch.setattr(
        recovery, "metadata_file", lambda _context: root.parent / "metadata.json"
    )
    return root, sidecar_path, state_path, operation


def _recover(**overrides):
    return recovery.partial_backend_metadata(
        **{
            "context": "owned-context",
            "project_id": "project-test",
            "tenant_id": "tenant-test",
            "region": "region-test",
            **overrides,
        }
    )


def test_partial_cluster_recovers_exact_native_backend_without_rewriting_state(
    partial_cluster,
):
    root, sidecar, state, _operation = partial_cluster
    before = [path.read_bytes() for path in (sidecar, state)]
    metadata = _recover()
    assert metadata["backend_state_root"] == str(root.resolve())
    assert metadata["backend_cluster_id"] == "cluster-test"
    assert metadata["backend_recovery_operation_id"] == _operation.operation_id
    assert [path.read_bytes() for path in (sidecar, state)] == before
    assert not (root.parent / "metadata.json").exists()


@pytest.mark.parametrize(
    "field", ["project_id", "tenant_id", "region", "cluster_name", "context", "backend"]
)
def test_partial_cluster_rejects_sidecar_identity_mismatch(partial_cluster, field):
    _root, sidecar, _state, _operation = partial_cluster
    payload = json.loads(sidecar.read_text())
    payload[field] = "another-owner"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity does not match"):
        _recover()


def test_partial_cluster_rejects_state_for_another_provider_project(partial_cluster):
    _root, _sidecar, state, _operation = partial_cluster
    payload = json.loads(state.read_text())
    payload["resources"][0]["instances"][0]["attributes"]["parent_id"] = "project-other"
    state.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cluster identity does not match"):
        _recover()


@pytest.mark.parametrize("file_index", [1, 2])
def test_partial_cluster_rejects_readable_or_symlinked_state(
    partial_cluster, file_index
):
    path = partial_cluster[file_index]
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        _recover()
    path.chmod(0o600)
    target = path.with_suffix(".original")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular owned state"):
        _recover()


def test_partial_cluster_needs_unambiguous_operation(partial_cluster, monkeypatch):
    operation = partial_cluster[3]
    monkeypatch.setattr(
        recovery, "list_operations", lambda **_kwargs: [operation, operation]
    )
    with pytest.raises(ValueError, match="exact --operation-id"):
        _recover()
    assert (
        _recover(operation_id=operation.operation_id)["backend_cluster_id"]
        == "cluster-test"
    )


def test_partial_cluster_cannot_recover_without_matching_journal(
    partial_cluster, monkeypatch
):
    monkeypatch.setattr(recovery, "list_operations", lambda **_kwargs: [])
    assert _recover() == {}

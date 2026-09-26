"""Guardrails for Agent-only stopped-daemon credential-cache recovery."""

from __future__ import annotations

import pytest

from npa.orchestration.skypilot import local_api as api


@pytest.fixture
def metadata_root(tmp_path, monkeypatch):
    root = tmp_path / "cloud-metadata"
    root.mkdir()
    monkeypatch.setattr(api, "_METADATA_TOKEN_ROOT", root)
    return root


def _record(tmp_path, metadata_root, *, project: str = "agent-project") -> dict:
    root = tmp_path / "npa"
    binding = {"NPA_CONFIG_DIR": "stable", "KUBECONFIG": "stable"}
    files = {
        str(root / "config.yaml"): "before",
        str(root / "credentials.yaml"): "before",
        str(tmp_path / "agent-home" / ".nebius" / "credentials.yaml"): "before",
        str(metadata_root / "token"): "before",
    }
    return {
        "project_alias": project,
        "environment_binding": binding,
        "identity_files": files,
    }


def _environment(tmp_path, *, project: str = "agent-project") -> dict[str, str]:
    return {
        "NPA_AGENT_ISOLATED_RECOVERY_REBIND": "v1",
        "NPA_CONFIG_DIR": str(tmp_path / "npa"),
        "NPA_SKYPILOT_PROJECT": project,
        "NPA_NEBIUS_CREDENTIAL_SOURCE": "instance_metadata",
        "HOME": str(tmp_path / "agent-home"),
    }


def test_agent_recovery_rebinds_only_staged_profile_and_metadata(
    tmp_path, metadata_root
):
    record = _record(tmp_path, metadata_root)
    root = tmp_path / "npa"
    current = {
        str(root / "config.yaml"): "after",
        str(root / "credentials.yaml"): "after",
        str(tmp_path / "agent-home" / ".nebius" / "credentials.yaml"): "after",
        str(metadata_root / "token"): "after",
    }

    assert api._agent_profile_rebind_allowed(
        record,
        binding=record["environment_binding"],
        files=current,
        environment=_environment(tmp_path),
    )


def test_agent_recovery_rejects_changed_cluster_identity_or_binding(
    tmp_path, metadata_root
):
    record = _record(tmp_path, metadata_root)
    root = tmp_path / "npa"
    current = {
        str(root / "config.yaml"): "after",
        str(root / "credentials.yaml"): "after",
        str(root / "clusters" / "target" / "kubeconfig"): "changed",
    }

    assert not api._agent_profile_rebind_allowed(
        record,
        binding=record["environment_binding"],
        files=current,
        environment=_environment(tmp_path),
    )
    assert not api._agent_profile_rebind_allowed(
        record,
        binding={**record["environment_binding"], "KUBECONFIG": "different"},
        files=record["identity_files"],
        environment=_environment(tmp_path),
    )


def test_agent_recovery_rejects_metadata_symlink_escape(tmp_path, metadata_root):
    outside = tmp_path / "outside-token"
    outside.write_text("fixture token")
    token = metadata_root / "token"
    token.symlink_to(outside)
    record = _record(tmp_path, metadata_root)
    current = {**record["identity_files"], str(token): "after"}

    assert not api._agent_profile_rebind_allowed(
        record,
        binding=record["environment_binding"],
        files=current,
        environment=_environment(tmp_path),
    )

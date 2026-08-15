from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.cluster import identity
from npa.cluster.exceptions import ClusterNotFoundError
from npa.cluster.state import ClusterState


def _state(
    kubeconfig: Path,
    *,
    project_id: str = "project-a",
    provider_name: str = "",
) -> ClusterState:
    return ClusterState(
        name="npa-cluster",
        cluster_id="cluster-a",
        project_id=project_id,
        region="region-a",
        node_count=1,
        node_platform="cpu-d3",
        node_preset="4vcpu-16gb",
        k8s_version="1.30",
        subnet_id="subnet-a",
        created_at="2026-08-06T00:00:00Z",
        endpoint="https://api.cluster.example",
        kubeconfig_path=str(kubeconfig),
        provider_name=provider_name,
    )


def _write_kubeconfig(path: Path, *, current: str = "npa-cluster") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "current-context": current,
                "contexts": [
                    {
                        "name": "npa-cluster",
                        "context": {"cluster": "cluster-a", "user": "operator"},
                    }
                ],
                "clusters": [
                    {
                        "name": "cluster-a",
                        "cluster": {"server": "https://api.cluster.example"},
                    }
                ],
                "users": [{"name": "operator", "user": {"token": "fixture"}}],
            }
        ),
        encoding="utf-8",
    )


def _configure(monkeypatch, kubeconfig: Path, *, state_project: str = "project-a"):
    from npa.clients import config

    state = _state(kubeconfig, project_id=state_project)
    monkeypatch.setattr(
        config,
        "list_projects",
        lambda: {
            "selected": {"project_id": "project-a"},
            "other": {"project_id": "project-b"},
        },
    )
    monkeypatch.setattr(config, "default_project_name", lambda: "selected")
    monkeypatch.setattr(identity, "load_cluster_state", lambda context: state)
    monkeypatch.setattr(identity, "list_local_clusters", lambda: [state])
    monkeypatch.setattr(identity, "existing_kubeconfig", lambda context: kubeconfig)
    return state


class _Client:
    def __init__(self, remote=None, error: Exception | None = None) -> None:  # noqa: ANN001
        self.remote = remote or SimpleNamespace(
            id="cluster-a",
            project_id="project-a",
            name="npa-cluster",
            endpoint="https://api.cluster.example",
        )
        self.error = error
        self.calls = []

    def get_cluster(self, cluster_id: str, *, project_id: str = ""):
        self.calls.append((cluster_id, project_id))
        if self.error:
            raise self.error
        return self.remote


def test_explicit_project_and_context_take_precedence(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig)
    _configure(monkeypatch, kubeconfig)
    client = _Client()

    verified = identity.resolve_verified_cluster_identity(
        project="selected", context="npa-cluster", client=client
    )

    assert verified.project_alias == "selected"
    assert verified.context == "npa-cluster"
    assert client.calls == [("cluster-a", "project-a")]


def test_provider_name_may_differ_from_unique_local_context(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig)
    state = _state(kubeconfig, provider_name="fleet-cluster")
    from npa.clients import config

    monkeypatch.setattr(
        config, "list_projects", lambda: {"selected": {"project_id": "project-a"}}
    )
    monkeypatch.setattr(config, "default_project_name", lambda: "selected")
    monkeypatch.setattr(identity, "load_cluster_state", lambda context: state)
    monkeypatch.setattr(identity, "existing_kubeconfig", lambda context: kubeconfig)
    client = _Client(
        remote=SimpleNamespace(
            id="cluster-a",
            project_id="project-a",
            name="fleet-cluster",
            endpoint="https://api.cluster.example",
        )
    )

    verified = identity.resolve_verified_cluster_identity(
        project="selected", context="npa-cluster", client=client
    )

    assert verified.context == "npa-cluster"
    assert verified.cluster_name == "fleet-cluster"


def test_context_project_mismatch_fails_before_provider_access(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig)
    _configure(monkeypatch, kubeconfig, state_project="project-b")
    client = _Client()

    with pytest.raises(identity.ClusterIdentityError, match="belongs to project"):
        identity.resolve_verified_cluster_identity(
            project="selected", context="npa-cluster", client=client
        )

    assert client.calls == []


def test_stale_current_context_is_refused(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig, current="unrelated")
    _configure(monkeypatch, kubeconfig)

    with pytest.raises(identity.ClusterIdentityError, match="current-context"):
        identity.resolve_verified_cluster_identity(
            project="selected", context="npa-cluster", client=_Client()
        )


def test_explicit_kubeconfig_must_match_saved_exact_file(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    saved = tmp_path / "saved"
    other = tmp_path / "other"
    _write_kubeconfig(saved)
    _write_kubeconfig(other)
    _configure(monkeypatch, saved)

    with pytest.raises(identity.ClusterIdentityError, match="does not match"):
        identity.resolve_verified_cluster_identity(
            project="selected",
            context="npa-cluster",
            kubeconfig=other,
            client=_Client(),
        )


def test_conclusive_provider_absence_is_verified_convergence(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig)
    _configure(monkeypatch, kubeconfig)

    verified = identity.resolve_verified_cluster_identity(
        project="selected",
        context="npa-cluster",
        client=_Client(error=ClusterNotFoundError("gone")),
    )

    assert verified.cluster_absent is True


def test_ambiguous_inferred_cluster_is_refused(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    _write_kubeconfig(kubeconfig)
    state = _configure(monkeypatch, kubeconfig)
    monkeypatch.setattr(identity, "list_local_clusters", lambda: [state, state])

    with pytest.raises(identity.ClusterIdentityError, match="exactly one"):
        identity.resolve_verified_cluster_identity(client=_Client())

from __future__ import annotations

import pytest

from npa.cluster.config import (
    DEFAULT_K8S_VERSION,
    DEFAULT_NODE_PLATFORM,
    DEFAULT_NODE_PRESET,
    DEFAULT_REGION,
    ClusterConfig,
    resolve_project_id,
)
from npa.cluster.exceptions import ClusterConfigError


def test_default_cluster_config_values() -> None:
    config = ClusterConfig(name="w8cluster-test")

    assert config.region == DEFAULT_REGION
    assert config.node_count == 1
    assert config.node_platform == DEFAULT_NODE_PLATFORM
    assert config.node_preset == DEFAULT_NODE_PRESET
    assert config.k8s_version == DEFAULT_K8S_VERSION
    assert config.wait is True
    assert config.public_node_ip is False


@pytest.mark.parametrize("name", ["bad_name", "-bad", "bad-", "", "x" * 64])
def test_invalid_cluster_names_rejected(name: str) -> None:
    with pytest.raises(ClusterConfigError):
        ClusterConfig(name=name)


def test_uppercase_timestamp_style_name_is_allowed() -> None:
    config = ClusterConfig(name="w8cluster-test-20260514T214600Z")

    assert config.name.endswith("Z")


def test_us_central1_region_is_allowed() -> None:
    config = ClusterConfig(name="cluster-a", region="us-central1")
    assert config.region == "us-central1"


def test_invalid_region_rejected() -> None:
    with pytest.raises(ClusterConfigError, match="unsupported region"):
        ClusterConfig(name="cluster-a", region="us-east1")


def test_invalid_node_shape_rejected() -> None:
    with pytest.raises(ClusterConfigError, match="unsupported preset"):
        ClusterConfig(name="cluster-a", node_platform="cpu-e2", node_preset="4vcpu-8gb")


def test_project_id_resolution_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-env")

    assert resolve_project_id("project-explicit") == "project-explicit"


def test_project_id_resolution_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_CLUSTER_PROJECT_ID", "project-env")

    assert resolve_project_id() == "project-env"

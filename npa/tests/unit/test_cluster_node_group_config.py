from __future__ import annotations

import pytest

from npa.cluster.config import NodeGroupConfig
from npa.cluster.exceptions import ClusterConfigError
from npa.cluster.node_group import GPU_TYPE_DEFAULTS, default_node_group_name, resolve_gpu_preset


def test_node_group_config_defaults_to_private_h100() -> None:
    config = NodeGroupConfig(cluster_name="cluster-a", gpu_type="h100")

    assert config.name == "cluster-a-h100-gpu"
    assert config.platform == "gpu-h100-sxm"
    assert config.node_preset == "1gpu-16vcpu-200gb"
    assert config.node_count == 1
    assert config.public_ip is False


def test_resolve_preset_allows_override() -> None:
    assert resolve_gpu_preset("h100", "8gpu-128vcpu-1600gb") == "8gpu-128vcpu-1600gb"


def test_supported_gpu_types_are_v1_surface() -> None:
    assert set(GPU_TYPE_DEFAULTS) == {"h100", "h200", "l40s", "rtx6000"}


def test_invalid_gpu_type_rejected() -> None:
    with pytest.raises(ClusterConfigError, match="unsupported GPU type"):
        NodeGroupConfig(cluster_name="cluster-a", gpu_type="b300")


@pytest.mark.parametrize(
    ("autoscaling_min", "autoscaling_max"),
    [(1, None), (3, 2), (-1, 2), (0, 0)],
)
def test_invalid_autoscaling_bounds_rejected(autoscaling_min: int | None, autoscaling_max: int | None) -> None:
    with pytest.raises(ClusterConfigError):
        NodeGroupConfig(
            cluster_name="cluster-a",
            gpu_type="h100",
            autoscaling_min=autoscaling_min,
            autoscaling_max=autoscaling_max,
        )


def test_default_name_is_truncated_to_cluster_name_limit() -> None:
    name = default_node_group_name("x" * 63, "h100")

    assert len(name) == 63
    assert name.endswith("-h100-gpu")

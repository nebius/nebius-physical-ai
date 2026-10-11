from __future__ import annotations

import pytest

from npa.cluster.config import NodeGroupConfig
from npa.cluster.exceptions import ClusterConfigError
from npa.cluster.node_group import (
    GPU_TYPE_DEFAULTS,
    default_node_group_name,
    resolve_gpu_preset,
)


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
    assert set(GPU_TYPE_DEFAULTS) == {"b200", "h100", "h200", "l40s", "rtx6000"}


def test_b200_defaults_to_reviewed_fabric_shape_and_storage() -> None:
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="b200",
        gpu_cluster_id="computegpucluster-test",
    )

    assert config.platform == "gpu-b200-sxm"
    assert config.node_preset == "8gpu-160vcpu-1792gb"
    assert config.boot_disk_size_gib == 1280
    assert config.gpu_cluster_id == "computegpucluster-test"


def test_b200_accepts_a_provider_alternate_replicated_disk_shape() -> None:
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="b200",
        gpu_cluster_id="computegpucluster-test",
        boot_disk_type=" NETWORK_SSD_IO_M3 ",
        boot_disk_size_gib=1302,
    )

    assert config.boot_disk_type == "network_ssd_io_m3"
    assert config.boot_disk_size_gib == 1302


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_cluster_id": ""}, "--gpu-cluster-id"),
        ({"node_count": 2}, "one fixed 8-GPU node"),
        ({"node_preset": "1gpu-20vcpu-224gb"}, "reviewed gpu-b200-sxm"),
        ({"boot_disk_size_gib": 128}, "at least 1280 GiB"),
        ({"boot_disk_type": "network_hdd"}, "boot disk type"),
        (
            {
                "boot_disk_type": "network_ssd_io_m3",
                "boot_disk_size_gib": 1280,
            },
            "multiple of 93 GiB",
        ),
        ({"wait": False}, "requires --wait"),
    ],
)
def test_b200_rejects_unverified_attachment_shapes(overrides, message: str) -> None:
    options = {"gpu_cluster_id": "computegpucluster-test"}
    options.update(overrides)
    with pytest.raises(ClusterConfigError, match=message):
        NodeGroupConfig(
            cluster_name="cluster-a",
            gpu_type="b200",
            **options,
        )


def test_non_b200_rejects_gpu_cluster_attachment() -> None:
    with pytest.raises(ClusterConfigError, match="only for b200"):
        NodeGroupConfig(
            cluster_name="cluster-a",
            gpu_type="h100",
            gpu_cluster_id="computegpucluster-test",
        )


def test_invalid_gpu_type_rejected() -> None:
    with pytest.raises(ClusterConfigError, match="unsupported GPU type"):
        NodeGroupConfig(cluster_name="cluster-a", gpu_type="b300")


@pytest.mark.parametrize(
    ("autoscaling_min", "autoscaling_max"),
    [(1, None), (3, 2), (-1, 2), (0, 0)],
)
def test_invalid_autoscaling_bounds_rejected(
    autoscaling_min: int | None, autoscaling_max: int | None
) -> None:
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


def test_capacity_block_group_is_trimmed() -> None:
    config = NodeGroupConfig(
        cluster_name="cluster-a",
        gpu_type="h100",
        capacity_block_group=" capacityblockgroup-test ",
    )

    assert config.capacity_block_group == "capacityblockgroup-test"

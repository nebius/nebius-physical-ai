# Verify the owned SkyPilot API configuration preserves authentication and lifecycle boundaries.
"""Security and ownership boundaries of the operator-owned upstream API."""

from pathlib import Path

import yaml


_PLATFORM_DIRECTORY = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development/platform"


def test_api_control_endpoint_is_loopback_and_not_host_networked():
    """Reject platform settings that expose unauthenticated control access.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the platform configuration violates its boundary.
    """
    configuration = yaml.safe_load((_PLATFORM_DIRECTORY / "compose.yaml").read_text())
    service = configuration["services"]["api"]
    assert len(configuration["services"]) == 1
    assert service["ports"] == ["127.0.0.1:${SKY_API_PORT:-46590}:46580"]
    assert service.get("network_mode") != "host"
    assert not service.get("privileged", False)
    assert service["cap_drop"] == ["ALL"]
    assert service["cap_add"] == ["DAC_OVERRIDE"]
    assert service["security_opt"] == ["no-new-privileges:true"]


def test_credentials_are_required_readonly_bind_mounts_without_host_socket():
    """Limit platform credentials to explicit read-only mount boundaries.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the platform configuration violates its boundary.
    """
    service = yaml.safe_load((_PLATFORM_DIRECTORY / "compose.yaml").read_text())["services"]["api"]
    bind_mounts = []
    for mount in service["volumes"]:
        if isinstance(mount, dict):
            bind_mounts.append(mount)
    assert {mount["target"] for mount in bind_mounts} == {
        "/root/.kube/config", "/root/.nebius", "/usr/local/bin/nebius",
    }
    for mount in bind_mounts:
        assert mount["type"] == "bind"
        assert mount["read_only"] is True
        assert mount["bind"]["create_host_path"] is False
        assert ":?" in mount["source"]
    assert set(service["environment"]) == {
        "HOME", "KUBECONFIG", "SKYPILOT_DISABLE_USAGE_COLLECTION",
    }


def test_upstream_service_has_persistent_owned_state_and_no_gpu_lifecycle():
    """Keep platform state owned independently of GPU development clusters.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the platform configuration violates its boundary.
    """
    configuration = yaml.safe_load((_PLATFORM_DIRECTORY / "compose.yaml").read_text())
    service = configuration["services"]["api"]
    assert service["image"].startswith("berkeleyskypilot/skypilot@sha256:")
    assert len(service["image"].split("@sha256:")[1]) == 64
    assert "build" not in service
    assert service["command"] == ["sky", "api", "start", "--host", "0.0.0.0", "--foreground"]
    assert service["init"] is True
    assert "api-state:/root/.sky" in service["volumes"]
    assert configuration["volumes"] == {"api-state": None}
    assert "deploy" not in service
    assert "gpus" not in service
    assert "devices" not in service

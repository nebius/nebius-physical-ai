from __future__ import annotations

from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_INSTANCE_TYPE,
    apply_controller_override,
    controller_resources_kubernetes,
    controller_resources_nebius_vm,
    default_controller_resources,
)


def test_default_controller_resources_returns_kubernetes_default() -> None:
    resources = default_controller_resources()

    assert resources == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "autostop": False,
    }
    assert "disk_size" not in resources


def test_nebius_vm_controller_resources_remain_available() -> None:
    resources = controller_resources_nebius_vm()

    assert resources == {
        "cloud": "nebius",
        "region": "eu-north1",
        "instance_type": DEFAULT_CONTROLLER_INSTANCE_TYPE,
        "cpus": 2,
        "memory": 8,
        "disk_size": 64,
        "autostop": False,
    }


def test_apply_controller_override_injects_idempotently() -> None:
    first = apply_controller_override({"name": "dag"})
    second = apply_controller_override(first)

    assert first == second
    assert first["jobs"]["controller"]["resources"] == controller_resources_kubernetes()
    assert "disk_size" not in first["jobs"]["controller"]["resources"]


def test_apply_controller_override_can_emit_nebius_vm_fallback() -> None:
    config = apply_controller_override({"name": "dag"}, controller_backend="nebius")

    assert config["jobs"]["controller"]["resources"] == controller_resources_nebius_vm()


def test_apply_controller_override_preserves_explicitly_larger_kubernetes_controller() -> None:
    existing = {
        "jobs": {
            "controller": {
                "resources": {
                    "cloud": "kubernetes",
                    "cpus": 8,
                    "memory": 32,
                }
            }
        }
    }

    config = apply_controller_override(existing)

    expected = {**existing["jobs"]["controller"]["resources"], "autostop": False}
    assert config["jobs"]["controller"]["resources"] == expected


def test_apply_controller_override_drops_kubernetes_disk_size() -> None:
    config = apply_controller_override(
        {
            "jobs": {
                "controller": {
                    "resources": {
                        "cloud": "kubernetes",
                        "cpus": 4,
                        "memory": 16,
                        "disk_size": 50,
                    }
                }
            }
        }
    )

    assert config["jobs"]["controller"]["resources"] == controller_resources_kubernetes()


def test_apply_controller_override_preserves_explicitly_larger_nebius_controller() -> None:
    existing = {
        "jobs": {
            "controller": {
                "resources": {
                    "cloud": "nebius",
                    "region": "eu-north1",
                    "instance_type": "cpu-e2_4vcpu-16gb",
                    "cpus": 4,
                    "memory": 16,
                    "disk_size": 128,
                    "autostop": {"idle_minutes": 10, "down": False},
                }
            }
        }
    }

    config = apply_controller_override(existing, controller_backend="nebius")

    expected = {**existing["jobs"]["controller"]["resources"], "autostop": False}
    assert config["jobs"]["controller"]["resources"] == expected


def test_apply_controller_override_disables_existing_controller_autostop() -> None:
    config = apply_controller_override(
        {
            "jobs": {
                "controller": {
                    "autostop": 10,
                    "resources": {
                        "cloud": "kubernetes",
                        "cpus": 4,
                        "memory": 16,
                    },
                }
            }
        }
    )

    assert config["jobs"]["controller"]["resources"]["autostop"] is False

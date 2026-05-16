from __future__ import annotations

from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_INSTANCE_TYPE,
    apply_controller_override,
    default_controller_resources,
)


def test_default_controller_resources_returns_documented_default() -> None:
    resources = default_controller_resources()

    assert resources == {
        "cloud": "nebius",
        "region": "eu-north1",
        "instance_type": DEFAULT_CONTROLLER_INSTANCE_TYPE,
        "cpus": 2,
        "memory": 8,
        "disk_size": 64,
        "autostop": {"idle_minutes": 5, "down": False},
    }


def test_apply_controller_override_injects_idempotently() -> None:
    first = apply_controller_override({"name": "dag"})
    second = apply_controller_override(first)

    assert first == second
    assert first["jobs"]["controller"]["resources"] == default_controller_resources()


def test_apply_controller_override_preserves_explicitly_larger_controller() -> None:
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

    assert apply_controller_override(existing) == existing

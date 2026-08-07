"""GPU quota / capacity preflight for `npa cluster up`.

Regression: a tenant whose GPU quota was 0 got `QuotaFailure` from Nebius while
Terraform printed `Still creating...` for 27 minutes with no user-visible reason.
"""

from __future__ import annotations

import json
import subprocess

from npa.cli.cluster import capacity


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _quota(limit: int, usage: int) -> str:
    # Shape taken from a real `nebius quotas quota-allowance get-by-name`.
    return json.dumps(
        {
            "metadata": {"name": "compute.instance.gpu.rtx6000"},
            "spec": {"limit": str(limit), "region": "us-central1"},
            "status": {"usage": str(usage), "unit": "count"},
        }
    )


def _advice(*, on_demand_level: str, preemptible_available: int) -> str:
    # Shape taken from a real `nebius capacity resource-advice list`.
    return json.dumps(
        {
            "items": [
                {
                    "spec": {
                        "region": "us-central1",
                        "compute_instance": {
                            "platform": "gpu-rtx6000",
                            "preset": {"name": "1gpu-24vcpu-218gb", "resources": {"gpu_count": 1}},
                        },
                    },
                    "status": {
                        "reserved": {"availability_level": "AVAILABILITY_LEVEL_LIMIT_REACHED"},
                        "on_demand": {"availability_level": on_demand_level, "limit": 0},
                        "preemptible": {
                            "availability_level": "AVAILABILITY_LEVEL_HIGH",
                            "available": preemptible_available,
                        },
                    },
                }
            ]
        }
    )


def _capture(quota: str, advice: str = ""):
    calls: list[list[str]] = []

    def _run(args: list[str]):
        calls.append(args)
        if args[1:4] == ["quotas", "quota-allowance", "get-by-name"]:
            return _completed(quota)
        if args[1:4] == ["capacity", "resource-advice", "list"]:
            return _completed(advice or "{}")
        return _completed("", returncode=1)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def test_gpu_quota_name_maps_the_platform() -> None:
    assert capacity.gpu_quota_name("gpu-rtx6000") == "compute.instance.gpu.rtx6000"
    assert capacity.gpu_quota_name("gpu-h100-sxm") == "compute.instance.gpu.h100"
    assert capacity.gpu_quota_name("gpu-l40s-pcie") == "compute.instance.gpu.l40s"
    assert capacity.gpu_quota_name("cpu-d3") == ""
    assert capacity.gpu_quota_name("") == ""


def test_exhausted_quota_reports_the_numbers_and_the_preemptible_option() -> None:
    capture = _capture(
        _quota(limit=0, usage=0),
        _advice(on_demand_level="AVAILABILITY_LEVEL_LIMIT_REACHED", preemptible_available=44),
    )

    message = capacity.gpu_capacity_error(
        capture,
        nebius_bin="nebius",
        tenant_id="tenant-a",
        region="us-central1",
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        required_gpus=1,
    )

    assert message
    assert "compute.instance.gpu.rtx6000 allows 0 GPU(s)" in message
    assert "requests 1" in message
    assert "QuotaFailure" in message
    # The live capacity numbers make the remedy concrete.
    assert "on-demand LIMIT_REACHED (limit 0)" in message
    assert "preemptible HIGH (available 44)" in message
    assert "gpu_nodes_preemptible = true" in message
    assert "resource-advice list" in message


def test_quota_in_use_by_other_workloads_is_still_a_failure() -> None:
    """The dev tenant's real shape: limit 2, usage 2, so nothing is free."""
    message = capacity.gpu_capacity_error(
        _capture(_quota(limit=2, usage=2)),
        nebius_bin="nebius",
        tenant_id="tenant-a",
        region="us-central1",
        platform="gpu-rtx6000",
        preset="8gpu-192vcpu-1744gb",
        required_gpus=8,
    )

    assert message
    assert "allows 2 GPU(s) with 2 in use (0 free)" in message


def test_unused_quota_of_zero_is_not_treated_as_unreadable() -> None:
    """Nebius omits `status.usage` when nothing is allocated.

    Regression: `limit: "0"` with no usage field made the gate return "unreadable"
    and fail open, so the apply started and hung on `Still creating...` until the
    Terraform timeout — the exact case the gate exists for.
    """
    payload = json.dumps(
        {
            "metadata": {"name": "compute.instance.gpu.rtx6000"},
            "spec": {"limit": "0", "region": "us-central1"},
            "status": {"state": "STATE_ACTIVE", "unit": "count"},  # no `usage`
        }
    )

    assert capacity.gpu_quota_headroom(
        _capture(payload),
        nebius_bin="nebius",
        tenant_id="tenant-a",
        region="us-central1",
        quota_name="compute.instance.gpu.rtx6000",
    ) == (0, 0)

    message = capacity.gpu_capacity_error(
        _capture(payload, _advice(on_demand_level="AVAILABILITY_LEVEL_LIMIT_REACHED", preemptible_available=44)),
        nebius_bin="nebius",
        tenant_id="tenant-a",
        region="us-central1",
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        required_gpus=1,
    )
    assert message
    assert "allows 0 GPU(s) with 0 in use (0 free)" in message


def test_a_quota_response_without_a_limit_is_still_unreadable() -> None:
    """No limit at all means no opinion; the gate must not invent one."""
    payload = json.dumps({"metadata": {}, "status": {"usage": "1"}})

    assert (
        capacity.gpu_quota_headroom(
            _capture(payload),
            nebius_bin="nebius",
            tenant_id="tenant-a",
            region="us-central1",
            quota_name="compute.instance.gpu.rtx6000",
        )
        is None
    )


def test_headroom_passes_quietly() -> None:
    assert (
        capacity.gpu_capacity_error(
            _capture(_quota(limit=8, usage=2)),
            nebius_bin="nebius",
            tenant_id="tenant-a",
            region="us-central1",
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            required_gpus=1,
        )
        is None
    )


def test_preemptible_node_groups_skip_the_on_demand_quota() -> None:
    capture = _capture(_quota(limit=0, usage=0))

    assert (
        capacity.gpu_capacity_error(
            capture,
            nebius_bin="nebius",
            tenant_id="tenant-a",
            region="us-central1",
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            required_gpus=1,
            preemptible=True,
        )
        is None
    )
    assert capture.calls == []  # type: ignore[attr-defined]


def test_unreadable_quota_never_blocks_a_provision() -> None:
    def _fails(args: list[str]):
        return _completed("", returncode=1)

    assert (
        capacity.gpu_capacity_error(
            _fails,
            nebius_bin="nebius",
            tenant_id="tenant-a",
            region="us-central1",
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            required_gpus=1,
        )
        is None
    )


def test_platform_advice_falls_back_to_another_preset_in_the_region() -> None:
    items = json.loads(_advice(on_demand_level="AVAILABILITY_LEVEL_LOW", preemptible_available=3))["items"]

    exact = capacity.platform_advice(
        items, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb", region="us-central1"
    )
    other_preset = capacity.platform_advice(
        items, platform="gpu-rtx6000", preset="8gpu-192vcpu-1744gb", region="us-central1"
    )
    other_region = capacity.platform_advice(
        items, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb", region="eu-north1"
    )

    assert exact is items[0]
    assert other_preset is items[0]
    assert other_region == {}

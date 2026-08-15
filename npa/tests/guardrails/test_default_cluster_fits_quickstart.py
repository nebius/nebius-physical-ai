"""The default cluster shape must be able to schedule the documented quickstart.

Regression: `deploy/cluster` defaulted the CPU node to `4vcpu-16gb` while the
Physical AI Data Factory spec — the copy-paste path in the README and the
runbook — requests `cpus: 4` / `memory: 16Gi`. A 4 vCPU node advertises about 3.9
allocatable CPU and ~15Gi after kubelet/system reserve, so the stock submit could
never be scheduled: `ResourcesUnavailableError`, on a cluster npa itself had just
provisioned to the documented shape.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIABLES_TF = REPO_ROOT / "deploy" / "cluster" / "variables.tf"
QUICKSTART_SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "physical-ai-data-factory.yaml"
)

#: Headroom a Kubernetes node never offers to pods (kubelet + system reserve).
#: Nebius presets advertise roughly `vcpu - 0.1` CPU and ~1Gi less memory; keep a
#: deliberately conservative margin so a spec that only *just* fits still fails.
CPU_RESERVE = 0.5
MEMORY_RESERVE_GIB = 2.0


def _terraform_default(name: str) -> str:
    text = VARIABLES_TF.read_text(encoding="utf-8")
    block = re.search(rf'variable "{name}" \{{(.*?)\n\}}', text, re.DOTALL)
    assert block, f"variable {name!r} not found in {VARIABLES_TF}"
    default = re.search(r'default\s*=\s*"([^"]+)"', block.group(1))
    assert default, f"variable {name!r} has no string default"
    return default.group(1)


def _preset_capacity(preset: str) -> tuple[float, float]:
    """Return ``(vcpu, memory_gib)`` from a Nebius preset name like ``8vcpu-32gb``."""
    match = re.match(r"(\d+)vcpu-(\d+)gb", preset)
    assert match, f"unrecognized CPU preset {preset!r}"
    return float(match.group(1)), float(match.group(2))


def _memory_gib(value: str) -> float:
    match = re.match(r"([\d.]+)\s*(Gi|G|Mi|M)?$", str(value).strip())
    assert match, f"unrecognized memory request {value!r}"
    amount = float(match.group(1))
    unit = match.group(2) or "Gi"
    if unit in {"Mi", "M"}:
        return amount / 1024
    return amount


def _cpu_profiles() -> list[tuple[str, dict]]:
    spec = yaml.safe_load(QUICKSTART_SPEC.read_text(encoding="utf-8")) or {}
    resources = spec.get("resources") or {}
    return [
        (name, profile)
        for name, profile in resources.items()
        if isinstance(profile, dict)
        and not str(profile.get("accelerators", "") or "").strip()
    ]


def test_default_cpu_node_can_schedule_the_quickstart_spec() -> None:
    vcpu, memory_gib = _preset_capacity(_terraform_default("cpu_nodes_preset"))
    profiles = _cpu_profiles()
    assert profiles, "no CPU-only resource profiles found in the quickstart spec"

    for name, profile in profiles:
        requested_cpu = float(profile.get("cpus", 0) or 0)
        requested_memory = _memory_gib(profile.get("memory", "0Gi"))
        assert requested_cpu <= vcpu - CPU_RESERVE, (
            f"resource profile {name!r} requests {requested_cpu} CPU, which a "
            f"{vcpu:g}-vCPU node cannot schedule after kubelet reserve. Raise "
            "cpu_nodes_preset in deploy/cluster/variables.tf or lower the request."
        )
        assert requested_memory <= memory_gib - MEMORY_RESERVE_GIB, (
            f"resource profile {name!r} requests {requested_memory:g}Gi, which a "
            f"{memory_gib:g}Gi node cannot schedule after kubelet reserve."
        )


def test_gpu_profiles_fit_the_default_gpu_preset() -> None:
    """The GPU node group has the same trap: its preset carries CPU/memory too."""
    gpu_preset = _terraform_default("gpu_nodes_preset")
    match = re.match(r"(\d+)gpu-(\d+)vcpu-(\d+)gb", gpu_preset)
    assert match, f"unrecognized GPU preset {gpu_preset!r}"
    vcpu, memory_gib = float(match.group(2)), float(match.group(3))

    spec = yaml.safe_load(QUICKSTART_SPEC.read_text(encoding="utf-8")) or {}
    for name, profile in (spec.get("resources") or {}).items():
        if (
            not isinstance(profile, dict)
            or not str(profile.get("accelerators", "") or "").strip()
        ):
            continue
        assert float(profile.get("cpus", 0) or 0) <= vcpu - CPU_RESERVE, name
        assert (
            _memory_gib(profile.get("memory", "0Gi")) <= memory_gib - MEMORY_RESERVE_GIB
        ), name


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("8vcpu-32gb", (8.0, 32.0)),
        ("4vcpu-16gb", (4.0, 16.0)),
        ("16vcpu-64gb", (16.0, 64.0)),
    ],
)
def test_preset_parsing(preset: str, expected: tuple[float, float]) -> None:
    assert _preset_capacity(preset) == expected


#: What the SkyPilot managed-jobs controller parks on the CPU pool for the whole
#: run. Import the production values so this scheduling guard covers the exact
#: topology submit renders instead of maintaining a second default.
from npa.orchestration.skypilot.controller import (  # noqa: E402
    DEFAULT_K8S_CONTROLLER_CPUS as JOBS_CONTROLLER_CPUS,
    DEFAULT_K8S_CONTROLLER_MEMORY_GB as JOBS_CONTROLLER_MEMORY_GIB,
)


def test_the_cpu_node_fits_the_jobs_controller_and_a_stage_together() -> None:
    """One CPU node must hold the controller and a CPU stage at the same time.

    With only the controller budgeted, a run whose GPU nodes went away left
    `generate-configs` unschedulable on the one remaining CPU node -- reported
    only as an opaque PENDING.
    """

    vcpu, memory_gib = _preset_capacity(_terraform_default("cpu_nodes_preset"))
    spec = yaml.safe_load(QUICKSTART_SPEC.read_text(encoding="utf-8"))

    cpu_profiles = [
        profile
        for profile in (spec.get("resources") or {}).values()
        if isinstance(profile, dict) and not profile.get("accelerators")
    ]
    assert cpu_profiles, "the quickstart declares no CPU-only resource profile"

    for profile in cpu_profiles:
        needed_cpus = float(profile.get("cpus", 0) or 0) + JOBS_CONTROLLER_CPUS
        needed_memory = (
            _memory_gib(profile.get("memory", "0Gi")) + JOBS_CONTROLLER_MEMORY_GIB
        )
        assert needed_cpus <= vcpu - CPU_RESERVE, (
            f"controller ({JOBS_CONTROLLER_CPUS} CPU) + stage ({profile.get('cpus')}) "
            f"exceeds the default CPU node ({vcpu} vCPU)"
        )
        assert needed_memory <= memory_gib - MEMORY_RESERVE_GIB, (
            f"controller ({JOBS_CONTROLLER_MEMORY_GIB}Gi) + stage ({profile.get('memory')}) "
            f"exceeds the default CPU node ({memory_gib}Gi)"
        )


def test_npa_and_terraform_agree_on_the_default_cpu_preset() -> None:
    from npa.cluster.config import DEFAULT_CPU_NODE_GROUP_PRESET

    assert _terraform_default("cpu_nodes_preset") == DEFAULT_CPU_NODE_GROUP_PRESET


def test_paidf_self_provisioning_keeps_the_controller_cpu_node_explicit() -> None:
    spec = yaml.safe_load(QUICKSTART_SPEC.read_text(encoding="utf-8"))
    directive = spec["resources"]["gpu"]["deployIfAbsent"]
    assert directive == {
        "cpuNodes": 1,
        "cpuPlatform": "cpu-d3",
        "cpuPreset": "8vcpu-32gb",
        "gpuNodes": 1,
        "gpuPlatform": "gpu-rtx6000",
        "gpuPreset": "1gpu-24vcpu-218gb",
    }


def test_paidf_declared_cpu_preset_clears_the_allocatable_preflight_threshold() -> None:
    """The self-provisioned preset cannot deadlock its own placement gate."""

    from npa.orchestration.npa_workflow.paidf_preflight import (
        _PAIDF_CPU_MILLICORES,
        _PAIDF_MEMORY_BYTES,
    )

    spec = yaml.safe_load(QUICKSTART_SPEC.read_text(encoding="utf-8"))
    preset = spec["resources"]["gpu"]["deployIfAbsent"]["cpuPreset"]
    vcpu, memory_gib = _preset_capacity(preset)

    assert int((vcpu - CPU_RESERVE) * 1000) >= _PAIDF_CPU_MILLICORES
    assert int((memory_gib - MEMORY_RESERVE_GIB) * 1024**3) >= _PAIDF_MEMORY_BYTES

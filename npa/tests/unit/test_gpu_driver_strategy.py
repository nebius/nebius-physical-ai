"""GPU driver strategy defaults, safety, and recipe compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.cluster.gpu_driver import (
    GpuDriverStrategyError,
    gpus_per_node,
    resolve_gpu_driver_strategy,
)
from npa.fleet.spec import ClusterSpec, FleetSpecError, NodePoolSpec, spec_from_mapping
from npa.fleet.tfvars import render_tfvars


@pytest.mark.parametrize(
    ("platform", "preset", "nodes", "expected_gpus"),
    [
        ("gpu-rtx6000", "1gpu-24vcpu-218gb", 3, 3),
        ("gpu-h200-sxm", "8gpu-128vcpu-1600gb", 2, 16),
        ("gpu-b300-sxm", "8gpu-192vcpu-2768gb", 4, 32),
    ],
)
def test_auto_is_generalized_managed_image_default(
    platform: str, preset: str, nodes: int, expected_gpus: int
) -> None:
    selection = resolve_gpu_driver_strategy(
        gpu_nodes=nodes, platform=platform, preset=preset
    )
    assert selection.effective_mode == "managed-image"
    assert selection.managed_driver_preset == "cuda13.0"
    assert nodes * gpus_per_node(preset) == expected_gpus


def test_cpu_only_strategy_is_inert_and_emits_no_recipe_gpu_input() -> None:
    cluster = ClusterSpec(
        name="cpu", cpu_nodes=NodePoolSpec(count=2), gpu_driver_mode="managed-image"
    )
    cluster.validate()
    tfvars = render_tfvars(cluster)
    assert "gpu_nodes_driverfull_image" not in tfvars
    assert "gpu_nodes_driver_preset" not in tfvars


def test_operator_requires_explicit_nvswitch_acknowledgement() -> None:
    cluster = ClusterSpec(
        name="unsafe",
        gpu_nodes=NodePoolSpec(
            count=1, platform="gpu-h200-sxm", preset="8gpu-128vcpu-1600gb"
        ),
        gpu_driver_mode="operator",
        infiniband_fabric="fabric-a",
    )
    with pytest.raises(FleetSpecError, match="unsafe for this NVSwitch topology"):
        cluster.validate()

    cluster.allow_unsafe_nvswitch_operator = True
    cluster.validate()
    tfvars = render_tfvars(cluster)
    assert "gpu_nodes_driverfull_image   = false" in tfvars


@pytest.mark.parametrize("mode", ["none", "automatic", "managed", "OPERATOR!"])
def test_invalid_driver_mode_fails_closed(mode: str) -> None:
    with pytest.raises(GpuDriverStrategyError, match="unsupported GPU driver mode"):
        resolve_gpu_driver_strategy(
            gpu_nodes=1,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            mode=mode,
        )


def _recipe(
    root: Path,
    *,
    driver_flag: bool = True,
    preset_variable: bool = True,
    managed_wiring: bool = True,
    device_plugin: bool = True,
    operator: bool = True,
    fixed_preset: str = "",
) -> Path:
    root.mkdir(parents=True)
    variables = []
    if driver_flag:
        variables.append('variable "gpu_nodes_driverfull_image" { type = bool }')
    if preset_variable:
        variables.append('variable "gpu_nodes_driver_preset" { type = string }')
    (root / "variables.tf").write_text("\n".join(variables))
    (root / "main.tf").write_text(
        "gpu_settings = { drivers_preset = local.device_preset }"
        if managed_wiring
        else ""
    )
    helm = []
    if device_plugin:
        helm.append('module "device-plugin" {}')
    if operator:
        helm.append('module "gpu-operator" {}')
    (root / "helm.tf").write_text("\n".join(helm))
    (root / "locals.tf").write_text(
        f'locals {{\n  device_preset = "{fixed_preset}"\n}}\n' if fixed_preset else ""
    )
    return root


def _gpu_cluster(**overrides) -> ClusterSpec:
    values = {
        "name": "gpu",
        "gpu_nodes": NodePoolSpec(
            count=2, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb"
        ),
    }
    values.update(overrides)
    return ClusterSpec(**values)


def test_alternate_recipe_receives_configurable_managed_preset(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path / "recipe")
    tfvars = render_tfvars(
        _gpu_cluster(managed_driver_preset="cuda13.7"), recipe_dir=recipe
    )
    assert "gpu_nodes_driverfull_image   = true" in tfvars
    assert 'gpu_nodes_driver_preset     = "cuda13.7"' in tfvars


def test_legacy_fixed_preset_is_compatible_only_when_exact(tmp_path: Path) -> None:
    recipe = _recipe(
        tmp_path / "legacy", preset_variable=False, fixed_preset="cuda13.0"
    )
    compatible = render_tfvars(_gpu_cluster(), recipe_dir=recipe)
    assert "gpu_nodes_driverfull_image   = true" in compatible
    assert "gpu_nodes_driver_preset" not in compatible

    with pytest.raises(GpuDriverStrategyError, match="cannot select managed driver"):
        render_tfvars(_gpu_cluster(managed_driver_preset="cuda13.7"), recipe_dir=recipe)


@pytest.mark.parametrize(
    ("recipe_kwargs", "message"),
    [
        ({"driver_flag": False}, "gpu_nodes_driverfull_image"),
        ({"managed_wiring": False}, "gpu_settings.drivers_preset"),
        ({"device_plugin": False}, "device-plugin"),
    ],
)
def test_incompatible_managed_recipe_fails_before_terraform(
    tmp_path: Path, recipe_kwargs: dict[str, bool], message: str
) -> None:
    recipe = _recipe(tmp_path / "recipe", **recipe_kwargs)
    with pytest.raises(GpuDriverStrategyError, match=message):
        render_tfvars(_gpu_cluster(), recipe_dir=recipe)


def test_operator_recipe_requires_operator_component(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path / "recipe", operator=False)
    with pytest.raises(GpuDriverStrategyError, match="no GPU Operator"):
        render_tfvars(_gpu_cluster(gpu_driver_mode="operator"), recipe_dir=recipe)


def test_operator_only_alternate_recipe_needs_no_managed_image_flag(
    tmp_path: Path,
) -> None:
    recipe = _recipe(tmp_path / "recipe", driver_flag=False)
    tfvars = render_tfvars(_gpu_cluster(gpu_driver_mode="operator"), recipe_dir=recipe)
    assert "gpu_nodes_driverfull_image" not in tfvars
    assert "gpu_nodes_driver_preset" not in tfvars


def test_fleet_yaml_defaults_and_explicit_driver_overrides() -> None:
    spec = spec_from_mapping(
        {
            "apiVersion": "npa.fleet/v0.0.1",
            "name": "drivers",
            "defaults": {
                "gpu_nodes": {
                    "count": 1,
                    "platform": "gpu-rtx6000",
                    "preset": "1gpu-24vcpu-218gb",
                },
                "gpu_driver_mode": "managed-image",
                "managed_driver_preset": "cuda13.7",
                "gpu_health_stabilization_seconds": 45,
                "gpu_cuda_smoke": False,
            },
            "projects": [
                {
                    "name": "a",
                    "clusters": [
                        {},
                        {
                            "name": "debug",
                            "gpu_driver_mode": "operator",
                        },
                    ],
                }
            ],
        }
    )
    spec.validate()
    managed, operator = spec.projects[0].clusters
    assert managed.gpu_driver_mode == "managed-image"
    assert managed.managed_driver_preset == "cuda13.7"
    assert managed.gpu_health_stabilization_seconds == 45
    assert managed.gpu_cuda_smoke is False
    assert operator.gpu_driver_mode == "operator"

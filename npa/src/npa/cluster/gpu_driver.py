"""Shared GPU-driver strategy for NPA-provisioned Managed Kubernetes clusters.

Nebius Managed Kubernetes can boot GPU nodes from a driver-full image through
``gpu_settings.drivers_preset``.  That path brings the host driver and NVSwitch
fabric services up with the node image, before Kubernetes advertises GPUs.  It
avoids the ordering race between the in-cluster GPU/Network Operators and the
host InfiniBand devices that can leave an NVSwitch node in managed recovery.

Both ``npa cluster up`` and ``npa fleet deploy`` resolve this same contract.
Fleet additionally inspects alternate k8s-training recipes so a selected mode
cannot be silently dropped by an older or incompatible variable surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

GPU_DRIVER_MODES = frozenset({"auto", "managed-image", "operator"})
DEFAULT_MANAGED_DRIVER_PRESET = "cuda13.0"
_PRESET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GPU_COUNT_RE = re.compile(r"^(\d+)gpu-")
_VARIABLE_RE = re.compile(r'(?m)^\s*variable\s+"([^"]+)"\s*\{')
_FIXED_PRESET_RE = re.compile(
    r'(?m)^\s*(?:device_preset|gpu_nodes_driver_preset)\s*=\s*"([^"]+)"\s*$'
)


class GpuDriverStrategyError(ValueError):
    """Raised when the requested strategy is unsafe or recipe-incompatible."""


@dataclass(frozen=True)
class GpuDriverSelection:
    """Resolved driver behavior for one requested cluster topology."""

    requested_mode: str
    effective_mode: str
    managed_driver_preset: str
    gpu_enabled: bool
    nvswitch: bool
    unsafe_operator_acknowledged: bool = False

    @property
    def uses_managed_image(self) -> bool:
        return self.gpu_enabled and self.effective_mode == "managed-image"


@dataclass(frozen=True)
class RecipeGpuDriverCapabilities:
    """Relevant k8s-training recipe inputs and implementation hooks."""

    driverfull_flag: bool
    managed_preset_variable: bool
    managed_gpu_settings: bool
    managed_device_plugin: bool
    operator_component: bool
    fixed_managed_preset: str = ""

    @property
    def supports_managed_image(self) -> bool:
        return (
            self.driverfull_flag
            and self.managed_gpu_settings
            and self.managed_device_plugin
        )


CANONICAL_RECIPE_CAPABILITIES = RecipeGpuDriverCapabilities(
    driverfull_flag=True,
    managed_preset_variable=True,
    managed_gpu_settings=True,
    managed_device_plugin=True,
    operator_component=True,
)


def gpus_per_node(preset: str) -> int:
    """Return the generalized GPU count encoded by a Nebius preset."""

    match = _GPU_COUNT_RE.match(str(preset or "").strip().lower())
    return int(match.group(1)) if match else 0


def is_nvswitch_topology(
    *, platform: str, preset: str, enable_gpu_cluster: bool = False
) -> bool:
    """Whether a requested node topology uses an NVSwitch-class GPU system.

    ``enable_gpu_cluster`` is authoritative when set.  The multi-GPU SXM/NVL
    shape also catches a single-node NVSwitch system, where no cross-node GPU
    cluster object is required.
    """

    normalized = str(platform or "").strip().lower()
    return bool(
        enable_gpu_cluster
        or (
            gpus_per_node(preset) > 1 and ("-sxm" in normalized or "-nvl" in normalized)
        )
    )


def resolve_gpu_driver_strategy(
    *,
    gpu_nodes: int,
    platform: str,
    preset: str,
    mode: str = "auto",
    managed_driver_preset: str = DEFAULT_MANAGED_DRIVER_PRESET,
    enable_gpu_cluster: bool = False,
    allow_unsafe_nvswitch_operator: bool = False,
) -> GpuDriverSelection:
    """Validate and resolve the stable ``auto|managed-image|operator`` contract."""

    requested_mode = str(mode or "auto").strip().lower()
    if requested_mode not in GPU_DRIVER_MODES:
        choices = ", ".join(sorted(GPU_DRIVER_MODES))
        raise GpuDriverStrategyError(
            f"unsupported GPU driver mode {mode!r}; expected one of {choices}"
        )

    driver_preset = str(managed_driver_preset or DEFAULT_MANAGED_DRIVER_PRESET).strip()
    if not driver_preset or not _PRESET_RE.fullmatch(driver_preset):
        raise GpuDriverStrategyError(
            "managed GPU driver preset must start with an alphanumeric character "
            "and contain only letters, digits, '.', '_' or '-'"
        )

    gpu_enabled = int(gpu_nodes) > 0
    nvswitch = gpu_enabled and is_nvswitch_topology(
        platform=platform,
        preset=preset,
        enable_gpu_cluster=enable_gpu_cluster,
    )
    effective_mode = (
        "managed-image" if gpu_enabled and requested_mode == "auto" else requested_mode
    )
    if not gpu_enabled:
        # The selection is intentionally inert for CPU-only clusters.  Keeping
        # the requested value in the report makes plan output truthful without
        # requiring a GPU recipe surface or deploying GPU components.
        effective_mode = requested_mode

    if (
        gpu_enabled
        and effective_mode == "operator"
        and nvswitch
        and not allow_unsafe_nvswitch_operator
    ):
        raise GpuDriverStrategyError(
            "GPU driver mode 'operator' is unsafe for this NVSwitch topology: "
            "the in-cluster driver/Fabric Manager can race host InfiniBand device "
            "creation and enter recurring managed recovery. Use 'auto' or "
            "'managed-image'. For a controlled diagnostic only, explicitly set "
            "allow_unsafe_nvswitch_operator=true and recreate the GPU node group."
        )

    return GpuDriverSelection(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        managed_driver_preset=driver_preset,
        gpu_enabled=gpu_enabled,
        nvswitch=nvswitch,
        unsafe_operator_acknowledged=bool(
            gpu_enabled
            and effective_mode == "operator"
            and nvswitch
            and allow_unsafe_nvswitch_operator
        ),
    )


def inspect_recipe_gpu_driver_capabilities(
    recipe_dir: Path,
) -> RecipeGpuDriverCapabilities:
    """Inspect a materialized k8s-training root without executing Terraform."""

    variables_path = recipe_dir / "variables.tf"
    main_path = recipe_dir / "main.tf"
    helm_path = recipe_dir / "helm.tf"
    locals_path = recipe_dir / "locals.tf"
    variables = variables_path.read_text() if variables_path.is_file() else ""
    main = main_path.read_text() if main_path.is_file() else ""
    helm = helm_path.read_text() if helm_path.is_file() else ""
    locals_tf = locals_path.read_text() if locals_path.is_file() else ""
    declared = set(_VARIABLE_RE.findall(variables))
    fixed = _FIXED_PRESET_RE.search(locals_tf)
    return RecipeGpuDriverCapabilities(
        driverfull_flag="gpu_nodes_driverfull_image" in declared,
        managed_preset_variable="gpu_nodes_driver_preset" in declared,
        managed_gpu_settings=("gpu_settings" in main and "drivers_preset" in main),
        managed_device_plugin=(
            'module "device-plugin"' in helm or "nvidia-device-plugin" in helm
        ),
        operator_component=(
            'module "gpu-operator"' in helm or "nvidia-gpu-operator" in helm
        ),
        fixed_managed_preset=fixed.group(1) if fixed else "",
    )


def validate_recipe_gpu_driver_compatibility(
    selection: GpuDriverSelection,
    capabilities: RecipeGpuDriverCapabilities,
    *,
    recipe_label: str,
) -> None:
    """Fail before apply if an alternate recipe cannot honor *selection*."""

    if not selection.gpu_enabled:
        return
    if selection.effective_mode == "operator":
        if not capabilities.operator_component:
            raise GpuDriverStrategyError(
                f"{recipe_label} cannot honor operator GPU drivers: no GPU Operator "
                "component was detected. Use the vendored recipe or a compatible "
                "upstream ref."
            )
        return
    if not capabilities.driverfull_flag:
        raise GpuDriverStrategyError(
            f"{recipe_label} cannot honor GPU driver mode "
            f"{selection.requested_mode!r}: it does not declare "
            "variable 'gpu_nodes_driverfull_image'. Use the vendored recipe or "
            "an upstream ref with the managed-driver strategy inputs."
        )
    if not capabilities.supports_managed_image:
        missing: list[str] = []
        if not capabilities.managed_gpu_settings:
            missing.append("gpu_settings.drivers_preset wiring")
        if not capabilities.managed_device_plugin:
            missing.append("managed-image NVIDIA device-plugin installation")
        raise GpuDriverStrategyError(
            f"{recipe_label} cannot honor managed-image GPU drivers; missing "
            + ", ".join(missing)
            + ". Use the vendored recipe or a compatible upstream ref."
        )
    if capabilities.managed_preset_variable:
        return
    if capabilities.fixed_managed_preset == selection.managed_driver_preset:
        return
    fixed = capabilities.fixed_managed_preset or "not detectable"
    raise GpuDriverStrategyError(
        f"{recipe_label} cannot select managed driver preset "
        f"{selection.managed_driver_preset!r}: it has no "
        "'gpu_nodes_driver_preset' input (fixed preset: "
        f"{fixed!r}). Use the vendored recipe, a compatible upstream ref, or "
        "select the recipe's fixed preset explicitly."
    )


def recipe_driver_tfvars(
    selection: GpuDriverSelection,
    capabilities: RecipeGpuDriverCapabilities,
    *,
    recipe_label: str,
) -> list[str]:
    """Render only the compatible recipe inputs for a resolved selection."""

    validate_recipe_gpu_driver_compatibility(
        selection, capabilities, recipe_label=recipe_label
    )
    if not selection.gpu_enabled:
        return []
    managed = selection.effective_mode == "managed-image"
    lines = []
    if capabilities.driverfull_flag:
        lines.append(
            "gpu_nodes_driverfull_image   = " + ("true" if managed else "false")
        )
    if managed and capabilities.managed_preset_variable:
        escaped = selection.managed_driver_preset.replace("\\", "\\\\").replace(
            '"', '\\"'
        )
        lines.append(f'gpu_nodes_driver_preset     = "{escaped}"')
    return lines

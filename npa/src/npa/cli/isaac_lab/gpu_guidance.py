"""Describe the GPU choices supported by an Isaac Lab workbench deployment."""


def gpu_selection_error() -> str:
    """Explain the explicit GPU selection required by a rendered workbench.

    Args:
        None.
    Returns:
        CLI guidance for selecting an RT-capable deployment GPU.
    Raises:
        None.
    """
    return (
        "GPU selection is required for Isaac Lab deploy. Provide --gpu-type and --gpu-preset.\n"
        "  Suggested starting points:\n"
        "    Simulation workloads (L40S): --gpu-type gpu-l40s-a --gpu-preset 1gpu-40vcpu-160gb\n"
        "    RTX Pro 6000 fallback: --gpu-type gpu-rtx-pro-6000 --gpu-preset 1gpu-24vcpu-218gb\n"
        "  A deployed workbench is the render surface, so it needs RT cores; headless\n"
        "  training may select H100/H200/B200 via `isaac-lab train --runtime serverless`."
    )

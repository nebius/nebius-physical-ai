"""SDK wrappers for the Workbench NuRec / NRE CLI.

Neural reconstruction: a real sensor capture (NCore V4) becomes a 3D Gaussian
reconstruction, a renderable USDZ, and novel-view renders, driven by NVIDIA's
public NRE container on an RT-core GPU. See
``skills/workflows/neural-reconstruction/SKILL.md``.
"""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

check = make_cli_wrapper(
    "npa.cli.nurec",
    "check_cmd",
    "Check NRE container access, dataset download rights, and GPU suitability.",
)
fetch = make_cli_wrapper(
    "npa.cli.nurec",
    "fetch_cmd",
    "Download real NCore V4 shards and derive the rig->world pose edge NRE needs.",
)
reconstruct = make_cli_wrapper(
    "npa.cli.nurec",
    "reconstruct_cmd",
    "Train a 3DGUT Gaussian reconstruction into a renderable USDZ.",
)
render = make_cli_wrapper(
    "npa.cli.nurec",
    "render_cmd",
    "Render novel views from a trained reconstruction.",
)
visualize = make_cli_wrapper(
    "npa.cli.nurec",
    "visualize_cmd",
    "Build the run's Rerun recording for the NPA agent viewer.",
)
finalize = make_cli_wrapper(
    "npa.cli.nurec", "finalize_cmd", "Aggregate a NuRec run tree into a final report."
)
status = make_cli_wrapper(
    "npa.cli.nurec", "status_cmd", "Summarize a NuRec run prefix, stage by stage."
)

__all__ = [
    "check",
    "fetch",
    "finalize",
    "reconstruct",
    "render",
    "status",
    "visualize",
]

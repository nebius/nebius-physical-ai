"""Content Agents render stages restore OVRTX's Kubernetes runtime contract."""

from npa.orchestration.npa_workflow.skypilot_render import (
    render_run_preamble_for_tool,
)


def test_render_stages_require_graphics_mounts_and_start_xvfb() -> None:
    for tool_ref in (
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
    ):
        preamble = render_run_preamble_for_tool(tool_ref, config={})

        assert 'ctypes.CDLL("libGLX_nvidia.so.0")' in preamble
        assert "GPU Operator graphics driver mounts" in preamble
        assert "/usr/local/bin/npa-content-agents-entrypoint /bin/true" in preamble
        assert 'export DISPLAY=":$npa_ovrtx_display"' in preamble
        assert 'npa_ovrtx_lock="/tmp/.X$npa_ovrtx_display-lock"' in preamble
        assert "trap npa_cleanup_ovrtx_display EXIT" in preamble
        assert 'kill "$npa_ovrtx_xvfb_pid"' in preamble


def test_cpu_stages_do_not_start_a_display_or_require_driver_mounts() -> None:
    for tool_ref in (
        "workbench.content_agents.acquire",
        "workbench.content_agents.package",
    ):
        assert render_run_preamble_for_tool(tool_ref, config={}) == ""

"""Content Agents render stages restore OVRTX's Kubernetes runtime contract."""

from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_run_preamble_for_tool,
    render_setup_for_tool,
)


def test_render_stages_require_graphics_mounts_and_start_xvfb() -> None:
    for tool_ref in (
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
    ):
        preamble = render_run_preamble_for_tool(tool_ref, config={})

        bootstrap = (
            "/opt/venv/bin/python -m npa.workflows.content_agents bootstrap-runtime"
        )
        assert preamble.startswith('if [ -n "$PYTHONPATH" ]')
        assert "/opt/npa-runtime:/opt/content-agents:" in preamble
        assert preamble.index("/opt/npa-runtime") < preamble.index(bootstrap)
        assert 'ctypes.CDLL("libGLX_nvidia.so.0")' in preamble
        assert preamble.index(bootstrap) < preamble.index("libGLX_nvidia.so.0")
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
        preamble = render_run_preamble_for_tool(tool_ref, config={})
        assert "/opt/npa-runtime:/opt/content-agents:" in preamble
        assert "bootstrap-runtime" not in preamble
        assert "libGLX_nvidia.so.0" not in preamble
        assert "Xvfb" not in preamble


def test_all_content_agent_stages_verify_the_narrow_baked_runtime() -> None:
    for tool_ref in (
        "workbench.content_agents.acquire",
        "workbench.content_agents.materials",
        "workbench.content_agents.physics",
        "workbench.content_agents.validate",
        "workbench.content_agents.package",
    ):
        setup = render_setup_for_tool(
            tool_ref, config={}, options=SkypilotRenderOptions()
        )

        assert 'npa_baked_python="/opt/venv/bin/python"' in setup
        assert "/opt/npa-runtime:/opt/content-agents:" in setup
        assert "from npa.workflows.content_agents import inspect_image" in setup
        assert "Content Agents narrow baked runtime verified" in setup
        assert "/tmp/npa-python" in setup
        assert "command -v npa" not in setup
        assert "NPA_SRC_S3_URI" not in setup
        assert "bootstrap-runtime" not in setup

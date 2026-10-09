"""Isolate the NuRec adapter's Python dependencies from NVIDIA's native runtime."""

_NATIVE_TOOLS = frozenset(
    f"workbench.nurec.{verb}" for verb in ("check", "fetch", "reconstruct", "render")
)


def render_nurec_adapter_setup(tool_ref: str) -> str:
    """Prepare a separate adapter interpreter before installing staged NPA source.

    Args:
        tool_ref: The workflow stage's resolved tool reference.
    Returns:
        Worker setup shell, or an empty string for non-native stages.
    Raises:
        None.
    """
    if tool_ref not in _NATIVE_TOOLS:
        return ""
    # NRE's Debian packages lack pip RECORD metadata. Even pip's PEP 668
    # override cannot upgrade them safely; the adapter must own its dependencies.
    return (
        "set -e\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        'npa_nurec_base_python="$(command -v python3)"\n'
        '"$npa_nurec_base_python" -m venv --without-pip /tmp/npa-nurec-venv\n'
        "if command -v uv >/dev/null 2>&1; then\n"
        "  uv pip install -q --python /tmp/npa-nurec-venv/bin/python pip\n"
        'elif "$npa_nurec_base_python" -m pip --version >/dev/null 2>&1; then\n'
        '  "$npa_nurec_base_python" -m pip --python /tmp/npa-nurec-venv/bin/python install -q pip\n'
        "else\n"
        "  /tmp/npa-nurec-venv/bin/python -m ensurepip --upgrade\n"
        "fi\n"
        "export NPA_BAKED_PYTHON=/tmp/npa-nurec-venv/bin/python\n"
        "export NPA_LIGHT_WORKBENCH_TOOL=nurec\n"
        'export PATH="/tmp/npa-nurec-venv/bin:$PATH"\n'
    )

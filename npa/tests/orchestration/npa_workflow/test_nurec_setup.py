"""Exercise NuRec adapter isolation without changing the vendor interpreter."""

import json
import os
import shlex
import subprocess
import sys
import sysconfig

import pytest

from npa.orchestration.npa_workflow.nurec_setup import render_nurec_adapter_setup
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_setup_for_tool,
)


@pytest.mark.parametrize("verb", ["check", "fetch", "reconstruct", "render"])
def test_native_setup_isolates_before_installing_npa(verb):
    """Verify install ordering. Args: verb. Returns: None. Raises: AssertionError."""
    setup = render_setup_for_tool(
        f"workbench.nurec.{verb}", config={}, options=SkypilotRenderOptions()
    )
    assert setup.index("-m venv --without-pip") < setup.index("npa_pip_install()")
    assert setup.index("export NPA_BAKED_PYTHON=") < setup.index("npa_setup_python=")
    assert setup.index("export NPA_LIGHT_WORKBENCH_TOOL=nurec") < setup.index(
        "npa_pip_install()"
    )


@pytest.mark.parametrize(
    "tool_ref",
    [
        "workbench.nurec.visualize",
        "workbench.nurec.finalize",
        "workbench.nurec.convert_colmap",
        "workbench.lerobot.train",
    ],
)
def test_other_stages_keep_their_declared_runtime(tool_ref):
    """Preserve other runtimes. Args: tool_ref. Returns: None. Raises: AssertionError."""
    assert render_nurec_adapter_setup(tool_ref) == ""


def test_adapter_installs_without_access_to_vendor_site_packages(tmp_path):
    """Run isolated setup. Args: tmp_path. Returns: None. Raises: AssertionError."""
    binary = tmp_path / "bin"
    binary.mkdir()
    parent = binary / "python3"
    parent.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = -m ] && [ "$2" = pip ]; then exit 1; fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    parent.chmod(0o755)
    target = tmp_path / "adapter"
    setup = render_nurec_adapter_setup("workbench.nurec.check")
    venv_command = next(
        line for line in setup.splitlines() if "-m venv --without-pip" in line
    )
    setup = setup.replace(shlex.split(venv_command)[-1], str(target))
    probe = 'import json,sys; print(json.dumps({"prefix":sys.prefix,"paths":sys.path}))'
    script = setup + f'"$NPA_BAKED_PYTHON" -c {shlex.quote(probe)}\n'
    environment = dict(os.environ, HOME=str(tmp_path), PATH=str(binary))
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["/bin/bash", "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["prefix"] == str(target)
    assert sysconfig.get_path("purelib") not in observed["paths"]
    installed = subprocess.run(
        [str(target / "bin/python"), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(target) in installed.stdout

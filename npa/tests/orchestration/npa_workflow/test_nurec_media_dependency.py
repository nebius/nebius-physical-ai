"""Bind NuRec receipt decoding to its installed and executable runtime dependency."""

from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from npa.orchestration.npa_workflow.skypilot_render import (
    NUREC_FFMPEG_PIN,
    SkypilotRenderOptions,
    render_setup_for_tool,
)


def _setup(tool: str = "render") -> str:
    return render_setup_for_tool(
        f"workbench.nurec.{tool}", config={}, options=SkypilotRenderOptions()
    )


def _probe() -> str:
    line = next(
        line for line in _setup().splitlines() if "nurec runtime deps ready" in line
    )
    command = shlex.split(line)
    assert command[:2] == ["$npa_nurec_py", "-c"]
    return command[2]


@pytest.mark.parametrize("tool", ["reconstruct", "render", "visualize"])
def test_nurec_installs_decoder_into_recorded_interpreter(tool):
    setup = _setup(tool)
    assert NUREC_FFMPEG_PIN == "imageio-ffmpeg==0.6.0"
    install = next(line for line in setup.splitlines() if NUREC_FFMPEG_PIN in line)
    assert install.startswith("npa_nurec_pip ")
    assert "import ncore, rerun, imageio_ffmpeg" in setup
    assert "imageio_ffmpeg.get_ffmpeg_exe()" in setup


def test_nurec_audit_declares_same_decoder_pin():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    extras = tomllib.loads(project.read_text())["project"]["optional-dependencies"]
    assert NUREC_FFMPEG_PIN in extras["nurec-audit"]


def test_nurec_decoder_setup_does_not_change_other_tool_setup():
    setup = render_setup_for_tool(
        "workbench.vlm_eval.run", config={}, options=SkypilotRenderOptions()
    )
    assert NUREC_FFMPEG_PIN not in setup
    assert "nurec runtime deps ready" not in setup


def _run_probe(tmp_path: Path, *, setup: str = "") -> subprocess.CompletedProcess[str]:
    """Run the generated runtime probe in its own recorded Python interpreter."""
    script = tmp_path / "nurec_dependency_probe.py"
    script.write_text(
        "import sys\n"
        "from types import ModuleType\n"
        "for name in ('ncore', 'rerun'):\n"
        "    sys.modules[name] = ModuleType(name)\n" + setup + "\n" + _probe() + "\n"
    )
    return subprocess.run(
        [sys.executable, "-I", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_nurec_probe_requires_decoder_import(tmp_path):
    result = _run_probe(tmp_path, setup="sys.modules['imageio_ffmpeg'] = None\n")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "imageio_ffmpeg" in result.stderr
    assert "nurec runtime deps ready" not in result.stdout


def test_nurec_probe_requires_actual_ffmpeg_executable(tmp_path):
    result = _run_probe(
        tmp_path,
        setup=(
            "import imageio_ffmpeg\n"
            "def unavailable():\n"
            "    raise RuntimeError('decoder executable unavailable')\n"
            "imageio_ffmpeg.get_ffmpeg_exe = unavailable\n"
        ),
    )
    assert result.returncode != 0
    assert "RuntimeError: decoder executable unavailable" in result.stderr
    assert "nurec runtime deps ready" not in result.stdout


def test_nurec_probe_accepts_installed_decoder(tmp_path):
    result = _run_probe(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "nurec runtime deps ready\n"


def test_nurec_probe_ignores_ambient_pythonpath(tmp_path, monkeypatch):
    (tmp_path / "imageio_ffmpeg.py").write_text(
        "raise RuntimeError('ambient decoder must not replace installed package')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = _run_probe(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "nurec runtime deps ready\n"

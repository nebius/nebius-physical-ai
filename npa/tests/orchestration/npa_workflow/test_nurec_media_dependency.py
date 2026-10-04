"""Bind NuRec receipt decoding to its installed and executable runtime dependency."""

from pathlib import Path
import shlex
import sys
from types import ModuleType

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


def test_nurec_probe_requires_decoder_import(monkeypatch, capsys):
    for name in ("ncore", "rerun"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(ModuleNotFoundError, match="imageio_ffmpeg"):
        exec(_probe(), {})
    assert "nurec runtime deps ready" not in capsys.readouterr().out


def test_nurec_probe_requires_actual_ffmpeg_executable(monkeypatch, capsys):
    import imageio_ffmpeg

    for name in ("ncore", "rerun"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    def unavailable():
        raise RuntimeError("decoder executable unavailable")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", unavailable)
    with pytest.raises(RuntimeError, match="decoder executable unavailable"):
        exec(_probe(), {})
    assert "nurec runtime deps ready" not in capsys.readouterr().out


def test_nurec_probe_accepts_installed_decoder(monkeypatch, capsys):
    for name in ("ncore", "rerun"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    exec(_probe(), {})
    assert capsys.readouterr().out == "nurec runtime deps ready\n"

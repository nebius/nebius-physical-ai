from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from npa.cli import skypilot as skypilot_cli
from npa.cli.main import app


runner = CliRunner()


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_installed_venv(path: Path, *, version: str = "0.12.2") -> Path:
    bin_dir = path / "bin"
    _write_executable(bin_dir / "python", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "pip", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "sky", f"#!/bin/sh\nprintf 'SkyPilot {version}\\n'\n")
    return path


def test_skypilot_registered_under_root_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "skypilot" in result.output


def test_skypilot_bootstrap_idempotent_existing_install(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")

    first = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])
    second = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already installed" in first.output
    assert "already installed" in second.output
    assert str((venv / "bin" / "sky").resolve()) in second.output
    marker = venv / skypilot_cli.MARKER_FILE
    assert marker.exists()
    assert '"version": "0.12.2"' in marker.read_text(encoding="utf-8")


def test_skypilot_bootstrap_rejects_existing_version_mismatch(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", version="0.12.1")

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 1
    assert "Version conflict" in result.output
    assert "0.12.1" in result.output


def test_skypilot_path_can_come_from_flag_or_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    flag_venv = _fake_installed_venv(tmp_path / "flag-venv")
    env_venv = _fake_installed_venv(tmp_path / "env-venv")

    flag_result = runner.invoke(app, ["skypilot", "status", "--path", str(flag_venv), "--bin-path"])
    monkeypatch.setenv(skypilot_cli.VENV_PATH_ENV, str(env_venv))
    env_result = runner.invoke(app, ["skypilot", "status", "--bin-path"])

    assert flag_result.exit_code == 0
    assert flag_result.output.strip() == str((flag_venv / "bin" / "sky").resolve())
    assert env_result.exit_code == 0
    assert env_result.output.strip() == str((env_venv / "bin" / "sky").resolve())


def test_skypilot_bootstrap_reports_missing_python(tmp_path: Path) -> None:
    missing_python = tmp_path / "missing-python"

    result = runner.invoke(
        app,
        [
            "skypilot",
            "bootstrap",
            "--path",
            str(tmp_path / "new-venv"),
            "--python",
            str(missing_python),
        ],
    )

    assert result.exit_code == 1
    assert "Unable to create SkyPilot venv" in result.output
    assert "install Python with venv support" in result.output


def test_skypilot_bootstrap_reports_network_failure_from_pip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = tmp_path / "sky-venv"
    _write_executable(venv / "bin" / "python", "#!/bin/sh\nexit 0\n")
    _write_executable(venv / "bin" / "pip", "#!/bin/sh\nexit 0\n")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1:4] == ["-m", "pip", "install"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Temporary failure in name resolution")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(skypilot_cli.subprocess, "run", fake_run)

    with pytest.raises(skypilot_cli.SkyPilotBootstrapError, match="Network failure"):
        skypilot_cli.bootstrap_skypilot(venv_path=venv)


def test_skypilot_install_package_pins_click_after_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = tmp_path / "sky-venv"
    _write_executable(venv / "bin" / "python", "#!/bin/sh\nexit 0\n")
    _write_executable(venv / "bin" / "pip", "#!/bin/sh\nexit 0\n")
    state = skypilot_cli.inspect_venv(venv)
    installs: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1:4] == ["-m", "pip", "install"]:
            installs.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(skypilot_cli.subprocess, "run", fake_run)

    skypilot_cli._install_package(state, "skypilot==0.12.2")

    assert any(cmd[-1] == "click>=8.1,<8.2" for cmd in installs), installs


def test_skypilot_bootstrap_can_install_local_tiny_package(tmp_path: Path) -> None:
    package_dir = tmp_path / "fake-skypilot"
    sky_pkg = package_dir / "sky"
    sky_pkg.mkdir(parents=True)
    (package_dir / "setup.py").write_text(
        "\n".join(
            [
                "from setuptools import setup",
                "setup(",
                "    name='fake-skypilot',",
                "    version='0.12.2',",
                "    packages=['sky'],",
                "    entry_points={'console_scripts': ['sky=sky.cli:main']},",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    (sky_pkg / "__init__.py").write_text("__version__ = '0.12.2'\n", encoding="utf-8")
    (sky_pkg / "cli.py").write_text(
        "\n".join(
            [
                "def main():",
                "    import sys",
                "    if '--version' in sys.argv:",
                "        print('SkyPilot 0.12.2')",
                "    elif len(sys.argv) > 1 and sys.argv[1] == 'check':",
                "        print('checks passed')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = skypilot_cli.bootstrap_skypilot(
        venv_path=tmp_path / "sky-venv",
        python_bin=sys.executable,
        package_spec=os.fspath(package_dir),
        extras=("test",),
    )

    assert result.installed is True
    assert result.reused is False
    assert result.sky_bin.is_file()
    assert '"extras": [\n    "test"\n  ]' in result.marker_path.read_text(encoding="utf-8")


def _intercept_sky_check(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]):
    """Capture only the `sky check` invocation; delegate other calls.

    ``_run_no_raise`` is also used by ``inspect_venv`` for version/import probes,
    so a blanket stub would make the venv look uninstalled.
    """

    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001 - test stub
        if cmd[-1] == "check":
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(
                cmd, 0, stdout="checks passed", stderr=""
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)


def test_verify_pins_kubeconfig_from_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    kubeconfig = tmp_path / "kube.yaml"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    captured: dict[str, object] = {}
    _intercept_sky_check(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["skypilot", "verify", "--path", str(venv), "--kubeconfig", str(kubeconfig)],
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"][-1] == "check"
    assert captured["env"]["KUBECONFIG"] == str(kubeconfig)


def test_verify_without_kubeconfig_inherits_ambient_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    captured: dict[str, object] = {}
    _intercept_sky_check(monkeypatch, captured)

    result = runner.invoke(app, ["skypilot", "verify", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert captured["env"] is None


def test_verify_fails_clearly_on_missing_kubeconfig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    missing = tmp_path / "absent" / "kube.yaml"

    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001 - test stub
        if cmd[-1] == "check":
            raise AssertionError("sky check must not run when kubeconfig is missing")
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)

    result = runner.invoke(
        app,
        ["skypilot", "verify", "--path", str(venv), "--kubeconfig", str(missing)],
    )

    assert result.exit_code == 1
    assert "Kubeconfig not found" in result.output

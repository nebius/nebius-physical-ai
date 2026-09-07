from __future__ import annotations

import json
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


def _fake_installed_venv(
    path: Path, *, version: str = "0.12.2", kubernetes_version: str = "30.1.0"
) -> Path:
    bin_dir = path / "bin"
    # inspect_venv probes the venv python for `import sky` and the kubernetes
    # client version; both go through the same interpreter stub.
    _write_executable(
        bin_dir / "python",
        "#!/bin/sh\n"
        'case "$2" in\n'
        f"  *kubernetes*) printf '{kubernetes_version}\\n' ;;\n"
        "esac\n"
        "exit 0\n",
    )
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


def test_skypilot_failed_upgrade_preserves_existing_version_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", version="0.12.1")
    before = {
        path.relative_to(venv): path.read_bytes()
        for path in venv.rglob("*")
        if path.is_file()
    }

    def fail_create(_path: Path, _python: object) -> None:
        raise skypilot_cli.SkyPilotBootstrapError("offline resolver failure")

    monkeypatch.setattr(skypilot_cli, "_create_venv", fail_create)

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 1
    assert "offline resolver failure" in result.output
    after = {
        path.relative_to(venv): path.read_bytes()
        for path in venv.rglob("*")
        if path.is_file()
    }
    assert after == before


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

    def fake_create(path: Path, _python: object) -> None:
        _write_executable(path / "bin" / "python", "#!/bin/sh\nexit 0\n")
        _write_executable(path / "bin" / "pip", "#!/bin/sh\nexit 0\n")

    monkeypatch.setattr(skypilot_cli, "_create_venv", fake_create)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1:4] == ["-m", "pip", "install"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Temporary failure in name resolution")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(skypilot_cli.subprocess, "run", fake_run)

    with pytest.raises(skypilot_cli.SkyPilotBootstrapError, match="Network failure"):
        skypilot_cli.bootstrap_skypilot(venv_path=venv)


def test_skypilot_install_package_pins_runtime_dependencies_after_install(
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
    assert any(cmd[-1] == skypilot_cli.KUBERNETES_CLIENT_SPEC for cmd in installs), installs


def test_skypilot_bootstrap_can_install_local_tiny_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercises venv-creation + install mechanics with the test interpreter; the
    # SkyPilot Python-support policy is out of scope here (and the CI matrix runs
    # this on Python versions outside SkyPilot's supported range), so treat the
    # interpreter as supported.
    monkeypatch.setattr(skypilot_cli, "_is_supported_python", lambda _v: True)
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
        if "check" in cmd:
            captured["cmd"] = cmd
            captured["env"] = env
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Kubernetes: enabled [compute]\nchecks passed",
                stderr="",
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


def test_verify_scopes_existing_kubeconfig_to_its_current_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    kubeconfig = tmp_path / "kube.yaml"
    kubeconfig.write_text(
        "apiVersion: v1\ncurrent-context: fleet-exact\n", encoding="utf-8"
    )
    captured: dict[str, object] = {}
    _intercept_sky_check(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["skypilot", "verify", "--path", str(venv), "--kubeconfig", str(kubeconfig)],
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"][-4:] == [
        "check",
        "--config",
        'kubernetes.allowed_contexts=["fleet-exact"]',
        "kubernetes",
    ]


def test_verify_rejects_zero_exit_when_kubernetes_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001
        if cmd[-1] == "check":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Kubernetes: disabled\nNo infra to check/enabled.", stderr=""
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)
    result = runner.invoke(
        app,
        [
            "skypilot",
            "verify",
            "--path",
            str(venv),
            "--controller-backend",
            "kubernetes",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["kubernetes_enabled"] is False


def test_bare_verify_keeps_legacy_runtime_semantics_without_kubernetes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001
        if cmd[-1] == "check":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Kubernetes: disabled\nchecks passed", stderr=""
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)
    result = runner.invoke(
        app,
        ["skypilot", "verify", "--path", str(venv), "--output-format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["kubernetes_required"] is False
    assert payload["kubernetes_enabled"] is False


def test_verify_with_kubeconfig_requires_kubernetes_without_backend_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    kubeconfig = tmp_path / "kube.yaml"
    kubeconfig.write_text(
        "apiVersion: v1\ncurrent-context: fleet-exact\n", encoding="utf-8"
    )
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001
        if "check" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Kubernetes: disabled\nchecks passed", stderr=""
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)
    result = runner.invoke(
        app,
        [
            "skypilot",
            "verify",
            "--path",
            str(venv),
            "--kubeconfig",
            str(kubeconfig),
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["kubernetes_required"] is True
    assert payload["kubernetes_enabled"] is False


def test_verify_without_kubeconfig_inherits_ambient_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    captured: dict[str, object] = {}
    _intercept_sky_check(monkeypatch, captured)

    result = runner.invoke(app, ["skypilot", "verify", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert captured["env"] is None


def test_verify_kubernetes_mode_marks_nebius_profile_optional(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001
        if cmd[-1] == "check":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "Kubernetes: enabled [compute]\n"
                    "Unable to create Nebius profile\nSetup completed\n"
                ),
                stderr="",
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)
    result = runner.invoke(
        app,
        [
            "skypilot",
            "verify",
            "--path",
            str(venv),
            "--controller-backend",
            "kubernetes",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["nebius_profile"] == "skipped_not_required"
    assert "Unable to create" not in result.output


def test_verify_nebius_mode_fails_required_profile_without_success_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001
        if cmd[-1] == "check":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Unable to create Nebius profile\nSetup completed\n",
                stderr="",
            )
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)
    result = runner.invoke(
        app,
        [
            "skypilot",
            "verify",
            "--path",
            str(venv),
            "--controller-backend",
            "nebius",
        ],
    )

    assert result.exit_code == 1
    assert "status: failed" in result.output
    assert "failed_required" in result.output
    assert "Setup completed" not in result.output


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


def test_is_supported_python_range() -> None:
    assert skypilot_cli._is_supported_python((3, 11)) is True
    assert skypilot_cli._is_supported_python((3, 12)) is True
    assert skypilot_cli._is_supported_python((3, 14)) is False
    assert skypilot_cli._is_supported_python((3, 8)) is False
    assert skypilot_cli._is_supported_python(None) is False


def test_resolve_python_bin_rejects_explicit_unsupported(monkeypatch, tmp_path: Path) -> None:
    fake = _write_executable(tmp_path / "py314", '#!/bin/sh\necho "3 14"\n')
    with pytest.raises(skypilot_cli.SkyPilotBootstrapError) as exc:
        skypilot_cli._resolve_python_bin(str(fake))
    assert "3.14" in str(exc.value)
    assert "supported range" in str(exc.value)


def test_resolve_python_bin_autoselects_supported_when_default_too_new(monkeypatch) -> None:
    # Default interpreter reports 3.14; a supported python3.12 is on PATH.
    def fake_detect(executable):
        return (3, 14) if str(executable) == sys.executable else (3, 12)

    monkeypatch.setattr(skypilot_cli, "_detect_python_version", fake_detect)
    monkeypatch.setattr(
        skypilot_cli.shutil,
        "which",
        lambda name: "/usr/bin/python3.12" if name == "python3.12" else None,
    )
    assert skypilot_cli._resolve_python_bin(None) == "/usr/bin/python3.12"


def test_resolve_python_bin_errors_when_no_supported_interpreter(monkeypatch) -> None:
    monkeypatch.setattr(skypilot_cli, "_detect_python_version", lambda _e: (3, 14))
    monkeypatch.setattr(skypilot_cli.shutil, "which", lambda _name: None)
    with pytest.raises(skypilot_cli.SkyPilotBootstrapError) as exc:
        skypilot_cli._resolve_python_bin(None)
    assert "supported range" in str(exc.value)


def test_resolve_python_bin_passes_through_unknown_version(monkeypatch) -> None:
    # An interpreter whose version can't be determined is passed through so the
    # normal venv-creation error still surfaces (no false rejection).
    monkeypatch.setattr(skypilot_cli, "_detect_python_version", lambda _e: None)
    assert skypilot_cli._resolve_python_bin("/some/python") == "/some/python"


def _config_path() -> Path:
    from npa.clients import config as config_module

    return config_module.CONFIG_PATH


def test_bootstrap_persists_sky_bin_to_config(tmp_path: Path) -> None:
    """Bootstrap must survive the shell it ran in.

    Regression: bootstrap only printed `export NPA_SKYPILOT_BIN=...`, so the
    next shell (and every `workflow submit` in it) failed with "SkyPilot CLI
    executable is not configured".
    """
    import yaml

    venv = _fake_installed_venv(tmp_path / "sky-venv")

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert "saved: skypilot.sky_bin" in result.output
    saved = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    assert saved["skypilot"]["sky_bin"] == str((venv / "bin" / "sky").resolve())


def test_bootstrap_saved_sky_bin_resolves_without_env(tmp_path: Path) -> None:
    """The persisted value is the one NPA's resolver reads."""
    from npa.orchestration.skypilot._bin import resolve_sky_bin

    venv = _fake_installed_venv(tmp_path / "sky-venv")
    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])
    assert result.exit_code == 0, result.output

    # No NPA_SKYPILOT_BIN in the environment: config alone must resolve it.
    assert os.environ.get("NPA_SKYPILOT_BIN") in (None, "")
    assert resolve_sky_bin() == (venv / "bin" / "sky").resolve()


def test_bootstrap_no_save_leaves_config_untouched(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")

    result = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(venv), "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert "saved: skypilot.sky_bin" not in result.output
    assert "export NPA_SKYPILOT_BIN=" in result.output
    assert not _config_path().exists()


def test_bootstrap_still_succeeds_when_saving_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv")
    monkeypatch.setattr(skypilot_cli, "_persist_sky_bin", lambda _bin: "")

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert "could not save skypilot.sky_bin" in result.output


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        (None, False),
        ("", False),
        ("30.1.0", True),
        ("31.0.0", True),
        # SkyPilot 0.12.2 already excludes 32.0.0 itself.
        ("32.0.0", False),
        ("32.0.1", True),
        ("35.0.0", True),
        # 36.0.0 renamed the generated openapi type names, which turns every
        # pod_config into an import of `kubernetes.client.models.dict[str, str]`.
        ("36.0.0", False),
        ("36.0.3", False),
        ("37.1.0", False),
    ],
)
def test_kubernetes_client_supported_matches_measured_breakage(
    version: str | None, supported: bool
) -> None:
    assert skypilot_cli.kubernetes_client_supported(version) is supported


def test_skypilot_install_package_pins_kubernetes_client(
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

    monkeypatch.setattr(skypilot_cli.subprocess, "run", fake_run)

    skypilot_cli._install_package(state, "skypilot==0.12.2")

    assert any(
        cmd[-1] == skypilot_cli.KUBERNETES_CLIENT_SPEC for cmd in installs
    ), installs
    assert "<36" in skypilot_cli.KUBERNETES_CLIENT_SPEC


def test_skypilot_bootstrap_repairs_a_reused_venv_with_a_broken_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", kubernetes_version="36.0.3")
    original_bytes = (venv / "bin" / "sky").read_bytes()

    def fake_create(path: Path, _python: object) -> None:
        _fake_installed_venv(path, kubernetes_version="34.1.0")

    monkeypatch.setattr(skypilot_cli, "_create_venv", fake_create)
    monkeypatch.setattr(skypilot_cli, "_ensure_pip", lambda _state: None)
    monkeypatch.setattr(skypilot_cli, "_install_package", lambda _state, _spec: None)

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert skypilot_cli.inspect_venv(venv).kubernetes_version == "34.1.0"
    assert (venv / "bin" / "sky").read_bytes() == original_bytes


def test_skypilot_uninstall_removes_venv_and_clears_saved_bin(tmp_path: Path) -> None:
    """`npa skypilot uninstall` is the inverse of bootstrap.

    The teardown report left ~/.npa/skypilot-venv and skypilot.sky_bin in
    config.yaml behind with no npa command to remove them.
    """
    import yaml

    venv = _fake_installed_venv(tmp_path / "sky-venv")
    boot = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])
    assert boot.exit_code == 0, boot.output
    assert venv.exists()
    assert yaml.safe_load(_config_path().read_text(encoding="utf-8"))["skypilot"]["sky_bin"]

    result = runner.invoke(app, ["skypilot", "uninstall", "--path", str(venv), "--yes"])

    assert result.exit_code == 0, result.output
    assert not venv.exists()
    assert "Removed SkyPilot venv" in result.output
    assert "Cleared skypilot.sky_bin" in result.output
    saved = yaml.safe_load(_config_path().read_text(encoding="utf-8")) or {}
    assert "sky_bin" not in saved.get("skypilot", {})


def test_skypilot_uninstall_is_idempotent_without_a_venv(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["skypilot", "uninstall", "--path", str(tmp_path / "absent"), "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "No SkyPilot venv" in result.output


def test_skypilot_controller_cleanup_requires_confirmation_and_is_npa_only(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    calls: list[object] = []
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_jobs_controller",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(ok=True, resources_removed=[], errors=[], commands=[]),
    )

    plan = runner.invoke(app, ["skypilot", "cleanup-controller", "--json"])
    assert plan.exit_code == 1
    assert json.loads(plan.output)["outcome"] == "confirmation_required"
    assert calls == []

    cleanup = runner.invoke(
        app,
        [
            "skypilot",
            "cleanup-controller",
            "--yes",
            "--sky-bin",
            "/npa/pinned/sky",
            "--json",
        ],
    )
    assert cleanup.exit_code == 0, cleanup.output
    cleanup_payload = json.loads(cleanup.output)
    assert cleanup_payload["outcome"] == "cleaned"
    assert cleanup_payload["remote_absence_verified"] is True
    assert cleanup_payload["local_metadata_cleared"] is True
    assert cleanup_payload["overall_verified"] is True
    assert cleanup_payload["verified"] is True
    assert calls == [{"project": "", "context": "", "sky_bin": "/npa/pinned/sky"}]

    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_jobs_controller",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("controller unavailable")),
    )
    failed = runner.invoke(
        app, ["skypilot", "cleanup-controller", "--yes", "--json"]
    )
    assert failed.exit_code == 2
    assert json.loads(failed.output)["outcome"] == "verification_failed"


def test_skypilot_controller_cleanup_forwards_explicit_orphan_recovery_attestation(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_jobs_controller",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(ok=True, resources_removed=[], errors=[], commands=[]),
    )

    result = runner.invoke(
        app,
        [
            "skypilot",
            "cleanup-controller",
            "--yes",
            "--recover-orphan-controller",
            "--attest-no-active-jobs",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "project": "",
            "context": "",
            "sky_bin": None,
            "recover_orphan_controller": True,
            "attest_no_active_jobs": True,
        }
    ]


def test_skypilot_uninstall_refuses_to_delete_the_npa_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --path pointing at the running interpreter must be rejected, not wiped."""
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "npa-venv"))
    (tmp_path / "npa-venv").mkdir()

    result = runner.invoke(
        app, ["skypilot", "uninstall", "--path", str(tmp_path / "npa-venv"), "--yes"]
    )

    assert result.exit_code != 0
    assert (tmp_path / "npa-venv").exists()


def test_skypilot_bootstrap_leaves_a_good_client_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", kubernetes_version="31.0.0")
    installs: list[list[str]] = []
    original = skypilot_cli._run_no_raise

    def fake_run(cmd, *, env=None):  # noqa: ANN001 - test stub
        if cmd[1:4] == ["-m", "pip", "install"]:
            installs.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return original(cmd, env=env)

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", fake_run)

    result = runner.invoke(app, ["skypilot", "bootstrap", "--path", str(venv)])

    assert result.exit_code == 0, result.output
    assert installs == []


def test_skypilot_status_reports_and_fails_on_a_broken_client(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", kubernetes_version="36.0.3")

    result = runner.invoke(app, ["skypilot", "status", "--path", str(venv)])

    assert result.exit_code == 1, result.output
    assert "kubernetes_client: 36.0.3" in result.output
    assert "pod_config" in result.output


def test_skypilot_verify_fails_on_a_broken_client(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky-venv", kubernetes_version="36.0.3")

    result = runner.invoke(app, ["skypilot", "verify", "--path", str(venv)])

    assert result.exit_code == 1, result.output
    assert "npa skypilot bootstrap" in result.output


def test_bootstrap_preserves_unrelated_npa_state_byte_for_byte(tmp_path: Path) -> None:
    root = tmp_path / "custom-npa"
    sentinels = {
        "config.yaml": b"projects: {customer: {region: test}}\n",
        "credentials.yaml": b"tokens: {MODEL_KEY: sentinel}\n",
        "agents/customer/agent/record.json": b'{"id":"agent-sentinel"}\n',
        "operations/op.json": b'{"state":"running"}\n',
        "teardown-receipts/r.json": b'{"receipt":"sentinel"}\n',
        "runs/run-1/runtime.json": b'{"wave":"sentinel"}\n',
    }
    for name, body in sentinels.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    venv = _fake_installed_venv(root / "managed" / "sky")

    result = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(venv), "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert {name: (root / name).read_bytes() for name in sentinels} == sentinels


def test_bootstrap_rejects_symlink_and_parent_targets(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    linked = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(link / "sky"), "--no-save"]
    )
    broad = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(tmp_path / ".npa"), "--no-save"]
    )

    assert linked.exit_code == 1
    assert "symlink" in linked.output
    assert broad.exit_code == 1
    assert "parent NPA state" in broad.output


def test_bootstrap_recovers_interrupted_exchange(tmp_path: Path) -> None:
    target = tmp_path / "sky"
    previous, journal = skypilot_cli._bootstrap_paths(target)
    _fake_installed_venv(previous)
    skypilot_cli._write_bootstrap_journal(
        journal,
        {"target": str(target), "staging": ".sky.staging-dead", "previous": previous.name},
    )

    result = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(target), "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert (target / "bin" / "sky").is_file()
    assert not previous.exists()
    assert not journal.exists()


def test_bootstrap_lock_is_owner_only(tmp_path: Path) -> None:
    venv = _fake_installed_venv(tmp_path / "sky")

    result = runner.invoke(
        app, ["skypilot", "bootstrap", "--path", str(venv), "--no-save"]
    )

    assert result.exit_code == 0, result.output
    lock = venv.with_name(f".{venv.name}{skypilot_cli.BOOTSTRAP_LOCK_SUFFIX}")
    assert lock.stat().st_mode & 0o777 == 0o600

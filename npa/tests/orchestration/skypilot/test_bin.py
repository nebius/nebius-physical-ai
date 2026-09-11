from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import ANY

import pytest
import yaml

from npa import controller_ownership as ownership
from npa.controller_ownership import ControllerOwner, bind_controller_owner, controller_owner
from npa.orchestration.skypilot import _bin as bin_module
from npa.orchestration.skypilot._bin import (
    REQUIRED_SKYPILOT_VERSION,
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    clear_skypilot_version_cache,
    ensure_skypilot_version,
    resolve_config,
    resolve_isolated_config_dir,
    resolve_sky_bin,
)


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _isolated_skypilot_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)


def test_resolve_sky_bin_explicit_path_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = _executable(tmp_path / "explicit-sky")
    env = _executable(tmp_path / "env-sky")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))

    assert resolve_sky_bin(explicit) == explicit.resolve()


def test_resolve_sky_bin_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _executable(tmp_path / "env-sky")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert resolve_sky_bin() == env.resolve()


def test_resolve_sky_bin_config_file_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    discovered = _executable(tmp_path / "config-sky")
    config = tmp_path / "config.yaml"
    config.write_text(f"skypilot:\n  sky_bin: {discovered}\n", encoding="utf-8")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)

    assert resolve_sky_bin() == discovered.resolve()


def test_resolve_sky_bin_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing.yaml")

    with pytest.raises(SkyPilotNotInstalledError, match="SkyPilot CLI executable is not configured"):
        resolve_sky_bin()


def test_resolve_config_precedence_explicit_env_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = _executable(tmp_path / "explicit-sky")
    env = _executable(tmp_path / "env-sky")
    default = _executable(tmp_path / "default-sky")
    config_global = tmp_path / "config-global.yaml"
    env_global = tmp_path / "env-global.yaml"
    explicit_global = tmp_path / "explicit-global.yaml"
    config_isolated = tmp_path / "config-isolated"
    env_isolated = tmp_path / "env-isolated"
    explicit_isolated = tmp_path / "explicit-isolated"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "skypilot:",
                f"  sky_bin: {default}",
                f"  global_config_path: {config_global}",
                f"  isolated_config_dir: {config_isolated}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", str(env_global))
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(env_isolated))

    resolved = resolve_config(
        sky_bin=explicit,
        global_config_path=explicit_global,
        isolated_config_dir=explicit_isolated,
    )
    assert resolved.sky_bin == explicit.resolve()
    assert resolved.global_config_path == explicit_global
    assert resolved.isolated_config_dir == explicit_isolated

    resolved = resolve_config()
    assert resolved.sky_bin == env.resolve()
    assert resolved.global_config_path == env_global
    assert resolved.isolated_config_dir == env_isolated

    monkeypatch.delenv("NPA_SKYPILOT_BIN")
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG")
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR")
    resolved = resolve_config()
    assert resolved.sky_bin == default.resolve()
    assert resolved.global_config_path == config_global
    assert resolved.isolated_config_dir == config_isolated


def test_resolve_isolated_config_dir_does_not_require_sky_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_isolated = tmp_path / "config-isolated"
    env_isolated = tmp_path / "env-isolated"
    explicit_isolated = tmp_path / "explicit-isolated"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"skypilot:\n  isolated_config_dir: {config_isolated}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(env_isolated))

    assert resolve_isolated_config_dir(explicit_isolated) == explicit_isolated
    assert resolve_isolated_config_dir() == env_isolated

    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR")
    assert resolve_isolated_config_dir() == config_isolated


def test_resolve_isolated_config_dir_defaults_to_shared_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing.yaml")

    assert resolve_isolated_config_dir() is None


def test_resolve_config_makes_relative_runtime_paths_cwd_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sky = _executable(tmp_path / "sky")
    workdir = tmp_path / "operator"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    resolved = resolve_config(
        sky_bin=sky,
        global_config_path=Path("sky.yaml"),
        isolated_config_dir=Path("sky-state"),
    )

    assert resolved.global_config_path == workdir / "sky.yaml"
    assert resolved.isolated_config_dir == workdir / "sky-state"


def test_resolve_config_rejects_unknown_config_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sky = _executable(tmp_path / "sky")
    config = tmp_path / "config.yaml"
    config.write_text(f"skypilot:\n  sky_bin: {sky}\n  typo_key: true\n", encoding="utf-8")
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)

    with pytest.raises(SkyPilotConfigError, match="typo_key.*Valid keys"):
        resolve_config()


def _controller_owner() -> ControllerOwner:
    return ControllerOwner(
        project_alias="demo",
        project_id="project-a",
        cluster_id="cluster-a",
        cluster_name="npa-cluster",
        context="npa-cluster",
        context_fingerprint="immutable-fingerprint",
        operation_id="operation-a",
    )


def test_production_owner_writer_remains_readable_by_runtime_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sky = _executable(tmp_path / "sky")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "projects": {"demo": {"project_id": "project-a"}},
                "skypilot": {"sky_bin": str(sky)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", config)

    owner = _controller_owner()
    bind_controller_owner(owner)
    resolved = resolve_config(npa_config_path=config)

    assert resolved.sky_bin == sky.resolve()
    assert controller_owner() == owner
    saved = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert saved["skypilot"]["controller_owner"] == owner.to_dict()


def test_owner_metadata_survives_cancel_verify_and_controller_cleanup_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.orchestration.skypilot import cleanup, workflow

    sky = _executable(tmp_path / "sky")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "projects": {"demo": {"project_id": "project-a"}},
                "skypilot": {"sky_bin": str(sky)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", config)
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda value: value)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: value)
    owner = _controller_owner()
    bind_controller_owner(owner)

    workflow_calls: list[list[str]] = []

    def workflow_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        workflow_calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='[{"job_id":"job-1","status":"SUCCEEDED"}]',
            stderr="",
        )

    cleanup_calls: list[list[str]] = []

    def cleanup_run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        cleanup_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", workflow_run)
    monkeypatch.setattr(cleanup, "_run", cleanup_run)

    status = workflow.workflow_status("job-1")
    cancelled = cleanup._cancel_job(
        "job-1", isolated_config_dir=None, config_path=None, sky_bin=None
    )
    controller = cleanup._down_jobs_controller(
        "sky-jobs-controller-target",
        isolated_config_dir=None,
        config_path=None,
        sky_bin=None,
    )

    assert status.status == "SUCCEEDED"
    assert cancelled.ok
    assert controller.ok
    assert any(call[1:3] == ["jobs", "queue"] for call in workflow_calls)
    assert any(call[1:3] == ["jobs", "cancel"] for call in cleanup_calls)
    assert any(call[1:3] == ["down", "--yes"] for call in cleanup_calls)
    assert controller_owner() == owner
    assert ownership.clear_controller_owner(
        "demo", project_id="project-a", cluster_id="cluster-a"
    )
    assert controller_owner() is None
    assert resolve_config(npa_config_path=config).sky_bin == sky.resolve()


def test_legacy_project_owner_migrates_without_blocking_runtime_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sky = _executable(tmp_path / "sky")
    owner = _controller_owner()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "controller_owner": owner.to_dict(),
                    }
                },
                "skypilot": {"sky_bin": str(sky)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", config)

    bind_controller_owner(owner)

    assert resolve_config(npa_config_path=config).sky_bin == sky.resolve()
    saved = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert saved["skypilot"]["controller_owner"] == owner.to_dict()
    assert "controller_owner" not in saved["projects"]["demo"]


def test_malformed_owner_is_left_to_ownership_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sky = _executable(tmp_path / "sky")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "skypilot": {
                    "sky_bin": str(sky),
                    "controller_owner": {"project_alias": "incomplete"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", config)

    assert resolve_config(npa_config_path=config).sky_bin == sky.resolve()
    assert controller_owner() is None


def test_ensure_skypilot_version_accepts_required_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    sky = _executable(tmp_path / "sky")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[-1] != "--version":
            return bin_module.subprocess.CompletedProcess(
                cmd, 0, stdout="3.12 34.1.0\n", stderr=""
            )
        return bin_module.subprocess.CompletedProcess(cmd, 0, stdout=f"SkyPilot {REQUIRED_SKYPILOT_VERSION}\n", stderr="")

    monkeypatch.setattr(bin_module.subprocess, "run", fake_run)

    assert ensure_skypilot_version(sky) == sky.resolve()
    assert ensure_skypilot_version(sky) == sky.resolve()
    assert calls == [
        [str(sky.resolve()), "--version"],
        [str(sky.resolve().parent / "python"), "-c", ANY],
    ]


def test_ensure_skypilot_version_rejects_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    sky = _executable(tmp_path / "sky")

    def fake_run(cmd, **kwargs):
        return bin_module.subprocess.CompletedProcess(cmd, 0, stdout="SkyPilot 0.12.1\n", stderr="")

    monkeypatch.setattr(bin_module.subprocess, "run", fake_run)

    with pytest.raises(SkyPilotVersionError, match="expected 0.12.2, got 0.12.1"):
        ensure_skypilot_version(sky)


@pytest.mark.parametrize(
    ("runtime", "match"),
    [
        ("3.14 34.1.0\n", "Python 3.14"),
        ("3.12 36.0.0\n", "kubernetes 36.0.0"),
        ("3.12 32.0.0\n", "kubernetes 32.0.0"),
    ],
)
def test_ensure_skypilot_version_rejects_runtime_drift_before_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: str, match: str
) -> None:
    clear_skypilot_version_cache()
    sky = _executable(tmp_path / "bin" / "sky")

    def fake_run(cmd, **kwargs):
        output = f"SkyPilot {REQUIRED_SKYPILOT_VERSION}\n" if cmd[-1] == "--version" else runtime
        return bin_module.subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(bin_module.subprocess, "run", fake_run)

    with pytest.raises(SkyPilotVersionError, match=match):
        ensure_skypilot_version(sky)

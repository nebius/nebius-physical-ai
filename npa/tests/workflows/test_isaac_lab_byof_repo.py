from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_isaac_lab_byof_repo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_isaac_lab_byof_repo", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_sanitizes_stale_nebius_tokens(monkeypatch) -> None:
    module = _load_module()
    captured_env: dict[str, str] = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/stale-token")
    monkeypatch.setattr(module.subprocess, "run", fake_subprocess_run)

    module._run(["echo", "ok"])

    assert "NEBIUS_IAM_TOKEN" not in captured_env
    assert "NEBIUS_IAM_TOKEN_FILE" not in captured_env


def test_docker_login_uses_profile_token_for_password_stdin(monkeypatch) -> None:
    module = _load_module()
    seen: dict[str, object] = {}

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd == ["nebius", "iam", "get-access-token"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="profile-token\n", stderr="")
        if cmd[:4] == ["docker", "login", "-u", "iam"]:
            seen["stdin"] = stdin
            seen["env"] = env
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(module, "_run", fake_run)
    module._docker_login_nebius("cr.example.nebius.cloud", env={"DOCKER_CONFIG": "/tmp/docker-auth"})

    assert seen["stdin"] == "profile-token"
    assert seen["env"] == {"DOCKER_CONFIG": "/tmp/docker-auth"}


def test_main_reports_403_base_image_hint(monkeypatch, capsys) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/example/project/npa-isaac-lab:test",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd == ["nebius", "iam", "get-access-token"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="profile-token\n", stderr="")
        if cmd[:2] == ["docker", "build"]:
            raise RuntimeError("403 Forbidden while pulling ISAAC_BASE_IMAGE")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(["--run-id", "leisaac-hint-case", "--skip-run"])

    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert "hint" in output
    assert "Grant pull access for the base image" in output["hint"]

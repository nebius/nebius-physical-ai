from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_repo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_byof_repo", SCRIPT_PATH)
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
            raise RuntimeError("403 Forbidden while pulling BYOF_BASE_IMAGE")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(["--run-id", "leisaac-hint-case", "--base-profile", "isaac-lab", "--skip-run"])

    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert "hint" in output
    assert "Pass --base-image from an accessible registry" in output["hint"]


def test_main_reports_403_push_hint(monkeypatch, capsys) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "nvcr.io/nvidia/isaac-lab:2.3.2",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd == ["nebius", "iam", "get-access-token"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="profile-token\n", stderr="")
        if cmd[:2] == ["docker", "push"]:
            raise RuntimeError("command failed (1): docker push ... 403 Forbidden")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(["--run-id", "leisaac-push-403", "--base-profile", "isaac-lab", "--skip-run"])

    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert "hint" in output
    assert "Registry push was denied" in output["hint"]


def test_main_derives_base_registry_from_target_image(monkeypatch, capsys) -> None:
    module = _load_module()
    seen_registries: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/default/project",
    )

    def fake_container_image_for_tool(tool: str, *, registry: str, **_kwargs):
        assert tool == "isaac-lab"
        seen_registries.append(registry)
        return f"{registry}/npa-isaac-lab:test"

    monkeypatch.setattr(module, "container_image_for_tool", fake_container_image_for_tool)
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["noop"], 0, stdout="", stderr=""),
    )

    rc = module.main(
        [
            "--run-id",
            "leisaac-base-registry",
            "--image",
            "cr.eu-north1.nebius.cloud/custom/proj/npa-isaac-lab-leisaac:test",
            "--base-profile",
            "isaac-lab",
            "--skip-build",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert "cr.eu-north1.nebius.cloud/custom/proj" in seen_registries
    output = json.loads(capsys.readouterr().out)
    assert "cr.eu-north1.nebius.cloud/custom/proj/npa-isaac-lab:test" in output["base_image_candidates"]


def test_main_retries_build_with_fallback_base_image(monkeypatch, capsys) -> None:
    module = _load_module()
    build_args: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/default/project",
    )

    def fake_container_image_for_tool(tool: str, registry: str | None = None, **_kwargs):
        assert tool == "isaac-lab"
        if registry == "cr.eu-north1.nebius.cloud/custom/proj":
            return "cr.eu-north1.nebius.cloud/custom/proj/npa-isaac-lab:fallback"
        if registry == "cr.eu-north1.nebius.cloud/default/project":
            return "cr.eu-north1.nebius.cloud/default/project/npa-isaac-lab:default"
        return "ghcr.io/nebius/npa-isaac-lab:stable"

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd == ["nebius", "iam", "get-access-token"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="profile-token\n", stderr="")
        if cmd[:2] == ["docker", "build"]:
            base = next((part for part in cmd if part.startswith("BYOF_BASE_IMAGE=")), "")
            build_args.append(base)
            if base.endswith(":stable") or base.endswith(":default"):
                raise RuntimeError("403 Forbidden while pulling BYOF_BASE_IMAGE")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "container_image_for_tool", fake_container_image_for_tool)
    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "leisaac-fallback-case",
            "--image",
            "cr.eu-north1.nebius.cloud/custom/proj/npa-isaac-lab-leisaac:test",
            "--base-profile",
            "isaac-lab",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert build_args[0].endswith(":stable")
    assert any(item.endswith(":fallback") for item in build_args)
    output = json.loads(capsys.readouterr().out)
    assert output["base_image"].endswith(":fallback")


def test_main_forwards_yaml_override_to_runner(monkeypatch) -> None:
    module = _load_module()
    seen: dict[str, object] = {}

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
        if cmd and cmd[0] == sys.executable and str(module.ISAAC_RUNNER) in cmd:
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout='{"status":"submitted"}\n', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "leisaac-yaml-forward",
            "--skip-build",
            "--base-profile",
            "isaac-lab",
            "--yaml",
            "/tmp/isaac-lab-rtxpro.yaml",
        ]
    )

    assert rc == 0
    cmd = seen.get("cmd")
    assert isinstance(cmd, list)
    assert "--yaml" in cmd
    assert "/tmp/isaac-lab-rtxpro.yaml" in cmd


def test_main_forwards_datagen_workload_to_datagen_runner(monkeypatch) -> None:
    module = _load_module()
    seen: dict[str, object] = {}

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
        if cmd and cmd[0] == sys.executable and str(module.DATAGEN_RUNNER) in cmd:
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout='{"status":"submitted"}\n', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "leisaac-datagen-forward",
            "--skip-build",
            "--base-profile",
            "isaac-lab",
            "--workload",
            "datagen",
            "--task",
            "LeIsaac-SO101-PickOrange-v0",
            "--num-envs",
            "4",
            "--num-demos",
            "10",
            "--yaml",
            "/tmp/byof-datagen-rtxpro-smoke.yaml",
        ]
    )

    assert rc == 0
    cmd = seen.get("cmd")
    assert isinstance(cmd, list)
    assert str(module.DATAGEN_RUNNER) in cmd
    assert "--task" in cmd and "LeIsaac-SO101-PickOrange-v0" in cmd
    assert "--num-envs" in cmd and "4" in cmd
    assert "--num-demos" in cmd and "10" in cmd
    assert "--yaml" in cmd and "/tmp/byof-datagen-rtxpro-smoke.yaml" in cmd


def test_base_image_candidates_ubuntu_profile_default() -> None:
    module = _load_module()
    candidates = module._base_image_candidates(
        profile="ubuntu",
        image="cr.eu-north1.nebius.cloud/example/project/npa-byof:test",
        registry="cr.eu-north1.nebius.cloud/example/project",
        explicit_base="",
    )
    assert candidates == ["ubuntu:22.04"]


def test_base_image_candidates_explicit_base_overrides_profile() -> None:
    module = _load_module()
    candidates = module._base_image_candidates(
        profile="ubuntu",
        image="cr.eu-north1.nebius.cloud/example/project/npa-byof:test",
        registry="cr.eu-north1.nebius.cloud/example/project",
        explicit_base="ubuntu:24.04",
    )
    assert candidates == ["ubuntu:24.04"]


def test_base_image_candidates_isaac_lab_profile(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/example/project/npa-isaac-lab:test",
    )
    candidates = module._base_image_candidates(
        profile="isaac-lab",
        image="cr.eu-north1.nebius.cloud/example/project/npa-isaac-lab-leisaac:test",
        registry="cr.eu-north1.nebius.cloud/example/project",
        explicit_base="",
    )
    assert "nvcr.io/nvidia/isaac-lab:2.3.2" in candidates
    assert "nvcr.io/nvidia/isaac-sim:4.5.0" in candidates


def test_main_ubuntu_profile_uses_byof_base_image_build_arg(monkeypatch, capsys) -> None:
    module = _load_module()
    build_args: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "cr.eu-north1.nebius.cloud/example/project",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd == ["nebius", "iam", "get-access-token"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="profile-token\n", stderr="")
        if cmd[:2] == ["docker", "build"]:
            build_args.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "ubuntu-byof-case",
            "--repo-url",
            "https://github.com/example/demo.git",
            "--repo-ref",
            "main",
            "--base-profile",
            "ubuntu",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert any(part == "BYOF_BASE_IMAGE=ubuntu:22.04" for part in build_args)
    output = json.loads(capsys.readouterr().out)
    assert output["base_profile"] == "ubuntu"
    assert output["base_image"] == "ubuntu:22.04"


def test_dockerfile_writes_metadata_without_python_dependency() -> None:
    module = _load_module()
    text = module._dockerfile_text()
    assert "BYOF_BASE_IMAGE" in text
    assert "npa_source_metadata.json" in text
    assert "printf" in text
    assert "/opt/byof" in text
    assert "USER ubuntu" in text
    assert "npa.packaging.tier=\"interactive\"" in text
    assert "useradd" in text
    assert "python3" in text
    assert "NOPASSWD:ALL" in text
    assert "sudo" in text
    assert "mkdir -p /workspace" in text


def test_compat_shim_delegates_to_run_byof_repo() -> None:
    shim_path = ROOT / "npa" / "scripts" / "run_isaac_lab_byof_repo.py"
    spec = importlib.util.spec_from_file_location("run_isaac_lab_byof_repo_shim", shim_path)
    assert spec and spec.loader
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.main.__module__ == "run_byof_repo"

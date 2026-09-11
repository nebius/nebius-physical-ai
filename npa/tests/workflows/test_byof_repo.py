from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_repo.py"
LIBERO_YAML_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-libero-b200-gpu.yaml"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_byof_repo", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _accepted_wan_base_args(module) -> list[str]:
    digest = module.wan_accepted_image_manifest()["oci_digest"]
    return [
        "--base-profile",
        "prebuilt",
        "--base-image",
        f"registry.example/project/npa-wan2-2@{digest}",
    ]


def test_openpi_terms_fail_before_registry_or_build(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.delenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", raising=False)
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: pytest.fail("registry resolved before terms gate"),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("command ran before terms gate"),
    )

    rc = module.main(
        [
            "--repo-url",
            "https://github.com/Physical-Intelligence/openpi.git",
            "--solution-name",
            "openpi",
            "--skip-run",
        ]
    )

    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert "Gemma Terms of Use" in output["error"]
    assert "Gemma Prohibited Use Policy" in output["error"]


@pytest.mark.parametrize("value", ["yes", "TRUE", "1", "YES "])
def test_openpi_terms_gate_requires_exact_yes(monkeypatch, value) -> None:
    from npa.workflows.byof.openpi import require_openpi_terms

    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", value)
    with pytest.raises(ValueError, match="OpenPI pi0.5 requires scoped"):
        require_openpi_terms()


def test_openpi_terms_gate_accepts_scoped_yes(monkeypatch) -> None:
    from npa.workflows.byof.openpi import require_openpi_terms

    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    require_openpi_terms()


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


def test_run_redacts_private_source_values_from_command_and_captured_failure(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    private_values = (
        "github-private-token-canary",
        "https://github.com/example/private-source.git",
        "private-ref-canary",
    )

    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="stdout accidentally contained " + " ".join(private_values),
            stderr="stderr accidentally contained " + " ".join(private_values),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_subprocess_run)
    with pytest.raises(RuntimeError) as exc_info:
        module._run(
            ["tool", *private_values], capture=True, redactions=private_values
        )

    combined = str(exc_info.value) + capsys.readouterr().out
    for private_value in private_values:
        assert private_value not in combined
    assert "<redacted>" in combined


def test_private_build_uses_only_secret_mounts_and_sanitized_metadata(
    monkeypatch, capsys, tmp_path
) -> None:
    module = _load_module()
    repo_url = "https://github.com/example/private-source.git"
    repo_ref = "private-ref-canary"
    token = "github-private-token-canary"
    token_path = tmp_path / "token"
    url_path = tmp_path / "url"
    ref_path = tmp_path / "ref"
    prune_path = tmp_path / "prune-path"
    token_path.write_text(token, encoding="utf-8")
    url_path.write_text(repo_url, encoding="utf-8")
    ref_path.write_text(repo_ref, encoding="utf-8")
    prune_path.write_text("private-assets/render-only", encoding="utf-8")
    for path in (token_path, url_path, ref_path, prune_path):
        path.chmod(0o600)

    @contextmanager
    def fake_secrets(*_args, **_kwargs):
        yield SimpleNamespace(
            token=token_path,
            repo_url=url_path,
            repo_ref=ref_path,
            source_prune_path=prune_path,
            repository_sha256="a" * 64,
            ref_sha256="b" * 64,
            source_prune_path_sha256="c" * 64,
            redaction_values=(
                token,
                repo_url,
                repo_ref,
                "private-assets/render-only",
            ),
        )

    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "build"]:
            seen["cmd"] = list(cmd)
            seen["dockerfile"] = (Path(cmd[-1]) / "Dockerfile").read_text(
                encoding="utf-8"
            )
            seen["redactions"] = kwargs.get("redactions")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )
    monkeypatch.setattr(module, "private_repository_secrets", fake_secrets)
    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(
        [
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--repo-auth",
            "github",
            "--source-prune-path",
            "private-assets/render-only",
            "--run-id",
            "private-build-test",
            "--skip-push",
            "--skip-run",
        ]
    )

    assert rc == 0
    command = " ".join(seen["cmd"])
    dockerfile = str(seen["dockerfile"])
    output = capsys.readouterr().out
    for value in (token, repo_url, repo_ref, "private-assets/render-only"):
        assert value not in command
        assert value not in dockerfile
        assert value not in output
    assert command.count("--secret") == 4
    assert "npa_byof_repo_token" in command
    assert "npa_byof_source_prune_path" in command
    assert "BYOF_SOURCE_CACHE_KEY=" + "a" * 64 + "b" * 64 + "c" * 64 in command
    assert "type=secret,id=npa_byof_repo_token" in dockerfile
    assert "type=secret,id=npa_byof_source_prune_path" in dockerfile
    assert "username=x-access-token" in dockerfile
    assert "password=" in dockerfile
    assert 'ARG OSS_REPO_URL=""' in dockerfile
    assert 'ARG OSS_REPO_REF=""' in dockerfile
    assert "private-byof" in dockerfile
    assert '"commit":"<private-commit>"' in dockerfile
    assert '"commit_sha256":"%s"' in dockerfile
    assert '"source_prune_path":"%s"' in dockerfile
    assert '"source_prune_path_sha256":"%s"' in dockerfile
    assert '"source_pruned":%s' in dockerfile
    assert '"git_objects_removed":%s' in dockerfile
    assert "<private-source-prune-path>" in dockerfile
    assert "rm -rf /opt/byof/.git" in dockerfile
    assert "BYOF_SOURCE_PRUNE_PATH=private-assets/render-only" not in command
    assert seen["redactions"] == (
        token,
        repo_url,
        repo_ref,
        "private-assets/render-only",
    )
    summary = json.loads(output)
    assert summary["repo_url"] == "<private-repository>"
    assert summary["source_identity"] == {
        "repository_sha256": "a" * 64,
        "ref_sha256": "b" * 64,
        "source_prune_path_sha256": "c" * 64,
    }
    assert summary["source_prune_path"] == "<private-source-prune-path>"


def test_failed_private_build_redacts_summary_stdout_stderr_and_exception(
    monkeypatch, capsys, tmp_path
) -> None:
    module = _load_module()
    repo_url = "https://github.com/example/private-source.git"
    repo_ref = "missing-private-ref-canary"
    token = "github-private-token-canary"
    private_values = (token, repo_url, repo_ref)
    token_path = tmp_path / "token"
    url_path = tmp_path / "url"
    ref_path = tmp_path / "ref"
    prune_path = tmp_path / "prune-path"
    for path, value in zip(
        (token_path, url_path, ref_path), private_values, strict=True
    ):
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    prune_path.write_text("", encoding="utf-8")
    prune_path.chmod(0o600)

    @contextmanager
    def fake_secrets(*_args, **_kwargs):
        yield SimpleNamespace(
            token=token_path,
            repo_url=url_path,
            repo_ref=ref_path,
            source_prune_path=prune_path,
            repository_sha256="a" * 64,
            ref_sha256="b" * 64,
            source_prune_path_sha256=hashlib.sha256(b"").hexdigest(),
            redaction_values=private_values,
        )

    def failed_build(cmd, **kwargs):
        assert cmd[:2] == ["docker", "build"]
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="BuildKit clone output: " + " ".join(private_values),
            stderr="Git failure output: " + " ".join(private_values),
        )

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )
    monkeypatch.setattr(module, "private_repository_secrets", fake_secrets)
    monkeypatch.setattr(module.subprocess, "run", failed_build)

    rc = module.main(
        [
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--repo-auth",
            "github",
            "--run-id",
            "private-build-failure-test",
            "--skip-push",
            "--skip-run",
        ]
    )

    assert rc == 1
    published = capsys.readouterr()
    combined = published.out + published.err
    for private_value in private_values:
        assert private_value not in combined
    assert '"repo_url": "<private-repository>"' in combined
    assert '"repo_ref": "<private-ref>"' in combined
    assert combined.count("<redacted>") >= 3


def test_main_reports_403_base_image_hint(monkeypatch, capsys) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "registry.example/example/project/npa-isaac-lab:test",
    )
    monkeypatch.setenv("NPA_BYOF_SKIP_REGISTRY_REFRESH", "1")

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd[:2] == ["docker", "build"]:
            raise RuntimeError("403 Forbidden while pulling BYOF_BASE_IMAGE")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        ["--run-id", "leisaac-hint-case", "--base-profile", "isaac-lab", "--skip-run"]
    )

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
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "nvcr.io/nvidia/isaac-lab:2.3.2",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd[:2] == ["docker", "push"]:
            raise RuntimeError("command failed (1): docker push ... 403 Forbidden")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        ["--run-id", "leisaac-push-403", "--base-profile", "isaac-lab", "--skip-run"]
    )

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
        lambda *_args, **_kwargs: "registry.example/default/project",
    )

    def fake_container_image_for_tool(tool: str, *, registry: str, **_kwargs):
        assert tool == "isaac-lab"
        seen_registries.append(registry)
        return f"{registry}/npa-isaac-lab:test"

    monkeypatch.setattr(
        module, "container_image_for_tool", fake_container_image_for_tool
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["noop"], 0, stdout="", stderr=""
        ),
    )

    rc = module.main(
        [
            "--run-id",
            "leisaac-base-registry",
            "--image",
            "registry.example/custom/proj/npa-isaac-lab-leisaac:test",
            "--base-profile",
            "isaac-lab",
            "--skip-build",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert "registry.example/custom/proj" in seen_registries
    output = json.loads(capsys.readouterr().out)
    assert (
        "registry.example/custom/proj/npa-isaac-lab:test"
        in output["base_image_candidates"]
    )


def test_main_retries_build_with_fallback_base_image(monkeypatch, capsys) -> None:
    module = _load_module()
    build_args: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/default/project",
    )

    def fake_container_image_for_tool(
        tool: str, registry: str | None = None, **_kwargs
    ):
        assert tool == "isaac-lab"
        if registry == "registry.example/custom/proj":
            return "registry.example/custom/proj/npa-isaac-lab:fallback"
        if registry == "registry.example/default/project":
            return "registry.example/default/project/npa-isaac-lab:default"
        return "ghcr.io/nebius/npa-isaac-lab:stable"

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd[:2] == ["docker", "build"]:
            base = next(
                (part for part in cmd if part.startswith("BYOF_BASE_IMAGE=")), ""
            )
            build_args.append(base)
            if base.endswith(":stable") or base.endswith(":default"):
                raise RuntimeError("403 Forbidden while pulling BYOF_BASE_IMAGE")
        if cmd[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Digest: sha256:" + "a" * 64 + "\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        module, "container_image_for_tool", fake_container_image_for_tool
    )
    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "leisaac-fallback-case",
            "--image",
            "registry.example/custom/proj/npa-isaac-lab-leisaac:test",
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
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "registry.example/example/project/npa-isaac-lab:test",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd and cmd[0] == sys.executable and str(module.ISAAC_RUNNER) in cmd:
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"status":"submitted"}\n', stderr=""
            )
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
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "registry.example/example/project/npa-isaac-lab:test",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd and cmd[0] == sys.executable and str(module.DATAGEN_RUNNER) in cmd:
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"status":"submitted"}\n', stderr=""
            )
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


def test_main_forwards_solution_smoke_to_container_runner(monkeypatch) -> None:
    module = _load_module()
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "resolve_byof_kubernetes_target",
        lambda *_args, **_kwargs: type(
            "Target",
            (),
            {
                "kubeconfig": "/tmp/kubeconfig",
                "context": "customer-mk8s",
                "namespace": "workbench",
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "storage_env_for_project",
        lambda *_args, **_kwargs: {
            "AWS_ENDPOINT_URL": "https://storage.example",
            "AWS_ACCESS_KEY_ID": "key",
        },
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if (
            cmd
            and cmd[0] == sys.executable
            and str(module.CONTAINER_VERIFY_RUNNER) in cmd
        ):
            seen["cmd"] = list(cmd)
            seen["env"] = dict(env or {})
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"status":"submitted"}\n', stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "oss-solution-smoke",
            "--skip-build",
            "--base-profile",
            "ubuntu",
            "--workload",
            "solution-smoke",
            "--smoke-command",
            "python3 -c 'print(42)'",
            "--solution-name",
            "demo-solution",
            "--capability-name",
            "demo-capability",
            "--smoke-artifact-name",
            "demo_artifact.json",
        ]
    )

    assert rc == 0
    cmd = seen.get("cmd")
    assert isinstance(cmd, list)
    assert str(module.CONTAINER_VERIFY_RUNNER) in cmd
    assert "--smoke-command" in cmd
    assert "python3 -c 'print(42)'" in cmd
    assert "--solution-name" in cmd and "demo-solution" in cmd
    assert "--capability-name" in cmd and "demo-capability" in cmd
    assert "--smoke-artifact-name" in cmd and "demo_artifact.json" in cmd
    assert "--no-direct-launch" not in cmd
    env = seen.get("env")
    assert isinstance(env, dict)
    assert env["KUBECONFIG"] == "/tmp/kubeconfig"
    assert env["KUBECONTEXT"] == "customer-mk8s"
    assert env["NPA_BYOF_K8S_CONTEXT"] == "customer-mk8s"
    assert env["NPA_BYOF_K8S_NAMESPACE"] == "workbench"
    assert env["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert env["AWS_ACCESS_KEY_ID"] == "key"


def test_main_forces_libero_solution_smoke_through_managed_scheduler(
    monkeypatch,
) -> None:
    module = _load_module()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/example/project",
    )
    monkeypatch.setattr(
        module,
        "resolve_byof_kubernetes_target",
        lambda *_args, **_kwargs: type(
            "Target",
            (),
            {
                "kubeconfig": str(Path("/") / "tmp" / "kubeconfig"),
                "context": "execution-context",
                "namespace": "workbench",
            },
        )(),
    )
    monkeypatch.setattr(module, "storage_env_for_project", lambda *_a, **_k: {})

    def fake_run(cmd, **_kwargs):
        if str(module.CONTAINER_VERIFY_RUNNER) in cmd:
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"status":"submitted"}\n', stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--run-id",
            "libero-managed-route",
            "--skip-build",
            "--base-profile",
            "ubuntu",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "libero",
            "--yaml",
            str(LIBERO_YAML_PATH),
        ]
    )

    assert rc == 0
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert cmd.count("--no-direct-launch") == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--solution-name", "LIBERO"], "exact --solution-name libero"),
        (["--solution-name", "libero", "--run-id", "short"], "SkyPilot run_id"),
        (
            [
                "--solution-name",
                "libero",
                "--run-id",
                "../libero-escape",
            ],
            "SkyPilot run_id",
        ),
        (
            [
                "--yaml",
                "byof-solution-smoke-libero-b200-gpu",
                "--run-id",
                "generic-safe-run-id",
            ],
            "exact --solution-name libero",
        ),
        (
            [
                "--solution-name",
                "libero",
                "--run-id",
                "libero-wrong-workload",
                "--workload",
                "datagen",
                "--yaml",
                str(LIBERO_YAML_PATH),
            ],
            "solution-smoke workload",
        ),
        (
            [
                "--solution-name",
                "libero",
                "--run-id",
                "libero-wrong-profile",
                "--workload",
                "solution-smoke",
                "--yaml",
                "generic.yaml",
            ],
            "exact B200 solution-smoke profile",
        ),
    ],
)
def test_main_refuses_ambiguous_libero_identity_before_registry_or_build(
    monkeypatch, extra, message
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_a, **_k: pytest.fail("registry must not resolve before refusal"),
    )

    with pytest.raises(ValueError, match=message):
        module.main(["--skip-build", "--skip-run", *extra])


def test_main_publishes_verified_wan_rrd_after_success(monkeypatch, capsys) -> None:
    module = _load_module()
    published: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )
    monkeypatch.setattr(module, "_live_runner_env", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout='{"status":"success"}\n', stderr=""
        ),
    )

    def fake_postprocess(key, context):
        published.update(key=key, uri=context.run_prefix_uri, project=context.project)
        return {"status": "verified", "capability": "wan2.2_verified_rerun_recording"}

    monkeypatch.setattr(module, "run_registered_postprocess", fake_postprocess)
    rc = module.main(
        [
            "--run-id",
            "wan-generic-run",
            *_accepted_wan_base_args(module),
            "--skip-build",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "wan2.2-multigpu",
            "--capability-name",
            "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
            "--smoke-artifact-name",
            "wan2_2_ti2v_5b_multigpu.json",
            "--output-root",
            "s3://example/wan2.2-multigpu",
            "--project",
            "wan-project",
        ]
    )

    assert rc == 0
    assert published == {
        "key": "wan2.2-multigpu",
        "uri": "s3://example/wan2.2-multigpu/wan-generic-run/",
        "project": "wan-project",
    }
    output = capsys.readouterr().out
    assert '"postprocess": {' in output
    assert '"capability": "wan2.2_verified_rerun_recording"' in output


def test_main_fails_before_launch_when_registered_wan_has_no_output_root(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    launched = False

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )

    def unexpected_run(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("registered Wan smoke must fail before launch")

    monkeypatch.setattr(module, "_run", unexpected_run)
    rc = module.main(
        [
            "--run-id",
            "wan-no-output-root",
            *_accepted_wan_base_args(module),
            "--skip-build",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "wan2.2",
            "--capability-name",
            "wan2.2_ti2v_5b_text_to_video",
            "--smoke-artifact-name",
            "wan2_2_ti2v_5b_text_to_video.json",
        ]
    )

    assert rc == 1
    assert launched is False
    output = capsys.readouterr().out
    assert "requires --output-root" in output
    assert '"status": "failed"' in output


@pytest.mark.parametrize("solution_name", [None, "wan2.2-typo"])
def test_immutable_wan_base_cannot_skip_postprocess_by_changing_solution_label(
    monkeypatch, capsys, solution_name: str | None
) -> None:
    module = _load_module()
    launched = False

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )

    def unexpected_run(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("invalid Wan postprocess contract must fail before launch")

    monkeypatch.setattr(module, "_run", unexpected_run)
    argv = [
        "--run-id",
        "wan-label-bypass",
        "--workload",
        "solution-smoke",
        *_accepted_wan_base_args(module),
        "--capability-name",
        "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
        "--smoke-artifact-name",
        "wan2_2_ti2v_5b_multigpu.json",
        "--output-root",
        "s3://example/wan2.2-multigpu",
    ]
    if solution_name is not None:
        argv.extend(["--solution-name", solution_name])

    assert module.main(argv) == 1
    assert launched is False
    output = capsys.readouterr().out
    assert "cannot disable verified RRD postprocessing" in output
    assert '"status": "failed"' in output


def test_registered_wan_cannot_report_success_with_skip_run(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )

    assert (
        module.main(
            [
                "--run-id",
                "wan-skip-run",
                *_accepted_wan_base_args(module),
                "--workload",
                "solution-smoke",
                "--solution-name",
                "wan2.2",
                "--capability-name",
                "wan2.2_ti2v_5b_text_to_video",
                "--smoke-artifact-name",
                "wan2_2_ti2v_5b_text_to_video.json",
                "--output-root",
                "s3://example/wan2.2",
                "--skip-run",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "cannot use --skip-run" in output
    assert '"status": "failed"' in output


def test_main_fails_closed_when_wan_rrd_publication_fails(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )
    monkeypatch.setattr(module, "_live_runner_env", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout='{"status":"success"}\n', stderr=""
        ),
    )
    monkeypatch.setattr(
        module,
        "run_registered_postprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("RRD verify failed")
        ),
    )

    rc = module.main(
        [
            "--run-id",
            "wan-generic-run",
            *_accepted_wan_base_args(module),
            "--skip-build",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "wan2.2",
            "--capability-name",
            "wan2.2_ti2v_5b_text_to_video",
            "--smoke-artifact-name",
            "wan2_2_ti2v_5b_text_to_video.json",
            "--output-root",
            "s3://example/wan2.2",
        ]
    )

    assert rc == 1
    output = capsys.readouterr().out
    assert '"status": "failed"' in output
    assert '"error": "RRD verify failed"' in output


def test_main_fails_closed_when_registered_postprocess_returns_no_result(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )
    monkeypatch.setattr(module, "_live_runner_env", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout='{"status":"success"}\n', stderr=""
        ),
    )
    monkeypatch.setattr(module, "run_registered_postprocess", lambda *_args: None)

    rc = module.main(
        [
            "--run-id",
            "wan-missing-postprocess",
            *_accepted_wan_base_args(module),
            "--skip-build",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "wan2.2",
            "--capability-name",
            "wan2.2_ti2v_5b_text_to_video",
            "--smoke-artifact-name",
            "wan2_2_ti2v_5b_text_to_video.json",
            "--output-root",
            "s3://example/wan2.2",
        ]
    )

    assert rc == 1
    output = capsys.readouterr().out
    assert "returned no verified postprocess result" in output
    assert '"status": "failed"' in output


def test_base_image_candidates_ubuntu_profile_default() -> None:
    module = _load_module()
    candidates = module._base_image_candidates(
        profile="ubuntu",
        image="registry.example/example/project/npa-byof:test",
        registry="registry.example/example/project",
        explicit_base="",
    )
    assert candidates == ["ubuntu:22.04"]


def test_base_image_candidates_explicit_base_overrides_profile() -> None:
    module = _load_module()
    candidates = module._base_image_candidates(
        profile="ubuntu",
        image="registry.example/example/project/npa-byof:test",
        registry="registry.example/example/project",
        explicit_base="ubuntu:24.04",
    )
    assert candidates == ["ubuntu:24.04"]


def test_prebuilt_profile_resolves_only_registered_tool_image(monkeypatch) -> None:
    module = _load_module()
    seen: dict[str, str] = {}

    def resolve(tool: str, *, registry: str):
        seen.update(tool=tool, registry=registry)
        return f"{registry}/npa-wan2-2:immutable"

    monkeypatch.setattr(module, "container_image_for_tool", resolve)
    candidates = module._base_image_candidates(
        profile="prebuilt",
        image="registry.example/project/npa-byof:test",
        registry="registry.example/project",
        explicit_base="tool://wan2-2",
    )
    digest = module.wan_accepted_image_manifest()["oci_digest"]
    assert candidates == [f"registry.example/project/npa-wan2-2@{digest}"]
    assert seen == {"tool": "wan2-2", "registry": "registry.example/project"}


def test_registered_wan_refuses_a_nonaccepted_base_digest(monkeypatch, capsys) -> None:
    module = _load_module()
    launched = False
    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/project",
    )

    def unexpected_run(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("nonaccepted Wan digest must fail before launch")

    monkeypatch.setattr(module, "_run", unexpected_run)
    rc = module.main(
        [
            "--run-id",
            "wan-wrong-digest",
            "--base-profile",
            "prebuilt",
            "--base-image",
            "registry.example/project/npa-wan2-2@sha256:" + "0" * 64,
            "--skip-build",
            "--workload",
            "solution-smoke",
            "--solution-name",
            "wan2.2",
            "--capability-name",
            "wan2.2_ti2v_5b_text_to_video",
            "--smoke-artifact-name",
            "wan2_2_ti2v_5b_text_to_video.json",
            "--output-root",
            "s3://example/wan2.2",
        ]
    )
    assert rc == 1
    assert launched is False
    assert "exact GPU-accepted prebuilt image digest" in capsys.readouterr().out


def test_registered_wan_allows_only_the_explicit_cli_acceptance_candidate() -> None:
    module = _load_module()
    candidate = (
        "ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:" + "a" * 64
    )
    args = module.argparse.Namespace(
        workload="solution-smoke",
        solution_name="wan2.2",
        capability_name="wan2.2_ti2v_5b_text_to_video",
        smoke_artifact_name="wan2_2_ti2v_5b_text_to_video.json",
        wan_acceptance_candidate_image=candidate,
        skip_build=True,
    )

    assert (
        module._required_postprocess_key(
            args, base_image=candidate, base_profile="prebuilt"
        )
        == "wan2.2"
    )


def test_registered_wan_ambient_live_environment_cannot_authorize_candidate(
    monkeypatch,
) -> None:
    module = _load_module()
    candidate = (
        "ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:" + "a" * 64
    )
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    monkeypatch.setenv("NPA_BYOF_WAN22_LIVE_GPU", "1")
    monkeypatch.setenv("NPA_BYOF_WAN22_REUSE_IMAGE", candidate)
    args = module.argparse.Namespace(
        workload="solution-smoke",
        solution_name="wan2.2",
        capability_name="wan2.2_ti2v_5b_text_to_video",
        smoke_artifact_name="wan2_2_ti2v_5b_text_to_video.json",
        wan_acceptance_candidate_image="",
        skip_build=True,
    )

    with pytest.raises(ValueError, match="exact GPU-accepted prebuilt image"):
        module._required_postprocess_key(
            args, base_image=candidate, base_profile="prebuilt"
        )

def test_closed_postprocess_registry_ignores_unregistered_solution() -> None:
    from npa.workflows.byof.postprocess import (
        PostprocessContext,
        run_registered_postprocess,
    )

    result = run_registered_postprocess(
        "untrusted.module:callable",
        PostprocessContext("s3://bucket/prefix/", None),
    )
    assert result is None


def test_closed_postprocess_registry_normalizes_solution_casing(monkeypatch) -> None:
    from npa.workflows.byof import postprocess
    from npa.workflows.byof.postprocess import (
        PostprocessContext,
        has_registered_postprocess,
        run_registered_postprocess,
    )

    seen: list[str] = []
    monkeypatch.setitem(
        postprocess.POSTPROCESSORS,
        "wan2.2",
        lambda context: {"run_prefix_uri": seen.append(context.run_prefix_uri)},
    )
    assert has_registered_postprocess("  WAN2.2  ")
    assert (
        run_registered_postprocess(
            "Wan2.2", PostprocessContext("s3://bucket/prefix/", None)
        )
        is not None
    )
    assert seen == ["s3://bucket/prefix/"]


def test_base_image_candidates_isaac_lab_profile(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "container_image_for_tool",
        lambda *_args, **_kwargs: "registry.example/example/project/npa-isaac-lab:test",
    )
    candidates = module._base_image_candidates(
        profile="isaac-lab",
        image="registry.example/example/project/npa-isaac-lab-leisaac:test",
        registry="registry.example/example/project",
        explicit_base="",
    )
    assert "nvcr.io/nvidia/isaac-lab:2.3.2" in candidates
    assert "nvcr.io/nvidia/isaac-sim:4.5.0" in candidates


def test_main_ubuntu_profile_uses_byof_base_image_build_arg(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    build_args: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/example/project",
    )

    def fake_run(cmd, *, stdin=None, capture=False, env=None):
        if cmd[:2] == ["docker", "build"]:
            build_args.extend(cmd)
        if cmd[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Digest: sha256:" + "b" * 64 + "\n",
                stderr="",
            )
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
            "--build-command",
            "python3 -m pip install -e .",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert any(part == "BYOF_BASE_IMAGE=ubuntu:22.04" for part in build_args)
    assert any(part == "BYOF_BASE_IMAGE_DIGEST=" for part in build_args)
    assert any(
        part == "BYOF_BUILD_COMMAND=python3 -m pip install -e ." for part in build_args
    )
    output = json.loads(capsys.readouterr().out)
    assert output["base_profile"] == "ubuntu"
    assert output["base_image"] == "ubuntu:22.04"
    assert output["build_command"] == "python3 -m pip install -e ."
    assert output["build"] == {
        "digest": "sha256:" + "b" * 64,
        "ok": True,
        "pushed": True,
        "runtime_image": (
            "registry.example/example/project/npa-byof@sha256:" + "b" * 64
        ),
    }
    assert output["image"] == output["build"]["runtime_image"]


def test_resolve_pushed_image_digest_fails_closed_without_digest(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="Name: registry/image:tag\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="immutable sha256 digest"):
        module._resolve_pushed_image_digest("registry.example:5000/team/image:tag")


def test_dockerfile_writes_metadata_without_python_dependency() -> None:
    module = _load_module()
    text = module._dockerfile_text()
    assert "BYOF_BASE_IMAGE" in text
    assert "BYOF_BUILD_COMMAND" in text
    assert "npa.byof.build.v1" in text
    assert "build_command_executed" in text
    assert "build_command_sha256" in text
    assert "base_image_reference" in text
    assert "base_image_digest" in text
    assert "base_image_digest_pinned" in text
    assert 'org.opencontainers.image.base.name="${BYOF_BASE_IMAGE}"' in text
    assert 'org.opencontainers.image.base.digest="${BYOF_BASE_IMAGE_DIGEST}"' in text
    assert "sha256sum" in text
    assert "npa_build_metadata.json" in text
    assert "npa_source_metadata.json" in text
    assert "BYOF_SOURCE_PRUNE_PATH" in text
    assert 'observed_commit="$(git -C /opt/byof rev-parse HEAD)"' in text
    assert 'rm -rf -- "/opt/byof/${source_prune_path}" /opt/byof/.git' in text
    assert "printf" in text
    assert "/opt/byof" in text
    assert "USER ubuntu" in text
    assert 'npa.packaging.tier="interactive"' in text
    assert "useradd" in text
    assert "python3" in text
    assert "NOPASSWD:ALL" in text
    assert "sudo" in text
    assert "mkdir -p /workspace" in text
    assert "openssh-server" in text
    assert "rsync" in text
    assert "curl" in text
    assert "wget" in text
    assert "gcc" in text
    assert "patch" in text
    assert "pciutils" in text
    assert "fuse3" in text
    assert "linux-libc-dev" in text
    assert "netcat-openbsd" in text
    assert "npa-skypilot-bootstrap-guard verify" in text
    assert "NPA_SKYPILOT_BOOTSTRAP_APT_BYPASSED" in text
    assert "--kill-after=5s" in text
    assert "ssh-keygen -A" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text
    assert "ENV HOME=/home/ubuntu" in text
    assert 'exec \\"$@\\"' in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert (
        'org.nebius.npa.byof-bootstrap-guard="skypilot-0.12.2-v1"' in text
    )


def test_immutable_image_digest_accepts_only_exact_digest_suffix() -> None:
    module = _load_module()
    digest = "sha256:" + "a" * 64

    assert (
        module._immutable_image_digest(f"registry.example/team/image@{digest}")
        == digest
    )
    assert module._immutable_image_digest("registry.example/team/image:latest") == ""
    assert (
        module._immutable_image_digest(f"registry.example/team/image@{digest}:tag")
        == ""
    )


def test_dockerfile_refreshes_linux_libc_dev_without_broad_upgrade() -> None:
    module = _load_module()
    text = module._dockerfile_text()
    install_layer = text.split(
        "apt-get install -y --no-install-recommends", 1
    )[1].split("&& rm -rf /var/lib/apt/lists/*", 1)[0]

    assert "linux-libc-dev" in install_layer.split()
    assert "linux-libc-dev=" not in install_layer
    assert "apt-get upgrade" not in text
    assert "apt-get dist-upgrade" not in text


def test_dockerfile_refuses_missing_security_refresh_package() -> None:
    module = _load_module()
    text = module._dockerfile_text()
    without_linux_libc_dev = text.replace(" linux-libc-dev", "", 1)

    with pytest.raises(RuntimeError, match="missing required security refresh"):
        module._validate_byof_image_security_refresh(without_linux_libc_dev)


def _bootstrap_guard_fixture(tmp_path, module, *, missing: str = ""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "bootstrap-apt.state"
    contract_failure = tmp_path / "bootstrap-contract.failed"
    sky_failure = tmp_path / "apt-ssh-setup.failed"
    apt_complete = tmp_path / "apt-ssh-setup.complete"
    apt_calls = tmp_path / "real-apt.calls"
    real_apt = tmp_path / "real-apt-get"
    real_apt.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NPA_TEST_APT_CALLS\"\n",
        encoding="utf-8",
    )
    real_apt.chmod(0o755)

    dpkg_query = bin_dir / "dpkg-query"
    dpkg_query.write_text(
        """#!/bin/sh
format=
package=
for argument in "$@"; do
    case "$argument" in -f=*) format=${argument#-f=} ;; esac
    package=$argument
done
if [ "${NPA_TEST_MISSING:-}" = netcat ]; then
    case "$package" in netcat|netcat-openbsd|netcat-traditional) exit 1 ;; esac
elif [ "$package" = "${NPA_TEST_MISSING:-}" ]; then
    exit 1
fi
case "$format" in
    '${Status}') printf '%s' 'install ok installed' ;;
    '${Provides}')
        if [ "${NPA_TEST_MISSING:-}" != fuse ]; then
            printf '%s' 'fuse (= 3.10.5)'
        fi
        ;;
    *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    dpkg_query.chmod(0o755)

    for command_name in (
        "sudo",
        "sshd",
        "rsync",
        "service",
        "curl",
        "wget",
        "nc",
        "gcc",
        "patch",
        "lspci",
        "fusermount",
        "fusermount3",
    ):
        (bin_dir / command_name).symlink_to("/bin/true")

    guard_text = module._skypilot_bootstrap_guard_script()
    skypilot_tmp = Path("/") / "tmp"
    replacements = {
        "/usr/bin/apt-get": str(real_apt),
        "/usr/bin/timeout": shutil.which("timeout") or "/usr/bin/timeout",
        str(skypilot_tmp / "npa-skypilot-bootstrap-apt.state"): str(state),
        str(skypilot_tmp / "npa-skypilot-bootstrap-contract.failed"): str(
            contract_failure
        ),
        str(skypilot_tmp / "apt-ssh-setup.failed"): str(sky_failure),
        str(skypilot_tmp / "apt_ssh_setup_complete"): str(apt_complete),
    }
    for source, target in replacements.items():
        guard_text = guard_text.replace(source, target)

    guard = bin_dir / "npa-skypilot-bootstrap-guard"
    guard.write_text(guard_text, encoding="utf-8")
    guard.chmod(0o755)
    (bin_dir / "apt-get").symlink_to(guard)
    (bin_dir / "timeout").symlink_to(guard)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "NPA_TEST_APT_CALLS": str(apt_calls),
        "NPA_TEST_MISSING": missing,
        "SKYPILOT_POD_NODE_TYPE": "head",
    }
    return {
        "apt_calls": apt_calls,
        "bin_dir": bin_dir,
        "contract_failure": contract_failure,
        "env": env,
        "guard": guard,
        "sky_failure": sky_failure,
        "state": state,
    }


@pytest.mark.parametrize(
    "missing",
    (
        "rsync",
        "curl",
        "wget",
        "netcat",
        "gcc",
        "patch",
        "pciutils",
        "fuse",
        "fuse3",
        "openssh-server",
    ),
)
def test_bootstrap_guard_rejects_every_missing_skypilot_package(
    tmp_path, missing
) -> None:
    module = _load_module()
    assert module.SKYPILOT_BOOTSTRAP_PACKAGE_CAPABILITIES == (
        "rsync",
        "curl",
        "wget",
        "netcat",
        "gcc",
        "patch",
        "pciutils",
        "fuse",
        "fuse3",
        "openssh-server",
    )
    fixture = _bootstrap_guard_fixture(tmp_path, module, missing=missing)

    result = subprocess.run(
        [fixture["guard"], "verify"],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 86
    assert "NPA_SKYPILOT_BOOTSTRAP_FAILED status=86" in result.stderr
    assert fixture["contract_failure"].is_file()
    assert fixture["sky_failure"].is_file()


def test_complete_bootstrap_contract_bypasses_only_skypilot_apt_setup(
    tmp_path,
) -> None:
    module = _load_module()
    fixture = _bootstrap_guard_fixture(tmp_path, module)
    apt_get = fixture["bin_dir"] / "apt-get"

    update = subprocess.run(
        [apt_get, "update", "-o", "Acquire::Retries=0"],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
    )
    install = subprocess.run(
        [apt_get, "install", "-o", "Dpkg::Options::=--force-confold", "-y", "fuse"],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert update.returncode == 0
    assert install.returncode == 0
    assert "operation=update" in update.stdout
    assert "operation=install package=fuse provider=fuse3" in install.stdout
    assert fixture["state"].read_text(encoding="utf-8") == "complete\n"
    assert not fixture["apt_calls"].exists()

    ordinary_apt = subprocess.run(
        [apt_get, "update"],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ordinary_apt.returncode == 0
    assert fixture["apt_calls"].read_text(encoding="utf-8") == "update\n"


def test_bootstrap_timeout_kills_nonterminating_descendant_and_marks_failure(
    tmp_path,
) -> None:
    module = _load_module()
    fixture = _bootstrap_guard_fixture(tmp_path, module)
    timeout_guard = fixture["bin_dir"] / "timeout"
    guard_text = timeout_guard.resolve().read_text(encoding="utf-8")
    timeout_guard.resolve().write_text(
        guard_text.replace("--kill-after=5s", "--kill-after=0.2s"),
        encoding="utf-8",
    )
    descendant_pid_path = tmp_path / "descendant.pid"
    command = (
        "trap '' TERM; "
        "/bin/bash --noprofile --norc -c "
        "'trap \"\" TERM; printf \"%s\\n\" \"$$\" > \"$1\"; while :; do :; done' "
        f"descendant {descendant_pid_path} & wait"
    )

    result = subprocess.run(
        [timeout_guard, "0.1s", "/bin/bash", "--noprofile", "--norc", "-c", command],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )

    assert result.returncode in {124, 137}
    assert "NPA_SKYPILOT_BOOTSTRAP_FAILED" in result.stderr
    assert "detail=deadline-exceeded" in result.stderr
    assert fixture["contract_failure"].is_file()
    assert fixture["sky_failure"].is_file()
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and Path(f"/proc/{descendant_pid}").exists():
            state = Path(f"/proc/{descendant_pid}/stat").read_text().split()[2]
            if state == "Z":
                break
            time.sleep(0.02)
        if Path(f"/proc/{descendant_pid}").exists():
            assert Path(f"/proc/{descendant_pid}/stat").read_text().split()[2] == "Z"
    finally:
        try:
            os.kill(descendant_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize(
    "value",
    ["/absolute", "../escape", "assets/../escape", ".git/objects", "with space"],
)
def test_source_prune_path_rejects_unsafe_values(value: str) -> None:
    module = _load_module()

    with pytest.raises(argparse.ArgumentTypeError, match="safe relative repository path"):
        module._source_prune_path(value)


def test_source_prune_path_is_passed_to_the_clone_layer(
    monkeypatch, capsys
) -> None:
    module = _load_module()
    build_args: list[str] = []

    monkeypatch.setattr(
        module,
        "resolve_container_registry",
        lambda *_args, **_kwargs: "registry.example/example/project",
    )

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["docker", "build"]:
            build_args.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run", fake_run)
    rc = module.main(
        [
            "--repo-url",
            "https://github.com/example/demo.git",
            "--repo-ref",
            "0123456789abcdef0123456789abcdef01234567",
            "--source-prune-path",
            "assets/render-only",
            "--skip-push",
            "--skip-run",
        ]
    )

    assert rc == 0
    assert "BYOF_SOURCE_PRUNE_PATH=assets/render-only" in build_args
    assert json.loads(capsys.readouterr().out)["source_prune_path"] == (
        "assets/render-only"
    )


def test_compat_shim_delegates_to_run_byof_repo() -> None:
    shim_path = ROOT / "npa" / "scripts" / "run_isaac_lab_byof_repo.py"
    spec = importlib.util.spec_from_file_location(
        "run_isaac_lab_byof_repo_shim", shim_path
    )
    assert spec and spec.loader
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.main.__module__ == "run_byof_repo"

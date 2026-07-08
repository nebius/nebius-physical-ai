from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_container_verify.py"
YAML_PATH = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "byof-container-smoke-rtxpro.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_byof_container_verify", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_workflow_injects_solution_smoke_metadata() -> None:
    module = _load_module()
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://bucket/prefix",
        image="registry.example/npa-byof:demo",
        smoke_command="python -c 'print(42)'",
        solution_name="demo-solution",
        capability_name="demo-capability",
        smoke_artifact_name="demo_artifact.json",
    )

    task = docs[1]
    envs = task["envs"]
    assert envs["BYOF_SMOKE_COMMAND"] == "python -c 'print(42)'"
    assert envs["BYOF_SOLUTION_NAME"] == "demo-solution"
    assert envs["BYOF_CAPABILITY_NAME"] == "demo-capability"
    assert envs["BYOF_SMOKE_ARTIFACT_NAME"] == "demo_artifact.json"
    assert envs["S3_OUTPUT_PREFIX"] == "s3://bucket/prefix/byof-demo/"
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_default_infra_uses_resolved_kubernetes_context(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_K8S_CONTEXT", "customer-mk8s")
    monkeypatch.delenv("NPA_BYOF_INFRA", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_INFRA", raising=False)
    assert module._default_infra() == "k8s/customer-mk8s"


def test_ensure_infra_enabled_runs_sky_check_for_kubernetes(monkeypatch) -> None:
    module = _load_module()
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout='{"default": {"Kubernetes": ["compute"]}}', stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
    )

    assert seen == [
        ["/opt/sky", "api", "stop"],
        ["/opt/sky", "check", "kubernetes", "-o", "json", "--config", "/tmp/skypilot.yaml"],
    ]


def test_ensure_infra_enabled_skips_non_kubernetes(monkeypatch) -> None:
    module = _load_module()
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="aws/us-east-1")
    assert called is False


def test_direct_launch_uses_sky_launch_with_down(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    rendered_yaml = tmp_path / "workflow.yaml"
    rendered_yaml.write_text("name: demo\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rc = module._direct_launch(
        rendered_yaml=rendered_yaml,
        run_id="byof-demo",
        outputs={"summary": "s3://bucket/summary.json"},
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
        cleanup=True,
    )

    assert rc == 0
    assert seen["cmd"] == [
        "/opt/sky",
        "launch",
        "--yes",
        "--cluster",
        "byof-demo",
        "--name",
        "byof-demo",
        "--down",
        "--infra",
        "k8s/customer-mk8s",
        "--config",
        "/tmp/skypilot.yaml",
        str(rendered_yaml),
    ]
    output = capsys.readouterr().out
    assert '"mode": "direct-launch"' in output

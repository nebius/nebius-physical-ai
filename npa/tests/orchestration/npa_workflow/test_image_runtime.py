"""Customer image runtimes keep the workflow lifecycle without NPA bootstrap."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex

import pytest
import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow import skypilot_render as renderer
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    instrument_workflow_yaml,
)


IMAGE = "registry.example/application@sha256:" + "a" * 64


def _spec(tmp_path: Path, *, config=None, image=IMAGE, run=None, tool_ref=""):
    state = {"resources": "application", "terminal": True}
    if tool_ref:
        state["toolRef"] = tool_ref
    else:
        state["run"] = run or {"argv": ["/opt/application/bin/python", "driver.py"]}
    raw = {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "image-runtime"},
        "config": {"runtime_setup": "image"} if config is None else config,
        "resources": {
            "application": {"cloud": "kubernetes", "cpus": 2, "image": image}
        },
        "states": {"application": state},
    }
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_spec(path)


def _render(spec, **options):
    plan = build_plan(spec, run_id="test-image-runtime")
    return renderer.build_skypilot_task_doc(
        spec,
        plan.steps[0],
        run_id="test-image-runtime",
        options=renderer.SkypilotRenderOptions(
            materialize_registry_secrets=False, **options
        ),
    )


def test_image_runtime_never_resolves_or_injects_npa_source(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_SRC_OVERLAY", "1")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-source")

    def unexpected_source_lookup():
        pytest.fail("an application image must not resolve NPA source configuration")

    monkeypatch.setattr(renderer, "resolve_src_s3_uri", unexpected_source_lookup)
    doc = _render(_spec(tmp_path))

    assert "setup" not in doc
    assert doc["resources"]["image_id"] == "docker:" + IMAGE
    assert doc["envs"]["NPA_TASK_IMAGE"] == IMAGE
    assert "NPA_SRC_S3_URI" not in doc["envs"]
    assert "NPA_SRC_OVERLAY" not in doc["envs"]
    assert "NPA_SIM2REAL_SOURCE_SHA" not in doc["envs"]
    assert "npa-src" not in doc["run"]
    assert "npa-shim" not in doc["run"]
    assert "pip install" not in doc["run"]
    assert "import npa" not in doc["run"]
    assert "PYTHONPATH" not in doc["run"]
    assert doc["run"].endswith("\n/opt/application/bin/python driver.py\n")


@pytest.mark.parametrize("mode", ["", "Image", "skip", True, None, {}])
def test_invalid_runtime_setup_rejected_before_planning(tmp_path, mode):
    with pytest.raises(NpaWorkflowError, match="runtime_setup must be"):
        _spec(tmp_path, config={"runtime_setup": mode})


@pytest.mark.parametrize("required", [True, "1", "true", "yes", "on"])
def test_image_runtime_rejects_conflicting_npa_attestation(tmp_path, required):
    with pytest.raises(NpaWorkflowError, match="cannot be combined"):
        _spec(
            tmp_path,
            config={"runtime_setup": "image", "require_baked_npa": required},
        )


def test_image_runtime_rejects_toolref_before_planning(tmp_path):
    with pytest.raises(NpaWorkflowError, match="toolRef requires the NPA runtime"):
        _spec(tmp_path, tool_ref="workbench.lancedb.backfill_clip")


def test_config_override_cannot_bypass_toolref_runtime(tmp_path):
    spec = _spec(
        tmp_path,
        config={
            "lance_table": "images", "lance_uri": "s3://example-bucket/lance/",
            "lancedb_endpoint": "http://localhost:8686",
        },
        tool_ref="workbench.lancedb.backfill_clip",
    )
    with pytest.raises(NpaWorkflowError, match="toolRef requires the NPA runtime"):
        merge_config_overrides(spec, {"runtime_setup": "image"})


@pytest.mark.parametrize(
    "image",
    ["", "ubuntu:24.04", "registry.example/app:latest", "app@sha256:" + "a" * 64,
     "registry.example/app@sha256:abc", "https://registry.example/app@sha256:" + "a" * 64],
)
def test_image_runtime_requires_resolved_immutable_image(tmp_path, image):
    with pytest.raises(renderer.NpaWorkflowRenderError, match="immutable image"):
        _render(_spec(tmp_path, image=image))


def test_image_runtime_checks_final_image_override(tmp_path):
    spec = _spec(tmp_path, image="registry.example/application:runtime")
    doc = _render(spec, image_overrides={"*": IMAGE})
    assert doc["resources"]["image_id"] == "docker:" + IMAGE
    with pytest.raises(renderer.NpaWorkflowRenderError, match="immutable image"):
        _render(_spec(tmp_path), image_overrides={"*": "registry.example/app:latest"})


def test_image_runtime_preserves_cache_and_workflow_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("NPA_MODEL_CACHE_DISABLED", raising=False)
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", "/opt/application-cache")
    monkeypatch.setenv("NPA_MODEL_CACHE_PVC", "example-cache")
    doc = _render(
        _spec(tmp_path), execution_attempt_id="application-attempt",
        execution_fence_sequence=3, execution_fence_attempt=2,
    )
    assert doc["envs"]["HF_HOME"].startswith("/opt/application-cache/")
    assert doc["envs"]["NPA_WORKFLOW_RUN_ID"] == "test-image-runtime"
    assert doc["envs"]["NPA_WORKFLOW_ATTEMPT_ID"] == "application-attempt"
    assert doc["envs"]["NPA_WORKFLOW_FENCE_SEQUENCE"] == "3"
    assert doc["envs"]["NPA_WORKFLOW_FENCE_ATTEMPT"] == "2"
    assert "mkdir -p /opt/application-cache/" in doc["run"]
    pod = doc["config"]["kubernetes"]["pod_config"]["spec"]
    assert any(
        volume.get("persistentVolumeClaim", {}).get("claimName") == "example-cache"
        for volume in pod["volumes"]
    )


def test_image_runtime_retains_registry_credential_hook(tmp_path, monkeypatch):
    calls = []

    def registry_hook(doc, *, materialize):
        calls.append((doc["resources"]["image_id"], materialize))

    monkeypatch.setattr(renderer, "_inject_operator_registry_docker_secrets", registry_hook)
    _render(_spec(tmp_path))
    assert calls == [("docker:" + IMAGE, False)]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_image_runtime_executes_selected_interpreter_with_literal_argv(
    tmp_path, exit_code
):
    argument = "literal spaces; $(touch unexpected); `touch unexpected`"
    program = (
        "import json,os,sys; "
        "print(json.dumps({'arg':sys.argv[1], 'path':os.environ['PATH'], "
        "'pythonpath':os.environ['PYTHONPATH'], 'executable':sys.executable})); "
        f"sys.exit({exit_code})"
    )
    spec = _spec(tmp_path, run={"argv": [sys.executable, "-c", program, argument]})
    doc = _render(spec)
    env = dict(os.environ, PATH="/application-only", PYTHONPATH="/application-source")
    result = subprocess.run(
        ["/bin/bash", "-c", doc["run"]], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == {
        "arg": argument, "path": "/application-only",
        "pythonpath": "/application-source", "executable": sys.executable,
    }
    assert not (tmp_path / "unexpected").exists()


def test_image_runtime_preserves_raw_shell_program(tmp_path):
    spec = _spec(tmp_path, run={"shell": "printf '%s\\n' 'application result'; exit 9"})
    result = subprocess.run(
        ["bash", "-c", _render(spec)["run"]], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 9
    assert result.stdout == "application result\n"


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("command_form", ["argv", "shell"])
def test_image_runtime_finalizes_durable_status_after_application_exit(
    tmp_path, exit_code, command_form
):
    program = (
        "import json,sys; "
        "print(json.dumps({'application': 'completed', 'executable': sys.executable})); "
        f"sys.exit({exit_code})"
    )
    argv = [sys.executable, "-c", program]
    run = {"argv": argv} if command_form == "argv" else {"shell": shlex.join(argv)}
    original = _render(_spec(tmp_path, run=run))
    sky_yaml = tmp_path / "sky.yaml"
    sky_yaml.write_text(yaml.safe_dump(original), encoding="utf-8")
    mount = tmp_path / "state-mount"
    state = WorkflowS3Config(
        bucket="example-bucket", prefix="example-run", endpoint_url="https://s3.example.invalid"
    )
    instrumented = instrument_workflow_yaml(
        sky_yaml, run_id="test-image-runtime", state=state, mount_path=str(mount)
    )
    doc = yaml.safe_load(instrumented.yaml_text)

    result = subprocess.run(
        ["/bin/bash", "-c", doc["run"]], cwd=tmp_path,
        env={**os.environ, **doc["envs"]},
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == {
        "application": "completed", "executable": sys.executable,
    }
    log_dir = mount / state.prefix / "logs" / original["name"]
    status = json.loads((log_dir / "status.json").read_text())
    assert status["state"] == ("SUCCEEDED" if exit_code == 0 else "FAILED")
    assert status["tier"] == ("WORKS" if exit_code == 0 else "PARTIAL")
    assert status["error_summary"] == ("" if exit_code == 0 else "exit 7")
    assert status["start"] and status["end"]
    assert status["start_time"] == status["start"]
    assert status["end_time"] == status["end"]
    assert status["end"] >= status["start"]
    assert (log_dir / "run.log").read_text() == result.stdout


def test_default_npa_mode_still_bootstraps_and_uses_interpreter_shim(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-source")
    default = _render(_spec(tmp_path, config={}))
    explicit = _render(_spec(tmp_path, config={"runtime_setup": "npa"}))
    assert default == explicit
    assert "pip install" in default["setup"]
    assert "npa-shim" in default["run"]
    assert default["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-source"


def test_attested_npa_mode_keeps_its_existing_source_check(tmp_path):
    doc = _render(_spec(tmp_path, config={"require_baked_npa": True, "source_sha": "b" * 40}))
    assert "NPA_IMAGE_SOURCE_SHA" in doc["setup"]
    assert "actual != expected" in doc["setup"]
    assert doc["envs"]["NPA_SIM2REAL_SOURCE_SHA"] == "b" * 40
    assert "npa-shim" in doc["run"]


@pytest.mark.parametrize(("nodes", "gpus_per_node"), [(1, 1), (2, 1), (1, 2)])
def test_ray_session_renders_rank_aware_image_and_compilable_bootstrap(nodes, gpus_per_node):
    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/workbench/npa-workflows/ray-clip-development-session.yaml"
    )
    spec = merge_config_overrides(load_spec(path), {
        "nodes": str(nodes), "gpus_per_node": str(gpus_per_node),
        "accelerator": f"RTXPRO6000:{gpus_per_node}",
    })
    plan = build_plan(spec, run_id="ray-session-compile")
    assert len(plan.steps) == 1
    doc = _render(spec)
    assert doc["resources"]["image_id"] == "docker:" + spec.config["runtime_image"]
    assert doc.get("num_nodes", 1) == nodes
    assert doc["resources"]["accelerators"] == f"RTXPRO6000:{gpus_per_node}"
    assert "setup" not in doc
    assert "NPA_SRC_S3_URI" not in doc["envs"]
    # Both programs actually pass their language parsers after config/run token
    # resolution. This catches indentation/quoting drift in the long-lived
    # session before acquiring a GPU; it deliberately does not execute bootstrap.
    shell = plan.steps[0].shell
    bootstrap = shell.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(bootstrap, str(path), "exec")
    result = subprocess.run(
        ["bash", "-n", "-c", doc["run"]], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ray_session_application_ports_are_disjoint_from_management_runtime():
    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/workbench/npa-workflows/ray-clip-development-session.yaml"
    )
    shell = build_plan(load_spec(path), run_id="ray-port-check").steps[0].shell
    bootstrap = shell.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    arguments = {
        node.value.split("=", 1)[0]: node.value.split("=", 1)[1]
        for node in ast.walk(ast.parse(bootstrap))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
        and "=" in node.value
    }
    # SkyPilot 0.12.2 starts its management Ray with object port 8076 and
    # worker range 11002..65535: Ray 2.9.3 expands an omitted maximum to 65535.
    # Ray defaults also reserve Client 10001 and Dashboard agent HTTP 52365.
    # These ports were a concrete collision in the first application recipe.
    management_fixed = {6380, 8266, 8076, 10001, 52365}
    app_ports = [
        int(arguments[option])
        for option in (
            "--port", "--dashboard-port", "--object-manager-port",
            "--node-manager-port", "--ray-client-server-port",
            "--dashboard-agent-listen-port", "--dashboard-agent-grpc-port",
            "--runtime-env-agent-port", "--metrics-export-port",
        )
    ]
    first = int(arguments["--min-worker-port"])
    last = int(arguments["--max-worker-port"])
    assert 1 <= first <= last <= 10999
    assert set(range(first, last + 1)).isdisjoint(range(11002, 65536))
    assert len(app_ports) == len(set(app_ports))
    assert set(app_ports).isdisjoint(management_fixed)
    assert all(1 <= port <= 10999 for port in app_ports)
    assert all(not first <= port <= last for port in app_ports)
    assert all(not 11002 <= port <= 65535 for port in app_ports)
    assert arguments["--dashboard-host"] == "127.0.0.1"

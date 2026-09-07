"""Detached operator runtimes must not borrow state or activate host profiles."""

from __future__ import annotations

import ast
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli import agent
from npa.clients import nebius


def test_fresh_imports_resolve_all_operator_roots_from_config_dir(tmp_path):
    root = tmp_path / "private-runtime"
    root.mkdir()
    script = """
import json
from npa.clients import config, credentials
from npa.cluster import state
from npa.deploy import provisioner
from npa.orchestration.skypilot import _bin
from npa.cli.agent import _auth_secret_path
print(json.dumps([str(path) for path in (
    config.CONFIG_PATH, credentials.CREDENTIALS_PATH, state.CLUSTERS_DIR,
    provisioner.working_dir_path('owned', 'agent'),
    provisioner._TF_PLUGIN_CACHE_DIR, _bin.CONFIG_PATH,
    _auth_secret_path('owned', 'agent'),
)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "NPA_CONFIG_DIR": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(value) for value in json.loads(result.stdout)]
    assert len(paths) == 7
    assert all(path.is_relative_to(root) for path in paths)
    assert paths[-1] == root / "agents" / "owned" / "agent" / "auth.env"


def test_auth_creation_and_cleanup_stay_in_selected_runtime(tmp_path, monkeypatch):
    from npa.deploy import provisioner

    root = tmp_path / "selected"
    other = tmp_path / "other" / "agents" / "owned" / "agent" / "auth.env"
    other.parent.mkdir(parents=True)
    other.write_text("unrelated state")
    monkeypatch.setenv("NPA_CONFIG_DIR", str(root))
    monkeypatch.setattr(provisioner, "_WORKBENCH_BASE", root / "workbenches")
    owned_tf = provisioner.working_dir_path("owned", "agent")
    owned_tf.mkdir(parents=True)
    (owned_tf / "terraform.tfstate").write_text("owned state")
    secret = agent._write_auth_secret(
        project_alias="owned", name="agent", user="test", password="synthetic"
    )
    assert secret == root / "agents" / "owned" / "agent" / "auth.env"
    assert secret.stat().st_mode & 0o777 == 0o600
    agent._cleanup_agent_local_files("owned", "agent")
    assert not secret.exists()
    assert not owned_tf.exists()
    assert other.read_text() == "unrelated state"


@pytest.mark.parametrize(
    ("npa_profile", "provider_profile", "args", "expected"),
    [
        ("scoped", "ambient", ["iam", "get-access-token"], ["--profile", "scoped"]),
        ("", "ambient", ["iam", "whoami"], ["--profile", "ambient"]),
        ("scoped", "ambient", ["--profile", "explicit", "iam", "whoami"], []),
        ("scoped", "ambient", ["--profile=explicit", "iam", "whoami"], []),
        ("", "", ["iam", "whoami"], []),
    ],
)
def test_provider_calls_select_profile_without_mutating_configuration(
    tmp_path, monkeypatch, npa_profile, provider_profile, args, expected
):
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", npa_profile)
    monkeypatch.setenv("NEBIUS_PROFILE", provider_profile)
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "synthetic-stale-token")
    monkeypatch.setattr(nebius, "_require_nebius", lambda: "nebius")
    calls = []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="observed", stderr="")

    monkeypatch.setattr(nebius.subprocess, "run", capture)
    assert nebius._run(args) == "observed"
    assert calls[0][0] == ["nebius", *expected, *args]
    assert "NEBIUS_IAM_TOKEN" not in calls[0][1]["env"]
    assert len(calls) == 1


def test_deploy_does_not_activate_profile_before_project_resolution(monkeypatch):
    class ReachedProjectResolution(Exception):
        pass

    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "scoped")
    monkeypatch.setattr(agent, "_resolve_project_alias", lambda _project: "owned")
    monkeypatch.setattr(agent.shutil, "which", lambda _name: "/usr/bin/nebius")

    def resolve(*_args, **_kwargs):
        raise ReachedProjectResolution

    def unexpected(*_args, **_kwargs):
        raise AssertionError("No provider mutation before project resolution")

    monkeypatch.setattr(agent, "resolve_environment", resolve)
    monkeypatch.setattr(agent.subprocess, "run", unexpected)
    result = CliRunner().invoke(agent.app, ["deploy", "--project", "owned"])
    assert isinstance(result.exception, ReachedProjectResolution), result.output


def test_rendered_backend_reads_selected_configuration_and_cluster_root(
    tmp_path, monkeypatch
):
    helpers = runpy.run_path(str(Path(__file__).with_name("test_agent_backend_render.py")))
    source = helpers["_render_backend_body"](monkeypatch)
    functions = {
        "_load_agent_config_yaml", "_agent_project_alias", "_agent_k8s_backends"
    }
    nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    assert len(nodes) == 3
    root = tmp_path / "selected"
    root.mkdir()
    (root / "config.yaml").write_text("default_project: owned\n")
    monkeypatch.setenv("NPA_CONFIG_DIR", str(root))
    seen = {}

    def assemble(**kwargs):
        seen.update(kwargs)
        return {}

    namespace = {
        "Path": Path, "os": os, "NPA_PROJECT_ALIAS": "fallback",
        "NPA_CLUSTER_TERRAFORM_DIR": tmp_path / "terraform",
        "_agent_npa_ready": lambda: (True, ""),
        "_agent_cloud_mk8s_clusters": lambda _alias: [],
        "assemble_k8s_backend_inventory": assemble,
        "_configured_healthy_agent_exists": lambda *_args: False,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "backend", "exec"), namespace)
    namespace["_agent_k8s_backends"]()
    assert seen["alias"] == "owned"
    assert seen["clusters_root"] == root / "clusters"
    assert seen["config"] == {"default_project": "owned"}

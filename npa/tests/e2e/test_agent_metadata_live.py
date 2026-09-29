"""Verify staged metadata credentials on an isolated real CPU agent lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import httpx
import pytest

from npa.cli import agent
from npa.cli.agent_deployment import assert_live_deployment


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        not os.environ.get("NPA_AGENT_METADATA_LIVE_CONFIG"),
        reason="Requires private configuration for a disposable CPU agent.",
    ),
]


def _write_evidence(directory: Path, name: str, value: str) -> None:
    path = directory / name
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def _invoke(args: list[str], evidence: Path, label: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "npa", *args], capture_output=True, text=True
    )
    _write_evidence(evidence, f"{label}.stdout", result.stdout)
    _write_evidence(evidence, f"{label}.stderr", result.stderr)
    assert result.returncode == 0, f"{label} failed; inspect private evidence"
    return json.loads(result.stdout) if "--json" in args else {}


@pytest.fixture
def deployment():
    path = Path(os.environ["NPA_AGENT_METADATA_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "Live config must be owner-only"
    config = json.loads(path.read_text())
    args = config["deploy_args"]
    assert args[:2] == ["agent", "deploy"] and "--agent-only" in args
    project, name = (args[args.index(flag) + 1] for flag in ("--project", "--name"))
    assert not agent._agent_record(project, name), "Choose an unused agent name"
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0, "Evidence must be owner-only"
    try:
        yield args, project, name, evidence
    finally:
        # The project can have agents outside this private configuration copy.
        # Keep its shared IAM and verify only this run's exact infrastructure.
        cleanup = _invoke(
            [
                "agent",
                "destroy",
                "--project",
                project,
                "--name",
                name,
                "--keep-iam",
                "--yes",
                "--json",
            ],
            evidence,
            "destroy",
        )
        assert cleanup["infrastructure_absent"] is True
        assert cleanup["verified"] is True


def _request(client, route: str, evidence: Path, label: str) -> tuple[dict, float]:
    started = time.monotonic()
    response = client.get(route)
    elapsed = time.monotonic() - started
    _write_evidence(evidence, f"{label}.json", response.text)
    assert response.status_code == 200, f"{label} did not return HTTP 200"
    return response.json(), elapsed


def _deployed_source_hashes(project: str, name: str, evidence: Path) -> dict:
    record = agent._agent_record(project, name)
    ssh = agent.SSHClient(
        config=agent.resolve_ssh_config(
            ssh_host=record["public_ip"],
            ssh_user=record.get("ssh_user", "ubuntu"),
            ssh_key=record["ssh_key_path"],
            project=None,
            name=None,
        ).ssh
    )
    modules = ["agent", "agent_access_runtime", "agent_resources", "agent_env_files"]
    script = (
        "import hashlib, importlib, json; from pathlib import Path; "
        f"names={modules!r}; "
        "files={name:Path(importlib.import_module('npa.cli.'+name).__file__) "
        "for name in names}; files['rendered_backend']=Path('/opt/npa-agent/backend.py'); "
        "print(json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() "
        "for name,path in files.items()}))"
    )
    _, stdout, _ = ssh.run_or_raise(
        "sudo /opt/npa-agent/venv/bin/python -c " + shlex.quote(script)
    )
    hashes = json.loads(stdout)
    _write_evidence(evidence, "deployed-source-hashes.json", json.dumps(hashes))
    source = Path(agent.__file__).parent
    for name in modules:
        expected = hashlib.sha256((source / f"{name}.py").read_bytes()).hexdigest()
        assert hashes[name] == expected, f"Deployed {name} source differs"
    return hashes


def test_fresh_metadata_agent_access_and_inventory(deployment):
    args, project, name, evidence = deployment
    _invoke(args, evidence, "deploy")
    status = _invoke(
        ["agent", "status", "--project", project, "--name", name, "--json"],
        evidence,
        "status",
    )
    assert status["health"] is True and status["basic_auth_enforced"] is True
    base = status["public_url"].rstrip("/")
    auth = agent._load_auth_secret(str(agent._auth_secret_path(project, name)))
    with httpx.Client(base_url=base, verify=False, timeout=None) as anonymous:
        for route in ("/", "/api/access?refresh=true", "/api/resources"):
            assert anonymous.get(route).status_code == 401
    with httpx.Client(base_url=base, auth=auth, verify=False, timeout=None) as client:
        manifest, _ = _request(client, "/api/deployment", evidence, "deployment")
        access, access_seconds = _request(
            client, "/api/access?refresh=true", evidence, "access"
        )
        resources, resources_seconds = _request(
            client, "/api/resources", evidence, "resources"
        )
    expected = agent._agent_record(project, name)["deployment"]
    assert_live_deployment(expected, manifest)
    identity = access["identity"]
    assert identity["credential_source"] == "instance_metadata"
    assert identity["credential_profile"] == "cursor-sa"
    assert identity["credential_config"] == "/root/.nebius/config.yaml"
    assert isinstance(resources, dict) and not resources.get("error")
    hashes = _deployed_source_hashes(project, name, evidence)
    _record_result(evidence, manifest, hashes, access_seconds, resources_seconds)


def _record_result(evidence, manifest, hashes, access_seconds, resources_seconds):
    _write_evidence(
        evidence,
        "result.json",
        json.dumps(
            {
                "source_commit": manifest["commit"],
                "source_tree": manifest["source_tree"],
                "access_http_status": 200,
                "access_seconds": access_seconds,
                "resources_http_status": 200,
                "resources_seconds": resources_seconds,
                "anonymous_statuses": [401, 401, 401],
                "deployed_source_sha256": hashes,
            },
            indent=2,
        )
        + "\n",
    )

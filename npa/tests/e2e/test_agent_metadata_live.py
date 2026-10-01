"""Verify staged metadata credentials on an isolated real CPU agent lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import ssl
import subprocess
import sys
import time

import httpx
import pytest
from typer.testing import CliRunner

from npa.cli import agent
from npa.cli.main import app
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


def _exact_option_value(args: list[str], flag: str) -> str:
    positions = [
        index
        for index, value in enumerate(args)
        if value == flag or value.startswith(flag + "=")
    ]
    assert len(positions) == 1 and args[positions[0]] == flag, (
        f"Live deploy requires exactly one separate {flag} value"
    )
    index = positions[0]
    assert index + 1 < len(args), f"Live deploy {flag} has no value"
    raw_value = str(args[index + 1])
    value = raw_value.strip()
    assert value == raw_value, f"Live deploy {flag} must not contain outer whitespace"
    assert value and not value.startswith("-"), f"Live deploy {flag} is invalid"
    return value


def _private_evidence_directory(raw_path: str) -> Path:
    evidence = Path(raw_path).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    assert not evidence.is_relative_to(checkout), (
        "Live evidence directory must be outside the checkout"
    )
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0, "Evidence must be owner-only"
    return evidence


def _invoke_deploy(
    args: list[str], evidence: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Run the real CLI while retaining the exact locally rendered backend digest."""
    expected_backend_digests: set[str] = set()
    install_services = agent.install_agent_services

    def capture_install(
        ssh,
        *,
        setup_script: str,
        stage_source,
        resuming: bool,
    ) -> bool:
        match = re.search(
            r"cat <<'PY' \| sudo tee /opt/npa-agent/backend\.py >/dev/null\n"
            r"(?P<body>.*?)\nPY\n",
            setup_script,
            flags=re.DOTALL,
        )
        assert match, "Rendered installer did not contain backend.py"
        expected_backend_digests.add(
            hashlib.sha256((match.group("body") + "\n").encode()).hexdigest()
        )
        return install_services(
            ssh,
            setup_script=setup_script,
            stage_source=stage_source,
            resuming=resuming,
        )

    monkeypatch.setattr(agent, "install_agent_services", capture_install)
    result = CliRunner().invoke(app, args)
    _write_evidence(evidence, "deploy.stdout", result.stdout)
    _write_evidence(evidence, "deploy.stderr", result.stderr)
    assert result.exit_code == 0, "deploy failed; inspect private evidence"
    assert len(expected_backend_digests) == 1, (
        "Deploy did not produce one stable rendered backend"
    )
    expected = next(iter(expected_backend_digests))
    _write_evidence(evidence, "expected-rendered-backend.sha256", expected + "\n")
    return expected


@pytest.fixture
def deployment():
    path = Path(os.environ["NPA_AGENT_METADATA_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "Live config must be owner-only"
    config = json.loads(path.read_text())
    args = config["deploy_args"]
    assert args[:2] == ["agent", "deploy"] and "--agent-only" in args
    project = _exact_option_value(args, "--project")
    name = _exact_option_value(args, "--name")
    assert not agent._agent_record(project, name), "Choose an unused agent name"
    evidence = _private_evidence_directory(str(config["evidence_dir"]))
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


def _deployed_ssh(project: str, name: str):
    record = agent._agent_record(project, name)
    return agent.SSHClient(
        config=agent.resolve_ssh_config(
            ssh_host=record["public_ip"],
            ssh_user=record.get("ssh_user", "ubuntu"),
            ssh_key=record["ssh_key_path"],
            project=None,
            name=None,
        ).ssh
    )


def _deployed_tls(project: str, name: str, evidence: Path) -> ssl.SSLContext:
    # SSH checks the provider-pinned host key before trusting this certificate.
    _, certificate, _ = _deployed_ssh(project, name).run_or_raise(
        "sudo cat /etc/nginx/ssl/npa-agent.crt"
    )
    _write_evidence(evidence, "server-certificate.pem", certificate)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cadata=certificate)
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    return context


def _deployed_source_hashes(
    project: str,
    name: str,
    evidence: Path,
    *,
    expected_backend_sha256: str,
) -> dict:
    modules = ["agent", "agent_access_runtime", "agent_resources", "agent_env_files"]
    script = (
        "import hashlib, importlib, json; from pathlib import Path; "
        f"names={modules!r}; "
        "files={name:Path(importlib.import_module('npa.cli.'+name).__file__) "
        "for name in names}; files['rendered_backend']=Path('/opt/npa-agent/backend.py'); "
        "print(json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() "
        "for name,path in files.items()}))"
    )
    _, stdout, _ = _deployed_ssh(project, name).run_or_raise(
        "sudo /opt/npa-agent/venv/bin/python -c " + shlex.quote(script)
    )
    hashes = json.loads(stdout)
    _write_evidence(evidence, "deployed-source-hashes.json", json.dumps(hashes))
    source = Path(agent.__file__).parent
    for name in modules:
        expected = hashlib.sha256((source / f"{name}.py").read_bytes()).hexdigest()
        assert hashes[name] == expected, f"Deployed {name} source differs"
    assert re.fullmatch(r"[0-9a-f]{64}", expected_backend_sha256)
    assert hashes["rendered_backend"] == expected_backend_sha256, (
        "Deployed rendered backend differs from the exact local installer"
    )
    return hashes


def _reject_untrusted_certificate(base: str) -> None:
    # No trusted roots: the same public endpoint must fail before HTTP auth.
    untrusted = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with httpx.Client(base_url=base, verify=untrusted, timeout=None) as client:
        with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
            client.get("/api/access?refresh=true")


def test_fresh_metadata_agent_access_and_inventory(deployment, monkeypatch):
    args, project, name, evidence = deployment
    expected_backend_sha256 = _invoke_deploy(args, evidence, monkeypatch)
    status = _invoke(
        ["agent", "status", "--project", project, "--name", name, "--json"],
        evidence,
        "status",
    )
    assert status["health"] is True and status["basic_auth_enforced"] is True
    base = status["public_url"].rstrip("/")
    assert base.startswith("https://"), "Live proof requires public HTTPS"
    tls = _deployed_tls(project, name, evidence)
    _reject_untrusted_certificate(base)
    auth = agent._load_auth_secret(str(agent._auth_secret_path(project, name)))
    with httpx.Client(base_url=base, verify=tls, timeout=None) as anonymous:
        for route in ("/", "/api/access?refresh=true", "/api/resources"):
            assert anonymous.get(route).status_code == 401
    with httpx.Client(base_url=base, auth=auth, verify=tls, timeout=None) as client:
        manifest, _ = _request(client, "/api/deployment", evidence, "deployment")
        access, access_seconds = _request(
            client, "/api/access?refresh=true", evidence, "access"
        )
        resources, resources_seconds = _request(
            client, "/api/resources", evidence, "resources"
        )
    expected = agent.build_deployment_manifest(
        project_alias=project,
        name=name,
        bootstrap_timestamp=manifest["bootstrap_timestamp"],
    )
    assert_live_deployment(expected, manifest)
    assert access.get("ok") is True
    assert access.get("status") in {"available", "partial"}
    assert (access.get("capabilities") or {}).get("project_discovery", {}).get(
        "status"
    ) == "available"
    identity = access["identity"]
    assert identity["credential_source"] == "instance_metadata"
    assert identity["credential_profile"] == "cursor-sa"
    assert identity["credential_config"] == "/root/.nebius/config.yaml"
    assert resources.get("ok") is True and not resources.get("error")
    categories = {
        str(category.get("id") or ""): category
        for category in resources.get("categories") or []
        if isinstance(category, dict)
    }
    for category_id in ("project", "tenant"):
        category = categories.get(category_id) or {}
        assert category.get("status") == "discovered"
        assert int(category.get("discovered_count") or 0) > 0
    hashes = _deployed_source_hashes(
        project,
        name,
        evidence,
        expected_backend_sha256=expected_backend_sha256,
    )
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
                "tls_hostname_verified": True,
                "tls_untrusted_certificate_rejected": True,
                "tls_trust_source": "provider-pinned SSH",
                "server_certificate_sha256": hashlib.sha256(
                    ssl.PEM_cert_to_DER_cert(
                        (evidence / "server-certificate.pem").read_text()
                    )
                ).hexdigest(),
                "deployed_source_sha256": hashes,
            },
            indent=2,
        )
        + "\n",
    )

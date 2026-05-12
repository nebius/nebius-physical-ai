from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys
import time
from typing import Iterator

import pytest

from npa.cli.cosmos import DEFAULT_MODEL
from npa.clients.config import (
    remove_workbench_config,
    update_workbench_serverless_endpoint,
    write_config,
)
from npa.clients.serverless import (
    AuthError,
    EndpointInfo,
    EndpointSpec,
    EndpointStatus,
    NotEnoughResourcesError,
    ServerlessClient,
)
from npa.deploy.images import container_image_for_tool

from ._serverless_fallback import FallbackChain


pytestmark = pytest.mark.e2e_serverless


def _skip_if_not_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT"):
        pytest.skip("NPA_E2E_SERVERLESS_PROJECT not set")


def _image() -> str:
    return os.environ.get("NPA_E2E_SERVERLESS_IMAGE") or container_image_for_tool("cosmos")


def _platform() -> str:
    return os.environ.get("NPA_E2E_SERVERLESS_PLATFORM", "gpu-h200-sxm")


def _preset() -> str:
    return os.environ.get("NPA_E2E_SERVERLESS_PRESET", "1gpu-16vcpu-200gb")


def _subnet_id(project_id: str) -> str:
    override = os.environ.get("NPA_E2E_SERVERLESS_SUBNET_ID", "")
    if override:
        return override
    result = subprocess.run(
        ["nebius", "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to list subnets for {project_id}: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"No subnets found for {project_id}")
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(((item.get("status") or {}).get("state") or "")).upper() == "READY":
            subnet_id = ((item.get("metadata") or {}).get("id") or "")
            if subnet_id:
                return str(subnet_id)
    first = items[0]
    subnet_id = ((first.get("metadata") or {}).get("id") or "") if isinstance(first, dict) else ""
    if not subnet_id:
        raise RuntimeError(f"Could not parse subnet ID for {project_id}")
    return str(subnet_id)


def _run_npa(args: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _endpoint_spec(project_id: str, name: str) -> EndpointSpec:
    return EndpointSpec(
        name=name,
        project_id=project_id,
        image=_image(),
        platform=_platform(),
        preset=_preset(),
        container_ports=[8080],
        env={
            "COSMOS_MODEL_ID": DEFAULT_MODEL,
            "COSMOS_SERVER_PORT": "8080",
        },
        container_command="/bin/bash",
        subnet_id=_subnet_id(project_id),
        args=(
            "-lc "
            "'cd /opt/cosmos && "
            "exec /opt/cosmos/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8080'"
        ),
    )


def _create_with_fallback(client: ServerlessClient, name: str) -> tuple[str, EndpointInfo]:
    chain = FallbackChain.instance()
    last_error: BaseException | None = None
    while True:
        project_id = chain.current_project()
        if project_id is None:
            pytest.skip(f"All projects in fallback chain are NER-exhausted: {last_error}")
        try:
            return project_id, client.create_endpoint(_endpoint_spec(project_id, name))
        except NotEnoughResourcesError as exc:
            last_error = exc
            chain.mark_ner(project_id)
        except AuthError:
            raise


@pytest.fixture(scope="module")
def cosmos_endpoint() -> Iterator[dict[str, str]]:
    _skip_if_not_e2e()
    timestamp = int(time.time())
    name = f"npa-e2e-cosmos-{timestamp}"
    client = ServerlessClient(timeout=900, poll_interval=15)
    state = {
        "name": name,
        "project_id": "",
        "project_key": "",
        "endpoint_id": "",
        "endpoint_name": name,
        "url": "",
        "created": "false",
        "alias_created": "false",
    }

    try:
        project_id, info = _create_with_fallback(client, name)
        chain = FallbackChain.instance()
        project_key = chain.project_key(project_id)
        state.update(
            {
                "project_id": project_id,
                "project_key": project_key,
                "endpoint_id": info.id,
                "endpoint_name": info.name or name,
                "url": info.url,
                "created": "true",
            }
        )

        running = client.wait_for_running(project_id, info.id or name, timeout=900, poll_interval=15)
        state["url"] = running.url or info.url
        update_workbench_serverless_endpoint(
            project_key,
            name,
            endpoint_id=running.id or info.id,
            endpoint_name=running.name or info.name or name,
            project_id=project_id,
            url=running.url or info.url,
            image=_image(),
            platform=_platform(),
            preset=_preset(),
            container_port=8080,
            auth="none",
        )
        write_config(
            {
                "projects": {
                    project_key: {
                        "workbenches": {
                            name: {
                                "model": DEFAULT_MODEL,
                                "backend": "basic",
                            },
                        },
                    },
                },
            }
        )
        state["alias_created"] = "true"
        yield state
    finally:
        if state["created"] == "true":
            try:
                client.delete_endpoint(
                    state["project_id"],
                    state["endpoint_id"] or state["endpoint_name"],
                )
                print(
                    f"E2E TEARDOWN OK: deleted endpoint {state['endpoint_name']} from project {state['project_id']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"!!! E2E TEARDOWN FAILED for {state['endpoint_name']} in project {state['project_id']}: {exc}",
                    flush=True,
                )
                print(
                    f"!!! ORPHANED ENDPOINT in project {state['project_id']}: {state['endpoint_name']} ({state['endpoint_id']})",
                    flush=True,
                )
        if state["alias_created"] == "true":
            remove_workbench_config(state["project_key"], state["name"])


def test_e2e_client_endpoint_reaches_running(cosmos_endpoint: dict[str, str]) -> None:
    assert cosmos_endpoint["endpoint_id"]
    assert cosmos_endpoint["project_id"]
    assert cosmos_endpoint["url"]


def test_e2e_client_get_endpoint(cosmos_endpoint: dict[str, str]) -> None:
    info = ServerlessClient().get_endpoint(
        cosmos_endpoint["project_id"],
        cosmos_endpoint["endpoint_id"],
    )

    assert info.status is EndpointStatus.RUNNING
    assert info.name == cosmos_endpoint["endpoint_name"]


def test_e2e_client_list_contains_endpoint(cosmos_endpoint: dict[str, str]) -> None:
    endpoints = ServerlessClient().list_endpoints(cosmos_endpoint["project_id"])

    assert any(endpoint.id == cosmos_endpoint["endpoint_id"] for endpoint in endpoints)


def test_e2e_client_logs(cosmos_endpoint: dict[str, str]) -> None:
    logs = ServerlessClient().get_endpoint_logs(
        cosmos_endpoint["project_id"],
        cosmos_endpoint["endpoint_id"],
        tail=20,
    )

    assert isinstance(logs, str)


def test_e2e_cli_status_json(cosmos_endpoint: dict[str, str]) -> None:
    result = _run_npa(
        [
            "workbench",
            "cosmos",
            "-p",
            cosmos_endpoint["project_key"],
            "-n",
            cosmos_endpoint["name"],
            "status",
            "--output",
            "json",
        ],
        timeout=120,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert '"runtime": "serverless"' in result.stdout
    assert '"serverless_status": "running"' in result.stdout


def test_e2e_cli_serve_prewarm(cosmos_endpoint: dict[str, str]) -> None:
    result = _run_npa(
        [
            "workbench",
            "cosmos",
            "-p",
            cosmos_endpoint["project_key"],
            "-n",
            cosmos_endpoint["name"],
            "serve",
        ],
        timeout=180,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "status: prewarmed" in result.stdout


def test_e2e_cli_infer_prompt(cosmos_endpoint: dict[str, str]) -> None:
    result = _run_npa(
        [
            "workbench",
            "cosmos",
            "-p",
            cosmos_endpoint["project_key"],
            "-n",
            cosmos_endpoint["name"],
            "infer",
            "--prompt",
            "A robot arm stacks colored cubes",
            "--poll-interval",
            "5",
            "--timeout",
            "600",
        ],
        timeout=720,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "job_id:" in result.stdout
    assert "Generation complete" in result.stdout


def test_e2e_cli_deploy_dry_run_uses_selected_project(cosmos_endpoint: dict[str, str]) -> None:
    result = _run_npa(
        [
            "workbench",
            "cosmos",
            "-p",
            cosmos_endpoint["project_key"],
            "-n",
            f"{cosmos_endpoint['name']}-dry-run",
            "deploy",
            "--runtime",
            "serverless",
            "--image",
            _image(),
            "--platform",
            _platform(),
            "--preset",
            _preset(),
            "--subnet-id",
            _subnet_id(cosmos_endpoint["project_id"]),
            "--dry-run",
            "--output",
            "json",
        ],
        timeout=120,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert '"status": "dry_run"' in result.stdout
    assert cosmos_endpoint["project_id"] in result.stdout


def test_e2e_force_ner_detection() -> None:
    _skip_if_not_e2e()
    if os.environ.get("NPA_E2E_FORCE_NER") != "1":
        pytest.skip("NPA_E2E_FORCE_NER not set")
    project_id = FallbackChain.instance().current_project()
    if project_id is None:
        pytest.skip("All projects are NER-exhausted")
    spec = _endpoint_spec(project_id, f"npa-e2e-force-ner-{int(time.time())}")
    spec = EndpointSpec(
        **{
            **spec.__dict__,
            "platform": "gpu-h200-sxm",
            "preset": "999gpu-999vcpu-999tb",
        }
    )
    with pytest.raises(NotEnoughResourcesError):
        ServerlessClient(timeout=120).create_endpoint(spec)

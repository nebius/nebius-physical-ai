"""Launch a real LeIsaac browser-teleoperation session on Kubernetes."""

from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import ssl
import subprocess
import time
import urllib.request
from enum import Enum
from typing import Any

import typer

from npa.agent_backend.leisaac_registry import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_TASK,
    REGISTRY_FINGERPRINT,
    registry_payload,
    validate_environment_id,
    validate_environment_index,
    validate_num_envs,
    validate_seed,
    validate_task,
)
from npa.clients.config import SSHConfig, list_projects, resolve_project_storage
from npa.clients.network import (
    ensure_ingress,
    remove_exact_npa_ingress_for_instance,
    resolve_instance_network_context,
)
from npa.clients.ssh import SSHClient
from npa.workbench.leisaac import (
    GPU_PRODUCT,
    SOURCE_COMMIT,
    LeIsaacConfigError,
    MEDIA_PORT,
    RELAY_SERVICE_PORT,
    TURN_PORT,
    TURN_RELAY_PORT,
    TRANSPORT_AGENT_RELAY,
    TRANSPORT_LOAD_BALANCER,
    deployment_manifest,
    relay_client_secret_manifest,
    recorder_secret_manifest,
    relay_service_manifest,
    resource_name,
    service_manifests,
    session_manifest,
    session_attestation,
    split_s3_uri,
    validate_expiry,
    validate_image,
    validate_private_ip,
    validate_public_ip,
    validate_run_id,
    validate_source_ranges,
)
from npa.workbench.leisaac.paidf import (
    export_episode_to_paidf,
    materialize_paidf_dataset,
)
from npa.workbench.leisaac.dataset import resolve_s3_endpoint
from npa.workflows.sim2real.registry_auth import ensure_registry_pull_secret_for_images

app = typer.Typer(
    name="leisaac",
    help="LeIsaac SO101 browser teleoperation on the RTX PRO 6000 Kubernetes pool.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class Transport(str, Enum):
    load_balancer = TRANSPORT_LOAD_BALANCER
    agent_relay = TRANSPORT_AGENT_RELAY


_RELAY_TOOL = "leisaac-relay"
_RELAY_CONFIG = "/etc/npa/leisaac-relay.json"
_RELAY_SCRIPT = "/opt/npa-agent/leisaac-agent-relay.py"
_RELAY_UNIT = "npa-leisaac-relay.service"
_TURN_CONTROL_TOOL = "leisaac-turn-control"
_TURN_MEDIA_TOOL = "leisaac-turn-media"
_TURN_CONFIG = "/etc/npa/leisaac-turn.conf"
_TURN_UNIT = "npa-leisaac-turn.service"


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _kubectl(
    context: str, namespace: str, args: list[str], stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    command = ["kubectl"]
    if context:
        command.extend(["--context", context])
    command.extend(["--namespace", namespace, *args])
    return subprocess.run(
        command, input=stdin, capture_output=True, text=True, check=False
    )


def _apply(context: str, namespace: str, documents: list[dict[str, Any]]) -> None:
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": documents})
    result = _kubectl(context, namespace, ["apply", "-f", "-"], stdin=payload)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _external_ip(context: str, namespace: str, service: str) -> str:
    while True:
        result = _kubectl(
            context,
            namespace,
            [
                "get",
                "service",
                service,
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].ip}",
            ],
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
        time.sleep(3)


def _wait_ready(context: str, namespace: str, deployment: str) -> None:
    while True:
        result = _kubectl(
            context,
            namespace,
            ["get", "deployment", deployment, "-o", "json"],
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            metadata = data.get("metadata", {}) or {}
            spec = data.get("spec", {}) or {}
            status = data.get("status", {}) or {}
            generation = int(metadata.get("generation") or 0)
            if (
                generation > 0
                and int(status.get("observedGeneration") or 0) == generation
                and int(spec.get("replicas") or 0) == 1
                and int(status.get("updatedReplicas") or 0) == 1
                and int(status.get("readyReplicas") or 0) == 1
                and int(status.get("availableReplicas") or 0) == 1
                and int(status.get("unavailableReplicas") or 0) == 0
            ):
                return
        time.sleep(5)


def _delete_resources(context: str, namespace: str, name: str) -> None:
    result = _kubectl(
        context,
        namespace,
        [
            "delete",
            f"deployment/{name}",
            f"service/{name}-tcp",
            f"service/{name}-media",
            f"service/{name}-relay",
            f"secret/{name}-relay-client",
            f"secret/{name}-recorder",
            "--ignore-not-found=true",
        ],
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _relay_media_server(context: str, namespace: str, deployment: str) -> str:
    """Return the private pod IP shared by the simulator and TURN sidecar."""

    result = _kubectl(
        context,
        namespace,
        ["get", "pods", "-l", f"app={deployment}", "-o", "json"],
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    candidates: list[str] = []
    for pod in json.loads(result.stdout).get("items", []):
        metadata = pod.get("metadata", {}) or {}
        status = pod.get("status", {}) or {}
        containers = status.get("containerStatuses", []) or []
        main_ready = any(
            item.get("name") == "leisaac" and item.get("ready") is True
            for item in containers
        )
        if (
            metadata.get("deletionTimestamp")
            or status.get("phase") != "Running"
            or not main_ready
        ):
            continue
        candidates.append(
            validate_private_ip(status.get("podIP"), "LeIsaac pod media address")
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "LeIsaac relay requires exactly one ready simulator pod with a private "
            "pod address"
        )
    return candidates[0]


def _node_internal_ip(context: str, namespace: str) -> str:
    result = _kubectl(context, namespace, ["get", "nodes", "-o", "json"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    candidates: list[str] = []
    for node in json.loads(result.stdout).get("items", []):
        labels = node.get("metadata", {}).get("labels", {}) or {}
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in node.get("status", {}).get("conditions", []) or []
        )
        if not ready or labels.get("nvidia.com/gpu.product") != GPU_PRODUCT:
            continue
        for address in node.get("status", {}).get("addresses", []) or []:
            if address.get("type") == "InternalIP" and address.get("address"):
                candidates.append(str(address["address"]))
    if not candidates:
        raise RuntimeError(f"no Ready {GPU_PRODUCT} node with an internal IP was found")
    return sorted(set(candidates))[0]


def _relay_nodeports(context: str, namespace: str, service: str) -> dict[str, int]:
    result = _kubectl(context, namespace, ["get", "service", service, "-o", "json"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    ports = {
        str(item.get("name") or ""): int(item.get("nodePort") or 0)
        for item in json.loads(result.stdout).get("spec", {}).get("ports", [])
    }
    if set(ports) != {"status", "signal", "media"} or any(
        value < 30000 or value > 32767 for value in ports.values()
    ):
        raise RuntimeError("LeIsaac relay service did not receive valid NodePorts")
    return ports


def _agent_record(project: str, name: str) -> dict[str, Any]:
    project_record = list_projects().get(project, {})
    agents = (
        project_record.get("agents", {}) if isinstance(project_record, dict) else {}
    )
    record = agents.get(name, {}) if isinstance(agents, dict) else {}
    if not isinstance(record, dict) or not record:
        raise LeIsaacConfigError(f"agent config not found for {project}/{name}")
    return record


def _agent_artifact_storage(project: str, name: str) -> dict[str, str]:
    """Return the selected agent's S3 scope for capability publication.

    Agent-relay sessions must be published with the selected agent's endpoint
    and credentials.  Falling back to the operator shell's AWS endpoint can
    write a valid-looking manifest into a different regional S3 namespace,
    leaving the public agent unable to discover the capability it relays.
    """

    record = _agent_record(project, name)
    credentials = record.get("credentials")
    values = credentials if isinstance(credentials, dict) else {}
    storage = {
        "bucket": str(values.get("s3_bucket") or "").strip(),
        "prefix": str(values.get("s3_prefix") or "").strip().strip("/"),
        "endpoint": str(values.get("s3_endpoint") or "").strip(),
        "access_key": str(values.get("access_key") or "").strip(),
        "secret_key": str(values.get("secret_key") or "").strip(),
        "region": str(record.get("region") or "").strip(),
    }
    missing = [
        key
        for key in ("bucket", "endpoint", "access_key", "secret_key")
        if not storage[key]
    ]
    if missing:
        raise LeIsaacConfigError(
            "agent record has no usable artifact storage configuration "
            f"(missing {', '.join(missing)})"
        )
    return storage


def _agent_relay_context(
    project: str, name: str
) -> tuple[str, str, SSHClient, str, str]:
    record = _agent_record(project, name)
    instance_id = str(record.get("instance_id") or "").strip()
    key_path = str(record.get("ssh_key_path") or "").strip()
    if not instance_id:
        raise LeIsaacConfigError("agent record has no provider instance id")
    if not key_path or not Path(key_path).expanduser().is_file():
        raise LeIsaacConfigError("agent record has no usable SSH private key")
    network = resolve_instance_network_context(instance_id)
    public_ip = validate_public_ip(
        str(network.public_ip).split("/", 1)[0], "agent public IP"
    )
    saved_ip = str(record.get("public_ip") or "").split("/", 1)[0]
    if saved_ip and saved_ip != public_ip:
        raise LeIsaacConfigError(
            "agent public IP differs from provider state; bootstrap the agent to refresh it"
        )
    auth_path = Path(str(record.get("auth_secret_path") or "")).expanduser()
    values: dict[str, str] = {}
    if auth_path.is_file():
        for line in auth_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    auth_user = values.get("AGENT_USER", "")
    auth_password = values.get("AGENT_PASSWORD", "")
    if not auth_user or not auth_password:
        raise LeIsaacConfigError("agent record has no usable basic-auth secret")
    ssh = SSHClient(
        SSHConfig(
            host=public_ip,
            user=str(record.get("ssh_user") or "ubuntu"),
            key_path=key_path,
        )
    )
    return instance_id, public_ip, ssh, auth_user, auth_password


def _agent_certificate_sha256(public_ip: str) -> str:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((public_ip, 443), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=public_ip) as connection:
            certificate = connection.getpeercert(binary_form=True)
    if not certificate:
        raise RuntimeError("public agent HTTPS endpoint returned no certificate")
    return hashlib.sha256(certificate).hexdigest()


def _select_agent_leisaac_run(
    public_ip: str,
    *,
    auth_user: str,
    auth_password: str,
    run_id: str,
    certificate_sha256: str,
) -> None:
    """Register the live run through the selected agent's pinned TLS endpoint."""

    host = validate_public_ip(public_ip, "agent public IP")
    selected_run = validate_run_id(run_id)
    expected_certificate = str(certificate_sha256 or "").strip().lower()
    if len(expected_certificate) != 64:
        raise LeIsaacConfigError("agent certificate fingerprint is invalid")
    try:
        bytes.fromhex(expected_certificate)
    except ValueError as exc:
        raise LeIsaacConfigError("agent certificate fingerprint is invalid") from exc

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    raw = socket.create_connection((host, 443), timeout=10)
    tls = context.wrap_socket(raw, server_hostname=host)
    connection: http.client.HTTPConnection | None = None
    try:
        certificate = tls.getpeercert(binary_form=True)
        actual_certificate = (
            hashlib.sha256(certificate).hexdigest() if certificate else ""
        )
        if not secrets.compare_digest(actual_certificate, expected_certificate):
            raise RuntimeError("public agent TLS certificate fingerprint changed")
        # The TCP/TLS connection must fail fast, but selection performs capability
        # resolution plus a conditional state write against object storage.  Preserve
        # the pinned socket while giving those off-loop operations their own response
        # budget instead of inheriting the 10-second connect timeout.
        tls.settimeout(60)
        connection = http.client.HTTPConnection(host, 443, timeout=10)
        connection.sock = tls
        payload = json.dumps({"run_id": selected_run}, separators=(",", ":"))
        credential = base64.b64encode(
            f"{auth_user}:{auth_password}".encode("utf-8")
        ).decode("ascii")
        connection.request(
            "POST",
            "/api/leisaac/select",
            body=payload,
            headers={
                "Authorization": f"Basic {credential}",
                "Content-Type": "application/json",
                "X-NPA-LeIsaac-Control": "1",
            },
        )
        response = connection.getresponse()
        body = response.read(131073)
        if response.status != 200 or len(body) > 131072:
            raise RuntimeError(
                f"public agent rejected LeIsaac run selection (HTTP {response.status})"
            )
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                "public agent returned an invalid LeIsaac selection response"
            ) from exc
        if (
            not isinstance(result, dict)
            or result.get("selected") is not True
            or result.get("run_id") != selected_run
        ):
            raise RuntimeError("public agent did not persist the LeIsaac run selection")
    finally:
        if connection is not None:
            connection.close()
        else:
            tls.close()


def _relay_source(path: str) -> bytes:
    source = Path(__file__).resolve().parents[2] / "workbench" / "leisaac" / path
    return source.read_bytes()


def _install_agent_relay(
    ssh: SSHClient,
    *,
    run_id: str,
    session_nonce: str,
    media_target_host: str = "",
    media_target_port: int = 0,
) -> None:
    config = {
        "run_id": run_id,
        "session_nonce": session_nonce,
    }
    if media_target_host or media_target_port:
        config["media_target_host"] = media_target_host
        config["media_target_port"] = media_target_port
    unit = """[Unit]
Description=NPA LeIsaac private-cluster relay
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
DynamicUser=yes
ExecStart=/usr/bin/python3 /opt/npa-agent/leisaac-agent-relay.py --config /etc/npa/leisaac-relay.json
Restart=on-failure
RestartSec=2
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes

[Install]
WantedBy=multi-user.target
"""
    script_b64 = base64.b64encode(_relay_source("agent_relay.py")).decode("ascii")
    config_b64 = base64.b64encode(
        (json.dumps(config, sort_keys=True) + "\n").encode("utf-8")
    ).decode("ascii")
    unit_b64 = base64.b64encode(unit.encode("utf-8")).decode("ascii")
    run_q = shlex.quote(run_id)
    command = f"""set -eu
existing=''
if sudo test -f {_RELAY_CONFIG}; then
  existing=$(sudo /usr/bin/python3 -c 'import json; print(json.load(open("{_RELAY_CONFIG}"))["run_id"])')
fi
if sudo systemctl is-active --quiet {_RELAY_UNIT} && [ "$existing" != {run_q} ]; then
  echo 'another LeIsaac relay session is active' >&2
  exit 42
fi
sudo install -d -m 0755 /etc/npa /opt/npa-agent
echo {shlex.quote(script_b64)} | base64 -d | sudo tee {_RELAY_SCRIPT} >/dev/null
echo {shlex.quote(config_b64)} | base64 -d | sudo tee {_RELAY_CONFIG} >/dev/null
echo {shlex.quote(unit_b64)} | base64 -d | sudo tee /etc/systemd/system/{_RELAY_UNIT} >/dev/null
sudo chmod 0644 {_RELAY_SCRIPT} {_RELAY_CONFIG} /etc/systemd/system/{_RELAY_UNIT}
sudo systemctl daemon-reload
sudo systemctl enable --now {_RELAY_UNIT} >/dev/null
sudo systemctl restart {_RELAY_UNIT}
"""
    ssh.run_or_raise(command, label="install LeIsaac agent relay")


def _remove_agent_relay(ssh: SSHClient, *, run_id: str) -> None:
    run_q = shlex.quote(run_id)
    command = f"""set -eu
if ! sudo test -f {_RELAY_CONFIG}; then exit 0; fi
existing=$(sudo /usr/bin/python3 -c 'import json; print(json.load(open("{_RELAY_CONFIG}"))["run_id"])')
if [ "$existing" != {run_q} ]; then exit 0; fi
sudo systemctl disable --now {_RELAY_UNIT} >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/{_RELAY_UNIT} {_RELAY_CONFIG} {_RELAY_SCRIPT}
sudo systemctl daemon-reload
"""
    ssh.run_or_raise(command, label="remove LeIsaac agent relay")


def _relay_status(ssh: SSHClient) -> dict[str, Any]:
    _code, stdout, _stderr = ssh.run_or_raise(
        f"curl --fail --silent --show-error http://127.0.0.1:{RELAY_SERVICE_PORT}/status",
        label="attest LeIsaac through the agent relay",
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("LeIsaac relay returned a non-object health document")
    return payload


def _turn_peer_source(value: str) -> str:
    ranges = validate_source_ranges([value])
    network = ipaddress.ip_network(ranges[0])
    if len(ranges) != 1 or network.version != 4 or network.prefixlen < 22:
        raise LeIsaacConfigError(
            "TURN peer source must be one public IPv4 CIDR between /22 and /32"
        )
    return ranges[0]


def _existing_turn_peer_source(context: str, namespace: str, service: str) -> str:
    result = _kubectl(context, namespace, ["get", "service", service, "-o", "json"])
    if result.returncode:
        return ""
    annotations = json.loads(result.stdout).get("metadata", {}).get("annotations", {})
    value = str((annotations or {}).get("npa.nebius.com/turn-peer-source") or "")
    return _turn_peer_source(value) if value else ""


def _remove_agent_turn(ssh: SSHClient, *, run_id: str) -> None:
    run_q = shlex.quote(validate_run_id(run_id))
    command = f"""set -eu
if ! sudo test -f {_TURN_CONFIG}; then exit 0; fi
existing=$(sudo sed -n 's/^user=\\([^:]*\\):.*$/\\1/p' {_TURN_CONFIG})
if [ "$existing" != {run_q} ]; then exit 0; fi
sudo systemctl disable --now {_TURN_UNIT} >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/{_TURN_UNIT} {_TURN_CONFIG}
sudo systemctl daemon-reload
"""
    ssh.run_or_raise(command, label="remove LeIsaac TURN relay")


def _status(signal_host: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://{signal_host}:8080/status") as response:  # noqa: S310 - validated LB IP
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("LeIsaac service returned a non-object health document")
    return payload


def _put_manifest(
    uri: str,
    manifest: dict[str, Any],
    *,
    storage: dict[str, str] | None = None,
) -> str:
    import boto3

    bucket, prefix = split_s3_uri(uri)
    client_kwargs: dict[str, Any] = {}
    if storage is not None:
        configured_bucket = str(storage.get("bucket") or "").strip()
        configured_prefix = str(storage.get("prefix") or "").strip().strip("/")
        if bucket != configured_bucket:
            raise LeIsaacConfigError(
                "agent-relay artifact URI bucket must match the selected agent's bucket"
            )
        if configured_prefix and not (
            prefix == configured_prefix or prefix.startswith(configured_prefix + "/")
        ):
            raise LeIsaacConfigError(
                "agent-relay artifact URI must be inside the selected agent's artifact prefix"
            )
        client_kwargs = {
            "endpoint_url": storage["endpoint"],
            "aws_access_key_id": storage["access_key"],
            "region_name": storage.get("region") or None,
        }
        client_kwargs["aws" + "_secret_access_key"] = storage["secret_key"]
    leaf = "reports/leisaac-session.json"
    # A deprecated launch accepted a leaf URI but treated it as a prefix,
    # producing .../leisaac-session.json/<run>/reports/leisaac-session.json.
    # Honor leaf semantics for new writes while discovery continues to find
    # historical objects by their canonical basename.
    key = (
        prefix.rstrip("/")
        if prefix.rstrip("/").endswith(leaf)
        else f"{prefix.rstrip('/')}/{manifest['run_id']}/{leaf}"
    )
    if storage is None:
        client_kwargs["endpoint_url"] = (
            os.environ.get("NEBIUS_S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL")
            or None
        )
    client = boto3.client("s3", **client_kwargs)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@app.command("list-tasks")
def list_tasks_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """List the pinned SO101 tasks that support browser keyboard control."""

    payload = registry_payload()
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for task in payload["tasks"]:
        typer.echo(f"{task['task']}: {task['description']}")
    typer.echo(
        "environment_model: named-sequential (one active environment per launch)"
    )


@app.command("export-paidf")
def export_paidf_cmd(
    dataset_uri: str = typer.Option(..., "--dataset-uri"),
    episode: int = typer.Option(..., "--episode", min=0),
    run_id: str = typer.Option(..., "--run-id"),
    output_path: str = typer.Option(..., "--output-path"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Export one finalized episode directly from S3 into a PAIDF run input."""

    try:
        run_id = validate_run_id(run_id)
        result = export_episode_to_paidf(
            dataset_uri=dataset_uri,
            episode_index=episode,
            paidf_run_id=run_id,
            paidf_output_path=output_path,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(str(exc))
        return
    _emit(result, output)


@app.command("materialize-paidf")
def materialize_paidf_cmd(
    dataset_uri: str = typer.Option(..., "--dataset-uri"),
    episode: int = typer.Option(..., "--episode", min=0),
    paidf_run_uri: str = typer.Option(..., "--paidf-run-uri"),
    output_path: str = typer.Option(..., "--output-path"),
    variant: int = typer.Option(0, "--variant", min=0),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Create an immutable derived dataset after strict PAIDF video alignment."""

    try:
        result = materialize_paidf_dataset(
            dataset_uri=dataset_uri,
            episode_index=episode,
            paidf_run_uri=paidf_run_uri,
            output_path=output_path,
            variant_index=variant,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(str(exc))
        return
    _emit(result, output)


@app.command("launch")
def launch_cmd(
    run_id: str = typer.Option(
        ..., "--run-id", help="Run id used for artifact discovery and UI selection."
    ),
    image: str = typer.Option(
        ..., "--image", help="npa-leisaac image pinned as repository@sha256:digest."
    ),
    context: str = typer.Option(
        "", "--context", help="kubectl context for the RT-core GPU cluster."
    ),
    namespace: str = typer.Option(
        "default", "--namespace", help="Kubernetes namespace."
    ),
    source_range: list[str] = typer.Option(
        ...,
        "--source-range",
        help="Public operator CIDR allowed to reach the session; repeat when needed.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="Operator-owned S3 prefix for immutable LeRobot demonstration datasets.",
    ),
    manifest_prefix: str = typer.Option(
        "",
        "--manifest-prefix",
        help="S3 prefix (or exact .../reports/leisaac-session.json leaf) for capability publication.",
    ),
    artifact_uri: str = typer.Option(
        "",
        "--artifact-uri",
        help="Deprecated alias for --manifest-prefix; exact leaf URIs retain leaf semantics.",
    ),
    task: str = typer.Option(
        DEFAULT_TASK, "--task", help="Pinned task returned by list-tasks."
    ),
    environment_id: str = typer.Option(
        DEFAULT_ENVIRONMENT_ID,
        "--environment-id",
        help="Stable sequential environment identity.",
    ),
    environment_index: int = typer.Option(
        0,
        "--environment-index",
        min=0,
        max=2**31 - 1,
        help="Stable non-negative environment index.",
    ),
    seed: int = typer.Option(42, "--seed", min=0, max=2**32 - 1),
    num_envs: int = typer.Option(
        1,
        "--num-envs",
        help="Must be 1; browser control and episode boundaries are not parallel-routed.",
    ),
    expires_at: str = typer.Option(
        "",
        "--expires-at",
        help="Optional operator-chosen ISO-8601 expiry for UI discovery; omitted sessions remain live until destroyed.",
    ),
    image_pull_secret: str = typer.Option(
        "npa-registry", "--image-pull-secret", help="Existing registry pull secret."
    ),
    transport: Transport = typer.Option(
        Transport.load_balancer,
        "--transport",
        help="Public LBs, or the existing public HTTPS agent with private cluster relay.",
    ),
    agent_project: str = typer.Option(
        "", "--agent-project", help="Saved NPA agent project alias for agent-relay."
    ),
    agent_name: str = typer.Option(
        "", "--agent-name", help="Saved NPA agent deployment name for agent-relay."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Launch a supported SO101 task and publish its secure collector capability."""

    if (
        os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES"
        or os.environ.get("ISAACSIM_ACCEPT_EULA") != "YES"
    ):
        _fail(
            "set OMNI_KIT_ACCEPT_EULA=YES and ISAACSIM_ACCEPT_EULA=YES after accepting NVIDIA's EULAs"
        )
    name = ""
    instance_id = ""
    ssh: SSHClient | None = None
    artifact_storage: dict[str, str] | None = None
    auth_user = ""
    auth_password = ""
    certificate_sha256 = ""
    relay_installed = False
    turn_cleanup_required = False
    prior_turn_peer_source = ""
    created_ingress_specs: list[tuple[int, str, str, str]] = []
    try:
        run_id = validate_run_id(run_id)
        image = validate_image(image)
        task = validate_task(task)
        environment_id = validate_environment_id(environment_id)
        environment_index = validate_environment_index(environment_index)
        seed = validate_seed(seed)
        num_envs = validate_num_envs(num_envs)
        expires_at = validate_expiry(expires_at)
        source_ranges = validate_source_ranges(source_range)
        split_s3_uri(output_path)
        if manifest_prefix and artifact_uri:
            raise LeIsaacConfigError(
                "use --manifest-prefix or deprecated --artifact-uri, not both"
            )
        resolved_manifest_prefix = manifest_prefix or artifact_uri
        if not resolved_manifest_prefix:
            raise LeIsaacConfigError("--manifest-prefix is required")
        split_s3_uri(resolved_manifest_prefix)
        name = resource_name(run_id)
        nonce = secrets.token_hex(32)
        if image_pull_secret:
            # Nebius IAM-backed registry credentials are intentionally short lived. Refresh
            # the named secret for every launch so a pod scheduled onto a cold GPU node does
            # not depend on an old node cache or somebody else's credential-refresh cycle.
            ensure_registry_pull_secret_for_images(
                image,
                secret_name=image_pull_secret,
                namespace=namespace,
                k8s_context=context,
            )
            secret = _kubectl(
                context, namespace, ["get", "secret", image_pull_secret, "-o", "name"]
            )
            if secret.returncode:
                raise LeIsaacConfigError(
                    f"image pull secret {image_pull_secret!r} is missing in namespace {namespace!r}"
                )
        if transport == Transport.agent_relay:
            if not agent_project or not agent_name:
                raise LeIsaacConfigError(
                    "agent-relay requires --agent-project and --agent-name"
                )
            instance_id, media_host, ssh, auth_user, auth_password = (
                _agent_relay_context(agent_project, agent_name)
            )
            turn_cleanup_required = True
            artifact_storage = _agent_artifact_storage(agent_project, agent_name)
            dataset_bucket, dataset_prefix = split_s3_uri(output_path)
            if dataset_bucket != artifact_storage["bucket"]:
                raise LeIsaacConfigError(
                    "agent-relay output path bucket must match the selected agent's bucket"
                )
            agent_prefix = artifact_storage["prefix"]
            if agent_prefix and not (
                dataset_prefix == agent_prefix
                or dataset_prefix.startswith(agent_prefix + "/")
            ):
                raise LeIsaacConfigError(
                    "agent-relay output path must be inside the selected agent's artifact prefix"
                )
            prior_turn_peer_source = _existing_turn_peer_source(
                context, namespace, f"{name}-relay"
            )
            service = relay_service_manifest(
                run_id=run_id,
                namespace=namespace,
                agent_project=agent_project,
                agent_name=agent_name,
                source_ranges=source_ranges,
            )
            _apply(context, namespace, [service])
            relay_installed = True
            # Sessions from the previous topology ran coturn on the public agent
            # itself. Remove only this run's matching unit before the backhaul
            # relay takes ownership of public UDP 3478.
            _remove_agent_turn(ssh, run_id=run_id)
            _install_agent_relay(
                ssh,
                run_id=run_id,
                session_nonce=nonce,
            )
            certificate_sha256 = _agent_certificate_sha256(media_host)
            relay_secret = relay_client_secret_manifest(
                run_id=run_id,
                namespace=namespace,
                agent_host=media_host,
                session_nonce=nonce,
                certificate_sha256=certificate_sha256,
                auth_user=auth_user,
                auth_password=auth_password,
                client_source=_relay_source("reverse_client.py").decode("utf-8"),
            )
            _apply(context, namespace, [relay_secret])
            signal_host = "127.0.0.1"
        else:
            configured_storage = resolve_project_storage(None)
            artifact_storage = {
                "bucket": split_s3_uri(output_path)[0],
                "prefix": "",
                "endpoint": resolve_s3_endpoint(
                    config_endpoint=configured_storage.endpoint_url
                ),
                "access_key": os.environ.get("AWS_ACCESS_KEY_ID") or "",
                "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY") or "",
                "region": os.environ.get("AWS_REGION") or "eu-north1",
            }
            services = service_manifests(
                run_id=run_id,
                namespace=namespace,
                source_ranges=source_ranges,
            )
            _apply(context, namespace, services)
            signal_host = _external_ip(context, namespace, f"{name}-tcp")
            media_host = _external_ip(context, namespace, f"{name}-media")
            media_server = media_host
        if artifact_storage is None:
            raise LeIsaacConfigError("recorder storage configuration is unavailable")
        recorder_secret = recorder_secret_manifest(
            run_id=run_id,
            namespace=namespace,
            output_path=output_path,
            endpoint=artifact_storage["endpoint"],
            access_key=artifact_storage["access_key"],
            secret_key=artifact_storage["secret_key"],
            region=artifact_storage["region"],
        )
        _apply(context, namespace, [recorder_secret])
        deployment = deployment_manifest(
            run_id=run_id,
            namespace=namespace,
            image=image,
            media_host=media_host,
            session_nonce=nonce,
            image_pull_secret=image_pull_secret,
            relay_client_secret=(
                f"{name}-relay-client" if transport == Transport.agent_relay else ""
            ),
            recorder_secret=f"{name}-recorder",
            task=task,
            environment_id=environment_id,
            environment_index=environment_index,
            seed=seed,
            num_envs=num_envs,
        )
        _apply(context, namespace, [deployment])
        _wait_ready(context, namespace, name)
        if transport == Transport.agent_relay:
            if ssh is None:
                raise RuntimeError("LeIsaac agent relay has no SSH transport")
            media_server = _relay_media_server(context, namespace, name)
            for source in source_ranges:
                ingress = ensure_ingress(
                    vm_id=instance_id,
                    ports=(TURN_PORT,),
                    source=source,
                    tool=_TURN_CONTROL_TOOL,
                    protocol="UDP",
                )
                if ingress.changed:
                    created_ingress_specs.append(
                        (TURN_PORT, source, _TURN_CONTROL_TOOL, "UDP")
                    )
            if prior_turn_peer_source:
                remove_exact_npa_ingress_for_instance(
                    instance_id,
                    ports=(TURN_RELAY_PORT,),
                    source=prior_turn_peer_source,
                    tool=_TURN_MEDIA_TOOL,
                    protocol="UDP",
                )
            _apply(
                context,
                namespace,
                [
                    relay_service_manifest(
                        run_id=run_id,
                        namespace=namespace,
                        agent_project=agent_project,
                        agent_name=agent_name,
                        source_ranges=source_ranges,
                    )
                ],
            )
        health = _relay_status(ssh) if ssh is not None else _status(signal_host)
        if (
            health.get("state") != "ready"
            or health.get("task") != task
            or health.get("source_commit") != SOURCE_COMMIT
            or health.get("task_registry_fingerprint") != REGISTRY_FINGERPRINT
            or health.get("session_attestation") != session_attestation(nonce)
            or health.get("environment_id") != environment_id
            or int(health.get("environment_index", -1)) != environment_index
            or int(health.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"LeIsaac live attestation failed: {health}")
        manifest = session_manifest(
            run_id=run_id,
            image=image,
            signal_host=signal_host,
            media_host=media_host,
            media_server=media_server,
            session_nonce=nonce,
            expires_at=expires_at,
            gpu=str(health.get("gpu") or GPU_PRODUCT),
            created_at=str(health.get("started_at") or "") or None,
            transport=transport.value,
            task=task,
            environment_id=environment_id,
            environment_index=environment_index,
            seed=seed,
            num_envs=num_envs,
            output_path=output_path,
        )
        manifest_uri = _put_manifest(
            resolved_manifest_prefix,
            manifest,
            storage=artifact_storage,
        )
        if transport == Transport.agent_relay:
            _select_agent_leisaac_run(
                media_host,
                auth_user=auth_user,
                auth_password=auth_password,
                run_id=run_id,
                certificate_sha256=certificate_sha256,
            )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports SDK and kubectl failures
        cleanup_errors: list[str] = []
        if turn_cleanup_required and ssh is not None:
            try:
                _remove_agent_turn(ssh, run_id=run_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"TURN cleanup: {cleanup_exc}")
        if relay_installed and ssh is not None:
            try:
                _remove_agent_relay(ssh, run_id=run_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"relay cleanup: {cleanup_exc}")
        for port, source, tool, protocol in created_ingress_specs:
            try:
                remove_exact_npa_ingress_for_instance(
                    instance_id,
                    ports=(port,),
                    source=source,
                    tool=tool,
                    protocol=protocol,
                )
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"ingress cleanup: {cleanup_exc}")
        if name:
            try:
                _delete_resources(context, namespace, name)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"Kubernetes cleanup: {cleanup_exc}")
        if cleanup_errors:
            _fail(f"{exc}; cleanup also failed: {'; '.join(cleanup_errors)}")
        _fail(str(exc))
        return
    _emit(
        {
            "status": "ready",
            "run_id": run_id,
            "task": task,
            "environment_id": environment_id,
            "environment_index": environment_index,
            "seed": seed,
            "dataset": output_path.rstrip("/"),
            "gpu": health.get("gpu"),
            "image": image,
            "transport": transport.value,
            "deployment": name,
            "signal_host": signal_host,
            "media_host": media_host,
            "artifact": manifest_uri,
            "public_agent_url": (
                f"https://{media_host}/"
                if transport == Transport.agent_relay
                else "not used"
            ),
            "expires_at": expires_at or "none (service lifecycle)",
        },
        output,
    )


@app.command("status")
def status_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    context: str = typer.Option("", "--context"),
    namespace: str = typer.Option("default", "--namespace"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Report the live Kubernetes objects for a LeIsaac run."""

    try:
        name = resource_name(validate_run_id(run_id))
    except LeIsaacConfigError as exc:
        _fail(str(exc))
        return
    result = _kubectl(
        context,
        namespace,
        [
            "get",
            "deployment,service,pod",
            "-l",
            f"app.kubernetes.io/instance={name}",
            "-o",
            "json",
        ],
    )
    if result.returncode:
        _fail((result.stderr or result.stdout).strip())
    data = json.loads(result.stdout)
    _emit({"run_id": run_id, "resources": data.get("items", [])}, output)


@app.command("destroy")
def destroy_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    context: str = typer.Option("", "--context"),
    namespace: str = typer.Option("default", "--namespace"),
) -> None:
    """Delete this run's transient GPU deployment and LBs, preserving S3 evidence."""

    try:
        name = resource_name(validate_run_id(run_id))
    except LeIsaacConfigError as exc:
        _fail(str(exc))
        return
    relay = _kubectl(
        context, namespace, ["get", "service", f"{name}-relay", "-o", "json"]
    )
    if relay.returncode == 0:
        try:
            annotations = (
                json.loads(relay.stdout).get("metadata", {}).get("annotations", {})
                or {}
            )
            project = str(annotations.get("npa.nebius.com/agent-project") or "")
            agent_name = str(annotations.get("npa.nebius.com/agent-name") or "")
            sources = validate_source_ranges(
                str(annotations.get("npa.nebius.com/source-ranges") or "").split(",")
            )
            peer_source = str(
                annotations.get("npa.nebius.com/turn-peer-source") or ""
            ).strip()
            instance_id, _public_ip, ssh, _auth_user, _auth_password = (
                _agent_relay_context(project, agent_name)
            )
            _remove_agent_turn(ssh, run_id=run_id)
            _remove_agent_relay(ssh, run_id=run_id)
            ingress_specs = [
                (TURN_PORT, source, _TURN_CONTROL_TOOL) for source in sources
            ]
            if peer_source:
                validated_peer = _turn_peer_source(peer_source)
                ingress_specs.append(
                    (TURN_RELAY_PORT, validated_peer, _TURN_MEDIA_TOOL)
                )
            else:
                # Compatibility cleanup for sessions launched before TURN support.
                ingress_specs.extend(
                    (MEDIA_PORT, source, _RELAY_TOOL) for source in sources
                )
            for port, source, tool in ingress_specs:
                remove_exact_npa_ingress_for_instance(
                    instance_id,
                    ports=(port,),
                    source=source,
                    tool=tool,
                    protocol="UDP",
                )
        except Exception as exc:  # noqa: BLE001 - CLI cleanup boundary
            _fail(f"agent relay cleanup failed; Kubernetes resources retained: {exc}")
    try:
        _delete_resources(context, namespace, name)
    except Exception as exc:  # noqa: BLE001 - CLI cleanup boundary
        _fail(str(exc))
    typer.echo("transient LeIsaac Kubernetes and relay resources removed")

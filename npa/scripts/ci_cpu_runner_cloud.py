"""Keep disposable CI worker identities and Nebius operations in private local state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid


def _save(path: Path, value: dict | list) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _command(root: Path, command: list[str], payload=None, *, absent=False):
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return json.loads(result.stdout) if result.stdout.strip() else None
    missing = "code = NotFound" in result.stderr or "(HTTP 404)" in result.stderr
    if absent and missing:
        return None
    diagnostic = root / f"error-{uuid.uuid4().hex}.json"
    _save(diagnostic, {"stdout": result.stdout, "stderr": result.stderr})
    raise RuntimeError(
        f"Provider request failed; private diagnostics: {diagnostic.name}"
    )


def _github(
    root: Path, config: dict, endpoint: str, *, method="GET", payload=None, absent=False
):
    command = ["gh", "api", "--method", method]
    command.append(f"repos/{config['repository']}/{endpoint}")
    if payload is not None:
        command.extend(["--input", "-"])
    return _command(root, command, payload, absent=absent)


def _pages(root: Path, config: dict, endpoint: str, key: str) -> list[dict]:
    separator = "&" if "?" in endpoint else "?"
    command = [
        "gh",
        "api",
        "--paginate",
        "--slurp",
        f"repos/{config['repository']}/{endpoint}{separator}per_page=100",
    ]
    return [item for page in _command(root, command) for item in page[key]]


def _nebius(
    root: Path,
    config: dict,
    service: str,
    operation: str,
    payload: dict,
    *,
    absent=False,
):
    command = ["nebius", "--no-browser", "--no-check-update", "--format", "json"]
    if config.get("profile"):
        command.extend(["--profile", config["profile"]])
    command.extend([*service.split(), operation, "--file", "/dev/stdin"])
    return _command(root, command, payload, absent=absent)


def _owned(resource: dict, config: dict) -> None:
    metadata = resource["metadata"]
    if metadata["parent_id"] != config["project_id"]:
        raise ValueError("Resource belongs to a different project")
    labels = metadata.get("labels", {})
    if (
        labels.get("npa-owner") != config["owner"]
        or labels.get("npa-purpose") != "ci-runners"
    ):
        raise ValueError("Resource ownership does not match this runner pool")


def _nebius_items(root: Path, config: dict, service: str, parent: str) -> list[dict]:
    request = {"parent_id": parent, "page_size": 100}
    items = []
    tokens = set()
    while True:
        page = _nebius(root, config, service, "list", request)
        items.extend(page.get("items", []))
        token = page.get("next_page_token")
        if not token:
            return items
        if token in tokens:
            raise RuntimeError("Cloud inventory pagination did not advance")
        tokens.add(token)
        request["page_token"] = token


def _verify_project(root: Path, config: dict) -> None:
    receipt = config["project_create_response"]
    if receipt["metadata"]["id"] != config["project_id"]:
        raise ValueError("The project creation receipt does not match the pool")
    project = _nebius(root, config, "iam project", "get", {"id": config["project_id"]})
    for candidate in (receipt, project):
        if candidate["metadata"]["parent_id"] != config["tenant_id"]:
            raise ValueError("Project tenant differs from its creation receipt")
        if candidate["spec"]["region"] != config["region"]:
            raise ValueError("Project region differs from its creation receipt")
        if candidate["metadata"].get("labels", {}).get("npa-owner") != config["owner"]:
            raise ValueError("Project ownership differs from its creation receipt")


def _capacity_requirements(config: dict, count: int) -> dict[str, int]:
    cores = int(config["preset"].split("vcpu-", 1)[0])
    return {
        "compute.instance.non-gpu.vcpu": count * cores,
        "compute.instance.count": count,
        "compute.disk.count": count,
        "compute.disk.size.network-ssd": count * 64 * 1024**3,
        "vpc.ipv4-address.public.count": count,
        "vpc.allocation.count": count * 2,
    }


def _quota_headroom(quota: dict, name: str, *, inherited: bool):
    spec, status = quota["spec"], quota["status"]
    expected_unit = "byte" if name == "compute.disk.size.network-ssd" else "count"
    if status.get("unit") != expected_unit:
        raise ValueError(f"Unknown quota unit for {name}")
    if "limit" not in spec and inherited:
        return None
    usage = status.get("usage")
    if usage is None and status.get("usage_state") == "USAGE_STATE_NOT_USED":
        usage = 0
    if "limit" not in spec or usage is None:
        raise ValueError(f"Unverified quota capacity for {name}")
    return int(spec["limit"]) - int(usage)


def _check_capacity(root: Path, config: dict, count: int) -> None:
    requirements = _capacity_requirements(config, count)
    for field in ("tenant_id", "project_id"):
        quotas = _nebius_items(root, config, "quotas quota-allowance", config[field])
        regional = {
            q["metadata"]["name"]: q
            for q in quotas
            if q["spec"].get("region") == config["region"]
        }
        for name, needed in requirements.items():
            if name not in regional:
                raise ValueError(f"Missing capacity evidence for {name}")
            free = _quota_headroom(
                regional[name], name, inherited=field == "project_id"
            )
            if free is not None and free < needed:
                raise RuntimeError(
                    f"Insufficient {name}: need {needed}, available {free}"
                )


_LAUNCHER = """#!/bin/bash
set -euo pipefail
export AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache
export RUNNER_TOOL_CACHE=/opt/hostedtoolcache
cd /opt/actions-runner
exec ./run.sh --jitconfig "$(cat /run/npa-runner-jit)"
"""
_SERVICE = """[Unit]
After=network-online.target docker.service
Wants=network-online.target
[Service]
User=runner
WorkingDirectory=/opt/actions-runner
ExecStart=/usr/local/bin/npa-runner-job
ExecStopPost=+/usr/bin/systemctl poweroff
Restart=no
StandardOutput=journal+console
StandardError=journal+console
[Install]
WantedBy=multi-user.target
"""


def _worker_cloud_init(jit: str) -> str:
    files = [
        {
            "path": "/run/npa-runner-jit",
            "owner": "runner:runner",
            "permissions": "0600",
            "content": jit,
        },
        {
            "path": "/usr/local/bin/npa-runner-job",
            "permissions": "0755",
            "content": _LAUNCHER,
        },
        {
            "path": "/etc/systemd/system/npa-runner.service",
            "permissions": "0644",
            "content": _SERVICE,
        },
    ]
    return "#cloud-config\n" + json.dumps(
        {
            "ssh_pwauth": False,
            "write_files": files,
            "runcmd": [
                [
                    "chown",
                    "-R",
                    "runner:runner",
                    "/opt/actions-runner",
                    "/opt/hostedtoolcache",
                ],
                ["systemctl", "daemon-reload"],
                ["systemctl", "start", "npa-runner"],
            ],
        }
    )


def _instance_request(config: dict, name: str, jit: str) -> dict:
    labels = {"npa-owner": config["owner"], "npa-purpose": "ci-runners"}
    return {
        "metadata": {"parent_id": config["project_id"], "name": name, "labels": labels},
        "spec": {
            "hostname": name,
            "resources": {"platform": "cpu-d3", "preset": config["preset"]},
            "boot_disk": {
                "attach_mode": "READ_WRITE",
                "device_id": "boot",
                "managed_disk": {
                    "name": f"{name}-boot",
                    "labels": labels,
                    "spec": {
                        "type": "NETWORK_SSD",
                        "size_gibibytes": 64,
                        "source_image_id": config["image_id"],
                    },
                },
            },
            "network_interfaces": [
                {
                    "name": "eth0",
                    "subnet_id": config["subnet_id"],
                    "ip_address": {},
                    "public_ip_address": {},
                    "security_groups": [{"id": config["worker_security_group_id"]}],
                }
            ],
            "cloud_init_user_data": _worker_cloud_init(jit),
        },
    }


def _register_worker(root: Path, config: dict, record: dict) -> None:
    response = _github(
        root,
        config,
        "actions/runners/generate-jitconfig",
        method="POST",
        payload={
            "name": record["name"],
            "runner_group_id": 1,
            "labels": [config["label"]],
            "work_folder": "_work",
        },
    )
    record.update(
        runner_id=response["runner"]["id"], jit=response["encoded_jit_config"]
    )
    _save(root / "workers" / f"{record['name']}.json", record)


def _create_worker(root: Path, config: dict, name: str) -> None:
    record_path = root / "workers" / f"{name}.json"
    record = _load(record_path)
    if "runner_id" not in record:
        _register_worker(root, config, record)
    instance = _nebius(
        root,
        config,
        "compute instance",
        "get-by-name",
        {
            "parent_id": config["project_id"],
            "name": name,
        },
        absent=True,
    )
    if instance is None:
        instance = _nebius(
            root,
            config,
            "compute instance",
            "create",
            _instance_request(config, name, record["jit"]),
        )
    _owned(instance, config)
    record.update(instance_id=instance["metadata"]["id"], phase="provisioned")
    record.pop("jit", None)
    _save(record_path, record)


def _owned_worker(root: Path, config: dict, record: dict):
    by_id = "instance_id" in record
    request = (
        {"id": record["instance_id"]}
        if by_id
        else {
            "parent_id": config["project_id"],
            "name": record["name"],
        }
    )
    instance = _nebius(
        root,
        config,
        "compute instance",
        "get" if by_id else "get-by-name",
        request,
        absent=True,
    )
    if instance:
        _owned(instance, config)
        if instance["metadata"]["name"] != record["name"]:
            raise ValueError("Worker identity no longer matches its creation receipt")
        if "managed_disk" not in instance["spec"]["boot_disk"]:
            raise ValueError("Worker boot disk is no longer owned by its VM")
        record["instance_id"] = instance["metadata"]["id"]
    return instance


def _remove_registration(root: Path, config: dict, record: dict) -> None:
    runners = _pages(root, config, "actions/runners", "runners")
    for runner in runners:
        if runner["name"] != record["name"] and runner["id"] != record.get("runner_id"):
            continue
        if runner["name"] != record["name"] or runner["id"] != record.get(
            "runner_id", runner["id"]
        ):
            raise ValueError("Runner identity no longer matches the creation receipt")
        if runner["busy"]:
            raise RuntimeError(
                "Runner is still busy; drain its workflow before removal"
            )
        _github(root, config, f"actions/runners/{runner['id']}", method="DELETE")


def _delete_worker(root: Path, config: dict, record: dict) -> None:
    instance = _owned_worker(root, config, record)
    _remove_registration(root, config, record)
    if instance:
        _nebius(
            root, config, "compute instance", "delete", {"id": record["instance_id"]}
        )
    remaining = _owned_worker(root, config, record)
    if remaining is not None:
        raise RuntimeError("Worker deletion has not been verified")
    record["phase"] = "deleted"
    _save(root / "retired" / f"{record['name']}.json", record)
    (root / "workers" / f"{record['name']}.json").unlink()

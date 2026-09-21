"""Fresh same-authority read-only checks for terminal MK8s recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import yaml

from npa.clients.nebius import nebius_cli_env
from npa.cluster.absent_evidence import digest, pinned_bytes, require


def _bound_profile(authority: dict) -> dict:
    config = {
        "path": authority["provider_config_path"],
        "sha256": authority["provider_config_sha256"],
    }
    credentials = {
        "path": authority["credential_file_path"],
        "sha256": authority["credential_file_sha256"],
    }
    profile = yaml.safe_load(pinned_bytes(config))["profiles"][authority["profile"]]
    require(
        authority["key_backed"] is True
        and authority["attached_metadata_selected"] is False,
        "Recovery requires original key-backed authority",
    )
    require(
        set(profile)
        == {
            "endpoint",
            "auth-type",
            "parent-id",
            "tenant-id",
            "service-account-credentials-file-path",
        },
        "Only an explicit plain service-account profile is supported",
    )
    require(
        profile["auth-type"] == "service account"
        and profile["service-account-credentials-file-path"] == credentials["path"],
        "Original credential selector changed",
    )
    require(
        profile["parent-id"] == authority["project_id"]
        and profile["tenant-id"] == authority["tenant_id"],
        "Authority scope changed",
    )
    pinned_bytes(credentials)
    return config


def authority_environment(authority: dict) -> tuple[list[str], dict]:
    config = _bound_profile(authority)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NEBIUS_", "NPA_NEBIUS_"))
    }
    env.pop("NPA_REUSE_IAM_TOKEN", None)
    env = nebius_cli_env(env)
    env.update(
        NEBIUS_CONFIG_DIR=str(Path(config["path"]).parent),
        NEBIUS_PROFILE=authority["profile"],
        AWS_EC2_METADATA_DISABLED="true",
    )
    binary = shutil.which("nebius")
    require(bool(binary), "Nebius CLI is unavailable")
    return [binary, "--config", config["path"], "--profile", authority["profile"]], env


class AbsenceProvider:
    """Retain every actual provider response without changing cloud resources."""

    def __init__(self, authority: dict, output: Path):
        self.authority, self.output = authority, output
        self.records: list[dict] = []

    def query(self, args: list[str]) -> subprocess.CompletedProcess:
        argv, env = authority_environment(self.authority)
        result = subprocess.run(
            [*argv, *args, "--format", "json"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        authority_environment(self.authority)
        value = {
            "argv": [*argv, *args],
            "exit": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        path = self.output / f"provider-{len(self.records):04d}.json"
        data = (json.dumps(value, indent=2) + "\n").encode()
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
        self.records.append({"file": path.name, "sha256": digest(data)})
        return result

    def project(self) -> None:
        result = self.query(
            ["iam", "project", "get", "--id", self.authority["project_id"]]
        )
        require(result.returncode == 0, "Project authority is unavailable")
        payload = json.loads(result.stdout)
        require(
            payload["metadata"]["id"] == self.authority["project_id"]
            and payload["metadata"]["parent_id"] == self.authority["tenant_id"]
            and payload["status"]["state"] == "ACTIVE",
            "Wrong project authority",
        )

    def inventory(
        self, command: list[str], parent: str, *, absent_parent=False
    ) -> list[dict]:
        rows, token, seen = [], "", set()
        while True:
            args = [*command, "list", "--parent-id", parent]
            if token:
                args.extend(["--page-token", token])
            result = self.query(args)
            if result.returncode and absent_parent and not rows and not token:
                require_absent(result)
                return []
            require(result.returncode == 0, "Complete provider inventory unavailable")
            payload = json.loads(result.stdout)
            require(
                isinstance(payload, dict) and isinstance(payload.get("items"), list),
                "Incomplete provider inventory response",
            )
            rows.extend(payload["items"])
            token = payload.get("next_page_token", "")
            require(
                isinstance(token, str) and (not token or token not in seen),
                "Invalid or repeated inventory page token",
            )
            if not token:
                return rows
            seen.add(token)


def command_for(kind: str) -> list[str]:
    return {
        "cluster": ["mk8s", "cluster"],
        "node-group": ["mk8s", "node-group"],
        "application-release": ["applications", "v1alpha1", "k8s-release"],
    }[kind]


def require_absent(result: subprocess.CompletedProcess) -> None:
    require(
        result.returncode > 0
        and not result.stdout.strip()
        and re.findall(r"\bcode\s*=\s*([A-Za-z]+)\b", result.stderr) == ["NotFound"],
        "Exact resource absence is unverified (live, denied or unknown)",
    )


def verify_absence(provider: AbsenceProvider, evidence: dict) -> None:
    provider.project()
    resources, journal = evidence["resources"], evidence["journal"]
    cluster = next(r for r in resources if r["kind"] == "cluster")
    for resource in resources:
        require_absent(
            provider.query(
                [*command_for(resource["kind"]), "get", "--id", resource["id"]]
            )
        )
    clusters = provider.inventory(command_for("cluster"), journal["project_id"])
    for item in clusters:
        require(
            item["metadata"]["parent_id"] == journal["project_id"],
            "Foreign parent in project inventory",
        )
        require(
            item["metadata"]["name"] != journal["requested_name"]
            and item["metadata"]["id"] != cluster["id"],
            "Owned cluster name or ID remains live or ambiguous",
        )
    releases = provider.inventory(
        command_for("application-release"), journal["project_id"]
    )
    for item in releases:
        parent = item["metadata"]["parent_id"]
        cluster_id = item.get("spec", {}).get("cluster_id") or item.get("cluster_id")
        require(
            parent == journal["project_id"] and bool(cluster_id),
            "Application inventory cannot establish cluster identity",
        )
        require(cluster_id != cluster["id"], "A cluster application remains live")
    children = provider.inventory(
        command_for("node-group"), cluster["id"], absent_parent=True
    )
    require(not children, "Cluster child inventory is not empty")

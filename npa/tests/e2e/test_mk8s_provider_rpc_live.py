"""Read-only RPC-default checks around one operator-owned real MK8s lifecycle.

Set NPA_MK8S_RPC_LIVE_CONFIG to a private JSON configuration. Select the live
test after supported provision, and the cleanup test after supported destroy.
The test never creates, adopts, cancels, or destroys infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from npa.clients.nebius import nebius_cli_env

pytestmark = pytest.mark.e2e


def _bound_bytes(entry):
    path = Path(entry["path"])
    assert path.is_file() and not path.is_symlink(), "private input must be regular"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], "input binding changed"
    return data


def _pinned(config, name):
    return json.loads(_bound_bytes(config[name]))


def _authority(config, start):
    authority = config["authority"]
    assert start["authority"] == authority, "producing authority changed"
    assert authority["profile"] == config["profile"] and config["profile"]
    for key in ("project_id", "tenant_id"):
        assert authority[key] == config[key], "producing scope changed"
    profiles = yaml.safe_load(_bound_bytes(authority["config_file"]))["profiles"]
    profile = profiles[authority["profile"]]
    assert set(profile) == {
        "endpoint",
        "auth-type",
        "service-account-credentials-file-path",
        "parent-id",
        "tenant-id",
    }, "live verifier requires an explicit plain service-account profile"
    assert profile["auth-type"] == "service account"
    assert profile["endpoint"] == authority["endpoint"]
    assert profile["parent-id"] == config["project_id"]
    assert profile["tenant-id"] == config["tenant_id"]
    assert (
        profile["service-account-credentials-file-path"]
        == authority["credentials_file"]["path"]
    )
    _bound_bytes(authority["credentials_file"])
    return authority


def _provider_env(authority):
    # Reject unknown future selectors too; only explicitly bound selectors survive.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NEBIUS_", "NPA_NEBIUS_"))
    }
    env.pop("NPA_REUSE_IAM_TOKEN", None)
    env = nebius_cli_env(env)
    env["NEBIUS_CONFIG_DIR"] = str(Path(authority["config_file"]["path"]).parent)
    env["NEBIUS_PROFILE"] = authority["profile"]
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    return env


def _bindings(config, source_revision):
    start = _pinned(config, "provision_start")
    _authority(config, start)
    finish = _pinned(config, "provision_result")
    saved = _pinned(config, "deployment_sidecar")
    state = _pinned(config, "terraform_state")
    assert start["source_revision"] == source_revision == config["source_revision"]
    assert finish["started_at"] == start["started_at"], "producer attempt changed"
    argv = start["argv"]
    assert argv[argv.index("--context") + 1] == config["context"]
    assert argv[argv.index("--project") + 1] == config["project_alias"]
    assert finish["exit"] == 0, "provisioning did not pass its required gates"
    assert saved["project_id"] == config["project_id"]
    assert saved["cluster_name"] == config["cluster_name"]
    assert saved["status"] == "deployed"
    resources = {}
    for resource in state["resources"]:
        if resource["type"] not in {
            "nebius_mk8s_v1_cluster",
            "nebius_mk8s_v1_node_group",
        }:
            continue
        for instance in resource["instances"]:
            assert instance.get("status") != "tainted"
            resources.setdefault(resource["type"], []).append(
                instance["attributes"]["id"]
            )
    assert resources["nebius_mk8s_v1_cluster"] == [config["cluster_id"]]
    assert set(resources["nebius_mk8s_v1_node_group"]) == set(config["node_group_ids"])
    assert len(config["node_group_ids"]) == len(set(config["node_group_ids"])) > 0
    return saved


def _configured(phase):
    selected = os.environ.get("NPA_MK8S_RPC_LIVE_CONFIG")
    if not selected:
        pytest.skip(
            "supply exact private source/lifecycle bindings for a real owned run"
        )
    config = json.loads(Path(selected).read_text())
    if config["phase"] != phase:
        pytest.skip("select the phase corresponding to the real lifecycle state")
    repo = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    assert config["profile"], "an explicit operator profile is required"
    saved = _bindings(config, source)
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not evidence.is_symlink() and evidence.stat().st_mode & 0o077 == 0
    return config, saved, evidence


def _provider(config, evidence, label, args):
    authority = _authority(config, _pinned(config, "provision_start"))
    env = _provider_env(authority)
    argv = [
        "nebius",
        "--config",
        authority["config_file"]["path"],
        "--profile",
        authority["profile"],
        *args,
        "--format",
        "json",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    descriptor = os.open(
        evidence / (label + ".json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {
                "argv": argv,
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            stream,
        )
    return result


def _verify_project_authority(config, evidence):
    result = _provider(
        config,
        evidence,
        "project-authority",
        ["iam", "project", "get", "--id", config["project_id"]],
    )
    assert result.returncode == 0, "project authority must be verified before absence"
    payload = json.loads(result.stdout)
    assert payload["metadata"]["id"] == config["project_id"]
    assert payload["metadata"]["parent_id"] == config["tenant_id"]
    assert payload["status"]["state"] == "ACTIVE"


def _check_live_identity(payload, expected_id, expected_parent, expected_name=None):
    assert payload["metadata"]["id"] == expected_id
    assert payload["metadata"]["parent_id"] == expected_parent
    if expected_name is not None:
        assert payload["metadata"]["name"] == expected_name
    assert payload["status"]["state"] == "RUNNING"


def _check_absence(result):
    assert result.returncode != 0 and not result.stdout.strip()
    assert "code = NotFound" in result.stderr, (
        "absence requires typed NotFound, not denied/unknown"
    )


def test_mk8s_provider_rpc_live_deployment():
    config, saved, evidence = _configured("live")
    _verify_project_authority(config, evidence)
    receipt = saved["provider_rpc_deadlines"]
    minutes = config["apply_timeout_minutes"]
    assert receipt["apply_timeout_minutes"] == minutes
    assert receipt["inserted_defaults"] == {
        key: f"{minutes}m" for key in ("timeout", "per_retry_timeout", "auth_timeout")
    }
    provider = Path(config["materialized_provider"]["path"])
    digest = hashlib.sha256(provider.read_bytes()).hexdigest()
    assert (
        digest
        == config["materialized_provider"]["sha256"]
        == receipt["materialized_provider_sha256"]
    )
    result = _provider(
        config,
        evidence,
        "live-cluster",
        ["mk8s", "cluster", "get", "--id", config["cluster_id"]],
    )
    assert result.returncode == 0, "inspect private provider evidence"
    _check_live_identity(
        json.loads(result.stdout),
        config["cluster_id"],
        config["project_id"],
        config["cluster_name"],
    )
    for index, identity in enumerate(config["node_group_ids"]):
        result = _provider(
            config,
            evidence,
            f"live-node-group-{index}",
            ["mk8s", "node-group", "get", "--id", identity],
        )
        assert result.returncode == 0, "inspect private provider evidence"
        _check_live_identity(json.loads(result.stdout), identity, config["cluster_id"])


def test_mk8s_provider_rpc_live_cleanup():
    config, _saved, evidence = _configured("cleanup")
    _verify_project_authority(config, evidence)
    result = _provider(
        config,
        evidence,
        "absent-cluster",
        ["mk8s", "cluster", "get", "--id", config["cluster_id"]],
    )
    _check_absence(result)
    for index, identity in enumerate(config["node_group_ids"]):
        result = _provider(
            config,
            evidence,
            f"absent-node-group-{index}",
            ["mk8s", "node-group", "get", "--id", identity],
        )
        _check_absence(result)

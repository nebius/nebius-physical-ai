"""Real subprocess proof that npa storage/config teardown deletes owned storage.

Shells out to the real `npa` CLI (pinned `PYTHONPATH`/`sys.executable`) to
delete one root-provisioned disposable bucket and storage service account,
verifying provider-side absence afterward. It proves one real deletion cycle
happened. Separate credential-pruning concurrency coverage lives in
`npa/tests/cli/test_cleanup_teardown.py` as
`test_bucket_prune_survives_a_concurrent_unrelated_project_write`.

Env contract (every variable is required for the live path; anything missing
or unsafe makes the test skip before importing npa or running any command):

- ``NPA_INTEGRATION_E2E=1`` -- the repository-wide e2e opt-in.
- ``NPA_STORAGE_CLEANUP_LIVE_E2E=1`` -- purpose-specific mutation opt-in.
- ``NPA_E2E_PROJECT=<alias>`` -- the project alias to tear down.
- ``NPA_CONFIG_DIR=<path>`` -- an explicit, private, owner-only, pre-populated
  config directory (already holding a real `config.yaml`/`credentials.yaml`
  for *alias*). Must not be the host default `~/.npa`, must not live inside
  this repository checkout, and must not be a symlink.
- ``NPA_STORAGE_CLEANUP_LIVE_E2E_EVIDENCE_DIR=<path>`` -- an explicit,
  private, owner-only, pre-existing, and distinct directory. Every
  subprocess's raw output, a non-secret deletion-intent record, and a source
  fingerprint are written here; nothing sensitive is printed by the test.

The target project must hold exactly one storage bucket, one storage service
account, and the provider's own unique default network/subnet/security-group
(or none at all) -- and no other compute/VPC/mk8s/registry/serverless/
snapshot/GPU-cluster resource. Any extra or unrecognized resource makes the
test fail closed before any deletion is attempted. Run it with::

    NPA_INTEGRATION_E2E=1 NPA_STORAGE_CLEANUP_LIVE_E2E=1 \\
    NPA_E2E_PROJECT=<alias> \\
    NPA_CONFIG_DIR=/private/path/to/config \\
    NPA_STORAGE_CLEANUP_LIVE_E2E_EVIDENCE_DIR=/private/path/to/evidence \\
      npa/.venv/bin/python -m pytest \\
      npa/tests/e2e/test_config_storage_cleanup_live_e2e.py -q -s
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

_MODULE = sys.modules[__name__]

_MUTATION_OPT_IN_VAR = "NPA_STORAGE_CLEANUP_LIVE_E2E"
_EVIDENCE_DIR_VAR = "NPA_STORAGE_CLEANUP_LIVE_E2E_EVIDENCE_DIR"
_PROJECT_VAR = "NPA_E2E_PROJECT"
_CONFIG_DIR_VAR = "NPA_CONFIG_DIR"

_THIS_FILE = Path(__file__).resolve()
_NPA_ROOT = _THIS_FILE.parents[2]  # .../npa (contains src/, tests/)
_REPO_ROOT = _THIS_FILE.parents[3]  # repository checkout root

_TESTED_SOURCE_FILES = (
    "src/npa/cli/storage.py",
    "src/npa/clients/config.py",
    "src/npa/clients/nebius.py",
    "src/npa/clients/project_credential_store.py",
    "src/npa/cli/main.py",
    "tests/e2e/test_config_storage_cleanup_live_e2e.py",
)

_APP_INVOCATION = "from npa.cli.main import app; app()"

_DEPENDENCY_INVENTORY_SCRIPT = """
import json
import sys

from npa.clients.nebius import list_project_dependencies

print(json.dumps(list_project_dependencies(sys.argv[1])))
"""

_PROJECT_IDENTITY_SCRIPT = """
import dataclasses
import json
import sys

from npa.clients.nebius import get_project_identity

identity = get_project_identity(sys.argv[1], tenant_id=sys.argv[2])
print(json.dumps(None if identity is None else dataclasses.asdict(identity)))
"""

_DEFAULT_NETWORK_IDENTITY_SCRIPT = """
import dataclasses
import json
import sys

from npa.clients.nebius import get_project_default_network_identity

identity = get_project_default_network_identity(sys.argv[1])
print(json.dumps(None if identity is None else dataclasses.asdict(identity)))
"""

_SERVICE_ACCOUNT_IDENTITY_SCRIPT = """
import dataclasses
import json
import sys

from npa.clients.nebius import get_service_account_identity

identity = get_service_account_identity(sys.argv[1], project_id=sys.argv[2])
print(json.dumps(None if identity is None else dataclasses.asdict(identity)))
"""

_STORAGE_IAM_OBSERVATION_SCRIPT = """
import json
import sys

from npa.cli.storage import (
    _observation_dict, _observe_storage_iam, _resolve_storage_iam_context,
)

context = _resolve_storage_iam_context(
    project=sys.argv[1], project_id=sys.argv[2], service_account_id=sys.argv[3],
)
print(json.dumps(_observation_dict(_observe_storage_iam(context))))
"""

_BUCKET_CONFIG_SCRIPT = """
import json
import sys

from npa.clients.config import resolve_project_storage, resolve_terraform_state

storage = resolve_project_storage(
    sys.argv[1], include_shared_credentials=False, include_environment=False
)
terraform_state = resolve_terraform_state(sys.argv[1])
print(json.dumps({
    "configured_bucket_present": bool(storage.checkpoint_bucket),
    "terraform_state_bucket_present": bool(terraform_state.bucket),
}))
"""

# `list_project_dependencies` does not yet enumerate disk/instance snapshots
# or standalone GPU clusters as their own resource class, so this reuses the
# identical strict list-then-validate pattern directly against the raw
# provider CLI for just those two kinds.
_EXTRA_INVENTORY_SCRIPT = """
import json
import sys

from npa.clients.nebius import _iam_profile_args, _run_json

_KINDS = (
    ("compute_disk_snapshots", ("compute", "disk-snapshot", "list")),
    ("compute_gpu_clusters", ("compute", "gpu-cluster", "list")),
)


def _child_ids(command, profile_args, project_id):
    payload = _run_json([*profile_args, *command, "--parent-id", project_id, "--all"])
    items = [] if payload in ({}, {"items": None}) else payload.get("items")
    if not isinstance(items, list) or not all(isinstance(row, dict) for row in items):
        raise SystemExit(f"Nebius returned schema-invalid inventory for {command}")
    ids = []
    for row in items:
        metadata = row.get("metadata")
        child_id = str((metadata or {}).get("id") or "").strip()
        if not isinstance(metadata, dict) or not child_id:
            raise SystemExit(f"Nebius returned a {command} child without immutable identity")
        ids.append(child_id)
    return ids


_profile_args, _resolved_profile = _iam_profile_args(None)
_project_id = sys.argv[1]
_inventory = {kind: _child_ids(command, _profile_args, _project_id) for kind, command in _KINDS}
print(json.dumps(_inventory))
"""

# Provider dependency kinds ("npa.clients.nebius.list_project_dependencies")
# that must be empty in a project holding only the one disposable bucket and
# storage service account under test.
_STRICTLY_EMPTY_DEPENDENCY_KINDS = (
    "compute_instances",
    "compute_disks",
    "compute_filesystems",
    "vpc_allocations",
    "mk8s_clusters",
    "registries",
    "ai_endpoints",
    "ai_jobs",
)
_EXPECTED_SOLE_DEPENDENCY_COUNTS = {
    "storage_buckets": 1,
    "service_accounts": 1,
    "access_keys": 1,
}
# The real inventory always includes these three keys too (a project normally
# carries exactly one provider default network/subnet/security-group). They
# are part of the known schema so the exact-key equality check below does not
# reject real inventory, but their content is verified separately by
# `get_project_default_network_identity` (`_precheck_default_network_topology`),
# not by a blunt emptiness check here.
_VERIFIED_BY_DEFAULT_NETWORK_TOPOLOGY = (
    "vpc_networks",
    "vpc_subnets",
    "vpc_security_groups",
)
_ALL_DEPENDENCY_KINDS = (
    frozenset(_STRICTLY_EMPTY_DEPENDENCY_KINDS)
    | frozenset(_EXPECTED_SOLE_DEPENDENCY_COUNTS)
    | frozenset(_VERIFIED_BY_DEFAULT_NETWORK_TOPOLOGY)
)
_EXTRA_INVENTORY_KINDS = frozenset({"compute_disk_snapshots", "compute_gpu_clusters"})

# Independently re-derived (not copied) from the real production registry, so
# a schema-drift regression test can compare against production truth instead
# of trusting this file's own `_ALL_DEPENDENCY_KINDS`.
_PROJECT_CHILD_LIST_KINDS_SCRIPT = """
import json

from npa.clients.nebius import _PROJECT_CHILD_LIST_COMMANDS

kinds = sorted({kind for kind, _command, _supports_all in _PROJECT_CHILD_LIST_COMMANDS})
print(json.dumps(kinds))
"""

# CLI flags this test relies on, checked hermetically against real --help
# output so a renamed/removed flag is caught before it is ever run live.
_HELP_FLAG_EXPECTATIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("storage", "bucket", "list"), ("--project", "--json")),
    (
        ("storage", "bucket", "delete"),
        ("--project", "--id", "--yes", "--wait", "--json"),
    ),
    (
        ("storage", "service-account", "delete"),
        ("--project", "--id", "--dry-run", "--yes", "--json"),
    ),
    (("configure",), ("--forget-project",)),
    (("workbench", "workflow", "list"), ("--project", "--json")),
    (("workbench", "health", "preflight"), ("--project", "--checks", "--json")),
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_fingerprint() -> dict[str, Any]:
    """Hash the exact product source this test exercises, tied to git HEAD."""
    file_sha256 = {
        relative: _sha256_hex((_NPA_ROOT / relative).read_text(encoding="utf-8"))
        for relative in _TESTED_SOURCE_FILES
    }
    return {"git_head_commit": _git_head_commit(), "file_sha256": file_sha256}


def _ownership_reasons(label: str, resolved: Path) -> list[str]:
    """Require an owner-only, non-symlink, current-user-owned real directory."""
    try:
        info = os.lstat(resolved)
    except OSError as exc:
        return [f"{label} could not be inspected: {exc}"]
    if stat.S_ISLNK(info.st_mode):
        return [f"{label} must not be a symlink"]
    if not stat.S_ISDIR(info.st_mode):
        return [f"{label} must already exist as a real directory"]
    reasons: list[str] = []
    if info.st_uid != os.getuid():
        reasons.append(f"{label} must be owned by the current user")
    mode = stat.S_IMODE(info.st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        reasons.append(f"{label} must be owner-only (no group/other permissions)")
    return reasons


def _unsafe_path_reasons(
    label: str, raw: str, *, forbid_host_default: bool
) -> list[str]:
    """Return every reason *raw* is not an acceptable private directory."""
    if not raw:
        return [f"{label} is required and must be an explicit private path"]
    path = Path(raw)
    if not path.is_absolute():
        return [f"{label} must be an absolute path"]
    if path.is_symlink():
        return [f"{label} must not be a symlink"]
    resolved = path.resolve()
    reasons: list[str] = []
    if forbid_host_default and resolved == (Path.home() / ".npa").resolve():
        reasons.append(f"{label} must not be the host default ~/.npa")
    if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
        reasons.append(f"{label} must not be inside the repository checkout")
    reasons.extend(_ownership_reasons(label, resolved))
    return reasons


def _live_requirement_violations(env: Mapping[str, str]) -> list[str]:
    """Every reason the live mutation path must stay skipped, given *env*.

    Pure and side-effect free so the safety contract is unit-testable without
    importing npa or touching a network.
    """
    violations: list[str] = []
    if env.get("NPA_INTEGRATION_E2E", "").strip() != "1":
        violations.append("NPA_INTEGRATION_E2E=1 is required")
    if env.get(_MUTATION_OPT_IN_VAR, "").strip() != "1":
        violations.append(f"{_MUTATION_OPT_IN_VAR}=1 is required")
    if not env.get(_PROJECT_VAR, "").strip():
        violations.append(f"{_PROJECT_VAR} is required")

    config_dir = env.get(_CONFIG_DIR_VAR, "").strip()
    evidence_dir = env.get(_EVIDENCE_DIR_VAR, "").strip()
    config_reasons = _unsafe_path_reasons(
        _CONFIG_DIR_VAR, config_dir, forbid_host_default=True
    )
    evidence_reasons = _unsafe_path_reasons(
        _EVIDENCE_DIR_VAR, evidence_dir, forbid_host_default=False
    )
    violations.extend(config_reasons)
    violations.extend(evidence_reasons)
    if (
        not config_reasons
        and not evidence_reasons
        and Path(config_dir).resolve() == Path(evidence_dir).resolve()
    ):
        violations.append(f"{_CONFIG_DIR_VAR} and {_EVIDENCE_DIR_VAR} must be distinct")
    return violations


def _skip_unless_live_ready(env: Mapping[str, str]) -> None:
    violations = _live_requirement_violations(env)
    if violations:
        pytest.skip("Live storage cleanup e2e is not ready: " + "; ".join(violations))


def _assert_private_file(path: Path) -> None:
    """Require an owner-only, non-symlink, current-user-owned regular file."""
    info = os.lstat(path)
    assert not stat.S_ISLNK(info.st_mode), f"{path.name} must not be a symlink"
    assert stat.S_ISREG(info.st_mode), f"{path.name} must be a regular file"
    assert info.st_uid == os.getuid(), f"{path.name} must be owned by the current user"
    mode = stat.S_IMODE(info.st_mode)
    assert not (mode & (stat.S_IRWXG | stat.S_IRWXO)), f"{path.name} must be owner-only"


def _document_contains_value(node: object, needle: str) -> bool:
    """Recursively check whether *needle* appears as a leaf string anywhere."""
    if isinstance(node, dict):
        return any(_document_contains_value(value, needle) for value in node.values())
    if isinstance(node, list):
        return any(_document_contains_value(item, needle) for item in node)
    return node == needle


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    _assert_private_file(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return document if isinstance(document, dict) else {}


def _resolve_expected_identity(config_dir: Path, alias: str) -> dict[str, str]:
    """Resolve project/tenant/region for *alias* from the private config copy."""
    stanza = (_read_yaml(config_dir / "config.yaml").get("projects") or {}).get(alias)
    assert isinstance(stanza, dict), (
        "the project alias is not configured in the private NPA_CONFIG_DIR"
    )
    identity = {
        key: str(stanza.get(key, "") or "").strip()
        for key in ("project_id", "tenant_id", "region")
    }
    assert all(identity.values()), (
        "the configured project stanza is missing project_id/tenant_id/region"
    )
    return identity


def _help_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_NPA_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _live_env(config_dir: Path, evidence_dir: Path) -> dict[str, str]:
    env = _help_env()
    env["NPA_CONFIG_DIR"] = str(config_dir)
    env["NPA_TEARDOWN_RECEIPT_DIR"] = str(evidence_dir / "teardown-receipts")
    return env


def _run_python(
    script: str, argv: list[str], *, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one Python subprocess with no caller-imposed wall-clock timeout.

    Per-RPC safeguards live in the provider client itself; this test does not
    add a blanket destructive-operation time cap on top of them.
    """
    return subprocess.run(
        [sys.executable, "-c", script, *argv],
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _run_npa(
    args: list[str], *, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return _run_python(_APP_INVOCATION, args, env=env)


def _record_step(
    evidence_dir: Path,
    step: str,
    result: subprocess.CompletedProcess[str],
    *,
    command_description: str,
) -> Path:
    """Write raw output privately, plus a hash-only manifest safe to cite in failures."""
    (evidence_dir / f"{step}.stdout.txt").write_text(result.stdout or "")
    (evidence_dir / f"{step}.stderr.txt").write_text(result.stderr or "")
    manifest_path = evidence_dir / f"{step}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "step": step,
                "command": command_description,
                "returncode": result.returncode,
                "stdout_sha256": _sha256_hex(result.stdout or ""),
                "stderr_sha256": _sha256_hex(result.stderr or ""),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest_path


def _step_json(result: subprocess.CompletedProcess[str]) -> Any:
    try:
        return json.loads(result.stdout)
    except ValueError:
        return None


def _run_and_record(
    step: str,
    *,
    runner: Callable[[], subprocess.CompletedProcess[str]],
    evidence_dir: Path,
    command_description: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run one subprocess, record it, and require a zero exit before returning."""
    result = runner()
    manifest = _record_step(
        evidence_dir, step, result, command_description=command_description
    )
    assert result.returncode == 0, (
        f"live storage cleanup step {step!r} did not exit 0; see {manifest}"
    )
    return result, manifest


def _validated_string_id_lists(
    payload: Any, expected_kinds: frozenset[str], manifest: Path
) -> dict[str, list[str]]:
    """Require exactly the expected keys, each holding a list of string IDs.

    Missing keys, unknown/extra keys, and wrong-typed values all fail closed
    -- not just a known key that happens to be nonempty.
    """
    assert isinstance(payload, dict), (
        f"provider inventory was unreadable; see {manifest}"
    )
    assert set(payload) == expected_kinds, (
        f"provider inventory has missing or unknown resource classes; see {manifest}"
    )
    for kind, ids in payload.items():
        assert isinstance(ids, list) and all(isinstance(item, str) for item in ids), (
            f"{kind} inventory was not a list of IDs; see {manifest}"
        )
    return payload


def _precheck_provider_dependency_inventory(
    project_id: str, env: Mapping[str, str], evidence_dir: Path
) -> tuple[str, str]:
    """Inventory every real child resource class; fail closed on extras.

    Uses ``npa.clients.nebius.list_project_dependencies``, the same typed
    provider-API sweep NPA's own guarded project deletion uses.
    """
    result, manifest = _run_and_record(
        "01_provider_dependency_inventory",
        runner=lambda: _run_python(_DEPENDENCY_INVENTORY_SCRIPT, [project_id], env=env),
        evidence_dir=evidence_dir,
        command_description=f"npa.clients.nebius.list_project_dependencies({project_id!r})",
    )
    inventory = _validated_string_id_lists(
        _step_json(result), _ALL_DEPENDENCY_KINDS, manifest
    )
    nonempty = {
        kind: inventory[kind]
        for kind in _STRICTLY_EMPTY_DEPENDENCY_KINDS
        if inventory[kind]
    }
    assert not nonempty, (
        f"provider dependency inventory found an unexpected resource; see {manifest}"
    )
    for kind, expected_count in _EXPECTED_SOLE_DEPENDENCY_COUNTS.items():
        assert len(inventory[kind]) == expected_count, (
            f"{kind} inventory did not match the sole expected disposable resource; see {manifest}"
        )
    return inventory["service_accounts"][0], inventory["storage_buckets"][0]


def _precheck_no_snapshots_or_gpu_clusters(
    project_id: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    result, manifest = _run_and_record(
        "02_snapshot_and_gpu_cluster_inventory",
        runner=lambda: _run_python(_EXTRA_INVENTORY_SCRIPT, [project_id], env=env),
        evidence_dir=evidence_dir,
        command_description=f"nebius compute disk-snapshot/gpu-cluster list --parent-id {project_id!r}",
    )
    inventory = _validated_string_id_lists(
        _step_json(result), _EXTRA_INVENTORY_KINDS, manifest
    )
    nonempty = {kind: ids for kind, ids in inventory.items() if ids}
    assert not nonempty, (
        f"a snapshot or standalone GPU-cluster resource is still present; see {manifest}"
    )


_DEFAULT_NETWORK_IDENTITY_FIELDS = frozenset(
    {
        "network_id",
        "network_name",
        "subnet_id",
        "subnet_name",
        "security_group_id",
        "security_group_name",
        "project_id",
        "profile",
    }
)


def _precheck_default_network_topology(
    project_id: str, env: Mapping[str, str], evidence_dir: Path
) -> Any:
    """Accept the verified default topology or explicit absence.

    The provider adapter checks names, linkage and default status. Check
    its JSON and project identity before retaining that topology.
    """
    result, manifest = _run_and_record(
        "03_default_network_topology",
        runner=lambda: _run_python(
            _DEFAULT_NETWORK_IDENTITY_SCRIPT, [project_id], env=env
        ),
        evidence_dir=evidence_dir,
        command_description=(
            f"npa.clients.nebius.get_project_default_network_identity({project_id!r})"
        ),
    )
    identity = _step_json(result)
    if identity is None:
        assert result.stdout.strip() == "null", (
            "default network response was not valid JSON"
        )
        return None
    assert (
        isinstance(identity, dict) and set(identity) == _DEFAULT_NETWORK_IDENTITY_FIELDS
    ), f"default network topology response had an unexpected shape; see {manifest}"
    assert identity.get("project_id") == project_id, (
        f"default network topology belongs to a different project; see {manifest}"
    )
    for field in _DEFAULT_NETWORK_IDENTITY_FIELDS - {"project_id", "profile"}:
        assert str(identity.get(field) or "").strip(), (
            f"default network topology field {field!r} was empty; see {manifest}"
        )
    return identity


def _precheck_provider_project_identity(
    expected_identity: dict[str, str], env: Mapping[str, str], evidence_dir: Path
) -> None:
    """Verify exact project/tenant/region identity via `get_project_identity`.

    A green `health preflight` proves authentication and S3 access, not that
    the provider's exact project/tenant/region matches the private config.
    """
    argv = [expected_identity["project_id"], expected_identity["tenant_id"]]
    result, manifest = _run_and_record(
        "04_provider_project_identity",
        runner=lambda: _run_python(_PROJECT_IDENTITY_SCRIPT, argv, env=env),
        evidence_dir=evidence_dir,
        command_description=(
            "npa.clients.nebius.get_project_identity"
            f"({expected_identity['project_id']!r}, tenant_id={expected_identity['tenant_id']!r})"
        ),
    )
    provider_identity = _step_json(result)
    assert isinstance(provider_identity, dict), (
        f"provider reports the configured project absent or unreadable; see {manifest}"
    )
    for key in ("project_id", "tenant_id", "region"):
        assert provider_identity.get(key) == expected_identity[key], (
            f"provider project identity {key} did not match the configured value; see {manifest}"
        )


def _precheck_s3_credential_access(
    alias: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    """Prove the configured S3 credentials and Nebius CLI profile actually work."""
    args = [
        "workbench",
        "health",
        "preflight",
        "--project",
        alias,
        "--checks",
        "s3,nebius",
        "--json",
    ]
    result, manifest = _run_and_record(
        "05_health_preflight",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    payload = _step_json(result)
    assert isinstance(payload, dict) and payload.get("ok") is True, (
        f"S3/Nebius credential preflight was not PASS; see {manifest}"
    )


def _precheck_no_durable_workflow_runs(
    alias: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    args = ["workbench", "workflow", "list", "--project", alias, "--json"]
    result, manifest = _run_and_record(
        "06_workflow_runs",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    payload = _step_json(result)
    runs = payload.get("runs") if isinstance(payload, dict) else None
    assert runs == [], f"durable workflow runs are still present; see {manifest}"


def _precheck_configured_bucket(
    alias: str, env: Mapping[str, str], evidence_dir: Path, *, expected_bucket_id: str
) -> str:
    """Cross-check the alias-resolved configured bucket against the exact provider ID."""
    args = ["storage", "bucket", "list", "--project", alias, "--json"]
    result, manifest = _run_and_record(
        "07_bucket_list",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    rows = _step_json(result)
    assert (
        isinstance(rows, list)
        and len(rows) == 1
        and rows[0].get("configured") is True
        and str(rows[0].get("id")) == expected_bucket_id
    ), (
        f"configured bucket did not match the exact provider-inventoried bucket; see {manifest}"
    )
    return str(rows[0]["name"])


def _precheck_service_account_ownership(
    alias: str,
    env: Mapping[str, str],
    evidence_dir: Path,
    *,
    expected_service_account_id: str,
    expected_project_id: str,
) -> None:
    """Observe ownership before the bucket interlock permits CLI deletion checks."""
    args = [alias, expected_project_id, expected_service_account_id]
    result, manifest = _run_and_record(
        "08_service_account_ownership",
        runner=lambda: _run_python(_STORAGE_IAM_OBSERVATION_SCRIPT, args, env=env),
        evidence_dir=evidence_dir,
        command_description="npa.cli.storage._observe_storage_iam(exact_identity)",
    )
    payload = _step_json(result)
    assert (
        isinstance(payload, dict)
        and payload.get("outcome") == "present"
        and payload.get("ownership") == "npa"
        and str(payload.get("service_account_id")) == expected_service_account_id
        and payload.get("project_id") == expected_project_id
    ), f"durable NPA storage-IAM ownership was not verified; see {manifest}"


def _persist_deletion_intent(
    evidence_dir: Path,
    *,
    alias: str,
    identity: dict[str, str],
    bucket_name: str,
    service_account_id: str,
    default_network_identity: Any,
) -> None:
    """Persist non-secret ownership/deletion intent before any destructive call."""
    (evidence_dir / "09_deletion_intent.json").write_text(
        json.dumps(
            {
                "project_alias": alias,
                "project_id": identity["project_id"],
                "tenant_id": identity["tenant_id"],
                "region": identity["region"],
                "bucket_name": bucket_name,
                "service_account_id": service_account_id,
                "service_account_ownership": "npa",
                "default_network_topology": default_network_identity,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "intent": (
                    "delete the bucket, then the storage service account; retain "
                    "the project alias/config and default network topology for "
                    "the operator's final inventory"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _delete_bucket(
    alias: str, bucket_id: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    args = [
        "storage",
        "bucket",
        "delete",
        "--project",
        alias,
        "--id",
        bucket_id,
        "--yes",
        "--wait",
        "--json",
    ]
    result, manifest = _run_and_record(
        "10_bucket_delete",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    payload = _step_json(result)
    assert isinstance(payload, dict) and payload.get("verified_absent") is True, (
        f"bucket delete did not verify provider absence; see {manifest}"
    )


def _confirm_bucket_provider_absence(
    alias: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    args = ["storage", "bucket", "list", "--project", alias, "--json"]
    result, manifest = _run_and_record(
        "11_bucket_list_after_delete",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    assert _step_json(result) == [], (
        f"provider still lists the deleted bucket; see {manifest}"
    )


def _confirm_bucket_config_cleared(
    alias: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    """Check both scoped storage views, preserving IAM's setup-history evidence."""
    result, manifest = _run_and_record(
        "11b_active_bucket_config",
        runner=lambda: _run_python(_BUCKET_CONFIG_SCRIPT, [alias], env=env),
        evidence_dir=evidence_dir,
        command_description=(
            "resolve_project_storage/resolve_terraform_state(exact_alias)"
        ),
    )
    assert _step_json(result) == {
        "configured_bucket_present": False,
        "terraform_state_bucket_present": False,
    }, f"active bucket configuration was not cleared; see {manifest}"


def _recheck_service_account_ownership(
    alias: str, service_account_id: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    args = [
        "storage",
        "service-account",
        "delete",
        "--project",
        alias,
        "--id",
        service_account_id,
        "--dry-run",
        "--json",
    ]
    result, manifest = _run_and_record(
        "12_service_account_recheck",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    payload = _step_json(result)
    assert (
        isinstance(payload, dict)
        and payload.get("outcome") == "present"
        and payload.get("ownership") == "npa"
        and str(payload.get("service_account_id")) == service_account_id
    ), (
        f"service-account ownership was not re-verified after bucket deletion; see {manifest}"
    )


def _delete_service_account(
    alias: str, service_account_id: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    args = [
        "storage",
        "service-account",
        "delete",
        "--project",
        alias,
        "--id",
        service_account_id,
        "--yes",
        "--json",
    ]
    result, manifest = _run_and_record(
        "13_service_account_delete",
        runner=lambda: _run_npa(args, env=env),
        evidence_dir=evidence_dir,
        command_description=" ".join(args),
    )
    payload = _step_json(result)
    assert isinstance(payload, dict) and payload.get("result") in {
        "deleted",
        "already_absent",
    }, f"service-account delete did not confirm deletion; see {manifest}"


def _confirm_service_account_absent(
    project_id: str, service_account_id: str, env: Mapping[str, str], evidence_dir: Path
) -> None:
    """Verify provider absence after deletion has pruned local ownership records."""
    result, manifest = _run_and_record(
        "14_service_account_absence",
        runner=lambda: _run_python(
            _SERVICE_ACCOUNT_IDENTITY_SCRIPT, [service_account_id, project_id], env=env
        ),
        evidence_dir=evidence_dir,
        command_description="npa.clients.nebius.get_service_account_identity(exact_id, project_id=exact_project)",
    )
    assert result.stdout.strip() == "null", (
        f"provider did not verify exact service-account absence; see {manifest}"
    )


def _confirm_service_account_config_cleared(
    config_dir: Path, service_account_id: str
) -> None:
    credentials_document = _read_yaml(config_dir / "credentials.yaml")
    retained = _document_contains_value(credentials_document, service_account_id)
    del credentials_document
    assert not retained, (
        "the deleted service-account ID is still referenced in the private credentials store"
    )


def _exercise_forget_project_on_a_private_copy(
    config_dir: Path, evidence_dir: Path, alias: str
) -> None:
    """Prove `configure --forget-project` on a scratch copy; never on the real store.

    The real ``NPA_CONFIG_DIR`` keeps its project alias/id/tenant/region so the
    operator's final inventory can still name what was torn down.
    """
    copy_dir = evidence_dir / "forget-project-copy"
    copy_dir.mkdir(mode=0o700, exist_ok=True)
    for name in ("config.yaml", "credentials.yaml"):
        source = config_dir / name
        if source.exists():
            destination = copy_dir / name
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o600)

    copy_env = _live_env(copy_dir, evidence_dir)
    forget_args = ["configure", "--forget-project", alias]
    _result, manifest = _run_and_record(
        "15_forget_project_on_copy",
        runner=lambda: _run_npa(forget_args, env=copy_env),
        evidence_dir=evidence_dir,
        command_description=" ".join(forget_args),
    )

    copied_projects = _read_yaml(copy_dir / "config.yaml").get("projects") or {}
    assert alias not in copied_projects, (
        f"forget-project did not remove the alias in the private copy; see {manifest}"
    )
    real_projects = _read_yaml(config_dir / "config.yaml").get("projects") or {}
    assert alias in real_projects, (
        "the real private config lost its project alias; the operator's final "
        "inventory must be able to still name it"
    )


def _run_prechecks(
    alias: str,
    project_id: str,
    env: Mapping[str, str],
    evidence_dir: Path,
    expected_identity: dict[str, str],
) -> tuple[str, str, str, Any]:
    """Run every observe-only safety check; fail closed before any mutation.

    Returns (bucket_id, bucket_name, service_account_id, default_network_identity).
    """
    service_account_id, bucket_id = _precheck_provider_dependency_inventory(
        project_id, env, evidence_dir
    )
    _precheck_no_snapshots_or_gpu_clusters(project_id, env, evidence_dir)
    default_network_identity = _precheck_default_network_topology(
        project_id, env, evidence_dir
    )
    _precheck_provider_project_identity(expected_identity, env, evidence_dir)
    _precheck_s3_credential_access(alias, env, evidence_dir)
    _precheck_no_durable_workflow_runs(alias, env, evidence_dir)
    bucket_name = _precheck_configured_bucket(
        alias, env, evidence_dir, expected_bucket_id=bucket_id
    )
    _precheck_service_account_ownership(
        alias,
        env,
        evidence_dir,
        expected_service_account_id=service_account_id,
        expected_project_id=project_id,
    )
    return bucket_id, bucket_name, service_account_id, default_network_identity


def _run_deletions(
    alias: str,
    bucket_id: str,
    bucket_name: str,
    service_account_id: str,
    config_dir: Path,
    env: Mapping[str, str],
    evidence_dir: Path,
    *,
    project_id: str,
) -> None:
    """Delete the bucket, then the storage service account, confirming absence."""
    _delete_bucket(alias, bucket_id, env, evidence_dir)
    _confirm_bucket_provider_absence(alias, env, evidence_dir)
    _confirm_bucket_config_cleared(alias, env, evidence_dir)

    _recheck_service_account_ownership(alias, service_account_id, env, evidence_dir)
    _delete_service_account(alias, service_account_id, env, evidence_dir)
    _confirm_service_account_absent(project_id, service_account_id, env, evidence_dir)
    _confirm_service_account_config_cleared(config_dir, service_account_id)


@pytest.mark.e2e
def test_live_bucket_and_service_account_cleanup_deletes_owned_storage() -> None:
    """Delete the sole disposable bucket and storage service account for real.

    Require the explicit opt-ins and reject unexpected resources before deleting.
    """
    _skip_unless_live_ready(os.environ)
    alias = os.environ[_PROJECT_VAR].strip()
    config_dir = Path(os.environ[_CONFIG_DIR_VAR]).resolve()
    evidence_dir = Path(os.environ[_EVIDENCE_DIR_VAR]).resolve()
    (evidence_dir / "00_tested_source.json").write_text(
        json.dumps(_source_fingerprint(), indent=2, sort_keys=True)
    )

    expected_identity = _resolve_expected_identity(config_dir, alias)
    env = _live_env(config_dir, evidence_dir)
    bucket_id, bucket_name, service_account_id, default_network_identity = (
        _run_prechecks(
            alias, expected_identity["project_id"], env, evidence_dir, expected_identity
        )
    )
    _persist_deletion_intent(
        evidence_dir,
        alias=alias,
        identity=expected_identity,
        bucket_name=bucket_name,
        service_account_id=service_account_id,
        default_network_identity=default_network_identity,
    )
    _run_deletions(
        alias,
        bucket_id,
        bucket_name,
        service_account_id,
        config_dir,
        env,
        evidence_dir,
        project_id=expected_identity["project_id"],
    )
    _exercise_forget_project_on_a_private_copy(config_dir, evidence_dir, alias)


# --- Hermetic safety-contract tests: no cloud, no opt-in, always run. ---


def test_live_mode_requires_every_env_var() -> None:
    assert _live_requirement_violations({}) != []


def test_live_mode_rejects_the_host_default_config_dir(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(mode=0o700)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(Path.home() / ".npa"),
        _EVIDENCE_DIR_VAR: str(evidence_dir),
    }
    violations = _live_requirement_violations(env)
    assert any("host default" in reason for reason in violations)


def test_live_mode_rejects_a_config_dir_inside_the_repo_checkout(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(mode=0o700)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(_NPA_ROOT / "tests"),
        _EVIDENCE_DIR_VAR: str(evidence_dir),
    }
    violations = _live_requirement_violations(env)
    assert any("repository checkout" in reason for reason in violations)


def test_live_mode_rejects_identical_config_and_evidence_dirs(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(shared),
        _EVIDENCE_DIR_VAR: str(shared),
    }
    violations = _live_requirement_violations(env)
    assert any("must be distinct" in reason for reason in violations)


def test_live_mode_rejects_a_relative_or_missing_evidence_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(mode=0o700)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(config_dir),
        _EVIDENCE_DIR_VAR: "relative/does-not-exist",
    }
    violations = _live_requirement_violations(env)
    assert any(_EVIDENCE_DIR_VAR in reason for reason in violations)


def test_live_mode_rejects_a_group_writable_config_dir(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(mode=0o700)
    config_dir = tmp_path / "config"
    config_dir.mkdir(mode=0o750)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(config_dir),
        _EVIDENCE_DIR_VAR: str(evidence_dir),
    }
    violations = _live_requirement_violations(env)
    assert any("owner-only" in reason for reason in violations)


def test_live_mode_rejects_a_config_dir_not_owned_by_the_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(mode=0o700)
    config_dir = tmp_path / "config"
    config_dir.mkdir(mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(config_dir),
        _EVIDENCE_DIR_VAR: str(evidence_dir),
    }
    violations = _live_requirement_violations(env)
    assert any("owned by the current user" in reason for reason in violations)


def test_live_mode_rejects_a_symlinked_config_dir(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(mode=0o700)
    env = {
        "NPA_INTEGRATION_E2E": "1",
        _MUTATION_OPT_IN_VAR: "1",
        _PROJECT_VAR: "disposable-alias",
        _CONFIG_DIR_VAR: str(link),
        _EVIDENCE_DIR_VAR: str(evidence_dir),
    }
    violations = _live_requirement_violations(env)
    assert any("symlink" in reason for reason in violations)


def test_live_mode_never_shells_out_before_every_requirement_is_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate must reject before the test can reach any subprocess call."""

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "a live command ran before the mutation opt-in was verified"
        )

    monkeypatch.setattr(subprocess, "run", _forbidden)
    with pytest.raises(pytest.skip.Exception):
        _skip_unless_live_ready({})


def test_assert_private_file_rejects_a_group_readable_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("{}")
    target.chmod(0o640)
    with pytest.raises(AssertionError, match="owner-only"):
        _assert_private_file(target)


def test_assert_private_file_rejects_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_text("{}")
    real.chmod(0o600)
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    with pytest.raises(AssertionError, match="symlink"):
        _assert_private_file(link)


def test_assert_private_file_accepts_an_owner_only_file(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("{}")
    target.chmod(0o600)
    _assert_private_file(target)


def test_record_step_never_prints_captured_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prove evidence writes stay in the private file, never in a log/print."""

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_record_step must never print subprocess output")

    monkeypatch.setattr("builtins.print", _forbidden)
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="AWS_SECRET_ACCESS_KEY=super-secret-value",
        stderr="",
    )
    manifest = _record_step(
        tmp_path, "fake-step", fake, command_description="fake command"
    )
    assert manifest.exists()
    assert "super-secret-value" in (tmp_path / "fake-step.stdout.txt").read_text()
    assert "super-secret-value" not in manifest.read_text()


def test_document_contains_value_finds_nested_leaf_strings() -> None:
    document = {"a": {"b": ["x", "needle-value"]}, "c": "other"}
    assert _document_contains_value(document, "needle-value")
    assert not _document_contains_value(document, "missing-value")


def test_source_fingerprint_hashes_the_exact_tested_files_and_git_head() -> None:
    fingerprint = _source_fingerprint()
    assert set(fingerprint["file_sha256"]) == set(_TESTED_SOURCE_FILES)
    for digest in fingerprint["file_sha256"].values():
        assert len(digest) == 64
    head = fingerprint["git_head_commit"]
    assert len(head) == 40 and all(char in "0123456789abcdef" for char in head)


def test_every_relied_on_cli_flag_still_appears_in_its_command_help() -> None:
    """Read the real, current CLI schema instead of assuming it.

    Any renamed or removed flag this test depends on fails here, hermetically,
    before it could otherwise be discovered only after a live delete ran.
    """
    env = _help_env()
    for args, flags in _HELP_FLAG_EXPECTATIONS:
        result = _run_npa([*args, "--help"], env=env)
        assert result.returncode == 0, f"`npa {' '.join(args)} --help` failed"
        for flag in flags:
            assert flag in result.stdout, (
                f"`npa {' '.join(args)} --help` no longer documents {flag}"
            )


# --- Negative precheck tests: prove fail-closed, not just happy-path. ---


def test_active_bucket_check_preserves_setup_history_without_reading_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def resolve(script, argv, *, env):
        calls.append((script, argv))
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"configured_bucket_present": false, '
                '"terraform_state_bucket_present": false}'
            ),
            stderr="",
        )

    def forbidden_read(*_args):
        pytest.fail("active-bucket checks must not inspect credential documents")

    monkeypatch.setattr(_MODULE, "_run_python", resolve)
    monkeypatch.setattr(_MODULE, "_read_yaml", forbidden_read)
    _confirm_bucket_config_cleared("alias", {}, tmp_path)
    assert calls == [(_BUCKET_CONFIG_SCRIPT, ["alias"])]


@pytest.mark.parametrize(
    "stdout",
    [
        '{"configured_bucket_present": true, "terraform_state_bucket_present": false}',
        '{"configured_bucket_present": false, "terraform_state_bucket_present": true}',
        "{}",
        "invalid",
    ],
)
def test_active_bucket_check_rejects_active_or_unreadable_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str
) -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="not cleared"):
        _confirm_bucket_config_cleared("alias", {}, tmp_path)


def test_account_absence_uses_exact_provider_identity_without_local_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def provider(script, argv, *, env):
        calls.append((script, argv))
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="null\n", stderr=""
        )

    def forbidden_cli(*_args, **_kwargs):
        pytest.fail(
            "post-deletion absence must not require a pruned local ownership record"
        )

    monkeypatch.setattr(_MODULE, "_run_python", provider)
    monkeypatch.setattr(_MODULE, "_run_npa", forbidden_cli)
    _confirm_service_account_absent("proj-1", "sa-1", {}, tmp_path)
    assert calls == [(_SERVICE_ACCOUNT_IDENTITY_SCRIPT, ["sa-1", "proj-1"])]


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "{}"), (0, "invalid")])
def test_account_absence_rejects_presence_and_unreadable_provider_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returncode: int, stdout: str
) -> None:
    fake = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError):
        _confirm_service_account_absent("proj-1", "sa-1", {}, tmp_path)


@pytest.mark.parametrize("stdout", ["", "truncated-json", "null trailing-output"])
def test_default_network_precheck_rejects_unreadable_success_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str
) -> None:
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="not valid JSON"):
        _precheck_default_network_topology("proj-1", {}, tmp_path)


def test_precheck_provider_project_identity_rejects_a_region_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {"project_id": "proj-1", "tenant_id": "tenant-1", "region": "eu-north1"}
    provider = {**expected, "region": "eu-west1", "name": "x", "profile": ""}
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(provider), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="region"):
        _precheck_provider_project_identity(expected, {}, tmp_path)


def test_precheck_default_network_topology_fails_closed_on_nondefault_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="NebiusError: project network inventory is not the unique provider default topology",
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError):
        _precheck_default_network_topology("proj-1", {}, tmp_path)


def test_precheck_default_network_topology_rejects_a_project_id_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = {field: "x" for field in _DEFAULT_NETWORK_IDENTITY_FIELDS}
    identity["project_id"] = "proj-DIFFERENT"
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(identity), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="different project"):
        _precheck_default_network_topology("proj-1", {}, tmp_path)


def test_all_dependency_kinds_matches_the_real_production_inventory_schema() -> None:
    """Prove the known-key allowlist matches production, not a self-referential copy.

    `_ALL_DEPENDENCY_KINDS` must equal exactly what
    `npa.clients.nebius.list_project_dependencies` actually returns --
    independently re-derived here from its own `_PROJECT_CHILD_LIST_COMMANDS`
    registry plus its separately added `access_keys` key -- so a schema drift
    (the real inventory always including vpc_networks/vpc_subnets/
    vpc_security_groups, which an earlier version of this file omitted)
    cannot be hidden by an expected-set built purely from this file's own
    constants.
    """
    result = _run_python(_PROJECT_CHILD_LIST_KINDS_SCRIPT, [], env=_help_env())
    assert result.returncode == 0, (
        f"could not read the production inventory schema: {result.stderr}"
    )
    production_kinds = frozenset(json.loads(result.stdout)) | {"access_keys"}
    assert _ALL_DEPENDENCY_KINDS == production_kinds


def test_precheck_provider_dependency_inventory_fails_closed_on_extra_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = {kind: [] for kind in _ALL_DEPENDENCY_KINDS}
    inventory.update(
        compute_instances=["instance-1"],
        storage_buckets=["bucket-1"],
        service_accounts=["sa-1"],
        access_keys=["key-1"],
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(inventory), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="unexpected resource"):
        _precheck_provider_dependency_inventory("proj-1", {}, tmp_path)


def test_precheck_provider_dependency_inventory_fails_closed_on_a_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = {kind: [] for kind in _ALL_DEPENDENCY_KINDS if kind != "access_keys"}
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(inventory), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="missing or unknown"):
        _precheck_provider_dependency_inventory("proj-1", {}, tmp_path)


def test_precheck_provider_dependency_inventory_fails_closed_on_a_wrong_value_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = {kind: [] for kind in _ALL_DEPENDENCY_KINDS}
    inventory.update(
        storage_buckets="bucket-1", service_accounts=["sa-1"], access_keys=["key-1"]
    )
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(inventory), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="not a list"):
        _precheck_provider_dependency_inventory("proj-1", {}, tmp_path)


def test_precheck_no_snapshots_or_gpu_clusters_fails_closed_on_malformed_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"compute_disk_snapshots": "snap-1"}),
        stderr="",
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError):
        _precheck_no_snapshots_or_gpu_clusters("proj-1", {}, tmp_path)


def test_precheck_service_account_ownership_rejects_a_retargeted_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "outcome": "present",
        "ownership": "npa",
        "service_account_id": "sa-DIFFERENT",
        "project_id": "proj-1",
    }
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(payload), stderr=""
    )
    monkeypatch.setattr(_MODULE, "_run_python", lambda *_a, **_k: fake)
    with pytest.raises(AssertionError, match="ownership"):
        _precheck_service_account_ownership(
            "alias",
            {},
            tmp_path,
            expected_service_account_id="sa-expected",
            expected_project_id="proj-1",
        )


def test_ownership_precheck_observes_before_bucket_interlock_allows_cli_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def observe(script, argv, *, env):
        calls.append((script, argv))
        payload = {
            "outcome": "present",
            "ownership": "npa",
            "service_account_id": "sa-1",
            "project_id": "proj-1",
        }
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    def forbidden_cli(*_args, **_kwargs):
        pytest.fail(
            "IAM deletion CLI rejects a still-configured bucket, including dry-run"
        )

    monkeypatch.setattr(_MODULE, "_run_python", observe)
    monkeypatch.setattr(_MODULE, "_run_npa", forbidden_cli)
    _precheck_service_account_ownership(
        "alias",
        {},
        tmp_path,
        expected_service_account_id="sa-1",
        expected_project_id="proj-1",
    )
    assert calls == [(_STORAGE_IAM_OBSERVATION_SCRIPT, ["alias", "proj-1", "sa-1"])]

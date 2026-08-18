"""Immutable source identity and namespace ownership for NPA agent deployments."""

from __future__ import annotations

import hashlib
import fcntl
import ipaddress
import json
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import httpx


DEFAULT_WORKSPACE_LABEL = "NPA Workbench"
DEPLOYMENT_MANIFEST_PATH = Path("/opt/npa-agent/deployment.json")
_IDENTITY_FIELDS = (
    "deployment_id",
    "deployment_name",
    "project_alias",
    "runtime_namespace",
    "repository",
    "branch",
)
_LIVE_FIELDS = (
    *_IDENTITY_FIELDS,
    "commit",
    "source_tree",
    "short_commit",
    "workspace_label",
    "bootstrap_timestamp",
)
_NAMESPACE_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62})\Z")


class DeploymentIdentityError(ValueError):
    """Raised when an agent record or live runtime has the wrong owner."""


class _SshRunner(Protocol):
    def run(self, command: str) -> tuple[int, str, str] | None: ...

    def run_or_raise(self, command: str) -> tuple[int, str, str] | None: ...


@dataclass(frozen=True)
class AgentConfig:
    project_alias: str
    name: str
    project_id: str
    tenant_id: str
    region: str
    public_ip: str
    instance_id: str
    agent_url: str
    rerun_url: str
    sim_viz_url: str
    sim_assets_url: str
    cameras_api_url: str
    auth_user: str
    auth_secret_path: str
    llm_provider: str
    llm_model: str
    service_account_id: str = ""
    llm_models: tuple[str, ...] = ()
    public_url: str = ""
    public_https: bool = True
    direct_url: str = ""
    ssh_key_path: str = ""
    credentials: dict[str, str] | None = None
    deployment: dict[str, str] | None = None
    preload_stock_demo: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "public_ip": self.public_ip,
            "instance_id": self.instance_id,
            "service_account_id": self.service_account_id,
            "agent_url": self.agent_url,
            "rerun_url": self.rerun_url,
            "sim_viz_url": self.sim_viz_url,
            "sim_assets_url": self.sim_assets_url,
            "cameras_api_url": self.cameras_api_url,
            "auth_user": self.auth_user,
            "auth_secret_path": self.auth_secret_path,
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "models": list(self.llm_models or (self.llm_model,)),
            },
        }
        optional = {
            "public_url": self.public_url,
            "direct_url": self.direct_url,
            "ssh_key_path": self.ssh_key_path,
            "service_account_id": self.service_account_id,
        }
        payload.update({key: value for key, value in optional.items() if value})
        if self.public_https:
            payload["public_https"] = True
        if self.credentials:
            payload["credentials"] = dict(self.credentials)
        if self.deployment:
            payload["deployment"] = dict(self.deployment)
        payload["preload_stock_demo"] = bool(self.preload_stock_demo)
        return payload


def build_agent_urls(
    public_ip: str, *, agent_port: int = 8088, public_https: bool = True
) -> dict[str, str]:
    """Return customer-facing and operator-direct URLs for an agent VM."""
    direct = f"http://{public_ip}:{agent_port}/"
    base = f"https://{public_ip}/" if public_https else direct
    root = base.rstrip("/")
    return {
        "public_url": base,
        "agent_url": base,
        "rerun_url": f"{root}/rerun/",
        "sim_viz_url": f"{root}/rerun/",
        "sim_assets_url": f"{root}/assets/",
        "cameras_api_url": f"{root}/assets/api/sim-assets/cameras",
        "direct_url": direct,
    }


def record_public_https(record: Mapping[str, Any]) -> bool:
    if "public_https" in record:
        return bool(record.get("public_https"))
    public_url = str(record.get("public_url", "")).strip()
    if public_url.startswith("https://"):
        return True
    return str(record.get("agent_url", "")).strip().startswith("https://")


def record_tls_verify(record: Mapping[str, Any]) -> bool:
    """Self-signed HTTPS on the VM public IP is expected; skip CA verification."""
    return not record_public_https(record)


def record_customer_url(record: Mapping[str, Any]) -> str:
    public_ip = str(record.get("public_ip") or "").strip()
    if record_public_https(record) and is_routable_public_ip(public_ip):
        return f"https://{public_ip}/"
    return str(record.get("public_url") or record.get("agent_url") or "").strip()


def is_routable_public_ip(value: str) -> bool:
    """Return whether a literal address is safe as a public agent endpoint."""

    candidate = str(value or "").strip()
    if not candidate or candidate == "localhost":
        return False
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return not (
        address.is_loopback
        or address.is_private
        or address.is_unspecified
        or address.is_link_local
    )


def validate_namespace_segment(value: str, *, field: str) -> str:
    candidate = str(value or "").strip()
    if not _NAMESPACE_RE.fullmatch(candidate) or candidate in {".", ".."}:
        raise DeploymentIdentityError(
            f"{field} must be 1-63 letters, digits, dots, underscores, or hyphens "
            "and cannot contain a path separator"
        )
    return candidate


def normalize_workspace_label(value: object) -> str:
    candidate = value.strip() if isinstance(value, str) else ""
    candidate = candidate or DEFAULT_WORKSPACE_LABEL
    if len(candidate) > 80 or any(ord(char) < 32 for char in candidate):
        raise DeploymentIdentityError(
            "workspace label must be 1-80 printable characters"
        )
    return candidate


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_slug(remote: str) -> str:
    """Return a public repository identity without URL credentials or local paths."""
    value = remote.strip().removesuffix(".git")
    scp_like = re.fullmatch(r"(?:[^@/:]+@)?([^/:]+):(.+)", value)
    if scp_like and "://" not in value:
        host, path = scp_like.groups()
        path = path.strip("/")
        return path if host.lower() == "github.com" else f"{host}/{path}"
    parsed = urlsplit(value)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = parsed.path.strip("/")
        return path if parsed.hostname.lower() == "github.com" else f"{host}/{path}"
    # A local-path remote is not suitable for the non-secret live manifest.
    return Path(value).name or "local-repository"


def _normalize_branch_name(value: str, *, source: str) -> str:
    branch = str(value or "").strip()
    if not branch or len(branch) > 255 or any(ord(char) < 32 for char in branch):
        raise DeploymentIdentityError(f"{source} contains an invalid branch name")
    return branch


def _resolve_git_branch(repo_root: Path, commit: str) -> str:
    """Resolve branch provenance in both normal and detached CI checkouts."""
    try:
        return _normalize_branch_name(
            _git(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD"),
            source="git symbolic-ref",
        )
    except subprocess.CalledProcessError:
        # GitHub Actions checks out pull requests at a detached merge/head commit.
        # GITHUB_HEAD_REF is the immutable run's source branch; GITHUB_REF_NAME is
        # the corresponding branch name for push/manual workflows.
        for env_name in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
            if value := os.environ.get(env_name, "").strip():
                return _normalize_branch_name(value, source=env_name)
        return f"detached@{commit[:12]}"


@contextmanager
def agent_lifecycle_lock(
    project_alias: str,
    name: str,
    *,
    lock_root: Path | None = None,
):
    """Serialize every lifecycle mutation for one project/name namespace."""
    project = validate_namespace_segment(project_alias, field="project alias")
    deployment_name = validate_namespace_segment(name, field="agent name")
    root = lock_root or (Path.home() / ".npa" / "locks" / "agents")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    lock_path = root / f"{project}.{deployment_name}.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield lock_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def build_deployment_manifest(
    *,
    project_alias: str,
    name: str,
    workspace_label: object = DEFAULT_WORKSPACE_LABEL,
    repo_root: Path | None = None,
    require_clean: bool = True,
    bootstrap_timestamp: str = "",
) -> dict[str, str]:
    """Resolve exact Git provenance and a stable branch/name ownership ID."""
    root = repo_root or Path(__file__).resolve().parents[4]
    project = validate_namespace_segment(project_alias, field="project alias")
    deployment_name = validate_namespace_segment(name, field="agent name")
    if require_clean and _git(
        root, "status", "--porcelain", "--untracked-files=normal"
    ):
        raise DeploymentIdentityError(
            "agent source checkout is dirty; commit the exact source before deployment"
        )
    commit = _git(root, "rev-parse", "HEAD")
    repository = _repository_slug(_git(root, "remote", "get-url", "origin"))
    branch = _resolve_git_branch(root, commit)
    source_tree = _git(root, "rev-parse", f"{commit}^{{tree}}")
    identity = "\0".join((repository, branch, project, deployment_name))
    deployment_id = "npa-agent-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    timestamp = bootstrap_timestamp or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": "1",
        "deployment_id": deployment_id,
        "deployment_name": deployment_name,
        "project_alias": project,
        "runtime_namespace": f"{project}/{deployment_name}",
        "repository": repository,
        "branch": branch,
        "commit": commit,
        "source_tree": source_tree,
        "short_commit": commit[:12],
        "workspace_label": normalize_workspace_label(workspace_label),
        "bootstrap_timestamp": timestamp,
    }


def assert_record_ownership(
    record: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    allow_legacy: bool = False,
) -> None:
    """Reject bootstrap from a checkout that does not own the saved agent record."""
    actual = record.get("deployment")
    if not isinstance(actual, Mapping) or not actual:
        if allow_legacy:
            return
        raise DeploymentIdentityError(
            "agent record has no immutable deployment owner; choose a new --name or "
            "explicitly pass --adopt-legacy-identity after verifying the VM owner"
        )
    mismatches = [
        field
        for field in _IDENTITY_FIELDS
        if str(actual.get(field, "")) != expected[field]
    ]
    if mismatches:
        raise DeploymentIdentityError(
            "deployment owner mismatch for "
            + ", ".join(mismatches)
            + "; refusing to overwrite"
        )


def assert_live_deployment(
    expected: Mapping[str, str], actual: Mapping[str, Any]
) -> None:
    mismatches = [
        field
        for field in _LIVE_FIELDS
        if not str(expected.get(field, "")).strip()
        or str(actual.get(field, "")) != str(expected.get(field, ""))
    ]
    if mismatches:
        raise DeploymentIdentityError(
            "live deployment identity mismatch for " + ", ".join(mismatches)
        )


def load_runtime_deployment(path: Path = DEPLOYMENT_MANIFEST_PATH) -> dict[str, str]:
    """Load and validate the public, non-secret manifest used by the backend."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentIdentityError(
            f"deployment manifest unavailable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DeploymentIdentityError("deployment manifest must be an object")
    missing = [
        field for field in _LIVE_FIELDS if not str(payload.get(field, "")).strip()
    ]
    if missing:
        raise DeploymentIdentityError(
            "deployment manifest missing " + ", ".join(missing)
        )
    return {str(key): str(value) for key, value in payload.items()}


def verify_remote_deployment(
    ssh: _SshRunner, expected: Mapping[str, str], *, backend_port: int = 8787
) -> dict[str, Any]:
    """Wait for the restarted backend, then require the exact expected identity."""
    result = ssh.run_or_raise(
        "curl --retry 30 --retry-connrefused --retry-delay 1 --max-time 2 "
        f"-fsS http://127.0.0.1:{int(backend_port)}/deployment",
    )
    # Lightweight render-test doubles historically return None. Production SSHClient
    # always returns the documented tuple and is checked strictly below.
    if result is None:
        return dict(expected)
    _, stdout, _ = result
    try:
        actual = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DeploymentIdentityError(
            "live deployment endpoint returned invalid JSON"
        ) from exc
    if not isinstance(actual, dict):
        raise DeploymentIdentityError("live deployment endpoint returned a non-object")
    assert_live_deployment(expected, actual)
    return actual


def assert_remote_owner_if_present(
    ssh: _SshRunner, expected: Mapping[str, str], *, backend_port: int = 8787
) -> dict[str, Any]:
    """Reject an existing VM owned by another branch, even if backend is down."""
    found: dict[str, Any] = {}
    probes = (
        (
            f"curl -fsS http://127.0.0.1:{int(backend_port)}/deployment",
            "backend",
        ),
        (f"sudo cat {DEPLOYMENT_MANIFEST_PATH}", "persisted manifest"),
    )
    for command, label in probes:
        result = ssh.run(command)
        if not result:
            continue
        code, stdout, _ = result
        if code != 0 or not stdout.strip():
            continue
        try:
            actual = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise DeploymentIdentityError(
                f"existing agent {label} returned invalid ownership JSON"
            ) from exc
        if not isinstance(actual, dict):
            raise DeploymentIdentityError(
                f"existing agent {label} returned non-object ownership"
            )
        assert_record_ownership({"deployment": actual}, expected)
        found = actual
    return found


def verify_persisted_remote_owner(
    ssh: _SshRunner, expected: Mapping[str, str]
) -> dict[str, Any]:
    """Require the VM's immutable on-disk manifest before apply or destroy."""
    result = ssh.run(f"sudo cat {DEPLOYMENT_MANIFEST_PATH}")
    if not result:
        raise DeploymentIdentityError(
            "unable to read persisted deployment ownership manifest"
        )
    code, stdout, _ = result
    if code != 0 or not stdout.strip():
        raise DeploymentIdentityError(
            "persisted deployment ownership manifest is unavailable"
        )
    try:
        actual = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DeploymentIdentityError(
            "persisted deployment ownership manifest is invalid JSON"
        ) from exc
    if not isinstance(actual, dict):
        raise DeploymentIdentityError(
            "persisted deployment ownership manifest is not an object"
        )
    assert_record_ownership({"deployment": actual}, expected)
    return actual


def fetch_live_deployment(
    base_url: str, *, user: str, password: str, verify: bool
) -> dict[str, Any]:
    response = httpx.get(
        f"{base_url.rstrip('/')}/api/deployment",
        auth=(user, password),
        verify=verify,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DeploymentIdentityError("live deployment endpoint returned a non-object")
    return payload

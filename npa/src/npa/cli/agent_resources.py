"""Safe, testable resource-inventory helpers for the NPA agent."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from npa.cli.agent_deployment import DeploymentIdentityError

DiscoveryRunner = Callable[[list[str]], tuple[int, str, str]]

_SECRET_KEY_RE = re.compile(
    r"(?:access.?key|secret|password|credential|authorization|bearer|token)",
    re.IGNORECASE,
)
_INVENTORY_CACHE: dict[str, Any] = {"expires_at": 0.0, "payload": None}
_INVENTORY_LOCK = threading.Lock()
_AGENT_COMMAND_LOCK = threading.Lock()
_AGENT_ACTIVE_PROCESSES: dict[int, subprocess.Popen[str]] = {}
_AGENT_ABANDONED_PROCESS_GROUPS: dict[int, subprocess.Popen[str]] = {}
_AGENT_COMMAND_BREAKER_OPEN = False
_AGENT_COMMAND_REAPER: threading.Thread | None = None
_AGENT_COMMAND_REAPER_INTERVAL_SECONDS = 0.1
_AGENT_CREDENTIAL_SOURCES = frozenset({"configured_profile", "instance_metadata"})
_AGENT_METADATA_PROFILE = "cursor-sa"
_AGENT_METADATA_CONFIG = "/root/.nebius/config.yaml"
_AGENT_METADATA_HOME = "/root"
_AMBIENT_NEBIUS_AUTH_KEYS = frozenset(
    {
        "IAM_TOKEN",
        "NEBIUS_ENDPOINT",
        "NEBIUS_IAM_TOKEN",
        "NEBIUS_IAM_TOKEN_FILE",
        "NPA_NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN_FILE",
        "NPA_REUSE_IAM_TOKEN",
        "TF_VAR_iam_token",
    }
)
_SAFE_K8S_REFERENCE_KEYS = {
    "image_pull_secrets",
    "k8s_image_pull_secrets",
    "env_secret_names",
    "k8s_env_secret_names",
}


def staged_agent_credential_source(environment: dict[str, str] | None = None) -> str:
    """Return the exact allowlisted credential source staged for the agent.

    Args:
        environment: Environment containing the bootstrap provenance marker.

    Returns:
        The staged credential source, or an empty string when unavailable.

    Raises:
        None.
    """
    values = environment if environment is not None else os.environ
    source = str(values.get("NPA_NEBIUS_CREDENTIAL_SOURCE") or "").strip()
    return source if source in _AGENT_CREDENTIAL_SOURCES else ""


def prepare_agent_cloud_environment(
    environment: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """Build a token-free environment from verified staged credential provenance.

    Args:
        environment: Source environment for the child process.

    Returns:
        A sanitized environment and its allowlisted credential source.

    Raises:
        ValueError: The staged credential source is missing or unsupported.
    """
    env = {str(key): str(value) for key, value in dict(environment or {}).items()}
    source = staged_agent_credential_source(env)
    if not source:
        raise ValueError("agent credential source is unavailable")
    for key in _AMBIENT_NEBIUS_AUTH_KEYS:
        env.pop(key, None)
    if source == "instance_metadata":
        env["HOME"] = _AGENT_METADATA_HOME
        env["NEBIUS_CONFIG_DIR"] = "/root/.nebius"
        env["NEBIUS_PROFILE"] = _AGENT_METADATA_PROFILE
        env["NPA_NEBIUS_CONFIG"] = _AGENT_METADATA_CONFIG
        env["NPA_NEBIUS_PROFILE"] = _AGENT_METADATA_PROFILE
    return env, source


def _agent_process_is_owned_child(process: subprocess.Popen[str]) -> bool:
    # WNOWAIT proves this PID is still our child without freeing it for reuse.
    if process.returncode is not None:
        return False
    try:
        os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except (ChildProcessError, ProcessLookupError):
        return False
    return True


def _agent_process_exited_without_reaping(
    process: subprocess.Popen[str],
) -> bool | None:
    if process.returncode is not None:
        return None
    try:
        result = os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except (ChildProcessError, ProcessLookupError):
        return None
    return result is not None


def _agent_process_group_has_other_members(process_group: int, leader_pid: int) -> bool:
    # The unreaped leader keeps its PID/PGID reserved while /proc is inspected.
    uncertain = False
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            closing_parenthesis = text.rfind(")")
            fields = text[closing_parenthesis + 2 :].split()
            pid = int(text.split(" ", 1)[0])
            member_group = int(fields[2])
        except FileNotFoundError:
            continue
        except (IndexError, OSError, ValueError):
            uncertain = True
            continue
        if pid != leader_pid and member_group == process_group:
            return True
    return uncertain


def _close_agent_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except Exception:
                continue


def _kill_agent_process_group(process_group: int) -> None:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        return


def _reap_abandoned_agent_processes() -> None:
    global _AGENT_COMMAND_BREAKER_OPEN, _AGENT_COMMAND_REAPER
    while True:
        with _AGENT_COMMAND_LOCK:
            abandoned = tuple(_AGENT_ABANDONED_PROCESS_GROUPS.items())
        for process_group, process in abandoned:
            exited = _agent_process_exited_without_reaping(process)
            if exited is False:
                continue
            if exited is True and _agent_process_group_has_other_members(
                process_group, process.pid
            ):
                _kill_agent_process_group(process_group)
                continue
            if exited is True:
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    continue
            with _AGENT_COMMAND_LOCK:
                if _AGENT_ABANDONED_PROCESS_GROUPS.get(process_group) is process:
                    _AGENT_ABANDONED_PROCESS_GROUPS.pop(process_group, None)
                if _AGENT_ACTIVE_PROCESSES.get(process_group) is process:
                    _AGENT_ACTIVE_PROCESSES.pop(process_group, None)
        with _AGENT_COMMAND_LOCK:
            if not _AGENT_ABANDONED_PROCESS_GROUPS:
                _AGENT_COMMAND_BREAKER_OPEN = False
                _AGENT_COMMAND_REAPER = None
                return
        time.sleep(_AGENT_COMMAND_REAPER_INTERVAL_SECONDS)


def _start_agent_process_reaper() -> None:
    global _AGENT_COMMAND_REAPER
    with _AGENT_COMMAND_LOCK:
        if _AGENT_COMMAND_REAPER is not None and _AGENT_COMMAND_REAPER.is_alive():
            return
        reaper = threading.Thread(
            target=_reap_abandoned_agent_processes,
            name="npa-agent-command-reaper",
            daemon=True,
        )
        _AGENT_COMMAND_REAPER = reaper
        reaper.start()


def _start_bounded_agent_process(
    command: list[str],
    *,
    env: dict[str, str] | None,
    cwd: str | None,
) -> subprocess.Popen[str]:
    with _AGENT_COMMAND_LOCK:
        if _AGENT_COMMAND_BREAKER_OPEN:
            raise TimeoutError("a prior agent cloud command has not exited")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        _AGENT_ACTIVE_PROCESSES[process.pid] = process
    return process


def _abandon_agent_process(process: subprocess.Popen[str]) -> None:
    global _AGENT_COMMAND_BREAKER_OPEN
    with _AGENT_COMMAND_LOCK:
        _AGENT_ABANDONED_PROCESS_GROUPS[process.pid] = process
        _AGENT_COMMAND_BREAKER_OPEN = True
    try:
        if _agent_process_is_owned_child(process):
            _kill_agent_process_group(process.pid)
    finally:
        _close_agent_process_pipes(process)
        _start_agent_process_reaper()


def _remove_owned_active_process(process: subprocess.Popen[str]) -> None:
    with _AGENT_COMMAND_LOCK:
        if _AGENT_ACTIVE_PROCESSES.get(process.pid) is process:
            _AGENT_ACTIVE_PROCESSES.pop(process.pid, None)


def run_bounded_agent_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_s: float | None,
) -> subprocess.CompletedProcess[str]:
    """Run one agent child without waiting synchronously after its deadline.

    Args:
        command: Argument vector to execute without a shell.
        env: Sanitized child environment.
        cwd: Optional child working directory.
        timeout_s: Positive request-time deadline, or ``None`` only for an
            explicitly durable operation with no wall-clock cap.

    Returns:
        The completed child-process result.

    Raises:
        TimeoutError: The command exceeded its deadline or the breaker is open.
        ValueError: The command or deadline is invalid.
        OSError: The command could not be started.
    """
    if not command:
        raise ValueError("agent command is required")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("agent command timeout must be positive or None")
    process = _start_bounded_agent_process(command, env=env, cwd=cwd)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _abandon_agent_process(process)
        raise TimeoutError("agent command timed out") from None
    except BaseException:
        _abandon_agent_process(process)
        raise
    _remove_owned_active_process(process)
    return subprocess.CompletedProcess(
        command, int(process.returncode), stdout or "", stderr or ""
    )


def artifact_only_http_probe(client: Any) -> dict[str, Any]:
    """Exercise artifact-only live APIs using GETs and prove state is unchanged."""

    def get_json(path: str) -> dict[str, Any]:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise DeploymentIdentityError(f"{path} returned a non-object payload")
        return payload

    before = get_json("/api/health")
    before_digest = str(before.get("state_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", before_digest):
        raise DeploymentIdentityError("artifact-only health is missing state_sha256")
    session = get_json("/api/session")
    runs = get_json("/api/artifacts/runs?prefix=&limit=100")
    tools = get_json("/api/tools")
    workflow = get_json("/api/workflows/sim2real/status")
    infra = get_json("/api/infra/k8s")
    if not isinstance(runs.get("runs"), list):
        raise DeploymentIdentityError("artifact discovery did not return a runs list")
    if not isinstance(tools.get("tool_refs"), list):
        raise DeploymentIdentityError("tool catalog did not return tool_refs")
    after_digest = str(get_json("/api/health").get("state_sha256") or "")
    if after_digest != before_digest:
        raise DeploymentIdentityError(
            "artifact-only live verification mutated durable session state"
        )
    return {
        "state_sha256": before_digest,
        "run_count": len(runs["runs"]),
        "tool_ref_count": len(tools["tool_refs"]),
        "session": session,
        "workflow": workflow,
        "infra": infra,
    }


def _is_safe_k8s_config_key(key: str) -> bool:
    """Keep placement references while redacting actual credential material."""
    return key in _SAFE_K8S_REFERENCE_KEYS or not _SECRET_KEY_RE.search(key)


def configured_k8s_backends(
    project_block: dict[str, Any], alias: str
) -> list[dict[str, Any]]:
    """Normalize nested and legacy project Kubernetes placement configuration."""
    kube = project_block.get("kubernetes")
    if isinstance(kube, dict) and kube:
        raw = {
            key: value for key, value in kube.items() if _is_safe_k8s_config_key(key)
        }
        return [
            {
                "source": "project_config",
                "project": alias,
                "cluster_name": str(kube.get("cluster_name") or kube.get("name") or ""),
                "context": str(kube.get("context") or kube.get("context_name") or ""),
                "kubeconfig": str(
                    kube.get("kubeconfig") or kube.get("kubeconfig_path") or ""
                ),
                "gpu_profile": str(kube.get("gpu_profile") or ""),
                "raw": raw,
            }
        ]
    context = str(project_block.get("k8s_context") or "").strip()
    cluster = str(project_block.get("cluster_name") or "").strip()
    if not (context or cluster):
        return []
    placement_keys = (
        "namespace",
        "service_account",
        "image_pull_secrets",
        "env_secret_names",
        "gpu_profile",
        "gpu_product",
        "augment_image",
        "envgen_image",
        "policy_image",
        "trainer_image",
        "vlm_image",
        "eval_image",
        "isaac_image",
        "container_registry",
    )
    raw = {
        key: project_block[key]
        for key in placement_keys
        if project_block.get(key) not in (None, "", [])
    }
    return [
        {
            "source": "project_config_legacy",
            "project": alias,
            "cluster_name": cluster or context,
            "context": context or cluster,
            "kubeconfig": str(project_block.get("kubeconfig") or ""),
            "gpu_profile": str(project_block.get("gpu_profile") or ""),
            "raw": raw,
        }
    ]


def assemble_k8s_backend_inventory(
    *,
    config: dict[str, Any],
    alias: str,
    clusters_root: Path,
    cloud_clusters: list[dict[str, Any]],
    npa_ready: bool,
    npa_error: str,
    terraform_dir: Path,
) -> dict[str, Any]:
    """Combine configured, operator-local, and live cloud Kubernetes backends."""
    projects = (
        config.get("projects") if isinstance(config.get("projects"), dict) else {}
    )
    project_block = projects.get(alias) if isinstance(projects.get(alias), dict) else {}
    configured = configured_k8s_backends(project_block, alias)
    local_clusters: list[dict[str, Any]] = []
    if clusters_root.is_dir():
        for item in sorted(clusters_root.iterdir()):
            if item.is_dir():
                kubeconfig, state_path = item / "kubeconfig", item / "state.json"
                local_clusters.append(
                    {
                        "source": "local_state",
                        "cluster_name": item.name,
                        "context": item.name,
                        "kubeconfig": str(kubeconfig),
                        "kubeconfig_exists": kubeconfig.is_file(),
                        "state_exists": state_path.is_file(),
                    }
                )
    return {
        "ok": True,
        "project": alias,
        "configured": configured,
        "local_clusters": local_clusters,
        "cloud_clusters": cloud_clusters,
        "has_infra": bool(
            configured
            or any(x.get("kubeconfig_exists") for x in local_clusters)
            or cloud_clusters
        ),
        "agent_npa_ready": npa_ready,
        "agent_npa_error": npa_error,
        "terraform_dir": str(terraform_dir),
        "options": [
            "POST /api/infra/provision to let the agent create the minimal Kubernetes backend.",
            "Add projects.<alias>.kubernetes to ~/.npa/config.yaml on the agent to use an existing backend.",
            "Pass project/cluster_name in the workflow submit payload to target a known backend.",
        ],
    }


def validate_resource_inventory(payload: Any) -> dict[str, Any]:
    """Validate the live inventory contract and require at least one real reference."""
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise ValueError("tenant resource inventory endpoint did not return ok=true")
    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("tenant resource inventory endpoint returned no categories")
    if not any(
        isinstance(x, dict)
        and (
            int(x.get("discovered_count") or 0) > 0
            or int(x.get("configured_count") or 0) > 0
        )
        for x in categories
    ):
        raise ValueError(
            "tenant resource inventory has no configured or discovered resources"
        )
    states = {"discovered", "configured", "empty", "error"}
    if any(
        not isinstance(x, dict) or str(x.get("status") or "") not in states
        for x in categories
    ):
        raise ValueError("tenant resource inventory returned an invalid category state")
    return payload


def discover_mk8s_accelerators(
    cluster_id: str, command: list[str], command_env: dict[str, str]
) -> dict[str, Any]:
    """Return accelerator families grounded in a cluster's live node groups."""
    try:
        env, _source = prepare_agent_cloud_environment(command_env)
        proc = run_bounded_agent_command(
            [
                *command,
                "mk8s",
                "node-group",
                "list",
                "--parent-id",
                cluster_id,
                "--format",
                "json",
            ],
            env=env,
            timeout_s=30,
        )
        payload = json.loads(proc.stdout or "{}") if proc.returncode == 0 else {}
    except Exception:
        return {"available_accelerators": [], "gpu_platforms": []}
    platforms: list[str] = []
    accelerators: list[str] = []
    for group in payload.get("items", []) if isinstance(payload, dict) else []:
        spec = group.get("spec", {}) if isinstance(group, dict) else {}
        template = spec.get("template", {}) if isinstance(spec, dict) else {}
        resources = template.get("resources", {}) if isinstance(template, dict) else {}
        platform = str(resources.get("platform") or "").strip().lower()
        if platform:
            platforms.append(platform)
        mapping = {"gpu-rtx6000": "RTXPRO6000", "gpu-l40s": "L40S"}
        if platform in mapping:
            accelerators.append(mapping[platform])
        elif platform.startswith("gpu-"):
            accelerators.append(platform.removeprefix("gpu-").upper())
    available = sorted(set(accelerators))
    result: dict[str, Any] = {
        "available_accelerators": available,
        "gpu_platforms": sorted(set(platforms)),
    }
    if len(available) == 1:
        result["gpu_accelerator"] = available[0]
    return result


def classify_discovery_error(message: str) -> tuple[str, str]:
    """Return a stable error kind and public message without echoing CLI output."""
    lowered = str(message or "").lower()
    if (
        "permissiondenied" in lowered
        or "permission denied" in lowered
        or "forbidden" in lowered
    ):
        return (
            "permission_denied",
            "Credentials are authenticated but cannot enumerate this resource category.",
        )
    if (
        "unauthenticated" in lowered
        or "authentication" in lowered
        or "access token" in lowered
    ):
        return (
            "authentication_error",
            "Nebius authentication failed for this resource category.",
        )
    if "timed out" in lowered or "timeout" in lowered:
        return (
            "timeout",
            "Resource discovery timed out; configured references are still shown.",
        )
    if (
        "unknown command" in lowered
        or "not found" in lowered
        or "unsupported" in lowered
    ):
        return "unsupported", "This Nebius CLI cannot discover this resource category."
    return "discovery_error", "Nebius resource discovery failed for this category."


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "items",
        "resources",
        "instances",
        "clusters",
        "registries",
        "buckets",
        "networks",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if payload else []


def _safe_item(item: dict[str, Any], *, kind: str, source: str) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    result = {
        "kind": kind,
        "source": source,
        "id": str(metadata.get("id") or item.get("id") or "").strip(),
        "name": str(metadata.get("name") or item.get("name") or "").strip(),
        "status": str(
            status.get("state") or status.get("status") or item.get("state") or ""
        ).strip(),
    }
    return {
        key: value
        for key, value in result.items()
        if value != "" and not _SECRET_KEY_RE.search(key)
    }


def category_payload(
    category_id: str,
    label: str,
    *,
    configured: Iterable[dict[str, Any]] = (),
    discovered: Iterable[dict[str, Any]] = (),
    error: dict[str, str] | None = None,
    discovery_attempted: bool = True,
) -> dict[str, Any]:
    """Build a category with explicit discovered/configured/empty/error state."""
    configured_items = [dict(item) for item in configured if isinstance(item, dict)]
    discovered_items = [dict(item) for item in discovered if isinstance(item, dict)]
    if error:
        status = "error"
    elif discovered_items:
        status = "discovered"
    elif configured_items:
        status = "configured"
    else:
        status = "empty"
    result: dict[str, Any] = {
        "id": category_id,
        "label": label,
        "status": status,
        "configured": configured_items,
        "discovered": discovered_items,
        "configured_count": len(configured_items),
        "discovered_count": len(discovered_items),
        "discovery_attempted": bool(discovery_attempted),
    }
    if error:
        result["error"] = {
            "kind": str(error.get("kind") or "discovery_error"),
            "message": str(error.get("message") or "Resource discovery failed."),
        }
    return result


def discover_nebius_categories(
    *,
    project_id: str,
    tenant_id: str,
    profile: str,
    runner: DiscoveryRunner,
) -> list[dict[str, Any]]:
    """Discover a fixed read-only Nebius inventory using an injected runner."""
    project = str(project_id or "").strip()
    tenant = str(tenant_id or "").strip()
    safe_profile = (
        re.sub(r"[^A-Za-z0-9_.-]", "", str(profile or "").strip()) or "cursor-sa"
    )
    specs = [
        (
            "project",
            "Project",
            "project",
            ["iam", "project", "get", "--id", project],
            bool(project),
        ),
        (
            "tenant",
            "Tenant",
            "tenant",
            ["iam", "tenant", "get", "--id", tenant],
            bool(tenant),
        ),
        (
            "compute",
            "Compute",
            "instance",
            ["compute", "instance", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
        (
            "gpu_clusters",
            "GPU clusters",
            "gpu_cluster",
            ["compute", "gpu-cluster", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
        (
            "kubernetes",
            "Kubernetes",
            "cluster",
            ["mk8s", "cluster", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
        (
            "registry",
            "Container registry",
            "registry",
            ["registry", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
        (
            "storage",
            "Object storage",
            "bucket",
            ["storage", "bucket", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
        (
            "network",
            "Networks",
            "network",
            ["vpc", "network", "list", "--parent-id", project, "--all"],
            bool(project),
        ),
    ]
    categories: list[dict[str, Any]] = []
    for category_id, label, kind, args, can_attempt in specs:
        if not can_attempt:
            categories.append(
                category_payload(
                    category_id,
                    label,
                    error={
                        "kind": "not_configured",
                        "message": "No project or tenant identifier is configured for discovery.",
                    },
                    discovery_attempted=False,
                )
            )
            continue
        command = ["nebius", "--profile", safe_profile, *args, "--format", "json"]
        try:
            returncode, stdout, stderr = runner(command)
        except TimeoutError:
            returncode, stdout, stderr = 1, "", "timeout"
        except Exception:
            returncode, stdout, stderr = 1, "", "discovery failed"
        if returncode != 0:
            error_kind, public_message = classify_discovery_error(stderr or stdout)
            categories.append(
                category_payload(
                    category_id,
                    label,
                    error={"kind": error_kind, "message": public_message},
                )
            )
            continue
        try:
            payload = json.loads(stdout or "{}")
        except (TypeError, ValueError):
            categories.append(
                category_payload(
                    category_id,
                    label,
                    error={
                        "kind": "invalid_response",
                        "message": "Nebius discovery returned an unreadable response.",
                    },
                )
            )
            continue
        discovered = [
            _safe_item(item, kind=kind, source="nebius_cli")
            for item in _payload_items(payload)
        ]
        categories.append(category_payload(category_id, label, discovered=discovered))
    return categories


def run_resource_discovery_command(
    command: list[str],
    *,
    command_env: dict[str, str] | None = None,
    timeout_s: int = 30,
) -> tuple[int, str, str]:
    """Run one allowlisted, read-only Nebius command without a shell."""
    allowed = {"iam", "compute", "mk8s", "registry", "storage", "vpc"}
    if len(command) < 4 or command[0] != "nebius" or command[3] not in allowed:
        return 2, "", "unsupported resource discovery command"
    try:
        env, _source = prepare_agent_cloud_environment(command_env or dict(os.environ))
    except ValueError:
        return 2, "", "agent credential source is unavailable"
    env["NEBIUS_PROFILE"] = str(command[2])
    try:
        proc = run_bounded_agent_command(
            command,
            env=env,
            timeout_s=timeout_s,
        )
    except TimeoutError as exc:
        raise TimeoutError("resource discovery timed out") from exc
    except OSError:
        return 1, "", "resource discovery command failed to start"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _configured_references(
    project_block: dict[str, Any], env: dict[str, str], project_alias: str
) -> dict[str, list[dict[str, Any]]]:
    project_id = str(
        env.get("NEBIUS_PROJECT_ID") or project_block.get("project_id") or ""
    ).strip()
    tenant_id = str(
        env.get("NEBIUS_TENANT_ID") or project_block.get("tenant_id") or ""
    ).strip()
    references: dict[str, list[dict[str, Any]]] = {
        "project": (
            [
                {
                    "kind": "project",
                    "source": "staged_config",
                    "name": project_alias,
                    "id": project_id,
                }
            ]
            if project_id
            else []
        ),
        "tenant": (
            [{"kind": "tenant", "source": "staged_config", "id": tenant_id}]
            if tenant_id
            else []
        ),
        "compute": [
            {
                "kind": "agent",
                "source": "staged_agent",
                "name": str(env.get("NPA_AGENT_NAME") or "agent"),
            }
        ],
        "kubernetes": [],
        "registry": [],
        "storage": [],
        "network": [],
    }
    kube = (
        project_block.get("kubernetes")
        if isinstance(project_block.get("kubernetes"), dict)
        else {}
    )
    context = str(
        kube.get("context")
        or kube.get("context_name")
        or project_block.get("k8s_context")
        or ""
    ).strip()
    cluster_name = str(kube.get("cluster_name") or kube.get("name") or "").strip()
    if context or cluster_name:
        references["kubernetes"].append(
            {
                "kind": "cluster",
                "source": "staged_config",
                "name": cluster_name or context,
                "context": context,
            }
        )
    registry = str(
        env.get("NPA_REGISTRY") or project_block.get("container_registry") or ""
    ).strip()
    if registry:
        references["registry"].append(
            {"kind": "registry", "source": "staged_config", "name": registry}
        )
    bucket = str(
        env.get("NPA_AGENT_S3_BUCKET") or env.get("NEBIUS_S3_BUCKET") or ""
    ).strip()
    if bucket:
        references["storage"].append(
            {"kind": "bucket", "source": "staged_credentials", "name": bucket}
        )
    return references


def _local_categories(
    state: dict[str, Any], tool_refs: Iterable[str]
) -> list[dict[str, Any]]:
    families: dict[str, int] = {}
    for tool_ref in tool_refs:
        parts = str(tool_ref).split(".")
        family = parts[1] if len(parts) > 1 and parts[0] == "workbench" else parts[0]
        families[family] = families.get(family, 0) + 1
    workbench_items = [
        {"kind": "tool_family", "source": "agent_catalog", "name": name, "count": count}
        for name, count in sorted(families.items())
    ]
    run_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    run_map = (
        state.get("sim2real_runs")
        if isinstance(state.get("sim2real_runs"), dict)
        else {}
    )
    for run_id, details in run_map.items():
        token = str(run_id or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        detail = details if isinstance(details, dict) else {}
        run_items.append(
            {
                "kind": "run",
                "source": "agent_state",
                "name": token,
                "status": str(
                    detail.get("result") or detail.get("stage") or ""
                ).strip(),
            }
        )
    for key in ("latest_submit", "workflow_submit"):
        detail = state.get(key) if isinstance(state.get(key), dict) else {}
        run_id = str(detail.get("run_id") or "").strip()
        if run_id and run_id not in seen:
            seen.add(run_id)
            run_items.append(
                {
                    "kind": "run",
                    "source": "agent_state",
                    "name": run_id,
                    "status": str(detail.get("submit_mode") or "submitted"),
                }
            )
    return [
        category_payload("workbench", "Workbench tools", discovered=workbench_items),
        category_payload("workflows", "Workflows and runs", discovered=run_items),
    ]


def build_resource_inventory(
    *,
    config: dict[str, Any],
    env: dict[str, str],
    state: dict[str, Any],
    tool_refs: Iterable[str],
    runner: DiscoveryRunner,
    generated_at: str,
    metadata_token_available: bool,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Assemble configured, discovered, empty, and error resource states."""
    now = time.time()
    with _INVENTORY_LOCK:
        cached = _INVENTORY_CACHE.get("payload")
        if (
            not force_refresh
            and isinstance(cached, dict)
            and now < float(_INVENTORY_CACHE.get("expires_at") or 0)
        ):
            return cached
        alias = str(
            config.get("default_project")
            or env.get("NPA_AGENT_PROJECT_ALIAS")
            or "default"
        )
        projects = (
            config.get("projects") if isinstance(config.get("projects"), dict) else {}
        )
        project_block = (
            projects.get(alias) if isinstance(projects.get(alias), dict) else {}
        )
        project_id = str(
            env.get("NEBIUS_PROJECT_ID") or project_block.get("project_id") or ""
        ).strip()
        tenant_id = str(
            env.get("NEBIUS_TENANT_ID") or project_block.get("tenant_id") or ""
        ).strip()
        region = str(
            env.get("NEBIUS_REGION") or project_block.get("region") or ""
        ).strip()
        requested_profile = (
            "cursor-sa"
            if metadata_token_available
            else str(env.get("NEBIUS_PROFILE") or "cursor-sa")
        )
        profile = re.sub(r"[^A-Za-z0-9_.-]", "", requested_profile) or "cursor-sa"
        categories = discover_nebius_categories(
            project_id=project_id,
            tenant_id=tenant_id,
            profile=profile,
            runner=runner,
        )
        categories = merge_configured_references(
            categories, _configured_references(project_block, env, alias)
        )
        categories.extend(_local_categories(state, tool_refs))
        summary = inventory_summary(categories)
        error_kinds = {
            str((item.get("error") or {}).get("kind") or "")
            for item in categories
            if isinstance(item, dict) and isinstance(item.get("error"), dict)
        }
        if "authentication_error" in error_kinds:
            profile_status = "authentication_error"
        elif any(
            item.get("discovery_attempted") and item.get("status") != "error"
            for item in categories
        ):
            profile_status = "authenticated"
        elif error_kinds:
            profile_status = "error"
        else:
            profile_status = "not_checked"
        payload = {
            "ok": True,
            "generated_at": generated_at,
            "context": {
                "project_alias": alias,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
                "profile": profile,
                "profile_status": profile_status,
            },
            "summary": summary,
            "categories": categories,
        }
        _INVENTORY_CACHE["payload"] = payload
        _INVENTORY_CACHE["expires_at"] = now + 30.0
        return payload


def merge_configured_references(
    categories: list[dict[str, Any]], references: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Attach staged references without converting discovery errors into absence."""
    merged: list[dict[str, Any]] = []
    for category in categories:
        item = dict(category)
        configured = [
            dict(ref)
            for ref in references.get(str(item.get("id") or ""), [])
            if isinstance(ref, dict)
        ]
        item["configured"] = configured
        item["configured_count"] = len(configured)
        if item.get("status") == "empty" and configured:
            item["status"] = "configured"
        merged.append(item)
    return merged


def inventory_summary(categories: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = [item for item in categories if isinstance(item, dict)]
    return {
        "categories": len(rows),
        "discovered_categories": sum(
            item.get("status") == "discovered" for item in rows
        ),
        "configured_only_categories": sum(
            item.get("status") == "configured" for item in rows
        ),
        "empty_categories": sum(item.get("status") == "empty" for item in rows),
        "error_categories": sum(item.get("status") == "error" for item in rows),
        "configured_resources": sum(
            int(item.get("configured_count") or 0) for item in rows
        ),
        "discovered_resources": sum(
            int(item.get("discovered_count") or 0) for item in rows
        ),
    }


def format_resource_inventory(inventory: dict[str, Any]) -> str:
    """Format a concise zero-token grounded reply from an inventory response."""
    context = (
        inventory.get("context") if isinstance(inventory.get("context"), dict) else {}
    )
    summary = inventory_summary(inventory.get("categories") or [])
    lines = [
        "**Tenant resources** (grounded, read-only discovery):",
        f"- **project**: `{context.get('project_alias') or 'not configured'}`",
        f"- **project_id**: `{context.get('project_id') or 'not configured'}`",
        f"- **tenant_id**: `{context.get('tenant_id') or 'not configured'}`",
        f"- **region**: `{context.get('region') or 'not configured'}`",
        f"- **credential_profile**: `{context.get('profile') or 'not configured'}`",
        f"- **discovered_resources**: `{summary['discovered_resources']}`",
    ]
    for category in inventory.get("categories") or []:
        if not isinstance(category, dict):
            continue
        lines.append(
            f"- **{category.get('label') or category.get('id')}**: "
            f"status=`{category.get('status')}` discovered=`{category.get('discovered_count', 0)}` "
            f"configured=`{category.get('configured_count', 0)}`"
        )
        error = category.get("error") if isinstance(category.get("error"), dict) else {}
        if error:
            lines.append(
                f"  - discovery_error=`{error.get('kind')}` — {error.get('message')}"
            )
    lines.append(
        "- Open the **Tenant resources** panel or refresh `GET /api/resources` for details."
    )
    return "\n".join(lines)

"""Deploy / destroy / plan / status for npa-managed **fleets** of clusters.

A fleet is many backend-selected clusters across many projects in one tenant.
This module owns only fleet-level orchestration, per ``(project, cluster)``
target:

1. Resolving (or creating) the project under the tenant via the ``nebius`` CLI.
2. Selecting targets, scheduling concurrency, and aggregating results.
3. Persisting backend-discriminated inventory and shared-project network
   ownership.
4. Dispatching each target to its mk8s or soperator backend adapter.

Backend modules own desired state, rendering, apply, native status,
verification/reconciliation, and exact-identity destroy for one target.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import replace
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped]

from npa.cluster_backends.process import (
    require_bin as _require_bin,
    run_capture as _run_capture,
    run_stream as _run_stream,
    terraform_env as _terraform_env,
)
from npa.cluster_backends import (
    BackendOwnershipError,
    get_backend,
    require_backend_ownership,
)
from npa.cluster_backends.mk8s import (
    MK8sApplyRequest,
    MK8sDestroyRequest,
    MK8sExecutionScope,
    MK8sProjectIdentity,
    MK8sStatusRequest,
)
from npa.cluster_backends.mk8s_execution import (
    _PROVIDER_FIELD_MISSING,
    _ensure_private_directory,
    _get_project,
    _is_not_found_result,
    _load_env_sidecar,
    _load_json_file,
    _log,
    _nebius_argv,
    _prepare_install_dir,
    _provider_field,
    _tf_run,
    _write_json_file,
)
from npa.cluster_backends import mk8s_execution as _mk8s_execution
from npa.cluster_backends.soperator import (
    SoperatorApplyRequest,
    SoperatorDestroyRequest,
    SoperatorStatusRequest,
)
from npa.soperator.lifecycle import (
    SoperatorDeploymentValidationError,
    SoperatorStateCaptureError,
)
from npa.cluster_backends.quotas import preflight_region, shortfall_message
from npa.fleet.spec import ClusterSpec, FleetSpec, ProjectSpec
from npa.cluster_backends.mk8s_render import (
    validate_recipe_mig_compatibility,
)

# Deprecated test/developer compatibility names. Production orchestration calls
# the public backend adapter; these aliases keep direct helper consumers working
# while ownership remains in ``cluster_backends.mk8s_execution``.
_cluster_tf_env = _mk8s_execution._cluster_tf_env
_BACKEND_DEPLOY_CLUSTER = _mk8s_execution.deploy_cluster
_BACKEND_DESTROY_CLUSTER = _mk8s_execution.destroy_cluster
_BACKEND_IS_VERIFIED_UNCHANGED = _mk8s_execution.is_verified_unchanged_target
_persist_npa_cluster_identity = _mk8s_execution._persist_npa_cluster_identity
_remove_npa_cluster_identity = _mk8s_execution._remove_npa_cluster_identity
_provider_node_group_matches_pool = _mk8s_execution._provider_node_group_matches_pool
_run_to_log = _mk8s_execution._run_to_log
_terraform_managed_ids = _mk8s_execution._terraform_managed_ids
_terraform_outputs = _mk8s_execution._terraform_outputs
_write_env_sidecar = _mk8s_execution._write_env_sidecar
validate_gpu_health = _mk8s_execution.validate_gpu_health
wait_for_mig_ready = _mk8s_execution.wait_for_mig_ready
_LEGACY_EXECUTION_LOCK = threading.RLock()


def _write_kubeconfig(*args: Any, **kwargs: Any) -> None:
    # This compatibility seam temporarily forwards a formerly fleet-private
    # helper into the backend module. Keep the swap atomic: parallel fleet tests
    # and legacy embedders may patch different helpers at the same time.
    with _LEGACY_EXECUTION_LOCK:
        saved = _mk8s_execution._run_capture
        try:
            _mk8s_execution._run_capture = _run_capture
            _mk8s_execution._write_kubeconfig(*args, **kwargs)
        finally:
            _mk8s_execution._run_capture = saved


_LEGACY_HELPER_DEFAULTS = {
    name: globals()[name]
    for name in (
        "_cluster_tf_env",
        "_require_bin",
        "_terraform_env",
        "_prepare_install_dir",
        "_run_stream",
        "_terraform_managed_ids",
        "_terraform_outputs",
        "_write_kubeconfig",
        "_persist_npa_cluster_identity",
        "_tf_run",
        "_run_capture",
        "_get_project",
        "validate_gpu_health",
        "wait_for_mig_ready",
    )
}


def _call_legacy_execution(function: Callable[..., dict[str, Any]], **kwargs: Any):
    """Compatibility seam for callers that patched former fleet-private helpers."""

    names = (
        "_cluster_tf_env",
        "_require_bin",
        "_terraform_env",
        "_prepare_install_dir",
        "_run_stream",
        "_terraform_managed_ids",
        "_terraform_outputs",
        "_write_kubeconfig",
        "_persist_npa_cluster_identity",
        "_tf_run",
        "_run_capture",
        "_get_project",
        "validate_gpu_health",
        "wait_for_mig_ready",
    )
    with _LEGACY_EXECUTION_LOCK:
        saved = {name: getattr(_mk8s_execution, name) for name in names}
        try:
            for name in names:
                if globals()[name] is not _LEGACY_HELPER_DEFAULTS[name]:
                    setattr(_mk8s_execution, name, globals()[name])
            return function(**kwargs)
        finally:
            for name, value in saved.items():
                setattr(_mk8s_execution, name, value)


def _deploy_one_cluster(**kwargs: Any) -> dict[str, Any]:
    return _call_legacy_execution(_BACKEND_DEPLOY_CLUSTER, **kwargs)


def _destroy_one_cluster(**kwargs: Any) -> dict[str, Any]:
    return _call_legacy_execution(_BACKEND_DESTROY_CLUSTER, **kwargs)


def _is_verified_unchanged_target(**kwargs: Any) -> bool:
    return bool(_call_legacy_execution(_BACKEND_IS_VERIFIED_UNCHANGED, **kwargs))


_LEGACY_DEPLOY_COMPAT = _deploy_one_cluster
_LEGACY_DESTROY_COMPAT = _destroy_one_cluster


def _legacy_helpers_patched() -> bool:
    return any(
        globals()[name] is not default
        for name, default in _LEGACY_HELPER_DEFAULTS.items()
    )


logger = logging.getLogger(__name__)

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_K8S_TRAINING_SUBDIR = "k8s-training"
_MODULES_SUBDIR = "modules"
_TARGET_BACKEND_OWNER = ".npa-backend-owner.json"
# Pinned ref cloned when no local recipe is available, matching the repo-vendored
# copy (deploy/cluster/vendor + the single-cluster wrapper) so a fleet run from an
# installed package doesn't silently drift onto upstream ``main`` HEAD.
_PINNED_LIBRARY_REF = "main-v2026-05-25+local-cluster-patches"
_FILESYSTEM_VERIFIER = (
    Path("filesystem-csi-validation") / "01-verify-node-filesystem-mounts.sh"
)
_ENV_SIDECAR = ".npa-fleet-env.json"
_PROJECT_NETWORK_STATE = ".npa-fleet-network.json"
_FLEET_STATE = "fleet-state.json"
_MIN_TERRAFORM_VERSION = (1, 12, 0)


def _project_in_scope(
    project: ProjectSpec, only: list[str] | None, prefix: str
) -> bool:
    """A project matches ``--only-projects`` by its key **or** its display name."""

    if not only:
        return True
    return project.key() in only or project.display_name(prefix) in only


# --------------------------------------------------------------------------- #
# Nebius CLI env + tenant/region resolution
# --------------------------------------------------------------------------- #
def _nebius_cli_env() -> dict[str, str]:
    """Env for direct ``nebius`` CLI calls.

    A stale ambient ``NEBIUS_IAM_TOKEN`` (e.g. an expired cloud-env token) is
    preferred by the CLI over the active profile's exec-plugin, so pre-flight
    calls fail Unauthenticated even though the profile can mint a fresh token.
    Drop it unless the caller explicitly opts into reuse (``NPA_REUSE_IAM_TOKEN``).
    """

    env = os.environ.copy()
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not reuse:
        env.pop("NEBIUS_IAM_TOKEN", None)
    return env


def _nebius_config() -> dict[str, Any]:
    path = Path.home() / ".nebius" / "config.yaml"
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        logger.warning(
            "could not read Nebius config at %s (%s)", path, type(exc).__name__
        )
        return {}
    except yaml.YAMLError as exc:
        # Do not log the YAML exception text: parser snippets could contain
        # credentials from a malformed profile.
        logger.warning(
            "could not parse Nebius config at %s (%s)", path, type(exc).__name__
        )
        return {}
    if not isinstance(loaded, dict):
        logger.warning("ignoring Nebius config at %s because it is not a mapping", path)
        return {}
    return loaded


def _resolve_tenant_id(nebius_bin: str, explicit: str, profile: str = "") -> str:
    if explicit:
        return explicit
    cfg = _nebius_config()
    # With an explicit profile, that profile's tenant is authoritative: falling
    # back to the active profile's tenant would deploy into the wrong tenant.
    selected = profile or str(cfg.get("default", "") or "")
    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles"), dict) else {}
    prof = (
        profiles.get(selected, {}) if isinstance(profiles.get(selected), dict) else {}
    )
    tenant = str(prof.get("tenant-id", "") or "")
    if tenant:
        return tenant
    if profile:
        raise ValueError(
            f"tenant_id could not be resolved: profile {profile!r} has no 'tenant-id' in "
            "~/.nebius/config.yaml; set 'tenant_id' in the fleet spec"
        )
    # Fall back to the npa environment config.
    try:
        from npa.clients.config import resolve_environment

        envcfg = resolve_environment()
        if envcfg and envcfg.tenant_id:
            return envcfg.tenant_id
    except Exception as exc:
        logger.debug("tenant_id fallback via ~/.npa failed: %s", exc)
    raise ValueError(
        "tenant_id could not be resolved from the spec, ~/.nebius/config.yaml, or ~/.npa"
    )


def tenant_id_from_profile(profile: str) -> str:
    """Tenant recorded for *profile* in ``~/.nebius/config.yaml`` (config-only).

    Used by ``plan``, which must stay side-effect free: no ``nebius`` binary and
    no API calls, just the on-disk profile.
    """

    cfg = _nebius_config()
    selected = profile or str(cfg.get("default", "") or "")
    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles"), dict) else {}
    prof = (
        profiles.get(selected, {}) if isinstance(profiles.get(selected), dict) else {}
    )
    return str(prof.get("tenant-id", "") or "")


def _resolve_region(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        from npa.clients.config import resolve_environment

        envcfg = resolve_environment()
        if envcfg and envcfg.region:
            return envcfg.region
    except Exception as exc:
        logger.debug("region fallback via ~/.npa failed: %s", exc)
    raise ValueError("region could not be resolved from the spec or ~/.npa config")


def _resolve_ssh_public_key(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    for candidate in ("id_ed25519.pub", "id_rsa.pub"):
        path = Path.home() / ".ssh" / candidate
        if path.exists():
            return path.read_text().strip()
    raise ValueError(
        "ssh_public_key not set in spec and no ~/.ssh/id_ed25519.pub or id_rsa.pub found"
    )


# --------------------------------------------------------------------------- #
# k8s-training recipe source resolution
#
# The recipe is a *root* module that references sibling ``../modules/...`` and
# embeds its own provider config, so it cannot be sourced as a Terraform child
# module. We resolve the recipe *root* (the dir that contains both
# ``k8s-training/`` and ``modules/``), copy it per cluster, and run terraform
# inside the ``k8s-training`` copy.
# --------------------------------------------------------------------------- #
def _is_recipe_root(path: Path) -> bool:
    return (path / _K8S_TRAINING_SUBDIR / "variables.tf").exists() and (
        path / _MODULES_SUBDIR
    ).is_dir()


def _find_vendored_recipe_root() -> Path | None:
    """Walk up from this file to find the repo-vendored solutions-library root."""

    rel = Path("deploy") / "cluster" / "vendor" / "nebius-solutions-library"
    for base in Path(__file__).resolve().parents:
        candidate = base / rel
        if _is_recipe_root(candidate):
            return candidate
    return None


def _coerce_recipe_root(path: Path) -> Path:
    """Accept either a recipe root or a ``k8s-training`` dir and return the root."""

    path = path.expanduser().resolve()
    if _is_recipe_root(path):
        return path
    if path.name == _K8S_TRAINING_SUBDIR and _is_recipe_root(path.parent):
        return path.parent
    raise ValueError(
        f"{path} is not a k8s-training recipe (need a dir containing "
        "'k8s-training/' and 'modules/', or the 'k8s-training' dir itself)"
    )


def _resolve_recipe_root(
    k8s_training_dir: Path | None,
    *,
    ref: str | None,
    work_root: Path,
    on_status: Callable[[str], None] | None,
) -> Path:
    """Resolve the solutions-library recipe root (contains k8s-training + modules).

    Priority: explicit dir > env override > cloned ref (latest) > repo-vendored.
    Cloning satisfies "consume the latest k8s-training changes"; the vendored
    copy is the tested default when no ref/dir is requested.
    """

    if k8s_training_dir is not None:
        return _coerce_recipe_root(k8s_training_dir)

    env_dir = os.environ.get("NPA_K8S_TRAINING_DIR", "").strip()
    if env_dir:
        return _coerce_recipe_root(Path(env_dir))

    if ref:
        clone_dir = work_root / "nebius-solutions-library"
        if not _is_recipe_root(clone_dir):
            work_root.mkdir(parents=True, exist_ok=True)
            git = _require_bin("git")
            _log(
                on_status,
                f"cloning nebius-solutions-library@{ref} for latest k8s-training",
            )
            _run_stream(
                [
                    git,
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    ref,
                    _SOLUTIONS_LIBRARY_REPO,
                    str(clone_dir),
                ],
                timeout=600,
            )
        return clone_dir

    vendored = _find_vendored_recipe_root()
    if vendored is not None:
        return vendored
    # No vendored copy available (e.g. installed as a package without the repo's
    # deploy/cluster/vendor tree). Clone the SAME pinned ref the vendored copy
    # tracks rather than upstream HEAD, so an installed-package run stays
    # reproducible instead of silently drifting onto ``main``.
    return _resolve_recipe_root(
        None, ref=_PINNED_LIBRARY_REF, work_root=work_root, on_status=on_status
    )


# --------------------------------------------------------------------------- #
# Project resolution / creation
# --------------------------------------------------------------------------- #
def _list_projects(
    nebius_bin: str, tenant_id: str, env: dict[str, str], profile: str = ""
) -> list[dict[str, Any]]:
    result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "iam",
            "project",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not list projects (nebius exited {result.returncode})"
        )
    if not result.stdout.strip():
        raise RuntimeError("could not list projects (nebius returned no JSON)")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("could not parse project list JSON") from exc
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError("project list JSON has a non-list 'items' field")
    return [item for item in items if isinstance(item, dict)]


def _find_project_id(projects: list[dict[str, Any]], name: str) -> str:
    matches = [
        str((item.get("metadata", {}) or {}).get("id") or "")
        for item in projects
        if (item.get("metadata", {}) or {}).get("name") == name
        and (item.get("metadata", {}) or {}).get("id")
    ]
    if len(matches) > 1:
        raise ValueError(
            f"project name {name!r} is ambiguous under the selected tenant; "
            "use an explicit project_id"
        )
    return matches[0] if matches else ""


def _verify_existing_project(
    payload: dict[str, Any],
    *,
    project_id: str,
    tenant_id: str,
    name: str,
    region: str,
) -> None:
    """Fail closed unless a same-name resume has the exact requested identity."""

    metadata = payload.get("metadata", {}) or {}
    spec = payload.get("spec", {}) or {}
    status = payload.get("status", {}) or {}
    actual_region = str(
        spec.get("region") or status.get("region") or metadata.get("region") or ""
    )
    mismatches: list[str] = []
    if str(metadata.get("id") or "") != project_id:
        mismatches.append("provider id does not match the listed project")
    if str(metadata.get("name") or "") != name:
        mismatches.append("display name changed after project listing")
    parent_id = _provider_field(metadata, "parent_id", "parentId")
    if str(parent_id if parent_id is not _PROVIDER_FIELD_MISSING else "") != tenant_id:
        mismatches.append("project belongs to another tenant")
    if region and actual_region != region:
        mismatches.append(
            f"project region {actual_region or '<unavailable>'!r} does not match "
            f"requested region {region!r}"
        )
    if mismatches:
        raise ValueError(
            f"existing project {name!r} failed immutable identity verification: "
            + "; ".join(mismatches)
        )


def _create_project(
    nebius_bin: str,
    tenant_id: str,
    name: str,
    env: dict[str, str],
    *,
    region: str = "",
    profile: str = "",
) -> str:
    argv = [
        *_nebius_argv(nebius_bin, profile),
        "iam",
        "project",
        "create",
        "--parent-id",
        tenant_id,
        "--name",
        name,
    ]
    # Projects are regional in Nebius; pass the target region so the project (and
    # its clusters) land in the intended region rather than the tenant default.
    if region:
        argv += ["--region", region]
    argv += ["--format", "json"]
    result = _run_capture(argv, env=env)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"could not parse project create output for {name!r}: {exc}"
        ) from exc
    project_id = str(payload.get("metadata", {}).get("id") or payload.get("id") or "")
    if not project_id:
        raise ValueError(
            f"project create for {name!r} returned no id: {result.stdout[:200]}"
        )
    return project_id


def _list_subnets(
    nebius_bin: str, project_id: str, env: dict[str, str], profile: str = ""
) -> list[dict[str, Any]]:
    result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "vpc",
            "subnet",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not list subnets (nebius exited {result.returncode})"
        )
    if not result.stdout.strip():
        raise RuntimeError("could not list subnets (nebius returned no JSON)")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("could not parse subnet list JSON") from exc
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError("subnet list JSON has a non-list 'items' field")
    return [item for item in items if isinstance(item, dict)]


def ensure_subnet(
    nebius_bin: str,
    project_id: str,
    *,
    name_stem: str,
    env: dict[str, str],
    profile: str = "",
    network_state_path: Path | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Return ``(subnet_id, created_network_id)`` for *project_id*.

    The k8s-training root module requires an existing ``subnet_id`` (it does not
    create one). Freshly created projects may have no VPC yet, so create a
    network + subnet (inheriting the network's default IPv4 pools) when absent.
    ``created_network_id`` is non-empty only when this call created the network,
    so ``destroy`` can reclaim the network + subnet it created (an existing
    subnet is reused and left untouched).
    """

    network_state = _load_json_file(network_state_path) if network_state_path else {}
    owned_network_id = str(network_state.get("created_network_id") or "")
    owned_subnet_id = str(network_state.get("subnet_id") or "")
    subnets = _list_subnets(nebius_bin, project_id, env, profile)
    if subnets:
        # Deterministic pick: prefer a subnet whose name looks like the project
        # default, else the lowest id, so repeated runs choose the same subnet
        # instead of relying on list order.
        def _rank(sub: dict[str, Any]) -> tuple[int, int, str]:
            meta = sub.get("metadata", {})
            name = str(meta.get("name") or "")
            subnet_id = str(meta.get("id") or "")
            return (
                0 if subnet_id == owned_subnet_id else 1,
                0 if "default" in name else 1,
                subnet_id,
            )

        for sub in sorted(subnets, key=_rank):
            sid = str(sub.get("metadata", {}).get("id") or "")
            if sid:
                return sid, owned_network_id if sid == owned_subnet_id else ""
    _log(
        on_status,
        f"no subnet in project {project_id[:12]}...; creating network + subnet",
    )
    network_id = owned_network_id
    if not network_id:
        net = _run_capture(
            [
                *_nebius_argv(nebius_bin, profile),
                "vpc",
                "network",
                "create",
                "--parent-id",
                project_id,
                "--name",
                f"{name_stem}-net",
                "--format",
                "json",
            ],
            env=env,
        )
        try:
            network_id = str(
                json.loads(net.stdout or "{}").get("metadata", {}).get("id") or ""
            )
        except json.JSONDecodeError:
            network_id = ""
        if not network_id:
            raise ValueError(
                f"could not create network in project {project_id}: {net.stdout[:200]}"
            )
        # Persist ownership immediately. If subnet creation fails, destroy can
        # still recover the otherwise orphaned network on a later retry.
        if network_state_path:
            _write_json_file(
                network_state_path,
                {
                    "project_id": project_id,
                    "created_network_id": network_id,
                    "subnet_id": "",
                    "profile": profile,
                },
            )
    subnet_result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "vpc",
            "subnet",
            "create",
            "--parent-id",
            project_id,
            "--network-id",
            network_id,
            "--name",
            f"{name_stem}-subnet",
            "--format",
            "json",
        ],
        env=env,
    )
    try:
        subnet_id = str(
            json.loads(subnet_result.stdout or "{}").get("metadata", {}).get("id") or ""
        )
    except json.JSONDecodeError:
        subnet_id = ""
    if not subnet_id:
        raise ValueError(
            f"could not create subnet in project {project_id}: {subnet_result.stdout[:200]}"
        )
    if network_state_path:
        _write_json_file(
            network_state_path,
            {
                "project_id": project_id,
                "created_network_id": network_id,
                "subnet_id": subnet_id,
                "profile": profile,
            },
        )
    return subnet_id, network_id


def resolve_project_id(
    nebius_bin: str,
    tenant_id: str,
    project: ProjectSpec,
    *,
    prefix: str,
    create: bool,
    env: dict[str, str],
    region: str = "",
    profile: str = "",
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, bool]:
    """Return ``(project_id, created)`` for a project spec, creating if allowed."""

    if project.project_id:
        return project.project_id, False
    name = project.display_name(prefix)
    if not name:
        raise ValueError("project needs a name or project_id to resolve")
    existing = _list_projects(nebius_bin, tenant_id, env, profile)
    found = _find_project_id(existing, name)
    if found:
        verified = _get_project(nebius_bin, found, env, profile)
        _verify_existing_project(
            verified,
            project_id=found,
            tenant_id=tenant_id,
            name=name,
            region=region,
        )
        _log(on_status, f"project {name!r} exists ({found})")
        return found, False
    if not create:
        raise ValueError(
            f"project {name!r} not found under tenant and project creation is disabled"
        )
    _log(
        on_status,
        f"creating project {name!r} under tenant (region {region or 'default'})",
    )
    project_id = _create_project(
        nebius_bin, tenant_id, name, env, region=region, profile=profile
    )
    from npa.provisioning_journal import ProvisioningOperation, operation_context

    ownership = ProvisioningOperation.prepare(
        command="npa fleet deploy",
        project_alias=name,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        resource_type="project",
        requested_name=name,
        ownership_source="fleet-project-create",
        resume_command="npa fleet status",
        destroy_command="npa destroy --all --delete-project",
    )
    with operation_context(ownership):
        ownership.transition("mutating")
        ownership.record_resource(
            resource_type="nebius_project",
            requested_name=name,
            provider_id=project_id,
            ownership="created_by_this_operation",
            ownership_source="provider-create-response",
            project_id=project_id,
            labels={"tenant_id": tenant_id, "region": region},
        )
        ownership.transition("resource-created")
        ownership.transition("state-durable")
        ownership.commit()
    _log(on_status, f"created project {name!r} ({project_id})")
    return project_id, True


# --------------------------------------------------------------------------- #
# Per-cluster terraform materialization
# --------------------------------------------------------------------------- #


def _assert_terraform_version(terraform_bin: str) -> str:
    """Require the k8s-training recipe's Terraform >= 1.12 contract."""

    message = (
        "Terraform >= 1.12 is required by the k8s-training recipe; install a supported "
        "version or point NPA_TERRAFORM_BIN at one"
    )
    try:
        result = _run_capture(
            [terraform_bin, "version", "-json"], check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"{message} (version command failed: {type(exc).__name__})"
        ) from exc
    if result.returncode != 0:
        raise ValueError(f"{message} (version command exited {result.returncode})")
    try:
        payload = json.loads(result.stdout or "")
        version = str(payload.get("terraform_version") or "")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(
            f"{message} (could not parse 'terraform version -json')"
        ) from exc
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)([-+].+)?", version)
    if not match:
        raise ValueError(f"{message} (unparseable version {version!r})")
    core = tuple(int(match.group(i)) for i in range(1, 4))
    prerelease = bool(match.group(4) and match.group(4).startswith("-"))
    if core < _MIN_TERRAFORM_VERSION or (core == _MIN_TERRAFORM_VERSION and prerelease):
        raise ValueError(f"{message} (found {version})")
    return version


def _prewarm_plugin_cache(
    recipe_root: Path,
    *,
    region: str,
    cluster: ClusterSpec,
    ssh_public_key: str,
    work_root: Path,
    terraform_bin: str,
    nebius_bin: str,
    profile: str = "",
    on_status: Callable[[str], None] | None,
    log_path: Path | None = None,
) -> None:
    """Populate a shared terraform plugin cache with a single ``init`` before fan-out.

    Concurrent ``terraform init`` writes to a shared ``TF_PLUGIN_CACHE_DIR`` can
    corrupt provider binaries (a real failure hit in testing). Pre-warming the
    cache as the sole writer means the parallel per-cluster inits only *read* it.
    """

    cache_dir = Path(
        os.environ.get("TF_PLUGIN_CACHE_DIR") or (work_root / ".tf-plugin-cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set process-wide so every per-cluster env (via _terraform_env -> os.environ)
    # inherits the same warm cache.
    os.environ["TF_PLUGIN_CACHE_DIR"] = str(cache_dir)
    workdir = _prepare_install_dir(
        work_root / ".prewarm",
        recipe_root=recipe_root,
        region=region,
        cluster=cluster,
        ssh_public_key=ssh_public_key,
        on_status=None,
    )
    _log(on_status, f"pre-warming terraform provider cache at {cache_dir}")
    _tf_run(
        [terraform_bin, "init", "-input=false"],
        cwd=workdir,
        env=_terraform_env(nebius_bin, profile=profile),
        timeout=900,
        log_path=log_path,
    )


def _find_cluster_id_by_name(
    nebius_bin: str,
    project_id: str,
    cluster_name: str,
    env: dict[str, str],
    profile: str = "",
) -> str:
    result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "mk8s",
            "cluster",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not list Managed Kubernetes clusters (nebius exited {result.returncode})"
        )
    if not result.stdout.strip():
        raise RuntimeError(
            "could not list Managed Kubernetes clusters (nebius returned no JSON)"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "could not parse Managed Kubernetes cluster list JSON"
        ) from exc
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError(
            "Managed Kubernetes cluster list JSON has a non-list 'items' field"
        )
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata", {})
        if meta.get("name") == cluster_name and meta.get("id"):
            return str(meta["id"])
    return ""


# --------------------------------------------------------------------------- #
# Public lifecycle
# --------------------------------------------------------------------------- #
def _default_work_root() -> Path:
    return (Path.home() / ".npa" / "fleet").expanduser()


def plan_fleet(
    spec: FleetSpec,
    *,
    tenant_id: str | None = None,
    region: str | None = None,
    project_prefix: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Return the resolved deployment plan without touching infrastructure."""

    spec.validate()
    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    plan_profile = spec.profile if profile is None else profile
    # Show the tenant the deploy would actually use. Reporting
    # "(resolve-at-deploy)" while a named profile pins a specific tenant would
    # make plan useless as the pre-deploy check for targeting the right tenant.
    tenant = tenant_id or spec.tenant_id or tenant_id_from_profile(plan_profile)
    reg = region or spec.region
    plan_projects: list[dict[str, Any]] = []
    for project in spec.projects:
        clusters = []
        for cluster in project.clusters:
            if cluster.backend_name() == "soperator":
                assert cluster.soperator is not None
                resolved_soperator = replace(
                    cluster.soperator,
                    tenant_id=tenant or "",
                    project_id=project.project_id,
                    region=project.region or reg,
                )
                clusters.append(
                    {
                        "backend": "soperator",
                        **get_backend("soperator").plan(resolved_soperator),
                    }
                )
                continue
            backend_plan = get_backend("mk8s").plan(cluster)
            planned_cluster = {
                "name": backend_plan["name"],
                "cpu_nodes": backend_plan["cpu_nodes"],
                "cpu_preset": cluster.cpu_nodes.preset if cluster.cpu_nodes else "",
                "gpu_nodes": backend_plan["gpu_nodes"],
                "gpu_platform": backend_plan["gpu_platform"],
                "gpu_preset": backend_plan["gpu_preset"],
                "gpu_reservation": backend_plan["gpu_reservation"],
                "enable_gpu_cluster": backend_plan["enable_gpu_cluster"],
                "gpu_driver_mode": backend_plan["gpu_driver_mode"],
                "managed_driver_preset": backend_plan["managed_driver_preset"],
                "unsafe_nvswitch_operator": backend_plan["unsafe_nvswitch_operator"],
                "gpu_health_stabilization_seconds": backend_plan[
                    "gpu_health_stabilization_seconds"
                ],
                "gpu_cuda_smoke": backend_plan["gpu_cuda_smoke"],
                "enable_filestore": backend_plan["enable_filestore"],
                "filestore_disk_size_gibibytes": backend_plan[
                    "filestore_disk_size_gibibytes"
                ],
                "filestore_mount_path": backend_plan["filestore_mount_path"],
                "filestore_mount_tag": backend_plan["filestore_mount_tag"],
                "filesystem_csi_enabled": backend_plan[
                    "filesystem_csi_enabled"
                ],
                "k8s_version": backend_plan["k8s_version"],
                "mig": backend_plan["mig"],
            }
            if cluster.backend_explicit:
                planned_cluster["backend"] = "mk8s"
            clusters.append(planned_cluster)
        plan_projects.append(
            {
                "project_id": project.project_id or None,
                "display_name": project.display_name(prefix) or None,
                "will_create": not project.project_id,
                "clusters": clusters,
            }
        )
    backend_counts = {
        backend: sum(
            1
            for _project, cluster in spec.cluster_targets()
            if cluster.backend_name() == backend
        )
        for backend in ("mk8s", "soperator")
    }
    return {
        "name": spec.name,
        "tenant_id": tenant or "(resolve-at-deploy)",
        "region": reg or "(resolve-at-deploy)",
        "project_prefix": prefix,
        "profile": plan_profile or "(active)",
        "project_count": len(spec.projects),
        "cluster_count": len(spec.cluster_targets()),
        "backend_counts": backend_counts,
        "projects": plan_projects,
    }


def _persist_target_backend_owner(
    fleet_root: Path,
    *,
    fleet_name: str,
    project_key: str,
    cluster_name: str,
    expected_backend: str,
) -> None:
    """Fail closed on crash-residual backend evidence, then persist ownership."""

    target_root = fleet_root / project_key / cluster_name
    marker = target_root / _TARGET_BACKEND_OWNER
    evidence: list[dict[str, Any]] = []
    if marker.exists():
        try:
            payload = json.loads(marker.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise BackendOwnershipError(
                f"backend owner marker for {project_key}/{cluster_name} is unreadable"
            ) from exc
        if not isinstance(payload, dict):
            raise BackendOwnershipError(
                f"backend owner marker for {project_key}/{cluster_name} is malformed"
            )
        evidence.append(payload)
    mk8s_sidecar = target_root / ".npa-fleet-env.json"
    if mk8s_sidecar.exists():
        evidence.append({"backend": "mk8s"})
    soperator_sidecars = sorted(
        (target_root / "soperator").glob(
            "nebius-solutions-library*/soperator/installations/"
            + cluster_name
            + "/.npa-soperator-env.json"
        )
    )
    if len(soperator_sidecars) > 1:
        raise BackendOwnershipError(
            f"multiple soperator ownership sidecars exist for {project_key}/{cluster_name}"
        )
    if soperator_sidecars:
        evidence.append({"backend": "soperator"})
    discovered = {str(item.get("backend") or "mk8s") for item in evidence}
    if len(discovered) > 1 or (discovered and discovered != {expected_backend}):
        raise BackendOwnershipError(
            f"crash-residual state for {project_key}/{cluster_name} belongs to "
            f"backend(s) {sorted(discovered)}, but the spec selects {expected_backend!r}"
        )
    for item in evidence:
        for key, expected in (
            ("fleet_name", fleet_name),
            ("project_key", project_key),
            ("cluster_name", cluster_name),
        ):
            recorded = str(item.get(key) or "")
            if recorded and recorded != expected:
                raise BackendOwnershipError(
                    f"backend owner marker {key} does not match requested target"
                )
    _ensure_private_directory(fleet_root.parent)
    _ensure_private_directory(fleet_root)
    _ensure_private_directory(fleet_root / project_key)
    _ensure_private_directory(target_root)
    _write_json_file(
        marker,
        {
            "fleet_name": fleet_name,
            "project_key": project_key,
            "cluster_name": cluster_name,
            "backend": expected_backend,
        },
    )


def deploy_fleet(
    spec: FleetSpec,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch fleet targets through their shared backend adapters."""

    spec.validate()
    # Fail closed on every selected target's persisted owner before resolving
    # projects, creating shared networks, or invoking either backend.
    work_root = Path(kwargs.get("work_root") or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    state = _load_fleet_state(fleet_root)
    prior = {
        (str(item.get("project_key", "")), str(item.get("cluster_name", ""))): item
        for item in state.get("clusters", [])
        if isinstance(item, dict)
    }
    selected_prefix = kwargs.get("project_prefix")
    selected_prefix = (
        spec.project_prefix if selected_prefix is None else selected_prefix
    )
    selected_projects = kwargs.get("only_projects")
    selected_clusters = kwargs.get("only_clusters")
    for project, cluster in spec.cluster_targets():
        if not _project_in_scope(project, selected_projects, selected_prefix):
            continue
        if selected_clusters and cluster.name not in selected_clusters:
            continue
        saved = prior.get((project.key(), cluster.name))
        if saved is not None:
            require_backend_ownership(saved, cluster.backend_name())
        _persist_target_backend_owner(
            fleet_root,
            fleet_name=spec.name,
            project_key=project.key(),
            cluster_name=cluster.name,
            expected_backend=cluster.backend_name(),
        )
    mk8s_projects: list[ProjectSpec] = []
    soperator_targets: list[tuple[ProjectSpec, ClusterSpec]] = []
    for project in spec.projects:
        mk8s_clusters = [
            cluster for cluster in project.clusters if cluster.backend_name() == "mk8s"
        ]
        if mk8s_clusters:
            mk8s_projects.append(replace(project, clusters=mk8s_clusters))
        soperator_targets.extend(
            (project, cluster)
            for cluster in project.clusters
            if cluster.backend_name() == "soperator"
        )

    result: dict[str, Any] | None = None
    if mk8s_projects:
        result = _deploy_mk8s_fleet(replace(spec, projects=mk8s_projects), **kwargs)
    if not soperator_targets:
        assert result is not None
        result["backend_counts"] = _backend_counts(result.get("clusters", []))
        return result

    only_projects = kwargs.get("only_projects")
    only_clusters = kwargs.get("only_clusters")
    prefix = kwargs.get("project_prefix")
    prefix = spec.project_prefix if prefix is None else prefix
    profile = kwargs.get("profile")
    profile = spec.profile if profile is None else profile
    on_status = kwargs.get("on_status")
    create_projects = bool(kwargs.get("create_projects", True))
    continue_on_error = bool(kwargs.get("continue_on_error", True))
    timeout_minutes = int(kwargs.get("timeout_minutes", 120))
    stream_terraform = bool(kwargs.get("stream_terraform", True))
    _ensure_private_directory(fleet_root)
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    tenant_id = _resolve_tenant_id(nebius_bin, spec.tenant_id, profile)
    fleet_region = _resolve_region(spec.region)
    cli_env = _nebius_cli_env()
    if profile:
        cli_env["NEBIUS_PROFILE"] = profile
        cli_env["NPA_NEBIUS_PROFILE"] = profile
    soperator_results: list[dict[str, Any]] = []
    prepared: list[tuple[ProjectSpec, ClusterSpec, Any, str, str, Path]] = []
    for project, cluster in soperator_targets:
        if not _project_in_scope(project, only_projects, prefix):
            continue
        if only_clusters and cluster.name not in only_clusters:
            continue
        desired = cluster.soperator
        assert desired is not None
        key = (project.key(), cluster.name)
        if key in prior:
            require_backend_ownership(prior[key], "soperator")
        region = project.region or fleet_region
        try:
            project_id, _created = resolve_project_id(
                nebius_bin,
                tenant_id,
                project,
                prefix=prefix,
                create=create_projects,
                env=cli_env,
                region=region,
                profile=profile,
                on_status=on_status,
            )
            resolved = replace(
                desired,
                region=region,
                tenant_id=tenant_id,
                project_id=project_id,
                subnet_id="",
            )
            backend_root = fleet_root / project.key() / cluster.name / "soperator"
            if any(pool.capacity_mode() == "reserved" for pool in resolved.workers):
                preflight = get_backend("soperator").preflight(
                    resolved,
                    SoperatorApplyRequest(
                        provider_preflight=True,
                        provider_nebius_bin=nebius_bin,
                        provider_tenant_id=tenant_id,
                        provider_project_id=project_id,
                        provider_region=region,
                        provider_install_dir=backend_root
                        / "installations"
                        / cluster.name,
                        provider_work_root=backend_root,
                        provider_env=cli_env,
                        on_status=on_status,
                    ),
                )
                resolved = preflight.get("resolved_desired", resolved)
            subnet_id = desired.subnet_id
            if not subnet_id:
                subnet_id, _network = ensure_subnet(
                    nebius_bin,
                    project_id,
                    name_stem=project.key(),
                    env=cli_env,
                    profile=profile,
                    network_state_path=fleet_root
                    / project.key()
                    / _PROJECT_NETWORK_STATE,
                    on_status=on_status,
                )
            resolved = replace(resolved, subnet_id=subnet_id)
            prepared.append(
                (project, cluster, resolved, project_id, region, backend_root)
            )
        except Exception as exc:  # noqa: BLE001 - aggregate target preparation
            if not continue_on_error:
                raise
            soperator_results.append(
                {
                    "backend": "soperator",
                    "project_key": project.key(),
                    "cluster_name": cluster.name,
                    "region": region,
                    "status": "error",
                    "error": str(exc),
                }
            )

    def _apply_soperator_target(
        target: tuple[ProjectSpec, ClusterSpec, Any, str, str, Path],
    ) -> dict[str, Any]:
        project, cluster, resolved, project_id, region, backend_root = target
        try:
            deployed = get_backend("soperator").apply(
                resolved,
                SoperatorApplyRequest(
                    work_root=backend_root,
                    timeout_minutes=timeout_minutes,
                    stream_terraform_output=stream_terraform,
                    on_status=on_status,
                    profile=profile,
                ),
            )
            return {
                **deployed,
                "backend": "soperator",
                "project_key": project.key(),
                "project_id": project_id,
                "cluster_name": cluster.name,
                "region": region,
                "backend_state_root": str(backend_root),
            }
        except (SoperatorDeploymentValidationError, SoperatorStateCaptureError) as exc:
            # Terraform applied, but a real post-deploy validation gate failed.
            # Preserve the native degraded result and canonical recovery root.
            item = {
                **exc.result,
                "backend": "soperator",
                "project_key": project.key(),
                "cluster_name": cluster.name,
                "region": region,
                "backend_state_root": str(
                    fleet_root / project.key() / cluster.name / "soperator"
                ),
            }
            if not continue_on_error:
                raise
            return item
        except Exception as exc:  # noqa: BLE001 - aggregate per-target failure
            if not continue_on_error:
                raise
            return {
                "backend": "soperator",
                "project_key": project.key(),
                "cluster_name": cluster.name,
                "region": region,
                "status": "error",
                "error": str(exc),
            }

    concurrency = max(1, int(kwargs.get("concurrency", 1)))
    if concurrency > 1 and len(prepared) > 1 and continue_on_error:
        _log(
            on_status,
            f"deploying {len(prepared)} soperator target(s) with concurrency={concurrency}",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            soperator_results.extend(
                future.result()
                for future in concurrent.futures.as_completed(
                    [
                        pool.submit(_apply_soperator_target, target)
                        for target in prepared
                    ]
                )
            )
    else:
        for target in prepared:
            try:
                soperator_results.append(_apply_soperator_target(target))
            except (
                SoperatorDeploymentValidationError,
                SoperatorStateCaptureError,
            ) as exc:
                item = {
                    **exc.result,
                    "backend": "soperator",
                    "project_key": target[0].key(),
                    "cluster_name": target[1].name,
                    "region": target[4],
                    "backend_state_root": str(target[5]),
                }
                soperator_results.append(item)
                _upsert_fleet_state(fleet_root, {}, soperator_results)
                raise

    if result is None:
        result = {
            "name": spec.name,
            "tenant_id": tenant_id,
            "region": fleet_region,
            "project_prefix": prefix,
            "profile": profile,
            "clusters": [],
        }
    result["clusters"] = [*result.get("clusters", []), *soperator_results]
    result.update(_recount(result["clusters"]))
    result["backend_counts"] = _backend_counts(result["clusters"])
    _upsert_fleet_state(
        fleet_root,
        {key: value for key, value in result.items() if key != "clusters"},
        soperator_results,
    )
    return result


def _deploy_mk8s_fleet(
    spec: FleetSpec,
    *,
    k8s_training_dir: Path | None = None,
    k8s_training_ref: str | None = None,
    work_root: Path | None = None,
    project_prefix: str | None = None,
    create_projects: bool = True,
    only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None,
    timeout_minutes: int = 120,
    continue_on_error: bool = True,
    concurrency: int = 1,
    profile: str | None = None,
    preflight: bool = True,
    repair_stopped_placeholder: bool = False,
    stream_terraform: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy every ``(project, cluster)`` target in *spec*. Returns fleet metadata.

    ``only_projects`` / ``only_clusters`` narrow the action to a subset so callers
    can add one or many specific projects/clusters without touching the rest of a
    fleet (deploy is idempotent: existing clusters reconcile in place).

    ``concurrency`` > 1 applies that many clusters in parallel (each has its own
    isolated terraform state, so there is no cross-cluster lock contention). The
    provider plugin cache is pre-warmed once to avoid concurrent-init corruption,
    and each cluster streams to its own ``<install_dir>/deploy.log``. Project
    subnet discovery/creation is resolved once, sequentially, before the
    parallel apply phase. Clusters without an explicit ``subnet_id`` therefore
    share one authoritative project subnet without a create race; explicit
    per-cluster overrides remain unchanged.

    ``profile`` overrides the spec's ``profile``: every ``nebius`` CLI call and
    the minted terraform IAM token then use that ``~/.nebius`` profile, which is
    how one workstation deploys into several tenants (a service account is
    single-tenant) without switching the machine-wide active profile.

    ``preflight`` (default on) validates explicitly bound capacity blocks and
    compares the remaining on-demand requirements against tenant quota
    allowances before any apply. mk8s accepts a node group it cannot fill, so
    without this a capacity/quota wall shows up as terraform blocking on
    ``Still creating...`` until the timeout.

    ``stream_terraform`` defaults to the historical human-facing behavior.
    Machine-readable callers can set it false to keep stdout untouched and
    retain Terraform diagnostics in restrictive per-cluster logs.
    """

    spec.validate()
    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    nebius_profile = spec.profile if profile is None else profile
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    _assert_terraform_version(terraform_bin)
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    tenant_id = _resolve_tenant_id(nebius_bin, spec.tenant_id, nebius_profile)
    fleet_region = _resolve_region(spec.region)
    ssh_public_key = _resolve_ssh_public_key(spec.ssh_public_key)

    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    work_root.mkdir(parents=True, exist_ok=True)
    _ensure_private_directory(fleet_root)
    recipe_root = _resolve_recipe_root(
        k8s_training_dir, ref=k8s_training_ref, work_root=work_root, on_status=on_status
    )
    _log(on_status, f"k8s-training recipe root: {recipe_root}")
    for project in spec.projects:
        if not _project_in_scope(project, only_projects, prefix):
            continue
        for cluster in project.clusters:
            if only_clusters and cluster.name not in only_clusters:
                continue
            validate_recipe_mig_compatibility(
                cluster, recipe_root / _K8S_TRAINING_SUBDIR
            )

    cli_env = _nebius_cli_env()
    results: list[dict[str, Any]] = []

    base_meta = {
        "name": spec.name,
        "tenant_id": tenant_id,
        "region": fleet_region,
        "project_prefix": prefix,
        "profile": nebius_profile,
        "k8s_training_source": str(recipe_root),
    }

    # Phase 0: quota preflight, before any project is created. An exact
    # provider-present target whose saved rendered shape still matches the spec
    # has zero incremental demand. Everything else is conservatively treated as
    # new capacity, so missing/stale local state can never bypass preflight or
    # leave a newly created project behind when quota is unavailable.
    preflight_project_ids: dict[str, str] = {}
    if preflight:
        scoped: dict[str, list[ClusterSpec]] = {}
        new_projects_by_region: dict[str, int] = {}
        scoped_projects: list[tuple[ProjectSpec, str]] = []
        for project in spec.projects:
            if not _project_in_scope(project, only_projects, prefix):
                continue
            region = project.region or fleet_region
            project_has_scoped_cluster = False
            for cluster in project.clusters:
                if only_clusters and cluster.name not in only_clusters:
                    continue
                if _is_verified_unchanged_target(
                    project=project,
                    cluster=cluster,
                    prefix=prefix,
                    tenant_id=tenant_id,
                    region=region,
                    ssh_public_key=ssh_public_key,
                    fleet_root=fleet_root,
                    nebius_bin=nebius_bin,
                    profile=nebius_profile,
                    env=cli_env,
                ):
                    _log(
                        on_status,
                        f"capacity/quota preflight: {project.key()}/{cluster.name} "
                        "is provider-verified and unchanged; incremental demand is zero",
                    )
                    continue
                scoped.setdefault(region, []).append(cluster)
                project_has_scoped_cluster = True
            if project_has_scoped_cluster:
                scoped_projects.append((project, region))

        named_projects = [
            project for project, _region in scoped_projects if not project.project_id
        ]
        existing_projects = (
            _list_projects(nebius_bin, tenant_id, cli_env, nebius_profile)
            if named_projects
            else []
        )
        for project, region in scoped_projects:
            if project.project_id:
                continue
            name = project.display_name(prefix)
            found = _find_project_id(existing_projects, name)
            if found:
                preflight_project_ids[project.key()] = found
                _log(on_status, f"project {name!r} exists ({found})")
            else:
                new_projects_by_region[region] = (
                    new_projects_by_region.get(region, 0) + 1
                )
        if scoped:
            _preflight_quotas(
                nebius_bin,
                tenant_id=tenant_id,
                by_region=scoped,
                new_projects_by_region=new_projects_by_region,
                env=cli_env,
                profile=nebius_profile,
                on_status=on_status,
            )

    # Phase 1 (sequential, cheap): resolve/create each in-scope project and build
    # the flat list of (project, cluster) targets to apply.
    targets: list[dict[str, Any]] = []
    for project in spec.projects:
        if not _project_in_scope(project, only_projects, prefix):
            continue
        scoped_clusters = [
            cluster
            for cluster in project.clusters
            if not (only_clusters and cluster.name not in only_clusters)
        ]
        if not scoped_clusters:
            continue
        region = project.region or fleet_region
        try:
            if project.key() in preflight_project_ids:
                project_id = preflight_project_ids[project.key()]
                created = False
            else:
                project_id, created = resolve_project_id(
                    nebius_bin,
                    tenant_id,
                    project,
                    prefix=prefix,
                    create=create_projects,
                    env=cli_env,
                    region=region,
                    profile=nebius_profile,
                    on_status=on_status,
                )
            shared_subnet_id = ""
            if any(not cluster.subnet_id for cluster in scoped_clusters):
                shared_subnet_id, _created_network_id = ensure_subnet(
                    nebius_bin,
                    project_id,
                    name_stem=project.key(),
                    env=cli_env,
                    profile=nebius_profile,
                    network_state_path=fleet_root
                    / project.key()
                    / _PROJECT_NETWORK_STATE,
                    on_status=on_status,
                )
        except Exception as exc:  # noqa: BLE001 - report and continue
            _log(on_status, f"project {project.key()} FAILED to resolve: {exc}")
            if not continue_on_error:
                raise
            results.append(
                {"project_key": project.key(), "status": "error", "error": str(exc)}
            )
            continue
        for cluster in scoped_clusters:
            targets.append(
                {
                    "project": project,
                    "cluster": cluster,
                    "project_id": project_id,
                    "created": created,
                    "region": region,
                    "subnet_id": cluster.subnet_id or shared_subnet_id,
                }
            )

    parallel = concurrency > 1 and len(targets) > 1
    if parallel:
        prewarm_log = None
        if not stream_terraform:
            diagnostics_root = fleet_root / ".logs"
            _ensure_private_directory(diagnostics_root)
            prewarm_log = diagnostics_root / "terraform-prewarm.log"
        _prewarm_plugin_cache(
            recipe_root,
            region=targets[0]["region"],
            cluster=targets[0]["cluster"],
            ssh_public_key=ssh_public_key,
            work_root=work_root,
            terraform_bin=terraform_bin,
            nebius_bin=nebius_bin,
            profile=nebius_profile,
            on_status=on_status,
            log_path=prewarm_log,
        )

    def _run_target(t: dict[str, Any]) -> dict[str, Any]:
        log_path = None
        if parallel or not stream_terraform:
            log_path = (
                fleet_root / t["project"].key() / t["cluster"].name / "deploy.log"
            )
        if (
            _deploy_one_cluster is not _LEGACY_DEPLOY_COMPAT
            or _legacy_helpers_patched()
        ):
            return _deploy_one_cluster(
                spec=spec,
                project=t["project"],
                cluster=t["cluster"],
                project_id=t["project_id"],
                project_created=t["created"],
                subnet_id=t["subnet_id"],
                region=t["region"],
                tenant_id=tenant_id,
                ssh_public_key=ssh_public_key,
                fleet_root=fleet_root,
                recipe_root=recipe_root,
                terraform_bin=terraform_bin,
                nebius_bin=nebius_bin,
                profile=nebius_profile,
                timeout_minutes=timeout_minutes,
                on_status=on_status,
                log_path=log_path,
                repair_stopped_placeholder=repair_stopped_placeholder,
            )
        return get_backend("mk8s").apply(
            t["cluster"],
            MK8sApplyRequest(
                scope=MK8sExecutionScope(
                    fleet_name=spec.name,
                    tenant_id=tenant_id,
                    region=t["region"],
                    project_prefix=prefix,
                ),
                project=MK8sProjectIdentity(
                    project_key=t["project"].key(),
                    project_id=t["project_id"],
                    project_name=t["project"].name,
                    expected_provider_name=t["project"].display_name(prefix),
                ),
                project_id=t["project_id"],
                project_created=t["created"],
                subnet_id=t["subnet_id"],
                region=t["region"],
                tenant_id=tenant_id,
                ssh_public_key=ssh_public_key,
                fleet_root=fleet_root,
                recipe_root=recipe_root,
                terraform_bin=terraform_bin,
                nebius_bin=nebius_bin,
                profile=nebius_profile,
                timeout_minutes=timeout_minutes,
                on_status=on_status,
                log_path=log_path,
                repair_stopped_placeholder=repair_stopped_placeholder,
            ),
        )

    # Phase 2: apply -- sequentially (live stdout) or in a bounded thread pool.
    if parallel:
        _log(
            on_status,
            f"applying {len(targets)} cluster(s) with concurrency={concurrency} "
            "(per-cluster output in <install_dir>/deploy.log)",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run_target, t) for t in targets]
            for fut in concurrent.futures.as_completed(futures):
                results.append(
                    fut.result()
                )  # _deploy_one_cluster captures its own errors
    else:
        for t in targets:
            entry = _run_target(t)
            results.append(entry)
            if entry.get("status") == "error" and not continue_on_error:
                _upsert_fleet_state(fleet_root, base_meta, results)
                raise RuntimeError(entry.get("error") or "cluster deploy failed")

    result = {**base_meta, "clusters": results, **_recount(results)}
    # Persist a merged view so a targeted deploy doesn't clobber untouched clusters.
    _upsert_fleet_state(fleet_root, base_meta, results)
    if not continue_on_error and result["failed"]:
        raise RuntimeError(f"{result['failed']} cluster(s) failed")
    return result


def _preflight_quotas(
    nebius_bin: str,
    *,
    tenant_id: str,
    by_region: dict[str, list[ClusterSpec]],
    new_projects_by_region: dict[str, int],
    env: dict[str, str],
    profile: str,
    on_status: Callable[[str], None] | None,
) -> None:
    """Raise when reservations or tenant quota cannot cover the scoped clusters."""

    shortfalls = []
    for region, clusters in sorted(by_region.items()):
        _log(
            on_status,
            f"capacity/quota preflight: {len(clusters)} cluster(s) in {region}",
        )
        shortfalls += preflight_region(
            nebius_bin=nebius_bin,
            tenant_id=tenant_id,
            region=region,
            clusters=clusters,
            new_projects=new_projects_by_region.get(region, 0),
            env=env,
            profile=profile,
            run_capture=_run_capture,
            nebius_argv=_nebius_argv,
            on_status=on_status,
        )
    if shortfalls:
        raise ValueError(shortfall_message(shortfalls, tenant_id))


def _write_fleet_state(fleet_root: Path, result: dict[str, Any]) -> None:
    _write_json_file(fleet_root / _FLEET_STATE, result)


def _load_fleet_state(fleet_root: Path) -> dict[str, Any]:
    path = fleet_root / _FLEET_STATE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise RuntimeError(
                f"persisted fleet inventory at {path} is not an object; refusing lifecycle action"
            )
        return data
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"persisted fleet inventory at {path} is unreadable; refusing lifecycle action"
        ) from exc


def _recount(clusters: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "deployed": sum(1 for c in clusters if c.get("status") == "deployed"),
        "failed": sum(
            1
            for c in clusters
            if c.get("status") not in {"deployed", "destroyed", "absent"}
        ),
    }


def _backend_counts(clusters: list[dict[str, Any]]) -> dict[str, int]:
    """Return stable backend keys for pure and mixed fleet results."""

    return {
        backend: sum(1 for item in clusters if item.get("backend", "mk8s") == backend)
        for backend in ("mk8s", "soperator")
    }


def _upsert_fleet_state(
    fleet_root: Path, base_meta: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    """Merge this run's cluster results into the persisted fleet summary.

    A targeted (``only_projects``/``only_clusters``) deploy must not clobber the
    recorded state of clusters it did not touch, so entries are upserted by
    ``(project_key, cluster_name)`` rather than overwritten wholesale.
    """

    state = _load_fleet_state(fleet_root)
    clusters = (
        state.get("clusters", []) if isinstance(state.get("clusters"), list) else []
    )
    index = {
        (c.get("project_key"), c.get("cluster_name")): i for i, c in enumerate(clusters)
    }
    for entry in results:
        entry = {"backend": str(entry.get("backend") or "mk8s"), **entry}
        key = (entry.get("project_key"), entry.get("cluster_name"))
        if key[1] is None:  # project-level failure (no cluster) -- don't persist
            continue
        if key in index:
            require_backend_ownership(clusters[index[key]], entry["backend"])
            clusters[index[key]] = entry
        else:
            index[key] = len(clusters)
            clusters.append(entry)
    _write_fleet_state(
        fleet_root,
        {
            **base_meta,
            "clusters": clusters,
            **_recount(clusters),
            "backend_counts": _backend_counts(clusters),
        },
    )


def _prune_fleet_state(fleet_root: Path, removed_keys: set[tuple[str, str]]) -> None:
    """Drop destroyed ``(project_key, cluster_name)`` entries from the summary."""

    for project_key, cluster_name in removed_keys:
        (fleet_root / project_key / cluster_name / _TARGET_BACKEND_OWNER).unlink(
            missing_ok=True
        )
    state = _load_fleet_state(fleet_root)
    clusters = (
        state.get("clusters", []) if isinstance(state.get("clusters"), list) else []
    )
    kept = [
        c
        for c in clusters
        if (c.get("project_key"), c.get("cluster_name")) not in removed_keys
    ]
    _write_fleet_state(
        fleet_root,
        {
            **state,
            "clusters": kept,
            **_recount(kept),
            "backend_counts": _backend_counts(kept),
        },
    )


def _project_has_persisted_targets(fleet_root: Path, project_key: str) -> bool:
    """Return whether inventory proves any backend still owns this project network."""

    state = _load_fleet_state(fleet_root)
    return any(
        isinstance(item, dict) and str(item.get("project_key") or "") == project_key
        for item in state.get("clusters", [])
    )


def _reclaim_unused_project_networks(
    spec: FleetSpec,
    *,
    fleet_root: Path,
    nebius_bin: str,
    prefix: str,
    only_projects: list[str] | None,
    profile: str | None,
    on_status: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    """Reclaim fleet-owned networks only after every backend target is absent.

    Inventory is authoritative across backend-specific directory layouts. Local
    sidecars are an additional fail-closed recovery check, never a substitute
    for the fleet ownership record.
    """

    network_results: list[dict[str, Any]] = []
    for project in spec.projects:
        if not _project_in_scope(project, only_projects, prefix):
            continue
        if _project_has_persisted_targets(fleet_root, project.key()):
            continue
        project_root = fleet_root / project.key()
        if project_root.exists() or project_root.is_symlink():
            try:
                _ensure_private_directory(project_root)
                has_backend_state = any(
                    child.is_dir()
                    and (
                        (child / _ENV_SIDECAR).exists()
                        or (child / "soperator").exists()
                    )
                    for child in project_root.iterdir()
                )
            except (OSError, RuntimeError) as exc:
                network_results.append(
                    {
                        "project_key": project.key(),
                        "status": "destroy-incomplete",
                        "errors": [
                            "could not safely inspect project recovery state: "
                            f"{type(exc).__name__}"
                        ],
                    }
                )
                continue
        else:
            has_backend_state = False
        if has_backend_state:
            continue
        network_state_path = project_root / _PROJECT_NETWORK_STATE
        network_state = _load_json_file(network_state_path)
        network_id = str(network_state.get("created_network_id") or "")
        project_id = str(network_state.get("project_id") or project.project_id)
        if not network_id or not project_id:
            continue
        cleanup_profile = (spec.profile if profile is None else profile) or str(
            network_state.get("profile") or ""
        )
        errors = _reclaim_created_network(
            nebius_bin,
            project_id,
            network_id,
            str(network_state.get("subnet_id") or ""),
            _nebius_cli_env(),
            on_status,
            project.key(),
            profile=cleanup_profile,
        )
        if errors:
            network_results.append(
                {
                    "project_key": project.key(),
                    "status": "destroy-incomplete",
                    "errors": errors,
                }
            )
            continue
        try:
            network_state_path.unlink(missing_ok=True)
        except OSError as exc:
            network_results.append(
                {
                    "project_key": project.key(),
                    "status": "destroy-incomplete",
                    "errors": [
                        "cloud network teardown succeeded but local ownership "
                        f"cleanup failed: {type(exc).__name__}"
                    ],
                }
            )
        else:
            network_results.append(
                {"project_key": project.key(), "status": "destroyed"}
            )
    return network_results


def destroy_fleet(
    spec: FleetSpec,
    **kwargs: Any,
) -> dict[str, Any]:
    """Destroy each target only through the backend recorded in inventory."""

    spec.validate()
    work_root = Path(kwargs.get("work_root") or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    state = _load_fleet_state(fleet_root)
    persisted = {
        (str(item.get("project_key", "")), str(item.get("cluster_name", ""))): item
        for item in state.get("clusters", [])
        if isinstance(item, dict)
    }
    prefix = kwargs.get("project_prefix")
    prefix = spec.project_prefix if prefix is None else prefix
    only_projects = kwargs.get("only_projects")
    only_clusters = kwargs.get("only_clusters")
    mk8s_projects: list[ProjectSpec] = []
    sop_targets: list[tuple[ProjectSpec, ClusterSpec]] = []
    for project in spec.projects:
        selected = [
            cluster
            for cluster in project.clusters
            if _project_in_scope(project, only_projects, prefix)
            and not (only_clusters and cluster.name not in only_clusters)
        ]
        for cluster in selected:
            saved = persisted.get((project.key(), cluster.name))
            if saved is not None:
                require_backend_ownership(saved, cluster.backend_name())
        mk8s = [cluster for cluster in selected if cluster.backend_name() == "mk8s"]
        if mk8s:
            mk8s_projects.append(replace(project, clusters=mk8s))
        sop_targets.extend(
            (project, cluster)
            for cluster in selected
            if cluster.backend_name() == "soperator"
        )

    results: list[dict[str, Any]] = []
    if mk8s_projects:
        mk8s_kwargs = dict(kwargs)
        # Selection is already represented by the subset spec.
        mk8s_kwargs["only_projects"] = None
        mk8s_kwargs["only_clusters"] = None
        mk_result = _destroy_mk8s_fleet(
            replace(spec, projects=mk8s_projects), **mk8s_kwargs
        )
        if not sop_targets:
            return mk_result
        results.extend(mk_result.get("clusters", []))

    def _destroy_soperator_target(
        target: tuple[ProjectSpec, ClusterSpec],
    ) -> tuple[dict[str, Any], tuple[str, str] | None]:
        project, cluster = target
        desired = cluster.soperator
        assert desired is not None
        saved = persisted.get((project.key(), cluster.name), {})
        canonical_root = fleet_root / project.key() / cluster.name / "soperator"
        backend_root_text = str(saved.get("backend_state_root") or "")
        if (
            backend_root_text
            and Path(backend_root_text).expanduser().resolve()
            != canonical_root.resolve()
        ):
            raise ValueError(
                "persisted soperator backend_state_root does not match the "
                f"canonical fleet-owned root for {project.key()}/{cluster.name}"
            )
        backend_root = canonical_root
        try:
            native = get_backend("soperator").destroy(
                desired,
                SoperatorDestroyRequest(
                    work_root=backend_root,
                    timeout_minutes=int(kwargs.get("timeout_minutes", 120)),
                    on_status=kwargs.get("on_status"),
                    profile=(
                        spec.profile
                        if kwargs.get("profile") is None
                        else str(kwargs.get("profile") or "")
                    ),
                ),
            )
            if not native or native.get("status") != "destroyed":
                return (
                    {
                        "backend": "soperator",
                        "project_key": project.key(),
                        "cluster_name": cluster.name,
                        "status": "destroy-incomplete",
                        "error": str(
                            (native or {}).get("errors")
                            or "native destroy did not prove exact provider absence"
                        ),
                    },
                    None,
                )
            # Native destroy deliberately retains its source/install tree for
            # standalone recovery. Fleet inventory is the recovery authority;
            # after an authoritative successful destroy, remove only this
            # canonical target root so shared-network cleanup can prove absence.
            if backend_root.exists():
                shutil.rmtree(backend_root)
            cluster_root = backend_root.parent
            if cluster_root.exists() and not any(cluster_root.iterdir()):
                cluster_root.rmdir()
            item = {
                "backend": "soperator",
                "project_key": project.key(),
                "cluster_name": cluster.name,
                "status": "destroyed",
            }
            removed_key: tuple[str, str] | None = (project.key(), cluster.name)
        except Exception as exc:  # noqa: BLE001 - best-effort fleet aggregation
            item = {
                "backend": "soperator",
                "project_key": project.key(),
                "cluster_name": cluster.name,
                "status": "destroy-incomplete",
                "error": str(exc),
            }
            removed_key = None
        return item, removed_key

    removed: set[tuple[str, str]] = set()
    concurrency = max(1, int(kwargs.get("concurrency", 1)))
    if concurrency > 1 and len(sop_targets) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = [
                future.result()
                for future in concurrent.futures.as_completed(
                    [
                        pool.submit(_destroy_soperator_target, target)
                        for target in sop_targets
                    ]
                )
            ]
    else:
        outcomes = [_destroy_soperator_target(target) for target in sop_targets]
    for item, removed_key in outcomes:
        results.append(item)
        if removed_key is not None:
            removed.add(removed_key)
    if removed:
        _prune_fleet_state(fleet_root, removed)
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    network_results = _reclaim_unused_project_networks(
        spec,
        fleet_root=fleet_root,
        nebius_bin=nebius_bin,
        prefix=prefix,
        only_projects=only_projects,
        profile=kwargs.get("profile"),
        on_status=kwargs.get("on_status"),
    )
    failed = sum(
        1 for entry in results if entry.get("status") == "destroy-incomplete"
    ) + sum(
        1 for entry in network_results if entry.get("status") == "destroy-incomplete"
    )
    return {
        "name": spec.name,
        "clusters": results,
        "networks": network_results,
        **_recount(results),
        "backend_counts": _backend_counts(results),
        "failed": failed,
    }


def _destroy_mk8s_fleet(
    spec: FleetSpec,
    *,
    work_root: Path | None = None,
    project_prefix: str | None = None,
    only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None,
    timeout_minutes: int = 120,
    concurrency: int = 1,
    profile: str | None = None,
    stream_terraform: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Destroy the fleet's clusters (best-effort, per-target).

    Tears down each cluster **declared in the spec** that has a local install dir
    (its own terraform state), and reclaims any VPC network the fleet created.
    ``only_projects`` / ``only_clusters`` narrow the teardown to a subset so one or
    many specific clusters/projects can be removed without touching the rest.
    ``concurrency`` > 1 tears down that many clusters in parallel (each has its own
    state, so there is no lock contention). This does not enumerate clusters via
    the API, so a cluster created out-of-band is not reclaimed here.

    ``profile`` overrides the spec's ``profile``; when neither is set the profile
    recorded in each cluster's env sidecar at deploy time is used, so a teardown
    authenticates as the same principal that created the cluster.

    ``stream_terraform`` defaults to the historical human-facing behavior.
    Machine-readable callers can set it false to retain diagnostics under the
    fleet's private ``.logs`` tree without writing Terraform output to stdout.
    """

    spec.validate()
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    _assert_terraform_version(terraform_bin)
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    if fleet_root.exists() or fleet_root.is_symlink():
        _ensure_private_directory(fleet_root)

    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    targets: list[tuple[ProjectSpec, ClusterSpec]] = [
        (project, cluster)
        for project in spec.projects
        if _project_in_scope(project, only_projects, prefix)
        for cluster in project.clusters
        if not (only_clusters and cluster.name not in only_clusters)
    ]

    # Migrate legacy per-cluster network ownership into the project-level record
    # before any cluster directory can be removed. The project record prevents
    # two clusters sharing a subnet from both claiming/deleting the same VPC.
    for project, cluster in targets:
        install_dir = fleet_root / project.key() / cluster.name
        if install_dir.exists() or install_dir.is_symlink():
            try:
                _ensure_private_directory(install_dir.parent)
                _ensure_private_directory(install_dir)
            except RuntimeError:
                # _destroy_one_cluster reports the unsafe path as retained
                # per-cluster recovery state; never inspect through the link.
                continue
        saved = _load_env_sidecar(install_dir) or {}
        created_network_id = str(saved.get("created_network_id") or "")
        network_state_path = fleet_root / project.key() / _PROJECT_NETWORK_STATE
        if created_network_id and not network_state_path.exists():
            _write_json_file(
                network_state_path,
                {
                    "project_id": str(saved.get("project_id") or project.project_id),
                    "created_network_id": created_network_id,
                    "subnet_id": str(saved.get("subnet_id") or ""),
                    "profile": str(saved.get("profile") or spec.profile),
                },
            )

    parallel = concurrency > 1 and len(targets) > 1

    def _run(target: tuple[ProjectSpec, ClusterSpec]) -> dict[str, Any]:
        project, cluster = target
        log_path = None
        if parallel or not stream_terraform:
            # Successful destroy removes <install_dir>, so diagnostics live in
            # a sibling private tree that survives authoritative state cleanup.
            log_path = (
                fleet_root / ".logs" / project.key() / cluster.name / "destroy.log"
            )
        if (
            _destroy_one_cluster is not _LEGACY_DESTROY_COMPAT
            or _legacy_helpers_patched()
        ):
            return _destroy_one_cluster(
                spec=spec,
                project=project,
                cluster=cluster,
                fleet_root=fleet_root,
                terraform_bin=terraform_bin,
                nebius_bin=nebius_bin,
                profile=spec.profile if profile is None else profile,
                timeout_minutes=timeout_minutes,
                on_status=on_status,
                log_path=log_path,
            )
        result = get_backend("mk8s").destroy(
            cluster,
            MK8sDestroyRequest(
                scope=MK8sExecutionScope(
                    fleet_name=spec.name,
                    tenant_id=spec.tenant_id,
                    region=spec.region,
                    project_prefix=prefix,
                ),
                project=MK8sProjectIdentity(
                    project_key=project.key(),
                    project_id=project.project_id,
                    project_name=project.name,
                    expected_provider_name=project.display_name(prefix),
                ),
                fleet_root=fleet_root,
                terraform_bin=terraform_bin,
                nebius_bin=nebius_bin,
                profile=spec.profile if profile is None else profile,
                timeout_minutes=timeout_minutes,
                on_status=on_status,
                log_path=log_path,
            ),
        )
        return result or {
            "backend": "mk8s",
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "absent",
        }

    if parallel:
        _log(
            on_status,
            f"destroying {len(targets)} cluster(s) with concurrency={concurrency}",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            destroyed = [
                f.result()
                for f in concurrent.futures.as_completed(
                    [pool.submit(_run, t) for t in targets]
                )
            ]
    else:
        destroyed = [_run(t) for t in targets]

    removed_keys = {
        (d["project_key"], d["cluster_name"])
        for d in destroyed
        if d.get("status") == "destroyed"
    }
    if removed_keys:
        _prune_fleet_state(fleet_root, removed_keys)
    incomplete = [
        entry for entry in destroyed if entry.get("status") == "destroy-incomplete"
    ]
    if incomplete:
        previous = _load_fleet_state(fleet_root)
        base_meta = {key: value for key, value in previous.items() if key != "clusters"}
        base_meta.setdefault("name", spec.name)
        _upsert_fleet_state(fleet_root, base_meta, incomplete)

    network_results = _reclaim_unused_project_networks(
        spec,
        fleet_root=fleet_root,
        nebius_bin=nebius_bin,
        prefix=prefix,
        only_projects=only_projects,
        profile=profile,
        on_status=on_status,
    )
    failed = sum(
        1 for entry in destroyed if entry.get("status") == "destroy-incomplete"
    )
    failed += sum(
        1 for entry in network_results if entry.get("status") == "destroy-incomplete"
    )
    return {
        "name": spec.name,
        "clusters": destroyed,
        "networks": network_results,
        "backend_counts": _backend_counts(destroyed),
        "failed": failed,
    }


def _reclaim_created_network(
    nebius_bin: str,
    project_id: str,
    network_id: str,
    subnet_id: str,
    env: dict[str, str],
    on_status: Callable[[str], None] | None,
    label: str,
    profile: str = "",
) -> list[str]:
    """Delete a fleet-created subnet + network, attempting both and returning errors."""

    errors: list[str] = []
    if subnet_id:
        _log(on_status, f"[{label}] deleting fleet-created subnet {subnet_id}")
        try:
            result = _run_capture(
                [
                    *_nebius_argv(nebius_bin, profile),
                    "vpc",
                    "subnet",
                    "delete",
                    "--id",
                    subnet_id,
                ],
                env=env,
                check=False,
                timeout=600,
            )
            if result.returncode != 0 and not _is_not_found_result(result):
                errors.append(
                    f"subnet delete failed (nebius exited {result.returncode})"
                )
        except Exception as exc:  # noqa: BLE001 - report and continue with network cleanup
            errors.append(f"subnet delete failed: {type(exc).__name__}: {exc}")
    _log(on_status, f"[{label}] deleting fleet-created network {network_id}")
    try:
        result = _run_capture(
            [
                *_nebius_argv(nebius_bin, profile),
                "vpc",
                "network",
                "delete",
                "--id",
                network_id,
            ],
            env=env,
            check=False,
            timeout=600,
        )
        if result.returncode != 0 and not _is_not_found_result(result):
            errors.append(f"network delete failed (nebius exited {result.returncode})")
    except Exception as exc:  # noqa: BLE001 - return context to the caller
        errors.append(f"network delete failed: {type(exc).__name__}: {exc}")
    return errors


def fleet_status(
    spec: FleetSpec,
    *,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Report the last-known deployment state for the fleet."""

    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    state_path = fleet_root / _FLEET_STATE
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            entries = {
                (
                    str(item.get("project_key", "")),
                    str(item.get("cluster_name", "")),
                ): item
                for item in state.get("clusters", [])
                if isinstance(item, dict)
            }
            for project, cluster in spec.cluster_targets():
                saved = entries.get((project.key(), cluster.name))
                if saved is not None:
                    require_backend_ownership(saved, cluster.backend_name())
                if cluster.backend_name() == "soperator":
                    assert cluster.soperator is not None
                    backend_root = (
                        fleet_root / project.key() / cluster.name / "soperator"
                    )
                    persisted_root = str((saved or {}).get("backend_state_root") or "")
                    if (
                        persisted_root
                        and Path(persisted_root).expanduser().resolve()
                        != backend_root.resolve()
                    ):
                        raise ValueError(
                            "persisted soperator backend_state_root does not match "
                            f"the canonical fleet-owned root for {project.key()}/"
                            f"{cluster.name}"
                        )
                    try:
                        live = get_backend("soperator").status(
                            cluster.soperator,
                            SoperatorStatusRequest(work_root=backend_root),
                        )
                    except Exception as exc:  # noqa: BLE001 - aggregate target health
                        live = {
                            "backend": "soperator",
                            "status": "status-error",
                            "error": str(exc),
                        }
                    merged = {
                        **(saved or {}),
                        **live,
                        "backend": "soperator",
                        "project_key": project.key(),
                        "cluster_name": cluster.name,
                        "backend_state_root": str(backend_root),
                    }
                    entries[(project.key(), cluster.name)] = merged
                elif saved is not None:
                    try:
                        live = get_backend("mk8s").status(
                            cluster,
                            MK8sStatusRequest(
                                state=saved,
                                install_dir=fleet_root / project.key() / cluster.name,
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - aggregate backend status
                        live = {
                            "backend": "mk8s",
                            "status": "status-error",
                            "error": str(exc),
                        }
                    entries[(project.key(), cluster.name)] = {
                        **saved,
                        **live,
                        "backend": "mk8s",
                        "project_key": project.key(),
                        "cluster_name": cluster.name,
                    }
            if isinstance(state.get("clusters"), list):
                persisted_keys = {
                    (
                        str(item.get("project_key", "")),
                        str(item.get("cluster_name", "")),
                    )
                    for item in state["clusters"]
                    if isinstance(item, dict)
                }
                state["clusters"] = [
                    entries.get(
                        (
                            str(item.get("project_key", "")),
                            str(item.get("cluster_name", "")),
                        ),
                        item,
                    )
                    if isinstance(item, dict)
                    else item
                    for item in state["clusters"]
                ]
                state["clusters"].extend(
                    item for key, item in entries.items() if key not in persisted_keys
                )
                state.update(_recount(state["clusters"]))
            state["backend_counts"] = _backend_counts(state.get("clusters", []))
            return state
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"persisted fleet inventory at {state_path} is unreadable; "
                "refusing status without backend ownership proof"
            ) from exc
    # Reconstruct from per-cluster sidecars if the summary is missing.
    clusters: list[dict[str, Any]] = []
    for project in spec.projects:
        for cluster in project.clusters:
            if cluster.backend_name() == "soperator":
                assert cluster.soperator is not None
                backend_root = fleet_root / project.key() / cluster.name / "soperator"
                try:
                    status = get_backend("soperator").status(
                        cluster.soperator,
                        SoperatorStatusRequest(work_root=backend_root),
                    )
                except Exception as exc:  # noqa: BLE001 - aggregate target health
                    status = {
                        "backend": "soperator",
                        "status": "status-error",
                        "error": str(exc),
                    }
                clusters.append(
                    {
                        **status,
                        "backend": "soperator",
                        "project_key": project.key(),
                        "cluster_name": cluster.name,
                        "backend_state_root": str(backend_root),
                    }
                )
                continue
            install_dir = fleet_root / project.key() / cluster.name
            saved = _load_env_sidecar(install_dir)
            # Trust the sidecar's own status ("provisioning"/"deployed"); a
            # present sidecar does not imply a successful apply (it is written
            # before terraform runs), so never assume "deployed" from presence.
            status = str(saved.get("status") or "unknown") if saved else "unknown"
            clusters.append(
                {
                    "backend": "mk8s",
                    "project_key": project.key(),
                    "cluster_name": cluster.name,
                    "status": status,
                    **({k: v for k, v in saved.items()} if saved else {}),
                }
            )
    return {
        "name": spec.name,
        "clusters": clusters,
        **_recount(clusters),
        "backend_counts": _backend_counts(clusters),
    }

"""Deploy / destroy / plan / status for npa-managed **fleets** of clusters.

A fleet is many k8s-training clusters across many projects in one tenant. This
module orchestrates, per ``(project, cluster)`` target:

1. Resolving (or creating) the project under the tenant via the ``nebius`` CLI.
2. Materializing a self-contained Terraform wrapper that sources the
   ``k8s-training`` recipe (repo-vendored by default, or a freshly cloned
   upstream ``main``/ref so the fleet can consume the latest recipe changes).
3. Rendering ``terraform.tfvars`` and running ``terraform apply`` with the
   tenant/project/region/iam_token/ssh passed as ``TF_VAR_*`` env.
4. Writing an admin kubeconfig context and an env sidecar so ``destroy`` can
   reconstruct the required variables.

Reuses the terraform subprocess helpers from
``npa.cli.cluster.terraform_lifecycle`` and Nebius-CLI env hygiene mirroring
``npa.soperator.lifecycle``.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from npa.cli.cluster.terraform_lifecycle import (
    _require_bin,
    _run_capture,
    _run_stream,
    _terraform_env,
)
from npa.fleet.quotas import preflight_region, shortfall_message
from npa.fleet.spec import ClusterSpec, FleetSpec, ProjectSpec
from npa.fleet.tfvars import patch_provider_domain, provider_domain, render_tfvars

logger = logging.getLogger(__name__)

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_K8S_TRAINING_SUBDIR = "k8s-training"
_MODULES_SUBDIR = "modules"
# Pinned ref cloned when no local recipe is available, matching the repo-vendored
# copy (deploy/cluster/vendor + the single-cluster wrapper) so a fleet run from an
# installed package doesn't silently drift onto upstream ``main`` HEAD.
_PINNED_LIBRARY_REF = "main-v2026-05-25+local-cluster-patches"
_ENV_SIDECAR = ".npa-fleet-env.json"
_FLEET_STATE = "fleet-state.json"


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _project_in_scope(project: ProjectSpec, only: list[str] | None, prefix: str) -> bool:
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
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {"1", "true", "yes", "on"}
    if not reuse:
        env.pop("NEBIUS_IAM_TOKEN", None)
    return env


def _nebius_argv(nebius_bin: str, profile: str = "") -> list[str]:
    """Base argv for a ``nebius`` CLI call, pinned to *profile* when given.

    A Nebius service account belongs to exactly one tenant, so a fleet targeting
    another tenant must authenticate as that tenant's profile. Passing
    ``--profile`` per call keeps the machine's active profile untouched (and
    keeps concurrent runs against different tenants independent).
    """

    return [nebius_bin, "--profile", profile] if profile else [nebius_bin]


def _nebius_config() -> dict[str, Any]:
    path = Path.home() / ".nebius" / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _resolve_tenant_id(nebius_bin: str, explicit: str, profile: str = "") -> str:
    if explicit:
        return explicit
    cfg = _nebius_config()
    # With an explicit profile, that profile's tenant is authoritative: falling
    # back to the active profile's tenant would deploy into the wrong tenant.
    selected = profile or str(cfg.get("default", "") or "")
    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles"), dict) else {}
    prof = profiles.get(selected, {}) if isinstance(profiles.get(selected), dict) else {}
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
    prof = profiles.get(selected, {}) if isinstance(profiles.get(selected), dict) else {}
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
            _log(on_status, f"cloning nebius-solutions-library@{ref} for latest k8s-training")
            _run_stream(
                [git, "clone", "--depth", "1", "--branch", ref, _SOLUTIONS_LIBRARY_REPO, str(clone_dir)],
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
            "iam", "project", "list", "--parent-id", tenant_id, "--format", "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return list(json.loads(result.stdout).get("items", []))
    except json.JSONDecodeError:
        return []


def _find_project_id(projects: list[dict[str, Any]], name: str) -> str:
    for item in projects:
        meta = item.get("metadata", {})
        if meta.get("name") == name and meta.get("id"):
            return str(meta["id"])
    return ""


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
        *_nebius_argv(nebius_bin, profile), "iam", "project", "create",
        "--parent-id", tenant_id, "--name", name,
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
        raise ValueError(f"could not parse project create output for {name!r}: {exc}") from exc
    project_id = str(payload.get("metadata", {}).get("id") or payload.get("id") or "")
    if not project_id:
        raise ValueError(f"project create for {name!r} returned no id: {result.stdout[:200]}")
    return project_id


def _list_subnets(
    nebius_bin: str, project_id: str, env: dict[str, str], profile: str = ""
) -> list[dict[str, Any]]:
    result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return list(json.loads(result.stdout).get("items", []))
    except json.JSONDecodeError:
        return []


def ensure_subnet(
    nebius_bin: str,
    project_id: str,
    *,
    name_stem: str,
    env: dict[str, str],
    profile: str = "",
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

    subnets = _list_subnets(nebius_bin, project_id, env, profile)
    if subnets:
        # Deterministic pick: prefer a subnet whose name looks like the project
        # default, else the lowest id, so repeated runs choose the same subnet
        # instead of relying on list order.
        def _rank(sub: dict[str, Any]) -> tuple[int, str]:
            meta = sub.get("metadata", {})
            name = str(meta.get("name") or "")
            return (0 if "default" in name else 1, str(meta.get("id") or ""))

        for sub in sorted(subnets, key=_rank):
            sid = str(sub.get("metadata", {}).get("id") or "")
            if sid:
                return sid, ""
    _log(on_status, f"no subnet in project {project_id[:12]}...; creating network + subnet")
    net = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile), "vpc", "network", "create",
            "--parent-id", project_id, "--name", f"{name_stem}-net", "--format", "json",
        ],
        env=env,
    )
    try:
        network_id = str(json.loads(net.stdout or "{}").get("metadata", {}).get("id") or "")
    except json.JSONDecodeError:
        network_id = ""
    if not network_id:
        raise ValueError(f"could not create network in project {project_id}: {net.stdout[:200]}")
    sub = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile), "vpc", "subnet", "create",
            "--parent-id", project_id, "--network-id", network_id,
            "--name", f"{name_stem}-subnet", "--format", "json",
        ],
        env=env,
    )
    try:
        subnet_id = str(json.loads(sub.stdout or "{}").get("metadata", {}).get("id") or "")
    except json.JSONDecodeError:
        subnet_id = ""
    if not subnet_id:
        raise ValueError(f"could not create subnet in project {project_id}: {sub.stdout[:200]}")
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
        _log(on_status, f"project {name!r} exists ({found})")
        return found, False
    if not create:
        raise ValueError(
            f"project {name!r} not found under tenant and project creation is disabled"
        )
    _log(on_status, f"creating project {name!r} under tenant (region {region or 'default'})")
    project_id = _create_project(
        nebius_bin, tenant_id, name, env, region=region, profile=profile
    )
    _log(on_status, f"created project {name!r} ({project_id})")
    return project_id, True


# --------------------------------------------------------------------------- #
# Per-cluster terraform materialization
# --------------------------------------------------------------------------- #
def _write_env_sidecar(install_dir: Path, data: dict[str, str]) -> None:
    (install_dir / _ENV_SIDECAR).write_text(json.dumps(data, indent=2))


def _load_env_sidecar(install_dir: Path) -> dict[str, str] | None:
    path = install_dir / _ENV_SIDECAR
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _prepare_install_dir(
    install_dir: Path,
    *,
    recipe_root: Path,
    region: str,
    cluster: ClusterSpec,
    ssh_public_key: str,
    on_status: Callable[[str], None] | None = None,
) -> Path:
    """Materialize a per-cluster copy of the recipe and return the terraform workdir.

    Copies ``<recipe_root>/k8s-training`` and ``<recipe_root>/modules`` into the
    install dir (preserving the ``../modules`` relationship), patches the recipe
    provider domain for the region, and writes ``terraform.tfvars``. Returns the
    ``k8s-training`` copy where terraform must run.
    """

    install_dir.mkdir(parents=True, exist_ok=True)
    workdir = install_dir / _K8S_TRAINING_SUBDIR
    modules_dst = install_dir / _MODULES_SUBDIR
    # Refresh recipe files but preserve any existing terraform state/plugins.
    if workdir.exists():
        for item in workdir.iterdir():
            if item.name.startswith("terraform.tfstate") or item.name == ".terraform":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
    shutil.copytree(recipe_root / _K8S_TRAINING_SUBDIR, workdir, dirs_exist_ok=True)
    if modules_dst.exists():
        shutil.rmtree(modules_dst, ignore_errors=True)
    shutil.copytree(recipe_root / _MODULES_SUBDIR, modules_dst)

    provider_tf = workdir / "provider.tf"
    if provider_tf.exists():
        original = provider_tf.read_text()
        patched = patch_provider_domain(original, region)
        # Loud no-op guard: if the recipe drifts (renamed file, moved/renamed
        # provider block, or changed default domain) the literal replace silently
        # matches nothing and terraform would talk to the EU endpoint from a
        # non-EU region, failing confusingly at apply. Surface it here instead.
        target = provider_domain(region)
        if patched == original and target not in original:
            _log(
                on_status,
                f"WARNING: provider.tf domain not patched to {target} for region "
                f"{region!r} (recipe may have changed); check {provider_tf}",
            )
        provider_tf.write_text(patched)
    elif not region.startswith("eu"):
        _log(
            on_status,
            f"WARNING: no provider.tf in recipe copy at {workdir}; cannot patch "
            f"provider domain for region {region!r}",
        )

    (workdir / "terraform.tfvars").write_text(
        render_tfvars(cluster, ssh_public_key=ssh_public_key)
    )
    return workdir


def _run_to_log(
    args: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> None:
    """Run *args* streaming combined stdout/stderr to *log_path*; raise on failure.

    Used in parallel deploys so each cluster's terraform output goes to its own
    file instead of interleaving on the shared stdout.
    """

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        proc = subprocess.run(
            args, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)} (see {log_path})")


def _tf_run(
    args: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path | None
) -> None:
    """terraform runner: stream to stdout (sequential) or to a per-cluster log."""

    if log_path is not None:
        _run_to_log(args, cwd=cwd, env=env, timeout=timeout, log_path=log_path)
    else:
        _run_stream(args, cwd=cwd, env=env, timeout=timeout)


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
) -> None:
    """Populate a shared terraform plugin cache with a single ``init`` before fan-out.

    Concurrent ``terraform init`` writes to a shared ``TF_PLUGIN_CACHE_DIR`` can
    corrupt provider binaries (a real failure hit in testing). Pre-warming the
    cache as the sole writer means the parallel per-cluster inits only *read* it.
    """

    cache_dir = Path(os.environ.get("TF_PLUGIN_CACHE_DIR") or (work_root / ".tf-plugin-cache"))
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
    _run_stream(
        [terraform_bin, "init", "-input=false"],
        cwd=workdir,
        env=_terraform_env(nebius_bin, profile=profile),
        timeout=900,
    )


def _cluster_tf_env(
    nebius_bin: str,
    *,
    tenant_id: str,
    project_id: str,
    region: str,
    subnet_id: str,
    profile: str = "",
) -> dict[str, str]:
    env = _terraform_env(nebius_bin, profile=profile)
    env["TF_VAR_tenant_id"] = tenant_id
    env["TF_VAR_parent_id"] = project_id
    env["TF_VAR_region"] = region
    env["TF_VAR_subnet_id"] = subnet_id
    return env


def _terraform_outputs(terraform_bin: str, install_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    result = _run_capture(
        [terraform_bin, "output", "-json"], cwd=install_dir, env=env, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def _cluster_id_from_outputs(outputs: dict[str, Any]) -> str:
    value = outputs.get("kube_cluster", {}).get("value")
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    return ""


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
            "mk8s", "cluster", "list", "--parent-id", project_id, "--format", "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        items = json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return ""
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("name") == cluster_name and meta.get("id"):
            return str(meta["id"])
    return ""


def _write_kubeconfig(
    nebius_bin: str,
    cluster_id: str,
    kubeconfig_path: Path,
    context: str,
    env: dict[str, str],
    profile: str = "",
) -> None:
    """Write an admin kubeconfig for *cluster_id*.

    When a profile is given the nebius CLI bakes ``--profile`` into the
    kubeconfig's exec-credential args, so ``kubectl`` keeps authenticating as
    that tenant's principal rather than the machine's active profile.
    """

    kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
    _run_capture(
        [
            *_nebius_argv(nebius_bin, profile), "mk8s", "cluster", "get-credentials",
            "--id", cluster_id, "--external", "--force",
            "--kubeconfig", str(kubeconfig_path), "--context-name", context,
        ],
        env=env,
        check=False,
        timeout=180,
    )


def _context_name(fleet_name: str, project_key: str, cluster_name: str) -> str:
    return f"fleet-{fleet_name}-{project_key}-{cluster_name}"


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
            clusters.append(
                {
                    "name": cluster.name,
                    "cpu_nodes": cluster.cpu_count(),
                    "cpu_preset": cluster.cpu_nodes.preset if cluster.cpu_nodes else "",
                    "gpu_nodes": cluster.gpu_count(),
                    "gpu_platform": cluster.gpu_nodes.platform if cluster.gpu_nodes else "",
                    "gpu_preset": cluster.gpu_nodes.preset if cluster.gpu_nodes else "",
                    "gpu_reservation": (
                        "strict"
                        if cluster.gpu_nodes and cluster.gpu_nodes.capacity_block_group
                        else "on-demand"
                    ),
                    "enable_gpu_cluster": cluster.resolved_enable_gpu_cluster(),
                    "enable_filestore": cluster.enable_filestore,
                    "filestore_disk_size_gibibytes": cluster.filestore_disk_size_gibibytes,
                    "filestore_mount_path": cluster.filestore_mount_path,
                    "filestore_mount_tag": cluster.filestore_mount_tag,
                    "k8s_version": cluster.k8s_version or "backend-default",
                }
            )
        plan_projects.append(
            {
                "project_id": project.project_id or None,
                "display_name": project.display_name(prefix) or None,
                "will_create": not project.project_id,
                "clusters": clusters,
            }
        )
    return {
        "name": spec.name,
        "tenant_id": tenant or "(resolve-at-deploy)",
        "region": reg or "(resolve-at-deploy)",
        "project_prefix": prefix,
        "profile": plan_profile or "(active)",
        "project_count": len(spec.projects),
        "cluster_count": len(spec.cluster_targets()),
        "projects": plan_projects,
    }


def deploy_fleet(
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
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy every ``(project, cluster)`` target in *spec*. Returns fleet metadata.

    ``only_projects`` / ``only_clusters`` narrow the action to a subset so callers
    can add one or many specific projects/clusters without touching the rest of a
    fleet (deploy is idempotent: existing clusters reconcile in place).

    ``concurrency`` > 1 applies that many clusters in parallel (each has its own
    isolated terraform state, so there is no cross-cluster lock contention). The
    provider plugin cache is pre-warmed once to avoid concurrent-init corruption,
    and each cluster streams to its own ``<install_dir>/deploy.log``. Parallel
    deploys assume one cluster per project subnet (concurrent creates in a shared
    network can race on the same CIDR pool) -- give each cluster its own
    ``subnet_id`` (or its own project) when packing several into one project.

    ``profile`` overrides the spec's ``profile``: every ``nebius`` CLI call and
    the minted terraform IAM token then use that ``~/.nebius`` profile, which is
    how one workstation deploys into several tenants (a service account is
    single-tenant) without switching the machine-wide active profile.

    ``preflight`` (default on) validates explicitly bound capacity blocks and
    compares the remaining on-demand requirements against tenant quota
    allowances before any apply. mk8s accepts a node group it cannot fill, so
    without this a capacity/quota wall shows up as terraform blocking on
    ``Still creating...`` until the timeout.
    """

    spec.validate()
    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    nebius_profile = spec.profile if profile is None else profile
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    tenant_id = _resolve_tenant_id(nebius_bin, spec.tenant_id, nebius_profile)
    fleet_region = _resolve_region(spec.region)
    ssh_public_key = _resolve_ssh_public_key(spec.ssh_public_key)

    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    fleet_root.mkdir(parents=True, exist_ok=True)
    recipe_root = _resolve_recipe_root(
        k8s_training_dir, ref=k8s_training_ref, work_root=work_root, on_status=on_status
    )
    _log(on_status, f"k8s-training recipe root: {recipe_root}")

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

    # Phase 0: quota preflight, before any project is created. Regions come from
    # the spec, so this needs no project ids -- and running it first means a
    # quota-blocked deploy does not leave a freshly created, empty project behind.
    if preflight:
        scoped: dict[str, list[ClusterSpec]] = {}
        for project in spec.projects:
            if not _project_in_scope(project, only_projects, prefix):
                continue
            region = project.region or fleet_region
            for cluster in project.clusters:
                if only_clusters and cluster.name not in only_clusters:
                    continue
                scoped.setdefault(region, []).append(cluster)
        if scoped:
            _preflight_quotas(
                nebius_bin,
                tenant_id=tenant_id,
                by_region=scoped,
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
        region = project.region or fleet_region
        try:
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
        except Exception as exc:  # noqa: BLE001 - report and continue
            _log(on_status, f"project {project.key()} FAILED to resolve: {exc}")
            if not continue_on_error:
                raise
            results.append({"project_key": project.key(), "status": "error", "error": str(exc)})
            continue
        for cluster in project.clusters:
            if only_clusters and cluster.name not in only_clusters:
                continue
            targets.append(
                {
                    "project": project,
                    "cluster": cluster,
                    "project_id": project_id,
                    "created": created,
                    "region": region,
                }
            )

    parallel = concurrency > 1 and len(targets) > 1
    if parallel:
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
        )

    def _run_target(t: dict[str, Any]) -> dict[str, Any]:
        log_path = None
        if parallel:
            log_path = fleet_root / t["project"].key() / t["cluster"].name / "deploy.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return _deploy_one_cluster(
            spec=spec,
            project=t["project"],
            cluster=t["cluster"],
            project_id=t["project_id"],
            project_created=t["created"],
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
                results.append(fut.result())  # _deploy_one_cluster captures its own errors
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
    env: dict[str, str],
    profile: str,
    on_status: Callable[[str], None] | None,
) -> None:
    """Raise when reservations or tenant quota cannot cover the scoped clusters."""

    shortfalls = []
    for region, clusters in sorted(by_region.items()):
        _log(on_status, f"capacity/quota preflight: {len(clusters)} cluster(s) in {region}")
        shortfalls += preflight_region(
            nebius_bin=nebius_bin,
            tenant_id=tenant_id,
            region=region,
            clusters=clusters,
            env=env,
            profile=profile,
            run_capture=_run_capture,
            nebius_argv=_nebius_argv,
            on_status=on_status,
        )
    if shortfalls:
        raise ValueError(shortfall_message(shortfalls, tenant_id))


def _deploy_one_cluster(
    *,
    spec: FleetSpec,
    project: ProjectSpec,
    cluster: ClusterSpec,
    project_id: str,
    project_created: bool,
    region: str,
    tenant_id: str,
    ssh_public_key: str,
    fleet_root: Path,
    recipe_root: Path,
    terraform_bin: str,
    nebius_bin: str,
    profile: str = "",
    timeout_minutes: int,
    on_status: Callable[[str], None] | None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    project_key = project.key()
    install_dir = fleet_root / project_key / cluster.name
    context = _context_name(spec.name, project_key, cluster.name)
    label = f"{project_key}/{cluster.name}"
    try:
        created_network_id = ""
        if cluster.subnet_id:
            subnet_id = cluster.subnet_id
        else:
            subnet_id, created_network_id = ensure_subnet(
                nebius_bin,
                project_id,
                name_stem=cluster.name,
                env=_nebius_cli_env(),
                profile=profile,
                on_status=on_status,
            )
        workdir = _prepare_install_dir(
            install_dir,
            recipe_root=recipe_root,
            region=region,
            cluster=cluster,
            ssh_public_key=ssh_public_key,
            on_status=on_status,
        )
        env = _cluster_tf_env(
            nebius_bin,
            tenant_id=tenant_id,
            project_id=project_id,
            region=region,
            subnet_id=subnet_id,
            profile=profile,
        )
        # Written before apply so ``destroy`` can reconstruct TF_VAR_* and reclaim
        # the network we created even if apply fails midway. ``status`` starts as
        # "provisioning" and is promoted to "deployed" only on success, so the
        # status fallback never mislabels a failed cluster as deployed.
        sidecar = {
            "tenant_id": tenant_id,
            "project_id": project_id,
            "region": region,
            "subnet_id": subnet_id,
            "created_network_id": created_network_id,
            "cluster_name": cluster.name,
            "context": context,
            "profile": profile,
            "status": "provisioning",
        }
        _write_env_sidecar(install_dir, sidecar)
        _log(on_status, f"[{label}] terraform init" + (f" (-> {log_path})" if log_path else ""))
        _tf_run([terraform_bin, "init", "-input=false"], cwd=workdir, env=env, timeout=900, log_path=log_path)
        _log(
            on_status,
            f"[{label}] terraform apply (cpu={cluster.cpu_count()} gpu={cluster.gpu_count()} "
            f"{cluster.gpu_nodes.preset if cluster.gpu_nodes else ''})",
        )
        _tf_run(
            [terraform_bin, "apply", "-auto-approve", "-input=false"],
            cwd=workdir,
            env=env,
            timeout=timeout_minutes * 60,
            log_path=log_path,
        )
        outputs = _terraform_outputs(terraform_bin, workdir, env)
        cluster_id = _cluster_id_from_outputs(outputs)
        kubeconfig_path = install_dir / "kubeconfig"
        if cluster_id:
            _log(on_status, f"[{label}] writing kubeconfig context {context}")
            _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context, env, profile)
        _write_env_sidecar(install_dir, {**sidecar, "cluster_id": cluster_id, "status": "deployed"})
        return {
            "project_key": project_key,
            "project_id": project_id,
            "project_created": project_created,
            "cluster_name": cluster.name,
            "region": region,
            "cluster_id": cluster_id,
            "kube_context": context,
            "kubeconfig": str(kubeconfig_path) if cluster_id else "",
            "install_dir": str(install_dir),
            "status": "deployed",
        }
    except Exception as exc:  # noqa: BLE001 - capture per-cluster failure
        _log(on_status, f"[{label}] FAILED: {exc}")
        return {
            "project_key": project_key,
            "project_id": project_id,
            "cluster_name": cluster.name,
            "region": region,
            "install_dir": str(install_dir),
            "status": "error",
            "error": str(exc),
        }


def _write_fleet_state(fleet_root: Path, result: dict[str, Any]) -> None:
    try:
        (fleet_root / _FLEET_STATE).write_text(json.dumps(result, indent=2))
    except OSError:
        pass


def _load_fleet_state(fleet_root: Path) -> dict[str, Any]:
    path = fleet_root / _FLEET_STATE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _recount(clusters: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "deployed": sum(1 for c in clusters if c.get("status") == "deployed"),
        "failed": sum(1 for c in clusters if c.get("status") == "error"),
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
    clusters = state.get("clusters", []) if isinstance(state.get("clusters"), list) else []
    index = {(c.get("project_key"), c.get("cluster_name")): i for i, c in enumerate(clusters)}
    for entry in results:
        key = (entry.get("project_key"), entry.get("cluster_name"))
        if key[1] is None:  # project-level failure (no cluster) -- don't persist
            continue
        if key in index:
            clusters[index[key]] = entry
        else:
            index[key] = len(clusters)
            clusters.append(entry)
    _write_fleet_state(fleet_root, {**base_meta, "clusters": clusters, **_recount(clusters)})


def _prune_fleet_state(fleet_root: Path, removed_keys: set[tuple[str, str]]) -> None:
    """Drop destroyed ``(project_key, cluster_name)`` entries from the summary."""

    state = _load_fleet_state(fleet_root)
    clusters = state.get("clusters", []) if isinstance(state.get("clusters"), list) else []
    kept = [c for c in clusters if (c.get("project_key"), c.get("cluster_name")) not in removed_keys]
    _write_fleet_state(fleet_root, {**state, "clusters": kept, **_recount(kept)})


def destroy_fleet(
    spec: FleetSpec,
    *,
    work_root: Path | None = None,
    project_prefix: str | None = None,
    only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None,
    timeout_minutes: int = 120,
    concurrency: int = 1,
    profile: str | None = None,
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
    """

    spec.validate()
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name

    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    targets: list[tuple[ProjectSpec, ClusterSpec]] = [
        (project, cluster)
        for project in spec.projects
        if _project_in_scope(project, only_projects, prefix)
        for cluster in project.clusters
        if not (only_clusters and cluster.name not in only_clusters)
    ]

    parallel = concurrency > 1 and len(targets) > 1

    def _run(target: tuple[ProjectSpec, ClusterSpec]) -> dict[str, Any]:
        project, cluster = target
        log_path = None
        if parallel:
            log_path = fleet_root / project.key() / cluster.name / "destroy.log"
            if log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
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

    if parallel:
        _log(on_status, f"destroying {len(targets)} cluster(s) with concurrency={concurrency}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            destroyed = [f.result() for f in concurrent.futures.as_completed(
                [pool.submit(_run, t) for t in targets]
            )]
    else:
        destroyed = [_run(t) for t in targets]

    removed_keys = {
        (d["project_key"], d["cluster_name"]) for d in destroyed if d.get("status") != "absent"
    }
    if removed_keys:
        _prune_fleet_state(fleet_root, removed_keys)
    return {"name": spec.name, "clusters": destroyed}


def _destroy_one_cluster(
    *,
    spec: FleetSpec,
    project: ProjectSpec,
    cluster: ClusterSpec,
    fleet_root: Path,
    terraform_bin: str,
    nebius_bin: str,
    profile: str = "",
    timeout_minutes: int,
    on_status: Callable[[str], None] | None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    install_dir = fleet_root / project.key() / cluster.name
    label = f"{project.key()}/{cluster.name}"
    if not install_dir.exists():
        _log(on_status, f"[{label}] no install dir; skipping")
        return {"project_key": project.key(), "cluster_name": cluster.name, "status": "absent"}
    saved = _load_env_sidecar(install_dir) or {}
    project_id = str(saved.get("project_id") or "")
    subnet_id = str(saved.get("subnet_id") or "")
    # Fall back to the profile the cluster was deployed with so a teardown never
    # authenticates as the wrong tenant's principal.
    profile = profile or str(saved.get("profile") or "")
    workdir = install_dir / _K8S_TRAINING_SUBDIR
    env = _cluster_tf_env(
        nebius_bin,
        tenant_id=str(saved.get("tenant_id") or spec.tenant_id),
        project_id=project_id,
        region=str(saved.get("region") or spec.region),
        subnet_id=subnet_id,
        profile=profile,
    )
    _log(on_status, f"[{label}] terraform destroy" + (f" (-> {log_path})" if log_path else ""))
    _tf_run([terraform_bin, "init", "-input=false"], cwd=workdir, env=env, timeout=900, log_path=log_path)
    try:
        _tf_run(
            [terraform_bin, "destroy", "-auto-approve", "-input=false"],
            cwd=workdir,
            env=env,
            timeout=timeout_minutes * 60,
            log_path=log_path,
        )
        destroy_failed = False
    except Exception as exc:  # noqa: BLE001 - fall back to direct cleanup
        logger.debug("[%s] terraform destroy failed: %s", label, exc)
        destroy_failed = True
    status = "destroyed"
    if destroy_failed:
        _log(on_status, f"[{label}] terraform destroy reported errors; direct cleanup")
        if project_id:
            cid = _find_cluster_id_by_name(
                nebius_bin,
                project_id,
                str(saved.get("cluster_name") or cluster.name),
                env,
                profile,
            )
            if cid:
                _run_capture(
                    [*_nebius_argv(nebius_bin, profile), "mk8s", "cluster", "delete", "--id", cid],
                    env=env,
                    check=False,
                    timeout=timeout_minutes * 60,
                )
        status = "destroyed-with-fallback"
    # Reclaim the VPC network + subnet this fleet created (only when we created
    # them; a reused pre-existing subnet has no created_network_id and is left
    # untouched). Runs after the cluster is gone so the subnet is unused.
    created_network_id = str(saved.get("created_network_id") or "")
    if created_network_id and project_id:
        _reclaim_created_network(
            nebius_bin,
            project_id,
            created_network_id,
            subnet_id,
            env,
            on_status,
            label,
            profile=profile,
        )
    # Remove the whole per-cluster install dir so the cluster no longer appears
    # in `status`.
    shutil.rmtree(install_dir, ignore_errors=True)
    return {"project_key": project.key(), "cluster_name": cluster.name, "status": status}


def _reclaim_created_network(
    nebius_bin: str,
    project_id: str,
    network_id: str,
    subnet_id: str,
    env: dict[str, str],
    on_status: Callable[[str], None] | None,
    label: str,
    profile: str = "",
) -> None:
    """Best-effort delete of a fleet-created subnet + network (subnet first)."""

    if subnet_id:
        _log(on_status, f"[{label}] deleting fleet-created subnet {subnet_id}")
        _run_capture(
            [*_nebius_argv(nebius_bin, profile), "vpc", "subnet", "delete", "--id", subnet_id],
            env=env,
            check=False,
            timeout=600,
        )
    _log(on_status, f"[{label}] deleting fleet-created network {network_id}")
    _run_capture(
        [*_nebius_argv(nebius_bin, profile), "vpc", "network", "delete", "--id", network_id],
        env=env,
        check=False,
        timeout=600,
    )


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
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    # Reconstruct from per-cluster sidecars if the summary is missing.
    clusters: list[dict[str, Any]] = []
    for project in spec.projects:
        for cluster in project.clusters:
            install_dir = fleet_root / project.key() / cluster.name
            saved = _load_env_sidecar(install_dir)
            # Trust the sidecar's own status ("provisioning"/"deployed"); a
            # present sidecar does not imply a successful apply (it is written
            # before terraform runs), so never assume "deployed" from presence.
            status = str(saved.get("status") or "unknown") if saved else "unknown"
            clusters.append(
                {
                    "project_key": project.key(),
                    "cluster_name": cluster.name,
                    "status": status,
                    **({k: v for k, v in saved.items()} if saved else {}),
                }
            )
    return {"name": spec.name, "clusters": clusters}

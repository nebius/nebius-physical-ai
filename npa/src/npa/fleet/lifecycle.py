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

import json
import os
from pathlib import Path
from typing import Any, Callable

from npa.cli.cluster.terraform_lifecycle import (
    _require_bin,
    _run_capture,
    _run_stream,
    _terraform_env,
)
from npa.fleet.spec import ClusterSpec, FleetSpec, ProjectSpec
from npa.fleet.tfvars import render_main_tf, render_tfvars, variables_tf

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_K8S_TRAINING_SUBDIR = "k8s-training"
_ENV_SIDECAR = ".npa-fleet-env.json"
_FLEET_STATE = "fleet-state.json"


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


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


def _nebius_config() -> dict[str, Any]:
    path = Path.home() / ".nebius" / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _resolve_tenant_id(nebius_bin: str, explicit: str) -> str:
    if explicit:
        return explicit
    cfg = _nebius_config()
    default = str(cfg.get("default", "") or "")
    profiles = cfg.get("profiles", {}) if isinstance(cfg.get("profiles"), dict) else {}
    prof = profiles.get(default, {}) if isinstance(profiles.get(default), dict) else {}
    tenant = str(prof.get("tenant-id", "") or "")
    if tenant:
        return tenant
    # Fall back to the npa environment config.
    try:
        from npa.clients.config import resolve_environment

        envcfg = resolve_environment()
        if envcfg and envcfg.tenant_id:
            return envcfg.tenant_id
    except Exception:
        pass
    raise ValueError(
        "tenant_id could not be resolved from the spec, ~/.nebius/config.yaml, or ~/.npa"
    )


def _resolve_region(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        from npa.clients.config import resolve_environment

        envcfg = resolve_environment()
        if envcfg and envcfg.region:
            return envcfg.region
    except Exception:
        pass
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
# --------------------------------------------------------------------------- #
def _find_vendored_k8s_training() -> Path | None:
    """Walk up from this file to find the repo-vendored k8s-training recipe."""

    rel = Path("deploy") / "cluster" / "vendor" / "nebius-solutions-library" / _K8S_TRAINING_SUBDIR
    for base in Path(__file__).resolve().parents:
        candidate = base / rel
        if (candidate / "variables.tf").exists():
            return candidate
    return None


def _resolve_k8s_training_source(
    k8s_training_dir: Path | None,
    *,
    ref: str | None,
    work_root: Path,
    on_status: Callable[[str], None] | None,
) -> Path:
    """Resolve the k8s-training module dir.

    Priority: explicit dir > env override > cloned ref (latest) > repo-vendored.
    Cloning satisfies "consume the latest k8s-training changes"; the vendored
    copy is the tested default when no ref/dir is requested.
    """

    if k8s_training_dir is not None:
        path = k8s_training_dir.expanduser().resolve()
        if not (path / "variables.tf").exists():
            raise ValueError(f"{path} is not a k8s-training recipe dir (missing variables.tf)")
        return path

    env_dir = os.environ.get("NPA_K8S_TRAINING_DIR", "").strip()
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if not (path / "variables.tf").exists():
            raise ValueError(f"NPA_K8S_TRAINING_DIR={path} is missing variables.tf")
        return path

    if ref:
        clone_dir = work_root / "nebius-solutions-library"
        module = clone_dir / _K8S_TRAINING_SUBDIR
        if not (module / "variables.tf").exists():
            work_root.mkdir(parents=True, exist_ok=True)
            git = _require_bin("git")
            _log(on_status, f"cloning nebius-solutions-library@{ref} for latest k8s-training")
            _run_stream(
                [git, "clone", "--depth", "1", "--branch", ref, _SOLUTIONS_LIBRARY_REPO, str(clone_dir)],
                timeout=600,
            )
        return module

    vendored = _find_vendored_k8s_training()
    if vendored is not None:
        return vendored
    # No vendored copy available (e.g. installed package without repo). Clone main.
    return _resolve_k8s_training_source(
        None, ref="main", work_root=work_root, on_status=on_status
    )


# --------------------------------------------------------------------------- #
# Project resolution / creation
# --------------------------------------------------------------------------- #
def _list_projects(nebius_bin: str, tenant_id: str, env: dict[str, str]) -> list[dict[str, Any]]:
    result = _run_capture(
        [nebius_bin, "iam", "project", "list", "--parent-id", tenant_id, "--format", "json"],
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


def _create_project(nebius_bin: str, tenant_id: str, name: str, env: dict[str, str]) -> str:
    result = _run_capture(
        [
            nebius_bin, "iam", "project", "create",
            "--parent-id", tenant_id, "--name", name, "--format", "json",
        ],
        env=env,
    )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse project create output for {name!r}: {exc}") from exc
    project_id = str(payload.get("metadata", {}).get("id") or payload.get("id") or "")
    if not project_id:
        raise ValueError(f"project create for {name!r} returned no id: {result.stdout[:200]}")
    return project_id


def resolve_project_id(
    nebius_bin: str,
    tenant_id: str,
    project: ProjectSpec,
    *,
    prefix: str,
    create: bool,
    env: dict[str, str],
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, bool]:
    """Return ``(project_id, created)`` for a project spec, creating if allowed."""

    if project.project_id:
        return project.project_id, False
    name = project.display_name(prefix)
    if not name:
        raise ValueError("project needs a name or project_id to resolve")
    existing = _list_projects(nebius_bin, tenant_id, env)
    found = _find_project_id(existing, name)
    if found:
        _log(on_status, f"project {name!r} exists ({found})")
        return found, False
    if not create:
        raise ValueError(
            f"project {name!r} not found under tenant and project creation is disabled"
        )
    _log(on_status, f"creating project {name!r} under tenant")
    project_id = _create_project(nebius_bin, tenant_id, name, env)
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
    install_dir: Path, *, k8s_training_source: Path, region: str, cluster: ClusterSpec
) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    (install_dir / "main.tf").write_text(
        render_main_tf(k8s_training_source=str(k8s_training_source), region=region)
    )
    (install_dir / "variables.tf").write_text(variables_tf())
    (install_dir / "terraform.tfvars").write_text(render_tfvars(cluster))


def _cluster_tf_env(
    nebius_bin: str,
    *,
    tenant_id: str,
    project_id: str,
    region: str,
    ssh_public_key: str,
) -> dict[str, str]:
    env = _terraform_env(nebius_bin)
    env["TF_VAR_tenant_id"] = tenant_id
    env["TF_VAR_parent_id"] = project_id
    env["TF_VAR_region"] = region
    env["TF_VAR_ssh_public_key"] = ssh_public_key
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
    nebius_bin: str, project_id: str, cluster_name: str, env: dict[str, str]
) -> str:
    result = _run_capture(
        [nebius_bin, "mk8s", "cluster", "list", "--parent-id", project_id, "--format", "json"],
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
    nebius_bin: str, cluster_id: str, kubeconfig_path: Path, context: str, env: dict[str, str]
) -> None:
    kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
    _run_capture(
        [
            nebius_bin, "mk8s", "cluster", "get-credentials",
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
) -> dict[str, Any]:
    """Return the resolved deployment plan without touching infrastructure."""

    spec.validate()
    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    tenant = tenant_id or spec.tenant_id
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
                    "enable_gpu_cluster": cluster.resolved_enable_gpu_cluster(),
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
    timeout_minutes: int = 120,
    continue_on_error: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy every ``(project, cluster)`` target in *spec*. Returns fleet metadata."""

    spec.validate()
    prefix = project_prefix if project_prefix is not None else spec.project_prefix
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    tenant_id = _resolve_tenant_id(nebius_bin, spec.tenant_id)
    fleet_region = _resolve_region(spec.region)
    ssh_public_key = _resolve_ssh_public_key(spec.ssh_public_key)

    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name
    fleet_root.mkdir(parents=True, exist_ok=True)
    k8s_training_source = _resolve_k8s_training_source(
        k8s_training_dir, ref=k8s_training_ref, work_root=work_root, on_status=on_status
    )
    _log(on_status, f"k8s-training recipe: {k8s_training_source}")

    cli_env = _nebius_cli_env()
    results: list[dict[str, Any]] = []
    project_ids: dict[str, str] = {}

    for project in spec.projects:
        if only_projects and project.key() not in only_projects:
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
        project_ids[project.key()] = project_id

        for cluster in project.clusters:
            entry = _deploy_one_cluster(
                spec=spec,
                project=project,
                cluster=cluster,
                project_id=project_id,
                project_created=created,
                region=region,
                tenant_id=tenant_id,
                ssh_public_key=ssh_public_key,
                fleet_root=fleet_root,
                k8s_training_source=k8s_training_source,
                terraform_bin=terraform_bin,
                nebius_bin=nebius_bin,
                timeout_minutes=timeout_minutes,
                on_status=on_status,
            )
            results.append(entry)
            if entry.get("status") == "error" and not continue_on_error:
                raise RuntimeError(entry.get("error") or "cluster deploy failed")

    result = {
        "name": spec.name,
        "tenant_id": tenant_id,
        "region": fleet_region,
        "project_prefix": prefix,
        "k8s_training_source": str(k8s_training_source),
        "clusters": results,
        "deployed": sum(1 for r in results if r.get("status") == "deployed"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }
    _write_fleet_state(fleet_root, result)
    return result


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
    k8s_training_source: Path,
    terraform_bin: str,
    nebius_bin: str,
    timeout_minutes: int,
    on_status: Callable[[str], None] | None,
) -> dict[str, Any]:
    project_key = project.key()
    install_dir = fleet_root / project_key / cluster.name
    context = _context_name(spec.name, project_key, cluster.name)
    label = f"{project_key}/{cluster.name}"
    try:
        _prepare_install_dir(
            install_dir,
            k8s_training_source=k8s_training_source,
            region=region,
            cluster=cluster,
        )
        env = _cluster_tf_env(
            nebius_bin,
            tenant_id=tenant_id,
            project_id=project_id,
            region=region,
            ssh_public_key=ssh_public_key,
        )
        _write_env_sidecar(
            install_dir,
            {
                "tenant_id": tenant_id,
                "project_id": project_id,
                "region": region,
                "cluster_name": cluster.name,
                "context": context,
            },
        )
        _log(on_status, f"[{label}] terraform init")
        _run_stream([terraform_bin, "init", "-input=false"], cwd=install_dir, env=env, timeout=900)
        _log(
            on_status,
            f"[{label}] terraform apply (cpu={cluster.cpu_count()} gpu={cluster.gpu_count()} "
            f"{cluster.gpu_nodes.preset if cluster.gpu_nodes else ''})",
        )
        _run_stream(
            [terraform_bin, "apply", "-auto-approve", "-input=false"],
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )
        outputs = _terraform_outputs(terraform_bin, install_dir, env)
        cluster_id = _cluster_id_from_outputs(outputs)
        kubeconfig_path = install_dir / "kubeconfig"
        if cluster_id:
            _log(on_status, f"[{label}] writing kubeconfig context {context}")
            _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context, env)
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


def destroy_fleet(
    spec: FleetSpec,
    *,
    work_root: Path | None = None,
    project_prefix: str | None = None,
    only_projects: list[str] | None = None,
    timeout_minutes: int = 120,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Destroy every cluster in the fleet (best-effort, per-target)."""

    spec.validate()
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    work_root = (work_root or _default_work_root()).expanduser()
    fleet_root = work_root / spec.name

    destroyed: list[dict[str, Any]] = []
    for project in spec.projects:
        if only_projects and project.key() not in only_projects:
            continue
        for cluster in project.clusters:
            install_dir = fleet_root / project.key() / cluster.name
            label = f"{project.key()}/{cluster.name}"
            if not install_dir.exists():
                _log(on_status, f"[{label}] no install dir; skipping")
                destroyed.append({"project_key": project.key(), "cluster_name": cluster.name, "status": "absent"})
                continue
            saved = _load_env_sidecar(install_dir) or {}
            tenant_id = str(saved.get("tenant_id") or spec.tenant_id)
            project_id = str(saved.get("project_id") or "")
            region = str(saved.get("region") or spec.region)
            try:
                ssh_public_key = _resolve_ssh_public_key(spec.ssh_public_key)
            except ValueError:
                ssh_public_key = ""
            env = _cluster_tf_env(
                nebius_bin,
                tenant_id=tenant_id,
                project_id=project_id,
                region=region,
                ssh_public_key=ssh_public_key,
            )
            _log(on_status, f"[{label}] terraform destroy")
            _run_stream([terraform_bin, "init", "-input=false"], cwd=install_dir, env=env, timeout=900)
            destroy = _run_capture(
                [terraform_bin, "destroy", "-auto-approve", "-input=false"],
                cwd=install_dir,
                env=env,
                timeout=timeout_minutes * 60,
                check=False,
            )
            status = "destroyed"
            if destroy.returncode != 0:
                _log(on_status, f"[{label}] terraform destroy reported errors; direct cleanup")
                # Fall back to deleting the mk8s cluster directly by name.
                if project_id:
                    cid = _find_cluster_id_by_name(
                        nebius_bin, project_id, str(saved.get("cluster_name") or cluster.name), env
                    )
                    if cid:
                        _run_capture(
                            [nebius_bin, "mk8s", "cluster", "delete", "--id", cid],
                            env=env,
                            check=False,
                            timeout=timeout_minutes * 60,
                        )
                status = "destroyed-with-fallback"
            for stale in install_dir.glob("terraform.tfstate*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            destroyed.append(
                {"project_key": project.key(), "cluster_name": cluster.name, "status": status}
            )
    return {"name": spec.name, "clusters": destroyed}


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
            clusters.append(
                {
                    "project_key": project.key(),
                    "cluster_name": cluster.name,
                    "status": "deployed" if saved else "unknown",
                    **({k: v for k, v in saved.items()} if saved else {}),
                }
            )
    return {"name": spec.name, "clusters": clusters}

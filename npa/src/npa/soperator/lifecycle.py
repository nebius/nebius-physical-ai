"""Deploy / destroy / status for npa-managed soperator clusters.

Wraps the ``nebius/nebius-solutions-library`` soperator Terraform recipe and
applies the post-deploy fixes required to make the 4.1.0 stable release usable
(monitoring CRDs, the ``ncclInspectorPreConf`` CRD gap, the cluster-name-
prefixed slurm-scripts configmap). Node registration helpers are best-effort.

Reuses the terraform subprocess helpers from ``npa.cli.cluster.terraform_lifecycle``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from npa.cli.cluster.terraform_lifecycle import (
    _require_bin,
    _run_capture,
    _run_stream,
    _terraform_env,
)
from npa.clients.config import resolve_environment
from npa.soperator.spec import SoperatorSpec
from npa.soperator.tfvars import render_tfvars

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_PROMETHEUS_CRD_BASE = (
    "https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/"
    "v0.76.0/example/prometheus-operator-crd"
)
_PROMETHEUS_CRDS = (
    "monitoring.coreos.com_servicemonitors.yaml",
    "monitoring.coreos.com_podmonitors.yaml",
    "monitoring.coreos.com_probes.yaml",
)

# Sidecar written next to the generated tfvars so ``destroy`` can rebuild the
# same TF_VAR_* env the recipe requires. region/tenant/project/subnet/o11y are
# passed as env vars at apply time (not persisted in terraform.tfvars), so a
# later ``terraform destroy`` would fail on "No value for required variable"
# without these.
_ENV_SIDECAR = ".npa-soperator-env.json"


def _write_env_sidecar(
    install_dir: Path,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
    o11y_profile: str,
) -> None:
    (install_dir / _ENV_SIDECAR).write_text(
        json.dumps(
            {
                "region": region,
                "tenant_id": tenant_id,
                "project_id": project_id,
                "subnet_id": subnet_id,
                "o11y_profile": o11y_profile,
            },
            indent=2,
        )
    )


def _load_env_sidecar(install_dir: Path) -> dict[str, str] | None:
    path = install_dir / _ENV_SIDECAR
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _api_domain(region: str) -> str:
    """Nebius API domain for a region (the recipe hardcodes the EU domain)."""

    return "api.eu.nebius.cloud:443" if region.startswith("eu") else "api.nebius.cloud:443"


def _resolve_solutions_library(terraform_dir: Path | None, work_root: Path, ref: str) -> Path:
    """Return the soperator recipe dir, cloning the solutions-library if needed."""

    if terraform_dir is not None:
        path = terraform_dir.expanduser().resolve()
        if not (path / "installations" / "example").exists():
            raise ValueError(
                f"{path} is not a soperator recipe dir (missing installations/example)"
            )
        return path
    clone_dir = work_root / "nebius-solutions-library"
    if not (clone_dir / "soperator" / "installations" / "example").exists():
        work_root.mkdir(parents=True, exist_ok=True)
        git = _require_bin("git")
        _run_stream(
            [git, "clone", "--depth", "1", "--branch", ref, _SOLUTIONS_LIBRARY_REPO, str(clone_dir)],
            timeout=600,
        )
    return clone_dir / "soperator"


def _nebius_cli_env() -> dict[str, str]:
    """Environment for direct ``nebius`` CLI calls (pre-flight / cleanup).

    A stale ambient ``NEBIUS_IAM_TOKEN`` (e.g. an expired cloud-env token left in
    the parent process) is used by the CLI in preference to the active profile's
    exec-plugin, so pre-flight calls like ``vpc subnet list`` fail Unauthenticated
    even though the profile can mint a fresh token. Drop it so the CLI falls back
    to the auto-refreshing profile credential -- unless the caller explicitly opts
    into reuse (NPA_REUSE_IAM_TOKEN, e.g. CI injecting a short-lived token).
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


def _resolve_subnet(nebius_bin: str, project_id: str, env: dict[str, str]) -> str:
    result = _run_capture(
        [nebius_bin, "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        env=env,
    )
    payload = json.loads(result.stdout or "{}")
    items = payload.get("items") or []
    if not items:
        raise ValueError(f"no VPC subnet found in project {project_id}")
    return str(items[0].get("metadata", {}).get("id") or "")


def _prepare_installation(recipe_dir: Path, spec: SoperatorSpec, region: str) -> Path:
    """Create installations/<name> with the recipe files + generated tfvars."""

    example = recipe_dir / "installations" / "example"
    install_dir = recipe_dir / "installations" / spec.name
    install_dir.mkdir(parents=True, exist_ok=True)
    for item in ("main.tf", "variables.tf", "terraform.tf", "driver_presets.tf"):
        src = example / item
        if src.exists():
            shutil.copy2(src, install_dir / item)
    assets = example / "assets"
    if assets.exists():
        shutil.copytree(assets, install_dir / "assets", dirs_exist_ok=True)

    # Patch the hardcoded provider domain for the target region.
    terraform_tf = install_dir / "terraform.tf"
    if terraform_tf.exists():
        text = terraform_tf.read_text()
        text = text.replace("api.eu.nebius.cloud:443", _api_domain(region))
        terraform_tf.write_text(text)

    (install_dir / "terraform.tfvars").write_text(render_tfvars(spec))
    return install_dir


def _soperator_tf_env(
    nebius_bin: str,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
) -> dict[str, str]:
    env = _terraform_env(nebius_bin)
    env["TF_VAR_region"] = region
    env["TF_VAR_iam_tenant_id"] = tenant_id
    env["TF_VAR_iam_project_id"] = project_id
    # o11y is disabled in tfvars, but the variables are required to parse.
    env["TF_VAR_o11y_iam_tenant_id"] = tenant_id
    env["TF_VAR_o11y_profile"] = os.environ.get("NPA_NEBIUS_PROFILE", "") or "default"
    env["TF_VAR_vpc_subnet_id"] = subnet_id
    return env


def _terraform_cluster_id(terraform_bin: str, install_dir: Path, env: dict[str, str]) -> str:
    """Return the mk8s cluster id from Terraform state (empty if not found)."""

    result = _run_capture(
        [terraform_bin, "state", "pull"], cwd=install_dir, env=env, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ""
    for resource in state.get("resources", []):
        if resource.get("type") != "nebius_mk8s_v1_cluster":
            continue
        for instance in resource.get("instances", []):
            cid = instance.get("attributes", {}).get("id")
            if cid:
                return str(cid)
    return ""


def _find_cluster_id_by_name(
    nebius_bin: str, project_id: str, cluster_name: str, env: dict[str, str]
) -> str:
    """Return the mk8s cluster id matching *cluster_name* (empty if none)."""

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


def _refresh_kube_credentials(
    nebius_bin: str, cluster_id: str, context: str, env: dict[str, str]
) -> None:
    """Write an admin kubeconfig context for the cluster (recipe writes a limited SA)."""

    _run_capture(
        [
            nebius_bin, "mk8s", "cluster", "get-credentials",
            "--id", cluster_id, "--external", "--force", "--context-name", context,
        ],
        env=env,
        check=False,
    )


def _install_monitoring_crds(
    kubectl_bin: str, context: str, *, on_status: Callable[[str], None] | None = None
) -> None:
    """Install prometheus-operator CRDs the soperator operator chart requires.

    The operator chart creates a ServiceMonitor unconditionally; with telemetry
    off the recipe never installs its CRD, so the operator HelmRelease cannot
    install. These must be present before the operator reconciles.

    kubectl runs with the ambient ``NEBIUS_IAM_TOKEN`` stripped (via
    ``_nebius_cli_env``): a stale token shadows the kubeconfig exec-plugin and
    makes the apply fail Unauthenticated. Each apply is retried, and the
    ServiceMonitor CRD is confirmed registered before returning -- swallowing a
    failure here otherwise surfaces only ~an hour later as an operator
    HelmRelease InstallFailed and a ``wait_for_slurm_cluster_hr`` timeout.
    """

    kube_env = _nebius_cli_env()
    _log(on_status, "installing prometheus-operator CRDs (ServiceMonitor/PodMonitor/Probe)")
    for crd in _PROMETHEUS_CRDS:
        last: subprocess.CompletedProcess[str] | None = None
        for _attempt in range(3):
            last = _run_capture(
                [kubectl_bin, "--context", context, "apply", "--server-side", "-f",
                 f"{_PROMETHEUS_CRD_BASE}/{crd}"],
                env=kube_env,
                check=False,
            )
            if last.returncode == 0:
                break
            time.sleep(5)
        if last is None or last.returncode != 0:
            detail = (last.stderr or last.stdout).strip() if last else ""
            raise RuntimeError(
                f"failed to install prometheus-operator CRD {crd} after 3 attempts"
                + (f": {detail}" if detail else "")
            )
    # Confirm the ServiceMonitor CRD is actually registered: the operator chart
    # renders a ServiceMonitor and cannot install without it, so a no-op apply
    # (wrong context / swallowed auth error) must fail loudly here, not later.
    check = _run_capture(
        [kubectl_bin, "--context", context, "get", "crd",
         "servicemonitors.monitoring.coreos.com", "-o", "name"],
        env=kube_env,
        check=False,
    )
    if check.returncode != 0 or not check.stdout.strip():
        detail = (check.stderr or check.stdout).strip()
        raise RuntimeError(
            "prometheus-operator ServiceMonitor CRD not present after install"
            + (f": {detail}" if detail else "")
        )


def _patch_slurmcluster_crd(kubectl_bin: str, context: str) -> bool:
    """Patch the SlurmCluster CRD to accept plugStackConfig.ncclInspectorPreConf.

    Idempotent. Returns True once the CRD exists and the patch is applied. The
    CRD is created by the operator, so this only succeeds after the operator
    installs -- callers should retry until it returns True.
    """

    kube_env = _nebius_cli_env()
    got = _run_capture(
        [kubectl_bin, "--context", context, "get", "crd",
         "slurmclusters.slurm.nebius.ai", "-o", "name"],
        env=kube_env,
        check=False,
    )
    if got.returncode != 0 or not got.stdout.strip():
        return False
    _run_capture(
        [kubectl_bin, "--context", context, "patch", "crd",
         "slurmclusters.slurm.nebius.ai", "--type=json", "-p",
         '[{"op":"add","path":"/spec/versions/0/schema/openAPIV3Schema/'
         'properties/spec/properties/plugStackConfig/'
         'x-kubernetes-preserve-unknown-fields","value":true}]'],
        env=kube_env,
        check=False,
    )
    return True


def _ensure_scripts_configmap(kubectl_bin: str, context: str, namespace: str) -> bool:
    """Create the cluster-name-prefixed <ns>-slurm-scripts configmap.

    The nodesets chart mounts ``<ns>-slurm-scripts`` while the slurm-cluster
    chart creates the unprefixed ``slurm-scripts`` (a chart naming skew).
    Idempotent; returns True once the prefixed copy exists.
    """

    kube_env = _nebius_cli_env()
    target = f"{namespace}-slurm-scripts"
    exists = _run_capture(
        [kubectl_bin, "--context", context, "get", "cm", target, "-n", namespace, "-o", "name"],
        env=kube_env,
        check=False,
    )
    if exists.returncode == 0 and exists.stdout.strip():
        return True
    src = _run_capture(
        [kubectl_bin, "--context", context, "get", "cm", "slurm-scripts",
         "-n", namespace, "-o", "json"],
        env=kube_env,
        check=False,
    )
    if src.returncode != 0 or not src.stdout.strip():
        return False
    try:
        cm = json.loads(src.stdout)
    except json.JSONDecodeError:
        return False
    cm["metadata"] = {"name": target, "namespace": namespace}
    subprocess.run(
        [kubectl_bin, "--context", context, "apply", "-f", "-"],
        input=json.dumps(cm),
        text=True,
        env=kube_env,
        check=False,
    )
    return True


def _mid_apply_fix_loop(
    kubectl_bin: str,
    context: str,
    name: str,
    *,
    namespace: str = "soperator",
    stop: "threading.Event | None" = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Apply mid-apply fixes while phase 2 blocks on the slurm-cluster HelmRelease.

    The operator creates the SlurmCluster CRD and the slurm-scripts configmap
    *during* phase 2, and the slurm-cluster / nodesets HelmReleases then block on
    the CRD patch + the prefixed configmap. Poll and apply both as soon as they
    appear so phase 2 can converge unattended.
    """

    crd_done = False
    cm_done = False
    logged_crd = False
    logged_cm = False
    while stop is None or not stop.is_set():
        if not crd_done and _patch_slurmcluster_crd(kubectl_bin, context):
            crd_done = True
            if not logged_crd:
                _log(on_status, "mid-apply: patched SlurmCluster CRD (ncclInspectorPreConf)")
                logged_crd = True
        if not cm_done and _ensure_scripts_configmap(kubectl_bin, context, namespace):
            cm_done = True
            if not logged_cm:
                _log(on_status, f"mid-apply: ensured {namespace}-slurm-scripts configmap")
                logged_cm = True
        if crd_done and cm_done:
            return
        if stop is not None:
            stop.wait(15)
        else:
            time.sleep(15)


def deploy_cluster(
    spec: SoperatorSpec,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = "main",
    project: str | None = None,
    timeout_minutes: int = 90,
    apply_fixes: bool = True,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Deploy a soperator cluster described by *spec*. Returns cluster metadata."""

    spec.validate()
    envcfg = resolve_environment(
        project=project,
        project_id=spec.project_id or None,
        tenant_id=spec.tenant_id or None,
        region=spec.region or None,
    )
    region = spec.region or envcfg.region
    tenant_id = spec.tenant_id or envcfg.tenant_id
    project_id = spec.project_id or envcfg.project_id
    if not (region and tenant_id and project_id):
        raise ValueError(
            "region, tenant_id and project_id must be resolvable from the spec or ~/.npa config"
        )

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(terraform_dir, work_root, solutions_library_ref)
    install_dir = _prepare_installation(recipe_dir, spec, region)
    _log(on_status, f"Installation dir: {install_dir}")

    subnet_id = spec.subnet_id or _resolve_subnet(nebius_bin, project_id, _nebius_cli_env())
    env = _soperator_tf_env(
        nebius_bin,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
    )
    # Persist the env the recipe requires so ``destroy`` can reconstruct it.
    _write_env_sidecar(
        install_dir,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
        o11y_profile=env["TF_VAR_o11y_profile"],
    )

    context = f"nebius-{spec.name}-slurm"
    _log(on_status, "terraform init")
    _run_stream([terraform_bin, "init"], cwd=install_dir, env=env, timeout=900)

    if apply_fixes:
        # Two-phase apply: the soperator operator HelmRelease is reconciled inside
        # the full apply and blocks on the prometheus ServiceMonitor CRD. With
        # telemetry off, that CRD is not installed by the recipe, so a single
        # apply times out waiting for the operator. Phase 1 brings up the mk8s
        # cluster + node groups (and writes the kube context); we then refresh
        # admin credentials and install the monitoring CRDs so the operator can
        # install cleanly in phase 2.
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        _log(on_status, "terraform apply (phase 1: k8s cluster + node groups)")
        _run_stream(
            [terraform_bin, "apply", "-target=module.k8s", "-auto-approve"],
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )
        cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
        if cluster_id:
            _log(on_status, "refreshing kube admin credentials")
            _refresh_kube_credentials(nebius_bin, cluster_id, context, env)
        _log(on_status, "installing monitoring CRDs (before operator reconcile)")
        _install_monitoring_crds(kubectl_bin, context, on_status=on_status)
        _log(on_status, f"terraform apply (phase 2: operator + Slurm; {len(spec.workers)} worker pool(s))")
        # The SlurmCluster CRD is created by the operator *during* phase 2, and the
        # slurm-cluster HelmRelease then blocks on it accepting
        # plugStackConfig.ncclInspectorPreConf (a chart/CRD skew); the nodesets
        # chart likewise mounts a cluster-name-prefixed slurm-scripts configmap.
        # Both must be fixed mid-apply, so run a concurrent fixer while phase 2
        # blocks on wait_for_slurm_cluster_hr.
        stop = threading.Event()
        fixer = threading.Thread(
            target=_mid_apply_fix_loop,
            args=(kubectl_bin, context, spec.name),
            kwargs={"stop": stop, "on_status": on_status},
            daemon=True,
        )
        fixer.start()
        try:
            _run_stream(
                [terraform_bin, "apply", "-auto-approve"],
                cwd=install_dir,
                env=env,
                timeout=timeout_minutes * 60,
            )
        finally:
            stop.set()
            fixer.join(timeout=10)
    else:
        _log(on_status, f"terraform apply ({len(spec.workers)} worker pool(s))")
        _run_stream(
            [terraform_bin, "apply", "-auto-approve"],
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
        )

    result: dict[str, Any] = {
        "name": spec.name,
        "region": region,
        "project_id": project_id,
        "install_dir": str(install_dir),
        "kube_context": context,
        "worker_pools": [p.name for p in spec.workers],
        "docker_cache_pools": [p.name for p in spec.workers if p.docker_cache],
    }

    if apply_fixes:
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        apply_post_deploy_fixes(context, kubectl_bin, on_status=on_status)
        result["post_deploy_fixes"] = "applied"

    return result


def destroy_cluster(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = "main",
    project: str | None = None,
    timeout_minutes: int = 90,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Destroy an npa-managed soperator cluster by name."""

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(terraform_dir, work_root, solutions_library_ref)
    install_dir = recipe_dir / "installations" / name
    if not install_dir.exists():
        raise ValueError(f"no installation found for cluster {name!r} at {install_dir}")

    # ``terraform destroy`` still parses the config, so the region/tenant/project/
    # subnet/o11y variables (passed as env at apply time, never written to
    # terraform.tfvars) must be set or destroy fails on "No value for required
    # variable". Prefer the sidecar written at deploy time; fall back to
    # re-resolving from ~/.npa for installs predating the sidecar.
    saved = _load_env_sidecar(install_dir)
    if saved and saved.get("region") and saved.get("tenant_id") and saved.get("project_id"):
        env = _soperator_tf_env(
            nebius_bin,
            region=str(saved["region"]),
            tenant_id=str(saved["tenant_id"]),
            project_id=str(saved["project_id"]),
            subnet_id=str(saved.get("subnet_id") or ""),
        )
        if saved.get("o11y_profile"):
            env["TF_VAR_o11y_profile"] = str(saved["o11y_profile"])
    else:
        envcfg = resolve_environment(project=project)
        region = envcfg.region
        tenant_id = envcfg.tenant_id
        project_id = envcfg.project_id
        if not (region and tenant_id and project_id):
            raise ValueError(
                "cannot resolve region/tenant/project to destroy "
                f"{name!r}: no env sidecar at {install_dir / _ENV_SIDECAR} and "
                "~/.npa config is incomplete (pass --project)"
            )
        subnet_id = _resolve_subnet(nebius_bin, project_id, _nebius_cli_env())
        env = _soperator_tf_env(
            nebius_bin,
            region=region,
            tenant_id=tenant_id,
            project_id=project_id,
            subnet_id=subnet_id,
        )
    _log(on_status, f"terraform destroy: {name}")
    _run_stream([terraform_bin, "init"], cwd=install_dir, env=env, timeout=900)
    cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
    project_id = str(
        (saved or {}).get("project_id")
        or env.get("TF_VAR_iam_project_id")
        or ""
    )
    # An interrupted deploy can leave the cloud cluster running while local
    # Terraform state is empty, so cluster_id is blank here. Fall back to finding
    # the mk8s cluster by its recipe name (soperator-<name>) so destroy can still
    # tear it down instead of silently no-op'ing.
    if not cluster_id and project_id:
        cluster_id = _find_cluster_id_by_name(nebius_bin, project_id, f"soperator-{name}", env)
        if cluster_id:
            _log(on_status, f"terraform state empty; found cluster {cluster_id} by name")

    # Reclaim CSI-provisioned PVC disks (NFS + any dynamic volumes) BEFORE the
    # cluster is torn down. Deleting the mk8s cluster does NOT cascade-delete the
    # NETWORK_SSD_IO_M3 disks backing PVCs, so they leak against the (small) IO_M3
    # quota across deploy/destroy cycles. Delete the PVCs while the cluster is
    # still reachable so the CSI provisioner releases their backing disks.
    if cluster_id:
        context = f"nebius-{name}-slurm"
        _refresh_kube_credentials(nebius_bin, cluster_id, context, env)
        kubectl_bin = shutil.which(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        if kubectl_bin:
            _log(on_status, "reclaiming CSI PVC disks before teardown")
            _run_capture(
                [kubectl_bin, "--context", context, "delete", "pvc", "--all",
                 "--all-namespaces", "--wait=false", "--timeout=60s"],
                env=env,
                check=False,
                timeout=120,
            )
            time.sleep(20)  # give the CSI provisioner a moment to delete disks

    # Best-effort terraform destroy. The recipe's disk_cleanup local-exec and
    # occasional node-group deletion races can fail even when the cluster itself
    # is removable, so don't hard-fail here -- fall through to a direct delete +
    # state reset so the install dir is reusable and quota is freed.
    destroy = _run_capture(
        [terraform_bin, "destroy", "-auto-approve"],
        cwd=install_dir,
        env=env,
        timeout=timeout_minutes * 60,
        check=False,
    )
    if destroy.returncode != 0:
        _log(on_status, "terraform destroy reported errors; falling back to direct cleanup")

    # Ensure the mk8s cluster is actually gone (cascades node groups + instances).
    if cluster_id:
        still = _run_capture(
            [nebius_bin, "mk8s", "cluster", "get", "--id", cluster_id, "--format", "json"],
            env=env,
            check=False,
        )
        if still.returncode == 0 and still.stdout.strip():
            _log(on_status, f"deleting mk8s cluster {cluster_id} directly")
            _run_capture(
                [nebius_bin, "mk8s", "cluster", "delete", "--id", cluster_id],
                env=env,
                check=False,
                timeout=timeout_minutes * 60,
            )
            # Wait for the cluster to actually disappear before cleaning up VPC
            # allocations below. The delete call can return while the cluster (and
            # its cloud-controller-manager) still exists; if we delete the static-IP
            # allocation while the CCM is alive it will re-create a same-named orphan
            # that isn't in terraform state, and the next deploy fails with
            # "Allocation ... already exists" (AlreadyExists). Poll get until gone.
            deadline = time.monotonic() + timeout_minutes * 60
            while time.monotonic() < deadline:
                gone = _run_capture(
                    [nebius_bin, "mk8s", "cluster", "get", "--id", cluster_id, "--format", "json"],
                    env=env,
                    check=False,
                )
                if gone.returncode != 0 or not gone.stdout.strip():
                    break
                time.sleep(15)

    # Best-effort delete filesystems this cluster created (jail / controller-spool
    # / accounting are named ``soperator-<name>-*``) so they don't linger against
    # quota. NOTE: the recipe prefixes every filesystem with ``soperator-`` -- the
    # match below must use that full prefix, otherwise orphans survive the destroy
    # and the next deploy fails with "filesystem ... already exists" (AlreadyExists).
    if project_id:
        fs_list = _run_capture(
            [nebius_bin, "compute", "filesystem", "list", "--parent-id", project_id, "--format", "json"],
            env=env,
            check=False,
        )
        try:
            items = json.loads(fs_list.stdout or "{}").get("items", [])
        except json.JSONDecodeError:
            items = []
        for item in items:
            meta = item.get("metadata", {})
            fs_name = str(meta.get("name") or "")
            fs_id = str(meta.get("id") or "")
            if fs_id and fs_name.startswith(f"soperator-{name}-"):
                _log(on_status, f"deleting orphaned filesystem {fs_name}")
                _run_capture(
                    [nebius_bin, "compute", "filesystem", "delete", "--id", fs_id],
                    env=env,
                    check=False,
                )

    # Best-effort delete orphaned VPC allocations this cluster created. The recipe
    # provisions a static public IP named ``soperator-<name>-public-static-ip``
    # for the login LoadBalancer. The Nebius cloud-controller-manager can also
    # *re-create* a same-named allocation mid-teardown (a LoadBalancer-service
    # race) after terraform has already deleted the in-state copy, leaving an
    # orphan that isn't in state -- so a later ``terraform apply`` fails with
    # "Allocation with name 'soperator-<name>-public-static-ip' already exists".
    # These are safe to remove once the cluster (and its CCM) is gone. Runs after
    # the direct cluster delete above so the CCM can't re-create them again.
    if project_id:
        alloc_list = _run_capture(
            [nebius_bin, "vpc", "allocation", "list", "--parent-id", project_id, "--format", "json"],
            env=env,
            check=False,
        )
        try:
            items = json.loads(alloc_list.stdout or "{}").get("items", [])
        except json.JSONDecodeError:
            items = []
        for item in items:
            meta = item.get("metadata", {})
            alloc_name = str(meta.get("name") or "")
            alloc_id = str(meta.get("id") or "")
            if alloc_id and alloc_name.startswith(f"soperator-{name}-"):
                _log(on_status, f"deleting orphaned VPC allocation {alloc_name}")
                _run_capture(
                    [nebius_bin, "vpc", "allocation", "delete", "--id", alloc_id],
                    env=env,
                    check=False,
                )

    # Reset local terraform state so the install dir is clean for a redeploy.
    for stale in install_dir.glob("terraform.tfstate*"):
        try:
            stale.unlink()
        except OSError:
            pass
    _log(on_status, f"destroy complete: {name}")


def apply_post_deploy_fixes(
    context: str,
    kubectl_bin: str,
    *,
    namespace: str = "soperator",
    on_status: Callable[[str], None] | None = None,
    timeout_minutes: int = 20,
) -> None:
    """Apply the fixes the 4.1.0-stable recipe needs to reach a working Slurm.

    1. Install prometheus-operator CRDs (operator chart needs ServiceMonitor even
       when telemetry is off).
    2. Patch the SlurmCluster CRD to accept ``plugStackConfig.ncclInspectorPreConf``.
    3. Create the cluster-name-prefixed ``<ns>-slurm-scripts`` configmap the
       nodesets chart mounts (chart naming skew).
    """

    _install_monitoring_crds(kubectl_bin, context, on_status=on_status)

    _log(on_status, "post-deploy: patching SlurmCluster CRD + ensuring scripts configmap")
    deadline = time.monotonic() + timeout_minutes * 60
    crd_done = False
    cm_done = False
    while time.monotonic() < deadline and not (crd_done and cm_done):
        crd_done = crd_done or _patch_slurmcluster_crd(kubectl_bin, context)
        cm_done = cm_done or _ensure_scripts_configmap(kubectl_bin, context, namespace)
        if crd_done and cm_done:
            break
        time.sleep(15)
    if not crd_done:
        _log(on_status, "post-deploy: SlurmCluster CRD not present yet; skipped CRD patch")
    if not cm_done:
        _log(on_status, "post-deploy: slurm-scripts configmap not present yet; skipped")

    _register_slurm_workers(kubectl_bin, context, namespace, on_status=on_status)
    _log(on_status, "post-deploy: fixes applied")


def _register_slurm_workers(
    kubectl_bin: str,
    context: str,
    namespace: str,
    *,
    on_status: Callable[[str], None] | None = None,
    wait_minutes: int = 10,
) -> None:
    """Best-effort: bring DOWN worker nodes to IDLE (soperator 4.1.0 slurmrestd gap).

    In 4.1.0-stable the operator doesn't deploy slurmrestd, so soperator's
    dynamic-node registration can leave workers DOWN/not-responding with slurmctld
    resolving the bare short name. Set the FQDN NodeAddr and RESUME any node that
    isn't idle. Idempotent and non-fatal.
    """

    kube_env = _nebius_cli_env()
    ctl = ["exec", "-n", namespace, "controller-0", "-c", "slurmctld", "--"]

    def slurmctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _run_capture(
            [kubectl_bin, "--context", context, *ctl, *args], env=kube_env, check=False
        )

    deadline = time.monotonic() + wait_minutes * 60
    while time.monotonic() < deadline:
        info = slurmctl(["sinfo", "-h", "-N", "-o", "%N %t"])
        if info.returncode != 0 or not info.stdout.strip():
            time.sleep(15)
            continue
        down = [
            line.split()[0]
            for line in info.stdout.splitlines()
            if line.split() and (line.split()[1].endswith("*") or "down" in line.split()[1].lower())
        ]
        if not down:
            _log(on_status, "post-deploy: all Slurm worker nodes are responding")
            return
        for node in sorted(set(down)):
            fqdn = f"{node}.soperator-nodeset-svc.{namespace}.svc.cluster.local"
            slurmctl(["scontrol", "update", f"NodeName={node}", f"NodeAddr={fqdn}"])
            slurmctl(["scontrol", "update", f"NodeName={node}", "State=RESUME"])
        _log(on_status, f"post-deploy: registered worker node(s): {', '.join(sorted(set(down)))}")
        time.sleep(15)

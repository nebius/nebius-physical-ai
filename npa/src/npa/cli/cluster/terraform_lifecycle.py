"""Terraform-backed ``npa cluster up`` and ``npa cluster down`` commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

from npa.cluster.state import ClusterState, kubeconfig_file, save_cluster_state, utc_now_iso

_DEFAULT_TERRAFORM_SUBDIR = Path("deploy") / "cluster"
_DEFAULT_SKYPILOT_BIN = Path.home() / ".npa" / "skypilot-venv" / "bin" / "sky"
_DEFAULT_FILESTORE_SIZE_GIB = 1024
_GIB = 1024**3


def up_cmd(
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory. Defaults to ./deploy/cluster or the repo root deploy/cluster.",
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help="Kubeconfig output path. Defaults to ~/.npa/clusters/<cluster-name>/kubeconfig.",
    ),
    context_name: str = typer.Option(
        "",
        "--context",
        help="Kubeconfig context name. Defaults to the Terraform cluster name.",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--skip-validate",
        help="Validate nodes, GPU allocatable resources, GPU Operator pods, and default StorageClass.",
    ),
    sky_smoke: bool = typer.Option(
        True,
        "--sky-smoke/--skip-sky-smoke",
        help="Run a SkyPilot Kubernetes GPU smoke task and clean it up with sky down.",
    ),
    sky_gpus: str = typer.Option(
        "",
        "--sky-gpus",
        help="SkyPilot GPU demand for the smoke task. Defaults to auto-detecting the first Kubernetes GPU.",
    ),
    capacity_block_group: str = typer.Option(
        "",
        "--capacity-block-group",
        help=(
            "Optional private capacity block group ID for strict GPU node-group "
            "reservation selection. Equivalent to TF_VAR_capacity_block_group."
        ),
    ),
    validation_timeout: int = typer.Option(
        60,
        "--validation-timeout",
        help="Post-apply Kubernetes validation timeout in minutes.",
    ),
    timeout: int = typer.Option(120, "--timeout", help="Terraform apply timeout in minutes."),
) -> None:
    """Create or update the Terraform-managed NPA Kubernetes cluster."""

    tf_dir = _resolve_terraform_dir(terraform_dir)
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
    env = _terraform_env(nebius_bin)
    _apply_capacity_block_group_env(env, capacity_block_group)

    typer.echo(f"Terraform directory: {tf_dir}")
    _run_stream([terraform_bin, "init"], cwd=tf_dir, env=env, timeout=600)
    tfvars = _read_tfvars(tf_dir)
    _apply_capacity_block_group_tfvars(tfvars, capacity_block_group)
    _guard_unmanaged_duplicate(nebius_bin, terraform_bin, tf_dir, tfvars, env)
    _preflight_filestore_quota(nebius_bin, tfvars, env)

    _run_stream(
        [terraform_bin, "apply", "-auto-approve"],
        cwd=tf_dir,
        env=env,
        timeout=timeout * 60,
    )
    outputs = _terraform_outputs(terraform_bin, tf_dir, env)
    cluster = _cluster_output(outputs)
    cluster_id = str(cluster.get("id") or "")
    cluster_name = str(cluster.get("name") or tfvars.get("cluster_name") or "npa-cluster")
    if not cluster_id:
        raise typer.BadParameter("Terraform output kube_cluster.id is empty")

    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context)
    _save_terraform_cluster_state(tfvars, cluster, context, kubeconfig_path)

    typer.echo(f"Cluster ID: {cluster_id}")
    typer.echo(f"Cluster name: {cluster_name}")
    typer.echo(f"Kubeconfig: {kubeconfig_path}")

    if validate:
        validation = _validate_cluster(kubectl_bin, kubeconfig_path, tfvars, validation_timeout)
        typer.echo(
            "Validation: "
            f"{validation['ready_nodes']} Ready nodes, "
            f"{validation['total_gpus']} allocatable GPUs, "
            f"default StorageClass {validation['default_storage_class']}"
        )
    if sky_smoke:
        _run_skypilot_smoke(kubeconfig_path, context, cluster_name, sky_gpus)


def down_cmd(
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory. Defaults to ./deploy/cluster or the repo root deploy/cluster.",
    ),
    force: bool = typer.Option(False, "--force", help="Skip confirmation."),
    timeout: int = typer.Option(120, "--timeout", help="Terraform destroy timeout in minutes."),
) -> None:
    """Destroy the Terraform-managed NPA Kubernetes cluster."""

    tf_dir = _resolve_terraform_dir(terraform_dir)
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    env = _terraform_env(nebius_bin)
    if not force and not typer.confirm(f"Destroy Terraform-managed cluster in {tf_dir}?"):
        raise typer.Exit(1)
    _run_stream([terraform_bin, "init"], cwd=tf_dir, env=env, timeout=600)
    _run_stream(
        [terraform_bin, "destroy", "-auto-approve"],
        cwd=tf_dir,
        env=env,
        timeout=timeout * 60,
    )


def terraform_status(terraform_dir: Path | None = None) -> dict[str, Any] | None:
    """Return Terraform cluster outputs when state exists."""

    try:
        tf_dir = _resolve_terraform_dir(terraform_dir)
        terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
        env = os.environ.copy()
        return _terraform_outputs(terraform_bin, tf_dir, env)
    except Exception:
        return None


def _resolve_terraform_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise typer.BadParameter(f"Terraform directory does not exist: {path}")
        return path
    cwd_candidate = (Path.cwd() / _DEFAULT_TERRAFORM_SUBDIR).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = _find_repo_root(Path.cwd())
    if repo_root is not None:
        repo_candidate = (repo_root / _DEFAULT_TERRAFORM_SUBDIR).resolve()
        if repo_candidate.exists():
            return repo_candidate
    raise typer.BadParameter("Cannot find deploy/cluster; pass --terraform-dir")


def _find_repo_root(path: Path) -> Path | None:
    for current in [path, *path.parents]:
        if (current / ".git").exists():
            return current
    return None


def _require_bin(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    if Path(binary).exists():
        return binary
    raise typer.BadParameter(f"Required executable not found: {binary}")


def _terraform_env(nebius_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("TF_VAR_iam_token"):
        token = _run_capture([nebius_bin, "iam", "get-access-token"], env=env).stdout.strip()
        env["TF_VAR_iam_token"] = token
        env["NEBIUS_IAM_TOKEN"] = token
    return env


def _run_stream(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, env=env, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise typer.BadParameter(f"Command failed ({result.returncode}): {' '.join(args)}")
    return result


def _run_capture(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise typer.BadParameter(f"Command failed ({result.returncode}): {' '.join(args)}{suffix}")
    return result


def _read_tfvars(terraform_dir: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in [terraform_dir / "terraform.tfvars", *sorted(terraform_dir.glob("*.auto.tfvars"))]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$", line)
            if not match:
                continue
            key, raw_value = match.groups()
            values[key] = _parse_tfvar_scalar(raw_value)
    return values


def _apply_capacity_block_group_env(env: dict[str, str], capacity_block_group: str) -> None:
    value = capacity_block_group.strip()
    if value:
        env["TF_VAR_capacity_block_group"] = value


def _apply_capacity_block_group_tfvars(tfvars: dict[str, Any], capacity_block_group: str) -> None:
    value = capacity_block_group.strip()
    if value:
        tfvars["capacity_block_group"] = value


def _parse_tfvar_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _guard_unmanaged_duplicate(
    nebius_bin: str,
    terraform_bin: str,
    terraform_dir: Path,
    tfvars: dict[str, Any],
    env: dict[str, str],
) -> None:
    cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
    project_id = str(tfvars.get("parent_id") or os.environ.get("TF_VAR_parent_id") or "")
    if not project_id:
        typer.echo("Skipping duplicate cluster preflight: parent_id is not set in tfvars or env.", err=True)
        return
    result = _run_capture(
        [nebius_bin, "mk8s", "cluster", "list", "--parent-id", project_id, "--format", "json"],
        env=env,
    )
    payload = json.loads(result.stdout or "{}")
    matches = [
        item
        for item in payload.get("items", [])
        if item.get("metadata", {}).get("name") == cluster_name
    ]
    if not matches:
        return
    managed_ids = _terraform_state_cluster_ids(terraform_bin, terraform_dir, env)
    unmanaged = [
        item.get("metadata", {}).get("id")
        for item in matches
        if item.get("metadata", {}).get("id") not in managed_ids
    ]
    if unmanaged:
        ids = ", ".join(str(value) for value in unmanaged if value)
        raise typer.BadParameter(
            f"Cluster {cluster_name} already exists outside this Terraform state: {ids}"
        )


def _preflight_filestore_quota(nebius_bin: str, tfvars: dict[str, Any], env: dict[str, str]) -> None:
    enable_filestore = bool(_tfvar_value(tfvars, env, "enable_filestore", True))
    existing_filestore = str(_tfvar_value(tfvars, env, "existing_filestore", "") or "").strip()
    if not enable_filestore or existing_filestore:
        return
    tenant_id = str(_tfvar_value(tfvars, env, "tenant_id", "") or "").strip()
    region = str(_tfvar_value(tfvars, env, "region", "") or "").strip()
    if not tenant_id or not region:
        typer.echo(
            "Skipping shared filesystem quota preflight: tenant_id or region is not set in tfvars or env.",
            err=True,
        )
        return
    size_gib = int(_tfvar_value(tfvars, env, "filestore_disk_size_gibibytes", _DEFAULT_FILESTORE_SIZE_GIB))
    requested_bytes = size_gib * _GIB
    quota = _quota_allowance(
        nebius_bin,
        parent_id=tenant_id,
        region=region,
        name="compute.filesystem.size.network-ssd",
        env=env,
    )
    limit = _quota_limit(quota)
    usage = _quota_usage(quota)
    available = limit - usage
    if available < requested_bytes:
        raise typer.BadParameter(
            "Shared filesystem quota is insufficient for Terraform creation: "
            f"compute.filesystem.size.network-ssd available {available} bytes, "
            f"requested {requested_bytes} bytes in {region}. "
            "Provide existing_filestore or raise Shared Filesystem SSD quota before running apply."
        )


def _tfvar_value(tfvars: dict[str, Any], env: dict[str, str], key: str, default: Any) -> Any:
    if key in tfvars:
        return tfvars[key]
    return env.get(f"TF_VAR_{key}", default)


def _quota_allowance(
    nebius_bin: str,
    *,
    parent_id: str,
    region: str,
    name: str,
    env: dict[str, str],
) -> dict[str, Any]:
    result = _run_capture(
        [
            nebius_bin,
            "quotas",
            "quota-allowance",
            "get-by-name",
            "--parent-id",
            parent_id,
            "--region",
            region,
            "--name",
            name,
            "--format",
            "json",
        ],
        env=env,
    )
    return json.loads(result.stdout or "{}")


def _quota_limit(quota: dict[str, Any]) -> int:
    raw_limit = quota.get("spec", {}).get("limit")
    return int(raw_limit or 0)


def _quota_usage(quota: dict[str, Any]) -> int:
    raw_usage = quota.get("status", {}).get("usage")
    return int(raw_usage or 0)


def _terraform_state_cluster_ids(terraform_bin: str, terraform_dir: Path, env: dict[str, str]) -> set[str]:
    result = _run_capture(
        [terraform_bin, "state", "pull"],
        cwd=terraform_dir,
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    ids: set[str] = set()
    output_cluster_id = (
        state.get("outputs", {})
        .get("kube_cluster", {})
        .get("value", {})
        .get("id")
    )
    if output_cluster_id:
        ids.add(str(output_cluster_id))

    def walk(module: dict[str, Any]) -> None:
        for resource in module.get("resources", []):
            if resource.get("type") != "nebius_mk8s_v1_cluster":
                continue
            for instance in resource.get("instances", []):
                cluster_id = instance.get("attributes", {}).get("id")
                if cluster_id:
                    ids.add(str(cluster_id))
        for child in module.get("child_modules", []):
            walk(child)

    walk(state.get("values", {}).get("root_module", {}))
    for resource in state.get("resources", []):
        if resource.get("type") != "nebius_mk8s_v1_cluster":
            continue
        for instance in resource.get("instances", []):
            cluster_id = instance.get("attributes", {}).get("id")
            if cluster_id:
                ids.add(str(cluster_id))
    return ids


def _terraform_outputs(terraform_bin: str, terraform_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    result = _run_capture([terraform_bin, "output", "-json"], cwd=terraform_dir, env=env)
    return json.loads(result.stdout or "{}")


def _cluster_output(outputs: dict[str, Any]) -> dict[str, Any]:
    value = outputs.get("kube_cluster", {}).get("value")
    if not isinstance(value, dict):
        raise typer.BadParameter("Terraform output kube_cluster is missing")
    return value


def _write_kubeconfig(nebius_bin: str, cluster_id: str, kubeconfig_path: Path, context: str) -> None:
    kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
    _run_stream(
        [
            nebius_bin,
            "mk8s",
            "cluster",
            "get-credentials",
            "--id",
            cluster_id,
            "--force",
            "--kubeconfig",
            str(kubeconfig_path),
            "--external",
            "--context-name",
            context,
        ],
        timeout=120,
    )


def _save_terraform_cluster_state(
    tfvars: dict[str, Any],
    cluster: dict[str, Any],
    context: str,
    kubeconfig_path: Path,
) -> None:
    endpoints = cluster.get("endpoints") if isinstance(cluster.get("endpoints"), dict) else {}
    state = ClusterState(
        name=context,
        cluster_id=str(cluster.get("id") or ""),
        project_id=str(tfvars.get("parent_id") or ""),
        region=str(tfvars.get("region") or ""),
        node_count=int(tfvars.get("cpu_nodes_count") or 0) + int(tfvars.get("gpu_nodes_count") or 0),
        node_platform=str(tfvars.get("gpu_nodes_platform") or ""),
        node_preset=str(tfvars.get("gpu_nodes_preset") or ""),
        k8s_version=str(tfvars.get("k8s_version") or ""),
        subnet_id=str(tfvars.get("subnet_id") or ""),
        created_at=utc_now_iso(),
        last_seen_state="RUNNING",
        endpoint=str(endpoints.get("public_endpoint") or ""),
        kubeconfig_path=str(kubeconfig_path),
    )
    save_cluster_state(
        state,
        metadata={
            "managed_by": "npa cluster terraform",
            "event": "kubeconfig_written",
            "updated_at": utc_now_iso(),
            "teardown": "Run `npa cluster down --terraform-dir deploy/cluster --force` when finished.",
        },
    )


def _validate_cluster(
    kubectl_bin: str,
    kubeconfig_path: Path,
    tfvars: dict[str, Any],
    timeout_minutes: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_minutes * 60
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            return _validate_cluster_once(kubectl_bin, kubeconfig_path, tfvars)
        except typer.BadParameter as exc:
            last_error = str(exc)
            typer.echo(f"Validation pending: {last_error}")
            time.sleep(30)
    raise typer.BadParameter(
        f"Cluster validation did not pass within {timeout_minutes} minutes: {last_error}"
    )


def _validate_cluster_once(kubectl_bin: str, kubeconfig_path: Path, tfvars: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path)
    nodes = json.loads(_run_capture([kubectl_bin, "get", "nodes", "-o", "json"], env=env).stdout)
    ready_nodes = 0
    total_gpus = 0
    gpu_node_count = 0
    for node in nodes.get("items", []):
        conditions = node.get("status", {}).get("conditions", [])
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
            ready_nodes += 1
        gpu_count = int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu") or 0)
        if gpu_count:
            gpu_node_count += 1
            total_gpus += gpu_count

    expected_gpu_nodes = int(tfvars.get("gpu_nodes_count") or 0)
    expected_gpus = expected_gpu_nodes * _gpus_per_node(str(tfvars.get("gpu_nodes_preset") or ""))
    if expected_gpu_nodes and gpu_node_count != expected_gpu_nodes:
        raise typer.BadParameter(f"Expected {expected_gpu_nodes} GPU nodes, found {gpu_node_count}")
    if expected_gpus and total_gpus != expected_gpus:
        raise typer.BadParameter(f"Expected {expected_gpus} allocatable GPUs, found {total_gpus}")

    pods = json.loads(
        _run_capture([kubectl_bin, "get", "pods", "-n", "gpu-operator", "-o", "json"], env=env).stdout
    )
    if not pods.get("items"):
        raise typer.BadParameter("GPU Operator namespace has no pods")
    bad_pods = [
        pod.get("metadata", {}).get("name", "")
        for pod in pods.get("items", [])
        if pod.get("status", {}).get("phase") not in {"Running", "Succeeded"}
    ]
    if bad_pods:
        raise typer.BadParameter(f"GPU Operator pods are not ready: {', '.join(bad_pods)}")

    storage_classes = json.loads(
        _run_capture([kubectl_bin, "get", "storageclass", "-o", "json"], env=env).stdout
    )
    default_sc = ""
    for item in storage_classes.get("items", []):
        annotations = item.get("metadata", {}).get("annotations", {})
        if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
            default_sc = item.get("metadata", {}).get("name", "")
            break
    if default_sc != "csi-mounted-fs-path-sc":
        raise typer.BadParameter(f"Expected default StorageClass csi-mounted-fs-path-sc, found {default_sc}")
    return {
        "ready_nodes": ready_nodes,
        "gpu_nodes": gpu_node_count,
        "total_gpus": total_gpus,
        "default_storage_class": default_sc,
    }


def _gpus_per_node(preset: str) -> int:
    match = re.match(r"^(\d+)gpu-", preset)
    return int(match.group(1)) if match else 0


def _run_skypilot_smoke(kubeconfig_path: Path, context: str, cluster_name: str, sky_gpus: str) -> None:
    sky_bin = os.environ.get("NPA_SKYPILOT_BIN") or str(_DEFAULT_SKYPILOT_BIN)
    sky = _require_bin(sky_bin)
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path)
    infra = f"k8s/{context}"
    _run_stream([sky, "check", "kubernetes"], env=env, timeout=300)
    accelerator = sky_gpus.strip() or _detect_skypilot_gpu(sky, infra, env)
    smoke_name = _sky_cluster_name(cluster_name)
    try:
        _run_stream(
            [
                sky,
                "launch",
                "-c",
                smoke_name,
                "--infra",
                infra,
                "--gpus",
                accelerator,
                "-y",
                "nvidia-smi",
            ],
            env=env,
            timeout=1800,
        )
    finally:
        _run_stream([sky, "down", "--yes", smoke_name], env=env, timeout=600)
        _wait_for_sky_down(sky, smoke_name, env)
    typer.echo(f"SkyPilot smoke passed and {smoke_name} was removed.")


def _detect_skypilot_gpu(sky: str, infra: str, env: dict[str, str]) -> str:
    result = _run_capture([sky, "show-gpus", "--infra", infra, "--all"], env=env, timeout=300)
    for line in result.stdout.splitlines():
        if "RTX" not in line.upper() or "6000" not in line:
            continue
        columns = [column for column in re.split(r"\s{2,}", line.strip()) if column]
        if columns:
            return f"{columns[0]}:1"
    raise typer.BadParameter("Unable to auto-detect a Kubernetes GPU for SkyPilot; pass --sky-gpus")


def _sky_cluster_name(cluster_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9-]+", "-", cluster_name).strip("-").lower()
    return f"{normalized[:40]}-sky-smoke"


def _wait_for_sky_down(sky: str, cluster_name: str, env: dict[str, str]) -> None:
    for _ in range(30):
        result = _run_capture([sky, "status", "--refresh"], env=env, timeout=120, check=False)
        if cluster_name not in result.stdout:
            return
        time.sleep(10)
    raise typer.BadParameter(f"SkyPilot cluster {cluster_name} still appears in sky status")

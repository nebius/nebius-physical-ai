"""Terraform-backed ``npa cluster up`` and ``npa cluster down`` commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.cluster.state import ClusterState, kubeconfig_file, save_cluster_state, utc_now_iso

_DEFAULT_TERRAFORM_SUBDIR = Path("deploy") / "cluster"
_DEFAULT_SKYPILOT_BIN = Path.home() / ".npa" / "skypilot-venv" / "bin" / "sky"
_DEFAULT_FILESTORE_SIZE_GIB = 1024
_GIB = 1024**3
# deploy/cluster vendors nebius-solutions-library, whose k8s-rbac-bindings module
# declares `required_version >= 1.12.0` and whose o11y module uses `ephemeral`
# blocks (Terraform 1.10+). Terraform loads every referenced module during
# `init`, even the ones this config disables, so an older binary fails with a
# wall of "Unsupported Terraform Core version" / "Unsupported block type" errors
# from vendored files the operator never wrote. Check up front instead.
_MIN_TERRAFORM_VERSION = (1, 12, 0)


@resolve_typer_defaults
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
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project/tenant/region to use when tfvars omit them.",
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
    _preflight_terraform_version(terraform_bin)
    tfvars = _read_tfvars(tf_dir)
    _apply_project_tf_vars(env, project, tfvars)
    _guard_tfvars_iam_token(tf_dir, tfvars)
    _run_stream([terraform_bin, "init"], cwd=tf_dir, env=env, timeout=600)
    _apply_capacity_block_group_tfvars(tfvars, capacity_block_group)
    _guard_unmanaged_duplicate(nebius_bin, terraform_bin, tf_dir, tfvars, env)
    _preflight_filestore_quota(nebius_bin, tfvars, env)
    _preflight_gpu_capacity(nebius_bin, tfvars, env)

    apply_args = [
        terraform_bin,
        "apply",
        "-auto-approve",
        # -var beats terraform.tfvars, TF_VAR_* does not. Pass the flag value
        # explicitly so `--capacity-block-group` is not silently dropped by a
        # `capacity_block_group = ""` line in a checked-in tfvars file.
        *_capacity_block_group_var_args(capacity_block_group),
        *_ssh_public_key_var_args(tfvars, env),
    ]
    # Terraform prints only `Still creating...` while a node group retries, so a
    # cloud-side failure (QuotaFailure, no capacity) is invisible for as long as
    # the operator is willing to wait. Report node-group state alongside it, and
    # cancel the apply when the platform reports a refusal retrying cannot fix.
    watcher = _NodeGroupWatcher(nebius_bin, tfvars, env)
    watcher.start()
    try:
        _run_stream(
            apply_args,
            cwd=tf_dir,
            env=env,
            timeout=timeout * 60,
            cancel=lambda: watcher.fatal_reason,
        )
    except (typer.BadParameter, KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
        watcher.stop()
        _echo_apply_recovery(tf_dir, tfvars, isinstance(exc, KeyboardInterrupt))
        raise
    finally:
        watcher.stop()
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
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project/tenant/region to use when tfvars omit them.",
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
    _preflight_terraform_version(terraform_bin)
    tfvars = _read_tfvars(tf_dir)
    _apply_project_tf_vars(env, project, tfvars)
    _guard_tfvars_iam_token(tf_dir, tfvars)
    _run_stream([terraform_bin, "init"], cwd=tf_dir, env=env, timeout=600)
    _run_stream(
        [
            terraform_bin,
            "destroy",
            "-auto-approve",
            # Variable validation runs on destroy too, so the key has to resolve
            # here as well — but a teardown must not be blocked by a machine that
            # has no SSH key, since the value cannot affect what is destroyed.
            *_ssh_public_key_var_args(tfvars, env, allow_placeholder=True),
        ],
        cwd=tf_dir,
        env=env,
        timeout=timeout * 60,
    )


def kubeconfig_cmd(
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help="Managed Kubernetes cluster name. Defaults to the Terraform cluster_name, else npa-cluster.",
    ),
    project_id: str = typer.Option(
        "",
        "--project-id",
        help="Nebius project id holding the cluster. Defaults to tfvars/TF_VAR_parent_id, then the configured project.",
    ),
    project: str = typer.Option(
        "", "--project", help="NPA project alias whose saved project_id to use."
    ),
    context_name: str = typer.Option(
        "", "--context", help="Kubeconfig context name. Defaults to the cluster name."
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help="Kubeconfig output path. Defaults to ~/.npa/clusters/<context>/kubeconfig.",
    ),
    terraform_dir: Path | None = typer.Option(
        None, "--terraform-dir", help="Terraform cluster directory to read tfvars from."
    ),
) -> None:
    """Write a kubeconfig for a Managed Kubernetes cluster that already exists.

    An interrupted `npa cluster up` (or one provisioned elsewhere) leaves a running
    cluster with no local kubeconfig, which nothing could then use. This adopts it:
    it writes the kubeconfig and cluster state that `npa cluster status` and
    `npa workbench workflow submit --infra k8s/<context>` read.
    """
    from npa.clients.config import resolve_environment

    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    env = _terraform_env(nebius_bin)
    tfvars: dict[str, Any] = {}
    try:
        tfvars = _read_tfvars(_resolve_terraform_dir(terraform_dir))
    except typer.BadParameter:
        # Adopting a cluster does not require a Terraform directory at all.
        tfvars = {}

    name = cluster_name.strip() or str(tfvars.get("cluster_name") or "npa-cluster")
    resolved_project = project_id.strip() or str(
        tfvars.get("parent_id") or os.environ.get("TF_VAR_parent_id") or ""
    )
    if not resolved_project:
        saved = resolve_environment(project or None)
        resolved_project = str(getattr(saved, "project_id", "") or "")
    if not resolved_project:
        raise typer.BadParameter(
            "Cannot tell which Nebius project holds the cluster. Pass --project-id "
            "<id> (or --project <alias> after `npa configure`)."
        )

    result = _run_capture(
        [nebius_bin, "mk8s", "cluster", "list", "--parent-id", resolved_project, "--format", "json"],
        env=env,
    )
    matches = [
        item
        for item in (json.loads(result.stdout or "{}") or {}).get("items", [])
        if str((item.get("metadata") or {}).get("name", "")) == name
    ]
    if not matches:
        raise typer.BadParameter(
            f"No Managed Kubernetes cluster named {name!r} in project {resolved_project}. "
            f"List what exists with `nebius mk8s cluster list --parent-id {resolved_project}`."
        )
    metadata = matches[0].get("metadata") or {}
    cluster_id = str(metadata.get("id") or "")
    if not cluster_id:
        raise typer.BadParameter(f"Cluster {name!r} has no id in the Nebius response")

    context = context_name.strip() or name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context)
    _save_terraform_cluster_state(
        {**tfvars, "parent_id": resolved_project, "cluster_name": name},
        {"id": cluster_id, "name": name},
        context,
        kubeconfig_path,
    )
    typer.echo(f"Cluster ID: {cluster_id}")
    typer.echo(f"Kubeconfig: {kubeconfig_path}")
    typer.echo(f"Context: {context}")
    typer.echo(
        f"Submit against it with `--infra k8s/{context}` (npa resolves this file), "
        f"or export KUBECONFIG={kubeconfig_path} for kubectl."
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


def _apply_project_tf_vars(env: dict[str, str], project: str, tfvars: dict[str, Any]) -> None:
    """Fill TF_VAR_parent_id/tenant_id/region from ``~/.npa/config.yaml``.

    `npa provision-if-absent` exports these before calling `up`, so a cluster
    provisioned that way has no `terraform.tfvars` — and a later bare
    `npa cluster down --force` then failed with "No value for required variable",
    leaving the VPC/subnet orphaned until the operator exported them by hand.
    Values already in tfvars or the environment win.
    """
    missing = [
        key
        for key, var in (("parent_id", "TF_VAR_parent_id"), ("tenant_id", "TF_VAR_tenant_id"), ("region", "TF_VAR_region"))
        if key not in tfvars and not str(env.get(var, "") or "").strip()
    ]
    if not missing:
        return
    try:
        from npa.clients.config import resolve_environment

        saved = resolve_environment(project or None)
    except Exception:  # noqa: BLE001 - no saved config is a normal first run
        saved = None
    if saved is None:
        return
    resolved: list[str] = []
    for key, var, value in (
        ("parent_id", "TF_VAR_parent_id", str(getattr(saved, "project_id", "") or "")),
        ("tenant_id", "TF_VAR_tenant_id", str(getattr(saved, "tenant_id", "") or "")),
        ("region", "TF_VAR_region", str(getattr(saved, "region", "") or "")),
    ):
        if key in missing and value:
            env[var] = value
            resolved.append(f"{key}={value}")
    if resolved:
        typer.echo(f"Using saved project settings from ~/.npa/config.yaml: {', '.join(resolved)}")


def _terraform_env(nebius_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    # A stale ambient IAM token (e.g. a cloud-env token) silently shadows the
    # intended Nebius profile and mints kubeconfig/registry credentials for the
    # wrong principal -- the cause of Forbidden jobs/pods/nodes and 401 image
    # pulls. Mint a fresh token by default after clearing any stale token; opt
    # back into reuse only when NPA_REUSE_IAM_TOKEN is explicitly set (e.g. CI
    # that injects a short-lived token intentionally).
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if reuse and env.get("TF_VAR_iam_token"):
        return env
    env.pop("TF_VAR_iam_token", None)
    env.pop("NEBIUS_IAM_TOKEN", None)
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
    cancel: Callable[[], str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *args*, streaming output.

    ``cancel`` is polled while the command runs; when it returns a non-empty
    reason the process is asked to stop (SIGINT first, which Terraform handles as
    a graceful shutdown that still persists state) and the reason is raised.
    """
    if cancel is None:
        result = subprocess.run(args, cwd=cwd, env=env, text=True, timeout=timeout, check=False)
        if result.returncode != 0:
            raise typer.BadParameter(f"Command failed ({result.returncode}): {' '.join(args)}")
        return result

    reason = ""
    process = subprocess.Popen(args, cwd=cwd, env=env, text=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while process.poll() is None:
            if not reason:
                reason = cancel() or ""
                if reason:
                    _stop_process(process)
            if deadline is not None and time.monotonic() >= deadline:
                _stop_process(process)
                raise subprocess.TimeoutExpired(args, timeout or 0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        _stop_process(process)
        raise
    returncode = process.returncode or 0
    if reason:
        raise typer.BadParameter(f"Cancelled `{' '.join(args[:2])}`: {reason}")
    if returncode != 0:
        raise typer.BadParameter(f"Command failed ({returncode}): {' '.join(args)}")
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout="", stderr="")


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Ask a process to stop: SIGINT (graceful for terraform), then SIGTERM/kill."""
    for signal_name, grace in (("interrupt", 30.0), ("terminate", 10.0)):
        if process.poll() is not None:
            return
        try:
            if signal_name == "interrupt":
                process.send_signal(signal.SIGINT)
            else:
                process.terminate()
        except OSError:  # pragma: no cover - process already gone
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:  # pragma: no cover - terraform ignoring both signals
        process.kill()


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


def _capacity_block_group_var_args(capacity_block_group: str) -> list[str]:
    value = capacity_block_group.strip()
    if not value:
        return []
    return ["-var", f"capacity_block_group={value}"]


#: Node-group SSH keys, most modern first. `ssh-keygen` has defaulted to ed25519
#: for years, and the rest of the CLI (agent deploy, the tfvars example) uses it.
_SSH_PUBLIC_KEY_NAMES = ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub")


def _ssh_public_key_var_args(
    tfvars: dict[str, Any], env: dict[str, str], *, allow_placeholder: bool = False
) -> list[str]:
    """Pin an SSH public key that exists on this machine.

    The vendored module validates ``fileexists(ssh_public_key.path)`` against a
    default of ``~/.ssh/id_rsa.pub``, so the zero-config path
    (``npa provision-if-absent``, which passes no key) fails at plan time on any
    machine that only has an ed25519 key. Resolve one here instead; an explicit
    ``ssh_public_key`` in tfvars or ``TF_VAR_ssh_public_key`` always wins.

    ``allow_placeholder`` is for destroy: variable validation still runs, but the
    value is irrelevant to tearing resources down, and a teardown must not be
    blocked by a missing key on the machine doing the cleanup.
    """
    if "ssh_public_key" in tfvars or str(env.get("TF_VAR_ssh_public_key", "") or "").strip():
        return []
    explicit = os.environ.get("NPA_SSH_PUBLIC_KEY", "").strip()
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path.home() / ".ssh" / name for name in _SSH_PUBLIC_KEY_NAMES]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ["-var", f'ssh_public_key={{path="{candidate}"}}']
    if allow_placeholder:
        return ["-var", 'ssh_public_key={key="ssh-ed25519 AAAA npa-teardown-placeholder"}']
    searched = ", ".join(str(path) for path in candidates)
    raise typer.BadParameter(
        f"No SSH public key found for the cluster node groups (looked at {searched}). "
        "Create one with `ssh-keygen -t ed25519`, point NPA_SSH_PUBLIC_KEY at an "
        "existing key, or set ssh_public_key in terraform.tfvars."
    )


def _preflight_terraform_version(terraform_bin: str) -> None:
    """Fail early when the terraform binary is older than the vendored modules need."""
    result = _run_capture([terraform_bin, "version", "-json"], check=False)
    version = ""
    if result.returncode == 0:
        try:
            version = str(json.loads(result.stdout or "{}").get("terraform_version") or "")
        except json.JSONDecodeError:
            version = ""
    parsed = _parse_semver(version)
    if parsed is None:
        # Never block on an unparseable version; terraform itself will complain.
        return
    if parsed >= _MIN_TERRAFORM_VERSION:
        return
    minimum = ".".join(str(part) for part in _MIN_TERRAFORM_VERSION)
    raise typer.BadParameter(
        f"Terraform {version} is too old for deploy/cluster: it vendors modules that "
        f"require Terraform >= {minimum} (and use `ephemeral` blocks). "
        f"Install a newer Terraform (https://developer.hashicorp.com/terraform/install), "
        f"then re-run; point NPA_TERRAFORM_BIN at it to keep the old binary on PATH."
    )


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", str(version or "").strip())
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def _guard_tfvars_iam_token(terraform_dir: Path, tfvars: dict[str, Any]) -> None:
    """Reject an ``iam_token`` pinned in tfvars.

    Terraform gives ``terraform.tfvars`` precedence over ``TF_VAR_*``, so a token
    left in that file (the example file used to ship a placeholder) shadows the
    fresh token ``_terraform_env`` mints — apply then fails with Unauthenticated,
    or succeeds until the pasted token expires an hour later.
    """
    if "iam_token" not in tfvars:
        return
    files = ", ".join(
        str(path.name)
        for path in [terraform_dir / "terraform.tfvars", *sorted(terraform_dir.glob("*.auto.tfvars"))]
        if path.exists()
    )
    raise typer.BadParameter(
        f"Remove the `iam_token` line from {files or 'terraform.tfvars'} in {terraform_dir}: "
        "Terraform prefers tfvars over the fresh token npa mints for every run, so a "
        "pinned token (or the example's <nebius-iam-token> placeholder) breaks apply "
        "with Unauthenticated. npa supplies iam_token automatically."
    )


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
            f"Cluster {cluster_name} already exists outside this Terraform state: {ids}. "
            f"Adopt it with `npa cluster kubeconfig --cluster-name {cluster_name}` (writes "
            "its kubeconfig and cluster state), pick another `cluster_name`, or delete it "
            f"with `nebius mk8s cluster delete --id {unmanaged[0]}`."
        )


def _preflight_filestore_quota(nebius_bin: str, tfvars: dict[str, Any], env: dict[str, str]) -> None:
    # The Terraform default is enable_filestore = false (deploy/cluster/variables.tf),
    # so the default FTUE / PAIDF cluster needs no Shared Filesystem SSD quota. Only
    # check the quota when the shared filesystem is explicitly opted into and is being
    # created (not attached via existing_filestore).
    enable_filestore = _tfvar_bool(tfvars, env, "enable_filestore", False)
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


def _preflight_gpu_capacity(nebius_bin: str, tfvars: dict[str, Any], env: dict[str, str]) -> None:
    """Fail before apply when the tenant's GPU quota cannot cover the node group."""
    from npa.cli.cluster.capacity import gpu_capacity_error

    gpu_nodes = int(_tfvar_value(tfvars, env, "gpu_nodes_count", 0) or 0)
    if gpu_nodes <= 0:
        return
    platform = str(_tfvar_value(tfvars, env, "gpu_nodes_platform", "gpu-rtx6000") or "").strip()
    preset = str(_tfvar_value(tfvars, env, "gpu_nodes_preset", "") or "").strip()
    tenant_id = str(_tfvar_value(tfvars, env, "tenant_id", "") or "").strip()
    region = str(_tfvar_value(tfvars, env, "region", "") or "").strip()
    if not tenant_id or not region:
        typer.echo(
            "Skipping GPU quota preflight: tenant_id or region is not set in tfvars or env.",
            err=True,
        )
        return
    preemptible = _tfvar_bool(tfvars, env, "gpu_nodes_preemptible", False)
    required = gpu_nodes * _gpus_per_node(preset)
    capture = lambda args: _run_capture(args, env=env, check=False)  # noqa: E731 - passed through
    message = gpu_capacity_error(
        capture,
        nebius_bin=nebius_bin,
        tenant_id=tenant_id,
        region=region,
        platform=platform,
        preset=preset,
        required_gpus=required,
        preemptible=preemptible,
    )
    if message:
        raise typer.BadParameter(message)
    if not preemptible and not _gpu_quota_was_readable(
        capture, nebius_bin=nebius_bin, tenant_id=tenant_id, region=region, platform=platform
    ):
        # Skipping silently is what let an unreadable quota become a node group
        # that retries for hours with no explanation.
        typer.echo(
            f"Warning: could not read the {platform} GPU quota for {region} "
            f"(tenant {tenant_id}), so this apply is not quota-checked. If the node "
            "group never leaves PROVISIONING, the platform is refusing it — check "
            "`nebius quotas quota-allowance get-by-name --parent-id <tenant> "
            f"--region {region} --name compute.instance.gpu.<model>`.",
            err=True,
        )


def _gpu_quota_was_readable(
    capture: Any, *, nebius_bin: str, tenant_id: str, region: str, platform: str
) -> bool:
    from npa.cli.cluster.capacity import gpu_quota_headroom, gpu_quota_name

    quota_name = gpu_quota_name(platform)
    if not quota_name:
        return True
    return (
        gpu_quota_headroom(
            capture,
            nebius_bin=nebius_bin,
            tenant_id=tenant_id,
            region=region,
            quota_name=quota_name,
        )
        is not None
    )


#: Node-group status text that means the platform has *refused* the group rather
#: than being slow: waiting these out cannot succeed. Terraform keeps printing
#: `Still creating...` and retries until its own timeout (two hours by default),
#: which is what turned an unavailable GPU into an open-ended hang.
_TERMINAL_NODE_GROUP_MARKERS = (
    "quotafailure",
    "quota exceeded",
    "quota_exceeded",
    "exceeded quota",
    "out of capacity",
    "insufficient capacity",
    "no capacity",
    "capacity not available",
)


def terminal_node_group_failure(status: dict[str, Any]) -> str:
    """Return the refusal text when *status* shows a failure retrying cannot fix."""
    for key, value in (status or {}).items():
        if not value or not any(
            token in str(key).lower()
            for token in ("error", "failure", "message", "condition", "reason", "state")
        ):
            continue
        text = str(value)
        lowered = text.lower()
        for marker in _TERMINAL_NODE_GROUP_MARKERS:
            if marker in lowered:
                return text
    return ""


class _NodeGroupWatcher:
    """Watch Managed-Kubernetes node groups while ``terraform apply`` runs.

    Terraform's own output is just `Still creating...`, so a node group that
    Nebius refuses (QuotaFailure, no capacity) looks identical to one that is
    provisioning normally — the operator waits, then interrupts, and is left with
    a half-created cluster and no reason. This prints each group's state as it
    changes and, when the platform reports a refusal that retrying cannot fix,
    cancels the apply (``on_fatal``) instead of waiting out the timeout.

    Polling is best-effort: any error inside the thread stops the watcher rather
    than affecting the apply.
    """

    def __init__(
        self,
        nebius_bin: str,
        tfvars: dict[str, Any],
        env: dict[str, str],
        *,
        interval: float = 45.0,
        on_fatal: Callable[[str], None] | None = None,
    ):
        self._nebius_bin = nebius_bin
        self._project_id = str(_tfvar_value(tfvars, env, "parent_id", "") or "").strip()
        self._cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
        self._env = env
        self._interval = interval
        self._on_fatal = on_fatal
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: dict[str, str] = {}
        self.fatal_reason = ""

    def start(self) -> None:
        if not self._project_id:
            return
        self._thread = threading.Thread(target=self._run, name="npa-node-group-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._poll()
            except Exception:  # noqa: BLE001 - observability must never break apply
                return

    def _poll(self) -> None:
        cluster_id = self._cluster_id()
        if not cluster_id:
            return
        result = _run_capture(
            [self._nebius_bin, "mk8s", "node-group", "list", "--parent-id", cluster_id, "--format", "json"],
            env=self._env,
            check=False,
        )
        if result.returncode != 0:
            return
        for item in (json.loads(result.stdout or "{}") or {}).get("items", []):
            name = str((item.get("metadata") or {}).get("name", "") or "")
            status = item.get("status") or {}
            line = _format_node_group_status(status)
            if name and self._seen.get(name) != line:
                self._seen[name] = line
                typer.echo(f"  node group {name}: {line}", err=True)
            refusal = terminal_node_group_failure(status)
            if refusal and not self.fatal_reason:
                self.fatal_reason = f"node group {name}: {refusal}"
                typer.echo(
                    f"  Nebius refused node group {name}: {refusal}. Retrying cannot fix "
                    "this, so the apply is being cancelled instead of waiting for the "
                    "Terraform timeout.",
                    err=True,
                )
                if self._on_fatal is not None:
                    self._on_fatal(self.fatal_reason)
                self._stop.set()
                return

    def _cluster_id(self) -> str:
        result = _run_capture(
            [self._nebius_bin, "mk8s", "cluster", "list", "--parent-id", self._project_id, "--format", "json"],
            env=self._env,
            check=False,
        )
        if result.returncode != 0:
            return ""
        for item in (json.loads(result.stdout or "{}") or {}).get("items", []):
            metadata = item.get("metadata") or {}
            if str(metadata.get("name", "")) == self._cluster_name:
                return str(metadata.get("id", "") or "")
        return ""


def _format_node_group_status(status: dict[str, Any]) -> str:
    """Summarize node-group status, keeping any failure text Nebius reports."""
    state = str(status.get("state", "") or "UNKNOWN")
    counts = (
        f"{status.get('ready_node_count', '?')}/{status.get('target_node_count', '?')} ready"
    )
    # The failure shape is not part of the documented schema (QuotaFailure arrives
    # as a message/condition), so surface anything that looks like one verbatim.
    details = [
        f"{key}={value}"
        for key, value in sorted(status.items())
        if value
        and any(token in key.lower() for token in ("error", "failure", "message", "condition", "reason"))
    ]
    return ", ".join([f"{state} ({counts})", *details])


def _echo_apply_recovery(tf_dir: Path, tfvars: dict[str, Any], interrupted: bool) -> None:
    """Say what may exist in the cloud after a failed or interrupted apply.

    An interrupted `cluster up` leaves a real cluster running with no local
    kubeconfig, which reads as "nothing happened" until the bill arrives.
    """
    cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
    reason = "interrupted" if interrupted else "failed"
    typer.echo("", err=True)
    typer.echo(
        f"terraform apply was {reason}. Cluster {cluster_name!r} may exist (partially) "
        "in the project, and no kubeconfig was written yet. Either:",
        err=True,
    )
    typer.echo(
        f"  - resume: re-run `npa cluster up --terraform-dir {tf_dir}` (idempotent; it "
        "finishes what was created and writes the kubeconfig), or",
        err=True,
    )
    typer.echo(
        f"  - tear it down: `npa cluster down --terraform-dir {tf_dir} --force`, or",
        err=True,
    )
    typer.echo(
        f"  - adopt what exists: `npa cluster kubeconfig --cluster-name {cluster_name}` "
        "writes the kubeconfig and cluster state for a cluster that is already running.",
        err=True,
    )
    typer.echo(
        "  - check what exists now: `nebius mk8s cluster list --parent-id <project-id>`.",
        err=True,
    )


def _tfvar_value(tfvars: dict[str, Any], env: dict[str, str], key: str, default: Any) -> Any:
    if key in tfvars:
        return tfvars[key]
    return env.get(f"TF_VAR_{key}", default)


def _tfvar_bool(tfvars: dict[str, Any], env: dict[str, str], key: str, default: bool) -> bool:
    """Read a boolean tfvar, treating ``TF_VAR_x`` strings the way Terraform does.

    ``TF_VAR_*`` values arrive as strings, and ``bool("false")`` is ``True``, so a
    documented ``TF_VAR_enable_filestore=false`` opt-out would otherwise be read as
    enabled.
    """
    value = _tfvar_value(tfvars, env, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        return default
    return bool(value)


def _shared_filesystem_requested(tfvars: dict[str, Any], env: dict[str, str]) -> bool:
    """Whether the config asks for a shared filesystem (created or attached).

    ``deploy/cluster`` turns ``existing_filestore`` into ``enable_filestore`` for the
    vendored module, so either one means the filesystem CSI and its
    ``csi-mounted-fs-path-sc`` default StorageClass are installed.
    """
    if _tfvar_bool(tfvars, env, "enable_filestore", False):
        return True
    return bool(str(_tfvar_value(tfvars, env, "existing_filestore", "") or "").strip())


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
    # The filesystem CSI (and its `csi-mounted-fs-path-sc` default StorageClass) is
    # only installed when the shared filesystem is enabled. With the default
    # FTUE / PAIDF shape (enable_filestore = false), the platform block-storage
    # StorageClass stays the default, so only enforce the filesystem CSI SC when
    # the shared filesystem was opted into.
    if _shared_filesystem_requested(tfvars, os.environ) and default_sc != "csi-mounted-fs-path-sc":
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

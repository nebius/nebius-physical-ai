"""Backend-owned native Managed Kubernetes execution primitives."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


from npa.cluster.gpu_driver import resolve_gpu_driver_strategy
from npa.cluster.gpu_health import GpuHealthConfig, validate_gpu_health
from npa.cluster_backends.process import (
    _redact as _redact_output,
    require_bin as _require_bin,
    run_capture as _run_capture,
    run_stream as _run_stream,
    terraform_plugin_cache_lock,
    terraform_env as _terraform_env,
)
from npa.cluster_backends.mk8s_model import (
    MK8sDesired,
    MK8sExecutionScope,
    MK8sNodePool,
    MK8sProjectIdentity,
)
from npa.cluster_backends.mig import wait_for_mig_ready
from npa.cluster_backends.mk8s_render import (
    gpu_node_group_layout,
    patch_provider_domain,
    provider_domain,
    render_tfvars,
)

logger = logging.getLogger(__name__)
_K8S_TRAINING_SUBDIR = "k8s-training"
_MODULES_SUBDIR = "modules"
_FILESYSTEM_VERIFIER = (
    Path("filesystem-csi-validation") / "01-verify-node-filesystem-mounts.sh"
)
_FILESYSTEM_VALIDATION_COMMON = Path("filesystem-csi-validation") / "common.sh"
_FILESYSTEM_SMOKE_MANIFEST = (
    Path("filesystem-csi-validation") / "manifests" / "01-csi-smoke-test.yaml"
)
_FILESYSTEM_STORAGE_CLASS = "csi-mounted-fs-path-sc"
_ENV_SIDECAR = ".npa-fleet-env.json"
_PROVIDER_FIELD_MISSING = object()


def verify_cluster(
    *,
    cluster: MK8sDesired,
    kubeconfig: Path,
    kubectl_bin: str = "",
    evidence_path: Path | None = None,
    on_status: Callable[[str], None] | None = None,
    run_capture: Callable[..., Any] = _run_capture,
    mig_verifier: Callable[..., Any] = wait_for_mig_ready,
    gpu_health_verifier: Callable[..., Any] = validate_gpu_health,
    validation_policy: str = "standalone-full",
    basic_validation_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Verify one mk8s target without routing back through a surface adapter."""

    if validation_policy not in {"fleet", "standalone-full", "skip"}:
        raise ValueError(f"unsupported mk8s validation policy {validation_policy!r}")
    if validation_policy == "skip":
        return {
            "backend": "mk8s",
            "cluster_name": cluster.name,
            "status": "validation-skipped",
            "verification": "skipped",
        }

    kubectl_bin = kubectl_bin or _require_bin(
        os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
    )

    if cluster.mig and cluster.mig.enabled:
        report = mig_verifier(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            expected_nodes=cluster.gpu_count(),
            reconcile=True,
            timeout_seconds=cluster.gpu_health_timeout_minutes * 60,
            cuda_smoke_image=cluster.gpu_cuda_smoke_image,
            on_status=on_status,
        )
        result = {
            "backend": "mk8s",
            "cluster_name": cluster.name,
            "status": "verified",
            "verification": "mig",
            "verified_nodes": len(report.nodes),
            "reconciled": True,
            "cuda_smoke": True,
            "mig": report.as_dict(),
        }
    elif cluster.gpu_count() > 0:
        gpu = cluster.gpu_nodes
        driver = resolve_gpu_driver_strategy(
            gpu_nodes=cluster.gpu_count(),
            platform=gpu.platform if gpu else "",
            preset=gpu.preset if gpu else "",
            mode=cluster.resolved_gpu_driver_mode(),
            managed_driver_preset=cluster.managed_driver_preset,
            enable_gpu_cluster=cluster.resolved_enable_gpu_cluster(),
            allow_unsafe_nvswitch_operator=cluster.allow_unsafe_nvswitch_operator,
        )
        health = gpu_health_verifier(
            run_capture,
            kubectl_bin=kubectl_bin,
            kubeconfig_path=kubeconfig,
            config=GpuHealthConfig(
                expected_nodes=cluster.cpu_count() + cluster.gpu_count(),
                expected_gpu_nodes=cluster.gpu_count(),
                gpu_preset=gpu.preset if gpu else "",
                gpu_platform=gpu.platform if gpu else "",
                driver_mode=driver.effective_mode,
                nvswitch=driver.nvswitch,
                stabilization_seconds=cluster.gpu_health_stabilization_seconds,
                timeout_seconds=cluster.gpu_health_timeout_minutes * 60,
                cuda_smoke=cluster.gpu_cuda_smoke,
                cuda_smoke_image=cluster.gpu_cuda_smoke_image,
                graphics_smoke=cluster.gpu_graphics_smoke,
                graphics_smoke_image=cluster.gpu_graphics_smoke_image,
            ),
            evidence_path=evidence_path,
            on_status=on_status,
        )
        result = {
            "backend": "mk8s",
            "cluster_name": cluster.name,
            "status": "verified",
            "verification": "gpu-health",
            "gpu_health": health,
        }
    else:
        result = {
            "backend": "mk8s",
            "cluster_name": cluster.name,
            "status": "verified",
            "verification": "control-plane",
        }
    if validation_policy == "standalone-full":
        result["cluster_basics"] = _wait_for_cluster_basics(
            cluster=cluster,
            kubeconfig=kubeconfig,
            kubectl_bin=kubectl_bin,
            run_capture=run_capture,
            timeout_seconds=basic_validation_timeout_seconds,
            on_status=on_status,
        )
    return result


def _validate_cluster_basics_once(
    *,
    cluster: MK8sDesired,
    kubeconfig: Path,
    kubectl_bin: str,
    run_capture: Callable[..., Any],
) -> dict[str, Any]:
    """Prove exact Ready-node count and the recipe's default StorageClass."""

    kube_env = os.environ.copy()
    kube_env["KUBECONFIG"] = str(kubeconfig)
    nodes_result = run_capture(
        [kubectl_bin, "get", "nodes", "-o", "json"], env=kube_env, check=False
    )
    if nodes_result.returncode != 0 or not nodes_result.stdout.strip():
        raise RuntimeError("Kubernetes node inventory is unreadable")
    try:
        nodes_payload = json.loads(nodes_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Kubernetes node inventory returned invalid JSON") from exc
    nodes = nodes_payload.get("items") if isinstance(nodes_payload, dict) else None
    if not isinstance(nodes, list):
        raise RuntimeError("Kubernetes node inventory has no valid items list")
    ready_nodes = sum(
        1
        for node in nodes
        if isinstance(node, dict)
        and any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in ((node.get("status") or {}).get("conditions") or [])
            if isinstance(condition, dict)
        )
    )
    expected_nodes = cluster.cpu_count() + cluster.gpu_count()
    if len(nodes) != expected_nodes or ready_nodes != expected_nodes:
        raise RuntimeError(
            f"expected exactly {expected_nodes} Ready nodes, found "
            f"{ready_nodes}/{len(nodes)} Ready"
        )

    storage_result = run_capture(
        [kubectl_bin, "get", "storageclass", "-o", "json"],
        env=kube_env,
        check=False,
    )
    if storage_result.returncode != 0 or not storage_result.stdout.strip():
        raise RuntimeError("Kubernetes StorageClass inventory is unreadable")
    try:
        storage_payload = json.loads(storage_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Kubernetes StorageClass inventory returned invalid JSON"
        ) from exc
    items = storage_payload.get("items") if isinstance(storage_payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Kubernetes StorageClass inventory has no valid items list")
    defaults = [
        str((item.get("metadata") or {}).get("name") or "")
        for item in items
        if isinstance(item, dict)
        and str(
            ((item.get("metadata") or {}).get("annotations") or {}).get(
                "storageclass.kubernetes.io/is-default-class"
            )
            or ""
        ).lower()
        == "true"
    ]
    expected_storage = (
        "csi-mounted-fs-path-sc"
        if cluster.enable_filestore or bool(cluster.existing_filestore)
        else "compute-csi-default-sc"
    )
    if defaults != [expected_storage]:
        rendered = ", ".join(defaults) if defaults else "none"
        raise RuntimeError(
            f"expected default StorageClass {expected_storage}, found {rendered}"
        )
    return {
        "ready_nodes": ready_nodes,
        "expected_nodes": expected_nodes,
        "default_storage_class": expected_storage,
    }


def _wait_for_cluster_basics(
    *,
    cluster: MK8sDesired,
    kubeconfig: Path,
    kubectl_bin: str,
    run_capture: Callable[..., Any],
    timeout_seconds: int,
    on_status: Callable[[str], None] | None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("mk8s basic validation timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_error = "validation did not run"
    while True:
        try:
            return _validate_cluster_basics_once(
                cluster=cluster,
                kubeconfig=kubeconfig,
                kubectl_bin=kubectl_bin,
                run_capture=run_capture,
            )
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "cluster node/StorageClass validation did not converge: "
                    + last_error
                ) from exc
            _log(on_status, f"validation pending: {last_error}")
            time.sleep(min(30, max(1, timeout_seconds)))


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _nebius_argv(nebius_bin: str, profile: str = "") -> list[str]:
    """Base argv for a ``nebius`` CLI call, pinned to *profile* when given.

    A Nebius service account belongs to exactly one tenant, so a fleet targeting
    another tenant must authenticate as that tenant's profile. Passing
    ``--profile`` per call keeps the machine's active profile untouched (and
    keeps concurrent runs against different tenants independent).
    """

    return [nebius_bin, "--profile", profile] if profile else [nebius_bin]


def _get_project(
    nebius_bin: str, project_id: str, env: dict[str, str], profile: str = ""
) -> dict[str, Any]:
    """Return one exact provider project without exposing provider output."""

    result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "iam",
            "project",
            "get",
            "--id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not verify existing project (nebius exited {result.returncode})"
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("could not parse existing project JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("existing project JSON is not an object")
    return payload


def _provider_field(payload: object, *spellings: str) -> object:
    """Read one provider field defensively across wire-format spellings.

    Multiple spellings are accepted only when their values agree. Missing or
    contradictory evidence stays unknown so identity/shape checks fail closed.
    """

    if not isinstance(payload, dict):
        return _PROVIDER_FIELD_MISSING
    values = [payload[key] for key in spellings if key in payload]
    if not values or any(value != values[0] for value in values[1:]):
        return _PROVIDER_FIELD_MISSING
    return values[0]


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Atomically write non-secret local recovery metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "could not load fleet recovery metadata %s (%s)", path, type(exc).__name__
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "ignoring fleet recovery metadata %s because it is not a mapping", path
        )
        return {}
    return data


def _write_env_sidecar(install_dir: Path, data: dict[str, Any]) -> None:
    _write_json_file(install_dir / _ENV_SIDECAR, data)


def _persist_npa_cluster_identity(
    *,
    context: str,
    cluster_id: str,
    project_id: str,
    region: str,
    cluster: MK8sDesired,
    subnet_id: str,
    kubeconfig_path: Path,
    fleet_name: str,
    project_key: str,
    endpoint: str = "",
    node_group_id: str = "",
    base_dir: Path | None = None,
) -> None:
    """Register a fleet cluster for project-scoped workflow/controller use."""

    from npa.cluster.state import (
        ClusterState,
        kubeconfig_file,
        load_cluster_state,
        save_cluster_state,
        utc_now_iso,
    )

    existing = load_cluster_state(context, base_dir=base_dir)
    if existing is not None and (
        existing.cluster_id != cluster_id or existing.project_id != project_id
    ):
        raise RuntimeError(
            f"NPA cluster context {context!r} already records a different immutable "
            "project/cluster identity; refusing to overwrite it"
        )
    if not kubeconfig_path.is_file():
        raise RuntimeError(
            "Fleet kubeconfig is unavailable; refusing to register incomplete "
            "NPA cluster identity"
        )
    installed_kubeconfig = kubeconfig_file(context, base_dir=base_dir)
    installed_kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    if kubeconfig_path.resolve() != installed_kubeconfig.resolve():
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=installed_kubeconfig.parent,
                prefix=f".{installed_kubeconfig.name}.",
                delete=False,
            ) as handle:
                with kubeconfig_path.open("rb") as source:
                    shutil.copyfileobj(source, handle)
                temporary = Path(handle.name)
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
            os.replace(temporary, installed_kubeconfig)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        installed_kubeconfig.chmod(stat.S_IRUSR | stat.S_IWUSR)
    primary_pool = (
        cluster.gpu_nodes
        if cluster.gpu_nodes and cluster.gpu_nodes.count > 0
        else cluster.cpu_nodes
    )
    state = ClusterState(
        name=context,
        cluster_id=cluster_id,
        project_id=project_id,
        region=region,
        node_count=cluster.cpu_count() + cluster.gpu_count(),
        node_platform=str(primary_pool.platform if primary_pool else ""),
        node_preset=str(primary_pool.preset if primary_pool else ""),
        k8s_version=cluster.k8s_version,
        subnet_id=subnet_id,
        created_at=existing.created_at if existing else utc_now_iso(),
        last_seen_state="RUNNING",
        last_seen_at=utc_now_iso(),
        node_group_id=node_group_id or (existing.node_group_id if existing else ""),
        endpoint=endpoint or (existing.endpoint if existing else ""),
        kubeconfig_path=str(installed_kubeconfig),
        provider_name=cluster.name,
    )
    save_cluster_state(
        state,
        base_dir=base_dir,
        metadata={
            "managed_by": "npa fleet",
            "fleet": fleet_name,
            "project_key": project_key,
            "event": "kubeconfig_written",
            "updated_at": utc_now_iso(),
            "teardown": (
                "Run `npa fleet destroy --spec <fleet-spec.yaml> "
                f"--only-projects {project_key} --only-clusters {cluster.name} --yes`."
            ),
        },
    )


def _remove_npa_cluster_identity(
    *,
    context: str,
    cluster_id: str,
    project_id: str,
    base_dir: Path | None = None,
) -> None:
    """Remove only the exact fleet identity after authoritative cloud teardown."""

    if not context:
        return
    from npa.cluster.state import delete_cluster_state, load_cluster_state

    existing = load_cluster_state(context, base_dir=base_dir)
    if existing is None:
        return
    if existing.cluster_id != cluster_id or existing.project_id != project_id:
        raise RuntimeError(
            f"NPA cluster context {context!r} no longer matches the fleet's immutable "
            "project/cluster identity; local identity was preserved"
        )
    delete_cluster_state(context, base_dir=base_dir)


def _load_env_sidecar(install_dir: Path) -> dict[str, str] | None:
    path = install_dir / _ENV_SIDECAR
    if not path.exists():
        return None
    data = _load_json_file(path)
    return data or None


def _prepare_install_dir(
    install_dir: Path,
    *,
    recipe_root: Path,
    region: str,
    cluster: MK8sDesired,
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

    # kubectl 1.36's `debug --quiet` suppresses both attached verifier output and
    # the generated debugger-pod name. That defeats success-evidence checking and
    # cleanup. Apply this compatibility shim to the materialized recipe so local,
    # ref-cloned, package-fallback, and vendored sources all receive the fix.
    verifier = workdir / _FILESYSTEM_VERIFIER
    if verifier.is_file():
        original = verifier.read_text()
        patched = re.sub(r"(?m)^[ \t]*--quiet[ \t]*\\\n", "", original)
        patched = patched.replace("kubectl debug --quiet ", "kubectl debug ")
        success_check = (
            '    if ! grep -Fq \\\n'
            '      "[result] PASS: shared filesystem host mount is writable '
            'virtiofs with reboot-safe nofail at ${MOUNT_POINT} on this node" \\\n'
            '      "${output_file}"; then'
        )
        if success_check in patched and "waiting for detached debugger evidence" not in patched:
            # kubectl 1.36 can win the pod-creation request but lose the
            # immediate attach race ("container debugger not found") and still
            # return success. Wait for the one-shot pod to terminate and append
            # its logs before deciding that the required marker is absent.
            fallback = (
                '    debug_pod_name="$(awk \'/Creating debugging pod / '
                '{ print $4 }\' "${output_file}" | tail -n 1)"\n'
                '    if ! grep -Fq "[result] PASS: shared filesystem host mount is '
                'writable virtiofs with reboot-safe nofail at ${MOUNT_POINT} on this '
                'node" "${output_file}" && [[ -n "${debug_pod_name}" ]]; then\n'
                '      echo "[check] waiting for detached debugger evidence" | '
                'tee -a "${output_file}"\n'
                '      kubectl wait -n "${TEST_NAMESPACE}" '
                '--for=jsonpath=\'{.status.phase}\'=Succeeded '
                '"pod/${debug_pod_name}" >/dev/null 2>&1 || true\n'
                '      kubectl logs -n "${TEST_NAMESPACE}" "${debug_pod_name}" '
                '| tee -a "${output_file}" || true\n'
                '    fi\n'
            )
            patched = patched.replace(success_check, fallback + success_check)
        mount_tag_marker = 'MOUNT_TAG="${MOUNT_TAG:-data}"'
        mount_tag_replacement = (
            'if [[ -z "${MOUNT_TAG:-}" ]]; then\n'
            f"  MOUNT_TAG={shlex.quote(cluster.filestore_mount_tag)}\n"
            "fi"
        )
        patched = patched.replace(mount_tag_marker, mount_tag_replacement)
        if patched != original:
            verifier.write_text(patched)
            _log(on_status, "patched filesystem verifier for kubectl debug compatibility")

    # The upstream verifier tries to infer the mount path by parsing the
    # cloud-init Terraform template. That template intentionally contains the
    # unresolved token ``${filestore_mount_path}``, so the parser can return the
    # token literally and make every real mount check fail before reaching
    # fstab, df, or the write probe. Bind the materialized validation scripts to
    # the already-validated fleet value while preserving an explicit runtime
    # MOUNT_POINT override.
    validation_common = workdir / _FILESYSTEM_VALIDATION_COMMON
    if validation_common.is_file():
        original = validation_common.read_text()
        marker = 'MOUNT_POINT="${MOUNT_POINT:-$(default_mount_point)}"'
        replacement = (
            'if [[ -z "${MOUNT_POINT:-}" ]]; then\n'
            f"  MOUNT_POINT={shlex.quote(cluster.filestore_mount_path)}\n"
            "fi"
        )
        patched = original.replace(marker, replacement)
        if patched != original:
            validation_common.write_text(patched)
            _log(on_status, "bound filesystem verifier to rendered mount path")

    # Some managed control planes do not admission-default a classless PVC even
    # while this recipe's filesystem class is correctly annotated as the sole
    # default. A smoke probe must exercise the intended CSI driver rather than
    # wait forever on a classless claim. The validation script separately
    # asserts the bound PVC uses this exact class; cluster-basics validation
    # proves the same class carries the default annotation.
    smoke_manifest = workdir / _FILESYSTEM_SMOKE_MANIFEST
    if smoke_manifest.is_file():
        original = smoke_manifest.read_text()
        marker = "spec:\n  accessModes:\n"
        replacement = (
            f"spec:\n  storageClassName: {_FILESYSTEM_STORAGE_CLASS}\n  accessModes:\n"
        )
        patched = original.replace(marker, replacement, 1)
        if patched != original:
            smoke_manifest.write_text(patched)
            _log(on_status, "pinned filesystem smoke PVC to the verified CSI class")

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
        render_tfvars(cluster, ssh_public_key=ssh_public_key, recipe_dir=workdir)
    )
    return workdir


def _ensure_private_directory(path: Path) -> None:
    """Create/open one directory without following a final symlink and set 0700."""

    try:
        path.mkdir(mode=0o700, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"could not securely prepare Terraform diagnostics directory {path}: "
            f"{type(exc).__name__}"
        ) from exc
    try:
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _open_private_log(log_path: Path):
    """Open an append-only 0600 regular file without following a final symlink."""

    _ensure_private_directory(log_path.parent)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
    )
    try:
        parent_fd = os.open(log_path.parent, directory_flags)
        try:
            fd = os.open(log_path.name, flags, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise RuntimeError(
            f"could not securely open Terraform diagnostics log {log_path}: "
            f"{type(exc).__name__}"
        ) from exc
    try:
        target_stat = os.fstat(fd)
        if not stat.S_ISREG(target_stat.st_mode):
            raise RuntimeError(
                f"Terraform diagnostics log target is not a regular file: {log_path}"
            )
        if target_stat.st_nlink != 1:
            raise RuntimeError(
                f"Terraform diagnostics log target has multiple hard links: {log_path}"
            )
        # A pre-existing file may have been created under a permissive umask.
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


def _ensure_private_log_parent(log_path: Path, fleet_root: Path) -> None:
    """Prepare every diagnostics directory below *fleet_root* as 0700."""

    try:
        relative_parent = log_path.parent.relative_to(fleet_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Terraform diagnostics log escapes the fleet run directory: {log_path}"
        ) from exc
    _ensure_private_directory(fleet_root)
    current = fleet_root
    for part in relative_parent.parts:
        current /= part
        _ensure_private_directory(current)


def _run_to_log(
    args: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> None:
    """Run *args* through the shared redacting process layer into a private log."""

    with _open_private_log(log_path) as fh:
        fh.write(f"\n$ {_redact_output(' '.join(args))}\n")
        fh.flush()
        try:
            _run_stream(
                args,
                cwd=cwd,
                env=env,
                timeout=timeout,
                output_sink=fh,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (see {log_path})") from exc


def _tf_run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    log_path: Path | None,
) -> None:
    """terraform runner: stream to stdout (sequential) or to a per-cluster log."""

    if log_path is not None:
        _run_to_log(args, cwd=cwd, env=env, timeout=timeout, log_path=log_path)
    else:
        _run_stream(args, cwd=cwd, env=env, timeout=timeout)


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


def _terraform_outputs(
    terraform_bin: str, install_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
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


def _cluster_endpoint_from_outputs(outputs: dict[str, Any]) -> str:
    value = outputs.get("kube_cluster", {}).get("value")
    if not isinstance(value, dict):
        return ""
    endpoints = value.get("endpoints")
    return (
        str(endpoints.get("public_endpoint") or "")
        if isinstance(endpoints, dict)
        else ""
    )


def _terraform_managed_ids(
    terraform_bin: str,
    install_dir: Path,
    env: dict[str, str],
    resource_type: str,
) -> list[str]:
    """Read exact managed IDs of one type from the just-applied state."""

    pulled = _run_capture(
        [terraform_bin, "state", "pull"],
        cwd=install_dir,
        env=env,
        check=False,
    )
    if pulled.returncode != 0 or not pulled.stdout.strip():
        raise RuntimeError("applied Terraform state could not be read")
    try:
        state = json.loads(pulled.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("applied Terraform state returned invalid JSON") from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise RuntimeError("applied Terraform state has no valid resource inventory")
    ids: set[str] = set()
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("mode", "managed") != "managed"
        ):
            continue
        if str(resource.get("type") or "") != resource_type:
            continue
        instances = resource.get("instances")
        if not isinstance(instances, list):
            raise RuntimeError(f"applied Terraform {resource_type} state is malformed")
        for instance in instances:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            resource_id = (
                str(attributes.get("id") or "") if isinstance(attributes, dict) else ""
            )
            if not resource_id:
                raise RuntimeError(
                    f"applied Terraform {resource_type} state is missing an exact ID"
                )
            ids.add(resource_id)
    return sorted(ids)


def _terraform_instance_address(
    resource: dict[str, Any], instance: dict[str, Any]
) -> str:
    """Return the exact Terraform address for one state instance."""

    prefix = f"{resource.get('module')}." if resource.get("module") else ""
    address = f"{prefix}{resource.get('type')}.{resource.get('name')}"
    if "index_key" in instance:
        address += f"[{json.dumps(instance['index_key'], separators=(',', ':'))}]"
    return address


def _node_group_template_fingerprint(template: dict[str, Any]) -> dict[str, Any]:
    """Select identity-bearing node-template fields, excluding computed values."""

    boot_disk = template.get("boot_disk") or {}
    reservation = template.get("reservation_policy") or {}
    gpu_settings = template.get("gpu_settings") or {}
    network_interfaces = template.get("network_interfaces") or []
    filesystems = template.get("filesystems") or []
    disk_size = boot_disk.get("size_gibibytes")
    try:
        disk_size = int(disk_size) if disk_size is not None else None
    except (TypeError, ValueError):
        pass
    return {
        "resources": template.get("resources") or {},
        "boot_disk": {
            "type": boot_disk.get("type"),
            "size_gibibytes": disk_size,
        },
        "reservation_policy": {
            "policy": reservation.get("policy"),
            "reservation_ids": reservation.get("reservation_ids") or [],
        },
        "preemptible": bool(template.get("preemptible", False)),
        "network_interfaces": [
            {"subnet_id": item.get("subnet_id")}
            for item in network_interfaces
            if isinstance(item, dict)
        ],
        "filesystems": [
            {
                "existing_filesystem": item.get("existing_filesystem"),
                "mount_tag": item.get("mount_tag"),
                "attach_mode": item.get("attach_mode"),
            }
            for item in filesystems
            if isinstance(item, dict)
        ],
        "gpu_cluster": template.get("gpu_cluster") or None,
        "gpu_settings": {"drivers_preset": gpu_settings.get("drivers_preset")},
    }


def _tainted_node_group_matches_desired(
    *,
    provider_payload: dict[str, Any],
    state_attributes: dict[str, Any],
    pool: MK8sNodePool,
    cluster: MK8sDesired,
    cluster_id: str,
    subnet_id: str,
    expected_node_count: int | None = None,
) -> bool:
    """Prove a tainted state entry still owns the exact desired live pool."""

    metadata = provider_payload.get("metadata") or {}
    spec = provider_payload.get("spec") or {}
    status = provider_payload.get("status") or {}
    template = spec.get("template") or {}
    state_template = state_attributes.get("template") or {}
    if not all(
        isinstance(value, dict)
        for value in (metadata, spec, status, template, state_template)
    ):
        return False
    parent_id = _provider_field(metadata, "parent_id", "parentId")
    if parent_id is _PROVIDER_FIELD_MISSING:
        return False
    try:
        fixed_node_count = int(spec.get("fixed_node_count"))
    except (TypeError, ValueError):
        return False
    if (
        str(metadata.get("id") or "") != str(state_attributes.get("id") or "")
        or str(parent_id) != cluster_id
        or str(metadata.get("name") or "") != str(state_attributes.get("name") or "")
        or str(status.get("state") or "") not in {"PROVISIONING", "RUNNING"}
        or fixed_node_count
        != (pool.count if expected_node_count is None else expected_node_count)
        or _node_group_template_fingerprint(template)
        != _node_group_template_fingerprint(state_template)
    ):
        return False

    resources = template.get("resources") or {}
    reservation = template.get("reservation_policy") or {}
    interfaces = template.get("network_interfaces") or []
    filesystems = template.get("filesystems") or []
    gpu_settings = template.get("gpu_settings") or {}
    expected_disk_size = (
        cluster.resolved_gpu_disk_size_gib()
        if pool.is_gpu()
        else (pool.disk_size_gib or 128)
    )
    try:
        boot_disk_size = int((template.get("boot_disk") or {}).get("size_gibibytes"))
    except (TypeError, ValueError):
        return False
    if (
        resources.get("platform") != pool.platform
        or resources.get("preset") != pool.preset
        or boot_disk_size != expected_disk_size
        or bool(template.get("preemptible", False)) != pool.preemptible
        or len(interfaces) != 1
        or not isinstance(interfaces[0], dict)
        or interfaces[0].get("subnet_id") != subnet_id
    ):
        return False
    if pool.capacity_block_group:
        if (
            reservation.get("policy") != "STRICT"
            or reservation.get("reservation_ids") != [pool.capacity_block_group]
        ):
            return False
    elif reservation.get("policy") or reservation.get("reservation_ids"):
        return False
    if cluster.enable_filestore:
        if (
            len(filesystems) != 1
            or not isinstance(filesystems[0], dict)
            or filesystems[0].get("mount_tag") != cluster.filestore_mount_tag
            or filesystems[0].get("attach_mode") != "READ_WRITE"
            or not filesystems[0].get("existing_filesystem")
        ):
            return False
    elif filesystems:
        return False
    if bool(template.get("gpu_cluster")) != cluster.resolved_enable_gpu_cluster():
        return False
    if pool.is_gpu():
        driver = resolve_gpu_driver_strategy(
            gpu_nodes=cluster.gpu_count(),
            platform=pool.platform,
            preset=pool.preset,
            mode=cluster.resolved_gpu_driver_mode(),
            managed_driver_preset=cluster.managed_driver_preset,
            enable_gpu_cluster=cluster.resolved_enable_gpu_cluster(),
            allow_unsafe_nvswitch_operator=cluster.allow_unsafe_nvswitch_operator,
        )
        expected_driver = (
            driver.managed_driver_preset if driver.uses_managed_image else None
        )
        if gpu_settings.get("drivers_preset") != expected_driver:
            return False
    return True


def _reconcile_tainted_node_groups(
    *,
    terraform_bin: str,
    workdir: Path,
    env: dict[str, str],
    cluster: MK8sDesired,
    project_id: str = "",
    subnet_id: str,
    nebius_bin: str,
    profile: str,
    on_status: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Safely untaint exact live node groups after an interrupted apply."""

    if not workdir.is_dir():
        return {}
    pulled = _run_capture(
        [terraform_bin, "state", "pull"], cwd=workdir, env=env, check=False
    )
    local_state = workdir / "terraform.tfstate"
    if pulled.returncode != 0:
        stderr = str(getattr(pulled, "stderr", "") or "")
        if "No state file was found" in stderr and not local_state.exists():
            return {}
        raise RuntimeError(
            "Terraform state could not be audited before node-group reconciliation"
        )
    if not pulled.stdout.strip():
        if not local_state.exists():
            return {}
        raise RuntimeError(
            "Terraform state was empty during node-group reconciliation"
        )
    try:
        state = json.loads(pulled.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Terraform state is unreadable during taint audit") from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise RuntimeError("Terraform state has no valid resource inventory")

    cluster_ids = {
        str(instance.get("attributes", {}).get("id") or "")
        for resource in resources
        if isinstance(resource, dict)
        and resource.get("mode", "managed") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_cluster"
        for instance in resource.get("instances", [])
        if isinstance(instance, dict)
    }
    cluster_ids.discard("")
    tainted: list[tuple[str, dict[str, Any], MK8sNodePool]] = []
    pools = {"cpu-only": cluster.cpu_nodes, "gpu": cluster.gpu_nodes}
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("mode", "managed") != "managed"
            or resource.get("type") != "nebius_mk8s_v1_node_group"
        ):
            continue
        pool = pools.get(str(resource.get("name") or ""))
        for instance in resource.get("instances", []):
            if not isinstance(instance, dict) or instance.get("status") != "tainted":
                continue
            if pool is None or pool.count <= 0:
                raise RuntimeError(
                    "refusing to reconcile a tainted node group with no exact desired pool"
                )
            attributes = instance.get("attributes")
            if not isinstance(attributes, dict) or not attributes.get("id"):
                raise RuntimeError(
                    "refusing to reconcile a tainted node group without exact state identity"
                )
            tainted.append((_terraform_instance_address(resource, instance), attributes, pool))
    if not tainted:
        return {}
    if len(cluster_ids) != 1:
        raise RuntimeError(
            "refusing to reconcile tainted node groups without one exact managed cluster"
        )
    cluster_id = next(iter(cluster_ids))
    gpu_state_instance_count = sum(
        len(resource.get("instances", []))
        for resource in resources
        if isinstance(resource, dict)
        and resource.get("mode", "managed") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_node_group"
        and resource.get("name") == "gpu"
    )
    cli_env = env.copy()
    gpu_nodes_per_group, _gpu_group_count = gpu_node_group_layout(cluster)
    adopted_ids: list[str] = []
    for address, attributes, pool in tainted:
        result = _run_capture(
            [
                *_nebius_argv(nebius_bin, profile),
                "mk8s",
                "node-group",
                "get",
                "--id",
                str(attributes["id"]),
                "--format",
                "json",
            ],
            env=cli_env,
            check=False,
        )
        expected_node_count = (
            pool.count
            if pool.is_gpu() and gpu_state_instance_count == 1
            else (gpu_nodes_per_group if pool.is_gpu() else pool.count)
        )
        if result.returncode != 0 and _is_not_found_result(result):
            state_only_payload = {
                "metadata": {
                    "id": attributes.get("id"),
                    "name": attributes.get("name"),
                    "parent_id": cluster_id,
                },
                "spec": {
                    "fixed_node_count": attributes.get("fixed_node_count"),
                    "template": attributes.get("template"),
                },
                "status": {"state": "PROVISIONING"},
            }
            if not _tainted_node_group_matches_desired(
                provider_payload=state_only_payload,
                state_attributes=attributes,
                pool=pool,
                cluster=cluster,
                cluster_id=cluster_id,
                subnet_id=subnet_id,
                expected_node_count=expected_node_count,
            ):
                raise RuntimeError(
                    "refusing to retain an absent node-group taint because Terraform "
                    "state does not match the exact desired topology"
                )
            _log(
                on_status,
                f"retained provider-confirmed absent node-group taint at {address} "
                "for replacement",
            )
            continue
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "refusing to reconcile a tainted node group with unreadable provider state"
            ) from exc
        if result.returncode != 0 or not isinstance(payload, dict) or not (
            _tainted_node_group_matches_desired(
                provider_payload=payload,
                state_attributes=attributes,
                pool=pool,
                cluster=cluster,
                cluster_id=cluster_id,
                subnet_id=subnet_id,
                expected_node_count=expected_node_count,
            )
        ):
            raise RuntimeError(
                "refusing to reconcile a tainted node group because live identity or "
                "desired topology does not match exact Terraform state"
            )
        status = payload.get("status") or {}
        ready = status.get("ready_node_count")
        target = status.get("target_node_count")
        if (
            pool.is_gpu()
            and expected_node_count == 1
            and status.get("state") == "PROVISIONING"
            and str(target) == "1"
            and ready in (None, 0, "0")
        ):
            if not project_id:
                raise RuntimeError(
                    "refusing to retain a failed split node-group taint without "
                    "exact project identity"
                )
            inventory = _run_capture(
                [
                    *_nebius_argv(nebius_bin, profile),
                    "compute",
                    "instance",
                    "list",
                    "--parent-id",
                    project_id,
                    "--all",
                    "--format",
                    "json",
                ],
                env=cli_env,
                check=False,
            )
            try:
                instances = json.loads(inventory.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "refusing split node-group replacement with unreadable worker inventory"
                ) from exc
            workers = [
                item
                for item in instances.get("items", [])
                if isinstance(item, dict)
                and ((item.get("metadata") or {}).get("labels") or {}).get(
                    "mk8s-node-group-id"
                )
                == str(attributes["id"])
            ]
            failed_worker = len(workers) == 1 and (
                (workers[0].get("status") or {}).get("state") == "STOPPED"
                and not (workers[0].get("status") or {}).get("reservation_id")
                and not (workers[0].get("status") or {}).get("disk_attachments")
            )
            if inventory.returncode != 0 or (workers and not failed_worker):
                raise RuntimeError(
                    "refusing split node-group replacement without exact failed-worker evidence"
                )
            _log(
                on_status,
                f"retained exact failed one-node group taint at {address} for replacement",
            )
            continue
        untaint = _run_capture(
            [terraform_bin, "untaint", address],
            cwd=workdir,
            env=env,
            check=False,
        )
        if untaint.returncode != 0:
            raise RuntimeError("Terraform could not safely clear an exact node-group taint")
        _log(on_status, f"reconciled exact tainted node-group state at {address}")
        adopted_ids.append(str(attributes["id"]))
    if not adopted_ids:
        return {}
    return {
        "cluster_id": cluster_id,
        "node_group_ids": sorted(adopted_ids),
    }


def _instance_matches_node_group(
    payload: dict[str, Any],
    *,
    project_id: str,
    cluster_id: str,
    node_group_id: str,
    template: dict[str, Any],
) -> bool:
    """Prove that one compute instance is owned by an exact node-group template."""

    metadata = payload.get("metadata") or {}
    spec = payload.get("spec") or {}
    labels = metadata.get("labels") or {}
    interfaces = spec.get("network_interfaces") or []
    expected_interfaces = template.get("network_interfaces") or []
    disk = ((spec.get("boot_disk") or {}).get("managed_disk") or {}).get("spec") or {}
    expected_disk = template.get("boot_disk") or {}
    return bool(
        isinstance(metadata, dict)
        and isinstance(spec, dict)
        and isinstance(labels, dict)
        and metadata.get("id")
        and _provider_field(metadata, "parent_id", "parentId") == project_id
        and labels.get("mk8s-cluster-id") == cluster_id
        and labels.get("mk8s-node-group-id") == node_group_id
        and spec.get("resources") == template.get("resources")
        and spec.get("reservation_policy") == template.get("reservation_policy")
        and not bool(spec.get("preemptible", False))
        and len(interfaces) == len(expected_interfaces) == 1
        and interfaces[0].get("subnet_id") == expected_interfaces[0].get("subnet_id")
        and spec.get("filesystems", []) == template.get("filesystems", [])
        and spec.get("gpu_cluster") == (template.get("gpu_cluster") or {})
        and disk.get("type") == expected_disk.get("type")
        and disk.get("size_gibibytes") == expected_disk.get("size_gibibytes")
    )


def _repair_exact_stopped_placeholder(
    *,
    terraform_bin: str,
    workdir: Path,
    env: dict[str, str],
    cluster: MK8sDesired,
    project_id: str,
    subnet_id: str,
    nebius_bin: str,
    profile: str,
    on_status: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Replace one exact stopped strict-reserved worker, only when opted in.

    Managed Kubernetes can retain an unbound, diskless STOPPED compute
    placeholder after a reservation scheduling timeout. Terraform sees the
    node group itself as unchanged and therefore cannot repair that element.
    This deliberately narrow recovery path proves state, ownership, topology,
    and the absence of an active node-group operation before deleting only the
    failed placeholder. It then uses resource-version guarded node-count
    updates to make the controller reconcile a replacement.
    """

    pool = cluster.gpu_nodes
    if (
        pool is None
        or pool.count != 2
        or not pool.capacity_block_group
        or pool.preemptible
    ):
        raise RuntimeError(
            "stopped-placeholder repair requires exactly two strict-reserved, "
            "non-preemptible GPU workers"
        )
    pulled = _run_capture(
        [terraform_bin, "state", "pull"], cwd=workdir, env=env, check=False
    )
    if pulled.returncode != 0 or not pulled.stdout.strip():
        raise RuntimeError("could not audit Terraform state before placeholder repair")
    try:
        state = json.loads(pulled.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Terraform state is unreadable before placeholder repair") from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise RuntimeError("Terraform state has no valid resource inventory")
    cluster_instances = [
        item
        for resource in resources
        if isinstance(resource, dict)
        and resource.get("mode", "managed") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_cluster"
        for item in resource.get("instances", [])
        if isinstance(item, dict) and isinstance(item.get("attributes"), dict)
    ]
    group_instances = [
        item
        for resource in resources
        if isinstance(resource, dict)
        and resource.get("mode", "managed") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_node_group"
        and resource.get("name") == "gpu"
        for item in resource.get("instances", [])
        if isinstance(item, dict) and isinstance(item.get("attributes"), dict)
    ]
    _nodes_per_group, expected_group_count = gpu_node_group_layout(cluster)
    if len(cluster_instances) != 1:
        raise RuntimeError(
            "placeholder repair requires one exact Terraform cluster"
        )
    cluster_id = str(cluster_instances[0]["attributes"].get("id") or "")
    if not cluster_id:
        raise RuntimeError("placeholder repair requires exact Terraform cluster identity")
    cli = _nebius_argv(nebius_bin, profile)

    def provider_json(args: list[str], message: str) -> dict[str, Any]:
        result = _run_capture([*cli, *args, "--format", "json"], env=env, check=False)
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(message) from exc
        if result.returncode != 0 or not isinstance(payload, dict):
            raise RuntimeError(message)
        return payload

    if len(cluster_instances) == 1 and len(group_instances) == expected_group_count > 1:
        # The split strict-reserved layout has no multi-node group element for
        # this legacy repair to touch. Explicit invocation stays idempotent.
        return {"status": "not-applicable-split-layout"}
    if (
        len(group_instances) == 1
        and expected_group_count == 2
        and str(group_instances[0]["attributes"].get("fixed_node_count")) == "1"
    ):
        # An interrupted create can leave the second provider group live after
        # Terraform has removed its state entry. Prove the exact recipe name,
        # template, zero-ready status, worker failure, and idle operation set
        # before deleting only that orphan so the next apply can recreate it.
        existing = group_instances[0]["attributes"]
        live_groups = provider_json(
            ["mk8s", "node-group", "list", "--parent-id", cluster_id, "--all"],
            "could not audit live node groups before split-orphan repair",
        )
        expected_name = f"{cluster.name}-ng-gpu-1"
        candidates = [
            item
            for item in live_groups.get("items", [])
            if isinstance(item, dict)
            and (item.get("metadata") or {}).get("name") == expected_name
            and (item.get("metadata") or {}).get("parent_id") == cluster_id
        ]
        if not candidates:
            _log(on_status, "split node group is absent from both state and provider")
            return {"status": "split-group-absent"}
        if len(candidates) != 1:
            raise RuntimeError(
                "refusing split-orphan repair without one exact provider node group"
            )
        orphan = candidates[0]
        orphan_id = str((orphan.get("metadata") or {}).get("id") or "")
        orphan_state = {
            **existing,
            "id": orphan_id,
            "name": expected_name,
            "fixed_node_count": 1,
        }
        orphan_status = orphan.get("status") or {}
        if not orphan_id or not _tainted_node_group_matches_desired(
            provider_payload=orphan,
            state_attributes=orphan_state,
            pool=pool,
            cluster=cluster,
            cluster_id=cluster_id,
            subnet_id=subnet_id,
            expected_node_count=1,
        ) or not (
            orphan_status.get("state") == "PROVISIONING"
            and str(orphan_status.get("target_node_count")) == "1"
            and orphan_status.get("ready_node_count") in (None, 0, "0")
        ):
            raise RuntimeError(
                "refusing split-orphan repair: provider group is not exact and zero-ready"
            )
        operations = provider_json(
            [
                "mk8s",
                "node-group",
                "operation",
                "list",
                "--resource-id",
                orphan_id,
                "--all",
            ],
            "could not audit split-orphan operations",
        )
        if operations.get("items"):
            raise RuntimeError("refusing split-orphan repair while an operation exists")
        inventory = provider_json(
            ["compute", "instance", "list", "--parent-id", project_id, "--all"],
            "could not audit split-orphan worker inventory",
        )
        workers = [
            item
            for item in inventory.get("items", [])
            if isinstance(item, dict)
            and ((item.get("metadata") or {}).get("labels") or {}).get(
                "mk8s-node-group-id"
            )
            == orphan_id
        ]
        failed_worker = len(workers) == 1 and (
            (workers[0].get("status") or {}).get("state") == "STOPPED"
            and not (workers[0].get("status") or {}).get("reservation_id")
            and not (workers[0].get("status") or {}).get("disk_attachments")
        )
        if workers and not failed_worker:
            raise RuntimeError(
                "refusing split-orphan repair without exact failed-worker evidence"
            )
        deleted = _run_capture(
            [
                *cli,
                "mk8s",
                "node-group",
                "delete",
                "--id",
                orphan_id,
                "--format",
                "json",
            ],
            env=env,
            check=False,
        )
        if deleted.returncode != 0:
            raise RuntimeError("provider could not delete the exact failed split orphan")
        _log(on_status, "removed one exact failed split node-group orphan")
        return {"status": "split-orphan-removed"}
    if len(cluster_instances) != 1 or len(group_instances) != 1:
        raise RuntimeError(
            "placeholder repair requires one exact Terraform cluster and GPU node group"
        )
    attributes = group_instances[0]["attributes"]
    node_group_id = str(attributes.get("id") or "")
    if not cluster_id or not node_group_id:
        raise RuntimeError("placeholder repair requires exact Terraform resource IDs")
    live_group = provider_json(
        ["mk8s", "node-group", "get", "--id", node_group_id],
        "could not read the exact node group before placeholder repair",
    )
    if not _tainted_node_group_matches_desired(
        provider_payload=live_group,
        state_attributes=attributes,
        pool=pool,
        cluster=cluster,
        cluster_id=cluster_id,
        subnet_id=subnet_id,
    ):
        raise RuntimeError("refusing placeholder repair: live node group does not match state")
    status = live_group.get("status") or {}
    if (
        status.get("state") != "PROVISIONING"
        or int(status.get("target_node_count", -1)) != 2
        or int(status.get("ready_node_count", -1)) != 1
    ):
        raise RuntimeError("refusing placeholder repair: node-group readiness is not 1 of 2")
    operations = provider_json(
        [
            "mk8s",
            "node-group",
            "operation",
            "list",
            "--resource-id",
            node_group_id,
            "--all",
        ],
        "could not audit node-group operations before placeholder repair",
    )
    if operations.get("items"):
        raise RuntimeError("refusing placeholder repair while a node-group operation exists")
    inventory = provider_json(
        ["compute", "instance", "list", "--parent-id", project_id, "--all"],
        "could not audit compute inventory before placeholder repair",
    )
    template = (live_group.get("spec") or {}).get("template") or {}
    workers = [
        item
        for item in inventory.get("items", [])
        if isinstance(item, dict)
        and ((item.get("metadata") or {}).get("labels") or {}).get(
            "mk8s-node-group-id"
        )
        == node_group_id
    ]
    if len(workers) != 2 or not all(
        _instance_matches_node_group(
            item,
            project_id=project_id,
            cluster_id=cluster_id,
            node_group_id=node_group_id,
            template=template,
        )
        for item in workers
    ):
        raise RuntimeError("refusing placeholder repair: worker inventory is not exact")
    running = [item for item in workers if (item.get("status") or {}).get("state") == "RUNNING"]
    stopped = [item for item in workers if (item.get("status") or {}).get("state") == "STOPPED"]
    if len(running) != 1 or len(stopped) != 1:
        raise RuntimeError("refusing placeholder repair: expected one RUNNING and one STOPPED worker")
    running_status = running[0].get("status") or {}
    stopped_status = stopped[0].get("status") or {}
    if (
        not running_status.get("reservation_id")
        or not running_status.get("disk_attachments")
        or stopped_status.get("reservation_id")
        or stopped_status.get("disk_attachments")
    ):
        raise RuntimeError("refusing placeholder repair: reservation or disk evidence is unsafe")
    failed_id = str((stopped[0].get("metadata") or {}).get("id") or "")
    refreshed = provider_json(
        ["compute", "instance", "get", "--id", failed_id],
        "could not refetch the stopped placeholder before deletion",
    )
    if refreshed != stopped[0]:
        raise RuntimeError("refusing placeholder repair: stopped placeholder changed before deletion")
    deleted = _run_capture(
        [*cli, "compute", "instance", "delete", "--id", failed_id, "--format", "json"],
        env=env,
        check=False,
    )
    if deleted.returncode != 0:
        raise RuntimeError("provider could not delete the exact stopped placeholder")
    after = provider_json(
        ["compute", "instance", "list", "--parent-id", project_id, "--all"],
        "could not verify compute inventory after placeholder deletion",
    )
    remaining = [
        item
        for item in after.get("items", [])
        if isinstance(item, dict)
        and ((item.get("metadata") or {}).get("labels") or {}).get(
            "mk8s-node-group-id"
        )
        == node_group_id
    ]
    running_id = str((running[0].get("metadata") or {}).get("id") or "")
    if len(remaining) == 2 and any(
        str((item.get("metadata") or {}).get("id") or "") == running_id
        for item in remaining
    ):
        _log(on_status, "controller created a replacement after exact placeholder deletion")
        return {"status": "replacement-created"}
    if len(remaining) != 1 or str(
        (remaining[0].get("metadata") or {}).get("id") or ""
    ) != running_id:
        raise RuntimeError("placeholder repair changed unexpected worker inventory")
    current = provider_json(
        ["mk8s", "node-group", "get", "--id", node_group_id],
        "could not refetch node group for guarded reconciliation",
    )
    resource_version = (current.get("metadata") or {}).get("resource_version")
    if not resource_version:
        raise RuntimeError("node group has no resource version for guarded reconciliation")
    updated = provider_json(
        [
            "mk8s",
            "node-group",
            "update",
            "--id",
            node_group_id,
            "--resource-version",
            str(resource_version),
            "--fixed-node-count",
            "1",
        ],
        "node-group guarded update to 1 failed during placeholder repair",
    )
    resource_version = (updated.get("metadata") or {}).get("resource_version")
    if not resource_version:
        raise RuntimeError("node-group update returned no revision for desired-count restore")
    # Restoring two is intentionally asynchronous. A synchronous provider call
    # waits for physical placement and can exit nonzero after the desired count
    # was already accepted, making a safe retry indistinguishable from a CAS
    # conflict. Terraform apply and the normal health gates own convergence.
    restored = _run_capture(
        [
            *cli,
            "mk8s",
            "node-group",
            "update",
            "--id",
            node_group_id,
            "--resource-version",
            str(resource_version),
            "--fixed-node-count",
            "2",
            "--async",
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if restored.returncode != 0:
        raise RuntimeError(
            "node-group guarded asynchronous restore to 2 failed during placeholder repair"
        )
    _log(on_status, "repaired one exact stopped strict-reserved worker placeholder")
    return {"status": "reconciled"}


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
    temporary = kubeconfig_path.with_name(f".{kubeconfig_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        command_env = env.copy()
        configured_kubectl = os.environ.get("NPA_KUBECTL_BIN", "").strip()
        with tempfile.TemporaryDirectory(prefix="npa-kube-bin-") as shim_dir:
            if configured_kubectl and not shutil.which(
                "kubectl", path=command_env.get("PATH", "")
            ):
                resolved_kubectl = (
                    shutil.which(configured_kubectl) or configured_kubectl
                )
                shim = Path(shim_dir) / "kubectl"
                shim.symlink_to(Path(resolved_kubectl).expanduser().resolve())
                command_env["PATH"] = os.pathsep.join(
                    (shim_dir, command_env.get("PATH", ""))
                )
            result = _run_capture(
                [
                    *_nebius_argv(nebius_bin, profile),
                    "mk8s",
                    "cluster",
                    "get-credentials",
                    "--id",
                    cluster_id,
                    "--external",
                    "--force",
                    "--kubeconfig",
                    str(temporary),
                    "--context-name",
                    context,
                ],
                env=command_env,
                check=False,
                timeout=180,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"credential generation failed (nebius exited {result.returncode})"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(
                "credential generation completed without a kubeconfig file"
            )
        os.replace(temporary, kubeconfig_path)
    finally:
        temporary.unlink(missing_ok=True)


def _context_name(fleet_name: str, project_key: str, cluster_name: str) -> str:
    return f"fleet-{fleet_name}-{project_key}-{cluster_name}"


def _normalize_tfvars_assignments(text: str) -> str:
    """Normalize only an HCL line's first unquoted assignment operator."""

    normalized: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        quote = ""
        escaped = False
        assignment = -1
        for index, character in enumerate(line):
            if escaped:
                escaped = False
                continue
            if character == "\\" and quote:
                escaped = True
                continue
            if quote:
                if character == quote:
                    quote = ""
                continue
            if character in {'"', "'"}:
                quote = character
                continue
            if character == "=":
                assignment = index
                break
        if assignment < 0:
            normalized.append(line)
        else:
            normalized.append(
                line[:assignment].rstrip() + "=" + line[assignment + 1 :].lstrip()
            )
    return "\n".join(normalized)


def _is_verified_unchanged_target(
    *,
    project: MK8sProjectIdentity,
    cluster: MK8sDesired,
    prefix: str,
    tenant_id: str,
    region: str,
    ssh_public_key: str,
    fleet_root: Path,
    nebius_bin: str,
    profile: str,
    env: dict[str, str],
) -> bool:
    """Prove that a target consumes no *additional* quota on this deploy."""

    install_dir = fleet_root / project.key() / cluster.name
    saved = _load_env_sidecar(install_dir) or {}
    project_id = str(saved.get("project_id") or "")
    cluster_id = str(saved.get("cluster_id") or "")
    if (
        str(saved.get("status") or "") != "deployed"
        or not project_id
        or not cluster_id
        or str(saved.get("tenant_id") or "") != tenant_id
        or str(saved.get("region") or "") != region
        or str(saved.get("cluster_name") or "") != cluster.name
    ):
        return False
    if project.project_id and project.project_id != project_id:
        return False
    tfvars_path = install_dir / _K8S_TRAINING_SUBDIR / "terraform.tfvars"
    try:
        saved_tfvars = tfvars_path.read_text(encoding="utf-8")
        rendered_tfvars = render_tfvars(cluster, ssh_public_key=ssh_public_key)

        if _normalize_tfvars_assignments(saved_tfvars) != _normalize_tfvars_assignments(
            rendered_tfvars
        ):
            return False
        provider_project = _get_project(nebius_bin, project_id, env, profile)
    except (OSError, RuntimeError, ValueError):
        return False
    metadata = provider_project.get("metadata", {}) or {}
    project_spec = provider_project.get("spec", {}) or {}
    project_status = provider_project.get("status", {}) or {}
    provider_region = str(
        project_spec.get("region")
        or project_status.get("region")
        or metadata.get("region")
        or ""
    )
    expected_name = project.display_name(prefix) if project.name else ""
    provider_parent_id = _provider_field(metadata, "parent_id", "parentId")
    if (
        str(metadata.get("id") or "") != project_id
        or str(
            provider_parent_id
            if provider_parent_id is not _PROVIDER_FIELD_MISSING
            else ""
        )
        != tenant_id
        or provider_region != region
        or (expected_name and str(metadata.get("name") or "") != expected_name)
    ):
        return False

    cluster_result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "mk8s",
            "cluster",
            "get",
            "--id",
            cluster_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    groups_result = _run_capture(
        [
            *_nebius_argv(nebius_bin, profile),
            "mk8s",
            "node-group",
            "list",
            "--parent-id",
            cluster_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if cluster_result.returncode != 0 or groups_result.returncode != 0:
        return False
    try:
        provider_cluster = json.loads(cluster_result.stdout or "{}")
        groups_payload = json.loads(groups_result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    if not isinstance(provider_cluster, dict) or not isinstance(groups_payload, dict):
        return False
    cluster_metadata = provider_cluster.get("metadata", {}) or {}
    cluster_status = provider_cluster.get("status", {}) or {}
    cluster_parent_id = _provider_field(cluster_metadata, "parent_id", "parentId")
    if (
        str(cluster_metadata.get("id") or "") != cluster_id
        or str(
            cluster_parent_id
            if cluster_parent_id is not _PROVIDER_FIELD_MISSING
            else ""
        )
        != project_id
        or str(cluster_metadata.get("name") or "") != cluster.name
        or str(cluster_status.get("state") or "") != "RUNNING"
    ):
        return False
    groups = groups_payload.get("items", [])
    if not isinstance(groups, list):
        return False
    expected_pools = [
        pool
        for pool in (cluster.cpu_nodes, cluster.gpu_nodes)
        if pool is not None and pool.count > 0
    ]
    if len(groups) != len(expected_pools):
        return False

    unmatched = [item for item in groups if isinstance(item, dict)]
    if len(unmatched) != len(groups):
        return False
    for pool in expected_pools:
        match_index = next(
            (
                index
                for index, item in enumerate(unmatched)
                if _provider_node_group_matches_pool(item, pool)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _provider_node_group_matches_pool(
    payload: dict[str, Any], pool: MK8sNodePool
) -> bool:
    """Compare one provider node-group payload with one desired pool."""

    spec = payload.get("spec", {}) or {}
    status = payload.get("status", {}) or {}
    template = spec.get("template", {}) or {}
    resources = template.get("resources", {}) or {}
    reservation_value = _provider_field(
        template, "reservation_policy", "reservationPolicy"
    )
    if reservation_value is _PROVIDER_FIELD_MISSING:
        reservation: dict[str, Any] = {}
    elif isinstance(reservation_value, dict):
        reservation = reservation_value
    else:
        return False
    reservation_ids_value = _provider_field(
        reservation, "reservation_ids", "reservationIds"
    )
    reservation_ids = (
        []
        if reservation_ids_value is _PROVIDER_FIELD_MISSING
        else reservation_ids_value
    )
    fixed_node_count_value = _provider_field(spec, "fixed_node_count", "fixedNodeCount")
    preemptible = _provider_field(template, "preemptible")
    try:
        fixed_node_count = int(fixed_node_count_value)
    except (TypeError, ValueError):
        return False
    strict_reserved = bool(
        pool.capacity_block_group
        and isinstance(reservation, dict)
        and str(reservation.get("policy") or "") == "STRICT"
        and reservation_ids == [pool.capacity_block_group]
    )
    effective_preemptible = (
        False
        if preemptible is _PROVIDER_FIELD_MISSING and strict_reserved
        else preemptible
    )
    expected_preemptible = bool(pool.preemptible)
    if (
        str(status.get("state") or "") != "RUNNING"
        or fixed_node_count != pool.count
        or str(resources.get("platform") or "") != pool.platform
        or str(resources.get("preset") or "") != pool.preset
        # A strict reservation proves non-preemptibility even when the provider
        # omits its redundant false field. Every other capacity mode requires
        # the exact provider boolean matching the desired pool; missing, null,
        # string, or relocated evidence fails closed.
        or effective_preemptible is not expected_preemptible
    ):
        return False
    if pool.capacity_block_group:
        return strict_reserved
    return not reservation_ids and not (
        isinstance(reservation, dict) and reservation.get("policy")
    )


def _post_deploy_validation_phase(cluster: MK8sDesired) -> str:
    if cluster.mig and cluster.mig.enabled:
        return "validating-mig"
    if cluster.gpu_count() > 0:
        return "validating-gpu-health"
    return "validating-cluster-basics"


def _deploy_one_cluster(
    *,
    spec: MK8sExecutionScope,
    project: MK8sProjectIdentity,
    cluster: MK8sDesired,
    project_id: str,
    project_created: bool,
    subnet_id: str,
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
    validation_policy: str = "fleet",
    basic_validation_timeout_minutes: int = 30,
    kubectl_bin: str = "",
    repair_stopped_placeholder: bool = False,
) -> dict[str, Any]:
    project_key = project.key()
    install_dir = fleet_root / project_key / cluster.name
    context = _context_name(spec.name, project_key, cluster.name)
    label = f"{project_key}/{cluster.name}"
    log_metadata = {"terraform_log": str(log_path)} if log_path is not None else {}
    try:
        gpu = cluster.gpu_nodes
        driver = resolve_gpu_driver_strategy(
            gpu_nodes=cluster.gpu_count(),
            platform=gpu.platform if gpu else "",
            preset=gpu.preset if gpu else "",
            mode=cluster.resolved_gpu_driver_mode(),
            managed_driver_preset=cluster.managed_driver_preset,
            enable_gpu_cluster=cluster.resolved_enable_gpu_cluster(),
            allow_unsafe_nvswitch_operator=cluster.allow_unsafe_nvswitch_operator,
        )
        if driver.unsafe_operator_acknowledged:
            _log(
                on_status,
                f"[{label}] WARNING: explicitly acknowledged unsafe operator-mode "
                "driver/Fabric Manager ordering on an NVSwitch topology",
            )
        _ensure_private_directory(fleet_root)
        _ensure_private_directory(install_dir.parent)
        _ensure_private_directory(install_dir)
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
        # Written before apply so ``destroy`` can reconstruct TF_VAR_* even if
        # apply fails midway. Project network ownership is recorded separately.
        # ``status`` starts as "provisioning" and becomes "deployed" only after
        # both apply and kubeconfig generation succeed.
        sidecar = {
            "backend": "mk8s",
            "tenant_id": tenant_id,
            "project_id": project_id,
            "region": region,
            "subnet_id": subnet_id,
            "cluster_name": cluster.name,
            "context": context,
            "profile": profile,
            "gpu_driver_mode": driver.effective_mode,
            "managed_driver_preset": (
                driver.managed_driver_preset if driver.uses_managed_image else ""
            ),
            "status": "provisioning",
        }
        _write_env_sidecar(install_dir, sidecar)
        _log(
            on_status,
            f"[{label}] terraform init" + (f" (-> {log_path})" if log_path else ""),
        )
        with terraform_plugin_cache_lock(env):
            _tf_run(
                [terraform_bin, "init", "-input=false"],
                cwd=workdir,
                env=env,
                timeout=900,
                log_path=log_path,
            )
        recovered_identity = _reconcile_tainted_node_groups(
            terraform_bin=terraform_bin,
            workdir=workdir,
            env=env,
            cluster=cluster,
            project_id=project_id,
            subnet_id=subnet_id,
            nebius_bin=nebius_bin,
            profile=profile,
            on_status=(
                (lambda message: _log(on_status, f"[{label}] {message}"))
                if on_status
                else None
            ),
        )
        if recovered_identity:
            sidecar = {
                **sidecar,
                **recovered_identity,
                "status": "reconciling",
            }
            _write_env_sidecar(install_dir, sidecar)
        if repair_stopped_placeholder:
            _repair_exact_stopped_placeholder(
                terraform_bin=terraform_bin,
                workdir=workdir,
                env=env,
                cluster=cluster,
                project_id=project_id,
                subnet_id=subnet_id,
                nebius_bin=nebius_bin,
                profile=profile,
                on_status=(
                    (lambda message: _log(on_status, f"[{label}] {message}"))
                    if on_status
                    else None
                ),
            )
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
        endpoint = _cluster_endpoint_from_outputs(outputs)
        node_group_ids = _terraform_managed_ids(
            terraform_bin,
            workdir,
            env,
            "nebius_mk8s_v1_node_group",
        )
        node_group_id = node_group_ids[0] if node_group_ids else ""
        kubeconfig_path = install_dir / "kubeconfig"
        if not cluster_id:
            message = "terraform apply succeeded but returned no Managed Kubernetes cluster id"
            _write_env_sidecar(
                install_dir,
                {
                    **sidecar,
                    "cluster_id": "",
                    "status": "deployed-credentials-failed",
                    "error": message,
                },
            )
            return {
                "project_key": project_key,
                "project_id": project_id,
                "cluster_name": cluster.name,
                "region": region,
                "install_dir": str(install_dir),
                "status": "deployed-credentials-failed",
                "error": message,
                **log_metadata,
            }
        _log(on_status, f"[{label}] writing kubeconfig context {context}")
        try:
            _write_kubeconfig(
                nebius_bin, cluster_id, kubeconfig_path, context, env, profile
            )
            _persist_npa_cluster_identity(
                context=context,
                cluster_id=cluster_id,
                project_id=project_id,
                region=region,
                cluster=cluster,
                subnet_id=subnet_id,
                kubeconfig_path=kubeconfig_path,
                fleet_name=spec.name,
                project_key=project_key,
                endpoint=endpoint,
                node_group_id=node_group_id,
            )
        except Exception as exc:  # noqa: BLE001 - retain applied state for credential retry
            message = str(exc)
            _write_env_sidecar(
                install_dir,
                {
                    **sidecar,
                    "cluster_id": cluster_id,
                    "status": "deployed-credentials-failed",
                    "error": message,
                },
            )
            _log(on_status, f"[{label}] credentials FAILED: {message}")
            return {
                "project_key": project_key,
                "project_id": project_id,
                "cluster_name": cluster.name,
                "region": region,
                "cluster_id": cluster_id,
                "kube_context": context,
                "kubeconfig": "",
                "install_dir": str(install_dir),
                "status": "deployed-credentials-failed",
                "error": message,
                **log_metadata,
            }
        gpu_health_path = install_dir / "gpu-health.json"
        verification: dict[str, Any] = {}
        if validation_policy == "skip":
            _log(on_status, f"[{label}] post-deploy validation explicitly skipped")
            verification = {
                "verification": "skipped",
                "status": "validation-skipped",
            }
        elif cluster.gpu_count() > 0 or validation_policy == "standalone-full":
            kubectl_bin = _require_bin(
                kubectl_bin or os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
            )
            is_mig = bool(cluster.mig and cluster.mig.enabled)
            phase_status = _post_deploy_validation_phase(cluster)
            _write_env_sidecar(
                install_dir,
                {
                    **sidecar,
                    "cluster_id": cluster_id,
                    "status": phase_status,
                    "gpu_health_evidence": str(gpu_health_path),
                },
            )
            _log(on_status, f"[{label}] {phase_status.replace('-', ' ')}")
            try:
                verification = verify_cluster(
                    cluster=cluster,
                    kubeconfig=kubeconfig_path,
                    kubectl_bin=kubectl_bin,
                    evidence_path=gpu_health_path,
                    run_capture=_run_capture,
                    mig_verifier=wait_for_mig_ready,
                    gpu_health_verifier=validate_gpu_health,
                    validation_policy=validation_policy,
                    basic_validation_timeout_seconds=(
                        basic_validation_timeout_minutes * 60
                    ),
                    on_status=(
                        (lambda message: _log(on_status, f"[{label}] {message}"))
                        if on_status
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - retain applied state/evidence
                failure_status = (
                    "deployed-mig-not-ready" if is_mig else "deployed-validation-failed"
                )
                message = f"MIG verification failed: {exc}" if is_mig else str(exc)
                _write_env_sidecar(
                    install_dir,
                    {
                        **sidecar,
                        "cluster_id": cluster_id,
                        "status": failure_status,
                        "gpu_health_evidence": str(gpu_health_path),
                        "error": message,
                    },
                )
                _log(on_status, f"[{label}] {message}")
                return {
                    "project_key": project_key,
                    "project_id": project_id,
                    "cluster_name": cluster.name,
                    "region": region,
                    "cluster_id": cluster_id,
                    "endpoint": endpoint,
                    "node_group_id": node_group_id,
                    "node_group_ids": node_group_ids,
                    "kube_context": context,
                    "kubeconfig": str(kubeconfig_path),
                    "install_dir": str(install_dir),
                    "status": failure_status,
                    "gpu_health_evidence": str(gpu_health_path),
                    "error": message,
                    **log_metadata,
                }
        _write_env_sidecar(
            install_dir,
            {
                **sidecar,
                "cluster_id": cluster_id,
                "status": "deployed",
                **(
                    {"gpu_health_evidence": str(gpu_health_path)}
                    if verification.get("verification") == "gpu-health"
                    else {}
                ),
            },
        )
        return {
            "project_key": project_key,
            "project_id": project_id,
            "project_created": project_created,
            "cluster_name": cluster.name,
            "region": region,
            "cluster_id": cluster_id,
            "endpoint": endpoint,
            "node_group_id": node_group_id,
            "node_group_ids": node_group_ids,
            "kube_context": context,
            "kubeconfig": str(kubeconfig_path) if cluster_id else "",
            "install_dir": str(install_dir),
            "status": "deployed",
            **(
                {
                    "gpu_health": verification["gpu_health"],
                    "gpu_health_evidence": str(gpu_health_path),
                }
                if verification.get("verification") == "gpu-health"
                else {}
            ),
            **(
                {"mig": verification["mig"]}
                if verification.get("verification") == "mig"
                else {}
            ),
            **(
                {"cluster_basics": verification["cluster_basics"]}
                if verification.get("cluster_basics")
                else {}
            ),
            "validation": str(verification.get("verification") or "fleet-default"),
            **log_metadata,
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
            **log_metadata,
        }


def _destroy_one_cluster(
    *,
    spec: MK8sExecutionScope,
    project: MK8sProjectIdentity,
    cluster: MK8sDesired,
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
    if not install_dir.exists() and not install_dir.is_symlink():
        _log(on_status, f"[{label}] no install dir; skipping")
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "absent",
        }
    retry_command = (
        "npa fleet destroy --spec <fleet-spec.yaml> "
        f"--only-projects {project.key()} --only-clusters {cluster.name} --yes"
    )
    log_metadata = {"terraform_log": str(log_path)} if log_path is not None else {}
    try:
        _ensure_private_directory(fleet_root)
        _ensure_private_directory(install_dir.parent)
        _ensure_private_directory(install_dir)
    except RuntimeError as exc:
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "destroy-incomplete",
            "errors": [str(exc)],
            "retry_command": retry_command,
            "install_dir": str(install_dir),
            **log_metadata,
        }
    saved = _load_env_sidecar(install_dir) or {}
    project_id = str(saved.get("project_id") or "")
    subnet_id = str(saved.get("subnet_id") or "")
    identity_errors = []
    # Pre-discriminator sidecars are read-compatible as mk8s only because this
    # is the canonical mk8s state root. An explicit different backend remains
    # a hard ownership mismatch.
    recorded_backend = str(saved.get("backend") or "mk8s")
    for label_name, recorded, expected in (
        ("backend", recorded_backend, "mk8s"),
        ("cluster name", str(saved.get("cluster_name") or ""), cluster.name),
        ("project", project_id, project.project_id),
        ("tenant", str(saved.get("tenant_id") or ""), spec.tenant_id),
        ("region", str(saved.get("region") or ""), spec.region),
    ):
        if not recorded or (expected and recorded != expected):
            identity_errors.append(
                f"persisted {label_name} identity does not match request"
            )
    if identity_errors:
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "destroy-incomplete",
            "errors": identity_errors,
            "retry_command": retry_command,
            "install_dir": str(install_dir),
            **log_metadata,
        }
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
    _log(
        on_status,
        f"[{label}] terraform destroy" + (f" (-> {log_path})" if log_path else ""),
    )
    errors: list[str] = []
    try:
        if log_path is not None:
            _ensure_private_log_parent(log_path, fleet_root)
        with terraform_plugin_cache_lock(env):
            _tf_run(
                [terraform_bin, "init", "-input=false"],
                cwd=workdir,
                env=env,
                timeout=900,
                log_path=log_path,
            )
        _tf_run(
            [terraform_bin, "destroy", "-auto-approve", "-input=false"],
            cwd=workdir,
            env=env,
            timeout=timeout_minutes * 60,
            log_path=log_path,
        )
    except Exception as exc:  # noqa: BLE001 - preserve state and try scoped fallback
        errors.append(f"terraform teardown failed: {exc}")
        logger.warning(
            "[%s] terraform teardown incomplete (%s)", label, type(exc).__name__
        )
        _log(
            on_status,
            f"[{label}] terraform teardown incomplete; trying cluster fallback",
        )
        exact_cluster_id = str(saved.get("cluster_id") or "")
        if project_id and exact_cluster_id:
            try:
                exact = _run_capture(
                    [
                        *_nebius_argv(nebius_bin, profile),
                        "mk8s",
                        "cluster",
                        "get",
                        "--id",
                        exact_cluster_id,
                        "--format",
                        "json",
                    ],
                    env=env,
                    check=False,
                    timeout=120,
                )
                if not _is_not_found_result(exact):
                    if exact.returncode != 0 or not exact.stdout.strip():
                        raise RuntimeError(
                            "exact Managed Kubernetes identity is unreadable"
                        )
                    payload = json.loads(exact.stdout)
                    metadata = (
                        payload.get("metadata") if isinstance(payload, dict) else None
                    )
                    parent = _provider_field(metadata, "parent_id", "parentId")
                    if (
                        not isinstance(metadata, dict)
                        or str(metadata.get("id") or "") != exact_cluster_id
                        or str(metadata.get("name") or "") != cluster.name
                        or parent is _PROVIDER_FIELD_MISSING
                        or str(parent) != project_id
                    ):
                        raise RuntimeError(
                            "exact Managed Kubernetes fallback identity does not "
                            "match persisted project/name ownership"
                        )
                fallback = _run_capture(
                    [
                        *_nebius_argv(nebius_bin, profile),
                        "mk8s",
                        "cluster",
                        "delete",
                        "--id",
                        exact_cluster_id,
                    ],
                    env=env,
                    check=False,
                    timeout=timeout_minutes * 60,
                )
                if fallback.returncode != 0 and not _is_not_found_result(fallback):
                    errors.append(
                        f"Managed Kubernetes exact-ID fallback delete failed (nebius exited "
                        f"{fallback.returncode})"
                    )
            except Exception as fallback_exc:  # noqa: BLE001 - report every fallback failure
                errors.append(f"Managed Kubernetes fallback failed: {fallback_exc}")
        else:
            errors.append(
                "exact persisted Managed Kubernetes cluster ID/project identity is missing; "
                "refusing destructive name inference"
            )
        try:
            _write_env_sidecar(
                install_dir,
                {
                    **saved,
                    "status": "destroy-incomplete",
                    "errors": errors,
                    "retry_command": retry_command,
                },
            )
        except OSError as state_exc:
            errors.append(
                f"could not update recovery metadata: {type(state_exc).__name__}"
            )
        _log(on_status, f"[{label}] state retained; retry with: {retry_command}")
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "destroy-incomplete",
            "errors": errors,
            "retry_command": retry_command,
            "install_dir": str(install_dir),
            **log_metadata,
        }

    # Terraform is the authoritative owner of all recipe resources. Only after
    # its successful destroy may the exact global cluster identity and local
    # fleet state be removed.
    try:
        _remove_npa_cluster_identity(
            context=str(saved.get("context") or ""),
            cluster_id=str(saved.get("cluster_id") or ""),
            project_id=project_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(
            f"cloud teardown succeeded but NPA cluster identity cleanup failed: "
            f"{type(exc).__name__}"
        )
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "destroy-incomplete",
            "errors": errors,
            "retry_command": retry_command,
            "install_dir": str(install_dir),
            **log_metadata,
        }
    try:
        shutil.rmtree(install_dir)
    except OSError as exc:
        errors.append(
            f"cloud teardown succeeded but local state cleanup failed: {type(exc).__name__}"
        )
        return {
            "project_key": project.key(),
            "cluster_name": cluster.name,
            "status": "destroy-incomplete",
            "errors": errors,
            "retry_command": retry_command,
            "install_dir": str(install_dir),
            **log_metadata,
        }
    return {
        "project_key": project.key(),
        "cluster_name": cluster.name,
        "status": "destroyed",
        **log_metadata,
    }


def _is_not_found_result(result: Any) -> bool:
    text = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".casefold()
    return "not found" in text or "not_found" in text or "does not exist" in text


deploy_cluster = _deploy_one_cluster
destroy_cluster = _destroy_one_cluster
is_verified_unchanged_target = _is_verified_unchanged_target

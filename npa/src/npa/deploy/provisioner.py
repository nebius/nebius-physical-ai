"""Terraform CLI wrapper for infrastructure provisioning."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# The Nebius Terraform provider prefers an ambient NEBIUS_IAM_TOKEN over the
# explicit `token = var.iam_token` in the provider block. A stale/expired token
# in the environment (a common trap: it can be carried by a long-lived tmux
# server or exported by a shell profile) therefore shadows the fresh token this
# deploy mints via `get_iam_token()`, breaking `terraform apply/destroy` with
# PermissionDenied / Unauthenticated even though the CLI works. Strip these keys
# from the Terraform subprocess env so the fresh `-var iam_token` is always used.
_IAM_TOKEN_ENV_KEYS = ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN")
_AWS_BACKEND_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
)


class ProvisionerError(Exception):
    pass


class TerraformBackendError(ProvisionerError):
    """A typed, recoverable Terraform backend failure."""


class BackendBucketMissingError(TerraformBackendError):
    pass


class BackendAuthenticationError(TerraformBackendError):
    pass


class BackendEndpointError(TerraformBackendError):
    pass


class BackendLockError(TerraformBackendError):
    pass


class BackendInitError(TerraformBackendError):
    pass


class StatePushError(TerraformBackendError):
    pass


class StatePullError(TerraformBackendError):
    pass


@dataclass(frozen=True, repr=False)
class TerraformBackendContext:
    """Secret-bearing credentials bound to one initialized Terraform directory."""

    access_key: str = ""
    secret_key: str = ""
    session_token: str = ""
    endpoint: str = ""
    region: str = ""
    addressing_style: str = "path"

    @classmethod
    def from_mapping(cls, values: dict[str, str]) -> TerraformBackendContext:
        return cls(
            access_key=str(values.get("access_key", "") or ""),
            secret_key=str(values.get("secret_key", "") or ""),
            session_token=str(values.get("session_token", "") or ""),
            endpoint=str(values.get("endpoint", "") or ""),
            region=str(values.get("region", "") or ""),
            addressing_style=str(values.get("addressing_style", "path") or "path"),
        )

    def environment(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.access_key:
            result["AWS_ACCESS_KEY_ID"] = self.access_key
        if self.secret_key:
            result["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        if self.session_token:
            result["AWS_SESSION_TOKEN"] = self.session_token
        return result

    def safe_identity(self) -> dict[str, str]:
        """Return only non-secret fields suitable for receipts and diagnostics."""

        return {
            "endpoint": self.endpoint,
            "region": self.region,
            "addressing_style": self.addressing_style,
            "credential_source": "project_saved" if self.access_key else "none",
        }


_BACKEND_CONTEXTS: dict[Path, TerraformBackendContext] = {}


def _context_key(tf_dir: str | Path) -> Path:
    return Path(tf_dir).expanduser().resolve()


def _set_backend_context(tf_dir: str | Path, context: TerraformBackendContext | None) -> None:
    key = _context_key(tf_dir)
    if context is None:
        _BACKEND_CONTEXTS.pop(key, None)
    else:
        _BACKEND_CONTEXTS[key] = context


def _backend_context(tf_dir: str | Path) -> TerraformBackendContext | None:
    return _BACKEND_CONTEXTS.get(_context_key(tf_dir))


def _uses_remote_s3_backend(tf_dir: str | Path) -> bool:
    backend_file = Path(tf_dir) / "backend.tf"
    try:
        return 'backend "s3"' in backend_file.read_text(encoding="utf-8")
    except OSError:
        return False


def _redact_backend_secrets(message: str, context: TerraformBackendContext | None) -> str:
    cleaned = str(message or "")
    if context is None:
        return cleaned
    for value in (context.access_key, context.secret_key, context.session_token):
        if value:
            cleaned = cleaned.replace(value, "<redacted>")
    return cleaned


_BUNDLED_TF_DIR = Path(__file__).parent / "terraform"
_WORKBENCH_BASE = Path.home() / ".npa" / "workbenches"
# Shared Terraform plugin cache so every fresh per-deploy work dir reuses
# already-downloaded providers instead of re-fetching them from
# registry.terraform.io. This makes `terraform init` faster and resilient to
# transient registry outages (a warm cache needs no network at all). Overridable
# via the standard TF_PLUGIN_CACHE_DIR env var.
_TF_PLUGIN_CACHE_DIR = Path.home() / ".npa" / "terraform-plugin-cache"
# Substrings that identify a transient `terraform init` failure worth retrying
# (registry/network hiccups) rather than a real configuration error.
_TRANSIENT_INIT_MARKERS = (
    "could not connect to registry",
    "failed to request discovery document",
    "timeout exceeded while awaiting headers",
    "failed to retrieve",
    "no such host",
    "connection reset",
    "connection refused",
    "temporary failure in name resolution",
    "i/o timeout",
    "tls handshake timeout",
    "eof",
)
DEFAULT_VM_BOOT_DISK_SIZE_GB = 100
DEFAULT_CONTAINER_BOOT_DISK_SIZE_GB = 250

# Template for the S3 backend block.  Bucket, key, endpoint, and region are
# baked into the file so that ``endpoints`` (the newer Terraform >=1.6 syntax)
# works correctly — it cannot be set via ``-backend-config`` flat keys.
# Only credentials are passed at init time via ``-backend-config``.
_BACKEND_TF_TEMPLATE = """\
# Auto-generated by npa — do not edit.
terraform {{
  backend "s3" {{
    bucket = "{bucket}"
    key    = "npa/terraform-state/{project}/{name}/terraform.tfstate"

    endpoints = {{
      s3 = "{endpoint}"
    }}
    region = "{region}"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    use_path_style              = true
  }}
}}
"""


def _require_terraform() -> str:
    tf = shutil.which("terraform")
    if tf is None:
        raise ProvisionerError(
            "terraform binary not found on PATH. "
            "Install it: https://developer.hashicorp.com/terraform/install"
        )
    return tf


def _tf_env(terraform_dir: str | Path) -> dict[str, str]:
    """Return the subprocess env for terraform with a warm plugin cache.

    Also strips stale Nebius IAM tokens (_IAM_TOKEN_ENV_KEYS) so a fresh
    ``-var iam_token`` is always used instead of an ambient token that the
    provider would otherwise prefer (causing PermissionDenied / Unauthenticated).
    """
    env = dict(os.environ)
    for key in _IAM_TOKEN_ENV_KEYS:
        env.pop(key, None)
    from npa.terraform_lock import TerraformLockError, configure_plugin_cache

    try:
        configure_plugin_cache(
            env,
            terraform_dir,
            default_root=_TF_PLUGIN_CACHE_DIR,
        )
    except (OSError, TerraformLockError) as exc:
        raise ProvisionerError(f"Unsafe Terraform plugin cache configuration: {exc}") from exc
    return env


def _run(
    args: list[str],
    *,
    cwd: str | Path,
    capture: bool = False,
    stream: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    tf = _require_terraform()
    cmd = [tf] + args
    environment = _tf_env(cwd)
    environment.update(env_overrides or {})
    backend_context = _backend_context(cwd)
    if args and args[0] != "init" and _uses_remote_s3_backend(cwd) and backend_context is None:
        raise BackendAuthenticationError(
            "Terraform S3 backend credentials unavailable for this initialized work directory; "
            "run the project-scoped NPA init/reconfigure path before state access."
        )
    if backend_context is not None:
        # Explicit project-scoped credentials win over unrelated shell/tmux/CI
        # AWS variables and over any accidental call-site override.
        for key in _AWS_BACKEND_ENV_KEYS:
            environment.pop(key, None)
        environment.update(backend_context.environment())
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "text": True,
        "env": environment,
    }

    if stream and args and args[0] in {"apply", "destroy"}:
        return _run_destroy_stream(
            cmd, cwd=Path(cwd), env=kwargs["env"], backend_context=backend_context
        )

    if capture:
        kwargs["capture_output"] = True
    elif stream:
        kwargs["stdout"] = sys.stdout
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs["capture_output"] = True

    result = subprocess.run(cmd, **kwargs)
    if result.stdout and capture:
        result.stdout = _redact_backend_secrets(result.stdout, backend_context)
    if result.stderr:
        result.stderr = _redact_backend_secrets(result.stderr, backend_context)
    if stream and result.stderr:
        sys.stderr.write(result.stderr)
        sys.stderr.flush()
    if result.returncode != 0 and capture:
        stderr = result.stderr or ""
        raise ProvisionerError(
            f"terraform {args[0]} failed (exit {result.returncode}):\n{stderr.strip()}"
        )
    return result


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DEPRECATED_GPU_OUTPUT_RE = re.compile(
    r"^\s*[+~-]\s+gpu_(?:platform|preset)\s+="
)


def _filter_destroy_output_line(line: str) -> str:
    """Hide deprecated Terraform output aliases from human destroy progress.

    The aliases remain in the machine-readable Terraform schema for compatible
    GPU callers, but a CPU agent's destroy plan must not label ``cpu-d3`` or
    ``8vcpu-32gb`` as GPU facts. Canonical ``platform``/``preset`` and
    ``cpu_platform``/``cpu_preset`` lines remain visible.
    """

    plain = _ANSI_ESCAPE_RE.sub("", line)
    return "" if _DEPRECATED_GPU_OUTPUT_RE.match(plain) else line


def _compact_local_exec_error_line(
    line: str, *, suppressing_body: bool
) -> tuple[str, bool]:
    """Collapse Terraform's quoted local-exec script while retaining its output."""

    marker = "': exit status"
    if not suppressing_body and "Error running command '" not in line:
        return line, False
    if not suppressing_body:
        suppressing_body = True
    if marker not in line:
        return "", suppressing_body
    tail = line.split(marker, 1)[1]
    status, _, output = tail.partition(". Output:")
    visible = (
        "Terraform local-exec provisioner failed "
        f"(exit status {status.strip() or 'unknown'}).\n"
    )
    if output.strip():
        visible += output.lstrip()
        if not visible.endswith("\n"):
            visible += "\n"
    return visible, False


def _run_destroy_stream(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    backend_context: TerraformBackendContext | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream Terraform destroy while filtering only deprecated output aliases."""

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    compact_local_exec = bool(cmd and Path(cmd[0]).name == "terraform" and "apply" in cmd)

    def _pump_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = _redact_backend_secrets(line, backend_context)
            stdout_lines.append(safe_line)
            visible = _filter_destroy_output_line(safe_line)
            if visible:
                sys.stdout.write(visible)
                sys.stdout.flush()

    def _pump_stderr() -> None:
        assert process.stderr is not None
        suppressing_body = False
        for line in process.stderr:
            safe_line = _redact_backend_secrets(line, backend_context)
            visible = safe_line
            if compact_local_exec:
                visible, suppressing_body = _compact_local_exec_error_line(
                    safe_line, suppressing_body=suppressing_body
                )
            if visible:
                stderr_lines.append(visible)
                sys.stderr.write(visible)
                sys.stderr.flush()

    threads = [
        threading.Thread(target=_pump_stdout, name="npa-terraform-stdout", daemon=True),
        threading.Thread(target=_pump_stderr, name="npa-terraform-stderr", daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def _build_var_args(tf_vars: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in tf_vars.items():
        args.extend(["-var", f"{key}={value}"])
    return args


#: Variable names whose values must not reach the process table. `ps` is readable
#: by every local user, so `-var iam_token=...` / `-var nebius_secret_key=...`
#: leaked the deploy's IAM token and the state bucket's S3 keys for the duration
#: of the apply. Those values go through a 0600 var-file instead.
_SENSITIVE_VAR_HINTS = ("token", "secret", "password", "api_key", "_key")
_SENSITIVE_VAR_FILE = "npa-sensitive.tfvars.json"


def _is_sensitive_var(name: str) -> bool:
    lowered = str(name or "").lower()
    if lowered.endswith("_key_path") or lowered.endswith("_path"):
        return False
    return any(hint in lowered for hint in _SENSITIVE_VAR_HINTS)


def _split_sensitive_vars(tf_vars: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    sensitive = {
        key: value
        for key, value in tf_vars.items()
        if _is_sensitive_var(key) and str(value or "") != ""
    }
    plain = {key: value for key, value in tf_vars.items() if key not in sensitive}
    return sensitive, plain


@contextmanager
def _var_args(tf_dir: Path, tf_vars: dict[str, str]) -> Iterator[list[str]]:
    """Yield terraform var arguments, keeping secret values out of argv.

    The file is written 0600 inside the per-deploy working directory (itself
    0700) and removed when the command finishes. It is deliberately *not* named
    ``*.auto.tfvars.json``, so it only applies to the command that passes it.
    """
    sensitive, plain = _split_sensitive_vars(tf_vars)
    if not sensitive:
        yield _build_var_args(plain)
        return
    path = Path(tf_dir) / _SENSITIVE_VAR_FILE
    path.write_text(json.dumps(sensitive, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - unusual filesystems
        pass
    try:
        yield [f"-var-file={path}", *_build_var_args(plain)]
    finally:
        path.unlink(missing_ok=True)


def _runtime_value(runtime: Any) -> str:
    return str(getattr(runtime, "value", runtime))


def resolve_boot_disk_size_gb(runtime: Any, disk_size: int | None = None) -> int | None:
    """Return the boot disk size override for Terraform.

    VM deploys intentionally return ``None`` by default so the Terraform
    module keeps its existing default. Container deploys need a larger boot
    disk because image pulls and unpacked layers are written to the root disk.
    """
    if disk_size is not None:
        if disk_size <= 0:
            raise ValueError(f"--disk-size must be positive, got {disk_size}")
        return disk_size
    if _runtime_value(runtime) == "container":
        return DEFAULT_CONTAINER_BOOT_DISK_SIZE_GB
    return None


def boot_disk_tf_vars(runtime: Any, disk_size: int | None = None) -> dict[str, str]:
    size = resolve_boot_disk_size_gb(runtime, disk_size)
    if size is None:
        return {}
    return {"boot_disk_size_gb": str(size)}


def apply_boot_disk_tf_vars(
    tf_vars: dict[str, str],
    runtime: Any,
    disk_size: int | None = None,
) -> None:
    disk_vars = boot_disk_tf_vars(runtime, disk_size)
    if disk_vars and (disk_size is not None or "boot_disk_size_gb" not in tf_vars):
        tf_vars.update(disk_vars)


# Platforms that reject older public CUDA12 images (need nvidia drivers 580.x).
DEFAULT_CUDA13_IMAGE_FAMILY = "ubuntu24.04-cuda13.0"
_PLATFORMS_REQUIRING_CUDA13_IMAGE = frozenset(
    {
        "gpu-rtx6000",
        "gpu-b300-sxm",
    }
)


def default_image_family_for_platform(gpu_platform: str) -> str | None:
    """Return a boot image family override for GPU platforms that need it."""
    platform = (gpu_platform or "").strip().lower()
    if platform in _PLATFORMS_REQUIRING_CUDA13_IMAGE:
        return DEFAULT_CUDA13_IMAGE_FAMILY
    return None


def apply_default_image_family(tf_vars: dict[str, str], gpu_platform: str) -> None:
    """Set ``image_family`` when unset and the GPU platform requires CUDA 13 / driver 580."""
    if "image_family" in tf_vars:
        return
    family = default_image_family_for_platform(gpu_platform)
    if family:
        tf_vars["image_family"] = family


# ── Working directory management ─────────────────────────────────────────


def prepare_working_dir(
    project: str,
    name: str,
    *,
    bucket: str,
    region: str,
    endpoint: str,
) -> Path:
    """Create or update a per-instance Terraform working directory.

    Copies the bundled .tf and .tpl files into
    ``~/.npa/workbenches/{project}/{name}/`` and writes a ``backend.tf``
    for S3 remote state.  Returns the working directory path.
    """
    work_dir = _WORKBENCH_BASE / project / name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy every file from the bundled Terraform directory.
    for src in _BUNDLED_TF_DIR.iterdir():
        if src.is_file():
            shutil.copy2(src, work_dir / src.name)

    # Write the S3 backend block with environment-specific values.
    backend_tf = _BACKEND_TF_TEMPLATE.format(
        bucket=bucket,
        project=project,
        name=name,
        endpoint=endpoint,
        region=region,
    )
    (work_dir / "backend.tf").write_text(backend_tf)

    return work_dir


def working_dir_path(project: str, name: str) -> Path:
    """Return the path to a workbench's Terraform working directory."""
    return _WORKBENCH_BASE / project / name


def cleanup_working_dir(project: str, name: str) -> None:
    """Remove the per-instance working directory."""
    work_dir = _WORKBENCH_BASE / project / name
    if work_dir.exists():
        shutil.rmtree(work_dir)


# ── Terraform commands ───────────────────────────────────────────────────


def _looks_transient_init_failure(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_INIT_MARKERS)


def _classify_backend_error(message: str, *, action: str) -> TerraformBackendError:
    """Map backend failures to stable exception types without echoing credentials."""

    lowered = str(message or "").lower()
    if "nosuchbucket" in lowered or "no such bucket" in lowered:
        kind: type[TerraformBackendError] = BackendBucketMissingError
    elif any(item in lowered for item in ("accessdenied", "forbidden", "403", "signature")):
        kind = BackendAuthenticationError
    elif any(
        item in lowered
        for item in ("no such host", "name resolution", "dial tcp", "dns", "endpoint")
    ):
        kind = BackendEndpointError
    elif any(item in lowered for item in ("state lock", "acquiring the state lock", "lock id")):
        kind = BackendLockError
    elif action == "state-push":
        kind = StatePushError
    elif action == "state-pull":
        kind = StatePullError
    else:
        kind = BackendInitError
    return kind(f"Terraform backend {action} failed: {message}")


def init(
    tf_dir: str | Path | None = None,
    backend_config: dict[str, str] | None = None,
    *,
    retries: int = 3,
    backoff_seconds: float = 4.0,
    sleep: Any = time.sleep,
    force_reconfigure: bool = False,
    disable_backend: bool = False,
) -> None:
    """Run terraform init.

    If *backend_config* is provided, credentials are supplied only through the
    Terraform subprocess environment. They never enter argv. ``-reconfigure``
    is always used so a rotated credential generation or changed backend target
    cannot be shadowed by cached ``.terraform`` metadata.

    Provider installation from registry.terraform.io can fail transiently
    (timeouts, DNS blips); those are retried with exponential backoff so a fresh
    VM is not rolled back over a momentary registry hiccup. Non-transient
    failures (e.g. a real config error) raise immediately.
    """
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    from npa.terraform_lock import TerraformLockError, validate_provider_lock

    try:
        validate_provider_lock(tf_dir)
    except TerraformLockError as exc:
        raise ProvisionerError(f"Terraform provider-lock preflight failed: {exc}") from exc
    lock_file = tf_dir / ".terraform.lock.hcl"
    lock_before = lock_file.read_bytes()
    args = ["init", "-input=false", "-lockfile=readonly"]
    backend_env: dict[str, str] = {}
    if disable_backend:
        args.append("-backend=false")
        _set_backend_context(tf_dir, None)
    elif backend_config is not None:
        args.append("-reconfigure")
        backend_context = TerraformBackendContext.from_mapping(backend_config)
        if _uses_remote_s3_backend(tf_dir) and (
            not backend_context.access_key or not backend_context.secret_key
        ):
            _set_backend_context(tf_dir, None)
            raise BackendAuthenticationError(
                "Terraform S3 backend credentials unavailable; restore the exact "
                "project-saved Object Storage HMAC credentials before init/apply/destroy."
            )
        _set_backend_context(tf_dir, backend_context)
        backend_env.update(backend_context.environment())
    elif force_reconfigure:
        args.append("-reconfigure")
        _set_backend_context(tf_dir, None)
    else:
        # A local init must not inherit a remote backend context if a long-lived
        # NPA process later reuses the same work directory.
        _set_backend_context(tf_dir, None)
    delay = backoff_seconds
    attempts = max(1, retries)
    attempt = 1
    checksum_retry_used = False
    isolated_cache: tempfile.TemporaryDirectory[str] | None = None
    try:
        while attempt <= attempts:
            try:
                retry_env = dict(backend_env)
                if isolated_cache is not None:
                    retry_env["TF_PLUGIN_CACHE_DIR"] = isolated_cache.name
                if retry_env:
                    _run(args, cwd=tf_dir, capture=True, env_overrides=retry_env)
                else:
                    _run(args, cwd=tf_dir, capture=True)
                lock_after = lock_file.read_bytes()
                if lock_after != lock_before:
                    raise ProvisionerError(
                        f"Terraform changed {lock_file} despite -lockfile=readonly. NPA "
                        "stopped before apply; restore and review the tracked lock."
                    )
                return
            except ProvisionerError as exc:
                try:
                    lock_after = lock_file.read_bytes()
                except OSError:
                    lock_after = b""
                if lock_after != lock_before:
                    raise ProvisionerError(
                        f"Terraform changed {lock_file} despite -lockfile=readonly. NPA "
                        "stopped before apply; restore and review the tracked lock."
                    ) from exc
                lowered = str(exc).lower()
                checksum_failure = any(
                    marker in lowered
                    for marker in (
                        "doesn't match any of the checksums",
                        "does not match any of the checksums",
                        "checksum mismatch",
                    )
                )
                if checksum_failure and not checksum_retry_used:
                    checksum_retry_used = True
                    isolated_cache = tempfile.TemporaryDirectory(prefix="npa-tf-plugin-cache-")
                    Path(isolated_cache.name).chmod(0o700)
                    attempts += 1
                    attempt += 1
                    sys.stderr.write(
                        "  terraform init found an invalid cached provider; retrying once "
                        "with an isolated empty cache while keeping the lock file read-only...\n"
                    )
                    sys.stderr.flush()
                    continue
                if checksum_failure:
                    raise ProvisionerError(
                        "Terraform provider checksum verification failed. NPA kept "
                        f"{lock_file} immutable and will not bypass verification. Verify "
                        "the registry/mirror, then regenerate all supported platform hashes "
                        f"with `terraform -chdir={tf_dir} providers lock -platform=...` "
                        "in a clean reviewed checkout and inspect the exact diff."
                    ) from exc
                if attempt >= attempts or not _looks_transient_init_failure(str(exc)):
                    raise _classify_backend_error(str(exc), action="init/reconfigure") from exc
                sys.stderr.write(
                    f"  terraform init attempt {attempt}/{attempts} hit a transient "
                    f"registry/network error; retrying in {delay:.0f}s...\n"
                )
                sys.stderr.flush()
                sleep(delay)
                delay *= 2
                attempt += 1
    finally:
        if isolated_cache is not None:
            isolated_cache.cleanup()


def plan(
    tf_dir: str | Path | None = None,
    tf_vars: dict[str, str] | None = None,
) -> str:
    """Run terraform plan, return human-readable summary."""
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    with _var_args(tf_dir, tf_vars or {}) as var_args:
        result = _run(["plan", "-input=false", "-no-color", *var_args], cwd=tf_dir, capture=True)
    return result.stdout


def apply(
    tf_dir: str | Path | None = None,
    tf_vars: dict[str, str] | None = None,
    *,
    stream: bool = True,
) -> dict[str, Any]:
    """Run terraform apply -auto-approve, return outputs dict."""
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    with _var_args(tf_dir, tf_vars or {}) as var_args:
        result = _run(
            ["apply", "-auto-approve", "-input=false", *var_args], cwd=tf_dir, stream=stream
        )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        detail = f":\n{stderr}" if stderr else ""
        raise ProvisionerError(f"terraform apply failed (exit {result.returncode}){detail}")
    return outputs(tf_dir)


def destroy(
    tf_dir: str | Path | None = None,
    tf_vars: dict[str, str] | None = None,
    *,
    stream: bool = True,
    targets: list[str] | None = None,
) -> None:
    """Run terraform destroy -auto-approve."""
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    args = ["destroy", "-auto-approve", "-input=false"]
    if targets:
        for target in targets:
            args.extend(["-target", target])
    with _var_args(tf_dir, tf_vars or {}) as var_args:
        result = _run([*args, *var_args], cwd=tf_dir, stream=stream)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        detail = f":\n{stderr}" if stderr else ""
        raise ProvisionerError(f"terraform destroy failed (exit {result.returncode}){detail}")


def state_list(tf_dir: str | Path | None = None) -> list[str]:
    """Run terraform state list and return managed resource addresses."""
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    # Terraform exits 1 for a valid, initialized backend whose state object does
    # not exist yet.  That is the expected first-deploy representation of an
    # empty namespace, not an inspection failure.  Keep every other nonzero
    # result fail-closed so auth/backend failures cannot be mistaken for empty
    # state by lifecycle ownership guards.
    result = _run(["state", "list"], cwd=tf_dir)
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        if "no state file was found" in detail.lower():
            return []
        suffix = f":\n{detail}" if detail else ""
        raise ProvisionerError(
            f"terraform state failed (exit {result.returncode}){suffix}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def state_pull(tf_dir: str | Path | None = None) -> bytes:
    """Pull the current state so callers can preserve and independently verify it."""

    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    try:
        result = _run(["state", "pull"], cwd=tf_dir, capture=True)
    except ProvisionerError as exc:
        raise _classify_backend_error(str(exc), action="state-pull") from exc
    data = (result.stdout or "").encode("utf-8")
    if not data.strip():
        raise StatePullError("Terraform state pull returned an empty document")
    return data


def state_push(state_path: str | Path, tf_dir: str | Path | None = None) -> None:
    """Push a preserved state copy, classifying backend failures for recovery."""

    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    path = Path(state_path)
    try:
        _run(["state", "push", "-force", str(path)], cwd=tf_dir, capture=True)
    except ProvisionerError as exc:
        raise _classify_backend_error(str(exc), action="state-push") from exc


def state_resource_id(address: str, tf_dir: str | Path | None = None) -> str:
    """Return the id of one Terraform state resource without dumping full state.

    ``terraform show -json`` also contains sensitive variables for this stack.
    Query only the NPA-owned network resource whose ID is needed for the narrow
    default-security-group teardown recovery.
    """

    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    result = _run(["state", "show", "-no-color", address], cwd=tf_dir, capture=True)
    for line in result.stdout.splitlines():
        match = re.match(r'^\s*id\s*=\s*"([^"\r\n]+)"\s*$', line)
        if match:
            return match.group(1).strip()
    return ""


def outputs(tf_dir: str | Path | None = None) -> dict[str, Any]:
    """Run terraform output -json, parse and return dict of {key: value}."""
    tf_dir = Path(tf_dir) if tf_dir else _BUNDLED_TF_DIR
    result = _run(["output", "-json"], cwd=tf_dir, capture=True)
    raw = json.loads(result.stdout)
    # Terraform output -json returns {"key": {"value": ..., "type": ...}}
    return {key: entry["value"] for key, entry in raw.items()}

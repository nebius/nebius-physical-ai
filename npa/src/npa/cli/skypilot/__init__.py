"""CLI helpers for managing the isolated SkyPilot runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import fcntl

import typer
from rich.console import Console
import yaml  # type: ignore[import-untyped]

from npa.orchestration.skypilot._bin import REQUIRED_SKYPILOT_VERSION
from npa.orchestration.skypilot.workflow_state import redact_text
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

app = typer.Typer(
    name="skypilot",
    help="Manage the isolated SkyPilot runtime used by NPA workflows.",
    no_args_is_help=True,
)

console = Console(stderr=True)

SKYPILOT_VERSION = REQUIRED_SKYPILOT_VERSION
UTC = timezone.utc
SKYPILOT_EXTRAS = ("nebius", "kubernetes")
SKYPILOT_PACKAGE = f"skypilot[{','.join(SKYPILOT_EXTRAS)}]=={SKYPILOT_VERSION}"
# SkyPilot 0.12.2 declares `kubernetes!=32.0.0,>=20.0.0` with no upper bound, so a
# fresh bootstrap resolves the newest client. Client 36.0.0 renamed the generated
# `openapi_types` entries from `dict(str, str)` to `dict[str, str]`, which SkyPilot's
# PodValidator turns into an import of `kubernetes.client.models.dict[str, str]`.
# Every pod_config then fails validation and the managed-jobs controller retries
# forever, so pin the client below the break.
KUBERNETES_CLIENT_MAX_EXCLUSIVE = "36"
KUBERNETES_CLIENT_SPEC = f"kubernetes>=20.0.0,!=32.0.0,<{KUBERNETES_CLIENT_MAX_EXCLUSIVE}"
DEFAULT_VENV_PATH = Path(
    os.environ.get("NPA_CONFIG_DIR", "").strip() or (Path.home() / ".npa")
) / "skypilot-venv"
VENV_PATH_ENV = "NPA_SKYPILOT_VENV_PATH"
PYTHON_ENV = "NPA_SKYPILOT_PYTHON"
MARKER_FILE = ".npa-bootstrap-ok"
BOOTSTRAP_SCHEMA_VERSION = "npa.skypilot.bootstrap.v1"
BOOTSTRAP_LOCK_SUFFIX = ".bootstrap.lock"
BOOTSTRAP_JOURNAL_SUFFIX = ".bootstrap-journal.json"
CONSTRAINTS_FILE = Path(__file__).with_name("constraints-0.12.2.txt")

# SkyPilot 0.12.2 (and its `kubernetes`/`ray` dependencies) build and import
# cleanly only on this Python range. On a too-new interpreter (e.g. 3.14 on a
# fresh image) `pip install skypilot[kubernetes]` pulls a kubernetes client whose
# typing/imports fail, so the venv is created but `import sky`/submits blow up.
# Guard the interpreter up front and auto-select a supported one when the default
# is out of range.
SKYPILOT_MIN_PYTHON = (3, 9)
SKYPILOT_MAX_PYTHON = (3, 12)
_PREFERRED_PYTHON_BINS = ("python3.12", "python3.11", "python3.10", "python3.9")


def _supported_python_range_str() -> str:
    return (
        f"{SKYPILOT_MIN_PYTHON[0]}.{SKYPILOT_MIN_PYTHON[1]}-"
        f"{SKYPILOT_MAX_PYTHON[0]}.{SKYPILOT_MAX_PYTHON[1]}"
    )


class SkyPilotBootstrapError(RuntimeError):
    """Raised when the isolated SkyPilot runtime cannot be bootstrapped."""


class _BootstrapLock:
    """Owner-only advisory lock serializing one exact managed environment."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> _BootstrapLock:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SkyPilotBootstrapError(
                f"Cannot acquire the SkyPilot bootstrap lock {self.path}: {exc}"
            ) from exc
        os.fchmod(fd, 0o600)
        self._handle = os.fdopen(fd, "a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_args: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


@dataclass(frozen=True)
class VenvState:
    path: Path
    python_bin: Path
    pip_bin: Path
    sky_bin: Path
    exists: bool
    has_python: bool
    has_pip: bool
    has_sky: bool
    version: str | None
    importable: bool
    marker_path: Path
    kubernetes_version: str | None = None

    @property
    def installed(self) -> bool:
        return self.version == SKYPILOT_VERSION and self.importable and self.has_sky

    @property
    def kubernetes_compatible(self) -> bool:
        """Whether the installed kubernetes client works with SkyPilot's pod_config."""

        return kubernetes_client_supported(self.kubernetes_version)


@dataclass(frozen=True)
class BootstrapResult:
    path: Path
    sky_bin: Path
    installed: bool
    reused: bool
    marker_path: Path


@app.command("bootstrap")
def bootstrap_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    python: str = typer.Option(
        "",
        "--python",
        help=f"Python executable used to create the isolated venv. Defaults to {PYTHON_ENV} or this interpreter.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help=(
            "Persist the resolved sky binary as skypilot.sky_bin in "
            "~/.npa/config.yaml so later shells resolve it without exporting "
            "NPA_SKYPILOT_BIN."
        ),
    ),
) -> None:
    """Install SkyPilot into an isolated, idempotent virtualenv."""

    try:
        result = bootstrap_skypilot(venv_path=path, python_bin=python or None)
    except SkyPilotBootstrapError as exc:
        _fail(str(exc))
        return

    state = "already installed" if result.reused else "installed"
    typer.echo(f"SkyPilot {SKYPILOT_VERSION} {state} at {result.path}")
    typer.echo(str(result.sky_bin))
    typer.echo(f"export NPA_SKYPILOT_BIN={shlex.quote(str(result.sky_bin))}")
    if save:
        # Printing an `export` line only helped the current shell: the next
        # shell (and every `npa workbench workflow submit` in it) failed with
        # "SkyPilot CLI executable is not configured". `skypilot.sky_bin` in
        # ~/.npa/config.yaml is the persistent form NPA already resolves.
        saved_path = _persist_sky_bin(result.sky_bin)
        if saved_path:
            typer.echo(f"saved: skypilot.sky_bin -> {saved_path}")
        else:
            typer.echo(
                "warning: could not save skypilot.sky_bin; export "
                "NPA_SKYPILOT_BIN in each shell instead.",
                err=True,
            )
    typer.echo(f"marker: {result.marker_path}")


def _persist_sky_bin(sky_bin: Path) -> str:
    """Write ``skypilot.sky_bin`` into ``~/.npa/config.yaml``; "" on failure."""

    try:
        from npa.clients.config import write_config

        return str(write_config({"skypilot": {"sky_bin": str(sky_bin)}}))
    except Exception:  # noqa: BLE001 - persisting is a convenience, never fatal
        return ""


def _clear_saved_sky_bin() -> bool:
    """Drop ``skypilot.sky_bin`` from ``~/.npa/config.yaml``; return if present."""

    try:
        from npa.clients.config import clear_skypilot_bin

        return clear_skypilot_bin()
    except Exception:  # noqa: BLE001 - never fatal during teardown
        return False


@app.command("uninstall")
def uninstall_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Remove the isolated SkyPilot venv and clear the saved sky binary.

    The inverse of `npa skypilot bootstrap`: deletes the venv and drops
    ``skypilot.sky_bin`` from ~/.npa/config.yaml, so a torn-down environment does
    not leave a dangling runtime or a persisted binary path behind.
    """

    venv_path = _resolve_venv_path(path)
    try:
        _validate_managed_venv_path(venv_path)
    except SkyPilotBootstrapError as exc:
        _fail(str(exc))
        return

    existed = venv_path.exists()
    if existed and not yes and sys.stdin.isatty():
        if not typer.confirm(f"Remove the SkyPilot venv at {venv_path}?", default=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=1)
    if existed:
        state = inspect_venv(venv_path)
        if not state.has_python or not state.marker_path.is_file():
            _fail(
                f"Refusing to delete {venv_path}: it is not an exact NPA-managed "
                f"SkyPilot environment (missing Python or {MARKER_FILE})."
            )
            return
        tombstone = venv_path.with_name(f".{venv_path.name}.uninstall-{os.getpid()}")
        if tombstone.exists() or tombstone.is_symlink():
            _fail(f"Refusing occupied uninstall staging path: {tombstone}")
            return
        os.replace(venv_path, tombstone)
        shutil.rmtree(tombstone)
        typer.echo(f"Removed SkyPilot venv {venv_path}.")
    else:
        typer.echo(f"No SkyPilot venv at {venv_path}.")

    if _clear_saved_sky_bin():
        typer.echo("Cleared skypilot.sky_bin from ~/.npa/config.yaml.")


@app.command("status")
def status_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    bin_path: bool = typer.Option(False, "--bin-path", help="Print only the resolved sky binary path."),
    project: str = typer.Option(
        "", "--project", help="Project alias for immutable controller verification."
    ),
    context: str = typer.Option(
        "", "--context", help="Kubernetes context for immutable controller verification."
    ),
) -> None:
    """Report the isolated SkyPilot runtime status."""

    state = inspect_venv(_resolve_venv_path(path))
    if bin_path:
        if not state.has_sky:
            _fail(f"SkyPilot binary is not installed at {state.sky_bin}. Run `npa skypilot bootstrap`.")
            return
        typer.echo(str(state.sky_bin))
        return

    if not state.installed:
        detail = f"found version {state.version}" if state.version else "sky binary missing or not executable"
        _fail(f"SkyPilot {SKYPILOT_VERSION} is not ready in {state.path}: {detail}. Run `npa skypilot bootstrap`.")
        return

    marker_age = _format_marker_age(state.marker_path)
    typer.echo(f"venv_path: {state.path}")
    typer.echo(f"sky_bin: {state.sky_bin}")
    typer.echo(f"version: {state.version}")
    typer.echo(f"marker: {state.marker_path}")
    typer.echo(f"marker_age: {marker_age}")
    typer.echo(f"kubernetes_client: {state.kubernetes_version or 'not installed'}")

    if not state.kubernetes_compatible:
        _fail(kubernetes_client_remedy(state.kubernetes_version))
        return

    try:
        from npa.controller_ownership import (
            ControllerOwner,
            verify_controller_owner,
            verify_recorded_controller_owner,
        )

        owner: ControllerOwner | None
        if project or context:
            if not project or not context:
                raise ValueError(
                    "Controller verification requires both --project and --context."
                )
            owner = verify_controller_owner(project, context)
        else:
            owner = verify_recorded_controller_owner()
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        "controller_owner: "
        + (
            f"{owner.project_alias}/{owner.context}/{owner.cluster_id}"
            if owner is not None
            else "unbound"
        )
    )

    result = _run_observable(
        [str(state.sky_bin), "check"], label="SkyPilot status check"
    )
    summary = _summarize_completed_process(result)
    typer.echo(f"sky_check: {summary}")


@app.command("cleanup-controller")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def cleanup_controller_cmd(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm teardown of the shared managed-jobs controller after workflows are terminal.",
    ),
    sky_bin: str = typer.Option(
        "", "--sky-bin", help="Pinned NPA SkyPilot executable override."
    ),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help=(
            "Exact NPA project alias. Defaults only to an unambiguous selected "
            "NPA project; never to a SkyPilot/Nebius profile."
        ),
    ),
    context: str = typer.Option(
        "",
        "--context",
        help=(
            "Exact NPA cluster context. Defaults only when the selected project "
            "has exactly one NPA-owned cluster record."
        ),
    ),
    receipt: str = typer.Option("", "--receipt", help="Opaque teardown receipt ID."),
    project_id: str = typer.Option("", "--project-id", help="Exact Nebius project ID."),
    cluster_id: str = typer.Option("", "--cluster-id", help="Exact immutable cluster ID."),
    cluster_name: str = typer.Option("", "--cluster-name", help="Exact provider cluster name."),
    recover_orphan_controller: bool = typer.Option(
        False,
        "--recover-orphan-controller",
        help=(
            "Delete exact controller pods when verified NPA ownership exists but "
            "SkyPilot metadata is absent; requires --attest-no-active-jobs."
        ),
    ),
    attest_no_active_jobs: bool = typer.Option(
        False,
        "--attest-no-active-jobs",
        help=(
            "Attest that exact workflow status/cancel evidence proves no active "
            "managed jobs before orphan-controller recovery."
        ),
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable result."
    ),
) -> None:
    """Tear down NPA's shared jobs controller after its managed jobs drain."""

    if not yes:
        message = (
            "Plan only: this shared managed-jobs controller may serve every SkyPilot "
            "workflow on its cluster. Verify all workflows are terminal, then re-run "
            "`npa skypilot cleanup-controller --project <alias> --context "
            "<exact-context> --yes`. NPA will verify immutable project/cluster identity "
            "and remote absence before removing local metadata. No state was changed."
        )
        if output_json:
            typer.echo(
                json.dumps(
                    {"outcome": "confirmation_required", "changed": False, "message": message},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            typer.echo(message)
        raise typer.Exit(code=1)

    from npa.orchestration.skypilot.cleanup import cleanup_jobs_controller

    payload: dict[str, Any]
    try:
        cleanup_kwargs: dict[str, Any] = {
            "project": project,
            "context": context,
            "sky_bin": sky_bin or None,
        }
        if recover_orphan_controller or attest_no_active_jobs:
            cleanup_kwargs.update(
                recover_orphan_controller=recover_orphan_controller,
                attest_no_active_jobs=attest_no_active_jobs,
            )
        cleanup_kwargs.update(
            {
                key: value
                for key, value in {
                    "receipt": receipt,
                    "project_id": project_id,
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                }.items()
                if value
            }
        )
        result = cleanup_jobs_controller(**cleanup_kwargs)
    except (OSError, RuntimeError, ValueError) as exc:
        payload = {
            "outcome": "verification_failed",
            "resources_removed": [],
            "errors": [str(exc)],
            "commands": [],
        }
    else:
        remote_absence_verified = bool(
            getattr(result, "remote_absence_verified", result.ok)
        )
        local_metadata_cleared = bool(getattr(result, "verified", result.ok))
        payload = {
            "outcome": (
                getattr(result, "outcome", "cleaned")
                if result.ok
                else "degraded_local_metadata"
                if remote_absence_verified
                else "verification_failed"
            ),
            "resources_removed": result.resources_removed,
            "errors": result.errors,
            "commands": result.commands,
            "identity_source": getattr(result, "identity_source", "live_configuration"),
            "receipt_id": getattr(result, "receipt_id", ""),
            "verified": bool(remote_absence_verified and local_metadata_cleared),
            "overall_verified": bool(
                remote_absence_verified and local_metadata_cleared
            ),
            "local_metadata_cleared": local_metadata_cleared,
            "no_op": getattr(result, "no_op", not result.resources_removed),
            "project_alias": getattr(result, "project_alias", project),
            "project_id": getattr(result, "project_id", project_id),
            "cluster_id": getattr(result, "cluster_id", cluster_id),
            "context": getattr(result, "context", context),
            "remote_absence_verified": remote_absence_verified,
        }
    if payload["outcome"] in {"cleaned", "already_absent"} and (
        payload.get("project_alias") or payload.get("project_id")
    ):
        try:
            from npa.controller_ownership import clear_controller_owner

            clear_controller_owner(
                str(payload.get("project_alias") or ""),
                project_id=str(payload.get("project_id") or ""),
                cluster_id=str(payload.get("cluster_id") or ""),
                context=str(payload.get("context") or ""),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            payload["outcome"] = (
                "degraded_local_metadata"
                if payload.get("remote_absence_verified")
                else "verification_failed"
            )
            payload["errors"].append(
                "controller cleanup succeeded but the exact local ownership record "
                f"could not be cleared: {exc}"
            )
    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"identity_source: {payload.get('identity_source', 'unavailable')}")
        if payload["resources_removed"]:
            typer.echo(
                "Removed SkyPilot controller state: "
                + ", ".join(payload["resources_removed"])
            )
        elif payload["outcome"] in {
            "cleaned",
            "already_absent",
            "degraded_local_metadata",
        }:
            typer.echo("SkyPilot jobs controller is already absent; nothing to remove.")
        else:
            typer.echo("SkyPilot controller state could not be verified; nothing was removed.")
        for error in payload["errors"]:
            typer.echo(f"Controller cleanup warning: {error}", err=True)
    if payload["outcome"] == "verification_failed":
        raise typer.Exit(code=2)


@app.command("bind-controller")
def bind_controller_cmd(
    project: str = typer.Option("", "--project", help="Exact NPA project alias."),
    context: str = typer.Option(..., "--context", help="Exact NPA Kubernetes context."),
    rebind: bool = typer.Option(
        False,
        "--rebind",
        help="Replace a different owner only after the managed-job queue is terminal.",
    ),
    sky_bin: str = typer.Option("", "--sky-bin", help="Pinned SkyPilot executable."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Bind the shared jobs controller to one immutable project/cluster identity."""

    from npa.controller_ownership import (
        ClusterOwnerIdentityMismatchError,
        bind_controller_owner,
        resolve_controller_candidate,
        verify_live_controller_candidate,
    )

    try:
        candidate = verify_live_controller_candidate(
            resolve_controller_candidate(project, context)
        )
        if rebind:
            from npa.orchestration.skypilot.cleanup import (
                NONTERMINAL_JOB_STATUSES,
                _all_jobs,
                _job_statuses,
            )

            snapshot = _all_jobs(
                isolated_config_dir=None,
                config_path=None,
                sky_bin=sky_bin or None,
            )
            active = sorted(
                job_id
                for job_id, status in _job_statuses(snapshot.jobs).items()
                if status in NONTERMINAL_JOB_STATUSES
            )
            if active:
                raise ClusterOwnerIdentityMismatchError(
                    "Controller rebind refused while managed jobs are non-terminal: "
                    + ", ".join(active)
                )
        owner = bind_controller_owner(candidate, allow_rebind=rebind)
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
        return
    payload = owner.to_dict()
    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"controller_owner: {owner.project_alias}/{owner.context}/{owner.cluster_id}")


@app.command("verify")
def verify_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help=(
            "Kubeconfig to verify against. Sets KUBECONFIG for `sky check` so "
            "verify does not silently run against an ambient/empty context and "
            "report a misleading 403 anonymous 'missing context'."
        ),
    ),
    cluster: str | None = typer.Option(
        None,
        "--cluster",
        help=(
            "NPA cluster name. Resolves ~/.npa/clusters/<name>/kubeconfig when "
            "--kubeconfig is not given."
        ),
    ),
    controller_backend: str | None = typer.Option(
        None,
        "--controller-backend",
        help=(
            "Controller backend: kubernetes (Nebius profile optional) or nebius "
            "(required). Defaults to kubernetes without making a bare legacy "
            "runtime check require cluster setup."
        ),
    ),
    output_format: str = typer.Option(
        "text",
        "--output-format",
        help="Output format: text or json.",
    ),
) -> None:
    """Run `sky check` against the isolated SkyPilot runtime."""

    state = inspect_venv(_resolve_venv_path(path))
    if not state.installed:
        detail = f"found version {state.version}" if state.version else "sky binary missing or not executable"
        _fail(f"SkyPilot {SKYPILOT_VERSION} is not ready in {state.path}: {detail}. Run `npa skypilot bootstrap`.")
        return

    if not state.kubernetes_compatible:
        _fail(kubernetes_client_remedy(state.kubernetes_version))
        return

    backend_was_explicit = controller_backend is not None
    backend = str(controller_backend or "kubernetes").strip().lower()
    if backend not in {"kubernetes", "nebius"}:
        _fail("--controller-backend must be kubernetes or nebius")
        return
    if output_format not in {"text", "json"}:
        _fail("--output-format must be text or json")
        return
    check_env, exact_context = _verify_kube_env(
        kubeconfig=kubeconfig, cluster=cluster
    )
    kubernetes_required = backend == "kubernetes" and bool(
        backend_was_explicit or kubeconfig is not None or cluster
    )
    check_cmd = [str(state.sky_bin), "check"]
    if backend == "kubernetes" and exact_context:
        # A pre-existing SkyPilot config can restrict allowed_contexts to other
        # clusters.  Verify the kubeconfig the operator explicitly selected,
        # rather than silently checking that stale allowlist and returning 0
        # with Kubernetes disabled.
        allowed_contexts = json.dumps([exact_context], separators=(",", ":"))
        check_cmd.extend(
            ["--config", f"kubernetes.allowed_contexts={allowed_contexts}", "kubernetes"]
        )
    result = _run_observable(
        check_cmd,
        label="SkyPilot verification",
        env=check_env,
        emit_progress=output_format != "json",
    )
    combined_lines = [
        line
        for line in "\n".join((result.stdout or "", result.stderr or "")).splitlines()
        if line.strip()
    ]
    profile_failure = any("unable to create nebius profile" in line.lower() for line in combined_lines)
    required = backend == "nebius"
    plain_output = re.sub(
        r"\x1b\[[0-?]*[ -/]*[@-~]", "", "\n".join(combined_lines)
    )
    kubernetes_enabled = bool(
        re.search(r"\bKubernetes:\s+enabled\b", plain_output, flags=re.IGNORECASE)
    )
    ok = (
        result.returncode == 0
        and not (profile_failure and required)
        and (not kubernetes_required or kubernetes_enabled)
    )
    if profile_failure and not required:
        profile_status = "skipped_not_required"
        detail = "Nebius profile skipped; not required for Kubernetes-controller mode"
    elif profile_failure:
        profile_status = "failed_required"
        detail = "Nebius profile creation failed and is required for Nebius-controller mode"
    else:
        profile_status = "available_or_not_reported"
        detail = "Nebius profile failure was not reported"
    filtered = [
        line
        for line in combined_lines
        if "unable to create nebius profile" not in line.lower()
        and not (profile_failure and required and "setup completed" in line.lower())
    ]
    payload = {
        "status": "ok" if ok else "failed",
        "controller_backend": backend,
        "kubernetes_required": kubernetes_required,
        "kubernetes_enabled": kubernetes_enabled,
        "nebius_profile": profile_status,
        "nebius_profile_detail": detail,
        "sky_check_returncode": result.returncode,
        "sky_check_output": filtered,
    }
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {payload['status']}")
        typer.echo(f"controller_backend: {backend}")
        typer.echo(f"nebius_profile: {profile_status} ({detail})")
        for line in filtered:
            typer.echo(line)
    if not ok:
        raise typer.Exit(result.returncode or 1)


def _verify_kube_env(
    *, kubeconfig: Path | None, cluster: str | None
) -> tuple[dict[str, str] | None, str]:
    """Build the env for `sky check`, pinning KUBECONFIG when known.

    Without an explicit kubeconfig/cluster the behavior is unchanged (inherits
    the ambient environment).
    """

    resolved = kubeconfig
    if resolved is None and cluster:
        from npa.cluster.state import kubeconfig_file

        resolved = kubeconfig_file(cluster)
    if resolved is None:
        return None, str(cluster or "").strip()
    resolved = resolved.expanduser()
    if not resolved.exists():
        _fail(
            f"Kubeconfig not found: {resolved}. Run `npa cluster up`/`deploy` first "
            "or pass an explicit --kubeconfig."
        )
    env = os.environ.copy()
    env["KUBECONFIG"] = str(resolved)
    exact_context = str(cluster or "").strip()
    if not exact_context:
        try:
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            payload = {}
        if isinstance(payload, dict):
            exact_context = str(payload.get("current-context") or "").strip()
    return env, exact_context


def bootstrap_skypilot(
    *,
    venv_path: Path | str | None = None,
    python_bin: str | os.PathLike[str] | None = None,
    package_spec: str = SKYPILOT_PACKAGE,
    expected_version: str = SKYPILOT_VERSION,
    extras: tuple[str, ...] = SKYPILOT_EXTRAS,
) -> BootstrapResult:
    """Create or upgrade one isolated environment as an atomic transaction.

    Package installation never runs in the active environment.  A complete
    sibling is built and self-checked first, then only the exact environment
    directory is exchanged under an owner-only lock.  The parent NPA state tree
    is never renamed, recreated, or recursively removed.
    """

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("bootstrap_skypilot")

    path = _resolve_venv_path(venv_path)
    _validate_managed_venv_path(path)
    lock_path = path.with_name(f".{path.name}{BOOTSTRAP_LOCK_SUFFIX}")
    with _BootstrapLock(lock_path):
        _recover_bootstrap_exchange(path)
        state = inspect_venv(path)
        if state.installed and state.kubernetes_compatible:
            _write_marker(
                state,
                package_spec=package_spec,
                expected_version=expected_version,
                extras=extras,
                reused=True,
            )
            return BootstrapResult(
                path=state.path,
                sky_bin=state.sky_bin,
                installed=True,
                reused=True,
                marker_path=state.marker_path,
            )
        if state.exists and not state.has_python:
            raise SkyPilotBootstrapError(
                f"Path collision: {path} exists but is not an NPA-managed Python "
                "virtualenv. Choose a different --path; NPA will not replace it."
            )

        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
        )
        staging.chmod(0o700)
        try:
            _create_venv(staging, python_bin)
            staged = inspect_venv(staging)
            _ensure_pip(staged)
            _install_package(staged, package_spec)
            staged = inspect_venv(staging)
            _validate_staged_runtime(staged, expected_version=expected_version)
            _relocate_staged_scripts(staging, path)
            _activate_staged_runtime(path, staging)
        except BaseException:
            # A failed stage is diagnostic evidence.  It is owner-only and named
            # in the raised error/parent; the previous active runtime is intact.
            raise
        final = inspect_venv(path)
        _validate_staged_runtime(final, expected_version=expected_version)
        _write_marker(
            final,
            package_spec=package_spec,
            expected_version=expected_version,
            extras=extras,
            reused=False,
        )
        return BootstrapResult(
            path=final.path,
            sky_bin=final.sky_bin,
            installed=True,
            reused=False,
            marker_path=final.marker_path,
        )


def _validate_managed_venv_path(path: Path) -> None:
    """Reject broad, symlinked, or unsafe bootstrap/delete targets."""

    _reject_npa_environment(path)
    if path == Path(path.anchor) or path == path.parent:
        raise SkyPilotBootstrapError(f"Refusing broad SkyPilot environment path: {path}")
    if path.name in {"", ".", "..", ".npa"}:
        raise SkyPilotBootstrapError(f"Refusing parent NPA state path: {path}")
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        if current.is_symlink():
            raise SkyPilotBootstrapError(
                f"Refusing SkyPilot environment path containing a symlink: {current}"
            )
    if raw.exists() and not raw.is_dir():
        raise SkyPilotBootstrapError(
            f"Path collision: {raw} exists and is not a directory. Choose another --path."
        )


def _bootstrap_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f".{path.name}.previous"),
        path.with_name(f".{path.name}{BOOTSTRAP_JOURNAL_SUFFIX}"),
    )


def _write_bootstrap_journal(journal: Path, payload: dict[str, str]) -> None:
    tmp = journal.with_name(f".{journal.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": BOOTSTRAP_SCHEMA_VERSION, **payload}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, journal)
    finally:
        if tmp.exists():
            tmp.unlink()


def _recover_bootstrap_exchange(path: Path) -> None:
    previous, journal = _bootstrap_paths(path)
    if not journal.exists():
        if previous.exists():
            raise SkyPilotBootstrapError(
                f"Unexpected prior-runtime directory {previous}; refusing to guess ownership."
            )
        return
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkyPilotBootstrapError(
            f"Bootstrap recovery journal is unreadable: {journal}"
        ) from exc
    if payload.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        raise SkyPilotBootstrapError(f"Unsupported bootstrap journal: {journal}")
    if not path.exists() and previous.is_dir() and not previous.is_symlink():
        os.replace(previous, path)
    elif path.exists() and previous.is_dir() and not previous.is_symlink():
        # Activation completed before interruption. Keep the new runtime and
        # retain no duplicate active environment.
        shutil.rmtree(previous)
    journal.unlink()


def _activate_staged_runtime(path: Path, staging: Path) -> None:
    previous, journal = _bootstrap_paths(path)
    if previous.exists() or previous.is_symlink():
        raise SkyPilotBootstrapError(f"Refusing occupied bootstrap backup path: {previous}")
    _write_bootstrap_journal(
        journal,
        {"target": str(path), "staging": staging.name, "previous": previous.name},
    )
    moved_previous = False
    try:
        if path.exists():
            os.replace(path, previous)
            moved_previous = True
        os.replace(staging, path)
    except BaseException:
        if moved_previous and not path.exists() and previous.exists():
            os.replace(previous, path)
        raise
    else:
        if previous.exists():
            shutil.rmtree(previous)
        journal.unlink(missing_ok=True)


def _relocate_staged_scripts(staging: Path, target: Path) -> None:
    """Rewrite venv console-script shebangs before the atomic directory swap."""

    source = os.fsencode(str(staging))
    destination = os.fsencode(str(target))
    bin_dir = staging / ("Scripts" if os.name == "nt" else "bin")
    for entry in bin_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            body = entry.read_bytes()
        except OSError:
            continue
        first_line, separator, remainder = body.partition(b"\n")
        if not first_line.startswith(b"#!") or source not in first_line:
            continue
        entry.write_bytes(first_line.replace(source, destination) + separator + remainder)


def _validate_staged_runtime(state: VenvState, *, expected_version: str) -> None:
    problems: list[str] = []
    if state.version != expected_version:
        problems.append(f"expected SkyPilot {expected_version}, got {state.version or 'unknown'}")
    if not state.has_sky:
        problems.append("sky executable is missing")
    if not state.importable:
        problems.append("import sky failed")
    if not state.kubernetes_version:
        problems.append("kubernetes client import/version check failed")
    elif not state.kubernetes_compatible:
        problems.append(kubernetes_client_remedy(state.kubernetes_version))
    if problems:
        raise SkyPilotBootstrapError(
            f"Staged SkyPilot runtime self-check failed in {state.path}: "
            + "; ".join(problems)
            + ". The previous runtime was preserved."
        )


def inspect_venv(path: Path | str) -> VenvState:
    resolved = Path(path).expanduser().resolve(strict=False)
    bin_dir = _venv_bin_dir(resolved)
    python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
    pip_bin = bin_dir / ("pip.exe" if os.name == "nt" else "pip")
    sky_bin = bin_dir / ("sky.exe" if os.name == "nt" else "sky")
    has_python = _is_executable(python_bin)
    has_pip = _is_executable(pip_bin)
    has_sky = _is_executable(sky_bin)
    version = _sky_version(sky_bin) if has_sky else None
    importable = _sky_importable(python_bin) if has_python else False
    kubernetes_version = _kubernetes_client_version(python_bin) if has_python else None
    return VenvState(
        path=resolved,
        python_bin=python_bin,
        pip_bin=pip_bin,
        sky_bin=sky_bin,
        exists=resolved.exists(),
        has_python=has_python,
        has_pip=has_pip,
        has_sky=has_sky,
        version=version,
        importable=importable,
        marker_path=resolved / MARKER_FILE,
        kubernetes_version=kubernetes_version,
    )


def _resolve_venv_path(path: Path | str | None) -> Path:
    value = path or os.environ.get(VENV_PATH_ENV) or DEFAULT_VENV_PATH
    # Do not resolve symlinks here: bootstrap/delete validation must observe and
    # reject every symlink component before any filesystem mutation.
    return Path(os.path.abspath(Path(value).expanduser()))


def _reject_npa_environment(path: Path) -> None:
    prefixes = [Path(sys.prefix).expanduser().resolve(strict=False)]
    if os.environ.get("VIRTUAL_ENV"):
        prefixes.append(Path(os.environ["VIRTUAL_ENV"]).expanduser().resolve(strict=False))
    for prefix in prefixes:
        if path == prefix or prefix in path.parents:
            raise SkyPilotBootstrapError(
                f"Refusing to install SkyPilot into the NPA Python environment: {path}. "
                "Suggested action: use the default ~/.npa/skypilot-venv path or pass a separate --path."
            )


def _detect_python_version(executable: str | os.PathLike[str]) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` of *executable*, or None when undeterminable."""
    result = _run_no_raise(
        [os.fspath(executable), "-c", "import sys;print(sys.version_info[0], sys.version_info[1])"]
    )
    if result.returncode != 0:
        return None
    try:
        major, minor = (int(part) for part in (result.stdout or "").split()[:2])
    except (ValueError, IndexError):
        return None
    return (major, minor)


def _is_supported_python(version: tuple[int, int] | None) -> bool:
    return version is not None and SKYPILOT_MIN_PYTHON <= version <= SKYPILOT_MAX_PYTHON


def _resolve_python_bin(python_bin: str | os.PathLike[str] | None) -> str:
    """Resolve the interpreter used to create the SkyPilot venv.

    An explicit ``--python`` / ``NPA_SKYPILOT_PYTHON`` is honored but rejected
    when its version is known-unsupported. Otherwise the current interpreter is
    used when supported; if it is too new/old (e.g. Python 3.14, which breaks the
    kubernetes client), a supported ``python3.x`` on PATH is auto-selected. A
    version we cannot determine is passed through so the normal venv-creation
    error still surfaces.
    """
    explicit = python_bin or os.environ.get(PYTHON_ENV)
    if explicit:
        version = _detect_python_version(explicit)
        if version is not None and not _is_supported_python(version):
            raise SkyPilotBootstrapError(
                f"Python {version[0]}.{version[1]} ({explicit}) is outside SkyPilot "
                f"{SKYPILOT_VERSION}'s supported range ({_supported_python_range_str()}); "
                "its kubernetes/ray dependencies fail to build/import on newer "
                "versions. Suggested action: pass --python for a supported "
                "interpreter (e.g. python3.12)."
            )
        return os.fspath(explicit)

    current = _detect_python_version(sys.executable)
    if _is_supported_python(current) or current is None:
        return sys.executable

    for name in _PREFERRED_PYTHON_BINS:
        candidate = shutil.which(name)
        if candidate and _is_supported_python(_detect_python_version(candidate)):
            return candidate

    raise SkyPilotBootstrapError(
        f"The default Python ({current[0]}.{current[1]}) is outside SkyPilot "
        f"{SKYPILOT_VERSION}'s supported range ({_supported_python_range_str()}) — "
        "its kubernetes client fails to import there — and no supported "
        f"python3.x was found on PATH. Suggested action: install a supported "
        "Python (e.g. python3.12) or pass --python <interpreter>."
    )


def _create_venv(path: Path, python_bin: str | os.PathLike[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    executable = _resolve_python_bin(python_bin)
    result = _run_no_raise([executable, "-m", "venv", str(path)])
    if result.returncode != 0:
        detail = _combined_output(result) or "no output"
        raise SkyPilotBootstrapError(
            f"Unable to create SkyPilot venv with {executable}: {detail}. "
            "Suggested action: install Python with venv support or pass --python."
        )


def _ensure_pip(state: VenvState) -> None:
    if not state.has_python:
        raise SkyPilotBootstrapError(
            f"Missing Python in SkyPilot venv: {state.python_bin}. "
            "Suggested action: remove the venv and rerun bootstrap."
        )
    result = _run_no_raise([str(state.python_bin), "-m", "pip", "--version"])
    if result.returncode == 0:
        return
    ensurepip = _run_no_raise([str(state.python_bin), "-m", "ensurepip", "--upgrade"])
    if ensurepip.returncode == 0:
        return
    detail = _combined_output(ensurepip) or _combined_output(result) or "no output"
    raise SkyPilotBootstrapError(
        f"Missing pip in SkyPilot venv {state.path}: {detail}. "
        "Suggested action: install Python with ensurepip support or recreate the venv."
    )


def _install_package(state: VenvState, package_spec: str) -> None:
    if not CONSTRAINTS_FILE.is_file():
        raise SkyPilotBootstrapError(
            f"SkyPilot compatibility constraints are missing: {CONSTRAINTS_FILE}"
        )
    result = _run_observable(
        [
            str(state.python_bin),
            "-m",
            "pip",
            "install",
            "--constraint",
            str(CONSTRAINTS_FILE),
            package_spec,
        ],
        label="SkyPilot bootstrap package install",
    )
    if result.returncode == 0:
        # SkyPilot 0.12.2 declares click<8.2, but pip can still resolve a newer
        # Click that breaks `sky launch --docker` flag parsing (backend_name=False).
        # Kubernetes client 36 also changed model type strings from
        # ``dict(str, str)`` to ``dict[str, str]``; SkyPilot 0.12.2 tries to
        # import the latter as a model module while validating pod_config and
        # fails before creating any Kubernetes workload. Re-pin both runtime
        # edges after install so bootstrap stays launchable.
        _pin_skypilot_click(state)
        _pin_skypilot_kubernetes(inspect_venv(state.path))
        return
    detail = _combined_output(result) or "no output"
    if _looks_like_network_failure(detail):
        raise SkyPilotBootstrapError(
            f"Network failure while installing {package_spec}: {detail}. "
            "Suggested action: verify package index connectivity and rerun bootstrap."
        )
    raise SkyPilotBootstrapError(
        f"pip failed while installing {package_spec}: {detail}. "
        "Suggested action: inspect the pip error above, fix the environment, and rerun bootstrap."
    )


def kubernetes_client_supported(version: str | None) -> bool:
    """Whether ``version`` of the kubernetes client is usable by SkyPilot.

    Unknown or malformed versions are incompatible: workflow mutation must fail
    closed when the isolated runtime cannot prove its dependency contract.
    """

    if not version:
        return False
    match = re.match(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    if major >= int(KUBERNETES_CLIENT_MAX_EXCLUSIVE):
        return False
    return (major, minor, patch) != (32, 0, 0)


def kubernetes_client_remedy(version: str | None) -> str:
    """One-line remedy for an incompatible kubernetes client in the SkyPilot venv."""

    return (
        f"SkyPilot's isolated venv has kubernetes client {version or 'unknown'}, which "
        "breaks pod_config validation ("
        "\"Invalid pod_config ... No module named 'kubernetes.client.models.dict[str, str]'\") "
        "and makes the managed-jobs controller retry forever. "
        f"Suggested action: rerun `npa skypilot bootstrap`, or pin manually with "
        f"`pip install '{KUBERNETES_CLIENT_SPEC}'` in the SkyPilot venv."
    )


def _pin_skypilot_kubernetes(state: VenvState) -> None:
    """Keep the kubernetes client below the version that breaks pod_config validation."""

    if state.kubernetes_compatible and state.kubernetes_version:
        return
    result = _run_no_raise(
        [str(state.python_bin), "-m", "pip", "install", KUBERNETES_CLIENT_SPEC]
    )
    if result.returncode == 0:
        return
    detail = _combined_output(result) or "no output"
    if _looks_like_network_failure(detail):
        raise SkyPilotBootstrapError(
            f"Network failure while pinning the kubernetes client for SkyPilot: {detail}. "
            "Suggested action: verify package index connectivity and rerun bootstrap."
        )
    raise SkyPilotBootstrapError(
        f"pip failed while pinning the kubernetes client for SkyPilot: {detail}. "
        "Suggested action: inspect the pip error above, fix the environment, and rerun bootstrap."
    )


def _pin_skypilot_click(state: VenvState) -> None:
    """Keep Click inside SkyPilot's declared range after bootstrap installs."""

    result = _run_no_raise(
        [str(state.python_bin), "-m", "pip", "install", "click>=8.1,<8.2"]
    )
    if result.returncode == 0:
        return
    detail = _combined_output(result) or "no output"
    if _looks_like_network_failure(detail):
        raise SkyPilotBootstrapError(
            f"Network failure while pinning click for SkyPilot: {detail}. "
            "Suggested action: verify package index connectivity and rerun bootstrap."
        )
    raise SkyPilotBootstrapError(
        f"pip failed while pinning click for SkyPilot: {detail}. "
        "Suggested action: inspect the pip error above, fix the environment, and rerun bootstrap."
    )


def _write_marker(
    state: VenvState,
    *,
    package_spec: str,
    expected_version: str,
    extras: tuple[str, ...],
    reused: bool,
) -> None:
    payload = {
        "version": expected_version,
        "install_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "extras": list(extras),
        "package": package_spec,
        "sky_bin": str(state.sky_bin),
        "reused_existing_venv": reused,
        "kubernetes_client": state.kubernetes_version,
        "kubernetes_client_spec": KUBERNETES_CLIENT_SPEC,
    }
    state.marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sky_version(sky_bin: Path) -> str | None:
    result = _run_no_raise([str(sky_bin), "--version"])
    output = _combined_output(result)
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    return match.group(1) if match else None


def _sky_importable(python_bin: Path) -> bool:
    result = _run_no_raise([str(python_bin), "-c", "import sky"])
    return result.returncode == 0


def _kubernetes_client_version(python_bin: Path) -> str | None:
    result = _run_no_raise(
        [str(python_bin), "-c", "import kubernetes; print(kubernetes.__version__)"]
    )
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip().splitlines()
    return value[0].strip() if value else None


def _run_no_raise(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=env
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))


def _run_observable(
    cmd: list[str],
    *,
    label: str,
    env: dict[str, str] | None = None,
    emit_progress: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a potentially long subprocess with sanitized periodic stderr progress."""

    from npa.progress import WaitProgress

    progress = WaitProgress(
        label,
        emit=(
            (lambda message: typer.echo(message, err=True))
            if emit_progress
            else (lambda _message: None)
        ),
    )
    progress.start("attempt=1 state=starting")
    stop = threading.Event()

    def report_wait() -> None:
        while not stop.wait(progress.interval):
            progress.tick("attempt=1 state=running")

    reporter = threading.Thread(
        target=report_wait, name="npa-skypilot-cli-progress", daemon=True
    )
    reporter.start()
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = _run_no_raise(cmd, env=env)
        return result
    finally:
        stop.set()
        reporter.join(timeout=1)
        progress.finish(
            "completed" if result is not None and result.returncode == 0 else "failed",
            "attempt=1",
        )


def _summarize_completed_process(result: subprocess.CompletedProcess[str]) -> str:
    status = "passed" if result.returncode == 0 else f"failed ({result.returncode})"
    output = _combined_output(result).splitlines()
    first_line = output[0] if output else "no output"
    return f"{status}: {first_line}"


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return redact_text(
        "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part and part.strip()
        )
    )


def _looks_like_network_failure(detail: str) -> bool:
    lowered = detail.lower()
    needles = (
        "temporary failure",
        "name resolution",
        "connection",
        "network",
        "timed out",
        "timeout",
        "unreachable",
    )
    return any(needle in lowered for needle in needles)


def _format_marker_age(marker_path: Path) -> str:
    if not marker_path.exists():
        return "missing"
    age_seconds = max(0, int(time.time() - marker_path.stat().st_mtime))
    return f"{age_seconds}s"


def _venv_bin_dir(path: Path) -> Path:
    return path / ("Scripts" if os.name == "nt" else "bin")


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)

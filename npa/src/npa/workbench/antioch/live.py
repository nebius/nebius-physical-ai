"""Supported Antioch service + tmux supervision for the OpenPI live example."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .runtime import ensure_runtime
from .vendor_cli import AntiochCli, AntiochCliError

REMOTE_CLIENT_ROOT = "/tmp/npa-live-client-current"
REMOTE_CLIENT_STAGING_PREFIX = "/tmp/npa-live-client-generation-"
REMOTE_CLIENT_UPLOAD_PREFIX = "/tmp/npa-live-client-upload-"
UPSTREAM_BUNDLE_FILES = ("ca.crt", "api-key", "endpoint.json")
RELAY_BUNDLE_FILES = (
    "relay-ca.crt",
    "relay-server.crt",
    "relay-server.key",
    "relay-api-key",
)
REQUIRED_BUNDLE_FILES = UPSTREAM_BUNDLE_FILES + RELAY_BUNDLE_FILES
RELAY_TARGET_PORT = 8_444
RELAY_PUBLISHED_PORT = 18_444


class AntiochLiveError(RuntimeError):
    """The continuing live-session controller failed closed."""


def live_state_root() -> Path:
    configured = os.environ.get("NPA_ANTIOCH_LIVE_STATE_DIR", "").strip()
    if configured:
        return Path(configured)
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    return (Path(state_home) if state_home else Path.home() / ".local/state") / (
        "npa/antioch-live"
    )


def _session_name(project_id: str) -> str:
    digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:12]
    return f"npa-antioch-live-{digest}"


def _tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["tmux", *args],
            text=True,
            capture_output=True,
            check=check,
        )
    except FileNotFoundError as exc:
        raise AntiochLiveError(
            "tmux is required for continuing Antioch sessions"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AntiochLiveError("tmux live-session operation failed") from exc


def _session_running(name: str) -> bool:
    return _tmux("has-session", "-t", name, check=False).returncode == 0


def _window_running(session: str, window: str) -> bool:
    return _tmux("list-panes", "-t", f"{session}:{window}", check=False).returncode == 0


def _state_path(project_id: str) -> Path:
    return live_state_root() / _session_name(project_id) / "state.json"


def _read_state(project_id: str) -> dict[str, Any]:
    path = _state_path(project_id)
    if not path.is_file() or path.is_symlink():
        raise AntiochLiveError("no exact live-session state exists for this project")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("project_id") != project_id:
        raise AntiochLiveError("live-session state identity is malformed")
    return value


def _validate_upstream_bundle(bundle: Path) -> None:
    for name in UPSTREAM_BUNDLE_FILES:
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            raise AntiochLiveError(f"client bundle is missing regular file {name!r}")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise AntiochLiveError(
                f"client bundle file {name!r} must not be group/world accessible"
            )
    endpoint = json.loads((bundle / "endpoint.json").read_text(encoding="utf-8"))
    if not isinstance(endpoint, dict) or endpoint.get("scheme") != "wss":
        raise AntiochLiveError("client bundle endpoint must use wss")
    if len((bundle / "api-key").read_text(encoding="utf-8").strip()) < 32:
        raise AntiochLiveError("client bundle API key is malformed")


def _validate_bundle(bundle: Path) -> None:
    _validate_upstream_bundle(bundle)
    for name in RELAY_BUNDLE_FILES:
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            raise AntiochLiveError(f"client bundle is missing regular file {name!r}")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise AntiochLiveError(
                f"client bundle file {name!r} must not be group/world accessible"
            )
    if len((bundle / "relay-api-key").read_text(encoding="utf-8").strip()) < 32:
        raise AntiochLiveError("relay API key is malformed")


def _relay_certificate() -> tuple[bytes, bytes, bytes]:
    """Create a run-local CA and localhost certificate for the declared port."""

    now = dt.datetime.now(dt.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65_537, key_size=3_072)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "NPA Antioch live relay CA")]
    )
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    server_key = rsa.generate_private_key(public_exponent=65_537, key_size=3_072)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "NPA Antioch live relay")]
    )
    server = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return (
        ca.public_bytes(serialization.Encoding.PEM),
        server.public_bytes(serialization.Encoding.PEM),
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


def _write_private(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, content)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _prepare_runtime_bundle(source: Path, destination: Path) -> None:
    _validate_upstream_bundle(source)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(destination, 0o700)
    for name in UPSTREAM_BUNDLE_FILES:
        _write_private(destination / name, (source / name).read_bytes())
    ca, certificate, private_key = _relay_certificate()
    for name, content in {
        "relay-ca.crt": ca,
        "relay-server.crt": certificate,
        "relay-server.key": private_key,
        "relay-api-key": (secrets.token_urlsafe(48) + "\n").encode(),
    }.items():
        _write_private(destination / name, content)
    _validate_bundle(destination)


def _prepare_copyable_bundle(source: Path, destination: Path) -> None:
    """Make transport-readable copies protected by an owner-only parent.

    Antioch's supported service copy preserves file modes but assigns files to
    the machine-side copy user.  A 0644 transport copy is therefore required
    for the uid-1000 service to read it.  The enclosing directory remains 0700,
    and the service immediately installs a separate uid-owned 0600 generation.
    """

    parent = destination.parent
    if parent.stat().st_mode & 0o077:
        raise AntiochLiveError("bundle transport parent must be owner-only")
    destination.mkdir(mode=0o700)
    os.chmod(destination, 0o700)
    for name in REQUIRED_BUNDLE_FILES:
        target = destination / name
        target.write_bytes((source / name).read_bytes())
        os.chmod(target, 0o644)


def _stage_private_bundle(
    cli: AntiochCli,
    *,
    runtime: Path,
    client_bundle: Path,
    attempts: int = 12,
) -> None:
    """Stage through supported service commands across container recreation."""

    last_error: Exception | None = None
    local_upload = runtime / f".bundle-upload-{uuid.uuid4().hex}"
    _prepare_copyable_bundle(client_bundle, local_upload)
    try:
        for attempt in range(attempts):
            upload = f"{REMOTE_CLIENT_UPLOAD_PREFIX}{uuid.uuid4().hex}"
            staging = f"{REMOTE_CLIENT_STAGING_PREFIX}{uuid.uuid4().hex}"
            upload_files = [f"{upload}/{name}" for name in REQUIRED_BUNDLE_FILES]
            remote_files = [f"{staging}/{name}" for name in REQUIRED_BUNDLE_FILES]
            try:
                cli.services_exec(
                    runtime,
                    "sim",
                    ["install", "-d", "-m", "0700", upload, staging],
                )
                for name in REQUIRED_BUNDLE_FILES:
                    cli.services_copy(
                        runtime,
                        local_upload / name,
                        f"sim:{upload}/{name}",
                    )
                for source, destination in zip(
                    upload_files, remote_files, strict=True
                ):
                    cli.services_exec(
                        runtime,
                        "sim",
                        ["install", "-m", "0600", source, destination],
                    )
                cli.services_exec(
                    runtime,
                    "sim",
                    [
                        "/bin/sh",
                        "-lc",
                        "test " + " -a ".join(f"-r {path}" for path in remote_files),
                    ],
                )
                cli.services_exec(runtime, "sim", ["rm", "-rf", upload])
                cli.services_exec(
                    runtime,
                    "sim",
                    [
                        "/bin/sh",
                        "-lc",
                        f"ln -s {shlex.quote(staging)} {REMOTE_CLIENT_ROOT}.new && "
                        f"mv -Tf {REMOTE_CLIENT_ROOT}.new {REMOTE_CLIENT_ROOT}",
                    ],
                )
                return
            except AntiochCliError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(5)
    finally:
        shutil.rmtree(local_upload, ignore_errors=True)
    raise AntiochLiveError(
        "the private live bundle did not survive service startup"
    ) from last_error


def _stage_runtime_source(
    cli: AntiochCli, *, runtime: Path, attempts: int = 12
) -> None:
    """Copy reviewed source, retrying across a service-container recreation."""

    source = runtime / "src"
    names = ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py")
    for name in names:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise AntiochLiveError(f"live runtime source {name!r} is unavailable")
    last_error: Exception | None = None
    for attempt in range(attempts):
        staging = f"/tmp/npa-live-source-{uuid.uuid4().hex}"
        try:
            cli.services_exec(
                runtime,
                "sim",
                ["install", "-d", "-m", "0700", staging, "/workspace/project/src"],
            )
            for name in names:
                cli.services_copy(
                    runtime,
                    source / name,
                    f"sim:{staging}/{name}",
                )
            for name in names:
                staged = f"{staging}/{name}"
                destination = f"/workspace/project/src/{name}"
                cli.services_exec(
                    runtime, "sim", ["install", "-m", "0644", staged, destination]
                )
                observed = cli.services_exec(runtime, "sim", ["sha256sum", destination])
                expected = hashlib.sha256((source / name).read_bytes()).hexdigest()
                if observed.split(maxsplit=1)[0] != expected:
                    raise AntiochLiveError(
                        f"live runtime source {name!r} failed verification"
                    )
            return
        except (AntiochCliError, AntiochLiveError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(5)
    raise AntiochLiveError(
        "the reviewed live source did not survive service startup"
    ) from last_error


def _cancel_remote_live_runs(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    scenario: str = "openpi_droid_live",
    attempts: int = 60,
) -> int:
    """Cancel only this project's exact live scenario before service teardown."""

    cancelled: set[str] = set()
    terminal: set[str] = set()
    live_phases = {"queued", "booting", "running"}
    terminal_stream_states = {"failed", "stopped", "idle"}
    stable_absence = 0
    for attempt in range(attempts):
        rows = cli.list_for_project(runtime, kind="scenario", project_id=project_id)
        candidates: dict[str, dict[str, Any]] = {
            str(row["scenario_run_id"]): row
            for row in rows
            if row.get("scenario") == scenario
            and row.get("phase") in live_phases
            and row.get("scenario_run_id")
            and str(row["scenario_run_id"]) not in terminal
        }
        machine = cli.machine_status(runtime, project_id=project_id)
        # Import locally to keep the legacy live module independent at import
        # time while sharing the exact supported Rome/direct-daemon contract.
        from .live_reconcile import _daemon_runtime_snapshot

        stream = _daemon_runtime_snapshot(
            machine, require_session_owner=False
        )["stream"]
        stream_run_id = str(stream.get("scenario_run_id") or "")
        stream_state = str(stream.get("state") or "").lower()
        stream_live = bool(stream_run_id) and stream_state not in terminal_stream_states
        if stream_live and stream_run_id not in candidates:
            if stream_run_id in terminal:
                stream_live = False
            else:
                raise AntiochLiveError(
                    "the active stream owner was absent from this project's exact live runs"
                )
        if not candidates and not stream_live:
            stable_absence += 1
            if stable_absence >= 3:
                return len(cancelled)
        else:
            stable_absence = 0
        for remote_id in candidates:
            try:
                result = cli.cancel(runtime, kind="scenario", remote_id=remote_id)
            except AntiochCliError as exc:
                # A booting run can terminalize between the supported list and
                # cancel calls. Treat only typed absence or the vendor's exact
                # unstructured absence response as terminal; every other
                # cancellation failure still blocks service teardown.
                missing = (
                    exc.http_status == 404
                    or exc.error_type in {"not_found", "scenario_not_found"}
                    or "was not found" in str(exc).lower()
                )
                if not missing:
                    raise
                terminal.add(remote_id)
                continue
            cancelled.add(remote_id)
            if result.get("phase") == "completed" or result.get("outcome"):
                terminal.add(remote_id)
        if attempt + 1 < attempts:
            time.sleep(2)
    raise AntiochLiveError(
        "the exact live scenario remained active; refusing service teardown"
    )


def _stage_project(source: Path, destination: Path, project_id: str) -> None:
    if not (source / "antioch.yaml").is_file():
        raise AntiochLiveError("live source is not an Antioch project")
    shutil.copytree(source, destination)
    os.chmod(destination, 0o700)
    # ``antioch services cp`` preserves the local source mode while transferring
    # through the assignment's SSH user.  The service itself runs as uid 1000,
    # so owner-only files become unreadable after that supported copy boundary.
    # These three files are reviewed public source (never the private bundle),
    # and must be readable by the unprivileged service container.
    for name in ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py"):
        staged_source = destination / "src" / name
        if not staged_source.is_file() or staged_source.is_symlink():
            raise AntiochLiveError(f"live runtime source {name!r} is unavailable")
        os.chmod(staged_source, 0o644)
    manifest_path = destination / "antioch.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("id") != "replace-at-runtime":
        raise AntiochLiveError(
            "live source must retain its non-deployable project placeholder"
        )
    manifest["id"] = project_id
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    os.chmod(manifest_path, 0o600)


def _write_supervisor(
    path: Path,
    *,
    cli_path: Path,
    python_path: Path,
    client_bundle: Path,
    stop_file: Path,
    active_state_path: Path,
    scenario_timeout_seconds: int,
    scenario_name: str = "openpi_droid_live",
    owner_identity: str = "",
    session_id: str = "",
) -> None:
    if scenario_timeout_seconds < 60:
        raise AntiochLiveError(
            "the per-run renewal boundary must be at least 60 seconds"
        )
    command = shlex.join(
        [
            str(cli_path),
            "scenario",
            "run",
            "--scenario",
            scenario_name,
            "--timeout",
            str(scenario_timeout_seconds),
            "--stream",
            "--verbose",
        ]
    )
    source_names = ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py")
    source_paths = {name: path.parent / "src" / name for name in source_names}
    for name, source_path in source_paths.items():
        if not source_path.is_file() or source_path.is_symlink():
            raise AntiochLiveError(f"live runtime source {name!r} is unavailable")
    source_hashes = {
        name: hashlib.sha256(source_path.read_bytes()).hexdigest()
        for name, source_path in source_paths.items()
    }
    source_check_expression = " -a ".join(
        f'"$(sha256sum /workspace/project/src/{name} 2>/dev/null | cut -d" " -f1)" '
        f"= {digest}"
        for name, digest in source_hashes.items()
    )
    source_check = shlex.join(
        [
            str(cli_path),
            "services",
            "exec",
            "sim",
            "/bin/sh",
            "-lc",
            f"test {source_check_expression}",
        ]
    )
    source_staging = f"/tmp/npa-live-supervisor-source-{uuid.uuid4().hex}"
    source_stage_commands = [
        shlex.join(
            [
                str(cli_path),
                "services",
                "exec",
                "sim",
                "install",
                "-d",
                "-m",
                "0700",
                source_staging,
                "/workspace/project/src",
            ]
        ),
        *[
            shlex.join(
                [
                    str(cli_path),
                    "services",
                    "cp",
                    str(source_paths[name]),
                    f"sim:{source_staging}/{name}",
                    "--json",
                ]
            )
            for name in source_names
        ],
        *[
            shlex.join(
                [
                    str(cli_path),
                    "services",
                    "exec",
                    "sim",
                    "install",
                    "-m",
                    "0644",
                    f"{source_staging}/{name}",
                    f"/workspace/project/src/{name}",
                ]
            )
            for name in source_names
        ],
        source_check,
    ]
    source_stage_block = " &&\n          ".join(source_stage_commands)
    service_check = shlex.join(
        [
            str(cli_path),
            "services",
            "exec",
            "sim",
            "/bin/true",
        ]
    )
    service_rebind = shlex.join([str(cli_path), "services", "up", "--json"])
    service_rebuild = shlex.join(
        [str(cli_path), "services", "build", "--service", "sim", "--json"]
    )
    bundle_hashes = {
        name: hashlib.sha256((client_bundle / name).read_bytes()).hexdigest()
        for name in REQUIRED_BUNDLE_FILES
    }
    bundle_check_expression = " -a ".join(
        f'"$(sha256sum {REMOTE_CLIENT_ROOT}/{name} 2>/dev/null | cut -d" " -f1)" '
        f"= {digest}"
        for name, digest in bundle_hashes.items()
    )
    bundle_check = shlex.join(
        [
            str(cli_path),
            "services",
            "exec",
            "sim",
            "/bin/sh",
            "-lc",
            f"test {bundle_check_expression}",
        ]
    )
    local_bundle_upload = path.parent / f".bundle-upload-{uuid.uuid4().hex}"
    _prepare_copyable_bundle(client_bundle, local_bundle_upload)
    bundle_upload = f"{REMOTE_CLIENT_UPLOAD_PREFIX}{uuid.uuid4().hex}"
    bundle_staging = f"{REMOTE_CLIENT_STAGING_PREFIX}{uuid.uuid4().hex}"
    upload_files = [f"{bundle_upload}/{name}" for name in REQUIRED_BUNDLE_FILES]
    remote_files = [f"{bundle_staging}/{name}" for name in REQUIRED_BUNDLE_FILES]
    stage_commands = [
        shlex.join(
            [
                str(cli_path),
                "services",
                "exec",
                "sim",
                "install",
                "-d",
                "-m",
                "0700",
                bundle_upload,
                bundle_staging,
            ]
        ),
        *[
            shlex.join(
                [
                    str(cli_path),
                    "services",
                    "cp",
                    str(local_bundle_upload / name),
                    f"sim:{bundle_upload}/{name}",
                    "--json",
                ]
            )
            for name in REQUIRED_BUNDLE_FILES
        ],
        *[
            shlex.join(
                [
                    str(cli_path),
                    "services",
                    "exec",
                    "sim",
                    "install",
                    "-m",
                    "0600",
                    source,
                    destination,
                ]
            )
            for source, destination in zip(upload_files, remote_files, strict=True)
        ],
        shlex.join(
            [
                str(cli_path),
                "services",
                "exec",
                "sim",
                "rm",
                "-rf",
                bundle_upload,
            ]
        ),
        shlex.join(
            [
                str(cli_path),
                "services",
                "exec",
                "sim",
                "/bin/sh",
                "-lc",
                f"ln -s {shlex.quote(bundle_staging)} {REMOTE_CLIENT_ROOT}.new && "
                f"mv -Tf {REMOTE_CLIENT_ROOT}.new {REMOTE_CLIENT_ROOT}",
            ]
        ),
        bundle_check,
    ]
    stage_block = " &&\n      ".join(stage_commands)
    module_root = Path(__file__).resolve().parents[3]
    reconcile = shlex.join(
        [
            "env",
            f"PYTHONPATH={module_root}",
            str(python_path),
            "-m",
            "npa.workbench.antioch.live_reconcile",
            "--cli",
            str(cli_path),
            "--runtime",
            ".",
            "--stop-file",
            str(stop_file),
            "--state-path",
            str(active_state_path),
            "--scenario",
            scenario_name,
            *(
                ["--owner-identity", owner_identity]
                if owner_identity
                else []
            ),
            *(["--session-id", session_id] if session_id else []),
        ]
    )
    content = f"""#!/bin/sh
set -u
(
  while [ ! -f {shlex.quote(str(stop_file))} ]; do
    if ! {service_check} >/dev/null 2>&1; then
      if ! {service_rebind} >/dev/null 2>&1; then
        {service_rebuild} >/dev/null 2>&1 && \
          {service_rebind} >/dev/null 2>&1 || true
      fi
    fi
    if ! {bundle_check} >/dev/null 2>&1; then
      {{
        {stage_block}
      }} >/dev/null 2>&1 || true
    fi
    if ! {source_check} >/dev/null 2>&1; then
      {{
        {source_stage_block}
      }} >/dev/null 2>&1 || true
    fi
    sleep 15
  done
) &
restager=$!
while [ ! -f {shlex.quote(str(stop_file))} ]; do
  {reconcile}
  reconcile_status=$?
  if [ "$reconcile_status" -eq 0 ]; then
    if [ -f {shlex.quote(str(stop_file))} ]; then
      break
    fi
    printf 'NPA_ANTIOCH_RECONCILED_TERMINAL\n'
    sleep 5
    continue
  fi
  if [ "$reconcile_status" -ne 3 ]; then
    printf 'NPA_ANTIOCH_RECONCILE_FAILED exit_code=%s\n' "$reconcile_status"
    sleep 5
    continue
  fi
  if ! {service_check} >/dev/null 2>&1 || \
     ! {source_check} >/dev/null 2>&1 || \
     ! {bundle_check} >/dev/null 2>&1; then
    printf 'NPA_ANTIOCH_SERVICE_NOT_READY\n'
    sleep 5
    continue
  fi
  {command}
  status=$?
  if [ -f {shlex.quote(str(stop_file))} ]; then
    break
  fi
  printf 'NPA_ANTIOCH_RENEWAL exit_code=%s\n' "$status"
  sleep 5
done
kill "$restager" >/dev/null 2>&1 || true
wait "$restager" >/dev/null 2>&1 || true
"""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o700)


def _write_relay_supervisor(
    path: Path,
    *,
    python_path: Path,
    client_bundle: Path,
    stop_file: Path,
    state_path: Path,
) -> None:
    command = shlex.join(
        [
            str(python_path),
            "-m",
            "npa.workbench.antioch.relay",
            "--bundle",
            str(client_bundle),
            "--local-port",
            str(RELAY_PUBLISHED_PORT),
            "--stop-file",
            str(stop_file),
            "--state-path",
            str(state_path),
        ]
    )
    content = f"""#!/bin/sh
set -u
while [ ! -f {shlex.quote(str(stop_file))} ]; do
  {command}
  status=$?
  if [ -f {shlex.quote(str(stop_file))} ]; then
    break
  fi
  printf 'NPA_ANTIOCH_RELAY_RESTART exit_code=%s\n' "$status"
  sleep 2
done
"""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o700)


def _write_bridge_supervisor(path: Path, *, cli_path: Path, stop_file: Path) -> None:
    """Probe the service-owned bridge with finite supported exec calls."""

    command = shlex.join(
        [
            str(cli_path),
            "services",
            "exec",
            "sim",
            "/usr/local/bin/python",
            "-c",
            "import socket; s=socket.create_connection(('127.0.0.1',8444),2); s.close()",
        ]
    )
    content = f"""#!/bin/sh
set -u
while [ ! -f {shlex.quote(str(stop_file))} ]; do
  {command}
  status=$?
  if [ -f {shlex.quote(str(stop_file))} ]; then
    break
  fi
  if [ "$status" -eq 0 ]; then
    printf 'NPA_ANTIOCH_BRIDGE_HEALTHY\n'
  else
    printf 'NPA_ANTIOCH_BRIDGE_NOT_READY exit_code=%s\n' "$status"
  fi
  sleep 10
done
"""
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o700)


def start_live(
    *,
    source: Path,
    project_id: str,
    client_bundle: Path,
    scenario_timeout_seconds: int = 14_400,
) -> dict[str, Any]:
    """Start the sim service and an indefinitely renewing streamed scenario."""

    if not project_id.strip() or project_id == "replace-at-runtime":
        raise AntiochLiveError("an assigned Antioch project ID is required")
    _validate_upstream_bundle(client_bundle)
    cli_path = ensure_runtime()
    cli = AntiochCli(cli_path)
    session = _session_name(project_id)
    if _session_running(session):
        return {"status": "already-running", "session": session}

    root = live_state_root() / session
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    runtime = root / f"runtime-{uuid.uuid4().hex}"
    _stage_project(source.resolve(), runtime, project_id)
    runtime_bundle = root / f"bundle-{uuid.uuid4().hex}"
    _prepare_runtime_bundle(client_bundle.resolve(), runtime_bundle)
    stop_file = runtime / ".stop"
    supervisor = runtime / ".supervise.sh"
    relay_supervisor = runtime / ".relay-supervise.sh"
    bridge_supervisor = runtime / ".bridge-supervise.sh"
    relay_state = runtime / "relay-state.json"
    active_state = runtime / "active-run.json"
    log = runtime / "live.log"
    relay_log = runtime / "relay.log"
    bridge_log = runtime / "bridge.log"
    _write_supervisor(
        supervisor,
        cli_path=cli_path,
        python_path=Path(sys.executable),
        client_bundle=runtime_bundle,
        stop_file=stop_file,
        active_state_path=active_state,
        scenario_timeout_seconds=scenario_timeout_seconds,
        scenario_name="openpi_droid_live",
    )
    _write_relay_supervisor(
        relay_supervisor,
        python_path=Path(sys.executable),
        client_bundle=runtime_bundle,
        stop_file=stop_file,
        state_path=relay_state,
    )
    _write_bridge_supervisor(
        bridge_supervisor,
        cli_path=cli_path,
        stop_file=stop_file,
    )

    service_started = False
    try:
        cli.services_build(runtime, service="sim")
        cli.services_up(runtime)
        service_started = True
        _stage_runtime_source(cli, runtime=runtime)
        _stage_private_bundle(
            cli,
            runtime=runtime,
            client_bundle=runtime_bundle,
        )
        _tmux(
            "new-session",
            "-d",
            "-s",
            session,
            "-n",
            "scenario",
            "-c",
            str(runtime),
            str(supervisor),
        )
        _tmux(
            "pipe-pane",
            "-o",
            "-t",
            f"{session}:scenario.0",
            f"exec >> {shlex.quote(str(log))} 2>&1",
        )
        _tmux(
            "new-window",
            "-d",
            "-t",
            session,
            "-n",
            "bridge",
            "-c",
            str(runtime),
            str(bridge_supervisor),
        )
        _tmux(
            "pipe-pane",
            "-o",
            "-t",
            f"{session}:bridge.0",
            f"exec >> {shlex.quote(str(bridge_log))} 2>&1",
        )
        _tmux(
            "new-window",
            "-d",
            "-t",
            session,
            "-n",
            "relay",
            "-c",
            str(runtime),
            str(relay_supervisor),
        )
        _tmux(
            "pipe-pane",
            "-o",
            "-t",
            f"{session}:relay.0",
            f"exec >> {shlex.quote(str(relay_log))} 2>&1",
        )
        time.sleep(2)
        if not _session_running(session):
            raise AntiochLiveError("the Antioch live supervisor exited during startup")
    except Exception:
        if _session_running(session):
            _tmux("kill-session", "-t", session, check=False)
        if service_started:
            try:
                cli.services_down(runtime)
            except AntiochCliError:
                pass
        raise

    state = {
        "schema": "npa.workbench.antioch-live.v1",
        "session": session,
        "runtime": str(runtime),
        "project_id": project_id,
        "scenario": "openpi_droid_live",
        "renewal_boundary_seconds": scenario_timeout_seconds,
        "service": "sim",
        "cli": str(cli_path),
        "relay_transport": "antioch-declared-port-double-wss",
        "bridge": "sim",
        "relay_state": str(relay_state),
        "active_state": str(active_state),
    }
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.chmod(state_path, 0o600)
    return {
        "status": "running",
        "session": session,
        "scenario": state["scenario"],
        "renewal_boundary_seconds": scenario_timeout_seconds,
        "credentials_in_process_arguments": False,
        "transport": state["relay_transport"],
    }


def status_live(*, project_id: str) -> dict[str, Any]:
    """Return local supervisor state without reading auth storage or process lists."""

    state = _read_state(project_id)
    session = str(state["session"])
    runtime = Path(str(state["runtime"]))
    log = runtime / "live.log"
    result: dict[str, Any] = {
        "status": "running" if _session_running(session) else "stopped",
        "session": session,
        "scenario": state["scenario"],
        "renewal_boundary_seconds": state["renewal_boundary_seconds"],
        "log_exists": log.is_file(),
        "runtime": str(runtime),
        "transport": state.get("relay_transport", "direct-wss"),
    }
    relay_state_path = Path(str(state.get("relay_state", "")))
    if relay_state_path.is_file() and not relay_state_path.is_symlink():
        relay = json.loads(relay_state_path.read_text(encoding="utf-8"))
        allowed = {
            "status",
            "connections",
            "reconnects",
            "forwarded_requests",
            "failures",
            "last_round_trip_ms",
            "last_error_type",
        }
        result["relay"] = {key: relay.get(key) for key in sorted(allowed)}
    active_state_path = Path(
        str(state.get("active_state") or runtime / "active-run.json")
    )
    if active_state_path.is_file() and not active_state_path.is_symlink():
        active = json.loads(active_state_path.read_text(encoding="utf-8"))
        result["active_run"] = {
            "scenario": active.get("scenario"),
            "scenario_run_id": active.get("scenario_run_id"),
            "status": active.get("status"),
        }
    return result


def stop_live(*, project_id: str, timeout_seconds: float = 120.0) -> dict[str, Any]:
    """Stop the exact streamed run first, then its exact supported sim service."""

    state = _read_state(project_id)
    session = str(state["session"])
    runtime = Path(str(state["runtime"]))
    stop_file = runtime / ".stop"
    stop_file.touch(mode=0o600, exist_ok=True)
    if _window_running(session, "scenario"):
        _tmux("send-keys", "-t", f"{session}:scenario.0", "C-c")
        deadline = time.monotonic() + timeout_seconds
        while _window_running(session, "scenario") and time.monotonic() < deadline:
            time.sleep(1)
        if _window_running(session, "scenario"):
            raise AntiochLiveError(
                "scenario cancellation did not finish; refusing to tear down its service"
            )
    cli_path = Path(str(state["cli"]))
    cli = AntiochCli(cli_path)
    cancelled_remote_runs = _cancel_remote_live_runs(
        cli,
        runtime=runtime,
        project_id=project_id,
        scenario="openpi_droid_live",
    )
    if _window_running(session, "relay"):
        _tmux("send-keys", "-t", f"{session}:relay.0", "C-c")
        deadline = time.monotonic() + timeout_seconds
        while _window_running(session, "relay") and time.monotonic() < deadline:
            time.sleep(1)
        if _window_running(session, "relay"):
            raise AntiochLiveError(
                "relay cancellation did not finish; refusing to tear down its service"
            )
    if _window_running(session, "bridge"):
        _tmux("send-keys", "-t", f"{session}:bridge.0", "C-c")
        deadline = time.monotonic() + timeout_seconds
        while _window_running(session, "bridge") and time.monotonic() < deadline:
            time.sleep(1)
        if _window_running(session, "bridge"):
            raise AntiochLiveError(
                "bridge cancellation did not finish; refusing to tear down its service"
            )
    cli.services_down(runtime)
    return {
        "status": "stopped",
        "session": session,
        "service_stopped_after_scenario": True,
        "cancelled_remote_runs": cancelled_remote_runs,
        "runtime_preserved": str(runtime),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--client-bundle", required=True)
    parser.add_argument("--scenario-timeout-seconds", type=int, default=14_400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = start_live(
        source=Path(args.source),
        project_id=args.project_id,
        client_bundle=Path(args.client_bundle),
        scenario_timeout_seconds=args.scenario_timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

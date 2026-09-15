"""Endpoint selection for workbench HTTP services."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import socket
import subprocess
import tempfile
import time
from typing import Any, Iterator
from urllib.parse import urlparse, urlunparse

from npa.clients.config import update_workbench_endpoint_strategy


class EndpointError(Exception):
    pass


@dataclass
class ActiveEndpoint:
    url: str
    strategy: str
    local_port: int | None = None


def _is_loopback_host(hostname: str | None) -> bool:
    return (hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


def _port_from_url(url: str) -> int:
    try:
        return int(urlparse(url or "").port or 0)
    except ValueError:
        return 0


def _replace_host_port(url: str, host: str, port: int) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    return urlunparse((scheme, netloc, parsed.path.rstrip("/"), "", "", "")).rstrip("/")


def _tcp_open(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_local_port(port: int, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _tcp_open("127.0.0.1", port, timeout=0.25):
            return
        time.sleep(0.1)
    raise EndpointError(f"SSH tunnel did not become ready on 127.0.0.1:{port}")


def _open_ssh_forward(cfg: Any, local_port: int, remote_port: int) -> subprocess.Popen:
    key_path = os.path.expanduser(cfg.ssh.key_path)
    from npa.deploy.ssh_trust import known_hosts_path

    selected_hosts = os.environ.get("NPA_SSH_KNOWN_HOSTS", "").strip()
    provider_hosts = known_hosts_path(cfg.ssh.host)
    if not selected_hosts and provider_hosts.is_file():
        selected_hosts = str(provider_hosts)
    control_directory = tempfile.TemporaryDirectory(prefix="npa-ssh-")
    control_path = os.path.join(control_directory.name, "control")
    target = f"{cfg.ssh.user}@{cfg.ssh.host}"
    cmd = [
        "ssh",
        "-M",
        "-S",
        control_path,
        "-N",
        "-o",
        "ControlPersist=no",
        "-o",
        "ForkAfterAuthentication=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        key_path,
        target,
    ]
    if selected_hosts:
        cmd[1:1] = ["-o", f"UserKnownHostsFile={os.path.expanduser(selected_hosts)}"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        control_directory.cleanup()
        raise EndpointError(f"Unable to start SSH tunnel: {exc}") from exc
    proc._npa_ssh_control_directory = control_directory
    proc._npa_ssh_control_path = control_path
    proc._npa_ssh_target = target
    return proc


def _wait_for_ssh_forward(
    proc: subprocess.Popen, local_port: int, remote_port: int, *, timeout: float = 10.0,
) -> None:
    """Ask the authenticated master to bind, and require its success response."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(proc._npa_ssh_control_path):
        if proc.poll() is not None:
            raise EndpointError("SSH authentication or host verification failed before forwarding")
        if time.monotonic() >= deadline:
            raise EndpointError("SSH control connection did not become ready")
        time.sleep(0.05)
    try:
        result = subprocess.run(
            [
                "ssh", "-S", proc._npa_ssh_control_path, "-O", "forward",
                "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
                proc._npa_ssh_target,
            ],
            capture_output=True, text=True, check=False,
            timeout=max(0.01, deadline - time.monotonic()),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EndpointError("Unable to confirm forwarding through the authenticated SSH master") from exc
    if result.returncode != 0 or proc.poll() is not None:
        raise EndpointError("SSH could not bind the requested local forwarding port")


def _close_process(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        for name in ("stdout", "stderr"):
            pipe = getattr(proc, name, None)
            if pipe is not None and hasattr(pipe, "close"):
                pipe.close()
        control_directory = getattr(proc, "_npa_ssh_control_directory", None)
        if isinstance(control_directory, tempfile.TemporaryDirectory):
            control_directory.cleanup()


def _is_byovm_config(cfg: Any) -> bool:
    runtime = str(getattr(cfg, "runtime", "") or "").lower()
    if runtime == "byovm":
        return True
    return getattr(cfg, "managed_lifecycle", None) is False


def _is_serverless_config(cfg: Any) -> bool:
    return str(getattr(cfg, "runtime", "") or "").lower() == "serverless"


def _is_ssh_strategy(strategy: str) -> bool:
    return strategy.replace("-", "_") in {"ssh", "ssh_fallback"}


def _public_endpoint_open(base_url: str, *, default_port: int = 0) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = _port_from_url(base_url) or int(default_port or 0)
    if not host or port <= 0:
        return False
    if _is_loopback_host(host):
        return True
    return _tcp_open(host, port, timeout=1.0)


def _persist_ssh_strategy(cfg: Any, remote_port: int) -> None:
    project = str(getattr(cfg, "project", "") or "")
    name = str(getattr(cfg, "name", "") or "")
    if not project or not name:
        return
    update_workbench_endpoint_strategy(project, name, "ssh_fallback", remote_port)
    setattr(cfg, "endpoint_strategy", "ssh_fallback")
    setattr(cfg, "service_port", remote_port)
    setattr(cfg, "endpoint_strategy_configured", True)
    setattr(cfg, "service_port_configured", True)


@contextmanager
def _ssh_service_endpoint(
    cfg: Any,
    *,
    base_url: str,
    default_port: int = 0,
    service_port: int | None = None,
    allow_existing_local_port: bool = True,
    require_new_forward: bool = False,
) -> Iterator[ActiveEndpoint]:
    parsed = urlparse(base_url)
    remote_port = (
        int(service_port or 0)
        or int(getattr(cfg, "service_port", 0) or 0)
        or _port_from_url(base_url)
        or int(default_port or 0)
    )
    if remote_port <= 0:
        raise EndpointError("SSH endpoint strategy requires a service port")

    if not require_new_forward and _is_loopback_host(parsed.hostname):
        yield ActiveEndpoint(
            url=base_url,
            strategy="ssh_fallback",
            local_port=_port_from_url(base_url) or remote_port,
        )
        return

    if allow_existing_local_port and _tcp_open("127.0.0.1", remote_port):
        yield ActiveEndpoint(
            url=_replace_host_port(base_url, "127.0.0.1", remote_port),
            strategy="ssh_fallback",
            local_port=remote_port,
        )
        return

    if (
        not getattr(cfg.ssh, "host", "")
        or not getattr(cfg.ssh, "user", "")
        or not getattr(cfg.ssh, "key_path", "")
    ):
        raise EndpointError("SSH endpoint strategy requires ssh host, user, and key path")

    local_port = _free_local_port()
    proc = _open_ssh_forward(cfg, local_port, remote_port)
    try:
        _wait_for_ssh_forward(proc, local_port, remote_port)
        yield ActiveEndpoint(
            url=_replace_host_port(base_url, "127.0.0.1", local_port),
            strategy="ssh_fallback",
            local_port=local_port,
        )
    finally:
        _close_process(proc)


@contextmanager
def service_endpoint(
    cfg: Any,
    *,
    default_port: int = 0,
    endpoint: str | None = None,
    service_port: int | None = None,
    require_ssh: bool = False,
) -> Iterator[ActiveEndpoint]:
    """Yield the HTTP endpoint that should be used for a live command.

    ``require_ssh`` always creates a new verified SSH forward, including when a
    saved endpoint is public or a loopback port already has a listener.

    Older configs default to the public endpoint. BYOVM configs that recorded
    ``endpoint_strategy: ssh_fallback`` get a transient local SSH forward unless the
    configured endpoint already points at localhost or the service port is
    already reachable locally. Legacy BYOVM aliases that did not record an
    endpoint strategy, or that still record the public BYOVM route, get a
    public reachability check; if the public endpoint is blocked but SSH-local
    routing succeeds, the SSH strategy is persisted back to the alias config.
    """
    base_url = (endpoint or getattr(cfg, "endpoint", "")).rstrip("/")
    strategy = str(getattr(cfg, "endpoint_strategy", "") or "public").lower()
    strategy_configured = bool(getattr(cfg, "endpoint_strategy_configured", False))
    service_port_configured = bool(getattr(cfg, "service_port_configured", False))

    if require_ssh:
        with _ssh_service_endpoint(
            cfg, base_url=base_url, default_port=default_port,
            service_port=service_port, allow_existing_local_port=False,
            require_new_forward=True,
        ) as active:
            remote_port = int(service_port or getattr(cfg, "service_port", 0) or _port_from_url(base_url) or default_port)
            if not _is_ssh_strategy(strategy) or not service_port_configured:
                _persist_ssh_strategy(cfg, remote_port)
            yield active
        return

    if _is_serverless_config(cfg):
        yield ActiveEndpoint(url=base_url, strategy="serverless")
        return

    ssh_strategy = _is_ssh_strategy(strategy)
    byovm_public = _is_byovm_config(cfg) and strategy == "public"
    legacy_byovm = _is_byovm_config(cfg) and (not strategy_configured or byovm_public)
    persist_missing_ssh_port = (
        _is_byovm_config(cfg)
        and ssh_strategy
        and strategy_configured
        and not service_port_configured
    )

    if not ssh_strategy and not legacy_byovm:
        yield ActiveEndpoint(url=base_url, strategy="public")
        return

    remote_port = (
        int(service_port or 0)
        or int(getattr(cfg, "service_port", 0) or 0)
        or _port_from_url(base_url)
        or int(default_port or 0)
    )
    if legacy_byovm and _public_endpoint_open(base_url, default_port=default_port):
        yield ActiveEndpoint(url=base_url, strategy="public")
        return

    with _ssh_service_endpoint(
        cfg,
        base_url=base_url,
        default_port=default_port,
        service_port=service_port,
        allow_existing_local_port=not legacy_byovm,
    ) as active:
        if legacy_byovm or persist_missing_ssh_port:
            _persist_ssh_strategy(cfg, remote_port)
        yield active

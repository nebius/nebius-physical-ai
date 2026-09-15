"""Confirm that the authenticated Kubernetes child owns its local listener."""

import os
import selectors
import subprocess
import time

from npa.clients.endpoint import EndpointError


def _wait_for_kubernetes_forward(
    proc: subprocess.Popen, local_port: int, remote_port: int, *, timeout: float = 10.0,
) -> None:
    """Wait for kubectl itself to confirm the requested local listener."""
    if proc.stdout is None:
        raise EndpointError("Kubernetes port-forward has no readiness output")
    ready_line = f"Forwarding from 127.0.0.1:{local_port} -> {remote_port}".encode()
    deadline = time.monotonic() + timeout
    pending = b""
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise EndpointError("Kubernetes port-forward did not confirm its local listener")
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            lines = (pending + chunk).split(b"\n")
            pending = lines.pop()
            if any(line.strip() == ready_line for line in lines):
                if proc.poll() is not None:
                    break
                os.set_blocking(proc.stdout.fileno(), False)
                return
    raise EndpointError("Kubernetes port-forward exited before confirming its local listener")

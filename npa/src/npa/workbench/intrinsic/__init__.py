"""npa.workbench.intrinsic - Intrinsic Core workbench toolRef.

Read-only validation of an Intrinsic Core deployment (Apache 2.0, announced
at ROSCon 2026): host/runtime preflight, ICON real-time control status, and
digital-twin (world) reachability.

Honest surface: everything here is a probe. Mutating cluster operations —
``inctl world reset``, ``inctl asset install``, ``inctl icon clear-faults``,
and friends — are deliberately not exposed.

The CLI (:mod:`npa.cli.workbench.intrinsic`) is a thin client; this module is
the primary SDK surface and never goes through ``make_cli_wrapper``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from typing import Any

#: toolRef identity for this workbench module.
TOOLREF = "workbench.intrinsic"

#: ROS 2 distribution Intrinsic Core requires.
SUPPORTED_ROS_DISTRO = "lyrical"

#: Environment variable overriding the Intrinsic ingress address (host:port).
ADDRESS_ENV = "INTRINSIC_ADDRESS"

#: Default Intrinsic ingress address (Envoy, plain channel).
DEFAULT_ADDRESS = "localhost:17080"

#: Ubuntu releases Intrinsic Core lists as primary targets.
PRIMARY_UBUNTU_RELEASES = ("24.04", "26.04")

#: Minimum Ubuntu release accepted (22.04 is listed as supported).
MIN_UBUNTU_RELEASE = (22, 4)

#: Namespace the Intrinsic Core runtime installs into.
_RUNTIME_NAMESPACE = "app-intrinsic-base"

#: Per-probe subprocess timeout, in seconds.
_TIMEOUT_S = 30

#: Markers that indicate an errored service in ``inctl service state`` output.
_ERROR_MARKERS = ("STATE_CODE_ERROR", "ERROR")


def resolve_address(address: str | None = None) -> str:
    """Resolve the Intrinsic ingress address (explicit > env > default)."""
    return (address or os.environ.get(ADDRESS_ENV) or DEFAULT_ADDRESS).strip()


def _run(
    argv: list[str], timeout: int = _TIMEOUT_S
) -> subprocess.CompletedProcess[str] | None:
    """Run argv (never shell); return None when the binary is missing/times out."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _check(
    name: str,
    ok: bool,
    detail: str,
    *,
    skipped: bool = False,
) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, "skipped": skipped}


def _read_os_release() -> dict[str, str]:
    """Parse /etc/os-release into a dict (mockable seam for tests)."""
    data: dict[str, str] = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    data[key] = value.strip('"')
    except OSError:
        pass
    return data


def _check_ubuntu() -> tuple[dict[str, Any], str | None]:
    """Ubuntu >= 22.04 passes; warn unless 24.04/26.04. Returns (check, warning)."""
    release = _read_os_release().get("VERSION_ID", "")
    warning = None
    try:
        parts = tuple(int(p) for p in release.split(".")[:2])
    except ValueError:
        parts = ()
    if len(parts) < 2 or parts < MIN_UBUNTU_RELEASE:
        return (
            _check(
                "ubuntu",
                False,
                f"Ubuntu {release or 'unknown'} < 22.04; Intrinsic Core needs "
                "Ubuntu 22.04+ (24.04/26.04 recommended)",
            ),
            None,
        )
    if release not in PRIMARY_UBUNTU_RELEASES:
        warning = (
            f"Ubuntu {release} is supported but not a primary target; "
            f"prefer {', '.join(PRIMARY_UBUNTU_RELEASES)}"
        )
    return _check("ubuntu", True, f"Ubuntu {release}"), warning


def _check_ros_distro() -> dict[str, Any]:
    ros_distro = (os.environ.get("ROS_DISTRO") or "").strip()
    if ros_distro != SUPPORTED_ROS_DISTRO:
        return _check(
            "ros-distro",
            False,
            f"ROS_DISTRO={ros_distro or '<unset>'}; Intrinsic Core requires "
            f"ROS 2 {SUPPORTED_ROS_DISTRO.capitalize()} Luth "
            f"(`source /opt/ros/{SUPPORTED_ROS_DISTRO}/setup.bash`)",
        )
    return _check("ros-distro", True, f"ROS_DISTRO={ros_distro} (Lyrical Luth)")


def _check_k3s() -> dict[str, Any]:
    proc = _run(["systemctl", "is-active", "k3s"])
    if proc is not None and proc.stdout.strip() == "active":
        return _check("k3s", True, "k3s service is active")
    proc = _run(["pgrep", "-x", "k3s"])
    if proc is not None and proc.returncode == 0:
        return _check("k3s", True, "k3s process is running (non-systemd host)")
    return _check(
        "k3s",
        False,
        "k3s is not running; start it per the Intrinsic Core getting-started "
        "guide (https://github.com/intrinsic-ai/intrinsic-core)",
    )


def _check_inctl() -> dict[str, Any]:
    if shutil.which("inctl"):
        return _check("inctl", True, f"inctl on PATH ({shutil.which('inctl')})")
    return _check(
        "inctl",
        False,
        "inctl (Intrinsic Control CLI) not on PATH; install the "
        "inctl-linux-amd64 release asset per the Intrinsic Core "
        "getting-started guide",
    )


def _check_gpu() -> tuple[dict[str, Any], str | None]:
    if shutil.which("nvidia-smi"):
        return _check("gpu", True, "nvidia-smi present"), None
    return (
        _check("gpu", True, "no GPU detected"),
        "nvidia-smi not found; perception/inference workloads need an NVIDIA "
        "GPU (RTX 3060/4060 or better)",
    )


def _split_address(address: str) -> tuple[str, int] | None:
    host, sep, port = address.rpartition(":")
    if not sep or not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


def _check_ingress(address: str) -> dict[str, Any]:
    target = _split_address(address)
    if target is None:
        return _check(
            "ingress",
            False,
            f"cannot parse address {address!r}; expected host:port",
        )
    host, port = target
    try:
        with socket.create_connection((host, port), timeout=10):
            pass
    except OSError as exc:
        return _check(
            "ingress",
            False,
            f"TCP connect to {address} failed ({exc}); is the Intrinsic Core "
            "runtime (k3s + Envoy ingress) up on this host?",
        )
    return _check("ingress", True, f"TCP connect to {address} ok")


def _check_kubectl_pods() -> dict[str, Any]:
    if not shutil.which("kubectl"):
        return _check(
            "runtime-pods",
            True,
            "kubectl not on PATH; runtime pod check skipped",
            skipped=True,
        )
    proc = _run(["kubectl", "get", "pods", "-n", _RUNTIME_NAMESPACE, "-o", "json"])
    if proc is None or proc.returncode != 0:
        return _check(
            "runtime-pods",
            False,
            f"kubectl get pods -n {_RUNTIME_NAMESPACE} failed; is k3s up with "
            "the Intrinsic Core runtime installed?",
        )
    try:
        items = json.loads(proc.stdout).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        return _check(
            "runtime-pods",
            False,
            "could not parse `kubectl get pods -o json` output",
        )
    not_running = [
        item.get("metadata", {}).get("name", "?")
        for item in items
        if item.get("status", {}).get("phase") != "Running"
    ]
    if not_running:
        return _check(
            "runtime-pods",
            False,
            f"pods not Running in {_RUNTIME_NAMESPACE}: " + ", ".join(not_running),
        )
    return _check(
        "runtime-pods",
        True,
        f"{len(items)} pod(s) Running in {_RUNTIME_NAMESPACE}",
    )


def _service_state_raw(address: str) -> tuple[str | None, str]:
    """Run the read-only ``inctl service state list``; return (stdout, error)."""
    if not shutil.which("inctl"):
        return None, "inctl not on PATH"
    proc = _run(
        ["inctl", "service", "state", "list", "--address", address, "--output", "json"]
    )
    if proc is None:
        return None, "inctl service state list did not complete"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or "inctl service state list failed"
    return proc.stdout, ""


def _check_service_state(address: str) -> dict[str, Any]:
    raw, error = _service_state_raw(address)
    if raw is None:
        return _check("service-state", False, error)
    if any(marker in raw for marker in _ERROR_MARKERS):
        return _check(
            "service-state",
            False,
            "inctl service state list reports errored services; "
            "inspect with `inctl service state list`",
        )
    return _check("service-state", True, "service states report no errors")


def preflight(address: str | None = None) -> dict[str, Any]:
    """Check Intrinsic Core host + runtime prerequisites.

    Returns ``{"ok", "detail", "address", "supported_distro", "checks",
    "warnings"}``. Never raises for a missing environment — callers decide
    how to surface the failure (the CLI exits 3 with remediation).
    """
    resolved = resolve_address(address)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    ubuntu_check, ubuntu_warning = _check_ubuntu()
    checks.append(ubuntu_check)
    if ubuntu_warning:
        warnings.append(ubuntu_warning)
    checks.append(_check_ros_distro())
    checks.append(_check_k3s())
    checks.append(_check_inctl())
    gpu_check, gpu_warning = _check_gpu()
    checks.append(gpu_check)
    if gpu_warning:
        warnings.append(gpu_warning)

    host_ok = all(c["ok"] for c in checks)
    if host_ok:
        # Runtime probes only make sense once the host surface is present.
        checks.append(_check_kubectl_pods())
        checks.append(_check_ingress(resolved))
        checks.append(_check_service_state(resolved))

    failed = [c for c in checks if not c["ok"] and not c["skipped"]]
    ok = not failed
    if ok:
        detail = (
            f"Intrinsic Core preflight OK ({len(checks)} checks, "
            f"{len(warnings)} warning(s))"
        )
    else:
        detail = "; ".join(f"{c['name']}: {c['detail']}" for c in failed)
    return {
        "ok": ok,
        "detail": detail,
        "address": resolved,
        "supported_distro": SUPPORTED_ROS_DISTRO,
        "checks": checks,
        "warnings": warnings,
    }


def icon_status(
    address: str | None = None, instance_name: str = "icon"
) -> dict[str, Any]:
    """Read-only ICON real-time control status via ``inctl icon status``.

    Returns ``{"ok", "detail", "instance", "address", "excerpt"}``. Never
    raises for a missing environment.
    """
    resolved = resolve_address(address)
    if not shutil.which("inctl"):
        return {
            "ok": False,
            "detail": "inctl not on PATH; cannot query ICON status",
            "instance": instance_name,
            "address": resolved,
            "excerpt": "",
        }
    proc = _run(
        [
            "inctl",
            "icon",
            "status",
            "--instance_name",
            instance_name,
            "--address",
            resolved,
        ]
    )
    if proc is None:
        return {
            "ok": False,
            "detail": "inctl icon status did not complete",
            "instance": instance_name,
            "address": resolved,
            "excerpt": "",
        }
    excerpt = (proc.stdout or proc.stderr or "").strip()[:2000]
    if proc.returncode != 0:
        return {
            "ok": False,
            "detail": f"inctl icon status failed for instance {instance_name!r}",
            "instance": instance_name,
            "address": resolved,
            "excerpt": excerpt,
        }
    interesting = [
        line.strip()
        for line in excerpt.splitlines()
        if re.search(r"state|fault|safety|operational", line, re.IGNORECASE)
    ]
    detail = "; ".join(interesting[:5]) or "ICON status query ok"
    return {
        "ok": True,
        "detail": detail,
        "instance": instance_name,
        "address": resolved,
        "excerpt": excerpt,
    }


def world_probe(address: str | None = None) -> dict[str, Any]:
    """Read-only digital-twin (world) reachability probe.

    There is no read-only ``inctl world`` query (``inctl world reset`` mutates
    state and is deliberately not used). This probe checks TCP ingress and
    then looks for a world/ObjectWorld entry in the read-only
    ``inctl service state list`` output. Never mutates state.
    """
    resolved = resolve_address(address)
    ingress = _check_ingress(resolved)
    if not ingress["ok"]:
        return {
            "ok": False,
            "detail": ingress["detail"],
            "address": resolved,
            "world_service_found": False,
        }
    raw, error = _service_state_raw(resolved)
    if raw is None:
        return {
            "ok": False,
            "detail": error,
            "address": resolved,
            "world_service_found": False,
        }
    found = bool(re.search(r"world", raw, re.IGNORECASE))
    if not found:
        return {
            "ok": False,
            "detail": "no world/ObjectWorld service in `inctl service state list`",
            "address": resolved,
            "world_service_found": False,
        }
    return {
        "ok": True,
        "detail": "world/ObjectWorld service present in service state list",
        "address": resolved,
        "world_service_found": True,
    }


__all__ = [
    "TOOLREF",
    "SUPPORTED_ROS_DISTRO",
    "ADDRESS_ENV",
    "DEFAULT_ADDRESS",
    "resolve_address",
    "preflight",
    "icon_status",
    "world_probe",
]

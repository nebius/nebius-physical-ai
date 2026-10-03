"""Deterministic runtime fetch for NVIDIA OVRTX used by Content Agents.

The public Content Agents image contains the reviewed Apache-2.0 source and this
bootstrap, but no OVRTX wheel or installed runtime.  On a render-bearing stage,
the operator downloads the exact upstream-locked runtime directly from NVIDIA's
anonymous package index into the standard NPA runtime cache.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


OVRTX_VERSION = "0.3.0.312915"
NUMPY_VERSION = "2.2.6"
PILLOW_VERSION = "12.3.0"
OVRTX_RUNTIME_LOCK_SHA256 = (
    "ed582577175e4a5b32f8b69ef9cdbfc3d7337f3786051d8b076e30a2652f6fa5"
)
OVRTX_RUNTIME_LOCK = Path(
    "/opt/content-agents/world_understanding/functions/graphics/"
    "pylock.ovrtx-runtime.toml"
)
RUNTIME_CACHE_ENV = "NPA_CONTENT_AGENTS_RUNTIME_CACHE"
READY_MARKER = ".npa-runtime-ready.json"
UPSTREAM_READY_MARKER = ".wu-managed-ovrtx-venv"

# The complete lock digest pins every transitive wheel.  These per-platform
# OVRTX entries additionally prove that the selected architecture resolves to
# the reviewed anonymous NVIDIA artifact, never to a mutable package alias.
OVRTX_WHEELS: Mapping[str, tuple[str, str]] = {
    "x86_64": (
        "https://pypi.nvidia.com/ovrtx/"
        "ovrtx-0.3.0.312915-py3-none-manylinux_2_35_x86_64.whl",
        "a6b2b3c357f6487451c8d71e96cc4f83156c08fd9747d10e1b65f3866bed4b8f",
    ),
    "aarch64": (
        "https://pypi.nvidia.com/ovrtx/"
        "ovrtx-0.3.0.312915-py3-none-manylinux_2_35_aarch64.whl",
        "958d254cebeb271ac397f5b760ff5def84a02d04b23f610c31d456c7c8e871af",
    ),
}
_ARCH_ALIASES = {"amd64": "x86_64", "arm64": "aarch64"}


class ContentAgentsRuntimeError(RuntimeError):
    """Raised when the exact reviewed OVRTX runtime cannot be proven ready."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _architecture(machine: str | None = None) -> str:
    raw = (machine or platform.machine()).strip().lower()
    architecture = _ARCH_ALIASES.get(raw, raw)
    if architecture not in OVRTX_WHEELS:
        raise ContentAgentsRuntimeError(
            f"OVRTX {OVRTX_VERSION} has no reviewed lock entry for architecture {raw!r}"
        )
    return architecture


def _lock_path(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("NPA_CONTENT_AGENTS_OVRTX_LOCK", str(OVRTX_RUNTIME_LOCK)))


def _verify_lock(lock_path: Path, architecture: str) -> None:
    if not lock_path.is_file():
        raise ContentAgentsRuntimeError(
            f"reviewed OVRTX runtime lock is absent: {lock_path}"
        )
    actual = _sha256(lock_path)
    if actual != OVRTX_RUNTIME_LOCK_SHA256:
        raise ContentAgentsRuntimeError(
            "OVRTX runtime lock digest differs from the reviewed v0.5.2 lock"
        )
    text = lock_path.read_text(encoding="utf-8")
    wheel_url, wheel_sha256 = OVRTX_WHEELS[architecture]
    for expected in (
        'name = "ovrtx"',
        f'version = "{OVRTX_VERSION}"',
        wheel_url,
        f'sha256 = "{wheel_sha256}"',
    ):
        if expected not in text:
            raise ContentAgentsRuntimeError(
                f"reviewed OVRTX lock omits the {architecture} immutable artifact"
            )


def _cache_root(environ: Mapping[str, str]) -> tuple[Path, str]:
    configured = str(environ.get(RUNTIME_CACHE_ENV, "")).strip()
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ContentAgentsRuntimeError(f"{RUNTIME_CACHE_ENV} must be absolute")
        return root, "configured-filesystem"
    cache_home = str(environ.get("XDG_CACHE_HOME", "")).strip()
    if cache_home:
        root = Path(cache_home).expanduser()
    else:
        root = Path(environ.get("HOME", str(Path.home()))).expanduser() / ".cache"
    return root / "npa" / "runtime-cache" / "content-agents", "node-ephemeral"


def runtime_path(
    *, environ: Mapping[str, str] | None = None, machine: str | None = None
) -> tuple[Path, str, str]:
    env = environ if environ is not None else os.environ
    architecture = _architecture(machine)
    root, cache_tier = _cache_root(env)
    identity = f"lock-{OVRTX_RUNTIME_LOCK_SHA256}"
    return (
        root / "ovrtx" / OVRTX_VERSION / architecture / identity,
        architecture,
        cache_tier,
    )


def _probe_runtime(target: Path) -> None:
    python = target / "bin" / "python"
    upstream_marker = target / UPSTREAM_READY_MARKER
    if not python.is_file() or not upstream_marker.is_file():
        raise ContentAgentsRuntimeError(
            "OVRTX cache lacks its verified upstream ready marker"
        )
    fields: dict[str, str] = {}
    for line in upstream_marker.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and value.strip():
            fields[key.strip()] = value.strip()
    if fields != {
        "ovrtx_version": OVRTX_VERSION,
        "runtime_lock_sha256": OVRTX_RUNTIME_LOCK_SHA256,
    }:
        raise ContentAgentsRuntimeError(
            "OVRTX upstream marker differs from the reviewed lock"
        )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata,json; print(json.dumps({name: "
            "importlib.metadata.version(name) for name in ('ovrtx','numpy','pillow')}))",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "WU_OVRTX_AUTO_PROVISION": "0"},
    )
    try:
        versions = json.loads(probe.stdout)
    except json.JSONDecodeError:
        versions = {}
    if probe.returncode != 0 or versions != {
        "ovrtx": OVRTX_VERSION,
        "numpy": NUMPY_VERSION,
        "pillow": PILLOW_VERSION,
    }:
        raise ContentAgentsRuntimeError(
            "OVRTX cache does not contain the exact reviewed runtime"
        )


def _ready_payload(target: Path, architecture: str, cache_tier: str) -> dict[str, Any]:
    return {
        "schema": "npa.content_agents.ovrtx_runtime.v1",
        "status": "ready",
        "ovrtx_version": OVRTX_VERSION,
        "architecture": architecture,
        "runtime_lock_sha256": OVRTX_RUNTIME_LOCK_SHA256,
        "cache_identity": target.name,
        "cache_tier": cache_tier,
    }


def inspect_ready_runtime(
    *, environ: Mapping[str, str] | None = None, machine: str | None = None
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    target, architecture, cache_tier = runtime_path(environ=env, machine=machine)
    marker = target / READY_MARKER
    if not marker.is_file():
        raise ContentAgentsRuntimeError(
            "exact OVRTX runtime is not ready; run bootstrap-runtime"
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentAgentsRuntimeError(
            "OVRTX runtime ready marker is invalid"
        ) from exc
    expected = _ready_payload(target, architecture, cache_tier)
    if payload != expected:
        raise ContentAgentsRuntimeError(
            "OVRTX runtime identity does not match its ready marker"
        )
    _probe_runtime(target)
    return {**payload, "runtime_path": str(target)}


def bootstrap_runtime(
    *, environ: Mapping[str, str] | None = None, machine: str | None = None
) -> dict[str, Any]:
    """Fetch and atomically publish the exact OVRTX runtime, once per cache."""

    env = dict(environ if environ is not None else os.environ)
    target, architecture, cache_tier = runtime_path(environ=env, machine=machine)
    lock_path = _lock_path(env)
    _verify_lock(lock_path, architecture)
    target.parent.mkdir(parents=True, exist_ok=True)
    locks = target.parents[2] / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    writer_lock = locks / f"{target.name}-{architecture}.lock"
    with writer_lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if target.exists():
            payload = inspect_ready_runtime(environ=env, machine=architecture)
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
            )
            try:
                child_env = {
                    **env,
                    "WU_OVRTX_VENV_DIR": str(temporary),
                    "WU_OVRTX_AUTO_PROVISION": "1",
                    "WU_OVRTX_LOCK_DIR": str(locks / "upstream"),
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                }
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "world_understanding.functions.graphics.render_ovrtx",
                        "--provision-only",
                    ],
                    check=False,
                    env=child_env,
                )
                if completed.returncode:
                    raise ContentAgentsRuntimeError(
                        "anonymous NVIDIA OVRTX runtime fetch failed"
                    )
                _probe_runtime(temporary)
                ready = _ready_payload(target, architecture, cache_tier)
                (temporary / READY_MARKER).write_text(
                    json.dumps(ready, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.rename(temporary, target)
                payload = inspect_ready_runtime(environ=env, machine=architecture)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    # The upstream renderer snapshots this environment at import time. Callers
    # invoke bootstrap before importing any renderer-bearing agent code.
    os.environ["WU_OVRTX_VENV_DIR"] = str(target)
    os.environ["WU_OVRTX_AUTO_PROVISION"] = "0"
    os.environ["WU_OVRTX_LOCK_DIR"] = str(locks / "upstream")
    return payload


def inspect_image() -> dict[str, Any]:
    """Prove source pins are installed while OVRTX itself is absent."""

    # Upstream uses uv's PEP 751 support to install the complete reviewed lock.
    # Treat the downloader as part of image readiness: a byte-clean image that
    # cannot perform its promised runtime fetch is not ready.
    if shutil.which("uv") is None:
        raise ContentAgentsRuntimeError(
            "public image lacks the pinned uv runtime bootstrap executable"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "world-understanding",
            "material-agent",
            "physics-agent",
            "validation-agent",
        )
    }
    if set(versions.values()) != {"0.5.2"}:
        raise ContentAgentsRuntimeError(
            f"Content Agents packages are not pinned: {versions}"
        )
    try:
        installed_ovrtx = importlib.metadata.version("ovrtx")
    except importlib.metadata.PackageNotFoundError:
        installed_ovrtx = None
    if installed_ovrtx is not None:
        raise ContentAgentsRuntimeError("public image must not contain OVRTX")
    lock_path = _lock_path(os.environ)
    _verify_lock(lock_path, _architecture())
    legacy = Path("/opt/content-agents/.ovrtx_venv")
    if legacy.exists():
        raise ContentAgentsRuntimeError(
            "public image contains the legacy baked OVRTX runtime"
        )
    return {
        "schema": "npa.content_agents.image_runtime_boundary.v1",
        "status": "image-ready",
        "packages": versions,
        "ovrtx_baked": False,
        "ovrtx_version": OVRTX_VERSION,
        "runtime_lock_sha256": OVRTX_RUNTIME_LOCK_SHA256,
    }

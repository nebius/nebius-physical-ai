"""Validate native viewport graphics or extract an exact-driver private fallback."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

from .errors import IsaacArenaError
from .hashing import file_sha256 as _sha256
from .optix_payload import _native_optix_weights, _prepare_optix_weights

_NVIDIA_DRIVER_VERSION = re.compile(r"^[0-9]{3}\.[0-9]+\.[0-9]+$")
_VULKAN_MANIFEST_VERSION = re.compile(r"^[0-9]{1,3}(?:\.[0-9]{1,3}){1,3}$")
_Runner = Callable[..., subprocess.CompletedProcess[str]]
_SIGNED_APT_OPTIONS = [
    "-o",
    "Acquire::AllowInsecureRepositories=false",
    "-o",
    "APT::Get::AllowUnauthenticated=false",
]


def _graphics_probe(env: dict[str, str], *, runner: _Runner = subprocess.run) -> bool:
    """Return whether NVIDIA's headless EGL/Vulkan and OptiX libraries load."""
    libraries = runner(
        [sys.executable, "-c", "import ctypes; ctypes.CDLL('libEGL_nvidia.so.0'); "
         "ctypes.CDLL('libnvoptix.so.1')"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if libraries.returncode != 0:
        return False
    try:
        vulkan = runner(
            ["vulkaninfo", "--summary"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return False
    report = f"{vulkan.stdout or ''}\n{vulkan.stderr or ''}".lower()
    return vulkan.returncode == 0 and "nvidia" in report


def _checked_command(
    argv: list[str],
    *,
    error: str,
    runner: _Runner,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    completed = runner(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        **kwargs,
    )
    if completed.returncode != 0:
        raise IsaacArenaError(error)
    return completed


def _loaded_driver_version(env: dict[str, str], runner: _Runner) -> str:
    driver = _checked_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        error="viewport graphics require a readable NVIDIA driver version",
        runner=runner,
        env=env,
    )
    versions = {
        line.strip() for line in (driver.stdout or "").splitlines() if line.strip()
    }
    if (
        len(versions) != 1
        or _NVIDIA_DRIVER_VERSION.fullmatch(next(iter(versions), "")) is None
    ):
        raise IsaacArenaError(
            "viewport graphics require one exact NVIDIA driver version"
        )
    return versions.pop()


def _matching_package_version(
    package: str,
    driver_version: str,
    env: dict[str, str],
    runner: _Runner,
) -> str:
    _checked_command(
        ["sudo", "apt-get", *_SIGNED_APT_OPTIONS, "update", "-qq"],
        error="viewport graphics package metadata refresh failed",
        runner=runner,
        env=env,
    )
    policy = _checked_command(
        ["apt-cache", "policy", package],
        error="viewport graphics package metadata is unavailable",
        runner=runner,
        env=env,
    )
    match = re.search(r"^\s*Candidate:\s*(\S+)\s*$", policy.stdout or "", re.MULTILINE)
    candidate = match.group(1) if match else ""
    if not candidate.startswith(f"{driver_version}-"):
        raise IsaacArenaError(
            "viewport graphics archive does not contain the exact loaded driver version"
        )
    return candidate


def _download_graphics_package(
    root: Path,
    package: str,
    candidate: str,
    env: dict[str, str],
    runner: _Runner,
) -> tuple[Path, Path]:
    download_dir = root / "download"
    extract_dir = root / "extracted"
    download_dir.mkdir(parents=True, exist_ok=False)
    extract_dir.mkdir(parents=True, exist_ok=False)
    _checked_command(
        ["apt-get", *_SIGNED_APT_OPTIONS, "download", f"{package}={candidate}"],
        error="viewport graphics package download failed",
        runner=runner,
        env=env,
        cwd=download_dir,
    )
    debs = sorted(download_dir.glob("*.deb"))
    if len(debs) != 1 or not debs[0].is_file() or debs[0].stat().st_size == 0:
        raise IsaacArenaError("viewport graphics download did not yield one package")
    return debs[0], extract_dir


def _validate_package_identity(
    deb: Path,
    package: str,
    candidate: str,
    env: dict[str, str],
    runner: _Runner,
) -> None:
    fields = {}
    for field in ("Package", "Version", "Architecture"):
        value = _checked_command(
            ["dpkg-deb", "--field", str(deb), field],
            error="viewport graphics package metadata validation failed",
            runner=runner,
            env=env,
        ).stdout.strip()
        fields[field] = value
    if fields != {"Package": package, "Version": candidate, "Architecture": "amd64"}:
        raise IsaacArenaError("viewport graphics package identity does not match")


def _extract_graphics_package(
    deb: Path,
    extract_dir: Path,
    env: dict[str, str],
    runner: _Runner,
) -> tuple[Path, Path]:
    _checked_command(
        ["dpkg-deb", "--extract", str(deb), str(extract_dir)],
        error="viewport graphics package extraction failed",
        runner=runner,
        env=env,
    )
    library_dir = extract_dir / "usr/lib/x86_64-linux-gnu"
    packaged_icd = extract_dir / "usr/share/vulkan/icd.d/nvidia_icd.json"
    if (
        not (library_dir / "libEGL_nvidia.so.0").exists()
        or not packaged_icd.is_file()
        or packaged_icd.is_symlink()
        or packaged_icd.stat().st_size > 4096
    ):
        raise IsaacArenaError(
            "viewport graphics package is missing headless EGL/Vulkan metadata"
        )
    return library_dir, packaged_icd


def _validated_headless_manifest(
    packaged_icd: Path, library_dir: Path
) -> dict[str, Any]:
    try:
        manifest = json.loads(packaged_icd.read_text(encoding="utf-8"))
        file_format_version = manifest["file_format_version"]
        packaged_entrypoint = manifest["ICD"]["library_path"]
        api_version = manifest["ICD"]["api_version"]
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise IsaacArenaError(
            "viewport graphics package has invalid Vulkan metadata"
        ) from exc
    if (
        file_format_version not in {"1.0.0", "1.0.1"}
        or packaged_entrypoint not in {"libGLX_nvidia.so.0", "libEGL_nvidia.so.0"}
        or _VULKAN_MANIFEST_VERSION.fullmatch(str(api_version)) is None
        or not (library_dir / packaged_entrypoint).exists()
    ):
        raise IsaacArenaError(
            "viewport graphics package has unsupported Vulkan metadata"
        )
    # NVIDIA documents EGL as a headless Vulkan entrypoint. Only validated
    # scalar fields enter the override; packaged bytes remain untouched.
    return {
        "file_format_version": file_format_version,
        "ICD": {"library_path": "libEGL_nvidia.so.0", "api_version": api_version},
    }


def _configure_headless_graphics(
    root: Path,
    library_dir: Path,
    packaged_icd: Path,
    env: dict[str, str],
) -> Path:
    manifest = _validated_headless_manifest(packaged_icd, library_dir)
    headless_icd = root / "nvidia-headless-icd.json"
    headless_icd.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    headless_icd.chmod(0o600)
    previous_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(library_dir) + (
        f":{previous_library_path}" if previous_library_path else ""
    )
    env["VK_ICD_FILENAMES"] = str(headless_icd)
    env["VK_DRIVER_FILES"] = str(headless_icd)
    return headless_icd


def _private_graphics_evidence(
    driver_version: str,
    package: str,
    candidate: str,
    package_sha256: str,
    packaged_icd: Path,
    headless_icd: Path,
    optix_weights: dict[str, Any],
    optix_library_sha256: str,
) -> dict[str, Any]:
    return {
        "mode": "runtime_package_extract",
        "validated": True,
        "driver_version": driver_version,
        "driver_branch": driver_version.split(".", 1)[0],
        "package": package,
        "package_version": candidate,
        "package_sha256": package_sha256,
        "package_manifest_sha256": _sha256(packaged_icd),
        "headless_icd_sha256": _sha256(headless_icd),
        "headless_icd_override": True,
        "icd_entrypoint": "libEGL_nvidia.so.0",
        "source": "Ubuntu signed NVIDIA driver archive",
        "runtime_fetch": True,
        "installed_on_node": False,
        "baked": False,
        "published": False,
        "redistribution": False,
        "optix_weights": optix_weights,
        "optix_library_sha256": optix_library_sha256,
    }


def _optix_library_sha256(library_dir: Path, driver_version: str) -> str:
    library = library_dir / f"libnvoptix.so.{driver_version}"
    entrypoint = library_dir / "libnvoptix.so.1"
    if (not library.is_file() or library.is_symlink() or library.stat().st_size == 0
            or entrypoint.resolve() != library.resolve()):
        raise IsaacArenaError("matching NVIDIA package has no exact-driver OptiX library")
    return _sha256(library)


def _prepare_viewport_graphics(
    root: Path,
    env: dict[str, str],
    *,
    runner: _Runner = subprocess.run,
) -> dict[str, Any]:
    """Validate native graphics or an exact driver package in private scratch."""
    native_weights = _native_optix_weights() if _graphics_probe(env, runner=runner) else None
    if native_weights is not None:
        return {
            "mode": "native",
            "validated": True,
            "runtime_fetch": False,
            "baked": False,
            "redistribution": False,
            "optix_weights": native_weights,
        }
    driver_version = _loaded_driver_version(env, runner)
    package = f"libnvidia-gl-{driver_version.split('.', 1)[0]}-server"
    candidate = _matching_package_version(package, driver_version, env, runner)
    deb, extract_dir = _download_graphics_package(root, package, candidate, env, runner)
    _validate_package_identity(deb, package, candidate, env, runner)
    package_sha256 = _sha256(deb)
    library_dir, packaged_icd = _extract_graphics_package(deb, extract_dir, env, runner)
    headless_icd = _configure_headless_graphics(root, library_dir, packaged_icd, env)
    optix_library_sha256 = _optix_library_sha256(library_dir, driver_version)
    optix_weights = _prepare_optix_weights(extract_dir)
    if not _graphics_probe(env, runner=runner):
        raise IsaacArenaError("driver-matched viewport graphics validation failed")
    return _private_graphics_evidence(
        driver_version,
        package,
        candidate,
        package_sha256,
        packaged_icd,
        headless_icd,
        optix_weights,
        optix_library_sha256,
    )

"""Hermetic production-shell checks for the LingBot-VA runtime cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "npa/docker/workbench/lingbot-va/lingbot_va_runtime.sh"
SOURCE_REQUIREMENTS = RUNTIME.with_name("runtime-source-requirements.txt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_target(
    cache_root: Path, requirements: Path, source_requirements: Path
) -> Path:
    """Match the shell's cache identity without reproducing its verification."""
    abi = subprocess.check_output(
        [
            "python3",
            "-c",
            "import sys,sysconfig; print(f'{sys.version_info.major}.{sys.version_info.minor}-{sysconfig.get_platform()}')",
        ],
        text=True,
    ).strip()
    identity = "|".join(
        (abi, _sha256(requirements), _sha256(source_requirements), _sha256(RUNTIME))
    )
    return cache_root / hashlib.sha256(identity.encode()).hexdigest()[:16]


def _write_module(site_packages: Path, relative: str, source: str = "") -> None:
    path = site_packages / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _prepare_verified_cache(
    tmp_path: Path,
    *,
    installed_libero_version: str,
) -> tuple[dict[str, str], Path]:
    """Build only the runtime imports and package metadata the shell verifies."""
    cache_root = tmp_path / "cache"
    requirements = tmp_path / "runtime-requirements.txt"
    requirements.write_text("# hermetic production-shell control\n", encoding="utf-8")
    source_requirements = tmp_path / "runtime-source-requirements.txt"
    source_requirements.write_text(SOURCE_REQUIREMENTS.read_text(), encoding="utf-8")
    target = _cache_target(cache_root, requirements, source_requirements)
    subprocess.run([sys.executable, "-m", "venv", str(target / "venv")], check=True)
    site_packages = Path(
        subprocess.check_output(
            [
                str(target / "venv/bin/python"),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            text=True,
        ).strip()
    )
    _write_module(
        site_packages,
        "torch/__init__.py",
        "__version__ = '2.13.0+cu130'\n"
        "class _Version:\n    cuda = '13.0'\n"
        "version = _Version()\n",
    )
    _write_module(site_packages, "torch/nn/__init__.py")
    _write_module(site_packages, "torch/nn/attention/__init__.py")
    _write_module(
        site_packages,
        "torch/nn/attention/flex_attention.py",
        "def flex_attention(*args, **kwargs):\n    return None\n",
    )
    _write_module(site_packages, "datasets/__init__.py", "__version__ = '5.0.1'\n")
    _write_module(site_packages, "wan_va/__init__.py")
    _write_module(site_packages, "wan_va/modules/__init__.py")
    _write_module(site_packages, "wan_va/modules/model.py")
    _write_module(site_packages, "libero/__init__.py")
    metadata = site_packages / f"libero-{installed_libero_version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: libero\nVersion: {installed_libero_version}\n",
        encoding="utf-8",
    )
    (target / ".complete").touch()
    cache_root.mkdir(exist_ok=True)
    (cache_root / "current").symlink_to(target)
    environment = os.environ.copy()
    environment.update(
        {
            "NPA_LINGBOT_VA_RUNTIME_CACHE": str(cache_root),
            "NPA_LINGBOT_VA_RUNTIME_REQUIREMENTS": str(requirements),
            "NPA_LINGBOT_VA_SOURCE_REQUIREMENTS": str(source_requirements),
            "NPA_LINGBOT_VA_RUNTIME_OFFLINE": "1",
        }
    )
    return environment, source_requirements


def _run_offline_ensure(
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNTIME), "ensure"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_offline_runtime_accepts_matching_installed_libero_metadata(
    tmp_path: Path,
) -> None:
    """The production shell validates actual installed package metadata."""
    environment, _ = _prepare_verified_cache(
        tmp_path,
        installed_libero_version="0.1.0",
    )

    result = _run_offline_ensure(environment)

    assert result.returncode == 0, result.stderr
    assert "libero=0.1.0" in result.stdout


def test_offline_runtime_rejects_wrong_installed_libero_metadata(
    tmp_path: Path,
) -> None:
    """A cache cannot be marked complete when its installed version is wrong."""
    environment, _ = _prepare_verified_cache(
        tmp_path,
        installed_libero_version="0.1.1",
    )

    result = _run_offline_ensure(environment)

    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_offline_runtime_rejects_stale_source_inventory_cache_binding(
    tmp_path: Path,
) -> None:
    """Changing the source declaration invalidates an otherwise complete cache."""
    environment, source_requirements = _prepare_verified_cache(
        tmp_path,
        installed_libero_version="0.1.0",
    )
    source_requirements.write_text("libero==0.1.1\n", encoding="utf-8")

    result = _run_offline_ensure(environment)

    assert result.returncode == 69
    assert "offline cache does not match the reviewed runtime identity" in result.stderr

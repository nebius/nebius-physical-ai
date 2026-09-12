"""Protect portable fixes without changing the pinned model dependency graph."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2] / "docker" / "workbench"
HARDENED_NLTK_VERSION = "3.10.3.post1+npa.cbc98458"


@pytest.mark.parametrize(
    "image", ["openpi", "alpamayo2-super", "cosmos3", "cosmos3-ray-serve"]
)
def test_portable_overlay_hashes_and_application_order(image: str) -> None:
    directory = ROOT / image
    lock = (directory / "security-upgrades-requirements.txt").read_text()
    entries = re.findall(
        r"^([\w.-]+(?:\[[^]]+\])?==[^\n]+)(.*?)(?=^[\w.-]+==|\Z)", lock, re.M | re.S
    )
    assert entries
    for requirement, hashes in entries:
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", requirement + hashes)
    docker = (directory / "Dockerfile").read_text()
    assert "--no-deps --require-hashes" in docker
    sync = docker.index("uv sync")
    overlay = docker.index("--no-deps --require-hashes", sync)
    check = docker.index('"pip","check"' if image == "openpi" else "pip check", overlay)
    assert sync < overlay < check
    assert "pillow==12.3.0" in lock
    assert "setuptools==84.0.0" in lock


def test_hardened_nltk_source_install_contract() -> None:
    installer = (ROOT / "common" / "install_hardened_nltk.sh").read_text()
    verifier = (ROOT / "common" / "verify_hardened_nltk.py").read_text()
    commit = "cbc98458b43de5f792f0382583c16df39e5c5117"
    archive_sha = "3c4a9e92b53e34074ae5b7bbf21109868b5ef620b40c68b8542a2db37f7d9d0c"

    assert f'NLTK_SOURCE_COMMIT="{commit}"' in installer
    assert f'NLTK_SOURCE_SHA256="{archive_sha}"' in installer
    assert f'NLTK_HARDENED_VERSION="{HARDENED_NLTK_VERSION}"' in installer
    assert "codeload.github.com/nltk/nltk/tar.gz/${NLTK_SOURCE_COMMIT}" in installer
    assert "sha256sum --check --strict" in installer
    assert "--no-deps --no-build-isolation" in installer
    assert f'HARDENED_VERSION = "{HARDENED_NLTK_VERSION}"' in verifier

    affected = {
        "cosmos3": ROOT / "cosmos3" / "security-upgrades-requirements.in",
        "cosmos3-nano-video": ROOT / "cosmos3-nano-video" / "requirements.txt",
        "cosmos3-ray-serve": (
            ROOT / "cosmos3-ray-serve" / "security-upgrades-requirements.in"
        ),
    }
    for image, manifest in affected.items():
        requirements = manifest.read_text()
        docker = (ROOT / image / "Dockerfile").read_text()
        assert not re.search(r"^nltk(?:\[[^]]+\])?\s*[=<>~!]", requirements, re.M)
        assert "docker/workbench/common/install_hardened_nltk.sh" in docker
        assert "docker/workbench/common/verify_hardened_nltk.py" in docker
        assert "/opt/npa/install_hardened_nltk.sh" in docker
        assert "/opt/npa/verify_hardened_nltk.py" in docker

    nano_requirements = affected["cosmos3-nano-video"].read_text()
    for dependency in ("click", "defusedxml", "joblib", "regex", "tqdm"):
        assert re.search(rf"^{dependency}==", nano_requirements, re.M)


def test_transfer_overlay_retains_exact_verified_wheel_urls() -> None:
    text = (ROOT / "cosmos2-transfer" / "security-overrides.txt").read_text()
    requirements = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    assert len(requirements) == len(set(requirements))
    assert all(
        re.fullmatch(
            r"https://files\.pythonhosted\.org/[^\s]+\.whl#sha256=[0-9a-f]{64}", line
        )
        for line in requirements
    )
    assert "pillow-12.3.0" in text
    assert "hydra_core-1.3.6" in text
    assert "msal-1.38.0" in text
    assert "protobuf-6.33.5" in text


@pytest.mark.parametrize(
    "wrong", [None, "torch", "torchvision", "torchcodec", "natten", "flash-attn"]
)
def test_cosmos3_accelerator_group_rejects_mixed_abis(
    monkeypatch: pytest.MonkeyPatch, wrong: str | None
) -> None:
    spec = importlib.util.spec_from_file_location(
        "cosmos3_verify_security", ROOT / "cosmos3" / "verify_env.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "torchcodec": "0.14.0+cu130",
        "natten": "0.21.6+cu130.torch213",
    }
    if wrong:
        versions[wrong] = "0.0.0"

    def version(name: str) -> str:
        if name not in versions:
            raise module.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(module.metadata, "version", version)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.13.0+cu130", version=SimpleNamespace(cuda="13.0")
        ),
    )
    monkeypatch.setitem(sys.modules, "natten", SimpleNamespace(__version__="0.21.6"))
    monkeypatch.setitem(
        sys.modules, "natten.functional", SimpleNamespace(attention=lambda: None)
    )
    if wrong:
        with pytest.raises(RuntimeError):
            module.check_torch_stack()
    else:
        assert "natten=" in module.check_torch_stack()


def test_python_ray_runtime_excludes_unused_java_worker_bundle() -> None:
    docker = (ROOT / "cosmos3-ray-serve" / "Dockerfile").read_text()
    verifier = (ROOT / "cosmos3-ray-serve" / "verify_env.py").read_text()
    assert '"jars" / "ray_dist.jar"' in docker
    assert "p.unlink(missing_ok=True)" in docker
    assert '"jars" / "ray_dist.jar"' in verifier
    assert "unused Java worker bundle remains" in verifier


@pytest.mark.parametrize("image", ["cosmos3", "cosmos3-ray-serve"])
def test_cosmos_images_patch_every_bootstrap_interpreter(image: str) -> None:
    docker = (ROOT / image / "Dockerfile").read_text()
    assert "docker/workbench/sonic/packaging-requirements.txt" in docker
    assert 'Path("/opt/uv/python").glob("*/bin/python3")' in docker
    assert '"--no-deps","--require-hashes"' in docker
    assert "flash-attn flash-attn-3-nv" in docker

"""SeedVR2 image provenance, no-weights, and publication-quarantine contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    SUPPORTED_TOOL_VERSIONS,
    UNVALIDATED_PUBLICATION_TOOLS,
)
from npa.smoke.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = ROOT / "npa" / "docker" / "workbench" / "seedvr2"


def test_seedvr2_image_is_registered_and_quarantined() -> None:
    assert CONTAINER_IMAGE_NAMES["seedvr2"] == "npa-seedvr2"
    assert SUPPORTED_TOOL_VERSIONS["seedvr2"] == "0.1.0-cu130-unbuilt"
    assert "seedvr2" in UNVALIDATED_PUBLICATION_TOOLS


def test_seedvr2_packaging_is_public_runtime_fetch_only() -> None:
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )
    entry = contract["images"]["seedvr2"]
    assert entry["redistribution"] == "public"
    assert entry["dockerfile"] == "seedvr2/Dockerfile"
    notes = entry["notes"].lower()
    assert "weights" in notes and "runtime" in notes
    assert "quarantined" in notes


def test_seedvr2_dockerfile_pins_source_and_refuses_weight_payloads() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text()
    assert "SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185" in dockerfile
    assert "APEX_SOURCE_REF=8a6508aaad6e75a2b939e33f308cd63d745d97f1" in dockerfile
    assert (
        "sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105"
        in dockerfile
    )
    assert "NPA_LIGHT_WORKBENCH_TOOL=seedvr2" in dockerfile
    assert "find /opt/seedvr2 -type f" in dockerfile
    assert "HF_TOKEN" not in dockerfile


def test_seedvr2_build_and_entrypoint_scripts_are_executable() -> None:
    assert os.access(DOCKER_DIR / "build.sh", os.X_OK)
    assert os.access(DOCKER_DIR / "entrypoint.sh", os.X_OK)
    assert "--push" in (DOCKER_DIR / "build.sh").read_text()
    assert (
        "Push requires the trusted publication workflow"
        in (DOCKER_DIR / "build.sh").read_text()
    )


def test_seedvr2_cuda_compilers_have_recorded_bounded_parallelism() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text()
    build_script = (DOCKER_DIR / "build.sh").read_text()

    assert "ARG FLASH_ATTN_MAX_JOBS=2" in dockerfile
    assert "ARG FLASH_ATTN_NVCC_THREADS=1" in dockerfile
    assert "ARG FLASH_ATTN_CUDA_ARCHS=90" in dockerfile
    assert "ARG APEX_MAX_JOBS=2" in dockerfile
    assert 'FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}"' in dockerfile
    assert 'MAX_JOBS="${FLASH_ATTN_MAX_JOBS}"' in dockerfile
    assert 'NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}"' in dockerfile
    assert 'MAX_JOBS="${APEX_MAX_JOBS}"' in dockerfile
    assert 'npa.build.flash-attn.cuda-archs="${FLASH_ATTN_CUDA_ARCHS}"' in dockerfile
    assert 'npa.build.flash-attn.max-jobs="${FLASH_ATTN_MAX_JOBS}"' in dockerfile
    assert (
        'npa.build.flash-attn.nvcc-threads="${FLASH_ATTN_NVCC_THREADS}"' in dockerfile
    )
    assert "MAX_JOBS=8" not in dockerfile

    assert '--build-arg "FLASH_ATTN_MAX_JOBS=$FLASH_ATTN_MAX_JOBS"' in build_script
    assert (
        '--build-arg "FLASH_ATTN_NVCC_THREADS=$FLASH_ATTN_NVCC_THREADS"' in build_script
    )
    assert '--build-arg "FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"' in build_script
    assert '--build-arg "APEX_MAX_JOBS=$APEX_MAX_JOBS"' in build_script

    requirements_end = dockerfile.index(
        "RUN curl --fail --location \\\n"
        '      "https://files.pythonhosted.org/packages/source/f/flash-attn/'
    )
    assert "seedvr2-requirements.lock" in dockerfile[:requirements_end]
    assert "flash-attn.tar.gz" not in dockerfile[:requirements_end]


def test_seedvr2_dependency_lock_uses_remediated_runtime_versions() -> None:
    requirements = (DOCKER_DIR / "requirements.in").read_text()
    lock = (DOCKER_DIR / "requirements.lock").read_text()

    for requirement in (
        "diffusers==0.38.0",
        "pillow==12.3.0",
        "safetensors==0.8.0",
        "setuptools==83.0.0",
        "torch==2.13.0",
        "torchvision==0.28.0",
    ):
        assert requirement in requirements
        assert requirement in lock
    assert "accelerate==" not in requirements
    assert "accelerate==" not in lock
    assert (
        'torchvision.__version__.startswith("0.28.0")'
        in (DOCKER_DIR / "Dockerfile").read_text()
    )


def test_seedvr2_blackwell_manifest_keeps_pending_build_unproven() -> None:
    manifest = json.loads(
        (ROOT / "npa/docker/workbench/blackwell-dc-images.json").read_text()
    )
    entry = next(row for row in manifest["images"] if row["name"] == "npa-seedvr2")
    assert entry["verdict"] == "unknown"
    assert entry["validation"] == "pending-build"
    assert "sm_90" in entry["build_target"]
    assert "sm_100" in entry["build_target"]
    assert "unproven" in entry["build_target"]


def test_seedvr2_golden_eval_runs_real_gpu_capability() -> None:
    golden = load_manifest()["seedvr2"].golden_eval
    assert golden.gpu == "required"
    assert golden.serverless_gpu == "h100"
    assert golden.module == "npa.smoke.test_seedvr2_functional"
    assert golden.status == "gpu-gated"

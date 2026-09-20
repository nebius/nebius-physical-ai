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
CONTAINER_TEMP_ROOT = Path("/") / "tmp"


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
    assert "cudnn" in notes and "separate devel stage" in notes
    assert "nvshmem" in notes
    assert "quarantined" in notes
    redistribution = (DOCKER_DIR / "REDISTRIBUTION.md").read_text()
    assert "development build stage that is absent" in redistribution
    assert "cudnn-runtime.json" in redistribution
    assert "nvshmem-runtime.json" in redistribution


def test_seedvr2_dockerfile_pins_source_and_refuses_weight_payloads() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text()
    assert "SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185" in dockerfile
    assert "APEX_SOURCE_REF=8a6508aaad6e75a2b939e33f308cd63d745d97f1" in dockerfile
    assert (
        "sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105"
        in dockerfile
    )
    assert "NPA_LIGHT_WORKBENCH_TOOL=seedvr2" in dockerfile
    assert "printf '%s\\n' \"$NPA_SOURCE_SHA\" > /opt/npa-source-revision" in dockerfile
    assert "LicenseRef-NVIDIA-CUDA-Toolkit" in dockerfile
    assert (
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@"
        "sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105 "
        "AS seedvr2-build"
    ) in dockerfile
    assert (
        "nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@"
        "sha256:4d242f206abc4b9588a6506cce2d88932cc879849395aae3785075179718cc49"
    ) in dockerfile
    assert "filter_cudnn_runtime.py" in dockerfile
    assert "filter_nvshmem_runtime.py" in dockerfile
    assert "cudnn-runtime.json" in dockerfile
    assert "nvshmem-runtime.json" in dockerfile
    assert "NVIDIA/nvshmem/v3.4.5-0/License.txt" in dockerfile
    assert (
        "1f5b7ada702926bc73327e6eb02dc2d41facc844cc4512ac900451bda06a459e" in dockerfile
    )
    assert "NVSHMEM-License-v3.4.5-0.txt" in dockerfile
    assert "COPY --from=seedvr2-build /opt/seedvr2-venv" in dockerfile
    build_stage, final_stage = dockerfile.split(
        "FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@", 1
    )
    assert "NPA_SOURCE_SHA" not in build_stage
    assert "SOURCE_DATE_EPOCH" not in build_stage
    assert final_stage.index("COPY src/npa") < final_stage.index("ARG NPA_SOURCE_SHA")
    assert final_stage.index("COPY src/npa") < final_stage.index(
        'LABEL org.opencontainers.image.revision="${NPA_SOURCE_SHA}"'
    )
    assert "/opt/seedvr2-venv/bin/pip check" in final_stage
    assert "from flash_attn import flash_attn_varlen_func" in final_stage
    assert "from apex.normalization import FusedLayerNorm" in final_stage
    assert "install -d -m 0700 -o ubuntu -g ubuntu /workspace/tmp" in final_stage
    assert "find /opt/seedvr2 -type f" in dockerfile
    assert "HF_TOKEN" not in dockerfile
    source_layer = dockerfile[
        dockerfile.index(
            "RUN curl --fail --location \\\n"
            '      "https://codeload.github.com/ByteDance-Seed/SeedVR/'
        ) : dockerfile.index("COPY pyproject.toml README.md")
    ]
    assert "/opt/seedvr2/neg_emb.pt" in source_layer
    assert "/opt/seedvr2/pos_emb.pt" in source_layer
    assert str(CONTAINER_TEMP_ROOT / "seedvr2.tar.gz") in source_layer
    assert source_layer.index("rm -f") < source_layer.index("find /opt/seedvr2")


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
    assert "scripts/measure_extension_arches.py" in dockerfile
    assert dockerfile.count("--skip-no-fatbin --exact sm_90 --json") == 2
    assert "--distribution flash-attn" in dockerfile
    assert "--distribution apex" in dockerfile
    assert (
        dockerfile.count(
            "/opt/seedvr2-venv/bin/python /opt/npa-tools/measure_extension_arches.py"
        )
        == 2
    )
    assert "/usr/share/doc/npa-seedvr2/extension-arches/flash-attn.json" in dockerfile
    assert "/usr/share/doc/npa-seedvr2/extension-arches/apex.json" in dockerfile
    assert "npa/scripts/measure_extension_arches.py" in build_script

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
    notices = (DOCKER_DIR / "THIRD_PARTY_NOTICES.md").read_text()

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
        if requirement.startswith(("diffusers", "safetensors", "torch")):
            assert f"`{requirement}`" in notices
    assert "accelerate==" not in requirements
    assert "accelerate==" not in lock
    assert (
        'torchvision.__version__.startswith("0.28.0")'
        in (DOCKER_DIR / "Dockerfile").read_text()
    )


def test_seedvr2_service_environment_is_hash_locked() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text()
    service_input = (DOCKER_DIR / "service-requirements.in").read_text()
    service_lock = (DOCKER_DIR / "service-requirements.lock").read_text()

    assert "hatchling==1.29.0" in service_input
    assert "uvicorn==0.52.4" in service_input
    assert "hatchling==1.29.0" in service_lock
    assert "uvicorn==0.52.4" in service_lock
    assert "--hash=sha256:" in service_lock
    assert "--require-hashes" in dockerfile
    assert str(CONTAINER_TEMP_ROOT / "seedvr2-service.lock") in dockerfile
    verifier = "/usr/bin/python3.12 -I -S /opt/npa-tools/verify_no_weight_payloads.py"
    assert "verify_no_weight_payloads.py" in dockerfile
    assert (
        "/opt/npa-venv/bin/python /opt/npa-tools/verify_no_weight_payloads.py"
        not in dockerfile
    )
    assert dockerfile.index("--run-id image-build --dry-run") < dockerfile.index(
        verifier
    )
    assert dockerfile.index("rm -rf /root/.cache") < dockerfile.index(verifier)
    assert "/usr/share/doc/npa-seedvr2/weight-payload-scan.json" in dockerfile
    assert 'test -z "$(find /opt/npa-src /opt/npa-venv' not in dockerfile
    assert "--no-build-isolation /opt/npa-src" in dockerfile
    assert "'uvicorn==" not in dockerfile


def test_seedvr2_blackwell_manifest_records_built_h100_arches() -> None:
    manifest = json.loads(
        (ROOT / "npa/docker/workbench/blackwell-dc-images.json").read_text()
    )
    entry = next(row for row in manifest["images"] if row["name"] == "npa-seedvr2")
    assert entry["verdict"] == "ready"
    assert entry["validation"] == "pending-hardware"
    assert entry["measured_torch"] == "2.13.0+cu130"
    assert "sm_100" in entry["measured_arch_list"]
    assert entry["measured_extension_sass"] == {
        "flash-attn": ["sm_90"],
        "apex": ["sm_90"],
    }
    assert "sm_90" in entry["build_target"]
    assert "sm_100" in entry["build_target"]
    assert "no PTX" in entry["build_target"]


def test_seedvr2_golden_eval_runs_real_gpu_capability() -> None:
    golden = load_manifest()["seedvr2"].golden_eval
    assert golden.gpu == "required"
    assert golden.serverless_gpu == "h100"
    assert golden.module == "npa.smoke.test_seedvr2_functional"
    assert golden.status == "needs-image-update"

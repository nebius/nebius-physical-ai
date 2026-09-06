from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/paidf-anomalygen-sky/Dockerfile"
ENTRYPOINT = ROOT / "npa/docker/workbench/paidf-anomalygen-sky/entrypoint.sh"


def test_paidf_anomalygen_sky_parent_and_runtime_are_fixed() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "nvcr.io" not in text
    assert "dbaf7d7d9003f048230f9026da5969e9e5931785" in text
    assert "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04@sha256:" in text
    assert "nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:" in text
    assert "USER ubuntu" in text
    assert "ENTRYPOINT [\"/usr/local/bin/npa-sky-entrypoint\"]" in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert "git -C /src fetch -q --depth 1 origin" in text
    assert "bash /tmp/build_wheels.sh --in-container" in text


def test_paidf_anomalygen_sky_bootstrap_source_is_complete() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for dependency in ("openssh-server", "rsync", "netcat-openbsd", "sudo"):
        assert dependency in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in dockerfile
    assert 'uv venv --python "${PYTHON_VERSION}" /opt/npa-venv' in dockerfile
    assert "uv pip install --python /opt/npa-venv/bin/python pip" in dockerfile
    assert "chown -R ubuntu:ubuntu /opt/npa-venv" in dockerfile
    assert 'export PATH="/opt/npa-venv/bin:$PATH"' in dockerfile
    assert "PATH=/opt/npa-venv/bin:${PATH}" in dockerfile
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'exec "$@"' in entrypoint
    assert "exec /bin/bash" in entrypoint


def test_paidf_anomalygen_does_not_retain_ssh_host_keys_in_install_layer() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    instructions = re.split(r"(?m)^(?=[A-Z]+\s)", dockerfile)
    install_layers = [
        instruction
        for instruction in instructions
        if instruction.startswith("RUN ") and "openssh-server" in instruction
    ]
    assert install_layers
    for instruction in install_layers:
        assert "rm -f /etc/ssh/ssh_host_*" in instruction
        assert instruction.index("openssh-server") < instruction.index(
            "rm -f /etc/ssh/ssh_host_*"
        )


def test_paidf_anomalygen_sky_preserves_cuda_forward_compatibility() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "CUDA_COMPAT_PATH=/usr/local/cuda/compat" in text
    assert (
        "LD_LIBRARY_PATH=${CUDA_COMPAT_PATH}:${NV_SITE}/cudnn/lib:"
        "${NV_SITE}/nccl/lib:${NV_SITE}/cusparselt/lib:${LD_LIBRARY_PATH}"
    ) in text
    assert "dpkg-query -W -f='${Version}' cuda-compat-13-2" in text
    assert '"595.58.03-1ubuntu1"' in text
    assert "libcuda.so.595.58.03" in text
    assert "libnvidia-ptxjitcompiler.so.595.58.03" in text


def test_paidf_anomalygen_security_patch_preserves_the_wheel_build() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    builder, runtime = text.split("FROM ${CUDA_RUNTIME_IMAGE} AS runtime", 1)
    assert "nltk" not in builder
    assert "nltk-3.10.3-py3-none-any.whl#sha256=" in runtime
    assert "ff9598a8e20518ee0d557745890cc4435b9578489e2dcbc69c4f81fa060caf7c" in runtime
    assert runtime.index("uv pip install -r /tmp/requirements-nodeps.txt") < runtime.index(
        "nltk-3.10.3-py3-none-any.whl#sha256="
    )


def test_paidf_wandb_security_update_includes_the_framework_compatibility_patch() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    builder, runtime = text.split("FROM ${CUDA_RUNTIME_IMAGE} AS runtime", 1)
    assert "wandb" not in builder
    wheel = "wandb-0.28.2-py3-none-manylinux_2_28_x86_64.whl#sha256="
    assert "1db698d107871c66b2dcbb0cf4dc2af1ddb159ba94e957e890158ec60ab2de54" in runtime
    assert "COPY paidf-anomalygen-sky/patch_wandb_run_id.py " in runtime
    patch = "python /usr/local/lib/npa/patch_wandb_run_id.py"
    assert runtime.index("uv pip install -r /tmp/requirements-nodeps.txt") < runtime.index(wheel)
    assert runtime.index(wheel) < runtime.index(patch)
    assert runtime.index(patch) < runtime.index("uv pip install -e . --no-deps")
    instructions = re.split(r"(?m)^(?=[A-Z]+\s)", runtime)
    requirement_layers = [
        instruction
        for instruction in instructions
        if instruction.startswith("RUN ") and "uv pip install -r /tmp/requirements.txt" in instruction
    ]
    assert len(requirement_layers) == 1
    assert wheel in requirement_layers[0]

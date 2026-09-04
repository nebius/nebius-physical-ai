ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-genesis@sha256:80cd1c4c7f7a5466533de29ee5cad1213202f348545346fea9ed6746fa03b1ca
FROM ${BASE_IMAGE} AS sanitized

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20260820T000000Z

WORKDIR /opt/npa
USER root
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-control-requirements.txt \
    /opt/npa/sim2real-control-requirements.txt
COPY docker/workbench/common/sim2real-genesis-requirements.txt \
    /opt/npa/sim2real-genesis-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint
COPY --chmod=0755 docker/workbench/common/install_workflow_runtime_prereqs.sh \
    /usr/local/bin/install-workflow-runtime-prereqs
COPY docker/workbench/common/sanitize_sim2real_envgen_parent.py \
    /usr/local/lib/npa/sanitize-sim2real-envgen-parent.py
COPY docker/workbench/common/envgen_compat /opt/npa/compat

# The public Genesis parent includes three payloads that are outside this
# capability's public packaging/security contract:
#
# * imageio-ffmpeg's wheel-bundled static FFmpeg;
# * the AGPL tetgen Python wrapper (used only for optional tetrahedralization);
# * optional W&B and Nsight Compute tooling, including separately compiled Go
#   binaries that are not part of canonical environment generation.
#
# Retain MoviePy while replacing imageio-ffmpeg with its hash-pinned
# pure-Python sdist. Remove the optional packages and their parent metadata
# before flattening the filesystem so their bytes cannot remain in a lower
# layer. Genesis 0.4.6 declares tetgen and LeRobot 0.4.4 declares W&B
# unconditionally, so remove those two Requires-Dist entries as part of
# constructing this deliberately capability-reduced runtime distribution.
RUN /usr/local/bin/install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip uninstall -y npa \
    && python -m pip uninstall -y transformers imageio-ffmpeg wandb tetgen \
    && python /usr/local/lib/npa/sanitize-sim2real-envgen-parent.py \
    && rm -rf /opt/nvidia/nsight-compute \
    && python -m pip install --no-deps \
      'imageio-ffmpeg @ https://files.pythonhosted.org/packages/44/bd/c3343c721f2a1b0c9fc71c1aebf1966a3b7f08c2eea8ed5437a2865611d6/imageio_ffmpeg-0.6.0.tar.gz#sha256=e2556bed8e005564a9f925bb7afa4002d82770d6b08825078b7697ab88ba1755' \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test -z "$(find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg python -c \
      'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/compat\\n/opt/npa/src\\n', encoding='utf-8')" \
    && env -u PYTHONPATH /opt/npa/venv/bin/python -c \
      'import boto3, genesis; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
    && python -c 'import importlib.metadata as m; assert not any(d.metadata["Name"].lower() == "transformers" for d in m.distributions())' \
    && python -c 'import importlib.metadata as m; names = {d.metadata["Name"].lower() for d in m.distributions()}; assert names.isdisjoint({"tetgen", "wandb"})' \
    && test ! -e /opt/nvidia/nsight-compute \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py

# Flatten the sanitized filesystem so the published image does not reference
# the Genesis parent layer that contained the removed static FFmpeg payload.
# COPY --from copies the merged filesystem, including whiteouts, while the
# scratch stage deliberately drops every parent blob from the final manifest.
FROM scratch AS runtime

ARG NPA_SOURCE_SHA
LABEL npa.tool="envgen" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

COPY --from=sanitized / /

# A scratch stage does not inherit image configuration. Preserve the CUDA,
# Genesis, cache, and workflow runtime contract explicitly.
ENV PATH=/opt/npa/venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.0.1 \
    TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3 12.0" \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/npa/hf_cache \
    HF_HUB_CACHE=/opt/npa/hf_cache/hub \
    TRANSFORMERS_CACHE=/opt/npa/hf_cache/transformers \
    NPA_INPUT_PATH=/input \
    NPA_OUTPUT_PATH=/output \
    INPUT_PATH=/input \
    OUTPUT_PATH=/output \
    GENESIS_VERSION=0.4.6 \
    LEROBOT_VERSION=0.4.4 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=device \
    NPA_GENESIS_HOME=/opt/genesis \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_BAKED_PYTHON=/opt/npa/venv/bin/python \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PYTHONPATH=/opt/npa/compat:/opt/npa/src

WORKDIR /opt/npa
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]

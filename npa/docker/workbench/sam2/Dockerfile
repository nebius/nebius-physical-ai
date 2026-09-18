# Public source-only job runtime. CUDA and model weights are fetched at runtime.
FROM ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837

ARG NPA_SOURCE_SHA
LABEL npa.tool="sam2" \
      npa.version="2b90b9f5ceec907a1c18123530e92e794ad901a4-runtime-fetch" \
      org.opencontainers.image.title="NPA sam2 native GPU workflow runtime" \
      org.opencontainers.image.description="Pinned OSS source and CPU dependencies; runtime-only CUDA and models" \
      npa.source_revision="${NPA_SOURCE_SHA}" \
      npa.byof.repo="https://github.com/facebookresearch/sam2.git" \
      npa.byof.ref="2b90b9f5ceec907a1c18123530e92e794ad901a4" \
      npa.source.repo="https://github.com/facebookresearch/sam2.git" \
      npa.source.ref="2b90b9f5ceec907a1c18123530e92e794ad901a4" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.weights="runtime-only"

USER root
WORKDIR /opt
COPY --chmod=0755 docker/workbench/common/model_source.sh /opt/npa/model_source.sh
RUN /opt/npa/model_source.sh \
      https://github.com/facebookresearch/sam2.git \
      2b90b9f5ceec907a1c18123530e92e794ad901a4 \
      sam2 setup.py pyproject.toml MANIFEST.in LICENSE LICENSE_cctorch README.md \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir --no-deps \
      protobuf==6.33.6 hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 iopath==0.1.10 portalocker==3.2.0 \
    && /opt/wan-base/bin/python -m pip freeze --all > /opt/byof/npa_baked_python_inventory.txt \
    && ! grep -Eiq '^(torch|torchvision|torchaudio|nvidia-)' /opt/byof/npa_baked_python_inventory.txt \
    && find /opt/wan-base -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/pip-*

COPY src/npa/solutions/video_generation.py src/npa/solutions/lingbot_camera.py \
     src/npa/solutions/perception_video.py src/npa/solutions/native_image_smoke.py \
     /opt/npa-native/npa/solutions/
COPY --chmod=0755 docker/workbench/common/model_runtime.sh /usr/local/bin/model-runtime
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-job-entrypoint
COPY --chmod=0444 docker/workbench/sam2/REDISTRIBUTION.md /usr/share/doc/npa-sam2/REDISTRIBUTION.md

USER ubuntu
WORKDIR /opt/byof
# Read-only probe of the documented SkyPilot passwordless_root_exemption.
# No installation or file mutation runs through sudo in this instruction.
# trivy:ignore:AVD-DS-0010
RUN model-runtime health \
    && test -w /workspace/.cache/npa/wan2-2/runtime \
    && sudo -n true \
    && command -v rsync \
    && test -x /usr/sbin/sshd
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["model-runtime", "health"]
ENTRYPOINT ["npa-job-entrypoint"]
CMD ["model-runtime", "health"]

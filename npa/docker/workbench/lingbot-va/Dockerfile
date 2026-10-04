# Source-only LingBot-VA runtime.  The parent is pinned for its neutral OS,
# ffmpeg, and SkyPilot bootstrap only; its CUDA 13 Wan runtime is deliberately
# not reused because the upstream LingBot-VA recipe pins PyTorch 2.9/CUDA 12.6.
FROM ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837

ARG NPA_SOURCE_SHA=unknown
ARG LINGBOT_VA_REF=7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
ARG LINGBOT_VA_URL=https://github.com/Robbyant/lingbot-va.git

LABEL npa.tool="lingbot-va" \
      npa.version="libero-long-runtime-fetch" \
      npa.source.repo="https://github.com/Robbyant/lingbot-va" \
      npa.source.ref="${LINGBOT_VA_REF}" \
      npa.model.base="robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c" \
      npa.model.libero_long="robbyant/lingbot-va-posttrain-libero-long@0e89d1e753019988aba484e8da2dc0810e264d9f" \
      npa.weights="runtime-only" \
      npa.dataset="operator-staged-only" \
      npa.cuda.runtime="runtime-fetch:torch-2.13.0-cu126" \
      org.opencontainers.image.title="NPA LingBot-VA LIBERO-Long runtime" \
      org.opencontainers.image.description="Pinned Apache-2.0 source; models, CUDA, and data are runtime-only" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV NPA_LINGBOT_VA_SOURCE=/opt/lingbot-va \
    NPA_LINGBOT_VA_RUNTIME_CACHE=/workspace/.cache/npa/lingbot-va/runtime \
    NPA_LINGBOT_VA_RUNTIME_REQUIREMENTS=/opt/npa/lingbot-va/runtime-requirements.txt \
    HF_HOME=/workspace/model-cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]
USER root
WORKDIR /opt

# Keep only source and project metadata from the exact Apache-2.0 revision.
# The checkout intentionally has no .git directory, LFS objects, weights, or
# dataset bytes in a layer.
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "${LINGBOT_VA_URL}" /tmp/lingbot-va \
    && git -C /tmp/lingbot-va checkout --detach "${LINGBOT_VA_REF}" \
    && test "$(git -C /tmp/lingbot-va rev-parse HEAD)" = "${LINGBOT_VA_REF}" \
    && install -d /opt/lingbot-va \
    && cp -a /tmp/lingbot-va/LICENSE.txt /tmp/lingbot-va/README.md /tmp/lingbot-va/INSTALL.md /tmp/lingbot-va/requirements.txt /tmp/lingbot-va/pyproject.toml /opt/lingbot-va/ \
    && cp -a /tmp/lingbot-va/wan_va /tmp/lingbot-va/evaluation /tmp/lingbot-va/script /opt/lingbot-va/ \
    && rm -rf /tmp/lingbot-va \
    && test -s /opt/lingbot-va/LICENSE.txt \
    && /opt/wan-base/bin/python -c 'from pathlib import Path; path = Path("/opt/lingbot-va/wan_va/modules/model.py"); source = path.read_text(); old = "try:\n    from flash_attn_interface import flash_attn_func\nexcept:\n    from flash_attn import flash_attn_func\n"; new = "try:\n    from flash_attn_interface import flash_attn_func\nexcept ImportError:\n    try:\n        from flash_attn import flash_attn_func\n    except ImportError:\n        flash_attn_func = None\n"; assert old in source; path.write_text(source.replace(old, new))' \
    && ! find /opt/lingbot-va -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' \) -print -quit | grep -q .

COPY --chmod=0444 docker/workbench/lingbot-va/runtime-requirements.txt /opt/npa/lingbot-va/runtime-requirements.txt
COPY --chmod=0755 docker/workbench/lingbot-va/lingbot_va_runtime.sh /usr/local/bin/lingbot-va-runtime
COPY --chmod=0755 docker/workbench/lingbot-va/lingbot_va_golden.sh /usr/local/bin/lingbot-va-golden
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-job-entrypoint
# `COPY --chmod` is intentional for source files, but its destination parents
# must remain traversable by the non-root workload user.
RUN install -d -m 0755 /opt/npa-native/npa/solutions /opt/npa-native/npa/clients /usr/share/doc/npa-lingbot-va
COPY --chmod=0444 src/npa/solutions/lingbot_va.py /opt/npa-native/npa/solutions/lingbot_va.py
# The LingBot adapter needs only four S3 methods.  Preserve credential lookup
# at runtime through Boto3's standard provider chain rather than copying the
# general storage client's explicit key-parameter source into image bytes.
COPY --chmod=0444 docker/workbench/lingbot-va/lingbot_va_storage.py /opt/npa-native/npa/clients/storage.py
COPY --chmod=0444 docker/workbench/lingbot-va/REDISTRIBUTION.md /usr/share/doc/npa-lingbot-va/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/lingbot-va/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-lingbot-va/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o ubuntu -g ubuntu \
      /workspace/.cache/npa/lingbot-va/runtime /workspace/model-cache/huggingface /opt/npa/lingbot-va \
    && lingbot-va-runtime health \
    && test -z "$(find /workspace/.cache/npa/lingbot-va/runtime -mindepth 1 -print -quit)" \
    && find /opt/lingbot-va -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/pip-*

VOLUME ["/workspace/.cache/npa/lingbot-va/runtime", "/workspace/model-cache/huggingface"]
USER ubuntu
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["lingbot-va-runtime", "health"]
ENTRYPOINT ["npa-job-entrypoint"]
CMD ["lingbot-va-runtime", "health"]

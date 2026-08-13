# npa-ltx2: LTX-2.5 video/audio world model, with none of LTX-2.5 in it.
#
# LTX-2.5 is not OSI open source. Its weights AND its ltx-core / ltx-pipelines
# source are all governed by the LTX-2.x Community License Agreement, whose
# Section 3 makes us responsible for imposing its use restrictions by contract on
# whoever we hand it to (3.1), delivering the full agreement to them (3.2), and
# not transferring to a Commercial Entity at all until they hold a paid licence
# (3.5). An anonymous `docker pull` establishes none of that.
#
# So this image bakes no LTX bytes. On first use it fetches upstream's pinned
# source and lets upstream's own uv resolution install the dependencies, and it
# fetches the gated weights with the operator's own Hugging Face token — after
# refusing unless the operator has declared their position under Sections 2.1 and
# 2.2. Lightricks delivers to each operator directly; we redistribute nothing of
# theirs. See REDISTRIBUTION.md for the full classification.
FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG LTX_SOURCE_URL=https://github.com/Lightricks/LTX-2.git
ARG LTX_SOURCE_REF=fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
ARG NPA_SOURCE_COMMIT=unknown
ARG NPA_RUNTIME_USER=ubuntu
ARG UV_VERSION=0.9.8

LABEL npa.tool="ltx2" \
      npa.version="2.5-rtfetch" \
      npa.source.repo="${LTX_SOURCE_URL}" \
      npa.source.ref="${LTX_SOURCE_REF}" \
      npa.source.provisioning="operator-accepted-runtime-fetch" \
      npa.weights="runtime-only" \
      npa.license.upstream="LTX-2.x Community License Agreement (2026-08-11), not OSI" \
      npa.license.payload="none-baked" \
      npa.security.runtime-user="uid-1000-with-skypilot-bootstrap-sudo" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    NPA_LTX_GATE=/opt/npa/ltx2/ltx_gate.py \
    NPA_LTX_RUNTIME_CACHE=/workspace/.cache/npa/ltx2/runtime \
    NPA_LTX_MODEL_CACHE=/workspace/model-cache/ltx-2.5 \
    NPA_LTX_SOURCE_REPO=${LTX_SOURCE_URL} \
    NPA_LTX_SOURCE_REF=${LTX_SOURCE_REF} \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git libgl1 libglib2.0-0 \
      openssh-server procps rsync sudo util-linux \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u "${NPA_RUNTIME_USER}" >/dev/null 2>&1 \
       || useradd -m -s /bin/bash -u 1000 "${NPA_RUNTIME_USER}") \
    && printf '%s\n' "${NPA_RUNTIME_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

# uv (Apache-2.0/MIT) drives upstream's own dependency resolution at run time,
# and huggingface_hub provides the `hf` CLI for the operator's gated weight pull.
# Neither carries any LTX or NVIDIA payload.
RUN python -m pip install --no-cache-dir \
      "uv==${UV_VERSION}" "huggingface-hub[cli]==0.36.0" \
    && rm -rf /root/.cache

# The licensing gate is the repo's own unit-tested module, copied verbatim. It is
# stdlib-only precisely so the tested code and the enforcing code are the same
# code; a shell re-implementation would be a second version of a legal control.
COPY --chmod=0444 src/npa/workbench/ltx2/licensing.py /opt/npa/ltx2/licensing.py
COPY --chmod=0555 docker/workbench/ltx2/ltx_gate.py /opt/npa/ltx2/ltx_gate.py
COPY --chmod=0755 docker/workbench/ltx2/ltx_runtime.sh /usr/local/bin/ltx-runtime
COPY --chmod=0755 docker/workbench/ltx2/entrypoint.sh /usr/local/bin/ltx-entrypoint
COPY --chmod=0755 docker/workbench/ltx2/smoke.sh /opt/npa/ltx2/smoke.sh
COPY --chmod=0444 docker/workbench/ltx2/REDISTRIBUTION.md /usr/share/doc/npa-ltx2/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/ltx2/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-ltx2/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace /workspace/.cache/npa/ltx2/runtime /workspace/model-cache/ltx-2.5 \
    && chmod 0755 /opt/npa \
    # Prove at build time that the refusal is real and that proving it fetched
    # nothing. `assert-refusal` exists rather than a bare `ltx-runtime ensure`
    # here: it scrubs the acceptance variables from its child environment, so the
    # build cannot succeed at downloading even if a builder leaked them, and the
    # build history stays free of the fetch verbs the payload scanner rejects.
    # The golden eval repeats this against the pushed image.
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'ltx-runtime health' \
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'ltx-runtime assert-refusal' \
      | grep -Fq NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK \
    && test -z "$(find /workspace -mindepth 2 -print -quit)" \
    # No LTX payload, no weights, no CUDA wheels may exist in any layer.
    && test -z "$(find / -xdev \( -name 'ltx_core' -o -name 'ltx_pipelines' \) -print -quit)" \
    && test -z "$(find / -xdev \( -name '*.safetensors' -o -name '*.ckpt' \) -print -quit)" \
    && test -z "$(find / -xdev -name 'nvidia' -type d -path '*site-packages*' -print -quit)"

VOLUME ["/workspace/.cache/npa/ltx2/runtime", "/workspace/model-cache/ltx-2.5"]
WORKDIR /workspace
USER ${NPA_RUNTIME_USER}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["ltx-runtime", "health"]
ENTRYPOINT ["ltx-entrypoint"]

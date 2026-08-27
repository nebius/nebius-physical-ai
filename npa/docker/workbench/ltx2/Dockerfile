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
# fetches the gated weights — both under the operator's own Hugging Face token,
# and neither without it. That token is the checkable part: Lightricks grants
# access to the gated repository only after a human accepts the terms there, and
# the agreement itself binds by conduct. Lightricks delivers to each operator
# directly; we redistribute nothing of theirs, and compliance with the agreement
# is the operator's own. See REDISTRIBUTION.md for the full classification.
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

# The decoded-video check is the repo's own unit-tested module, copied verbatim.
# It is stdlib-plus-ffmpeg only precisely so the tested code and the code that
# decides whether a run produced a real clip are the same code.
COPY --chmod=0444 src/npa/workbench/ltx2/video_check.py /opt/npa/ltx2/video_check.py
COPY --chmod=0555 docker/workbench/ltx2/validate_video.py /opt/npa/ltx2/validate_video.py
COPY --chmod=0755 docker/workbench/ltx2/ltx_runtime.sh /usr/local/bin/ltx-runtime
COPY --chmod=0755 docker/workbench/ltx2/entrypoint.sh /usr/local/bin/ltx-entrypoint
COPY --chmod=0755 docker/workbench/ltx2/smoke.sh /opt/npa/ltx2/smoke.sh
COPY --chmod=0444 docker/workbench/ltx2/REDISTRIBUTION.md /usr/share/doc/npa-ltx2/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/ltx2/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-ltx2/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace /workspace/.cache/npa/ltx2/runtime /workspace/model-cache/ltx-2.5 \
      /opt/byof \
    # The prebuilt BYOF runner needs provenance metadata, not the licensed
    # source tree. This NPA-owned JSON identifies what ltx-runtime will fetch
    # after the task receives the operator's entitlement.
    && printf '{"source":"operator-runtime-fetch","repo":"%s","ref":"%s"}\n' \
         "${LTX_SOURCE_URL}" "${LTX_SOURCE_REF}" \
         > /opt/byof/npa_source_metadata.json \
    && chown "${NPA_RUNTIME_USER}:${NPA_RUNTIME_USER}" \
         /opt/byof/npa_source_metadata.json \
    && chmod 0444 /opt/byof/npa_source_metadata.json \
    # BuildKit applies `COPY --chmod` to the parent directories it creates on the
    # way, so the 0444 file modes above also landed on /opt/npa/ltx2 and
    # /usr/share/doc/npa-ltx2 — directories with no execute bit, which the
    # runtime user cannot traverse. Every file inside then reads as Permission
    # denied, including the video check. Normalize the directories back.
    && chmod 0755 /opt/npa /opt/npa/ltx2 /usr/share/doc/npa-ltx2 \
    # Prove at build time that the refusal is real and that proving it fetched
    # nothing. `assert-refusal` exists rather than a bare `ltx-runtime ensure`
    # here: it scrubs the entitlement variables from its child environment, so
    # the build cannot succeed at downloading even if a builder leaked them, and
    # the build history stays free of the fetch verbs the payload scanner
    # rejects. The golden eval repeats this against the pushed image.
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'ltx-runtime health' \
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'ltx-runtime assert-refusal' \
      | grep -Fq NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK \
    # No files anywhere in the operator's workspace: the refusal must not have
    # written one. Directories are exempt because `install -d` above creates the
    # empty cache mount points, which are deeper than the tree this once
    # searched — so the old depth-based form failed on the image's own layout.
    && test -z "$(find /workspace -type f -print -quit)" \
    # The output validator decides whether a GPU run passed, so prove here that
    # it can tell a real clip from a plausible-looking one. ffmpeg's own synthetic
    # sources supply both cases; no LTX bytes are involved and both files are
    # removed within this layer.
    && ffmpeg -v error -f lavfi -i life=size=64x64:rate=24 -t 2 \
         -pix_fmt yuv420p /tmp/npa-ltx-moving.mp4 \
    && ffmpeg -v error -f lavfi -i color=c=black:size=64x64:rate=24:duration=2 \
         -pix_fmt yuv420p /tmp/npa-ltx-flat.mp4 \
    # The third case: textured but frozen. The validator advertises rejecting a
    # still repeated, and only the flat and moving cases were ever proved here.
    && ffmpeg -v error -f lavfi -i testsrc=size=64x64:rate=24:duration=1 \
         -frames:v 1 -pix_fmt yuv420p /tmp/npa-ltx-one.png \
    && ffmpeg -v error -loop 1 -i /tmp/npa-ltx-one.png -t 3 -r 24 \
         -pix_fmt yuv420p /tmp/npa-ltx-still.mp4 \
    && /opt/npa/ltx2/validate_video.py --video /tmp/npa-ltx-moving.mp4 >/dev/null \
    && ! /opt/npa/ltx2/validate_video.py --video /tmp/npa-ltx-flat.mp4 >/dev/null 2>&1 \
    && ! /opt/npa/ltx2/validate_video.py --video /tmp/npa-ltx-still.mp4 >/dev/null 2>&1 \
    && rm -f /tmp/npa-ltx-moving.mp4 /tmp/npa-ltx-flat.mp4 \
         /tmp/npa-ltx-still.mp4 /tmp/npa-ltx-one.png \
    # No LTX payload, no weights, no CUDA wheels may exist in any layer.
    && test -z "$(find / -xdev \( -name 'ltx_core' -o -name 'ltx_pipelines' \) -print -quit)" \
    && test -z "$(find / -xdev \( -name '*.safetensors' -o -name '*.ckpt' \) -print -quit)" \
    && test -z "$(find / -xdev -name 'nvidia' -type d -path '*site-packages*' -print -quit)"

VOLUME ["/workspace/.cache/npa/ltx2/runtime", "/workspace/model-cache/ltx-2.5"]
WORKDIR /workspace
USER ${NPA_RUNTIME_USER}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["ltx-runtime", "health"]
ENTRYPOINT ["ltx-entrypoint"]

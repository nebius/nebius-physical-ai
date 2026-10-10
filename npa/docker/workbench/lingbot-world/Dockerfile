# Public source-only job runtime. CUDA and model weights are fetched at runtime.
# The immutable default is the accepted public Wan parent. A separately scanned,
# operator-private parent may be supplied only for private qualification.
ARG WAN_BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837
ARG DEBIAN_SNAPSHOT=20261010T000000Z
ARG DEBIAN_PERL_SECURITY_VERSION=5.36.0-7+deb12u4

# Build PyAV from source against the parent image's audited system FFmpeg.  The
# PyPI wheel bundles its own FFmpeg-family libraries, including forbidden CUDA
# linkage and secret-shaped literal bytes that must not enter the final image.
FROM ${WAN_BASE_IMAGE} AS pyav-wheel-builder

ARG DEBIAN_SNAPSHOT
USER root
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/" \
      'Suites: bookworm bookworm-updates' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/" \
      'Suites: bookworm-security' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian-security.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libavcodec-dev \
      libavdevice-dev \
      libavfilter-dev \
      libavformat-dev \
      libavutil-dev \
      libswresample-dev \
      libswscale-dev \
    && /opt/wan-base/bin/python -m pip wheel --no-cache-dir --no-deps \
      --no-binary=av --wheel-dir /opt/npa-pyav-wheel av==17.1.0 \
    && test -n "$(find /opt/npa-pyav-wheel -maxdepth 1 -type f -name 'av-17.1.0-*.whl' -print -quit)"

FROM ${WAN_BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG DEBIAN_SNAPSHOT
ARG DEBIAN_PERL_SECURITY_VERSION
LABEL npa.tool="lingbot-world" \
      npa.version="a43bec7f8091c83e9b30b16b912f6fc906236fa6-runtime-fetch" \
      org.opencontainers.image.title="NPA lingbot-world native GPU workflow runtime" \
      org.opencontainers.image.description="Pinned OSS source and CPU dependencies; runtime-only CUDA and models" \
      npa.source_revision="${NPA_SOURCE_SHA}" \
      npa.byof.repo="https://github.com/Robbyant/lingbot-world.git" \
      npa.byof.ref="a43bec7f8091c83e9b30b16b912f6fc906236fa6" \
      npa.source.repo="https://github.com/Robbyant/lingbot-world.git" \
      npa.source.ref="a43bec7f8091c83e9b30b16b912f6fc906236fa6" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.weights="runtime-only"

USER root
# The digest-pinned Wan parent can retain vulnerable Bookworm Perl packages.
# Use the dated archives so the fixed revision remains available before
# fetching the pinned LingBot source.
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/" \
      'Suites: bookworm bookworm-updates' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
      'Types: deb' \
      "URIs: https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/" \
      'Suites: bookworm-security' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/debian-security.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade \
      "perl=${DEBIAN_PERL_SECURITY_VERSION}" \
      "perl-base=${DEBIAN_PERL_SECURITY_VERSION}" \
      "libperl5.36=${DEBIAN_PERL_SECURITY_VERSION}" \
      "perl-modules-5.36=${DEBIAN_PERL_SECURITY_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY --from=pyav-wheel-builder /opt/npa-pyav-wheel /opt/npa-pyav-wheel
COPY --chmod=0755 docker/workbench/common/model_source.sh /opt/npa/model_source.sh
RUN /opt/npa/model_source.sh \
      https://github.com/Robbyant/lingbot-world.git \
      a43bec7f8091c83e9b30b16b912f6fc906236fa6 \
      wan generate.py generate_fast.py pyproject.toml requirements.txt LICENSE.txt README.md \
    && cp -a /opt/npa/wan2-2/source/wan/utils /opt/byof/wan/ \
    && sed -i \
      's/^from easydict import EasyDict$/from wan.utils.easydict import EasyDict/' \
      /opt/byof/wan/configs/*.py \
    && sed -i '/^easydict$/d' /opt/byof/requirements.txt \
    && sed -i '/"easydict",/d' /opt/byof/pyproject.toml \
    && ! grep -Rqs '^from easydict import EasyDict$' /opt/byof/wan/configs \
    && ! grep -qx 'easydict' /opt/byof/requirements.txt \
    && ! grep -Fq '"easydict"' /opt/byof/pyproject.toml \
    && /opt/wan-base/bin/python -c 'import runpy; cls = runpy.run_path("/opt/byof/wan/utils/easydict.py")["EasyDict"]; assert cls({"nested": {"value": 7}}).nested.value == 7' \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir --no-deps \
      protobuf==6.33.6 scipy==1.15.3 /opt/npa-pyav-wheel/av-17.1.0-*.whl \
    && /opt/wan-base/bin/python -m pip freeze --all > /opt/byof/npa_baked_python_inventory.txt \
    && ! grep -Eiq '^(torch|torchvision|torchaudio|nvidia-)' /opt/byof/npa_baked_python_inventory.txt \
    && find /opt/wan-base -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf /opt/npa-pyav-wheel /root/.cache /tmp/pip-*

COPY src/npa/solutions/video_generation.py src/npa/solutions/lingbot_camera.py \
     src/npa/solutions/perception_video.py src/npa/solutions/native_image_smoke.py \
     /opt/npa-native/npa/solutions/
COPY --chmod=0755 docker/workbench/common/model_runtime.sh /usr/local/bin/model-runtime
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-job-entrypoint
COPY --chmod=0444 docker/workbench/lingbot-world/REDISTRIBUTION.md /usr/share/doc/npa-lingbot-world/REDISTRIBUTION.md

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

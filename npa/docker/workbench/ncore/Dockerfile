# syntax=docker/dockerfile:1.7
# CPU-only NVIDIA NCore COLMAP ingestion. Build context: npa/ from SOURCE_SHA.
# Reuse the supported Python base and immutable Debian security snapshot pattern.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime-base
ARG DEBIAN_SNAPSHOT=20260906T183022Z
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN sed -ri \
      -e "s|https?://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" \
      -e "s|https?://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' >/etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl netcat-openbsd openssh-server patch procps rsync sudo \
      libblas3 liblapack3 libgfortran5 libquadmath0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && visudo -cf /etc/sudoers.d/ubuntu \
    && install -d -m 0755 /run/sshd \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
      >/etc/ssh/sshd_config.d/npa.conf \
    && install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache

# Optional read-only BuildKit named context for already verified source archives.
# Default builds fetch the same hash-pinned public bytes.
FROM scratch AS native-source-cache
FROM runtime-base AS native-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ gfortran gpg make nasm ninja-build pkg-config libblas-dev liblapack-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
COPY docker/workbench/ncore/assemble_native_sources.py docker/workbench/ncore/native_build_support.py docker/workbench/ncore/build_ffmpeg.py \
     docker/workbench/ncore/native-source-lock.json docker/workbench/ncore/native-*-build.lock /build/
RUN --mount=type=bind,from=native-source-cache,target=/source-cache \
    python /build/assemble_native_sources.py --lock /build/native-source-lock.json \
      --cache /source-cache --output /opt/ncore/native-sources
RUN gpg --dearmor </opt/ncore/native-sources/ffmpeg-devel.asc >/build/ffmpeg-release.gpg \
    && gpgv --keyring /build/ffmpeg-release.gpg /opt/ncore/native-sources/ffmpeg-8.1.1.tar.xz.asc \
      /opt/ncore/native-sources/ffmpeg-8.1.1.tar.xz \
    && python /build/build_ffmpeg.py
COPY docker/workbench/ncore/build_native.py /build/
RUN python /build/build_native.py converter
RUN python /build/build_native.py reader \
    && cp /build/build_native.py /build/build_ffmpeg.py /build/native_build_support.py /build/assemble_native_sources.py /build/native-*-build.lock /opt/ncore/native-sources/

FROM runtime-base AS assembled
COPY --from=native-builder /opt/ncore /opt/ncore
COPY --from=native-builder /build/ncore-native/wheels /build/native-wheels
RUN printf '%s\n' /opt/ncore/ffmpeg/lib >/etc/ld.so.conf.d/ncore-ffmpeg.conf && ldconfig
COPY docker/workbench/ncore/source-lock.json docker/workbench/ncore/stage_upstream.py /build/
RUN python -B /build/stage_upstream.py --lock /build/source-lock.json --output /opt/ncore/src

COPY docker/workbench/ncore/*requirements.lock /build/
# Build tools are hash-locked too. asciitree's pure-Python sdist is built with
# these installed tools, never an unconstrained isolated build environment.
RUN python -m pip install --require-hashes -r /build/build-requirements.lock \
    && python -m venv --without-pip /opt/ncore/converter-venv \
    && python -m venv --without-pip /opt/venv \
    && python -m pip --python /opt/ncore/converter-venv install --require-hashes -r /build/build-requirements.lock \
    && python -m pip --python /opt/ncore/converter-venv install --no-deps --no-index /build/native-wheels/converter/*.whl \
    && python -m pip --python /opt/ncore/converter-venv install --no-binary=numpy,scipy,av --no-build-isolation --require-hashes -r /build/converter-requirements.lock \
    && python -m pip --python /opt/venv install --require-hashes -r /build/build-requirements.lock \
    && python -m pip --python /opt/venv install --no-deps --no-index /build/native-wheels/reader/*.whl \
    && python -m pip --python /opt/venv install --no-binary=numpy,scipy,av --no-build-isolation --require-hashes -r /build/npa-requirements.lock \
    && printf '%s\n' /opt/ncore/src/ncore /opt/ncore/src/pycolmap /opt/ncore/src/pycolmap/pycolmap \
      >/opt/ncore/converter-venv/lib/python3.12/site-packages/ncore-source.pth \
    && printf '%s\n' /opt/ncore/src/ncore /opt/ncore/src/pycolmap /opt/ncore/src/pycolmap/pycolmap \
      >/opt/venv/lib/python3.12/site-packages/ncore-source.pth

COPY pyproject.toml /opt/npa/pyproject.toml
COPY src/npa /opt/npa/src/npa
RUN /opt/venv/bin/python -m pip install --no-deps --no-build-isolation /opt/npa \
    && /opt/venv/bin/python -m pip check \
    && /opt/ncore/converter-venv/bin/python -m pip check

FROM runtime-base
ARG NPA_SOURCE_SHA
ARG SOURCE_SHA=${NPA_SOURCE_SHA}
ENV PATH=/opt/venv/bin:/opt/ncore/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/ubuntu \
    NPA_NCORE_CONVERTER=/opt/ncore/bin/colmap-convert \
    NPA_NCORE_SOURCE_REVISION=59c698d206da92b406a4f72619fce3b3a2c64bfd \
    NPA_SOURCE_SHA=${SOURCE_SHA}
COPY --from=assembled /opt/ncore /opt/ncore
COPY --from=assembled /opt/venv /opt/venv
COPY --from=assembled /opt/npa /opt/npa
COPY --from=assembled /etc/ld.so.conf.d/ncore-ffmpeg.conf /etc/ld.so.conf.d/ncore-ffmpeg.conf
COPY --chmod=0644 docker/workbench/ncore/Dockerfile docker/workbench/ncore/debian-native-lock.json docker/workbench/ncore/NATIVE-SOURCES.md /opt/ncore/native-sources/
RUN chmod -R a+rX /opt/ncore/native-sources && ldconfig
COPY --from=assembled /build/*requirements.lock /usr/share/doc/npa-ncore/
COPY --from=assembled /opt/ncore/src/ncore/LICENSE /usr/share/doc/npa-ncore/LICENSE-APACHE-2.0
COPY docker/workbench/ncore/source-lock.json docker/workbench/ncore/REDISTRIBUTION.md /usr/share/doc/npa-ncore/
COPY --chmod=0755 docker/workbench/ncore/colmap-convert /opt/ncore/bin/colmap-convert
COPY --chmod=0755 docker/workbench/ncore/verify_native.py /opt/ncore/bin/verify_native.py
COPY --chmod=0755 docker/workbench/ncore/verify-packaging.py /opt/ncore/bin/verify-packaging.py
COPY --chmod=0755 docker/workbench/ncore/entrypoint.sh /opt/ncore/bin/entrypoint.sh
# Tests run the actual SSH service lifecycle under the final uid, then remove
# generated host keys in this same layer. The label attests this proven behavior.
RUN /opt/venv/bin/python -c 'import os,re; assert re.fullmatch("[0-9a-f]{40}", os.environ["NPA_SOURCE_SHA"])' \
    && printf '%s\n' "${SOURCE_SHA}" >/usr/share/doc/npa-ncore/npa-source-sha \
    && su ubuntu -s /bin/bash -c '/opt/ncore/bin/entrypoint.sh /bin/true && sudo -n /usr/sbin/sshd -t && sudo -n service ssh restart && sudo -n service ssh stop && rsync --version >/dev/null && test -w /tmp && test -w "$HOME"' \
    && rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -name 'ssh_host_*' -print -quit)"
USER ubuntu
WORKDIR /workspace
RUN /opt/venv/bin/python /opt/ncore/bin/verify-packaging.py \
    && /opt/venv/bin/python /opt/ncore/bin/verify_native.py >/tmp/native-reader.json \
    && /opt/ncore/converter-venv/bin/python /opt/ncore/bin/verify_native.py >/tmp/native-converter.json \
    && sudo mv /tmp/native-reader.json /tmp/native-converter.json /opt/ncore/native-sources/
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      npa.source_revision="${SOURCE_SHA}" \
      npa.tool="ncore" npa.runtime="cpu" \
      npa.ncore.revision="59c698d206da92b406a4f72619fce3b3a2c64bfd"
ENTRYPOINT ["/opt/ncore/bin/entrypoint.sh"]
CMD ["/opt/venv/bin/python", "/opt/ncore/bin/verify-packaging.py"]

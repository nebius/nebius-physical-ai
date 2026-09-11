# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986
ARG APT_SNAPSHOT=20260903T121500Z
ARG CA_DEB_SHA256=6e8cdcc8c86103acd4fc14649eac62ff2037108389074a7b167567af33c32245

FROM scratch AS npa-ca-bootstrap
ARG APT_SNAPSHOT
ARG CA_DEB_SHA256
ADD --checksum=sha256:${CA_DEB_SHA256} \
    http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/pool/main/c/ca-certificates/ca-certificates_20260601~22.04.1_all.deb \
    /ca-certificates.deb

FROM ${BASE_IMAGE} AS build
ARG APT_SNAPSHOT
ARG NPA_SOURCE_SHA
USER root
RUN --mount=from=npa-ca-bootstrap,source=/ca-certificates.deb,target=/tmp/ca-certificates.deb,ro \
    test -n "${NPA_SOURCE_SHA}" && printf '%s' "${NPA_SOURCE_SHA}" | grep -Eq '^[0-9a-f]{40}$' && \
    dpkg-deb -x /tmp/ca-certificates.deb / && \
    update-ca-certificates && \
    printf '%s\n' \
      'Types: deb' \
      'URIs: https://snapshot.ubuntu.com/ubuntu/' \
      "Suites: ${APT_SNAPSHOT}" \
      'Components: main universe' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      'Check-Valid-Until: no' \
      > /etc/apt/sources.list.d/npa-snapshot.sources && \
    rm -f /etc/apt/sources.list && \
    apt-get update && \
    installed_perl="$(dpkg-query -W -f='${Version}' perl-base)" && \
    candidate_perl_base="$(apt-cache policy perl-base | sed -n 's/^  Candidate: //p')" && \
    candidate_perl="$(apt-cache policy perl | sed -n 's/^  Candidate: //p')" && \
    test "$installed_perl" = '5.34.0-3ubuntu1.8' && \
    test "$installed_perl" = "$candidate_perl_base" && \
    test "$installed_perl" = "$candidate_perl" && \
    apt-get install -y --no-install-recommends \
      build-essential=12.9ubuntu3 \
      ca-certificates=20260601~22.04.1 \
      libegl1=1.4.0-1 \
      libegl1-mesa-dev=23.2.1-1ubuntu3.1~22.04.4 \
      libgl1=1.4.0-1 \
      libgl1-mesa-dev=23.2.1-1ubuntu3.1~22.04.4 \
      libglm-dev=0.9.9.8+ds-2 \
      libglvnd-dev=1.4.0-1 \
      libjpeg-dev=8c-2ubuntu10 \
      libomp-dev=1:14.0-55~exp2 \
      pkg-config=0.29.2-1ubuntu3 \
      python3-dev=3.10.6-1~22.04 \
      python3-venv=3.10.6-1~22.04.1 && \
    ! grep -R -nE '^[[:space:]]*URIs:[[:space:]]*http://' /etc/apt/sources.list.d && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/npa-build
COPY docker/workbench/habitat-sim/source-manifest.json source-manifest.json
COPY docker/workbench/habitat-sim/licenses.json licenses.json
COPY docker/workbench/habitat-sim/apt-build.lock apt-build.lock
COPY docker/workbench/habitat-sim/apt-runtime.lock apt-runtime.lock
COPY docker/workbench/habitat-sim/requirements-build.lock requirements-build.lock
COPY docker/workbench/habitat-sim/requirements-runtime.lock requirements-runtime.lock
COPY docker/workbench/habitat-sim/runtime-payload.json runtime-payload.json
COPY docker/workbench/habitat-sim/prepare_source.py prepare_source.py
RUN python3 prepare_source.py --manifest source-manifest.json --output /opt/habitat-sim \
      --inventory-output /opt/npa-build/source-projection.json && \
    python3 -m venv /opt/build-venv && \
    /opt/build-venv/bin/pip install --disable-pip-version-check --no-cache-dir \
      --only-binary=:all: --no-deps --require-hashes -r requirements-build.lock && \
    /opt/build-venv/bin/pip install --disable-pip-version-check --no-cache-dir \
      --only-binary=:all: --no-deps --require-hashes -r requirements-runtime.lock && \
    cd /opt/habitat-sim && \
    HABITAT_BUILD_GUI_VIEWERS=OFF \
    HABITAT_WITH_BULLET=ON \
    HABITAT_WITH_CUDA=OFF \
    HABITAT_WITH_AUDIO=OFF \
    HABITAT_BUILD_TEST=OFF \
    HABITAT_BUILD_BASIS_COMPRESSOR=OFF \
    SKBUILD_CMAKE_BUILD_TYPE=Release \
      /opt/build-venv/bin/pip wheel . --no-build-isolation --no-deps \
      --wheel-dir /opt/wheelhouse && \
    python3 -m venv /opt/runtime-venv && \
    /opt/runtime-venv/bin/pip install --disable-pip-version-check --no-cache-dir \
      --only-binary=:all: --no-deps --require-hashes -r /opt/npa-build/requirements-runtime.lock && \
    /opt/runtime-venv/bin/pip install --disable-pip-version-check --no-cache-dir \
      --no-deps /opt/wheelhouse/habitat_sim-*.whl && \
    /opt/runtime-venv/bin/pip check && \
    find /opt/runtime-venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg-*' -delete && \
    ! find /opt/runtime-venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg-*' -print -quit | grep -q . && \
    mkdir -p /opt/source-projection/data /opt/source-projection/src && \
    cp /opt/habitat-sim/LICENSE /opt/habitat-sim/MANIFEST.in \
       /opt/habitat-sim/README.md /opt/habitat-sim/pyproject.toml \
       /opt/habitat-sim/setup.py /opt/source-projection/ && \
    cp /opt/habitat-sim/data/default.physics_config.json /opt/source-projection/data/ && \
    cp -a /opt/habitat-sim/src/CMakeLists.txt /opt/habitat-sim/src/cmake \
       /opt/habitat-sim/src/esp /opt/habitat-sim/src_python /opt/source-projection/ && \
    cp -a /opt/habitat-sim/src/deps /opt/source-projection/src/ && \
    mkdir -p /opt/notices && \
    cp source-manifest.json source-projection.json licenses.json apt-build.lock apt-runtime.lock \
       requirements-build.lock requirements-runtime.lock runtime-payload.json /opt/notices/

FROM ${BASE_IMAGE} AS runtime
ARG APT_SNAPSHOT
ARG NPA_SOURCE_SHA
USER root
RUN --mount=from=npa-ca-bootstrap,source=/ca-certificates.deb,target=/tmp/ca-certificates.deb,ro \
    dpkg-deb -x /tmp/ca-certificates.deb / && \
    update-ca-certificates && \
    printf '%s\n' \
      'Types: deb' \
      'URIs: https://snapshot.ubuntu.com/ubuntu/' \
      "Suites: ${APT_SNAPSHOT}" \
      'Components: main universe' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      'Check-Valid-Until: no' \
      > /etc/apt/sources.list.d/npa-snapshot.sources && \
    rm -f /etc/apt/sources.list && \
    apt-get update && \
    installed_perl="$(dpkg-query -W -f='${Version}' perl-base)" && \
    candidate_perl_base="$(apt-cache policy perl-base | sed -n 's/^  Candidate: //p')" && \
    candidate_perl="$(apt-cache policy perl | sed -n 's/^  Candidate: //p')" && \
    test "$installed_perl" = '5.34.0-3ubuntu1.8' && \
    test "$installed_perl" = "$candidate_perl_base" && \
    test "$installed_perl" = "$candidate_perl" && \
    apt-get install -y --no-install-recommends \
      ca-certificates=20260601~22.04.1 \
      libegl1=1.4.0-1 \
      libgl1=1.4.0-1 \
      libglvnd0=1.4.0-1 \
      libjpeg-turbo8=2.1.2-0ubuntu1 \
      libomp5-14=1:14.0.0-1ubuntu1.1 \
      netcat-openbsd=1.218-4ubuntu1 \
      openssh-client=1:8.9p1-3ubuntu0.16 \
      openssh-server=1:8.9p1-3ubuntu0.16 \
      python3=3.10.6-1~22.04.1 \
      rsync=3.2.7-0ubuntu0.22.04.7 \
      sudo=1.9.9-1ubuntu2.6 && \
    ! grep -R -nE '^[[:space:]]*URIs:[[:space:]]*http://' /etc/apt/sources.list.d && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/cache/debconf/* && \
    rm -f /etc/ssh/ssh_host_* && \
    install -d -o ubuntu -g ubuntu -m 0700 /home/ubuntu/.ssh && \
    printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/90-npa-skypilot && \
    chmod 0440 /etc/sudoers.d/90-npa-skypilot

COPY --from=build /opt/runtime-venv /opt/venv
COPY --from=build /opt/source-projection /usr/src/habitat-sim
COPY --from=build /opt/notices /usr/share/doc/npa-habitat-sim
COPY src/npa/__init__.py /opt/npa-runtime/npa/__init__.py
COPY src/npa/workflows/__init__.py /opt/npa-runtime/npa/workflows/__init__.py
COPY src/npa/workflows/habitat_sim_smoke.py /opt/npa-runtime/npa/workflows/habitat_sim_smoke.py
COPY docker/workbench/habitat-sim/entrypoint.sh /usr/local/bin/npa-habitat-entrypoint
COPY docker/workbench/habitat-sim/REDISTRIBUTION.md /usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md
RUN chmod 0755 /usr/local/bin/npa-habitat-entrypoint && \
    chown -R root:root /opt/venv /opt/npa-runtime /usr/src/habitat-sim \
      /usr/share/doc/npa-habitat-sim && \
    ! find / -xdev \( -name 'habitat-test-scenes.zip' -o -name 'skokloster-castle.glb' \
      -o -name 'skokloster-castle.navmesh' -o -name '.git' -o -name '.gitconfig' \) \
      -print -quit | grep -q .

ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/npa-runtime \
    MAGNUM_LOG=verbose \
    HABITAT_SIM_LOG=verbose \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
LABEL org.opencontainers.image.source="https://github.com/facebookresearch/habitat-sim" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="MIT" \
      npa.tool="habitat-sim" \
      npa.version="0.3.3-public-unbuilt" \
      org.nebius.npa.upstream-revision="57ee4941dc4765240f0f91f70b2c97a919bf9038" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      org.nebius.npa.publication-status="quarantined-unbuilt"
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-habitat-entrypoint"]
CMD ["python3", "-m", "npa.workflows.habitat_sim_smoke", "--help"]

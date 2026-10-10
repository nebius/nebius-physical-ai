# MJLab and its CUDA wheel closure; no checkpoints, motion data or credentials.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ARG NPA_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
ARG DEBIAN_SNAPSHOT=20261010T000000Z
LABEL npa.tool="mjlab" \
      npa.version="1.6.0-cuda13" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"
ENV PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NPA_BAKED_PYTHON=/usr/local/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    XDG_CACHE_HOME=/workspace/.cache
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
COPY docker/workbench/habitat-sim/bootstrap_sources.py /opt/mjlab-source/bootstrap_sources.py
COPY docker/workbench/mjlab/python_sources.py docker/workbench/mjlab/python-sources.lock.json /opt/mjlab-source/
RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SOURCE_DATE_EPOCH" =~ ^[1-9][0-9]*$ ]] \
    && install -d -m 0755 /usr/share/doc/npa-mjlab \
    && cp /var/lib/dpkg/status /usr/share/doc/npa-mjlab/base-status \
    && rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
       "deb https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm main" \
       "deb https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm-updates main" \
       "deb https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/ bookworm-security main" \
       "deb-src https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm main" \
       "deb-src https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm-updates main" \
       "deb-src https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/ bookworm-security main" \
       "deb-src https://snapshot.debian.org/archive/debian/20260814T000000Z/ bookworm main" \
       "deb-src https://snapshot.debian.org/archive/debian-security/20260814T000000Z/ bookworm-security main" \
       > /etc/apt/sources.list.d/npa-snapshot.list \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot \
    && apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl netcat-openbsd procps rsync sudo openssh-server wget \
       libegl1 libgl1 libglvnd0 libglib2.0-0 ffmpeg git \
    && cp /var/lib/dpkg/status /usr/share/doc/npa-mjlab/runtime-status \
    && python3 /opt/mjlab-source/bootstrap_sources.py fetch /usr/share/doc/npa-mjlab \
    && python3 /opt/mjlab-source/python_sources.py --lock /opt/mjlab-source/python-sources.lock.json \
       --output /usr/share/doc/npa-mjlab/python-sources \
    && rm -f /etc/ssh/ssh_host_* \
    && install -d -m 0755 /run/sshd /usr/share/doc/npa-mjlab/notices \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
       > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime
COPY docker/workbench/mjlab/requirements.lock /opt/mjlab-requirements.lock
# This shared wheel has the same exact version as cuRobo's audited runtime.
COPY docker/workbench/curobo/filter_cudnn_runtime.py /opt/filter_cudnn_runtime.py
COPY docker/workbench/open3d/notices/mcap-LICENSE.txt /usr/share/doc/npa-mjlab/notices/mcap-LICENSE.txt
RUN pip install --no-cache-dir --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/mjlab-requirements.lock \
    && python3 /opt/filter_cudnn_runtime.py > /usr/share/doc/npa-mjlab/cudnn-runtime.json \
    && find /usr/local/lib -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test -z "$(find /usr/local/lib -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && python3 -c 'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && python3 -c 'import imageio_ffmpeg, os; p="/tmp/mjlab-ffmpeg-probe.mp4"; w=imageio_ffmpeg.write_frames(p,(2,2),fps=1,macro_block_size=1); w.send(None); w.send(bytes([255,0,0]*4)); w.close(); r=imageio_ffmpeg.read_frames(p,pix_fmt="rgb24"); m=next(r); f=next(r); r.close(); assert m["size"] == (2,2) and len(f) == 12; os.unlink(p)' \
    && echo 'da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d  /usr/share/doc/npa-mjlab/notices/mcap-LICENSE.txt' | sha256sum -c - \
    && python3 -c 'import importlib.metadata as m; assert m.version("mcap") == "1.4.0"' \
    && curl --fail --location https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt \
       -o /usr/share/doc/npa-mjlab/notices/NVSHMEM-LICENSE.txt \
    && echo '43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f  /usr/share/doc/npa-mjlab/notices/NVSHMEM-LICENSE.txt' | sha256sum -c - \
    && chmod 0444 /usr/share/doc/npa-mjlab/notices/*
COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/npa-src \
    && pip check \
    && python3 -c 'import importlib.metadata as m; import uvicorn; assert m.version("mjlab") == "1.6.0"' \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache \
    && rm -rf /root/.cache
COPY docker/workbench/mjlab/REDISTRIBUTION.md /usr/share/doc/npa-mjlab/REDISTRIBUTION.md
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-workflow-entrypoint
WORKDIR /workspace
USER ubuntu
# Prove the declared bootstrap behavior before publishing its OCI attestation.
# This probe exercises the existing SkyPilot-only passwordless_root_exemption
# in packaging-contract.yaml; the service disables privilege escalation.
# trivy:ignore:AVD-DS-0010
RUN sudo -n true && test -w /tmp && test -w /home/ubuntu \
    && command -v rsync && command -v ssh-keygen && command -v service \
    && sudo ssh-keygen -A && sudo /usr/sbin/sshd -t \
    && sudo rm -f /etc/ssh/ssh_host_* \
    && test "$(npa-workflow-entrypoint /bin/sh -c 'printf bootstrap-argv-ok')" = bootstrap-argv-ok
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
CMD ["uvicorn", "npa.workbench.mjlab.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

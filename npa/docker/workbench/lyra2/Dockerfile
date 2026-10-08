# Lyra 2 reconstruction bootstrap. CUDA compiler and NPA are baked; Lyra source,
# its Python/CUDA inference environment, and model weights are runtime fetches.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20261008T000000Z

LABEL npa.tool="lyra2" \
      npa.version="2.0-rtfetch1" \
      npa.weights="operator-runtime-fetch" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_BAKED_PYTHON=/opt/npa/.venv/bin/python \
    NPA_LYRA_BASE_PYTHON=/usr/bin/python3.12 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    CUDA_HOME=/usr/local/cuda \
    PYTHONPATH=/opt/npa/src \
    PATH=/opt/npa/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

USER root
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' 'Types: deb' \
      "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      'Suites: noble noble-updates noble-backports noble-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 \
      openssh-server python3.12 python3.12-dev python3.12-venv rsync sudo \
    && rm -f /etc/ssh/ssh_host_* \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /opt/nvidia/nsight-compute \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-skypilot \
    && chmod 0440 /etc/sudoers.d/npa-skypilot

COPY docker/workbench/lyra2/requirements.lock /tmp/lyra2-requirements.lock
COPY docker/workbench/sonic/packaging-requirements.txt /tmp/lyra2-packaging.txt
RUN python3.12 -m venv /opt/npa/.venv \
    && /opt/npa/.venv/bin/python -m pip install --no-cache-dir --require-hashes \
      -r /tmp/lyra2-packaging.txt \
    && /opt/npa/.venv/bin/python -m pip install --no-cache-dir --require-hashes \
      -r /tmp/lyra2-requirements.lock \
    && rm -f /tmp/lyra2-requirements.lock /tmp/lyra2-packaging.txt

COPY pyproject.toml README.md /opt/npa/
COPY src/npa /opt/npa/src/npa
COPY workflows /opt/npa/workflows
COPY --chmod=0755 docker/workbench/lyra2/entrypoint.sh /usr/local/bin/lyra2-entrypoint
COPY --chmod=0644 docker/workbench/lyra2/REDISTRIBUTION.md /usr/share/doc/npa-lyra2/REDISTRIBUTION.md

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && /opt/npa/.venv/bin/python -m pip install --no-cache-dir --no-deps /opt/npa \
    && /opt/npa/.venv/bin/python -m pip check \
    && python3.12 -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-source.pth').write_text('/opt/npa/src\\n')" \
    && env -u PYTHONPATH python3.12 -c 'import npa' \
    && /opt/npa/.venv/bin/python -m npa.workflows.lyra_stage --help \
    && /opt/npa/.venv/bin/python -c "import importlib.util; assert importlib.util.find_spec('torch') is None" \
    && install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && chmod -R a+rX /opt/npa \
    && rm -rf /root/.cache /tmp/pip-*

WORKDIR /workspace
USER ubuntu
ENV HOME=/home/ubuntu
# Exercise the contract as the final user before attesting to its behavior.
RUN /usr/local/bin/lyra2-entrypoint /bin/sh -c \
    'set -eu; test -w /tmp; test -w "$HOME"; command -v rsync; command -v service; test -x /usr/sbin/sshd; sudo -n true' \
    && test "$(/usr/local/bin/lyra2-entrypoint /usr/bin/printf '%s' 'argv with spaces')" = 'argv with spaces'
ENTRYPOINT ["/usr/local/bin/lyra2-entrypoint"]

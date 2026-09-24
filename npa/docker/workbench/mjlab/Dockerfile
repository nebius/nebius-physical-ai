# MJLab and its CUDA wheel closure; no checkpoints, motion data or credentials.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ARG NPA_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
LABEL npa.tool="mjlab" \
      npa.version="1.6.0-cuda12-unbuilt" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"
ENV PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NPA_BAKED_PYTHON=/usr/local/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    XDG_CACHE_HOME=/workspace/.cache
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SOURCE_DATE_EPOCH" =~ ^[1-9][0-9]*$ ]] \
    && apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
       rsync sudo openssh-server libegl1 libgl1 libglvnd0 libglib2.0-0 ffmpeg git \
    && rm -f /etc/ssh/ssh_host_* \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime
COPY docker/workbench/mjlab/requirements.lock /opt/mjlab-requirements.lock
RUN pip install --no-cache-dir --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -r /opt/mjlab-requirements.lock
COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/npa-src \
    && pip check \
    && python3 -c 'import importlib.metadata as m; assert m.version("mjlab") == "1.6.0"' \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache \
    && rm -rf /root/.cache
COPY docker/workbench/mjlab/REDISTRIBUTION.md /usr/share/doc/npa-mjlab/REDISTRIBUTION.md
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
CMD ["uvicorn", "npa.workbench.mjlab.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

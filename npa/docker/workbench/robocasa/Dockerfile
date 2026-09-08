# Pinned RoboCasa simulation service; large kitchen archives remain runtime fetches.
FROM nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701

ARG NPA_SOURCE_SHA
ARG ROBOCASA_VERSION=0.1.0
LABEL npa.tool="robocasa" \
      npa.version="${ROBOCASA_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0 AND BSD-3-Clause AND CC-BY-4.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ROBOCASA_PORT=8791 \
    ROBOCASA_AUTH_MODE=token \
    NPA_BAKED_PYTHON=/opt/npa-venv/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NUMBA_CACHE_DIR=/workspace/.cache/numba \
    HF_HOME=/workspace/.cache/huggingface \
    PATH=/opt/npa-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl python3.12 python3.12-venv \
       gcc libc6-dev linux-libc-dev python3.12-dev \
       libegl1 libgl1 libglvnd0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && python3.12 -m venv /opt/npa-venv

COPY docker/workbench/robocasa/requirements.lock /opt/robocasa-requirements.lock
COPY docker/workbench/robocasa/policy-requirements.lock /opt/robocasa-policy-requirements.lock
COPY docker/workbench/curobo/filter_cudnn_runtime.py /opt/filter_cudnn_runtime.py
# The base has no cuDNN. Filter the exact wheel before its installation layer
# exists, retaining the unmodified runtime libraries and every license notice.
RUN pip install --no-cache-dir --no-compile --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/robocasa-requirements.lock \
    && mkdir -p /usr/share/doc/npa-robocasa \
    && python /opt/filter_cudnn_runtime.py > /usr/share/doc/npa-robocasa/cudnn-runtime.json \
    && pip install --no-cache-dir --no-compile --no-deps --require-hashes \
      -r /opt/robocasa-policy-requirements.lock

# The shared CUDA 13 wheel closure needs NVSHMEM's product supplement and
# third-party notices in addition to the wheel's embedded CUDA agreement.
RUN curl --fail --location https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt \
      -o /usr/share/doc/npa-robocasa/NVSHMEM-LICENSE.txt \
    && echo '43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f  /usr/share/doc/npa-robocasa/NVSHMEM-LICENSE.txt' | sha256sum -c -

# Preserve bundled assets, README credits and notices. Remove the unused MIT
# Omniverse adapter in the extraction layer; this service uses MuJoCo/EGL.
# Separate trainer extras are excluded; the selected ACT subset is documented.
RUN curl --fail --location https://codeload.github.com/ARISE-Initiative/robosuite/tar.gz/85abee228d1c43ab1939bce33028099945d453b4 \
      -o /tmp/robosuite.tar.gz \
    && echo '2369a5c3385bf122eeff3362e7f70aacf1d10436e860d12026ecd10b37c152cd  /tmp/robosuite.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/robosuite/source \
    && tar -xzf /tmp/robosuite.tar.gz --strip-components=1 -C /opt/robosuite/source \
    && rm /opt/robosuite/source/robosuite/scripts/render_dataset_with_omniverse.py \
    && pip install --no-cache-dir --no-compile --no-deps --no-build-isolation -e /opt/robosuite/source \
    && curl --fail --location https://codeload.github.com/robocasa/robocasa/tar.gz/8f3c96ec8d1bfcd8126cad2bca887da98d30e997 \
      -o /tmp/robocasa.tar.gz \
    && echo '1893328b5222ac0443287e593c161c696c77e3f8018f6d9f6bd900871d2caad3  /tmp/robocasa.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/robocasa/source \
    && tar -xzf /tmp/robocasa.tar.gz --strip-components=1 -C /opt/robocasa/source \
    && pip install --no-cache-dir --no-compile --no-deps --no-build-isolation -e /opt/robocasa/source \
    && rm /tmp/robosuite.tar.gz /tmp/robocasa.tar.gz

# Install the real NPA package, including the service's clients/storage and CLI
# path validation imports; a synthetic namespace containing one tool is incomplete.
COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
RUN pip install --no-cache-dir --no-compile --no-deps --no-build-isolation /opt/npa-src \
    && python -c "import gymnasium; assert gymnasium.__version__ == '0.29.1'; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.configs.policies import PreTrainedConfig; from lerobot.policies.factory import make_pre_post_processors; from npa.workbench.robocasa.service import create_app; assert create_app().title == 'NPA RoboCasa'" \
    && rm -rf /root/.cache /tmp/pip-*

COPY --chmod=0444 docker/workbench/robocasa/REDISTRIBUTION.md docker/workbench/robocasa/ASSET-NOTICES.md docker/workbench/robocasa/LICENSE-APACHE-2.0 docker/workbench/robocasa/LICENSE-SAWYER /usr/share/doc/npa-robocasa/
RUN (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache \
    && chown -R ubuntu:ubuntu /opt/robocasa/source/robocasa/models/assets

WORKDIR /workspace
USER ubuntu
EXPOSE 8791
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8791/health', timeout=3)"]
CMD ["uvicorn", "npa.workbench.robocasa.service:app", "--host", "0.0.0.0", "--port", "8791"]

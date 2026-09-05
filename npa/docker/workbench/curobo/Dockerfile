# Pinned Apache-2.0 cuRobo V2 plus MIT/BSD benchmark data; no model weights.
FROM nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701

# cuRobo V2 uses the pinned cuda.core/NVRTC and Warp wheel compilers.
# A runtime-only base avoids unused profiler binaries and development headers.
ARG NPA_SOURCE_SHA
# Build-only commit epoch makes pip emit reproducible PEP 552 checked-hash pyc.
ARG SOURCE_DATE_EPOCH
LABEL npa.tool="curobo" \
      npa.version="0.8.0-cuda13-b300-unbuilt" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/npa-venv/bin:${PATH} \
    NPA_BAKED_PYTHON=/opt/npa-venv/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_CUROBO_SOURCE=/opt/curobo \
    NPA_CUROBO_WORK_DIR=/workspace \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache \
    CUDA_HOME=/usr/local/cuda

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SOURCE_DATE_EPOCH" =~ ^[1-9][0-9]*$ ]]
RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates curl python3.12 python3.12-venv \
       openssh-server rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && python3.12 -m venv /opt/npa-venv

COPY docker/workbench/curobo/requirements.lock /opt/curobo-requirements.lock
COPY docker/workbench/curobo/filter_cudnn_runtime.py /opt/filter_cudnn_runtime.py
COPY docker/workbench/curobo/remove_scikit_image_recipe.py /opt/remove_scikit_image_recipe.py
# The current cuDNN supplement permits runtime-only redistribution. Use that
# narrower boundary even though this wheel embeds older, broader header terms.
# The non-cuDNN base contains none; filter the wheel in this SAME RUN before a
# layer exists. Keep runtime .so bytes and all distribution license notices.
RUN pip install --no-cache-dir --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /opt/curobo-requirements.lock \
    && mkdir -p /usr/share/doc/npa-curobo \
    && python /opt/filter_cudnn_runtime.py > /usr/share/doc/npa-curobo/cudnn-runtime.json \
    && python /opt/remove_scikit_image_recipe.py \
      --site-packages /opt/npa-venv/lib/python3.12/site-packages \
      > /usr/share/doc/npa-curobo/dependency-source-correction.json \
    && pip check

# The NVSHMEM wheel omits its release's product supplement and third-party
# notices. Add the complete official notice without replacing its wheel license.
# This URL can change; the exact reviewed release notice hash fails closed.
RUN curl --fail --location https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt -o /usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt \
    && echo '43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f  /usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt' | sha256sum -c - \
    && chmod 0444 /usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt

# Official archive hashes are checked before extraction. Keep upstream source and
# robot/dataset licenses in the image; no ambient repository or credential cache.
COPY docker/workbench/curobo/correct_package_metadata.py /opt/correct_package_metadata.py
RUN curl --fail --location https://codeload.github.com/NVlabs/curobo/tar.gz/8e734f3ced1df898990bcd92de40abce475907db -o /tmp/curobo.tar.gz \
    && echo 'ea6d8310b1ab109ceaac046b15b4de31ecb7c3b52c2a4c762c58316c81cbbf2f  /tmp/curobo.tar.gz' | sha256sum -c - \
    && mkdir /opt/curobo && tar -xzf /tmp/curobo.tar.gz --strip-components=1 -C /opt/curobo \
    && printf '%s\n' 8e734f3ced1df898990bcd92de40abce475907db > /opt/curobo/NPA_SOURCE_REVISION \
    && python /opt/correct_package_metadata.py /opt/curobo > /usr/share/doc/npa-curobo/metadata-correction.json \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 pip install --no-deps --no-build-isolation --no-cache-dir /opt/curobo \
    && curl --fail --location https://codeload.github.com/fishbotics/robometrics/tar.gz/81e3d1d605de84100d8ab880b43096aba221a48b -o /tmp/robometrics.tar.gz \
    && echo 'ce62a3e4a721831054a533fa3646e4a6043f3eb0b38065a623edfec51ff9e172  /tmp/robometrics.tar.gz' | sha256sum -c - \
    && mkdir /opt/robometrics && tar -xzf /tmp/robometrics.tar.gz --strip-components=1 -C /opt/robometrics \
    && printf '%s\n' 81e3d1d605de84100d8ab880b43096aba221a48b > /opt/robometrics/NPA_SOURCE_REVISION \
    && rm /tmp/curobo.tar.gz /tmp/robometrics.tar.gz

COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
COPY workflows/workbench/npa-workflows/sim2real.yaml /opt/npa-src/workflows/workbench/npa-workflows/sim2real.yaml
COPY workflows/workbench/npa-workflows/physical-ai-data-factory.yaml /opt/npa-src/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
RUN pip install --no-deps --no-build-isolation --no-cache-dir /opt/npa-src \
    && python -c 'import torch; assert {"sm_100", "sm_120"} <= set(torch._C._cuda_getArchFlags().split())' \
    && rm -rf /root/.cache /tmp/pip-*

COPY --chmod=0444 docker/workbench/curobo/REDISTRIBUTION.md /usr/share/doc/npa-curobo/REDISTRIBUTION.md
COPY --chmod=0755 docker/workbench/curobo/entrypoint.sh /usr/local/bin/curobo-entrypoint
RUN install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["curobo-entrypoint"]
CMD ["serve"]

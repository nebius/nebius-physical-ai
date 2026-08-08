# Public-eligible Wan 2.2 image: Apache-2.0 source plus OSS CPU dependencies.
# CUDA PyTorch and every nvidia-* distribution are runtime-fetched into a volume
# only after the operator accepts NVIDIA's terms. No model/tokenizer is baked.
FROM python:3.10-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc

ARG WAN_SOURCE_URL=https://github.com/Wan-Video/Wan2.2.git
ARG WAN_SOURCE_REF=42bf4cfaa384bc21833865abc2f9e6c0e67233dc
ARG NPA_SOURCE_COMMIT=unknown
ARG NPA_RUNTIME_USER=ubuntu

LABEL npa.tool="wan2-2" \
      npa.version="2.2-ti2v5b-runtime-fetch" \
      npa.source.repo="${WAN_SOURCE_URL}" \
      npa.source.ref="${WAN_SOURCE_REF}" \
      npa.cuda.provisioning="operator-accepted-runtime-fetch" \
      npa.weights="runtime-only" \
      npa.security.runtime-user="uid-1000-with-skypilot-bootstrap-sudo" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    NPA_WAN_BASE_PYTHON=/opt/wan-base/bin/python \
    NPA_WAN_RUNTIME_REQUIREMENTS=/opt/npa/wan2-2/runtime-requirements.txt \
    NPA_WAN_RUNTIME_CACHE=/workspace/.cache/npa/wan2-2/runtime \
    WAN22_CACHE_DIR=/workspace/model-cache/wan2.2 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-lc"]
USER root

RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git git-lfs libgl1 libglib2.0-0 \
      openssh-server procps rsync sudo util-linux \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u "${NPA_RUNTIME_USER}" >/dev/null 2>&1 \
       || useradd -m -s /bin/bash -u 1000 "${NPA_RUNTIME_USER}") \
    && printf '%s\n' "${NPA_RUNTIME_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

RUN git clone --filter=blob:none --no-checkout "${WAN_SOURCE_URL}" /opt/byof \
    && git -C /opt/byof checkout --detach "${WAN_SOURCE_REF}" \
    && test "$(git -C /opt/byof rev-parse HEAD)" = "${WAN_SOURCE_REF}" \
    && git -C /opt/byof lfs uninstall \
    && rm -rf /opt/byof/.git \
    && test -s /opt/byof/LICENSE.txt \
    && printf '%s\n' \
      '{' \
      '  "source": "oss-byof",' \
      '  "repo": "https://github.com/Wan-Video/Wan2.2.git",' \
      '  "ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"' \
      '}' > /opt/byof/npa_source_metadata.json

COPY --chmod=0444 docker/workbench/wan2-2/baked-constraints.txt /opt/npa/wan2-2/baked-constraints.txt
COPY --chmod=0444 docker/workbench/wan2-2/easydict_compat.py /opt/byof/wan/utils/easydict.py
COPY --chmod=0444 docker/workbench/wan2-2/wan_init_ti2v.py /opt/byof/wan/__init__.py
COPY --chmod=0444 docker/workbench/wan2-2/input_contract.py /opt/byof/npa_wan_input_contract.py

# CPU-only PyTorch is BSD-3-Clause and prevents OSS dependency resolution from
# selecting CUDA wheels. The CUDA build later shadows it in the runtime cache.
RUN python -m venv /opt/wan-base \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir --upgrade \
      "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1" \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --constraint /opt/npa/wan2-2/baked-constraints.txt \
      "torch==2.7.1+cpu" "torchvision==0.22.1+cpu" "torchaudio==2.7.1+cpu" \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir \
      --constraint /opt/npa/wan2-2/baked-constraints.txt \
      "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
      "diffusers==0.35.1" "transformers==4.51.3" "tokenizers==0.21.4" \
      "accelerate==1.9.0" "huggingface-hub==0.34.4" \
      "imageio==2.37.0" \
      "safetensors==0.5.3" "sentencepiece==0.2.0" "einops==0.8.1" \
      "ftfy==6.3.1" "tqdm==4.67.1" \
      "decord==0.6.0" "peft==0.17.0" \
      "dashscope==1.26.5" "boto3==1.39.11" \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir \
      --no-binary imageio-ffmpeg \
      --constraint /opt/npa/wan2-2/baked-constraints.txt \
      "imageio-ffmpeg==0.6.0" \
    && /opt/wan-base/bin/python -m pip install --no-cache-dir --no-deps -e /opt/byof \
    && sed -i \
      's/^from \.attention import flash_attention$/from .attention import attention as flash_attention/' \
      /opt/byof/wan/modules/model.py \
    && sed -i \
      's/^from easydict import EasyDict$/from wan.utils.easydict import EasyDict/' \
      /opt/byof/wan/configs/*.py \
    && grep -Fxq 'from .attention import attention as flash_attention' /opt/byof/wan/modules/model.py \
    && ! grep -Rqs '^from easydict import EasyDict$' /opt/byof/wan/configs \
    && test "$(/opt/wan-base/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" = /usr/bin/ffmpeg \
    && test -z "$(find /opt/wan-base -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && /opt/wan-base/bin/python -m compileall -q /opt/byof \
    && /opt/wan-base/bin/python -c 'import runpy; cls=runpy.run_path("/opt/byof/wan/utils/easydict.py")["EasyDict"]; cfg=cls({"nested":{"value":7}}); assert cfg.nested.value == 7' \
    && /opt/wan-base/bin/python -m pip freeze --all --exclude-editable > /opt/byof/npa_baked_python_inventory.txt \
    && diff -u \
      <(sed '/^#/d;/^$/d' /opt/npa/wan2-2/baked-constraints.txt) \
      /opt/byof/npa_baked_python_inventory.txt \
    && ! grep -Eiq '^nvidia-|cu(da|dnn|blas)|nccl' /opt/byof/npa_baked_python_inventory.txt \
    && test -z "$(find /opt/byof /opt/wan-base -type f \
      \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname '*.bin.index.json' \
         -o -iname 'model*.pt' -o -iname 'model*.pth' -o -iname 'weights*.pt' \
         -o -iname 'weights*.pth' \) \
      -print -quit)" \
    && rm -rf /root/.cache /tmp/pip-* \
    && ln -s /workspace/.cache/npa/wan2-2/runtime/current/venv /opt/byof/.venv

COPY --chmod=0444 docker/workbench/wan2-2/runtime-requirements.txt /opt/npa/wan2-2/runtime-requirements.txt
COPY --chmod=0755 docker/workbench/wan2-2/wan_runtime.sh /usr/local/bin/wan-runtime
COPY --chmod=0755 docker/workbench/wan2-2/entrypoint.sh /usr/local/bin/wan-entrypoint
COPY --chmod=0755 docker/workbench/wan2-2/smoke.sh /opt/npa/wan2-2/smoke.sh
COPY --chmod=0444 docker/workbench/wan2-2/REDISTRIBUTION.md /usr/share/doc/npa-wan2-2/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/wan2-2/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-wan2-2/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace /workspace/.cache/npa/wan2-2/runtime /workspace/model-cache/wan2.2 /opt/npa/wan2-2 \
    && chmod 0755 /opt/npa \
    && test -r /opt/byof/LICENSE.txt \
    && test -r /opt/npa/wan2-2/runtime-requirements.txt \
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'wan-runtime health' \
    && test -z "$(find /workspace/.cache/npa/wan2-2/runtime -mindepth 1 -print -quit)"

VOLUME ["/workspace/.cache/npa/wan2-2/runtime", "/workspace/model-cache/wan2.2"]
WORKDIR /opt/byof
USER ${NPA_RUNTIME_USER}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["wan-runtime", "health"]
ENTRYPOINT ["wan-entrypoint"]

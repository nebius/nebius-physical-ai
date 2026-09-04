# Apache-2.0 OpenPI source and pinned RLDS/JAX runtime. Gemma/pi0.5 weights,
# DROID data, credentials, terms acceptance, and populated caches are absent.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed

ARG OPENPI_SOURCE_URL=https://github.com/Physical-Intelligence/openpi.git
ARG OPENPI_SOURCE_REF=15a9616a00943ada6c20a0f158e3adb39df2ccac
ARG NPA_SOURCE_SHA=unknown

LABEL npa.tool="openpi" \
      npa.version="pi05-full-droid-rlds-cu128-jax062-nccl2275-rerun0314" \
      npa.source.repo="${OPENPI_SOURCE_URL}" \
      npa.source.ref="${OPENPI_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.dataset="operator-runtime-fetch" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    GIT_LFS_SKIP_SMUDGE=1 \
    HOME=/home/ubuntu \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    OPENPI_DATA_HOME=/workspace/.cache/openpi \
    NPA_OPENPI_RERUN_PYTHON=/opt/rerun-venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    WANDB_MODE=disabled \
    PATH=/opt/venv/bin:/opt/gsutil-venv/bin:${PATH}

SHELL ["/bin/bash", "-lc"]
USER root

COPY docker/workbench/openpi/gsutil-requirements.txt /tmp/gsutil-requirements.txt

RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git git-lfs linux-libc-dev openssh-server python3 python3-dev \
      python3-pip python3-venv rsync sudo curl fuse gcc netcat-openbsd openssh-client \
      patch pciutils procps wget \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

RUN git clone --filter=blob:none --no-checkout "${OPENPI_SOURCE_URL}" /opt/byof \
    && git -C /opt/byof checkout --detach "${OPENPI_SOURCE_REF}" \
    && test "$(git -C /opt/byof rev-parse HEAD)" = "${OPENPI_SOURCE_REF}" \
    && test -s /opt/byof/LICENSE \
    && git -C /opt/byof lfs uninstall \
    && rm -rf /opt/byof/.git \
    && printf '%s\n' \
      '{' \
      '  "source": "oss-byof",' \
      '  "repo": "https://github.com/Physical-Intelligence/openpi.git",' \
      '  "ref": "15a9616a00943ada6c20a0f158e3adb39df2ccac"' \
      '}' > /opt/byof/npa_source_metadata.json

# Upstream's frozen lock is authoritative except for the exact JAX/NCCL SM120
# compatibility overlay below. The upstream JAX 0.5.3 / NCCL 2.26.2 pair faults
# on the first cross-node collective; the replacement wheels were qualified on
# eight one-GPU RTX PRO 6000 nodes. Keep gsutil isolated from this environment.
RUN python3 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir 'uv==0.9.7' \
    && /opt/uv/bin/uv python install 3.11 \
    && /opt/uv/bin/uv venv --python 3.11 /opt/venv \
    && cd /opt/byof \
    && VIRTUAL_ENV=/opt/venv /opt/uv/bin/uv sync --active --frozen --group rlds --no-dev \
    && cd / \
    && /opt/uv/bin/uv pip install --python /opt/venv/bin/python --no-cache --no-deps \
      'jax==0.6.2' \
      'jaxlib==0.6.2' \
      'jax-cuda12-plugin==0.6.2' \
      'jax-cuda12-pjrt==0.6.2' \
      'ml-dtypes==0.5.1' \
      'nvidia-cudnn-cu12==9.10.2.21' \
      'nvidia-nccl-cu12==2.27.5' \
      'nvidia-nvshmem-cu12==3.2.5' \
    && /opt/venv/bin/python -c \
      'import importlib.metadata as m; expected={"jax":"0.6.2","jaxlib":"0.6.2","jax-cuda12-plugin":"0.6.2","jax-cuda12-pjrt":"0.6.2","ml-dtypes":"0.5.1","nvidia-cublas-cu12":"12.6.4.1","nvidia-cuda-cupti-cu12":"12.6.80","nvidia-cuda-nvcc-cu12":"12.9.41","nvidia-cuda-nvrtc-cu12":"12.6.77","nvidia-cuda-runtime-cu12":"12.6.77","nvidia-cudnn-cu12":"9.10.2.21","nvidia-cufft-cu12":"11.3.0.4","nvidia-cusolver-cu12":"11.7.1.2","nvidia-cusparse-cu12":"12.5.4.2","nvidia-nccl-cu12":"2.27.5","nvidia-nvjitlink-cu12":"12.6.85","nvidia-nvshmem-cu12":"3.2.5"}; assert all(m.version(k)==v for k,v in expected.items()), {k:m.version(k) for k in expected}' \
    && /opt/venv/bin/python -c \
      'from pathlib import Path; root=Path("/opt/venv/lib/python3.11/site-packages/nvidia"); required=("cuda_runtime/lib/libcudart.so.12","cudnn/lib/libcudnn.so.9","cufft/lib/libcufft.so.11","cusolver/lib/libcusolver.so.11","cusparse/lib/libcusparse.so.12","nccl/lib/libnccl.so.2","nvshmem/lib/libnvshmem_host.so.3"); assert all((root/path).is_file() for path in required), [path for path in required if not (root/path).is_file()]' \
    && JAX_PLATFORMS=cpu /opt/venv/bin/python -c \
      'import numpy as np; import chex, flax, jax, jax.numpy as jnp, ml_dtypes, orbax.checkpoint, tensorflow, tensorflow_datasets; import openpi.training.config as config; from jax.experimental import multihost_utils; from jax.extend import backend; from jax.sharding import Mesh, NamedSharding, PartitionSpec; cfg=config.get_config("pi05_droid"); value=multihost_utils.broadcast_one_to_all(jax.jit(lambda x: x+1)(jnp.asarray([1]))); assert cfg and np.asarray(value).tolist()==[2] and jax.tree.map and jax.random.key and backend and Mesh and NamedSharding and PartitionSpec' \
    && find /opt/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test "$(/opt/venv/bin/python -c \
      'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" = /usr/bin/ffmpeg \
    && test -z "$(find /opt/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && /opt/venv/bin/python -c \
      'import imageio_ffmpeg, os; p="/tmp/npa-openpi-ffmpeg-probe.mp4"; w=imageio_ffmpeg.write_frames(p,(2,2),fps=1,macro_block_size=1); w.send(None); w.send(bytes([0,0,0]*4)); w.close(); r=imageio_ffmpeg.read_frames(p,pix_fmt="rgb24"); m=next(r); f=next(r); r.close(); assert m["size"] == (2,2) and len(f) == 12; os.unlink(p)' \
    && /opt/uv/bin/uv pip install --python /opt/venv/bin/python --no-cache --no-deps \
      'boto3==1.42.91' \
      'botocore==1.42.91' \
      'jmespath==1.1.0' \
      'python-dateutil==2.9.0.post0' \
      's3transfer==0.16.0' \
      'six==1.17.0' \
      'urllib3==2.7.0' \
    && /opt/uv/bin/uv pip install --python /opt/venv/bin/python --no-cache \
      'deepdiff==8.6.1' 'numpy==1.26.4' \
    && rm -f /opt/venv/lib/python3.11/site-packages/wandb/bin/wandb-core \
    && /opt/venv/bin/python -c \
      'import boto3, importlib.metadata as m, numpy, os, tensorflow, torch, transformers, wandb; from transformers import GemmaForCausalLM; expected={"boto3":"1.42.91","botocore":"1.42.91","jmespath":"1.1.0","python-dateutil":"2.9.0.post0","s3transfer":"0.16.0","six":"1.17.0","urllib3":"2.7.0"}; assert all(m.version(k)==v for k,v in expected.items()), {k:m.version(k) for k in expected}; assert boto3.session.Session() is not None and m.version("deepdiff") == "8.6.1" and m.version("numpy") == "1.26.4" and m.version("tensorflow") == "2.15.0" and m.version("transformers") == "4.53.2"; assert os.environ["WANDB_MODE"] == "disabled" and GemmaForCausalLM and numpy is not None and tensorflow is not None and torch is not None and wandb is not None' \
    && /opt/venv/bin/python -c \
      'import subprocess; expected={"The package `tensorflow-cpu` requires `ml-dtypes~=0.2.0`, but `0.5.1` is installed","The package `torch` requires `nvidia-cudnn-cu12==9.5.1.17 ; platform_machine == x86_64 and sys_platform == linux`, but `9.10.2.21` is installed","The package `torch` requires `nvidia-nccl-cu12==2.26.2 ; platform_machine == x86_64 and sys_platform == linux`, but `2.27.5` is installed","The package `tensorflow` requires `ml-dtypes~=0.2.0`, but `0.5.1` is installed","The package `openpi` requires `jax[cuda12]==0.5.3`, but `0.6.2` is installed"}; result=subprocess.run(["/opt/uv/bin/uv","pip","check","--python","/opt/venv/bin/python"],capture_output=True,text=True); actual={line.replace(chr(39),"") for line in (result.stdout+result.stderr).splitlines() if line.startswith("The package `")}; assert result.returncode == 1 and actual == expected, {"returncode":result.returncode,"unexpected":sorted(actual-expected),"missing":sorted(expected-actual)}' \
    && /opt/uv/bin/uv venv --python 3.11 /opt/rerun-venv \
    && /opt/uv/bin/uv pip install --python /opt/rerun-venv/bin/python --no-cache \
      'pyyaml==6.0.3' 'rerun-sdk==0.31.4' \
    && /opt/rerun-venv/bin/python -c \
      'import importlib.metadata as m, numpy, rerun, yaml; assert m.version("pyyaml") == "6.0.3" and m.version("rerun-sdk") == "0.31.4" and int(m.version("numpy").split(".")[0]) >= 2 and rerun is not None and numpy is not None and yaml is not None' \
    && /opt/uv/bin/uv pip check --python /opt/rerun-venv/bin/python \
    && /opt/rerun-venv/bin/rerun rrd --help >/dev/null \
    && test ! -e /opt/venv/lib/python3.11/site-packages/wandb/bin/wandb-core \
    && /opt/uv/bin/uv venv --python 3.11 /opt/gsutil-venv \
    && /opt/uv/bin/uv pip install --python /opt/gsutil-venv/bin/python --no-cache \
      --no-deps --require-hashes -r /tmp/gsutil-requirements.txt \
    && /opt/uv/bin/uv pip check --python /opt/gsutil-venv/bin/python \
    && /opt/gsutil-venv/bin/gsutil version -l | grep -Fx 'gsutil version: 5.37' \
    && build_hash="$(sha256sum /opt/byof/uv.lock | cut -d' ' -f1)" \
    && printf '{"schema":"npa.byof.build.v1","build_command_executed":true,"build_command_sha256":"%s"}\n' \
      "$build_hash" > /opt/byof/npa_build_metadata.json \
    && rm -rf /tmp/uv-cache /tmp/gsutil-requirements.txt

RUN printf '%s\n' \
      '#include <cuda_runtime.h>' \
      '#include <cstdio>' \
      'int main(){int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess)return 2;if(n!=1)return 3;cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,0)!=cudaSuccess||p.major!=12||p.minor!=0)return 4;std::printf("devices=%d cc=12.0\\n",n);return 0;}' \
      > /tmp/npa-openpi-sm120.cu \
    && nvcc -O2 -arch=sm_120 /tmp/npa-openpi-sm120.cu -o /usr/local/bin/npa-openpi-sm120-probe \
    && rm -f /tmp/npa-openpi-sm120.cu \
    && rm -rf /opt/nvidia/nsight-compute \
    && test ! -e /opt/nvidia/nsight-compute

COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
COPY workflows/workbench/npa-workflows/sim2real.yaml \
  /opt/npa-src/workflows/workbench/npa-workflows/sim2real.yaml
COPY workflows/workbench/npa-workflows/physical-ai-data-factory.yaml \
  /opt/npa-src/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
COPY workflows/workbench/npa-workflows/openpi-pi05-full-droid-finetune.yaml \
  /opt/npa-src/workflows/workbench/npa-workflows/openpi-pi05-full-droid-finetune.yaml
RUN /opt/uv/bin/uv pip install --python /opt/venv/bin/python --no-cache --no-deps /opt/npa-src \
    && /opt/uv/bin/uv pip install --python /opt/rerun-venv/bin/python --no-cache --no-deps /opt/npa-src \
    && /opt/venv/bin/python -m npa.workflows.byof.openpi_full_droid --help >/dev/null \
    && /opt/rerun-venv/bin/python -m npa.workflows.byof.openpi_full_droid rrd-worker --help >/dev/null \
    && test -z "$(find /opt/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && test -z "$(find /opt/byof /opt/npa-src /opt/venv \
      \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \
         -o -iname '*.tfrecord' -o -iname '*.tfrecord-*' \) -print -quit)" \
    && rm -rf /root/.cache /tmp/uv-cache /tmp/pip-*

COPY --chmod=0755 docker/workbench/openpi/entrypoint.sh /usr/local/bin/openpi-entrypoint
COPY --chmod=0444 docker/workbench/openpi/REDISTRIBUTION.md /usr/share/doc/npa-openpi/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/openpi/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-openpi/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o ubuntu -g ubuntu \
      /workspace /workspace/.cache /workspace/.cache/openpi \
    && chown -R ubuntu:ubuntu /home/ubuntu /workspace /opt/byof \
    && test -r /opt/byof/LICENSE \
    && test -r /usr/share/doc/npa-openpi/REDISTRIBUTION.md

VOLUME ["/workspace/.cache/openpi"]
WORKDIR /opt/byof
USER ubuntu
ENTRYPOINT ["openpi-entrypoint"]
CMD ["bash"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ENV DEBIAN_FRONTEND=noninteractive UV_NO_PROGRESS=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface HF_LEROBOT_HOME=/workspace/cache/lerobot
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg git python3 python3-pip python3-venv rsync sudo netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/tools && /opt/tools/bin/pip install --no-cache-dir uv==0.11.23
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot \
    && cd /opt/lerobot \
    && git checkout --detach 1bc0bdfb20ad4f4f76dc8a68a0e0746d54f50de9 \
    && test "$(git rev-parse HEAD)" = 1bc0bdfb20ad4f4f76dc8a68a0e0746d54f50de9
# Keep the CUDA stack in separate layers: one combined layer exceeded Docker Desktop's registry upload connection.
RUN cd /opt/lerobot && /opt/tools/bin/uv sync --locked \
    --no-install-package torch --no-install-package torchvision \
    --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cublas-cu12
RUN cd /opt/lerobot && /opt/tools/bin/uv sync --locked
RUN cd /opt/lerobot && /opt/tools/bin/uv sync --locked \
    --extra training --extra flux3 --extra peft --extra diffusion
RUN /opt/tools/bin/uv pip install --python /opt/lerobot/.venv/bin/python \
    'natten==0.21.6+torch2110cu128' --find-links https://whl.natten.org/ \
    && /opt/lerobot/.venv/bin/python -c 'import torch, natten, lerobot; assert torch.__version__.startswith("2.11.")'
RUN (id -u ubuntu >/dev/null 2>&1 || useradd -m -u 1000 -s /bin/bash ubuntu) \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' ubuntu > /etc/sudoers.d/99-npa-runtime \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime \
    && install -d -o ubuntu -g ubuntu /workspace /opt/npa/src/npa/workbench/lerobot
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
# NPA's av 17 pin removes av.option, which this pinned LeRobot dataset reader needs.
RUN /opt/tools/bin/uv pip install --python /opt/lerobot/.venv/bin/python -e /opt/npa \
    && /opt/tools/bin/uv pip install --python /opt/lerobot/.venv/bin/python --no-deps 'av==15.1.0' \
    && /opt/lerobot/.venv/bin/python -c 'import av, lerobot.datasets, npa.cli.main, torch, natten; assert av.__version__ == "15.1.0" and torch.__version__.startswith("2.11.")'
ENV PYTHONPATH=/opt/npa/src NPA_BAKED_PYTHON=/opt/lerobot/.venv/bin/python \
    PATH=/usr/bin:/opt/tools/bin:/opt/lerobot/.venv/bin:$PATH
USER ubuntu
WORKDIR /workspace
CMD ["/opt/lerobot/.venv/bin/python", "-c", "import torch, natten, lerobot; print('FLUX 3 SO-101 runtime ready')"]

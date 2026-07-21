# Thin detector train/eval layer on the shared CUDA/torch base. NPA_CUDA_BASE
# defaults to the locally-built base tag; override with
# --build-arg NPA_CUDA_BASE=<registry>/npa-workbench-cuda-base:<tag> to build
# against a published base.
ARG NPA_CUDA_BASE=npa-workbench-cuda-base:cuda13.0-torch2.12.1
FROM ${NPA_CUDA_BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DETECTION_TRAINING_PORT=8790

WORKDIR /app

# libgl/libglib + the FastAPI + lancedb + pyarrow + pillow + boto3 + torch/CUDA
# stack come from the shared base; only the detector-specific torch extensions
# and COCO metrics are unique to this image.
COPY docker/workbench/detection-training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/npa/workbench
RUN touch /app/npa/__init__.py /app/npa/workbench/__init__.py
COPY src/npa/workbench/detection_training /app/npa/workbench/detection_training
# detection_training.schemas imports training_config at module load.
COPY src/npa/workbench/training_config.py /app/npa/workbench/training_config.py
# Golden-eval smoke (standalone; runs against the in-image service module).
COPY src/npa/smoke/test_detection_training_functional.py /app/smoke_functional.py

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app

EXPOSE 8790

USER ubuntu

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.environ.get('DETECTION_TRAINING_PORT','8790'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.detection_training.service:app", "--host", "0.0.0.0", "--port", "8790"]

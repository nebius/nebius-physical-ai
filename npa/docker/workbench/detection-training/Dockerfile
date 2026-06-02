FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DETECTION_TRAINING_PORT=8790

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/workbench/detection-training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/npa/workbench
RUN touch /app/npa/__init__.py /app/npa/workbench/__init__.py
COPY src/npa/workbench/detection_training /app/npa/workbench/detection_training

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app

EXPOSE 8790

USER ubuntu

CMD ["uvicorn", "npa.workbench.detection_training.service:app", "--host", "0.0.0.0", "--port", "8790"]

FROM python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/home/mlflow/.local/bin:$PATH \
    MLFLOW_HOME=/opt/mlflow

RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/mlflow --create-home --shell /usr/sbin/nologin mlflow \
    && mkdir -p /opt/mlflow /tmp/mlflow \
    && chown -R 10001:10001 /opt/mlflow /tmp/mlflow /home/mlflow

WORKDIR /opt/mlflow
COPY --chown=10001:10001 requirements.txt /opt/mlflow/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --upgrade -r /opt/mlflow/requirements.txt

COPY --chown=10001:10001 src/ /opt/mlflow/src/
COPY --chown=10001:10001 scripts/verify_workflow.py /opt/mlflow/src/verify_workflow.py
USER 10001:10001
EXPOSE 5000
ENTRYPOINT ["python", "/opt/mlflow/src/mlflow_server.py"]

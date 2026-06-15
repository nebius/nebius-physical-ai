# CPU preprocess image for SONIC retargeting/motion-lib conversion.
FROM python:3.11-slim

ARG RETARGETING_VERSION=0.1.1
ARG SONIC_REPO_URL=https://github.com/NVlabs/GR00T-WholeBodyControl.git
ARG SONIC_REPO_REF=a9d20b2ac0949244d94461a1a3263f38c5027c4a

LABEL npa.tool="retargeting" \
      npa.version="${RETARGETING_VERSION}" \
      npa.upstream="NVlabs/GR00T-WholeBodyControl" \
      npa.upstream_ref="${SONIC_REPO_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SONIC_HOME=/opt/sonic \
    NPA_RETARGETING_UPSTREAM_REF=${SONIC_REPO_REF} \
    NPA_RETARGETING_AUTO_FETCH=0

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/npa

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --sparse "${SONIC_REPO_URL}" "${SONIC_HOME}" \
    && cd "${SONIC_HOME}" \
    && git checkout "${SONIC_REPO_REF}" \
    && git sparse-checkout set gear_sonic/data_process \
    && rm -rf "${SONIC_HOME}/.git"

COPY pyproject.toml /opt/npa/pyproject.toml
COPY src/npa /opt/npa/src/npa

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --no-deps -e /opt/npa \
    && python -m pip install --no-cache-dir \
      "boto3~=1.42.91" \
      "httpx~=0.28.1" \
      "jinja2~=3.1.6" \
      "joblib>=1.3" \
      "matplotlib>=3.8" \
      "numpy>=1.24" \
      "pandas>=2.0" \
      "paramiko~=4.0.0" \
      "pillow>=10.0" \
      "pydantic>=2.0" \
      "pyarrow>=14.0" \
      "pyyaml~=6.0.3" \
      "rich~=15.0.0" \
      "scipy>=1.10" \
      "typer~=0.24.1"

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /opt/npa /opt/sonic

USER ubuntu
WORKDIR /opt/npa
ENTRYPOINT ["/bin/bash"]

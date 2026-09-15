# Digest pinned to the maintained Python security release.
# To update: re-resolve via registry API and update digest below.
FROM python:3.11.16-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ARG FIFTYONE_VERSION=1.21.0
ARG BOTO3_VERSION=1.43.6
ARG DATASETS_VERSION=5.0.1
ARG HUGGINGFACE_HUB_VERSION=1.14.0
ARG PYARROW_VERSION=24.0.0
ARG PILLOW_VERSION=12.3.0
# MongoDB server for FiftyOne's metadata DB. The prebuilt fiftyone_db wheel does
# NOT ship a mongod binary for this platform (trixie), so real curation
# (compute_uniqueness / compute_similarity / compute_visualization, which all need
# the DB) fails with "Could not find `mongod`". We bundle the generic Ubuntu 22.04
# mongod build into fiftyone_db's bin dir so FiftyOne launches its own DB with no
# external MongoDB required.
ARG MONGODB_VERSION=7.0.40
ARG MONGODB_SHA256=e4b3d7a11818f983d897ec9fcbf25779a6e122f0e7b7e25fa4ab8ac5d78a5a89

LABEL npa.tool="fiftyone" \
      npa.version="${FIFTYONE_VERSION}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    FIFTYONE_VERSION=${FIFTYONE_VERSION} \
    FIFTYONE_HOME=/opt/fiftyone \
    FIFTYONE_DATABASE_DIR=/opt/fiftyone/db \
    FIFTYONE_DEFAULT_APP_ADDRESS=127.0.0.1 \
    FIFTYONE_DEFAULT_APP_PORT=5151 \
    FIFTYONE_DO_NOT_TRACK=true \
    FIFTYONE_VENV=/opt/fiftyone/venv \
    PATH=/opt/fiftyone/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --chmod=0644 docker/workbench/sonic/packaging-requirements.txt /opt/npa-packaging-requirements.txt
COPY --chmod=0644 docker/workbench/curobo/remove_scikit_image_recipe.py /opt/npa-remove-scikit-image-recipe.py

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      netcat-openbsd \
      openssh-server \
      patch \
      procps \
      rsync \
      sudo \
    && (apt-get install -y --no-install-recommends libcurl4t64 \
        || apt-get install -y --no-install-recommends libcurl4) \
    && install -d -m 0755 /run/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && printf '%s\n' \
      'PasswordAuthentication no' \
      'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/99-npa-container.conf \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
      -r /opt/npa-packaging-requirements.txt

# SkyPilot invokes apt and service through sudo. Restricting executable names
# would be cosmetic because apt can install arbitrary root code; this grant is
# confined to the image's dedicated workflow user.
RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/fiftyone \
      /opt/fiftyone/db \
      /opt/fiftyone/datasets \
      /opt/fiftyone/smoke \
      /opt/fiftyone/venv \
      /opt/npa/docker/workbench/fiftyone \
    && printf '%s\n' \
      'export FIFTYONE_VENV=/opt/fiftyone/venv' \
      'export PATH=/opt/fiftyone/venv/bin:$PATH' \
      > /etc/profile.d/npa-fiftyone.sh

USER ubuntu
WORKDIR /opt/fiftyone

# Replace any wheel-provided database binary before this installation layer is
# committed, so an older executable cannot survive in an ancestor layer.
RUN python -m venv "${FIFTYONE_VENV}" \
    && python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
      -r /opt/npa-packaging-requirements.txt \
    && python -m pip install --no-cache-dir \
      "fiftyone==${FIFTYONE_VERSION}" \
      "boto3==${BOTO3_VERSION}" \
      "datasets==${DATASETS_VERSION}" \
      "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
      "pyarrow==${PYARROW_VERSION}" \
      "pillow==${PILLOW_VERSION}" \
      "paramiko==5.0.0" \
      "scikit-image==0.26.0" \
    && python /opt/npa-remove-scikit-image-recipe.py \
      --site-packages "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && curl --fail --location --silent --show-error \
      -o /tmp/mongodb.tgz "https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}.tgz" \
    && echo "${MONGODB_SHA256}  /tmp/mongodb.tgz" | sha256sum -c - \
    && tar xzf /tmp/mongodb.tgz -C /tmp \
    && DBBIN="$(python -c 'import fiftyone.db; print(fiftyone.db.FIFTYONE_DB_BIN_DIR)')" \
    && mkdir -p "$DBBIN" /opt/fiftyone/mongodb-notices \
    && cp "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/bin/mongod" "$DBBIN/mongod" \
    && chmod 0755 "$DBBIN/mongod" \
    && cp "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/LICENSE-Community.txt" \
      "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/MPL-2" \
      "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/THIRD-PARTY-NOTICES" \
      /opt/fiftyone/mongodb-notices/ \
    && rm -rf /tmp/mongodb.tgz "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}" \
    && "$DBBIN/mongod" --version | head -n 1 \
    && python -m pip check

COPY --chown=ubuntu:ubuntu docker/workbench/fiftyone/smoke_env.py /opt/npa/docker/workbench/fiftyone/smoke_env.py
COPY --chown=ubuntu:ubuntu docker/workbench/fiftyone/smoke_functional.py /opt/npa/docker/workbench/fiftyone/smoke_functional.py
COPY --chown=ubuntu:ubuntu --chmod=0755 docker/workbench/fiftyone/entrypoint.sh /opt/npa/docker/workbench/fiftyone/entrypoint.sh

RUN python -m py_compile /opt/npa/docker/workbench/fiftyone/smoke_env.py /opt/npa/docker/workbench/fiftyone/smoke_functional.py \
    && python - <<'PY'
from importlib import metadata
import os

expected = os.environ["FIFTYONE_VERSION"]
actual = metadata.version("fiftyone")
if actual != expected:
    raise RuntimeError(f"expected fiftyone {expected}, found {actual}")
print(f"FIFTYONE_BUILD_VERSION_OK {actual}")
PY

EXPOSE 5151
ENTRYPOINT ["/opt/npa/docker/workbench/fiftyone/entrypoint.sh"]

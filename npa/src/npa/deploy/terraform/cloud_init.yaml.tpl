#cloud-config

users:
  - name: ${ssh_user}
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ${ssh_public_key}

write_files:
%{ if workbench_type == "fiftyone" ~}
  - path: /etc/apt/apt.conf.d/99npa-network
    permissions: "0644"
    content: |
      Acquire::ForceIPv4 "true";
      Acquire::Languages "none";
      Acquire::Retries "5";

  - path: /etc/npa-fiftyone/env
    owner: ${ssh_user}:${ssh_user}
    permissions: "0600"
    content: |
      FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0
      FIFTYONE_DEFAULT_APP_PORT=${server_port}
      FIFTYONE_DATABASE_DIR=/opt/fiftyone/db
      FIFTYONE_DEFAULT_DATASET_DIR=/opt/fiftyone/datasets
      FIFTYONE_DATASET_ZOO_DIR=/opt/fiftyone/zoo/datasets
      FIFTYONE_MODEL_ZOO_DIR=/opt/fiftyone/zoo/models
      FIFTYONE_DO_NOT_TRACK=true
      FIFTYONE_DATASET_NAME=
      AWS_ACCESS_KEY_ID=${aws_access_key}
      AWS_SECRET_ACCESS_KEY=${aws_secret_key}
      AWS_ENDPOINT_URL=${s3_endpoint}
      NEBIUS_S3_ENDPOINT=${s3_endpoint}
      NEBIUS_S3_BUCKET=${s3_bucket}
      NEBIUS_REGION=${nebius_region}
      PYTHONUNBUFFERED=1

  - path: /opt/fiftyone/app.py
    owner: ${ssh_user}:${ssh_user}
    permissions: "0644"
    content: |
      from __future__ import annotations

      import os
      import signal
      import time

      import fiftyone as fo

      _stop = False


      def _handle_stop(signum, frame):
          global _stop
          _stop = True


      signal.signal(signal.SIGINT, _handle_stop)
      signal.signal(signal.SIGTERM, _handle_stop)

      dataset_name = os.environ.get("FIFTYONE_DATASET_NAME", "").strip()
      dataset = None
      if dataset_name:
          try:
              if dataset_name in fo.list_datasets():
                  dataset = fo.load_dataset(dataset_name)
              else:
                  print(f"Dataset {dataset_name!r} not found; launching empty app", flush=True)
          except Exception as exc:
              print(f"Could not load dataset {dataset_name!r}: {exc}", flush=True)

      address = os.environ.get("FIFTYONE_DEFAULT_APP_ADDRESS", "0.0.0.0")
      port = int(os.environ.get("FIFTYONE_DEFAULT_APP_PORT", "5151"))
      session = fo.launch_app(
          dataset,
          remote=True,
          address=address,
          port=port,
          auto=False,
      )
      print(f"NPA_FIFTYONE_APP_READY http://{address}:{port}", flush=True)

      try:
          while not _stop:
              time.sleep(1)
      finally:
          session.close()
%{ else ~}
  - path: /etc/apt/apt.conf.d/99lerobot-network
    permissions: "0644"
    content: |
      Acquire::ForceIPv4 "true";
      Acquire::Languages "none";
      Acquire::Retries "5";

  - path: /opt/lerobot/.env
    owner: ${ssh_user}:${ssh_user}
    permissions: "0600"
    content: |
      AWS_ACCESS_KEY_ID=${aws_access_key}
      AWS_SECRET_ACCESS_KEY=${aws_secret_key}
      NEBIUS_S3_ENDPOINT=${s3_endpoint}
      NEBIUS_S3_BUCKET=${s3_bucket}
      NEBIUS_REGION=${nebius_region}
      HF_LEROBOT_HOME=/opt/lerobot/hf_cache
      MUJOCO_GL=egl
      PYOPENGL_PLATFORM=egl
      PYTHONUNBUFFERED=1
%{ endif ~}

runcmd:
  - |
    set -e
    # runcmd string items are interpreted by /bin/sh, so keep this block POSIX-safe.
%{ if workbench_type == "cosmos" ~}
    COSMOS_DATA_DEVICE="/dev/disk/by-id/virtio-npa-cosmos-data"
    COSMOS_DATA_MOUNT="/opt/cosmos-data"

    echo "=== Cosmos data disk setup started - $(date) ==="
    for _ in $(seq 1 60); do
      if [ -e "$COSMOS_DATA_DEVICE" ]; then
        break
      fi
      sleep 2
    done
    if [ ! -e "$COSMOS_DATA_DEVICE" ]; then
      echo "ERROR: Cosmos data disk not found at $COSMOS_DATA_DEVICE"
      exit 1
    fi

    if ! blkid "$COSMOS_DATA_DEVICE" >/dev/null 2>&1; then
      mkfs.ext4 -F "$COSMOS_DATA_DEVICE"
    fi
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} "$COSMOS_DATA_MOUNT"
    COSMOS_DATA_UUID="$(blkid -s UUID -o value "$COSMOS_DATA_DEVICE")"
    if ! grep -q "$COSMOS_DATA_UUID" /etc/fstab; then
      echo "UUID=$COSMOS_DATA_UUID $COSMOS_DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
    fi
    mount "$COSMOS_DATA_MOUNT" || mount -a
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} \
      "$COSMOS_DATA_MOUNT/models" \
      "$COSMOS_DATA_MOUNT/hf_cache" \
      "$COSMOS_DATA_MOUNT/outputs"
    echo "=== Cosmos data disk setup complete - $(date) ==="
%{ endif ~}
%{ if workbench_type == "groot" ~}
    GROOT_DATA_DEVICE="/dev/disk/by-id/virtio-npa-groot-data"
    GROOT_DATA_MOUNT="/opt/groot-data"

    echo "=== GR00T data disk setup started - $(date) ==="
    for _ in $(seq 1 60); do
      if [ -e "$GROOT_DATA_DEVICE" ]; then
        break
      fi
      sleep 2
    done
    if [ ! -e "$GROOT_DATA_DEVICE" ]; then
      echo "ERROR: GR00T data disk not found at $GROOT_DATA_DEVICE"
      exit 1
    fi

    if ! blkid "$GROOT_DATA_DEVICE" >/dev/null 2>&1; then
      mkfs.ext4 -F "$GROOT_DATA_DEVICE"
    fi
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} "$GROOT_DATA_MOUNT"
    GROOT_DATA_UUID="$(blkid -s UUID -o value "$GROOT_DATA_DEVICE")"
    if ! grep -q "$GROOT_DATA_UUID" /etc/fstab; then
      echo "UUID=$GROOT_DATA_UUID $GROOT_DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
    fi
    mount "$GROOT_DATA_MOUNT" || mount -a
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} \
      "$GROOT_DATA_MOUNT/models" \
      "$GROOT_DATA_MOUNT/hf_cache" \
      "$GROOT_DATA_MOUNT/outputs" \
      "$GROOT_DATA_MOUNT/checkpoints" \
      "$GROOT_DATA_MOUNT/data_cache" \
      "$GROOT_DATA_MOUNT/checkpoint_cache" \
      "$GROOT_DATA_MOUNT/base_model_cache" \
      "$GROOT_DATA_MOUNT/eval_data_cache" \
      "$GROOT_DATA_MOUNT/config_cache"
    if [ -e /opt/groot ] && [ ! -L /opt/groot ]; then
      if [ -d /opt/groot ] && [ -z "$(ls -A /opt/groot)" ]; then
        rmdir /opt/groot
      else
        mkdir -p "$GROOT_DATA_MOUNT/legacy-root"
        mv /opt/groot/* "$GROOT_DATA_MOUNT/legacy-root/" 2>/dev/null || true
        rmdir /opt/groot 2>/dev/null || true
      fi
    fi
    if [ ! -e /opt/groot ]; then
      ln -s "$GROOT_DATA_MOUNT" /opt/groot
    fi
    chown -h ${ssh_user}:${ssh_user} /opt/groot
    chown -R ${ssh_user}:${ssh_user} "$GROOT_DATA_MOUNT"
    echo "=== GR00T data disk setup complete - $(date) ==="
%{ endif ~}
%{ if workbench_type == "fiftyone" ~}
    exec > /var/log/fiftyone-setup.log 2>&1
    echo "=== FiftyOne setup started - $(date) ==="

    FIFTYONE_HOME="/opt/fiftyone"
    FIFTYONE_VENV="$FIFTYONE_HOME/venv"

    export DEBIAN_FRONTEND=noninteractive
    apt-get update || { echo "ERROR: Failed to update apt package indexes"; exit 1; }
    apt-get install -y \
      build-essential \
      curl \
      ffmpeg \
      git \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      || { echo "ERROR: Failed to install system dependencies"; exit 1; }

    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} \
      "$FIFTYONE_HOME" \
      "$FIFTYONE_HOME/datasets" \
      "$FIFTYONE_HOME/db" \
      "$FIFTYONE_HOME/zoo/datasets" \
      "$FIFTYONE_HOME/zoo/models" \
      "/home/${ssh_user}/.fiftyone"
    chown -R ${ssh_user}:${ssh_user} "$FIFTYONE_HOME" "/home/${ssh_user}/.fiftyone" /etc/npa-fiftyone/env
    chmod 600 /etc/npa-fiftyone/env

    if [ ! -x "$FIFTYONE_VENV/bin/python" ] || \
       ! sudo -H -u ${ssh_user} "$FIFTYONE_VENV/bin/python" -c "from importlib import metadata; raise SystemExit(0 if metadata.version('fiftyone') == '${fiftyone_version}' else 1)" 2>/dev/null
    then
      rm -rf "$FIFTYONE_VENV"
      sudo -H -u ${ssh_user} python3 -m venv "$FIFTYONE_VENV" || { echo "ERROR: Failed to create FiftyOne venv"; exit 1; }
      sudo -H -u ${ssh_user} "$FIFTYONE_VENV/bin/python" -m pip install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip"; exit 1; }
      sudo -H -u ${ssh_user} "$FIFTYONE_VENV/bin/python" -m pip install "fiftyone==${fiftyone_version}" boto3 datasets huggingface_hub pyarrow pillow || { echo "ERROR: Failed to install FiftyOne"; exit 1; }
    fi

    sudo -H -u ${ssh_user} "$FIFTYONE_VENV/bin/python" - <<'PY'
    from importlib import metadata

    version = metadata.version("fiftyone")
    if version != "${fiftyone_version}":
        raise RuntimeError(f"expected fiftyone ${fiftyone_version}, found {version}")
    print("FIFTYONE_ENV_SMOKE_OK")
    PY

    cat > "/home/${ssh_user}/.fiftyone/config.json" <<JSON
    {
      "default_app_address": "0.0.0.0",
      "default_app_port": ${server_port}
    }
    JSON
    chown ${ssh_user}:${ssh_user} "/home/${ssh_user}/.fiftyone/config.json"
    chmod 600 "/home/${ssh_user}/.fiftyone/config.json"

    cat > /etc/systemd/system/npa-fiftyone-app.service <<UNIT
    [Unit]
    Description=NPA FiftyOne App
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=${ssh_user}
    WorkingDirectory=/opt/fiftyone
    EnvironmentFile=/etc/npa-fiftyone/env
    ExecStart=/opt/fiftyone/venv/bin/python /opt/fiftyone/app.py
    Restart=always
    RestartSec=10
    KillMode=control-group
    TimeoutStopSec=15
    SendSIGKILL=yes

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable npa-fiftyone-app
    systemctl restart npa-fiftyone-app

    for _ in $(seq 1 120); do
      if curl -fsS "http://127.0.0.1:${server_port}/" >/dev/null; then
        echo "NPA_FIFTYONE_APP_READY"
        echo "=== FiftyOne setup complete - $(date) ==="
        exit 0
      fi
      sleep 1
    done

    echo "WARNING: FiftyOne app did not respond on port ${server_port} before cloud-init readiness timeout"
    systemctl --no-pager status npa-fiftyone-app || true
    echo "=== FiftyOne setup complete with app readiness warning - $(date) ==="
    exit 0
%{ else ~}
%{ if workbench_type == "groot" ~}
    exec > /var/log/groot-base-setup.log 2>&1
    echo "=== GR00T base VM setup started - $(date) ==="

    export DEBIAN_FRONTEND=noninteractive
    apt-get update || { echo "ERROR: Failed to update apt package indexes"; exit 1; }
    apt-get install -y \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libsm6 \
      libxext6 \
      libxrender-dev \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      wget \
      || { echo "ERROR: Failed to install GR00T base dependencies"; exit 1; }

    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} /opt/groot /opt/isaac-lab
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} \
      /opt/groot/models \
      /opt/groot/hf_cache \
      /opt/groot/outputs \
      /opt/groot/checkpoints \
      /opt/groot/data_cache \
      /opt/groot/checkpoint_cache \
      /opt/groot/base_model_cache \
      /opt/groot/eval_data_cache \
      /opt/groot/config_cache
    usermod -aG video,render ${ssh_user} || true

    echo "=== GR00T base VM setup complete - $(date) ==="
%{ else ~}
%{ if workbench_type == "lerobot-container" ~}
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} /opt/lerobot
    chown ${ssh_user}:${ssh_user} /opt/lerobot/.env 2>/dev/null || true
    chmod 600 /opt/lerobot/.env 2>/dev/null || true
    exec > /var/log/lerobot-container-setup.log 2>&1
    echo "=== LeRobot container VM setup started - $(date) ==="

    DEPLOY_ROOT="/opt/lerobot"
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} \
      "$DEPLOY_ROOT/checkpoints" \
      "$DEPLOY_ROOT/job_status" \
      "$DEPLOY_ROOT/dataset_cache" \
      "$DEPLOY_ROOT/checkpoint_cache" \
      "$DEPLOY_ROOT/benchmarks" \
      "$DEPLOY_ROOT/hf_cache" \
      "/var/log/npa-lerobot"

    usermod -aG video,render ${ssh_user} || true

    echo "=== LeRobot container VM setup complete - $(date) ==="
%{ else ~}
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} /opt/lerobot
    chown ${ssh_user}:${ssh_user} /opt/lerobot/.env 2>/dev/null || true
    chmod 600 /opt/lerobot/.env 2>/dev/null || true
    exec > /var/log/lerobot-setup.log 2>&1
    echo "=== LeRobot setup started - $(date) ==="

    DEPLOY_ROOT="/opt/lerobot"
    LEROBOT_VENV="$DEPLOY_ROOT/venv"

    # Only create data, logs, and hf_cache dirs. Do NOT create runs/ -
    # LeRobot expects output_dir to not exist unless resuming.
    mkdir -p "$DEPLOY_ROOT/data" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/hf_cache"

    echo "Installing system dependencies..."
    apt-get update || { echo "ERROR: Failed to update apt package indexes"; exit 1; }
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      git \
      wget \
      curl \
      ca-certificates \
      python3 \
      python3-venv \
      python3-dev \
      python3-pip \
      ffmpeg \
      libssl-dev \
      libffi-dev \
      libsm6 \
      libxext6 \
      libxrender-dev \
      || { echo "ERROR: Failed to install system dependencies"; exit 1; }

    # Install Node.js 22.x (for npm)
    echo "Installing Node.js 22.x..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - || { echo "ERROR: Failed to add NodeSource repo"; exit 1; }
    apt-get install -y nodejs || { echo "ERROR: Failed to install nodejs"; exit 1; }
    echo "Node $(node --version), npm $(npm --version)"

    # Some CUDA images ship the compute driver without the NVIDIA EGL/GL user-space
    # libraries. LIBERO/robosuite need these for headless EGL rendering.
    ensure_nvidia_egl() {
      NVIDIA_CFG_PKG="$(dpkg-query -W -f='$${Package}\n' 'libnvidia-cfg1-*' 2>/dev/null | sed -n 's/^\(libnvidia-cfg1-[0-9][0-9]*\)$/\1/p' | head -n 1)"
      if [ -z "$NVIDIA_CFG_PKG" ]; then
        echo "WARNING: No libnvidia-cfg1-* package found; skipping EGL userspace repair"
        return 0
      fi

      NVIDIA_BRANCH="$${NVIDIA_CFG_PKG##libnvidia-cfg1-}"
      NVIDIA_VERSION="$(dpkg-query -W -f='$${Version}' "$NVIDIA_CFG_PKG" 2>/dev/null || true)"
      if [ -z "$NVIDIA_VERSION" ]; then
        echo "ERROR: Could not determine NVIDIA userspace version"
        return 1
      fi

      if ! ldconfig -p | grep -q 'libEGL_nvidia.so.0' || \
         ! grep -Rqs 'libEGL_nvidia.so.0' /usr/share/glvnd/egl_vendor.d 2>/dev/null
      then
        echo "Installing NVIDIA EGL/GL userspace for branch $NVIDIA_BRANCH ($NVIDIA_VERSION)..."
        apt-get install -y \
          "libnvidia-common-$NVIDIA_BRANCH=$NVIDIA_VERSION" \
          "libnvidia-gl-$NVIDIA_BRANCH=$NVIDIA_VERSION" \
          libnvidia-egl-gbm1 \
          libnvidia-egl-wayland1 \
          libnvidia-egl-xcb1 \
          libnvidia-egl-xlib1 \
          || { echo "ERROR: Failed to install NVIDIA EGL/GL userspace"; return 1; }
        ldconfig
      fi

      if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && ldconfig -p | grep -q 'libEGL_nvidia.so.0'; then
        install -d -m 0755 /usr/share/glvnd/egl_vendor.d
        printf '%s\n' \
          '{' \
          '    "file_format_version" : "1.0.0",' \
          '    "ICD" : {' \
          '        "library_path" : "libEGL_nvidia.so.0"' \
          '    }' \
          '}' \
          > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
      fi

      ldconfig -p | grep -q 'libEGL_nvidia.so.0' || {
        echo "ERROR: libEGL_nvidia.so.0 missing after userspace install"
        return 1
      }
      grep -Rqs 'libEGL_nvidia.so.0' /usr/share/glvnd/egl_vendor.d 2>/dev/null || {
        echo "ERROR: NVIDIA EGL vendor manifest missing after userspace install"
        return 1
      }
    }

    ensure_nvidia_egl || exit 1

    # Create venv and install LeRobot from PyPI with pusht env extra
    echo "Creating Python venv at $LEROBOT_VENV..."
    python3 -m venv "$LEROBOT_VENV" || { echo "ERROR: Failed to create venv"; exit 1; }

    echo "Upgrading pip and installing dependencies..."
    "$LEROBOT_VENV/bin/pip" install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip"; exit 1; }

    echo "Installing LeRobot ${lerobot_version}..."
    "$LEROBOT_VENV/bin/pip" install "lerobot[pusht,libero]==${lerobot_version}" boto3 wandb tensorboard num2words || { echo "ERROR: Failed to install packages"; exit 1; }

    # Verify installation
    echo "Verifying LeRobot installation..."
    "$LEROBOT_VENV/bin/python" -c "import lerobot; print(f'LeRobot {lerobot.__version__}')" || { echo "ERROR: LeRobot import failed"; exit 1; }
    "$LEROBOT_VENV/bin/python" -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch import failed"; exit 1; }

    # Expose the venv first in /usr/local/bin so lerobot works without activation.
    # Do not replace /usr/bin/python3: the venv's python3 symlink points back to the
    # system interpreter, so rewriting /usr/bin/python3 would create a symlink loop.
    ln -sf "$LEROBOT_VENV/bin/python"  /usr/local/bin/python
    ln -sf "$LEROBOT_VENV/bin/python3" /usr/local/bin/python3
    ln -sf "$LEROBOT_VENV/bin/pip"     /usr/local/bin/pip3

    # Global venv activation via /etc/profile.d (works for all users)
    cat > /etc/profile.d/lerobot.sh <<'GLOBAL_EOF'
    if [ -f /opt/lerobot/venv/bin/activate ]; then
      . /opt/lerobot/venv/bin/activate
    fi
    if [ -f /opt/lerobot/.env ]; then
      set -a
      . /opt/lerobot/.env
      set +a
    fi
    GLOBAL_EOF
    chmod 644 /etc/profile.d/lerobot.sh

    # Also add to user's .bashrc for non-login shells
    echo 'set -a; source /opt/lerobot/.env; set +a' >> /home/${ssh_user}/.bashrc
    echo 'source /opt/lerobot/venv/bin/activate'    >> /home/${ssh_user}/.bashrc

    # Add to root's .bashrc as well
    echo 'set -a; source /opt/lerobot/.env; set +a' >> /root/.bashrc
    echo 'source /opt/lerobot/venv/bin/activate'    >> /root/.bashrc

    # Headless EGL access requires the login user to be able to open the DRM nodes.
    usermod -aG video,render ${ssh_user} || true

    # Seed LIBERO's config so the first training run does not stop for an
    # interactive dataset-path prompt.
    LIBERO_ROOT="$("$LEROBOT_VENV/bin/python" -c 'from pathlib import Path; import sysconfig; print((Path(sysconfig.get_paths()["purelib"]).resolve() / "libero" / "libero"))')"
    LIBERO_DATASETS="$(dirname "$LIBERO_ROOT")/datasets"
    install -d -m 0755 -o ${ssh_user} -g ${ssh_user} "/home/${ssh_user}/.libero"
    printf '%s\n' \
      "benchmark_root: $LIBERO_ROOT" \
      "bddl_files: $LIBERO_ROOT/./bddl_files" \
      "init_states: $LIBERO_ROOT/./init_files" \
      "datasets: $LIBERO_DATASETS" \
      "assets: $LIBERO_ROOT/./assets" \
      > "/home/${ssh_user}/.libero/config.yaml"
    chown ${ssh_user}:${ssh_user} "/home/${ssh_user}/.libero/config.yaml"

    chown -R ${ssh_user}:${ssh_user} "$DEPLOY_ROOT"

    # Final verification: test python command
    echo "Testing system python command..."
    python -c "import lerobot; print(f'System python works: LeRobot {lerobot.__version__}')" || echo "WARNING: System python failed (will work after login)"

    echo "=== LeRobot setup complete - $(date) ==="
    echo "Setup logs saved to: /var/log/cloud-init-output.log"
    echo "LeRobot setup logs saved to: /var/log/lerobot-setup.log"
%{ endif ~}
%{ endif ~}
%{ endif ~}

#cloud-config

users:
  - name: ${ssh_user}
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ${ssh_public_key}

package_update: true

packages:
  - build-essential
  - git
  - wget
  - curl
  - ca-certificates
  - python3
  - python3-venv
  - python3-dev
  - python3-pip
  - ffmpeg
  - libssl-dev
  - libffi-dev
  - libsm6
  - libxext6
  - libxrender-dev

write_files:
  - path: /opt/lerobot/.env
    permissions: "0600"
    content: |
      AWS_ACCESS_KEY_ID=${aws_access_key}
      AWS_SECRET_ACCESS_KEY=${aws_secret_key}
      NEBIUS_S3_ENDPOINT=${s3_endpoint}
      NEBIUS_S3_BUCKET=${s3_bucket}
      NEBIUS_REGION=${nebius_region}
      HF_LEROBOT_HOME=/opt/lerobot/hf_cache
      MUJOCO_GL=egl
      PYTHONUNBUFFERED=1

runcmd:
  - |
    set -e
    # runcmd string items are interpreted by /bin/sh, so keep this block POSIX-safe.
    exec > /var/log/lerobot-setup.log 2>&1
    echo "=== LeRobot setup started — $(date) ==="

    DEPLOY_ROOT="/opt/lerobot"
    LEROBOT_VENV="$DEPLOY_ROOT/venv"

    # Only create data, logs, and hf_cache dirs. Do NOT create runs/ —
    # LeRobot expects output_dir to not exist unless resuming.
    mkdir -p "$DEPLOY_ROOT/data" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/hf_cache"

    # Install Node.js 22.x (for npm)
    echo "Installing Node.js 22.x..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - || { echo "ERROR: Failed to add NodeSource repo"; exit 1; }
    apt-get install -y nodejs || { echo "ERROR: Failed to install nodejs"; exit 1; }
    echo "✓ Node $(node --version), npm $(npm --version)"

    # Some CUDA images ship the compute driver without the NVIDIA EGL/GL user-space
    # libraries. LIBERO/robosuite need these for headless EGL rendering.
    NVIDIA_CFG_PKG="$(dpkg-query -W -f='$${Package}\n' 'libnvidia-cfg1-*' 2>/dev/null | sed -n 's/^\(libnvidia-cfg1-[0-9][0-9]*\)$/\1/p' | head -n 1)"
    if [ -n "$NVIDIA_CFG_PKG" ]; then
      NVIDIA_BRANCH="$${NVIDIA_CFG_PKG##libnvidia-cfg1-}"
      NVIDIA_VERSION="$(dpkg-query -W -f='$${Version}' "$NVIDIA_CFG_PKG" 2>/dev/null || true)"
      if [ -n "$NVIDIA_VERSION" ]; then
        echo "Installing NVIDIA EGL/GL userspace for branch $NVIDIA_BRANCH ($NVIDIA_VERSION)..."
        apt-get install -y \
          "libnvidia-common-$NVIDIA_BRANCH=$NVIDIA_VERSION" \
          "libnvidia-gl-$NVIDIA_BRANCH=$NVIDIA_VERSION" \
          libnvidia-egl-gbm1 \
          libnvidia-egl-wayland1 \
          libnvidia-egl-xcb1 \
          libnvidia-egl-xlib1 \
          || { echo "ERROR: Failed to install NVIDIA EGL/GL userspace"; exit 1; }
      fi
    fi

    # Create venv and install LeRobot from PyPI with pusht env extra
    echo "Creating Python venv at $LEROBOT_VENV..."
    python3 -m venv "$LEROBOT_VENV" || { echo "ERROR: Failed to create venv"; exit 1; }

    echo "Upgrading pip and installing dependencies..."
    "$LEROBOT_VENV/bin/pip" install --upgrade pip setuptools wheel || { echo "ERROR: Failed to upgrade pip"; exit 1; }

    echo "Installing LeRobot ${lerobot_version}..."
    "$LEROBOT_VENV/bin/pip" install "lerobot[pusht,libero]==${lerobot_version}" boto3 wandb tensorboard || { echo "ERROR: Failed to install packages"; exit 1; }

    # Verify installation
    echo "Verifying LeRobot installation..."
    "$LEROBOT_VENV/bin/python" -c "import lerobot; print(f'✓ LeRobot {lerobot.__version__}')" || { echo "ERROR: LeRobot import failed"; exit 1; }
    "$LEROBOT_VENV/bin/python" -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch import failed"; exit 1; }

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
    python -c "import lerobot; print(f'✓ System python works: LeRobot {lerobot.__version__}')" || echo "WARNING: System python failed (will work after login)"

    echo "=== LeRobot setup complete — $(date) ==="
    echo "Setup logs saved to: /var/log/cloud-init-output.log"
    echo "LeRobot setup logs saved to: /var/log/lerobot-setup.log"

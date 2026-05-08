#!/bin/bash
set -e

echo "=== Genesis Simulation Environment Setup ==="
echo "Target: GPU VM with CUDA 12.4+"

# Create conda environment (skip if it already exists)
echo "[1/6] Creating conda environment..."
if conda env list | grep -q "^genesis "; then
    echo "  Conda env 'genesis' already exists — reusing."
else
    conda create -n genesis python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate genesis

# Install PyTorch with CUDA 12.4
echo "[2/6] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Genesis
echo "[3/6] Installing Genesis..."
pip install genesis-world

# Install RL library (rsl-rl, same as Isaac Lab)
echo "[4/7] Installing rsl-rl and tensorboard..."
pip install rsl-rl-lib==2.2.4 tensorboard

# Install LeRobot (needed by eval_student to load student policy checkpoints)
echo "[5/7] Installing LeRobot for student policy loading..."
pip install lerobot

# Install NPA CLI (provides npa entrypoint for remote workflow commands)
echo "[6/7] Installing NPA CLI..."
if [ -d /opt/npa/repo ]; then
    pip install -e "/opt/npa/repo/npa[genesis]"
else
    echo "WARNING: NPA repo not found at /opt/npa/repo."
    echo "  Clone the repo and reinstall: git clone <repo> /opt/npa/repo && pip install -e /opt/npa/repo/npa[genesis]"
    echo "  Installing standalone dependencies instead..."
    pip install typer paramiko pyyaml rich boto3 httpx jinja2 pyarrow numpy
fi

# Verify npa entrypoint
echo "[7/8] Verifying npa CLI..."
if command -v npa &>/dev/null; then
    npa --help | head -5
    echo "npa CLI OK"
else
    echo "WARNING: npa CLI not on PATH. Remote workflow commands will not work."
fi

# EGL checks for headless GPU rendering
echo "[8/8] Checking EGL setup for headless rendering..."
echo "--- EGL vendor configuration ---"
ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || echo "No EGL vendor dir found"

echo "--- NVIDIA EGL library ---"
ls -la /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>/dev/null || echo "WARNING: libEGL_nvidia not found"

echo "--- Mesa EGL conflict check ---"
dpkg -l 2>/dev/null | grep mesa-egl && echo "WARNING: mesa-egl found — may conflict with nvidia EGL" || echo "OK: no mesa-egl conflict"

echo "--- Genesis import test ---"
python -c "import genesis; print(f'Genesis {genesis.__version__} imported OK')"

echo "--- PyTorch CUDA test ---"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

echo ""
echo "=== Genesis setup complete ==="
echo "Activate with: conda activate genesis"

#!/bin/bash
set -e

echo "=== LeRobot Training Environment Setup ==="
echo "Target: GPU VM with CUDA 12.4+"

# Create conda environment (skip if it already exists)
echo "[1/5] Creating conda environment..."
if conda env list | grep -q "^lerobot "; then
    echo "  Conda env 'lerobot' already exists — reusing."
else
    conda create -n lerobot python=3.12 -y
fi
eval "$(conda shell.bash hook)"
conda activate lerobot

# Install PyTorch with CUDA 12.4
echo "[2/5] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install LeRobot (pin to 0.5.1 to match cloud-init and npa adapter expectations)
echo "[3/5] Installing LeRobot 0.5.1..."
pip install "lerobot[pusht]==0.5.1"

# Install NPA CLI (provides npa entrypoint for remote workflow commands)
echo "[4/6] Installing NPA CLI..."
if [ -d /opt/npa/repo ]; then
    pip install -e "/opt/npa/repo/npa[adapter]"
else
    echo "WARNING: NPA repo not found at /opt/npa/repo."
    echo "  Clone the repo and reinstall: git clone <repo> /opt/npa/repo && pip install -e /opt/npa/repo/npa[adapter]"
    echo "  Installing standalone dependencies instead..."
    pip install typer paramiko pyyaml rich boto3 httpx jinja2 pyarrow numpy
fi

# Verify npa entrypoint
echo "[5/6] Verifying npa CLI..."
if command -v npa &>/dev/null; then
    npa --help | head -5
    echo "npa CLI OK"
else
    echo "WARNING: npa CLI not on PATH. Remote workflow commands will not work."
fi

# Verify installation
echo "[6/6] Verifying installation..."

echo "--- LeRobot import test ---"
python -c "from lerobot.datasets import LeRobotDataset; print('LeRobot imported OK')"

echo "--- PyTorch CUDA test ---"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

echo ""
echo "=== LeRobot setup complete ==="
echo "Activate with: conda activate lerobot"

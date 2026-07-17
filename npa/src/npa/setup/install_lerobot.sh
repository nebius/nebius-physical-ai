#!/bin/bash
set -e

# Optional override. Default matches [tool.npa.supported-tools].lerobot.
LEROBOT_VERSION="${LEROBOT_VERSION:-0.5.1}"

echo "=== LeRobot Training Environment Setup ==="
echo "Target: GPU VM with CUDA 12.4+"
echo "LeRobot version: ${LEROBOT_VERSION}"

case "$LEROBOT_VERSION" in
  0.5.1)
    LEROBOT_PIP_SPEC="lerobot[pusht]==${LEROBOT_VERSION}"
    ;;
  0.6.0)
    # Lean 0.6.0 base needs training + PushT extras for workbench train/eval.
    LEROBOT_PIP_SPEC="lerobot[training,evaluation,pusht]==${LEROBOT_VERSION}"
    ;;
  *)
    echo "ERROR: unsupported LEROBOT_VERSION=${LEROBOT_VERSION} (supported: 0.5.1, 0.6.0)" >&2
    exit 2
    ;;
esac

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
# For 0.6.0, upstream caps torch < 2.12; let the lerobot resolver pick a compatible wheel.
echo "[2/5] Installing PyTorch (CUDA 12.4)..."
if [ "$LEROBOT_VERSION" = "0.6.0" ]; then
    echo "  Skipping standalone torch pin for 0.6.0 (installed via lerobot deps)."
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Install LeRobot
echo "[3/5] Installing LeRobot ${LEROBOT_VERSION}..."
pip install "$LEROBOT_PIP_SPEC"

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
python -c "from lerobot.datasets import LeRobotDataset; import lerobot; print(f'LeRobot {lerobot.__version__} imported OK')"

echo "--- PyTorch CUDA test ---"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

echo ""
echo "=== LeRobot setup complete ==="
echo "Activate with: conda activate lerobot"
echo "Selected version: ${LEROBOT_VERSION}"

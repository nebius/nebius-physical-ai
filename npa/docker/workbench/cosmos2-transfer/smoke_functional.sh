#!/usr/bin/env bash
# Golden-eval CUDA probe for npa-cosmos2-transfer.
# Published transfer images install PyTorch in the cosmos-transfer2.5 uv venv
# (Python 3.10 + cu128), not on the default system/python alias.
set -euo pipefail

PROBE='import torch; assert torch.cuda.is_available(); t=torch.ones((64, 64), device="cuda"); print("[PASS] cuda", torch.cuda.get_device_name(0), float(t.sum()))'

for candidate in \
  /opt/cosmos/cosmos-transfer2.5/.venv/bin/python \
  /opt/cosmos/venv/bin/python \
  /workspace/.venv/bin/python; do
  if [[ -x "${candidate}" ]]; then
    "${candidate}" -c "${PROBE}"
    exit 0
  fi
done

for candidate in python3 python; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    if "${candidate}" -c "${PROBE}"; then
      exit 0
    fi
  fi
done

echo "[FAIL] no python with torch+cuda found" >&2
exit 1

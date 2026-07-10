"""Cosmos2-Transfer golden eval — a REAL capability test.

Runs an actual Cosmos-Transfer2.5 video-to-video world-transfer inference on a
bundled robot control example and asserts a non-trivial generated video is
produced. This exercises the container's real job (synthetic-data augmentation
via world transfer), not just a CUDA/import probe.

The transfer runtime + inference venv live in
``/opt/cosmos/cosmos-transfer2.5`` (Python 3.10 + torch cu128 + flash-attn). This
module shells out to that venv so it stays import-safe on the default interpreter
(the golden-eval driver also invokes ``smoke_functional.sh`` directly).
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

REPO = os.environ.get("COSMOS_TRANSFER_REPO", "/opt/cosmos/cosmos-transfer2.5")
SPEC = os.environ.get(
    "COSMOS_TRANSFER_SPEC", "assets/robot_example/depth/robot_depth_spec.json"
)
OUT = os.environ.get("COSMOS_TRANSFER_OUT", "outputs/golden-eval")


def main() -> int:
    venv_python = os.path.join(REPO, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        print(f"[FAIL] cosmos-transfer2.5 inference venv missing at {venv_python}")
        return 1

    out_dir = os.path.join(REPO, OUT)
    print(f"[run] cosmos-transfer2.5 inference: {SPEC} -> {OUT}")
    proc = subprocess.run(
        [venv_python, "examples/inference.py", "-i", SPEC, "-o", OUT],
        cwd=REPO,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[FAIL] inference exited {proc.returncode}")
        return proc.returncode

    videos = [
        f
        for f in glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True)
        if "control" not in os.path.basename(f)
    ]
    big = [f for f in videos if os.path.getsize(f) > 100_000]
    if not big:
        print(f"[FAIL] no non-trivial output video in {out_dir}: "
              f"{[(f, os.path.getsize(f)) for f in videos]}")
        return 1
    result = big[0]
    print(f"[PASS] cosmos-transfer2.5 generated {result} ({os.path.getsize(result)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

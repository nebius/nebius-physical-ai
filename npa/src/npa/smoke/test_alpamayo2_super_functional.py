"""Real Alpamayo 2 Super golden evaluation executed inside the GPU image."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from npa.workbench.alpamayo2_super.runtime import Alpamayo2SuperRequest, run_inference


def main() -> int:
    """Predict one upstream trajectory and verify the useful artifacts."""

    with tempfile.TemporaryDirectory(prefix="alpamayo2-super-golden-") as scratch:
        output = Path(scratch) / "artifacts"
        result = run_inference(
            Alpamayo2SuperRequest(
                output_path=str(output),
                diffusion_steps=10,
                run_id=os.environ.get("NPA_RUN_ID", "golden-eval"),
                runtime_image=os.environ.get("NPA_TASK_IMAGE", ""),
            )
        )
        trajectory = json.loads(
            (output / "trajectory.json").read_text(encoding="utf-8")
        )
        if result["status"] != "ok" or not trajectory:
            raise RuntimeError("Alpamayo 2 Super produced no trajectory")
        if (output / "trajectory.png").stat().st_size == 0:
            raise RuntimeError("Alpamayo 2 Super produced an empty visualization")
        if (output / "result.json").stat().st_size == 0:
            raise RuntimeError("Alpamayo 2 Super produced no provenance")
    print("ALPAMAYO2_SUPER_GOLDEN_EVAL_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Real flex-pi golden evaluation executed inside the RTX GPU image."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from npa.workbench.flex_pi.runtime import FlexPiRequest, run_inference


def main() -> int:
    """Predict one action chunk and verify useful semantic artifacts."""
    with tempfile.TemporaryDirectory(prefix="flex-pi-golden-") as scratch:
        output = Path(scratch) / "artifacts"
        result = run_inference(FlexPiRequest(
            input_path="/opt/flex-pi/npa/public_robotwin_sample.json",
            output_path=str(output), expected_gpu="RTXPRO6000",
            run_id=os.environ.get("NPA_RUN_ID", "golden-eval"),
            runtime_image=os.environ.get("NPA_TASK_IMAGE", ""),
        ))
        actions = json.loads((output / "actions.json").read_text(encoding="utf-8"))
        if result["status"] != "ok" or len(actions.get("actions", [])) != 32:
            raise RuntimeError("flex-pi produced no valid action chunk")
        if (output / "result.json").stat().st_size == 0:
            raise RuntimeError("flex-pi produced no provenance")
    print("FLEX_PI_GOLDEN_EVAL_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

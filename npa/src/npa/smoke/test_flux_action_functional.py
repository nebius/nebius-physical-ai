"""Qualify a FLUX Action image with operator-provided demonstrations and schedule."""

from __future__ import annotations

import json
import os


def main() -> None:
    """Run native fine-tuning and require its verified S3 result.

    Args: None; NPA_FLUX_* environment variables select handoffs and GPU count.
    Returns: None; prints the completed artifact receipt.
    Raises: KeyError for missing inputs; FluxActionError for unsuccessful training.
    """
    from npa.sdk.workbench.flux_action import finetune

    result = finetune(
        input_path=os.environ["NPA_FLUX_INPUT_URI"],
        recipe_uri=os.environ["NPA_FLUX_RECIPE_URI"],
        output_path=os.environ["NPA_FLUX_OUTPUT_URI"],
        processes=int(os.environ.get("NPA_FLUX_PROCESSES", "8")),
    )
    if result["status"] != "completed":
        raise RuntimeError("native training did not complete")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""SDK for Alpamayo 2 Super inference."""

from typing import Any

from npa.workbench.alpamayo2_super.runtime import Alpamayo2SuperRequest, run_inference


def infer(**kwargs: Any) -> dict[str, Any]:
    """Run the same inference implementation used by CLI/API/workflow."""

    return run_inference(Alpamayo2SuperRequest(**kwargs))


__all__ = ["Alpamayo2SuperRequest", "infer"]

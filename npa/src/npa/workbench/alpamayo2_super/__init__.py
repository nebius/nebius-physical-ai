"""NVIDIA Alpamayo 2 Super workbench integration."""

from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    run_inference,
)

__all__ = [
    "DEFAULT_DATASET_REPO",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "Alpamayo2SuperError",
    "Alpamayo2SuperRequest",
    "run_inference",
]

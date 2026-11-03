"""Apply the DROID-only reader and eager verification hooks before native training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy


def _install_adapter():
    import native_video

    settings = json.loads(Path(os.environ["NPA_DROID_SETTINGS"]).read_text())
    native_video.install(settings)
    return settings


def _materialized_avae(settings):
    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

    register_checkpoints()
    _CHECKPOINTS["s3://bucket/pretrained/tokenizers/audio/avae"].hf._path = settings[
        "avae_cache"
    ]


if __name__ == "__main__":
    configuration = _install_adapter()
    _materialized_avae(configuration)
    import verification

    verification.install()
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")

"""Evaluate native LIBERO policies using explicit, disjoint validation and test states."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    """Run LeRobot evaluation with a checked offset into LIBERO's fixed initial states.

    Args:
        None; native evaluation arguments and NPA_VLA_INIT_OFFSET select the run.
    Returns:
        None.
    Raises:
        ValueError: An initial-state partition would wrap around or use async workers.
        RuntimeError: Native evaluation fails.
    """
    from libero import libero as benchmark_runtime

    assets = Path(os.environ["NPA_VLA_ASSETS"])
    if not (assets / "scenes").is_dir():
        raise ValueError("pinned LIBERO assets are missing")
    benchmark_runtime._assets_path_cache = str(assets)
    import lerobot.envs.libero as libero
    import lerobot.scripts.lerobot_eval as native

    offset = int(os.environ["NPA_VLA_INIT_OFFSET"])
    original = libero.LiberoEnv.__init__
    libero.LiberoEnv.__init__ = _partitioned_reset(original, offset)
    try:
        native.main()
    finally:
        libero.LiberoEnv.__init__ = original


def _partitioned_reset(original, offset):
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if offset < 0 or self._init_states is None:
            raise ValueError(
                "fixed, nonnegative LIBERO initial-state partitions are required"
            )
        if offset + self._reset_stride > len(self._init_states):
            raise ValueError("evaluation initial-state partition would wrap")
        self.init_state_id += offset

    return initialize


if __name__ == "__main__":
    main()

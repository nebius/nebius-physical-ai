"""Run pinned native LIBERO evaluation with NumPy-only checkpoint globals."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from evaluate import _check_revision


def _numpy_state_globals():
    import numpy as np
    from numpy.core.multiarray import _reconstruct

    return [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.float64))]


def _main(args, forwarded):
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    import torch

    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for directory, name in (
        ("framework", "framework"),
        ("simulation/LIBERO", "libero"),
    ):
        _check_revision(args.shared_root / directory, sources[name])
    script = (
        args.shared_root
        / "framework/cosmos_framework/simulation/libero/closed_loop_eval.py"
    )
    sys.argv = [str(script), *forwarded]
    # LIBERO's pinned initial states contain NumPy arrays. Keep weights-only
    # loading and allow their exact constructors, instead of enabling pickle.
    with torch.serialization.safe_globals(_numpy_state_globals()):
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--shared-root", type=Path, required=True)
    arguments, remainder = parser.parse_known_args()
    _main(arguments, remainder)

"""Reuse the measured telemetry reducer for a selected rank of the four-node run."""

import argparse
import importlib.util
from pathlib import Path


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    source = (
        Path(__file__).resolve().parent.parent
        / "cosmos3-wam-live-training-16/export_window.py"
    )
    spec = importlib.util.spec_from_file_location("wam_window_core", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core._main(arguments)


if __name__ == "__main__":
    _main()

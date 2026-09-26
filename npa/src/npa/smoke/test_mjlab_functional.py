"""Qualify real MJLab training, checkpoint reload, episode measurement and ONNX."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from npa.workbench.mjlab.schemas import EvalRequest, ExportRequest, TrainRequest
from npa.workbench.mjlab.worker import _evaluate, _export, _train


def main() -> None:
    """Run a one-update Cartpole capability smoke on a real CUDA GPU.

    Args:
        None.
    Returns:
        None; prints success after all real operations complete.
    Raises:
        RuntimeError: Training, evaluation, export or artifact checks fail.
    """
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    with tempfile.TemporaryDirectory(prefix="mjlab-functional-") as directory:
        root = Path(directory)
        training = root / "training"
        training.mkdir()
        common = {"task": "Mjlab-Cartpole-Balance", "output_path": "s3://smoke/mjlab"}
        _train(TrainRequest(**common, iterations=1, num_envs=16), {}, training)
        inputs = {"checkpoint": {"path": str(training / "checkpoint.pt")}}
        evaluation, exported = root / "eval", root / "export"
        evaluation.mkdir()
        exported.mkdir()
        report = _evaluate(
            EvalRequest(**common, checkpoint="s3://smoke/checkpoint.pt", episodes=2),
            inputs,
            evaluation,
        )
        assert report["episodes_completed"] == 2
        _export(
            ExportRequest(**common, checkpoint="s3://smoke/checkpoint.pt"),
            inputs,
            exported,
        )
        assert (exported / "policy.onnx").stat().st_size > 0
        print("MJLab native training, measured episodes and checked ONNX passed")


if __name__ == "__main__":
    main()

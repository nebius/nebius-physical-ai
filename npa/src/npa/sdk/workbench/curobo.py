"""cuRobo SDK, using the same request models and operations as CLI/API."""

from __future__ import annotations

from npa.workbench.curobo import runtime
from npa.workbench.curobo.schemas import PrepareRequest, RunRequest


def prepare(*, output_path: str, mode: str = "both"):
    return runtime.prepare(PrepareRequest(output_path=output_path, mode=mode))


def benchmark(*, input_path: str, output_path: str, run_id: str):
    return runtime.benchmark(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def plan(*, input_path: str, output_path: str, run_id: str):
    return runtime.plan(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def validate(*, input_path: str, output_path: str, run_id: str):
    return runtime.validate(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def visualize(*, input_path: str, output_path: str, run_id: str):
    return runtime.visualize(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )

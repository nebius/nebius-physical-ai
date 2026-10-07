"""Run the existing policy workflow locally with generated inputs and real stage workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from npa.orchestration.npa_workflow import load_spec, validate_spec
from npa.orchestration.npa_workflow.interpreter import run_workflow
from npa.workbench.dataset.storage import write_json_uri
from .demo_fixture import prepare


class _LocalReferenceExecutor:
    def __init__(self, workspace):
        self.workspace = workspace.resolve()
        self.steps = []
        self.prefix = "s3://example-bucket/policy-training/reference-demo/"

    def local(self, value):
        if not value.startswith("s3://"):
            return value
        if not value.startswith(self.prefix):
            raise ValueError("reference runner cannot access external storage")
        suffix = value.removeprefix(self.prefix)
        target = self.workspace.joinpath(suffix).resolve()
        if not target.is_relative_to(self.workspace):
            raise ValueError("reference path escapes the workspace")
        return str(target)

    def execute(self, step):
        started = time.monotonic()
        command = [sys.executable, *[self.local(arg) for arg in step.argv[1:]]]
        command = [
            "npa.workflows.policy_training.demo_stage"
            if arg == "npa.workflows.policy_training"
            else arg
            for arg in command
        ]
        print(f"Running {step.state}", flush=True)
        with (self.workspace / "workers.log").open("ab") as logs:
            result = subprocess.run(command, stdout=logs, stderr=logs, check=False)
        if result.returncode:
            raise RuntimeError(f"{step.state} failed; inspect private worker logs")
        record = {
            "state": step.state,
            "status": "ok",
            "seconds": round(time.monotonic() - started, 3),
        }
        self.steps.append(record)
        return record

    def read_decision(self, bucket, key):
        return Path(self.local(f"s3://{bucket}/{key}")).read_text()


def run_demo(output: Path, spec_path: Path, *, video: bool = True) -> dict:
    """Run the measured reference workflow and export a portable demonstration.

    Args:
        output: New directory for private evidence and shareable HTML/MP4.
        spec_path: Existing policy-training-slurm workflow YAML.
        video: Whether to render the HTML walkthrough to MP4.
    Returns:
        Sanitized summary of the completed run.
    Raises:
        FileExistsError: Output exists; previous evidence is never overwritten.
        RuntimeError: A real stage or artifact export fails.
    """
    from .demo_report import export

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "private"
    workspace.mkdir(mode=0o700)
    prepare(workspace)
    spec = load_spec(spec_path)
    validate_spec(spec)
    executor = _LocalReferenceExecutor(workspace)
    report = run_workflow(
        spec,
        run_id="reference-demo",
        execute=True,
        step_executor=executor,
        decision_reader=executor.read_decision,
    )
    if report["status"] != "completed":
        raise RuntimeError("reference workflow did not complete")
    write_json_uri(str(workspace / "workflow.json"), report)
    return export(workspace, output, executor.steps, video=video)


def main() -> None:
    """Run the checked-in reference from a local checkout.

    Args:
        None; reads output path and optional HTML-only flag from argv.
    Returns:
        None.
    Raises:
        SystemExit: Invalid invocation or failed stage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--html-only", action="store_true")
    arguments = parser.parse_args()
    os.umask(0o077)
    summary = run_demo(arguments.output, arguments.spec, video=not arguments.html_only)
    print(
        json.dumps(
            {
                "status": "completed",
                "steps": len(summary["steps"]),
                "pretrain_attempts": len(summary["pretrain"]),
                "finetune_attempts": len(summary["finetune"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

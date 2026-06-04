"""Cosmos runner functional smoke checks without model weights."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from npa.workbench.cosmos.workflows import (
    COSMOS_AUGMENT_YAML,
    COSMOS_REASON_YAML,
    COSMOS_ATTRIBUTION,
    build_cosmos_augment_env,
    build_cosmos_reason_env,
    launch_cosmos_sky_workflow,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_augment_dry_run() -> CheckResult:
    try:
        env = build_cosmos_augment_env(
            source="s3://example-bucket/input/sim.mp4",
            output_path="s3://example-bucket/output/augment/",
            prompt="preserve robot motion",
            control="edge",
            image="registry.example/npa-cosmos:3.0.0",
        )
        result = launch_cosmos_sky_workflow(
            yaml_path=COSMOS_AUGMENT_YAML,
            env=env,
            accelerator="TESTGPU:1",
            dry_run=True,
        )
        if result.status != "dry_run" or "--gpus" not in result.command:
            return CheckResult("render augment dry-run", False, str(result.command))
        return CheckResult("render augment dry-run", True, env["NPA_COSMOS_ATTRIBUTION"])
    except Exception as exc:
        return CheckResult("render augment dry-run", False, _format_exception(exc))


def check_reason_dry_run() -> CheckResult:
    try:
        env = build_cosmos_reason_env(
            input_path="s3://example-bucket/input/rollout.mp4",
            output_path="s3://example-bucket/output/reason/",
            criteria_prompt="did the robot complete the task?",
            model_size="nano",
            image="registry.example/npa-cosmos:3.0.0",
        )
        result = launch_cosmos_sky_workflow(
            yaml_path=COSMOS_REASON_YAML,
            env=env,
            dry_run=True,
        )
        if result.status != "dry_run" or "Cosmos3-Nano" not in result.env.values():
            return CheckResult("render reason dry-run", False, str(result.command))
        return CheckResult("render reason dry-run", True, env["NPA_COSMOS_REASON_MODEL_ID"])
    except Exception as exc:
        return CheckResult("render reason dry-run", False, _format_exception(exc))


def check_attribution() -> CheckResult:
    if COSMOS_ATTRIBUTION != "Built on NVIDIA Cosmos":
        return CheckResult("check attribution", False, COSMOS_ATTRIBUTION)
    return CheckResult("check attribution", True, COSMOS_ATTRIBUTION)


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_augment_dry_run,
        check_reason_dry_run,
        check_attribution,
    ]
    results = []
    for check in checks:
        result = check()
        results.append(result)
        _print_result(result)
    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Daily comprehensive-workflow E2E accounting for the dev-VM test runner.

Used by ``scripts/dev-vm-daily-tests.sh`` (tier ``e2e-daily``). Subcommands:

- ``check``: fail if a required image lost >= 4-step workflow coverage
  (regression guard the runner gates on).
- ``report`` / ``plan-set``: print the coverage table / today's rotating
  comprehensive spec set; the runner logs these for visibility.
- ``images``: resolve every workbench image to its pinned registry ref and,
  with ``--inspect``, check registry presence (the runner's all-image check).
- ``gpu-case``: print today's rotating real-GPU workflow twin (the runner
  submits it in the ``gpu-daily`` path).

All CPU-only. Note the runner does *not* call ``check_workflow_images.py`` or
the ``plan-spec`` CLI: registry reachability is done here by ``images
--inspect``, and every spec is validated + planned by the pytest smoke suite
(``test_all_workflow_yamls.py`` et al.). ``plan-set`` and the coverage helpers
in ``daily_coverage`` are operator-facing/reporting aids, exercised by the unit
guard and logged by the runner.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys

from npa.orchestration.npa_workflow import daily_coverage as dc


def _day_index() -> int:
    return _dt.datetime.now(_dt.timezone.utc).timetuple().tm_yday


def _cmd_report(as_json: bool) -> int:
    summaries = dc.spec_step_summary()
    report = dc.image_coverage(summaries)
    comp = dc.comprehensive_specs(summaries)
    if as_json:
        print(
            json.dumps(
                {
                    "total_specs": len(summaries),
                    "comprehensive_specs": len(comp),
                    "min_steps": dc.MIN_COMPREHENSIVE_STEPS,
                    "covered": sorted(report.covered),
                    "required": sorted(report.required),
                    "reachable": sorted(report.reachable),
                    "missing": sorted(report.missing),
                    "exempt": sorted(dc.EXEMPT_IMAGE_TOOLS),
                    "covering_workflows": {
                        k: sorted(v) for k, v in sorted(report.covering_workflows.items())
                    },
                    "ok": report.ok,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"npa.workflow specs: {len(summaries)} "
        f"(>= {dc.MIN_COMPREHENSIVE_STEPS}-step: {len(comp)})"
    )
    print(f"covered by >= {dc.MIN_COMPREHENSIVE_STEPS}-step workflows: {sorted(report.covered)}")
    print(f"exempt (tracked gap, not yet in a comprehensive workflow): {sorted(dc.EXEMPT_IMAGE_TOOLS)}")
    print("image -> covering comprehensive workflows:")
    for image in sorted(report.covering_workflows):
        print(f"  {image}: {', '.join(report.covering_workflows[image])}")
    if report.missing:
        print(f"MISSING required coverage: {sorted(report.missing)}")
    return 0


def _cmd_check() -> int:
    report = dc.image_coverage()
    try:
        dc.assert_coverage(report)
    except AssertionError as exc:
        print(f"coverage check FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"coverage check OK: every required image is in a >= "
        f"{dc.MIN_COMPREHENSIVE_STEPS}-step workflow ({sorted(report.covered)})"
    )
    return 0


def _resolve_all_images(registry: str | None) -> dict[str, str]:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES, container_image_for_tool

    resolved: dict[str, str] = {}
    for tool in sorted(CONTAINER_IMAGE_NAMES):
        try:
            resolved[tool] = (
                container_image_for_tool(tool, registry=registry)
                if registry
                else container_image_for_tool(tool)
            )
        except (ValueError, KeyError) as exc:
            resolved[tool] = f"<unresolved: {exc}>"
    return resolved


def _inspect_image(image: str) -> bool | None:
    from npa.guardrails.skypilot import inspect_image_exists

    try:
        return inspect_image_exists(image)
    except (RuntimeError, OSError, subprocess.SubprocessError):
        # No crane/skopeo/docker installed (RuntimeError), or a probe hung /
        # timed out (subprocess.TimeoutExpired) / errored: presence is UNKNOWN,
        # not absent. This check is report-only and must never fail the run.
        return None


def _cmd_images(registry: str | None, do_inspect: bool, require: bool, as_json: bool) -> int:
    resolved = _resolve_all_images(registry)
    presence: dict[str, bool | None] = {}
    if do_inspect:
        for tool, ref in resolved.items():
            presence[tool] = None if ref.startswith("<unresolved") else _inspect_image(ref)

    if as_json:
        print(
            json.dumps(
                {
                    "registry": registry or "(default)",
                    "inspected": do_inspect,
                    "images": {
                        tool: {"ref": ref, "present": presence.get(tool)}
                        for tool, ref in sorted(resolved.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"workbench images (registry: {registry or '(default)'}):")
        for tool, ref in sorted(resolved.items()):
            mark = ""
            if do_inspect:
                state = presence.get(tool)
                mark = {True: " [present]", False: " [ABSENT]", None: " [unknown]"}[state]
            print(f"  {tool}: {ref}{mark}")

    if require and do_inspect:
        missing = sorted(t for t, ok in presence.items() if ok is False)
        if missing:
            print(f"required images ABSENT from registry: {missing}", file=sys.stderr)
            return 1
    return 0


def _cmd_gpu_case(day_index: int) -> int:
    from npa.orchestration.npa_workflow.submit_matrix import rotating_gpu_submit_case

    case = rotating_gpu_submit_case(day_index)
    if case is None:
        return 0
    # Second field: which live test can drive this twin. Specs with a parallel
    # group or a decision-driven loop only run under the runtime orchestrator;
    # asking the one-shot test for them collects zero tests.
    sys.stdout.write(f"{case.spec}\t{'runtime' if case.runtime else 'one-shot'}\n")
    return 0


def _cmd_plan_set(day_index: int, null_sep: bool) -> int:
    specs = dc.daily_plan_set(day_index)
    sep = "\0" if null_sep else "\n"
    sys.stdout.write(sep.join(str(s.path) for s in specs))
    if specs and not null_sep:
        sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="print coverage report")
    p_report.add_argument("--json", action="store_true")

    sub.add_parser("check", help="assert required images keep >= 4-step coverage")

    p_plan = sub.add_parser("plan-set", help="print today's specs to plan-spec")
    p_plan.add_argument("--day-index", type=int, default=_day_index())
    p_plan.add_argument("--print0", action="store_true", help="NUL-separate paths")

    p_gpu = sub.add_parser("gpu-case", help="print today's rotating real-GPU workflow spec")
    p_gpu.add_argument("--day-index", type=int, default=_day_index())

    p_images = sub.add_parser("images", help="resolve + optionally inspect every workbench image")
    p_images.add_argument("--registry", default=None)
    p_images.add_argument("--inspect", action="store_true", help="check registry presence")
    p_images.add_argument("--require", action="store_true", help="exit 1 if an image is absent")
    p_images.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "report":
        return _cmd_report(args.json)
    if args.command == "check":
        return _cmd_check()
    if args.command == "plan-set":
        return _cmd_plan_set(args.day_index, args.print0)
    if args.command == "gpu-case":
        return _cmd_gpu_case(args.day_index)
    if args.command == "images":
        return _cmd_images(args.registry, args.inspect, args.require, args.json)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

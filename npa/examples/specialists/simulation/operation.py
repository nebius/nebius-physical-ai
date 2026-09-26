"""Expose identical fixed validation and simulation commands to both experiment arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from workload import TASKS, _validate


def _simulation_receipt(report):
    cases = [
        {
            key: row[key]
            for key in ("mass_scale", "friction_scale", "accepted", "final_distance_m")
        }
        for row in report["episodes"]
    ]
    return {
        "task": report["task"],
        "accepted": report["accepted"],
        "total": report["total"],
        "cases": cases,
        "evidence": "Full videos and physics traces are bound by latest.json",
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validate", "simulate"])
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    try:
        _validate(Path.cwd(), arguments.task)
        if arguments.operation == "validate":
            result = {"status": "valid", "parameter_cases": 6}
        else:
            from physics import _simulate

            report = _simulate(Path.cwd(), arguments.task, arguments.seed)
            result = _simulation_receipt(report)
        print(json.dumps(result))
        return 0 if result.get("accepted", 6) == 6 else 1
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(_main())

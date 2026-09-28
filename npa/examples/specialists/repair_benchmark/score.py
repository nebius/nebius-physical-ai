"""Export all matched outcomes and test the frozen quality, latency and total-cost claim."""

import argparse
import json
from pathlib import Path
from statistics import median

from accounting import _account
from operation import _digest


def _read(path):
    return json.loads(path.read_text())


def _native(value):
    return {
        key: value[key]
        for key in ("status", "source_sha256", "verification", "seconds")
        if key in value
    }


def _arm(root, pair, arm, prices):
    directory = root / "pairs" / f"pair-{pair}" / arm
    path = directory / "outcome.json"
    if not path.exists():
        return {
            "pair": pair,
            "arm": arm,
            "status": "unfinished",
            "cost": {"complete": False},
        }
    outcome = _read(path)
    report = {
        key: outcome[key]
        for key in ("status", "agent_tool_seconds", "end_to_end_seconds")
    }
    report.update(
        pair=pair,
        arm=arm,
        outcome_sha256=_digest(path),
        lanes={name: _native(value) for name, value in outcome["lanes"].items()},
        combined=_native(outcome["combined"]),
        cost=_account(root, directory, prices),
    )
    report["source_patches"] = _patches(root, directory)
    report["all_operation_attempts"] = _attempts(directory)
    report["routing_decisions"] = _routes(directory)
    report["host_load"] = {
        phase: outcome[phase] for phase in ("host_before", "host_after")
    }
    return report


def _routes(directory):
    path = directory / "run/usage.json"
    if not path.exists():
        return []
    fields = (
        "provider",
        "model",
        "status",
        "api_call_attempted",
        "usage",
        "model_verified",
        "prompt_revision",
        "selected_model",
        "effective_model",
        "fallback",
        "latency_seconds",
    )
    return [
        {name: row[name] for name in fields if name in row}
        for row in _read(path).get("router", {}).get("responses", [])
    ]


def _attempts(directory):
    result = {}
    states = sorted((directory / "operations").glob("*"))
    states.append(directory / "combined/operations")
    for state in states:
        lane = "combined" if state.name == "operations" else state.name
        rows = []
        for path in sorted(state.glob("*/result.json")):
            value = _read(path)
            row = {
                "kind": path.parent.name.split("-", 1)[0],
                "receipt_sha256": _digest(path),
                **_native(value),
            }
            for name in ("returncode", "workload_returncode", "regression_checks"):
                if name in value:
                    row[name] = value[name]
            rows.append(row)
        result[lane] = rows
    return result


def _patches(root, directory):
    import difflib

    frozen = _read(root / "freeze.json")
    result = []
    for config_path in sorted((directory / "configs").glob("*.json")):
        config = _read(config_path)
        for target in config["targets"]:
            final = Path(config["workspace"]) / "npa/src" / target
            initial_hash = frozen["inputs"][str(final.relative_to(root))]
            # Calibration retains the immutable historical source before model work.
            initial = sorted(
                (root / "calibration" / config_path.stem).glob(
                    "check-*/source/" + target
                )
            )[0]
            if _digest(initial) != initial_hash:
                raise ValueError(
                    "first diagnosis does not bind the frozen historical source"
                )
            result.append(
                {
                    "path": "npa/src/" + target,
                    "initial_sha256": initial_hash,
                    "final_sha256": _digest(final),
                    "diff": "".join(
                        difflib.unified_diff(
                            initial.read_text().splitlines(True),
                            final.read_text().splitlines(True),
                            fromfile="a/" + target,
                            tofile="b/" + target,
                        )
                    ),
                }
            )
    return result


def _comparison(arms):
    grouped = {
        name: [row for row in arms if row["arm"] == name]
        for name in ("astra-only", "astra-tofa")
    }
    summary = {
        name: {
            "passed": sum(row["status"] == "passed" for row in rows),
            "total": len(rows),
        }
        for name, rows in grouped.items()
    }
    expected_pairs = all(
        {row["pair"] for row in rows} == {1, 2, 3} and len(rows) == 3
        for rows in grouped.values()
    )
    eligible = expected_pairs and all(
        row["status"] == "passed" and row["cost"]["complete"] for row in arms
    )
    result = {
        "quality": summary,
        "comparison_eligible": eligible,
        "claim_supported": False,
    }
    if not eligible:
        result["reason"] = (
            "All declared arms must complete with reconciled costs before claiming a win."
        )
        return result
    medians = {name: _medians(rows) for name, rows in grouped.items()}
    return {**result, **_efficiency(medians)}


def _medians(rows):
    return {
        "seconds": median(row["end_to_end_seconds"] for row in rows),
        "cost_minimum_usd": median(row["cost"]["total_usd"]["minimum"] for row in rows),
        "cost_maximum_usd": median(row["cost"]["total_usd"]["maximum"] for row in rows),
    }


def _efficiency(medians):
    baseline, hybrid = medians["astra-only"], medians["astra-tofa"]
    speed = 1 - hybrid["seconds"] / baseline["seconds"]
    savings = 1 - hybrid["cost_maximum_usd"] / baseline["cost_minimum_usd"]
    return {
        "medians": medians,
        "elapsed_reduction_fraction": speed,
        "conservative_model_cost_reduction_fraction": savings,
        "claim_supported": speed > 0 and savings > 0,
        "interpretation": "Equal verified completion; compare median total latency and conservatively priced model usage.",
    }


def _score(root):
    frozen = _read(root / "freeze.json")
    execution = _read(root / "execution.json")
    arms = [
        _arm(root, number, arm, execution["prices"])
        for number, order in enumerate(frozen["protocol"]["pairs"], 1)
        for arm in order
    ]
    return {
        "schema": "npa.specialists.repair-benchmark.results.v1",
        "protocol": frozen["protocol"],
        "freeze_sha256": _digest(root / "freeze.json"),
        "common_prompt_sha256": frozen["common_prompt_sha256"],
        "runtime_commit": frozen["runtime_commit"],
        "instrumentation_wrapper_sha256": execution["capture"]["wrapper_sha256"],
        "calibration": _calibration(root),
        "price_snapshot": execution["prices"],
        "all_declared_arms": arms,
        "comparison": _comparison(arms),
        "benchmark_development_conversation_cost_included": False,
        "benchmark_development_conversation_cost_usd": None,
        "cloud_or_gpu_workload": False,
    }


def _calibration(root):
    path = root / "calibration/result.json"
    value = _read(path)
    return {
        "receipt_sha256": _digest(path),
        "status": value["status"],
        "reference": _native(value["reference"]),
        "historical": {name: _native(row) for name, row in value["historical"].items()},
        "native_reference": _native(value["native_reference"]),
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = _score(options.root.resolve())
    with options.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["comparison"]))


if __name__ == "__main__":
    _main()

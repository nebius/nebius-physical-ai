"""Score retained physics evidence and estimate token cost without hiding missing usage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from physics import _score
from workload import TASKS


def _astra_cost(usage, price):
    cached = usage["cached_input_tokens"]
    written = usage["cache_write_input_tokens"]
    uncached = usage["input_tokens"] - cached - written
    if min(cached, written, uncached, usage["output_tokens"]) < 0:
        raise ValueError("invalid provider usage counters")
    # Reasoning output is a subset of output_tokens, not an additional charge.
    return (
        uncached * price["input_price_per_million_tokens"]
        + cached * price["cached_input_price_per_million_tokens"]
        + written * price["cache_write_price_per_million_tokens"]
        + usage["output_tokens"] * price["output_price_per_million_tokens"]
    ) / 1e6


def _astra_usage(directory, prices):
    events = [
        json.loads(line)
        for line in (directory / "codex.jsonl").read_text().splitlines()
    ]
    turns = [event["usage"] for event in events if event["type"] == "turn.completed"]
    totals = {
        key: sum(turn[key] for turn in turns)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    invalid = [
        event
        for event in events
        if event.get("item", {}).get("type")
        in {"command_execution", "file_change", "collab_agent_tool_call"}
    ]
    if invalid:
        raise ValueError("baseline used tools outside the matched Workbench interface")
    return {
        "usage_complete": bool(turns),
        "usage": {"gpt-6-astra": totals},
        "recorded_cost_usd": _astra_cost(totals, prices["gpt-6-astra"])
        if turns
        else None,
    }


def _specialist_usage(receipts, prices):
    totals, complete = {}, True
    for task in receipts.values():
        previous = {}
        for event in task["events"]:
            if event["type"] == "needs_attention" and not (
                previous.get("type") == "model" and previous.get("accepted") is False
            ):
                complete = False
            previous = event
            if event["type"] != "model":
                continue
            usage = event["usage"]
            if not {"prompt_tokens", "completion_tokens"} <= usage.keys():
                complete = False
                continue
            model = totals.setdefault(event["model"], {})
            for key, value in usage.items():
                model[key] = model.get(key, 0) + value
    cost = sum(
        (
            usage["prompt_tokens"] * prices[model]["input_price_per_million_tokens"]
            + usage["completion_tokens"]
            * prices[model]["output_price_per_million_tokens"]
        )
        / 1e6
        for model, usage in totals.items()
    )
    return {
        "usage_complete": complete and bool(totals),
        "usage": totals,
        "recorded_cost_usd": cost,
    }


def _task_outcome(directory, name, receipt):
    try:
        results = [
            event["result"]
            for event in receipt["events"]
            if event["type"] == "tool"
            and event.get("result", {}).get("operation") == "simulate"
            and event["result"].get("returncode") == 0
        ]
        if not results:
            raise ValueError("no completed simulation in the frozen trial receipts")
        outcome = _score(directory, name)
        observed = json.loads(results[-1]["stdout"])
        if any(observed[key] != outcome[key] for key in ("accepted", "total")):
            raise ValueError("physics differs from the frozen trial receipt")
        return outcome
    except (OSError, ValueError, KeyError) as error:
        # Exact diagnostics remain in private receipts; public summaries need the class.
        return {
            "task": name,
            "accepted": 0,
            "total": 6,
            "replay_verified": False,
            "validation_error": type(error).__name__,
        }


def _score_run(directory, prices):
    started = time.perf_counter()
    execution = json.loads((directory / "execution.json").read_text())
    receipts = json.loads((directory / "task-receipts.json").read_text())
    usage = (
        _specialist_usage(receipts, prices)
        if execution["arm"] == "specialists"
        else _astra_usage(directory, prices)
    )
    if execution.get("infrastructure_failure"):
        usage["usage_complete"] = False
    tasks = [
        _task_outcome(directory / "workspaces" / name, name, receipts[name])
        for name in TASKS
    ]
    accepted = sum(task["accepted"] for task in tasks)
    cost = usage["recorded_cost_usd"] if usage["usage_complete"] else None
    return {
        "round": directory.parent.name,
        "arm": execution["arm"],
        "seed": execution["seed"],
        "infrastructure_failure": execution.get("infrastructure_failure"),
        "agent_tool_seconds": execution["agent_tool_seconds"],
        "accepted": accepted,
        "total": 36,
        "tasks": tasks,
        **usage,
        "estimated_cost_usd": cost,
        "estimated_cost_per_accepted_episode_usd": cost / accepted
        if cost is not None and accepted
        else None,
        "verification_seconds": time.perf_counter() - started,
    }


def _score_directory(root, prices):
    runs = []
    for path in sorted(root.glob("round-*/*/execution.json")):
        result = _score_run(path.parent, prices["models"])
        (path.parent / "score.json").write_text(json.dumps(result, indent=2) + "\n")
        runs.append(result)
        fields = (
            "round",
            "arm",
            "accepted",
            "total",
            "agent_tool_seconds",
            "estimated_cost_usd",
        )
        print(json.dumps({key: result[key] for key in fields}), flush=True)
    return runs


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    options = parser.parse_args()
    prices = json.loads((options.root / "prices.json").read_text())
    report = {
        "schema": "npa.specialists.simulation_score.v1",
        "prices": prices,
        "runs": _score_directory(options.root, prices),
        "cost_basis": "Standard API-equivalent token estimates; not invoices. Token Factory cache discount assumed zero.",
        "excluded_costs": ["setup and coding", "host compute", "human intervention"],
        "verification_outside_agent_timer": True,
    }
    (options.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    _main()

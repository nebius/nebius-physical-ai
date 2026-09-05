"""Functional CPU validation for the exact-source Sim2Real controller image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec


def main() -> int:
    """Load and execute both real control-flow branches of the canonical workflow."""

    source_sha = os.environ.get("NPA_IMAGE_SOURCE_SHA", "")
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise RuntimeError("NPA_IMAGE_SOURCE_SHA must be an exact lowercase 40-hex commit")

    workflow = Path(
        os.environ.get(
            "NPA_SIM2REAL_WORKFLOW",
            "/opt/npa/workflows/workbench/npa-workflows/sim2real.yaml",
        )
    )
    spec = load_spec(workflow)
    plans = {
        decision: build_plan(
            spec,
            run_id=f"golden-control-{decision}",
            assume_decision=decision,
        ).to_dict()
        for decision in ("promote_checkpoint", "loop_back")
    }
    states = set(spec.states)
    planned_states = {
        step["state"] for plan in plans.values() for step in plan["steps"]
    }
    state_prefixes = {state[:8] for state in planned_states}
    expected_prefixes = {f"stage-{index:02d}" for index in range(1, 15)}
    if state_prefixes != expected_prefixes:
        raise RuntimeError(
            f"controller did not execute the complete 14-stage graph: {sorted(planned_states)}"
        )
    if plans["promote_checkpoint"] == plans["loop_back"]:
        raise RuntimeError("controller decisions produced identical execution plans")

    report = {
        "schema": "npa.golden.sim2real-control.v1",
        "source_sha": source_sha,
        "workflow": spec.name,
        "declared_state_count": len(states),
        "stage_count": len(state_prefixes),
        "branch_step_counts": {
            decision: len(plan["steps"]) for decision, plan in plans.items()
        },
        "plan_sha256": {
            decision: hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for decision, plan in plans.items()
        },
        "status": "passed",
    }
    output_dir = Path(os.environ.get("NPA_SMOKE_OUTPUT_DIR", "/tmp/npa-golden"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "sim2real-control-functional.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

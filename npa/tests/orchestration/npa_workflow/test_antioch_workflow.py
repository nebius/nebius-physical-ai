"""Antioch workflow catalog identity contract coverage."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    REPO_ROOT
    / "npa/workflows/workbench/npa-workflows/antioch-offline-policy-train.yaml"
)


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_shipped_antioch_workflow_renders_real_cli_identity_flags() -> None:
    plan = build_plan(load_spec(SPEC_PATH), run_id="antioch-workflow-run")
    simulate = plan.steps[0]

    assert simulate.argv[:4] == ["npa", "workbench", "antioch", "run"]
    assert _option_value(simulate.argv, "--workflow-run") == "antioch-workflow-run"
    assert _option_value(simulate.argv, "--state-id") == "antioch-simulate"


def test_two_antioch_stages_can_render_distinct_state_identities(tmp_path: Path) -> None:
    document = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    simulate = document["states"]["antioch-simulate"]
    simulate["params"] = {"antioch_state_id": "collect-primary"}
    simulate["next"] = "antioch-simulate-secondary"
    document["states"]["antioch-simulate-secondary"] = {
        **simulate,
        "description": "Run a second independently idempotent Antioch collection.",
        "params": {"antioch_state_id": "collect-secondary"},
        "next": "train-offline-policy",
    }
    document["states"]["train-offline-policy"]["needs"] = [
        "antioch-simulate-secondary"
    ]
    path = tmp_path / "two-antioch-stages.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    plan = build_plan(load_spec(path), run_id="shared-workflow-run")
    antioch_steps = [step for step in plan.steps if step.tool_ref == "workbench.antioch.run"]

    assert len(antioch_steps) == 2
    assert {
        _option_value(step.argv, "--state-id") for step in antioch_steps
    } == {"collect-primary", "collect-secondary"}
    assert {
        _option_value(step.argv, "--workflow-run") for step in antioch_steps
    } == {"shared-workflow-run"}

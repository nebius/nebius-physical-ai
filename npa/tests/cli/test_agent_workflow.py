from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from npa.cli import agent as agent_module
from npa.cli.agent_chat import (
    apis_for_intent,
    build_grounded_reply,
    match_chat_intent,
)
from npa.cli.agent_workflow import (
    generate_sim2real_loop_gate_yaml,
    generate_sim2real_two_step_yaml,
    plan_workflow_yaml_text,
    validate_workflow_yaml_text,
)
from npa.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_YAML = REPO_ROOT / "npa/workflows/workbench/npa-workflows/sim2real-two-step-agent.yaml"
runner = CliRunner()


def test_generate_sim2real_two_step_yaml_validates() -> None:
    yaml_text = generate_sim2real_two_step_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True
    assert result["name"] == "sim2real-two-step"
    assert set(result["states"]) == {"augment", "envgen"}


def test_example_yaml_file_validate_spec_cli() -> None:
    if not EXAMPLE_YAML.is_file():
        EXAMPLE_YAML.write_text(generate_sim2real_two_step_yaml(), encoding="utf-8")
    assert EXAMPLE_YAML.is_file()
    result = runner.invoke(app, ["workbench", "workflow", "validate-spec", str(EXAMPLE_YAML), "--json"])
    assert result.exit_code == 0
    assert "sim2real-two-step" in result.output


def test_plan_two_step_workflow_has_two_steps() -> None:
    yaml_text = EXAMPLE_YAML.read_text(encoding="utf-8")
    plan = plan_workflow_yaml_text(yaml_text, run_id="unit-demo")
    assert plan["ok"] is True
    assert len(plan["steps"]) == 2
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.cosmos2.transfer" in tool_refs
    assert "workbench.sim2real_envgen.raw_shard" in tool_refs


def test_generate_sim2real_loop_gate_yaml_validates() -> None:
    yaml_text = generate_sim2real_loop_gate_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True
    assert result["name"] == "sim2real-loop-gate-agent"
    assert {"augment", "refine", "vlm-critique", "quality-gate", "publish"}.issubset(set(result["states"]))


def test_plan_loop_gate_workflow_respects_assume_decision() -> None:
    yaml_text = generate_sim2real_loop_gate_yaml()
    plan = plan_workflow_yaml_text(yaml_text, run_id="loop-demo", assume_decision="promote_checkpoint")
    assert plan["ok"] is True
    step_states = [str(step.get("state")) for step in plan["steps"]]
    assert "quality-gate" in step_states
    assert "publish" in step_states


def test_match_create_workflow_intent() -> None:
    assert match_chat_intent("create a 2-step sim2real workflow") == "create_workflow"
    assert match_chat_intent("generate npa.workflow YAML for sim2real") == "create_workflow"


def test_create_workflow_grounded_reply_includes_yaml_fence() -> None:
    state: dict = {}
    reply = build_grounded_reply("create_workflow", state, ["workbench.cosmos2.transfer"])
    assert "```yaml" in reply
    assert "sim2real-two-step" in reply
    assert "augment" in reply
    assert "GET /api" not in reply


def test_create_workflow_apis() -> None:
    apis = apis_for_intent("create_workflow")
    assert any(path.endswith("draft") for path in apis)
    assert any("validate" in path for path in apis)


def test_bootstrap_embeds_workflow_endpoints() -> None:
    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.post("/workflows/validate")' in source
    assert '@app.post("/workflows/plan")' in source
    assert '@app.post("/workflows/submit")' in source
    assert '@app.get("/workflows/draft")' in source
    assert "workflowYaml" in source
    assert "validateWorkflowYaml" in source
    assert "generate_sim2real_two_step_yaml" in source
    assert "generate_sim2real_loop_gate_yaml" in source
    assert '@app.get("/sim-viz/runs")' in source
    assert '@app.post("/sim-viz/select-run")' in source
    assert "sim_viz_runs" in source
    embedded = agent_module._embedded_agent_workflow_source()
    assert "validate_workflow_yaml_text" in embedded


def test_lightweight_validation_without_tool_refs_still_parses() -> None:
    yaml_text = generate_sim2real_two_step_yaml()
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True


def test_lightweight_validation_handles_complex_edges(monkeypatch) -> None:
    def _import_fail(_yaml_text: str) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: complex-agent-graph
initial: start
states:
  start:
    toolRef: workbench.cosmos2.transfer
    next: gate
  gate:
    transitions:
      promote_checkpoint: train
      loop_back: start
  train:
    sequence:
      - state: eval
      - state: done
  eval:
    toolRef: workbench.sim2real_envgen.raw_shard
    terminal: true
  done:
    terminal: true
"""
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True
    assert result["name"] == "complex-agent-graph"


def test_lightweight_plan_walks_reachable_graph(monkeypatch) -> None:
    def _import_fail(*_args, **_kwargs) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._plan_with_npa", _import_fail)
    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: branching-agent-plan
initial: root
states:
  root:
    next: branch
  branch:
    transitions:
      promote_checkpoint: deploy
      loop_back: recover
  deploy:
    terminal: true
  recover:
    terminal: true
"""
    plan = plan_workflow_yaml_text(yaml_text, run_id="agent-branch-demo", tool_refs=frozenset())
    assert plan["ok"] is True
    states = [step["state"] for step in plan["steps"]]
    assert states == ["root", "branch", "deploy", "recover"]


def test_lightweight_validation_accepts_transition_list_goto(monkeypatch) -> None:
    def _import_fail(*_args, **_kwargs) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: goto-list-graph
initial: start
states:
  start:
    transitions:
      - when: promote_checkpoint
        goto: publish
      - when: loop_back
        goto: retry
  retry:
    next: publish
  publish:
    terminal: true
"""
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True
    assert result["name"] == "goto-list-graph"

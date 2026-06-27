from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli import agent as agent_module
from npa.cli.agent_chat import (
    apis_for_intent,
    build_grounded_reply,
    match_chat_intent,
)
from npa.cli.agent_workflow import (
    choose_workflow_template,
    generate_gpu_cross_region_yaml,
    generate_sim2real_loop_gate_yaml,
    generate_sim2real_two_step_yaml,
    generate_token_factory_gate_yaml,
    generate_vlm_rl_loop_yaml,
    generate_workflow_draft,
    generate_workflow_yaml,
    plan_workflow_yaml_text,
    validate_workflow_yaml_text,
)
from npa.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_YAML = REPO_ROOT / "npa/workflows/workbench/npa-workflows/sim2real-two-step-agent.yaml"

_GOLDEN_YAMLS = [
    "sim2real-gpu-cross-region-agent.yaml",
    "sim2real-two-step-agent.yaml",
    "sim2real-two-step.yaml",
    "sim2real-vlm-rl.yaml",
    "tokenfactory-cosmos-gate.yaml",
    "tokenfactory-rollout-judge.yaml",
    "vlm-eval-single.yaml",
    "bdd100k-pipeline.yaml",
]

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
    assert "generate_workflow_draft" in source
    assert '@app.get("/sim-viz/runs")' in source
    assert '@app.post("/sim-viz/select-run")' in source
    assert "sim_viz_runs" in source
    embedded = agent_module._embedded_agent_workflow_source()
    assert "validate_workflow_yaml_text" in embedded
    assert "Could not generate runnable workflow YAML yet" in source
    assert "chat returns YAML only after both validation and planning succeed" in source


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


# --- Complex YAML generator tests ---


def test_generate_vlm_rl_loop_yaml_validates() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"vlm-rl validate failed: {result.get('error')}"
    assert result["name"] == "sim2real-vlm-rl"
    states = set(result["states"])
    assert "augment" in states
    assert "envgen" in states
    assert "finalize" in states


def test_generate_vlm_rl_loop_yaml_plan_has_multiple_steps() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    plan = plan_workflow_yaml_text(yaml_text, run_id="vlm-rl-test", assume_decision="promote_checkpoint")
    assert plan["ok"] is True, f"vlm-rl plan failed: {plan.get('error')}"
    assert len(plan["steps"]) >= 3
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.cosmos2.transfer" in tool_refs
    assert "workbench.sim2real_envgen.raw_shard" in tool_refs


def test_generate_vlm_rl_loop_yaml_contains_loop_and_gate() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    assert "loop:" in yaml_text
    assert "transitions:" in yaml_text
    assert "promote_checkpoint" in yaml_text
    assert "loop_back" in yaml_text
    assert "writesDecision: true" in yaml_text


def test_generate_token_factory_gate_yaml_validates() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"token-factory validate failed: {result.get('error')}"
    assert result["name"] == "tokenfactory-cosmos-gate"
    states = set(result["states"])
    assert "reason-scene" in states
    assert "augment-scene" in states
    assert "publish" in states


def test_generate_token_factory_gate_yaml_plan() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    plan = plan_workflow_yaml_text(yaml_text, run_id="gate-test", assume_decision="promote_checkpoint")
    assert plan["ok"] is True, f"token-factory plan failed: {plan.get('error')}"
    assert len(plan["steps"]) >= 2
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.cosmos2.transfer" in tool_refs


def test_generate_token_factory_gate_yaml_contains_vlm_gate() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    assert "loop:" in yaml_text
    assert "transitions:" in yaml_text
    assert "promote_checkpoint" in yaml_text
    assert "vlm-critique" in yaml_text
    assert "quality-gate" in yaml_text


def test_generate_gpu_cross_region_yaml_validates() -> None:
    yaml_text = generate_gpu_cross_region_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"gpu-cross-region validate failed: {result.get('error')}"
    assert result["name"] == "sim2real-gpu-cross-region"
    states = set(result["states"])
    assert "primary-rollout" in states
    assert "transform-rollouts" in states
    assert "secondary-eval" in states
    assert "summarize-improvement" in states
    assert "finalize" in states


def test_generate_gpu_cross_region_yaml_includes_multi_region_resources() -> None:
    yaml_text = generate_gpu_cross_region_yaml()
    assert "gpu-primary:" in yaml_text
    assert "gpu-secondary:" in yaml_text
    assert "container-glue:" in yaml_text
    assert "project_primary" in yaml_text
    assert "project_secondary" in yaml_text
    assert "region_primary" in yaml_text
    assert "region_secondary" in yaml_text
    assert "transform-rollouts" in yaml_text
    assert "summarize-improvement" in yaml_text
    assert "workbench.data_transform.rollout_contract" in yaml_text
    assert "workbench.data_transform.improvement_summary" in yaml_text
    assert "rollout_source_schema" not in yaml_text
    assert "rollout_target_schema" not in yaml_text


def test_generate_gpu_cross_region_yaml_contract_edges_align() -> None:
    spec = yaml.safe_load(generate_gpu_cross_region_yaml())
    states = spec["states"]
    primary_out_schema = states["primary-rollout"]["outputs"][0]["schema"]
    transform_in_schema = states["transform-rollouts"]["inputs"][0]["schema"]
    transform_out_schema = states["transform-rollouts"]["outputs"][0]["schema"]
    secondary_in_schema = states["secondary-eval"]["inputs"][0]["schema"]

    assert primary_out_schema == transform_in_schema
    assert transform_out_schema == secondary_in_schema


def test_generate_gpu_cross_region_yaml_plan() -> None:
    yaml_text = generate_gpu_cross_region_yaml()
    plan = plan_workflow_yaml_text(yaml_text, run_id="gpu-cross-region-test")
    assert plan["ok"] is True, f"gpu-cross-region plan failed: {plan.get('error')}"
    states = [step["state"] for step in plan["steps"]]
    assert states == [
        "primary-rollout",
        "transform-rollouts",
        "secondary-eval",
        "summarize-improvement",
        "finalize",
    ]


def test_generate_workflow_yaml_dispatcher() -> None:
    two_step = generate_workflow_yaml("two-step")
    assert "sim2real-two-step" in two_step
    vlm_rl = generate_workflow_yaml("vlm-rl-loop")
    assert "sim2real-vlm-rl" in vlm_rl
    gate = generate_workflow_yaml("token-factory-gate")
    assert "tokenfactory-cosmos-gate" in gate
    loop_gate = generate_workflow_yaml("loop-gate")
    assert "sim2real-loop-gate-agent" in loop_gate
    cross_region = generate_workflow_yaml("gpu-cross-region")
    assert "sim2real-gpu-cross-region" in cross_region
    default = generate_workflow_yaml("unknown-template")
    assert "sim2real-two-step" in default


def test_choose_workflow_template_by_intent_and_text() -> None:
    selected = choose_workflow_template(
        user_text="create a multi-step outer loop with inner loop gate",
        intent="create_workflow",
    )
    assert selected["template"] == "vlm-rl-loop"
    selected_gate = choose_workflow_template(
        user_text="build tokenfactory quality gate workflow",
        intent="create_workflow",
    )
    assert selected_gate["template"] == "token-factory-gate"
    selected_multi_region = choose_workflow_template(
        user_text="create gpu workflow across two regions for one tenant",
        intent="create_workflow",
    )
    assert selected_multi_region["template"] == "gpu-cross-region"


def test_generate_workflow_draft_returns_selection_and_valid_yaml() -> None:
    draft = generate_workflow_draft(
        user_text="draft a tokenfactory gate workflow",
        intent="create_gate_workflow",
        tool_refs=frozenset(),
    )
    assert draft["template"] == "token-factory-gate"
    assert draft["validation"]["ok"] is True
    assert draft["plan"]["ok"] is True
    assert draft["runnable"] is True
    assert "metadata:" in draft["yaml"]
    assert "\n\n  scene_uri:" in draft["yaml"]


def test_generate_workflow_draft_sets_not_runnable_when_plan_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent_workflow.plan_workflow_yaml_text",
        lambda *_args, **_kwargs: {"ok": False, "error": "forced plan failure"},
    )
    draft = generate_workflow_draft(template="two-step", tool_refs=frozenset())
    assert draft["validation"]["ok"] is True
    assert draft["plan"]["ok"] is False
    assert draft["runnable"] is False


def test_generate_workflow_yaml_aliases() -> None:
    assert "sim2real-vlm-rl" in generate_workflow_yaml("vlm-rl")
    assert "sim2real-vlm-rl" in generate_workflow_yaml("vlm_rl_loop")
    assert "tokenfactory-cosmos-gate" in generate_workflow_yaml("gate")
    assert "tokenfactory-cosmos-gate" in generate_workflow_yaml("tokenfactory")
    assert "sim2real-loop-gate-agent" in generate_workflow_yaml("loop")


@pytest.mark.parametrize("yaml_name", _GOLDEN_YAMLS)
def test_golden_yaml_validates(yaml_name: str) -> None:
    """All golden NPA workflow YAMLs in the repo should parse and validate."""
    yaml_path = REPO_ROOT / "npa/workflows/workbench/npa-workflows" / yaml_name
    if not yaml_path.is_file():
        pytest.skip(f"golden YAML not found: {yaml_name}")
    yaml_text = yaml_path.read_text(encoding="utf-8")
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"{yaml_name} failed: {result.get('error')}"


@pytest.mark.parametrize("yaml_name", _GOLDEN_YAMLS)
def test_golden_yaml_plan_spec_cli(yaml_name: str) -> None:
    """Golden YAMLs should plan successfully with the CLI."""
    yaml_path = REPO_ROOT / "npa/workflows/workbench/npa-workflows" / yaml_name
    if not yaml_path.is_file():
        pytest.skip(f"golden YAML not found: {yaml_name}")
    result = runner.invoke(
        app,
        ["workbench", "workflow", "plan-spec", str(yaml_path), "--run-id", "golden-test",
         "--assume-decision", "promote_checkpoint", "--json"],
    )
    assert result.exit_code == 0, f"{yaml_name} plan-spec CLI failed:\n{result.output}"
    assert "golden-test" in result.output or "steps" in result.output


# --- Complex workflow intent routing tests ---


def test_match_vlm_rl_workflow_intent() -> None:
    assert match_chat_intent("create a VLM-RL loop workflow") == "create_vlm_rl_workflow"
    assert match_chat_intent("generate a sim2real vlm rl pipeline") == "create_vlm_rl_workflow"
    assert match_chat_intent("build a workflow with outer loop and inner loop gate") == "create_vlm_rl_workflow"


def test_match_gate_workflow_intent() -> None:
    assert match_chat_intent("create a token factory gate workflow") == "create_gate_workflow"
    assert match_chat_intent("generate a quality gate cosmos augment loop") == "create_gate_workflow"
    assert match_chat_intent("build a tokenfactory cosmos-gate spec") == "create_gate_workflow"


def test_create_vlm_rl_workflow_grounded_reply() -> None:
    state: dict = {}
    reply = build_grounded_reply("create_vlm_rl_workflow", state, [])
    assert "```yaml" in reply
    assert "sim2real-vlm-rl" in reply
    assert "VLM-RL" in reply
    assert "GET /api" not in reply


def test_create_gate_workflow_grounded_reply() -> None:
    state: dict = {}
    reply = build_grounded_reply("create_gate_workflow", state, [])
    assert "```yaml" in reply
    assert "tokenfactory-cosmos-gate" in reply
    assert "Token Factory" in reply
    assert "GET /api" not in reply


def test_vlm_rl_workflow_apis_include_plan() -> None:
    apis = apis_for_intent("create_vlm_rl_workflow")
    assert any("validate" in p for p in apis)
    assert any("plan" in p for p in apis)


def test_gate_workflow_apis_include_plan() -> None:
    apis = apis_for_intent("create_gate_workflow")
    assert any("validate" in p for p in apis)
    assert any("plan" in p for p in apis)

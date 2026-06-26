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
    embedded = agent_module._embedded_agent_workflow_source()
    assert "validate_workflow_yaml_text" in embedded


def test_lightweight_validation_without_tool_refs_still_parses() -> None:
    yaml_text = generate_sim2real_two_step_yaml()
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True

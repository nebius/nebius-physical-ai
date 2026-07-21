"""Capability parity checks: chat intents vs CLI/YAML tool surface."""

from __future__ import annotations

from npa.cli.agent_chat import (
    INTENT_APIS,
    apis_for_intent,
    build_grounded_reply,
    format_list_recordings,
    format_tools_catalog,
    format_workflow_execute_guidance,
    match_chat_intent,
)
from npa.cli.agent_workflow import choose_workflow_template, format_workflow_chat_reply
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG


SAMPLE_TOOL_REFS = sorted(TOOL_CATALOG.keys())


def test_chat_intents_cover_core_cli_yaml_categories() -> None:
    expected = {
        "create_workflow": "draft a simple 2-step sim2real workflow yaml",
        "create_vlm_rl_workflow": "draft a vlm rl outer loop workflow yaml",
        "create_gate_workflow": "create a token factory quality gate workflow",
        "create_loop_gate_workflow": "create a sim2real loop-gate workflow yaml",
        "create_rl_policy_workflow": "draft an rl policy training workflow yaml",
        "tools_catalog": "show the workbench toolRef catalog",
        "workflow_execute_guidance": "how do I execute an npa.workflow with run-spec --execute",
        "list_recordings": "list available recordings and run history",
        "sonic_capabilities": "what can sonic do",
        "lerobot_capabilities": "lerobot capabilities",
        "groot_capabilities": "what does groot support",
        "genesis_capabilities": "genesis capabilities",
        "mjlab_capabilities": "mjlab capabilities",
        "isaac_lab_capabilities": "what can isaac lab do",
        "onboard_solution": "containerize a github repo and onboard it into the workbench",
    }
    for intent, prompt in expected.items():
        assert match_chat_intent(prompt) == intent, prompt


def test_intent_apis_document_start_sim2real_and_plan() -> None:
    assert "workflows/sim2real/submit" in apis_for_intent("start_sim2real")
    assert "workflows/plan" in apis_for_intent("create_workflow")
    assert "sim-viz/runs" in apis_for_intent("list_recordings")
    assert INTENT_APIS["cosmos3"] == ["tools"]


def test_tools_catalog_reply_covers_full_catalog_families() -> None:
    reply = format_tools_catalog(SAMPLE_TOOL_REFS, sample_size=12)
    assert f"{len(SAMPLE_TOOL_REFS)} toolRefs" in reply
    assert "scheduler plan only" in reply.lower() or "plan-only" in reply.lower()
    assert "run-spec" in reply
    assert "Families:" in reply
    assert "sim2real" in reply.lower() or "token_factory" in reply.lower() or "cosmos" in reply.lower()


def test_workflow_execute_guidance_matrix_mentions_plan_only_gap() -> None:
    reply = format_workflow_execute_guidance()
    assert "plan-only" in reply.lower()
    assert "run-spec" in reply
    assert "--execute" in reply
    assert "Agent chat / UI" in reply


def test_list_recordings_formats_live_state() -> None:
    state = {
        "sim_viz": {
            "run_id": "franka-demo",
            "active_run_id": "franka-demo",
            "available_run_ids": ["franka-demo", "customer-run"],
        },
        "sim_viz_runs": [
            {"run_id": "franka-demo", "stage": "demo", "camera": "workspace"},
            {"run_id": "customer-run", "stage": "completed", "camera": "heldout-sim"},
        ],
        "sim_viz_recordings": [{"name": "sim2real.rrd"}],
    }
    reply = format_list_recordings(state)
    assert "franka-demo" in reply
    assert "customer-run" in reply
    assert "sim2real.rrd" in reply
    assert "active" in reply.lower()


def test_workbench_family_capability_replies_ground_on_catalog() -> None:
    state = {"sim_viz": {}, "selection": {}, "latest_submit": {}}
    sonic = build_grounded_reply("sonic_capabilities", state, SAMPLE_TOOL_REFS)
    assert "SONIC capabilities" in sonic
    assert "workbench.sonic" in sonic or "Catalog matches" in sonic
    lerobot = build_grounded_reply("lerobot_capabilities", state, SAMPLE_TOOL_REFS)
    assert "LeRobot capabilities" in lerobot
    isaac = build_grounded_reply("isaac_lab_capabilities", state, SAMPLE_TOOL_REFS)
    assert "Isaac Lab capabilities" in isaac


def test_new_workflow_intents_select_expected_templates() -> None:
    loop = choose_workflow_template(user_text="loop gate", intent="create_loop_gate_workflow")
    assert loop["template"] == "loop-gate"
    rl = choose_workflow_template(user_text="rl policy", intent="create_rl_policy_workflow")
    assert rl["template"] == "rl-policy-success"


def test_workflow_chat_reply_states_plan_only_submit() -> None:
    reply = format_workflow_chat_reply(
        "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: demo\n",
        {"ok": True, "status": "valid", "name": "demo", "states": ["a"]},
        template="two-step",
        plan={"ok": True, "steps": [{"state": "a", "tool_ref": "workbench.sim2real.status"}]},
        runnable=True,
    )
    assert "plan-only" in reply.lower()
    assert "run-spec" in reply


def test_onboard_reply_does_not_claim_chat_executes_end_to_end() -> None:
    reply = build_grounded_reply(
        "onboard_solution",
        {"sim_viz": {}, "selection": {}, "latest_submit": {}},
        SAMPLE_TOOL_REFS,
    )
    assert "end-to-end" not in reply.lower()
    assert "does not execute" in reply.lower() or "operator CLI" in reply
    assert "run_byof_repo.py" in reply

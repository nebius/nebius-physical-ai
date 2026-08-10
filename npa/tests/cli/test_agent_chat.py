from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from npa.cli import agent as agent_module
from npa.cli import agent_actions
from npa.cli.agent_chat import (
    build_grounded_reply,
    format_sim2real_status,
    match_chat_intent,
)


def _planner(script):
    """A deterministic model_call yielding scripted planner decisions (0 tokens)."""
    calls = {"n": 0}

    def _call(messages, *, tier="cheap"):
        obj = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return {"choices": [{"message": {"content": json.dumps(obj)}}], "usage": {"total_tokens": 0}}

    return _call


def test_match_sim2real_status_intent() -> None:
    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("What's the workflow status?") == "sim2real_status"
    assert match_chat_intent("create a 2-step sim2real workflow") == "create_workflow"
    assert match_chat_intent("create a gpu workflow across 2 different regions") == "create_workflow"
    assert match_chat_intent("generate an example simple workflow YAML") == "create_workflow"
    assert match_chat_intent("start the sim2real pipeline") == "start_sim2real"
    assert match_chat_intent("run actual Sim2Real now") == "start_sim2real"
    assert match_chat_intent("what sim2real run should I view?") == "find_artifacts"
    assert match_chat_intent("which run should I load") == "find_artifacts"
    assert match_chat_intent("watch the sim") == "watch_sim"
    assert match_chat_intent("track the rerun timeline") == "watch_sim"
    assert match_chat_intent("keep me posted with live updates on the sim run") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("retry blob iframe until ready") == "watch_sim"
    assert match_chat_intent("watch sim and refresh when rrd lands") == "watch_sim"
    assert match_chat_intent("watch rerun blob+iframe until success") == "watch_sim"
    assert match_chat_intent("wait until both blob and iframe are SUCCESS") == "watch_sim"
    assert match_chat_intent("watch rerun blob iframe until consecutive success") == "watch_sim"
    assert match_chat_intent("keep rerun blob iframe green before finishing") == "watch_sim"
    assert match_chat_intent("mark rerun blob iframe passed before finishing") == "watch_sim"
    assert match_chat_intent("rerun blob-iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("rerun: blob/iframe; wait -> SUCCESS") == "watch_sim"
    assert match_chat_intent("keep rerun blob iframe healthy before finishing") == "watch_sim"
    assert match_chat_intent("Rerun blob iframe until SUCCESS. Branch feat/npa-agent. Bootstrap rtxpro/agent.") == "watch_sim"
    assert (
        match_chat_intent(
            "Enhance NPA agent chat intent routing and Rerun blob iframe until SUCCESS. Branch feat/npa-agent. Bootstrap rtxpro/agent after changes."
        )
        == "watch_sim"
    )
    assert match_chat_intent("watch until RERUN_BLOB_SUCCESS and RERUN_MOUNT_SUCCESS") == "watch_sim"
    assert match_chat_intent("load franka then rerun blob iframe until SUCCESS") == "watch_sim"
    assert (
        match_chat_intent("add an open source repo, containerize, push to registry, and run LeIsaac")
        == "onboard_solution"
    )
    assert (
        match_chat_intent("onboard a new workbench solution from a github repo with container and sky smoke")
        == "onboard_solution"
    )
    assert (
        match_chat_intent(
            "onboard https://github.com/githubtraining/hellogitworld.git on Ubuntu, "
            "build the container, push to registry, and run a deploy smoke on live infra"
        )
        == "onboard_solution"
    )
    assert match_chat_intent("what artifacts can I view?") == "find_artifacts"
    assert match_chat_intent("create a LeIsaac BYOF Isaac Lab workflow for live infra") == "create_workflow"
    assert match_chat_intent("camera angle inspector with top-down frustum preview") == "cameras"
    assert match_chat_intent("select scene robot props and cameras before submit") == "sim_assets"
    assert match_chat_intent("what does cosmos support for finetuning") == "cosmos_capabilities"
    assert match_chat_intent("what does lancedb expose") == "lancedb_capabilities"
    assert match_chat_intent("run on live infra in tmux loop with gpu compatibility checks") == "live_infra_loop"
    assert match_chat_intent("show my tenant resources") == "tenant_resources"
    assert match_chat_intent("what resources can I access in this project?") == "tenant_resources"


def test_public_chat_session_payload_never_exposes_memory_locator() -> None:
    from npa.cli.agent_chat import public_chat_session_payload

    payload = public_chat_session_payload(
        {
            "id": "session-a",
            "title": "Run analysis",
            "chat_history": [{"role": "user", "content": "hello"}],
            "memory_uri": "s3://private-bucket/private-tenant/session-a.json",
        }
    )

    assert payload["memory_persisted"] is True
    assert payload["message_count"] == 1
    assert "memory_uri" not in payload
    assert "private-bucket" not in json.dumps(payload)


def test_catalog_composition_requires_semantic_or_tool_specific_goal() -> None:
    from npa.cli.agent_chat import goal_requests_catalog_composition

    assert not goal_requests_catalog_composition("create 2-step sim2real workflow")
    assert not goal_requests_catalog_composition("generate an example simple workflow YAML")
    assert goal_requests_catalog_composition("write YAML using the cosmos tool")
    assert goal_requests_catalog_composition("curate a dataset -> train -> evaluate")
    assert goal_requests_catalog_composition(
        "compose workbench.dataset.curate and workbench.rl.policy_train"
    )


def test_match_complex_non_stock_artifact_queries() -> None:
    assert (
        match_chat_intent(
            "For the non-stock customer Sim2Real run, discover what outputs I can view, "
            "load the run-specific Rerun recording, then show video/json/log artifacts."
        )
        == "find_artifacts"
    )
    assert (
        match_chat_intent(
            "Which customer run should I use if I need the non stock .rrd plus rollout video and report artifacts?"
        )
        == "find_artifacts"
    )


def test_match_complex_workflow_yaml_queries() -> None:
    assert (
        match_chat_intent(
            "Draft a VLM/RL outer-loop workflow YAML for non-stock assets with a Token Factory quality gate, "
            "promote_checkpoint transition, and loop_back transition."
        )
        == "create_vlm_rl_workflow"
    )
    assert (
        match_chat_intent(
            "Create a workflow yaml that runs policy rollout, heldout eval, and a VLM critic gate before finalizing."
        )
        == "create_vlm_rl_workflow"
    )


def test_match_watch_sim_intent_with_long_requirements_addendum() -> None:
    prompt = """
Enhance NPA agent chat intent routing and Rerun blob iframe until SUCCESS. Branch feat/npa-agent. Bootstrap rtxpro/agent.

--- REQUIREMENTS ADDENDUM (read and apply) ---

Simulation visualization: keep /rerun/ iframe primary, poll /api/sim-viz/status, and continue until both blob and iframe mount report SUCCESS.
Camera inspector: list cameras and frustum preview.
Sim assets panel: selection, catalog, and submit path.
verify-live gates: include sim_viz_url and cameras API checks.
"""
    assert match_chat_intent(prompt) == "watch_sim"


def test_format_sim2real_status_includes_run_id_and_stage() -> None:
    state = {
        "sim_viz": {
            "run_id": "agent-run-deadbeef",
            "stage": "demo",
            "camera": "workspace",
            "rerun_ready": True,
            "rrd_updated_at": "2026-06-25T00:00:00+00:00",
            "rerun_iframe_url": "/rerun/?url=/api/sim-viz/rrd&camera=workspace",
        },
        "latest_submit": {"run_id": "agent-run-deadbeef", "submitted_at": "2026-06-25T00:00:00+00:00"},
        "selection": {"robot_preset": "franka", "sim_backend": "isaac"},
    }
    reply = format_sim2real_status(state, rerun_ready=True)
    assert "run_id" in reply
    assert "agent-run-deadbeef" in reply
    assert "stage" in reply
    assert "demo" in reply
    assert "rerun_iframe_url" in reply
    assert "/rerun/" in reply
    assert "GET /api" not in reply


def test_build_grounded_reply_sim2real_status() -> None:
    state = {"sim_viz": {"run_id": "x", "stage": "idle"}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("sim2real_status", state, ["workbench.lerobot"], rerun_ready=False)
    assert "**stage**" in reply
    assert "GET /api" not in reply


def test_build_grounded_reply_watch_sim_mentions_success() -> None:
    state = {"sim_viz": {"run_id": "x", "stage": "running"}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("watch_sim", state, ["workbench.lerobot"], rerun_ready=True)
    assert "SUCCESS" in reply
    assert "blob" in reply
    assert "iframe mount" in reply
    assert "Rerun blob iframe until SUCCESS" in reply
    assert "RERUN_BLOB_SUCCESS=SUCCESS" in reply
    assert "RERUN_MOUNT_SUCCESS=SUCCESS" in reply
    assert "consecutive SUCCESS confirmations" in reply
    assert "**rrd_uri**" in reply


def test_watch_sim_apis_include_rrd_paths() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("watch_sim")
    assert "sim-viz/status" in apis
    assert "sim-viz/rrd" in apis
    assert "sim-viz/rrd-blob" in apis


def test_onboard_solution_reply_is_generic_and_runnable() -> None:
    state = {"sim_viz": {}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("onboard_solution", state, ["workbench.rl.policy_train"])
    assert "npa workbench byof run" in reply or "run_byof_repo.py" in reply
    assert "--base-profile" in reply or "--base-image" in reply
    assert "byof-onboard" in reply or "skills/workflows/byof-onboard" in reply
    assert "oss-solution-registry-onboard" in reply
    assert "upstream docs" in reply
    assert "live Nebius" in reply
    assert "solution-smoke" in reply
    assert "capability" in reply.lower()
    assert "<repo-url>" in reply
    assert "container-verify" in reply or "byof-onboard" in reply
    assert "registry" in reply.lower()
    assert "oss-onboarding-ladder" in reply


def test_onboard_solution_reply_uses_npa_registry_env(monkeypatch) -> None:
    from npa.cli.agent_chat import format_onboard_solution

    monkeypatch.setenv("NPA_REGISTRY", "cr.eu-north1.nebius.cloud/example/project")
    reply = format_onboard_solution()
    assert "cr.eu-north1.nebius.cloud/example/project" in reply
    assert "<resolved-from-~/.npa/config.yaml>" not in reply


def test_onboard_solution_apis_include_tools_and_workflow_gates() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("onboard_solution")
    assert "tools" in apis
    assert "workflows/validate" in apis
    assert "workflows/plan" in apis


def test_onboard_solution_does_not_shadow_create_workflow() -> None:
    assert match_chat_intent("create a LeIsaac BYOF Isaac Lab workflow for live infra") == "create_workflow"
    assert (
        match_chat_intent("containerize a github repo and onboard it into the workbench with sky smoke")
        == "onboard_solution"
    )


def test_find_artifacts_apis_include_discovery_and_load() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("find_artifacts")
    assert "artifacts/runs" in apis
    assert "artifacts/run/{run_id}" in apis
    assert "sim-viz/load-artifact" in apis


def test_component_capabilities_reply_is_targeted() -> None:
    state = {"sim_viz": {}, "selection": {}, "latest_submit": {}}
    cosmos_reply = build_grounded_reply(
        "cosmos_capabilities",
        state,
        ["workbench.cosmos2.transfer", "workbench.token_factory.reason"],
    )
    assert "Cosmos capabilities" in cosmos_reply
    assert "Fine-tuning / post-training" in cosmos_reply

    lancedb_reply = build_grounded_reply(
        "lancedb_capabilities",
        state,
        ["workbench.lancedb.import_bdd100k", "workbench.lancedb.backfill_clip"],
    )
    assert "LanceDB capabilities" in lancedb_reply
    assert "Data ingest" in lancedb_reply


def test_tenant_resources_reply_is_zero_token_grounded_inventory() -> None:
    state = {
        "resources": {
            "context": {
                "project_alias": "demo",
                "project_id": "project-test",
                "tenant_id": "tenant-test",
                "region": "us-central1",
                "profile": "cursor-sa",
            },
            "categories": [
                {
                    "id": "compute",
                    "label": "Compute",
                    "status": "discovered",
                    "configured": [],
                    "discovered": [{"name": "agent-demo"}],
                    "configured_count": 0,
                    "discovered_count": 1,
                },
                {
                    "id": "storage",
                    "label": "Object storage",
                    "status": "error",
                    "configured": [{"name": "configured-bucket"}],
                    "discovered": [],
                    "configured_count": 1,
                    "discovered_count": 0,
                    "error": {"kind": "permission_denied", "message": "Not enumerable."},
                },
            ],
        }
    }
    reply = build_grounded_reply("tenant_resources", state, [])
    assert "**Tenant resources**" in reply
    assert "**discovered_resources**: `1`" in reply
    assert "permission_denied" in reply


def test_live_infra_loop_reply_mentions_registry_and_gpu_checks() -> None:
    state = {"sim_viz": {}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("live_infra_loop", state, ["workbench.cosmos2.transfer"])
    assert "Live infra loop guidance" in reply
    assert "never `<your-registry-id>` placeholders" in reply or "no placeholders" in reply
    assert "sky gpus list" in reply
    assert "FAILED_PRECHECKS" in reply


def test_embedded_agent_chat_source_strips_future_import() -> None:
    source = agent_module._embedded_agent_chat_source()
    assert "from __future__ import annotations" not in source
    assert "match_chat_intent" in source
    assert "INTENT_APIS" in source
    assert "onboard_solution" in source
    assert "format_onboard_solution" in source


def test_match_soperator_intent() -> None:
    assert match_chat_intent("deploy a soperator cluster") == "soperator"
    assert match_chat_intent("deploy slurm on kubernetes") == "soperator"
    assert match_chat_intent("spin up a slurm cluster with docker cache") == "soperator"
    assert match_chat_intent("can npa deploy slurm-on-k8s?") == "soperator"


def test_soperator_grounded_reply_points_to_npa_deploy() -> None:
    reply = build_grounded_reply("soperator", {}, ["infra.soperator.deploy"])
    assert "POST /api/infra/soperator/deploy" in reply
    assert "POST /api/infra/soperator/validate" in reply
    assert "GET /api/infra/soperator/status/{name}" in reply
    assert "npa soperator deploy" in reply
    assert "npa.soperator/v0.0.1" in reply
    assert "docker_cache" in reply


def test_mk8s_provision_grounded_reply_points_to_agent_api() -> None:
    assert match_chat_intent("deploy an mk8s cluster for workflows") == "mk8s_provision"
    reply = build_grounded_reply("mk8s_provision", {}, [])
    assert "POST /api/infra/mk8s/provision" in reply
    assert "npa provision-if-absent" in reply
    assert "dry_run" in reply


def test_chat_action_mode_drives_readonly_loop_over_insights() -> None:
    # A read-only "action" turn now returns a real loop result (steps/tools_used
    # trace + grounded final), not the old "POST /api/agent/act" boilerplate.
    planner = _planner(
        [
            {"tool": "insights_compare", "args": {"base_run": "r1", "candidate_run": "r2"}},
            {"final": "Run `r2` regressed on **collision_rate** vs `r1`."},
        ]
    )
    tools = {"insights_compare": lambda args: {"regressed": ["collision_rate"], "improved": []}}
    result = agent_actions.run_chat_action_loop(
        "which runs regressed on collision rate", tools=tools, model_call=planner
    )
    assert result["mode"] == agent_actions.CHAT_ACTION_MODE
    assert result["grounded"] is False
    assert result["tools_used"] == ["insights_compare"]
    assert result["steps"], "chat action turn must return a step trace"
    assert result["needs_confirmation"] is False
    assert "collision_rate" in result["reply"]


def test_chat_action_mode_gpu_turn_still_needs_confirmation() -> None:
    launched = {"n": 0}

    def _submit(args):  # pragma: no cover - a chat turn must never auto-launch GPU
        launched["n"] += 1
        return {"run_id": "x"}

    planner = _planner([{ "tool": "sim2real_submit", "args": {"run_id": "x"}}])
    result = agent_actions.run_chat_action_loop(
        "launch a big sim2real run", tools={"sim2real_submit": _submit}, model_call=planner
    )
    assert launched["n"] == 0
    assert result["needs_confirmation"] is True
    assert result["proposed_action"]["tool"] == "sim2real_submit"


def test_author_workflow_requests_route_to_create_workflow_not_capabilities() -> None:
    # "write me a 2 step npa yaml that uses cosmos" must generate a workflow,
    # not fall through to the cosmos capabilities blurb.
    assert match_chat_intent("write me a 2 step npa yaml that uses cosmos") == "create_workflow"
    assert match_chat_intent("generate a 3-step npa spec that uses cosmos") == "create_workflow"
    assert match_chat_intent("build an npa yaml pipeline that uses lancedb") == "create_workflow"
    assert match_chat_intent("draft a two step npa.workflow spec") == "create_workflow"
    # Non-authoring cosmos questions still route to capabilities.
    assert match_chat_intent("what does cosmos support for finetuning") == "cosmos_capabilities"
    assert match_chat_intent("what can cosmos do") == "cosmos_capabilities"


def test_metric_resource_queries_fall_through_to_insights() -> None:
    # BUG #3: metric/resource/comparison qualifiers must NOT be intercepted by a
    # grounded run-listing intent — they fall through so the insights loop runs.
    for turn in (
        "list runs by gpu count",
        "how many gpus did each run use",
        "which runs used more than 2 gpus",
        "compare gpus between run-a and run-b",
        "list runs by accelerator count",
        "which runs regressed on success rate",
    ):
        assert match_chat_intent(turn) not in {"find_artifacts", "list_recordings"}, turn


def test_plain_run_listing_stays_grounded_zero_tokens() -> None:
    # No metric/resource qualifier -> still grounded run/recording history.
    assert match_chat_intent("list recent runs") == "list_recordings"
    assert match_chat_intent("what recordings do I have") == "list_recordings"
    assert match_chat_intent("show me my run history") == "list_recordings"
    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"


def test_has_metric_resource_qualifier_helper() -> None:
    from npa.cli.agent_chat import has_metric_resource_qualifier

    assert has_metric_resource_qualifier("by gpu count")
    assert has_metric_resource_qualifier("compare the runs")
    assert has_metric_resource_qualifier("regressed on collision rate")
    assert not has_metric_resource_qualifier("list recent runs")
    assert not has_metric_resource_qualifier("what recordings do I have")


def test_author_workflow_from_goal_composes_cosmos_from_live_catalog() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    result = author_workflow_from_goal(
        "write me a 2 step npa yaml that uses cosmos", tool_refs=frozenset(TOOL_CATALOG)
    )
    assert result["runnable"] is False
    assert result["context_errors"]
    assert len(result["states"]) == 2
    assert result["tool_refs"] and all(ref in TOOL_CATALOG for ref in result["tool_refs"])
    assert any("cosmos" in ref for ref in result["tool_refs"])
    assert "npa.workflow/v0.0.1" in result["yaml"]
    # Two cosmos tools match a 2-step cosmos goal, so no padding.
    assert result["padded_tool_refs"] == []


def test_author_workflow_semantically_chains_curate_train_eval() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

    result = author_workflow_from_goal(
        "3-step npa yaml: curate a dataset then train then eval",
        tool_refs=frozenset(TOOL_CATALOG),
    )

    assert result["runnable"] is False
    assert result["yaml"], "valid structure remains available for operator repair"
    assert result["context_errors"]
    refs = result["tool_refs"]
    assert len(refs) == 3
    assert "curat" in refs[0]
    assert "train" in refs[1]
    assert "eval" in refs[2] or "evaluate" in refs[2]
    assert result["padded_tool_refs"] == []
    assert len(result["data_flow"]) == 2

    spec = yaml.safe_load(result["yaml"])
    config = spec["config"]
    states = list(spec["states"].values())
    for index, link in enumerate(result["data_flow"]):
        output_key = link["output_config"]
        input_key = link["input_config"]
        assert states[index]["outputs"][0]["uri"] == f"{{{{config.{output_key}}}}}"
        assert states[index + 1]["inputs"][0]["uri"] == f"{{{{config.{output_key}}}}}"
        if input_key != output_key:
            assert config[input_key] == f"{{{{config.{output_key}}}}}"

    referenced_config = {
        key
        for ref in refs
        for token in argv_for_tool(ref)
        for key in re.findall(r"\{\{\s*config\.([a-zA-Z0-9_.-]+)\s*\}\}", str(token))
    }
    assert referenced_config <= set(config), "every live argv config token must resolve"


def test_author_workflow_keeps_semantic_flow_when_requested_count_differs() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    result = author_workflow_from_goal(
        "write a 4-step npa yaml: curate a dataset -> train a policy -> evaluate it",
        tool_refs=frozenset(TOOL_CATALOG),
    )

    assert result["runnable"] is False
    assert result["context_errors"]
    assert len(result["tool_refs"]) == 4
    assert "curat" in result["tool_refs"][0]
    assert "train" in result["tool_refs"][1]
    assert "eval" in result["tool_refs"][2] or "evaluate" in result["tool_refs"][2]
    assert result["padded_tool_refs"] == [result["tool_refs"][3]]
    assert "placeholder" in result["yaml"].lower()
    assert len(result["data_flow"]) >= 2


def test_author_workflow_understands_paraphrased_semantic_flow() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    result = author_workflow_from_goal(
        "Refine the dataset before fitting a policy, followed by benchmarking it.",
        tool_refs=frozenset(TOOL_CATALOG),
    )

    assert result["runnable"] is False
    assert result["context_errors"]
    assert len(result["tool_refs"]) == 3
    assert "curat" in result["tool_refs"][0]
    assert "train" in result["tool_refs"][1]
    assert "eval" in result["tool_refs"][2] or "evaluate" in result["tool_refs"][2]
    assert result["padded_tool_refs"] == []
    assert len(result["data_flow"]) == 2


def test_author_workflow_returns_no_yaml_when_validation_fails(mocker) -> None:
    from npa.cli import agent_workflow
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    mocker.patch.object(
        agent_workflow,
        "validate_workflow_yaml_text",
        return_value={"ok": False, "error": "invalid authored workflow"},
    )
    result = agent_workflow.author_workflow_from_goal(
        "write me a 2 step npa yaml that uses cosmos",
        tool_refs=frozenset(TOOL_CATALOG),
    )

    assert result["runnable"] is False
    assert result["yaml"] == ""


def test_author_workflow_flags_padded_placeholder_states() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    # More requested steps than goal-matched tools -> the extra state is padded from the catalog
    # and flagged as a placeholder for the operator to replace.
    #
    # The keyword is chosen rather than fixed. The author caps a workflow at MAX_AUTHORED_STEPS
    # states, so a keyword matching that many tools leaves no room to pad — which is exactly
    # what happened when the sixth cosmos toolRef landed and this test started asserting 6 > 6.
    max_steps = 6
    keyword = next(
        (
            candidate
            for candidate in ("cosmos", "sonic", "mjlab", "retargeting")
            if len([ref for ref in TOOL_CATALOG if candidate in ref.lower()]) < max_steps
        ),
        "",
    )
    assert keyword, "no catalog keyword leaves headroom for a padded state"
    matched = [ref for ref in TOOL_CATALOG if keyword in ref.lower()]
    n_steps = len(matched) + 1
    assert n_steps <= max_steps
    result = author_workflow_from_goal(
        f"write me a {n_steps} step npa yaml that uses {keyword}",
        tool_refs=frozenset(TOOL_CATALOG),
    )
    assert len(result["tool_refs"]) == n_steps
    assert result["padded_tool_refs"], "extra state should be flagged as padding"
    if result["runnable"]:
        assert "placeholder" in result["yaml"].lower()


def test_embedded_chat_action_branch_drives_loop_not_boilerplate() -> None:
    # Guard the /chat action branch wiring in the embedded backend f-string:
    # it must drive the bounded loop, not describe the POST /api/agent/act recipe.
    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "Use `POST /api/agent/act` with a JSON body carrying your goal" not in source
    assert "run_chat_action_loop(" in source
    assert '"insights_query": _tool_insights_query' in source


def test_foxglove_intent_routes_and_grounds() -> None:
    from npa.cli.agent_chat import build_grounded_reply, match_chat_intent

    for text in (
        "open foxglove",
        "foxglove status",
        "show me the mcap recording",
        "can I play this rosbag?",
    ):
        assert match_chat_intent(text) == "foxglove_viewer", text

    # Rerun / generic viewer turns must not be captured by the new intent.
    for text in ("watch the sim", "what is the current status", "open the rerun timeline"):
        assert match_chat_intent(text) != "foxglove_viewer", text

    reply = build_grounded_reply(
        "foxglove_viewer",
        {
            "foxglove": {
                "available": True,
                "embed_src": "https://embed.foxglove.dev/",
                "org_slug": "acme",
                "sdk_version": "0.58.0",
                "run_id": "run-7",
                "artifact_key": "run-7/reports/session.mcap",
                "data_source": {
                    "type": "remote-file",
                    "urls": ["https://agent.example/foxglove/data/tok-run.mcap"],
                },
            },
            "sim_viz": {"foxglove_ready": True, "run_id": "run-7"},
        },
        [],
    )
    assert "**Foxglove viewer**" in reply
    assert "`https://embed.foxglove.dev/`" in reply
    assert "remote-file" in reply
    assert "run-7" in reply
    assert "0.58.0" in reply
    # Honest about the cross-origin capture limit.
    assert "cross-origin iframe" in reply


def test_foxglove_grounded_reply_explains_unconfigured_state() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    reply = build_grounded_reply(
        "foxglove_viewer",
        {"foxglove": {"available": False, "reason": "Foxglove SDK assets are not installed."}},
        [],
    )
    assert "`False`" in reply
    assert "not installed" in reply
    assert "--foxglove-embed-src" in reply
    # Never claim a viewer is showing data when it is not configured.
    assert "ready" not in reply.lower().split("foxglove_ready")[0]


def test_keyword_skill_rules_lead_with_the_npa_workflow_skill() -> None:
    """A Cosmos 3 workflow ask must not be answered with the SkyPilot template.

    The two files share a name but differ in shape and submit command, so the
    declarative-spec skill has to come first for turns the intent router leaves
    unclassified.
    """
    from npa.cli.agent_chat import skill_names_for_keywords

    for text in (
        "write me a cosmos3 workflow yaml",
        "cosmos 3 npa spec please",
        "generate a COSMOS3 WORKFLOW",
    ):
        assert skill_names_for_keywords(text) == ["cosmos3-npa-workflow"], text

    # Unrelated or too-generic turns must not pull the skill in.
    for text in ("write me a workflow yaml", "what is cosmos3", "run cosmos2 transfer"):
        assert skill_names_for_keywords(text) == [], text

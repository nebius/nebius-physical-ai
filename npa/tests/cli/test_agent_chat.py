from __future__ import annotations

from npa.cli import agent as agent_module
from npa.cli.agent_chat import (
    build_grounded_reply,
    format_sim2real_status,
    match_chat_intent,
)


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
    assert "run_byof_repo.py" in reply
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
    assert "Cosmos component capabilities" in cosmos_reply
    assert "Fine-tuning / post-training" in cosmos_reply

    lancedb_reply = build_grounded_reply(
        "lancedb_capabilities",
        state,
        ["workbench.lancedb.import_bdd100k", "workbench.lancedb.backfill_clip"],
    )
    assert "LanceDB component capabilities" in lancedb_reply
    assert "Data ingest" in lancedb_reply


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

"""Exercise real GLM and DeepSeek specialists repairing public Workbench workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from npa.agent_backend.specialists.config import Operation, Profile, TeamConfig
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.worker import supervise
from npa.clients.token_factory import TokenFactoryClient

pytestmark = [pytest.mark.token_factory_e2e]
_ROOT = Path(__file__).resolve().parents[3]
_MODELS = {"glm": "zai-org/GLM-5.3", "deepseek": "deepseek-ai/DeepSeek-V4-Pro-0813"}
_WORKFLOWS = {
    "glm": ("paidf-cosmos3.yaml", "generate-configs"),
    "deepseek": ("sim2real.yaml", "stage-04-wave"),
}


def _operations(filename):
    operations = {}
    for verb in ("validate", "plan"):
        argv = [
            "{python}",
            "-m",
            "npa",
            "workbench",
            "workflow",
            verb + "-spec",
            "workflows/" + filename,
            "--json",
        ]
        if filename == "sim2real.yaml" and verb == "plan":
            argv += ["--assume-decision", "promote_checkpoint"]
        operations[verb] = Operation(
            argv=argv,
            description=f"{verb} the real workflow using Workbench; nonzero exit means failure.",
        )
    return operations


def _profile(tmp_path, name, model):
    filename, transition = _WORKFLOWS[name]
    source = (_ROOT / "workflows/main" / filename).read_text()
    original = "next: " + transition + "\n"
    assert source.count(original) == 1
    workspace = tmp_path / name
    (workspace / "workflows").mkdir(parents=True)
    (workspace / "workflows" / filename).write_text(
        source.replace(original, "next: " + transition + "-missing\n")
    )
    options = (
        {"reasoning_effort": "none"}
        if name == "deepseek"
        else {"chat_template_kwargs": {"reasoning_effort": "low"}}
    )
    return Profile(
        name=name,
        model=model,
        description="Repair and validate a Workbench workflow",
        instructions="Run validate first to capture the defect, read the workflow, repair it, then run validate and plan. Do not stop before both pass.",
        workspace=workspace,
        read_paths=["workflows"],
        write_paths=["workflows"],
        model_options=options,
        operations=_operations(filename),
    )


def _configuration(tmp_path):
    return TeamConfig(
        state_directory=tmp_path / "state",
        profiles=[_profile(tmp_path, name, model) for name, model in _MODELS.items()],
        default_profile="glm",
    )


def _wait_for(team, predicate):
    while True:
        state = team.status()
        failures = [
            task for task in state["tasks"] if task["status"] == "needs_attention"
        ]
        assert not failures, [(task["profile"], task["error"]) for task in failures]
        if predicate(state):
            return state
        time.sleep(0.2)


def _assert_results(team, config):
    summary = {"models": [], "restarted_workers": 2, "remote_workloads_submitted": 0}
    for profile in config.profiles:
        result = team.status(profile.name)
        events = result["events"]
        models = [event for event in events if event["type"] == "model"]
        operations = [
            event["result"]
            for event in events
            if event["type"] == "tool" and event["name"] == "run_operation"
        ]
        assert models and all(event["model"] == profile.model for event in models)
        assert any(item["returncode"] != 0 for item in operations)
        assert {
            item["operation"] for item in operations if item["returncode"] == 0
        } >= {"validate", "plan"}
        filename, transition = _WORKFLOWS[profile.name]
        repaired = (profile.workspace / "workflows" / filename).read_bytes()
        assert repaired == (_ROOT / "workflows/main" / filename).read_bytes()
        assert transition + "-missing" in team.patch(profile.name)
        summary["models"].append(
            {
                "model": profile.model,
                "status": result["status"],
                "model_responses": len(models),
                "operation_receipts": len(operations),
                "patch_recorded": True,
            }
        )
    return summary


def test_two_live_specialists_resume_and_repair_workflows(tmp_path):
    if os.environ.get("NPA_SPECIALISTS_LIVE") != "1":
        pytest.skip("set NPA_SPECIALISTS_LIVE=1 for paid hosted inference")
    assert set(_MODELS.values()) <= set(TokenFactoryClient().list_models())
    config = _configuration(tmp_path)
    path = tmp_path / "team.json"
    path.write_text(config.model_dump_json(indent=2))
    path.chmod(0o600)
    team = SpecialistTeam(config)
    for name in _MODELS:
        team.submit(
            f"Repair the broken transition in workflows/{_WORKFLOWS[name][0]} and prove validation and planning pass.",
            specialist=name,
            task_id=name,
        )
    with supervise(str(path)):
        _wait_for(team, lambda state: all(team.store._events(name) for name in _MODELS))
        for name in _MODELS:
            team.pause(specialist=name)
        before = team.store._workers()
    with supervise(str(path)):
        _wait_for(
            team,
            lambda state: all(
                team.store._workers()[name]["pid"] != before[name]["pid"]
                for name in _MODELS
            ),
        )
        for name in _MODELS:
            team.pause(specialist=name, paused=False)
        _wait_for(
            team,
            lambda state: all(task["status"] == "completed" for task in state["tasks"]),
        )
    summary = _assert_results(team, config)
    evidence = os.environ.get("NPA_SPECIALISTS_EVIDENCE", "")
    if evidence:
        Path(evidence).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))

"""Compare actual Astra coordination with and without independent Token Factory workers."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from npa.agent_backend.specialists.config import load_config, private_directory
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.worker import supervise

from evidence import _granted_tools, _receipts, _snapshot, _write_json
from workflow_bridge import _coordinator_store

HERE = Path(__file__).resolve().parent


def _settings(config_path, directory, arm, effort):
    settings = {
        "model_reasoning_effort": effort,
        "approval_policy": "never",
        "mcp_servers.workbench.command": sys.executable,
        "mcp_servers.workbench.args": [
            str(HERE / "workflow_bridge.py"),
            str(config_path),
            str(directory),
            arm,
        ],
        # This is a transport deadline, not a job runtime limit. Wait observations are <=60s.
        "mcp_servers.workbench.tool_timeout_sec": 120,
    }
    names = _granted_tools(arm, load_config(config_path))
    for name in names:
        settings[f"mcp_servers.workbench.tools.{name}.approval_mode"] = "approve"
    return settings


def _astra_argv(config_path, directory, arm, effort):
    argv = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "-m",
        "gpt-6-astra",
        "-s",
        "workspace-write",
        "-C",
        str(directory),
    ]
    for feature in ("shell_tool", "unified_exec", "multi_agent"):
        argv.extend(["--disable", feature])
    for key, value in _settings(config_path, directory, arm, effort).items():
        argv.extend(["-c", key + "=" + json.dumps(value)])
    return [*argv, "-"]


def _workspace_policies(team):
    return [
        {
            "name": profile.name,
            "description": profile.description,
            "instructions": profile.instructions,
            "read_paths": profile.read_paths,
            "write_paths": profile.write_paths,
            "operations": {
                name: operation.description
                for name, operation in profile.operations.items()
            },
            "required_operations": profile.required_operations,
            "observation_only_operations": [
                name
                for name, operation in profile.operations.items()
                if operation.observation_only
            ],
        }
        for profile in team.config.profiles
    ]


def _prompt(common, team, arm):
    instructions = (
        "Use only the configured Workbench MCP tools. Native shell/file tools and "
        "additional Codex agents are outside this comparison. Use run_operations for "
        "concurrent work across distinct workspaces. A submit receipt is not workload "
        "completion: inspect real status, logs and verification receipts. Never repeat "
        "an uncertain external effect. Report all failures and unfinished work honestly.\n"
    )
    if arm == "astra-tofa":
        instructions += (
            "Initially delegate each independent workspace task to its configured Token Factory specialist. Use stable "
            "task IDs. Prefer wait_specialists across active tasks, carrying its after_sequences "
            "cursors: it waits for completion or attention without waking on routine worker events. "
            "Inspect full specialist_status receipts when a task finishes or needs attention, "
            "then remove ended tasks from subsequent waits. "
            "A delegated workspace remains owned until its task ends; do not edit or run "
            "effectful operations concurrently there. If a specialist needs attention, inspect its receipts "
            "and use take_over only when its effects are resolved, then perform the recovery "
            "yourself. Finish after verifying every required result.\n"
        )
        if "dismiss_interrupted_observation" in _granted_tools(arm, team.config):
            instructions += (
                "For needs_attention tasks, explicitly declared observation_only operations permit fresh "
                "diagnostic reads while ownership remains blocked. Use dismiss_interrupted_observation only "
                "for a lost, originally classified observation. It records failure without replay or requeue; "
                "take_over still requires every uncertain effect to be resolved.\n"
            )
    return (
        instructions
        + "\nWorkspace policies:\n"
        + json.dumps(_workspace_policies(team))
        + "\n\n"
        + common
    )


def _prepare(config_path, prompt_path, directory, arm, effort):
    config = load_config(config_path)
    common = prompt_path.read_text()
    if not common.strip():
        raise ValueError("the common task prompt must not be empty")
    for profile in config.profiles:
        if directory.is_relative_to(profile.workspace):
            raise ValueError("experiment evidence must be outside editable workspaces")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_directory(directory)
    team = SpecialistTeam(config)
    if team.store._list():
        raise ValueError("a new trial requires a fresh specialist state directory")
    saved = directory / "team.json"
    saved.write_text(config.model_dump_json(indent=2) + "\n")
    (directory / "prompt.txt").write_text(common)
    _write_json(
        directory / "protocol.json",
        {
            "schema": "npa.specialists.workflow_experiment.v1",
            "arm": arm,
            "model": "gpt-6-astra",
            "reasoning_effort": effort,
            "common_prompt_sha256": hashlib.sha256(common.encode()).hexdigest(),
            "team_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
            "sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in HERE.glob("*.py")
            },
            "workload_budgets": None,
        },
    )
    return team, saved, _prompt(common, team, arm)


def _codex(config_path, directory, arm, effort, prompt):
    argv = _astra_argv(config_path, directory, arm, effort)
    _write_json(directory / "coordinator-config.json", {"argv": argv})
    with (
        (directory / "codex.jsonl").open("w") as output,
        (directory / "codex.stderr").open("w") as errors,
    ):
        result = subprocess.run(
            argv,
            input=prompt,
            text=True,
            stdout=output,
            stderr=errors,
            check=False,
        )
    return result.returncode


def _coordinator_ended(team, directory, hybrid):
    _write_json(directory / "coordinator-end-task-receipts.json", _receipts(team.store))
    if hybrid:
        for profile in team.config.profiles:
            # Finish the current durable node, without beginning another inference call.
            team.pause(specialist=profile.name)


def _run(config_path, prompt_path, directory, arm, effort):
    team, saved, prompt = _prepare(config_path, prompt_path, directory, arm, effort)
    coordinator = _coordinator_store(team, directory)
    hybrid = arm == "astra-tofa"
    started = time.perf_counter()
    execution = {"arm": arm, "started_epoch": time.time(), "exit_code": None}
    try:
        with supervise(str(saved)) if hybrid else nullcontext():
            try:
                execution["exit_code"] = _codex(saved, directory, arm, effort, prompt)
            finally:
                execution["coordinator_seconds"] = time.perf_counter() - started
                execution["coordinator_ended_epoch"] = time.time()
                _coordinator_ended(team, directory, hybrid)
    except BaseException as error:
        execution["error_type"] = type(error).__name__
        raise
    finally:
        execution["agent_tool_seconds"] = time.perf_counter() - started
        execution["ended_epoch"] = time.time()
        execution["remote_workloads_cancelled"] = False
        _snapshot(team, coordinator, directory, execution)
    return execution


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["astra-only", "astra-tofa"], required=True)
    parser.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument("--live", action="store_true")
    options = parser.parse_args()
    if not options.live:
        parser.error(
            "--live is required: this invokes paid models and configured operations"
        )
    os.umask(0o077)
    print(
        json.dumps(
            _run(
                options.team_config.resolve(),
                options.prompt.resolve(),
                options.output.resolve(),
                options.arm,
                options.effort,
            )
        )
    )


if __name__ == "__main__":
    _main()

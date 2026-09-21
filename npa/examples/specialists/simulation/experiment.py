"""Prepare matched workspaces and measure real specialist or single-Codex execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from npa.agent_backend.specialists.config import load_config
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.worker import supervise
from npa.clients.token_factory import resolve_config

from workload import TASKS, _prepare_workspace, _task_text

HERE = Path(__file__).resolve().parent
MODELS = ["zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4-Pro-0813"]


def _profile(root, name, index, seed):
    workspace = root / "workspaces" / name
    _prepare_workspace(workspace, name)
    model = MODELS[index % len(MODELS)]
    operations = {
        verb: {
            "argv": [
                "{python}",
                str(HERE / "operation.py"),
                verb,
                "--task",
                name,
                "--seed",
                str(seed),
            ],
            "description": "Check the requested scene and full parameter coverage"
            if verb == "validate"
            else "Execute all six real MuJoCo parameter cases; inspect actual accepted/total results",
        }
        for verb in ("validate", "simulate")
    }
    return {
        "name": name,
        "description": TASKS[name]["instruction"],
        "instructions": _task_text(name),
        "model": model,
        "workspace": str(workspace),
        "read_paths": ["TASK.md", "plan.json"],
        "write_paths": ["plan.json"],
        "operations": operations,
        "model_options": {"chat_template_kwargs": {"reasoning_effort": "low"}}
        if index % 2 == 0
        else {"reasoning_effort": "none"},
    }


def _prepare(root, seed, rotation):
    root.mkdir(parents=True, exist_ok=False)
    profiles = [
        _profile(root, name, index + rotation, seed) for index, name in enumerate(TASKS)
    ]
    config = {
        "state_directory": str(root / "state"),
        "profiles": profiles,
        "default_profile": next(iter(TASKS)),
        "router": "explicit",
    }
    path = root / "team.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    team = SpecialistTeam(load_config(path))
    for name in TASKS:
        team.submit(_task_text(name), specialist=name, task_id=name)
    return path


def _stack(path):
    team = SpecialistTeam(load_config(path))
    os.environ["NEBIUS_TOKEN_FACTORY_KEY"] = resolve_config().api_key
    with supervise(str(path)):
        while True:
            tasks = team.status()["tasks"]
            if all(
                task["status"] in {"completed", "needs_attention", "cancelled"}
                for task in tasks
            ):
                return {
                    "tasks": [
                        {"task": task["id"], "status": task["status"]} for task in tasks
                    ]
                }
            time.sleep(0.1)


def _astra_argv(path, effort):
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
        str(path.parent),
    ]
    for feature in ("shell_tool", "unified_exec", "multi_agent"):
        argv.extend(["--disable", feature])
    settings = {
        "model_reasoning_effort": effort,
        "approval_policy": "never",
        "mcp_servers.workbench.command": sys.executable,
        "mcp_servers.workbench.args": [str(HERE / "mcp_bridge.py"), str(path)],
    }
    for tool in (
        "read_file",
        "list_files",
        "edit_file",
        "run_operation",
        "run_operations",
    ):
        settings[f"mcp_servers.workbench.tools.{tool}.approval_mode"] = "approve"
    for key, value in settings.items():
        argv.extend(["-c", key + "=" + json.dumps(value)])
    return [*argv, "-"]


def _astra(path, effort):
    root = path.parent
    prompt = (
        "Complete all six independent Workbench tasks: " + ", ".join(TASKS) + ". "
        "Use only the configured workbench MCP tools. Native shell/file tools and other agents "
        "are outside this comparison. Read TASK.md and plan.json for each task, first run "
        "validate to capture its failure, repair plan.json, then validate and simulate. "
        "Use run_operations to batch validate or simulate across all six tasks concurrently. "
        "You may call tools for independent tasks concurrently; fan out simulations rather "
        "than waiting for one task before starting another. The same scope and tools are "
        "given to the other experiment arm. Finish only after inspecting all six real "
        "simulation receipts and report successes and failures accurately."
    )
    with (
        (root / "codex.jsonl").open("w") as output,
        (root / "codex.stderr").open("w") as errors,
    ):
        process = subprocess.run(
            _astra_argv(path, effort),
            input=prompt,
            text=True,
            stdout=output,
            stderr=errors,
        )
    return {
        "exit_code": process.returncode,
        "model": "gpt-6-astra",
        "reasoning_effort": effort,
    }


def _measure(root, arm, seed, rotation):
    path = _prepare(root, seed, rotation)
    started = time.perf_counter()
    result = (
        _stack(path)
        if arm == "specialists"
        else _astra(path, arm.removeprefix("astra-"))
    )
    result.update(arm=arm, agent_tool_seconds=time.perf_counter() - started, seed=seed)
    team = SpecialistTeam(load_config(path))
    receipts = {name: team.status(name) for name in TASKS}
    (root / "task-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (root / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=["specialists", "astra-medium", "astra-xhigh"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    arguments = parser.parse_args()
    if not arguments.live:
        parser.error("--live is required: this runs paid inference and real simulation")
    os.umask(0o077)
    print(
        json.dumps(
            _measure(
                arguments.output.resolve(),
                arguments.arm,
                arguments.seed,
                arguments.rotation,
            )
        )
    )


if __name__ == "__main__":
    _main()

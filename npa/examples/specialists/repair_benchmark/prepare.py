"""Freeze equal historical repair workspaces, tests and trusted runtime inputs for both arms."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from npa.agent_backend.specialists.config import TeamConfig

from operation import _digest, _write
from sandbox import TARGETS

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
TASKS = {
    "physics": (
        TARGETS[0],
        "Validate complete, aligned physical traces before judging. Reject missing evidence, "
        "wrong widths, non-real state/action/position arrays and too-short settling traces. "
        "Require one-dimensional contacts and success streams. Include non-finite goal and "
        "object-typed numeric evidence in finiteness checks. String phase metadata is valid. "
        "Avoid converting uint8 camera recordings into float64 copies. Preserve the existing "
        "real controller, thresholds, action order, phase lengths and positive/negative outcomes.",
    ),
    "publication": (
        TARGETS[1],
        "Publish accepted robot datasets transactionally for a single writer per run. "
        "Complete conversion and metadata enrichment in staging on the run filesystem, "
        "then publish and assign provenance indices. Encoder or metadata errors must leave "
        "raw episodes intact, no dataset and no phantom indices; retry must succeed. Reject "
        "all existing destinations, including dangling links. Resolve accepted source "
        "directories inside the run root and reject duplicates or aliases. Preserve language "
        "tasks, rejected episodes and contiguous accepted indices.",
    ),
    "adapter": (
        TARGETS[2],
        "Validate every simulation episode before any output creation or encoding. Require "
        "nonempty aligned streams, finite real numeric state/actions, consistent widths, "
        "uint8 RGB and matching resolutions for both cameras, including episode zero. "
        "Reject scalar streams, zero-width features, bool/complex/string numerics and invalid "
        "frame rates. Preserve FileNotFoundError for missing required arrays and exact "
        "length diagnostics: 'Episode N: STREAM has X frames but state has Y'. Preserve "
        "valid single and multiple episodes, statistics, language tasks and LeRobot v3 metadata.",
    ),
}
CHECK_FILES = (
    "workbench/test_robot_physics_trace_contract.py",
    "workbench/test_robot_export_transaction.py",
    "workbench/test_robot_export_provenance.py",
    "workbench/test_token_factory_robot_sdg.py",
    "test_sim_episode_stream_lengths.py",
    "test_adapter.py",
)
PROCESS = (
    "Run diagnose first, read the granted source and regression cases, and repair only "
    "the granted file. Keep new helpers under 40 lines with meaningful names and docstrings. "
    "Other components are fixed references during this lane; do not change them. Preserve "
    "valid behavior and fix requirements generally, without special-casing tests or inputs. "
    "After diagnose passes, submit the real three-case MuJoCo/LeRobot workflow, wait for "
    "completion, inspect logs and repair any failure, then run verify on the final revision. "
    "Use wait to observe running work without extra model polling. Completion requires "
    "current-source test receipts and independent native replay/reader verification. "
    "The host will combine all three final patches and run a six-case integration check."
)


def _copy_inputs(root):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(REPOSITORY / "npa/src", root / "source", ignore=ignore)
    shutil.copytree(HERE.parent / "robot_workflow", root / "driver", ignore=ignore)
    shutil.copytree(HERE, root / "harness", ignore=ignore)
    shutil.copytree(HERE.parent / "workflows", root / "coordinator", ignore=ignore)
    (root / "tests").mkdir()
    for relative in CHECK_FILES:
        shutil.copyfile(
            REPOSITORY / "npa/tests" / relative, root / "tests" / Path(relative).name
        )
    shutil.copyfile(HERE.parent / "workflow_usage.py", root / "workflow_usage.py")


def _candidate(root, workspace, target, baseline):
    path = workspace / "npa/src" / target
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = subprocess.check_output(
        ["git", "show", f"{baseline}:npa/src/{target}"], cwd=REPOSITORY
    )
    path.write_bytes(contents)
    for source in (root / "tests").glob("*.py"):
        shutil.copyfile(source, workspace / source.name)


def _operation_config(root, workspace, state, targets, options, combined=False):
    return {
        "workspace": str(workspace),
        "state": str(state),
        "targets": list(targets),
        "source": str(root / "source"),
        "tests": str(root / "tests"),
        "driver": str(root / "driver"),
        "matrix": str(
            root / "driver" / ("multimodel-scenes.json" if combined else "scenes.json")
        ),
        "python": str(options.native_python.absolute()),
        "test_python": sys.executable,
        "native_python": str(options.reader_python.absolute()),
    }


def _operations(root, config, native_failure_handoff=False):
    descriptions = {
        "diagnose": "Run all immutable regression checks against the current candidate snapshot.",
        "submit": "Submit real native simulation, conversion and independent replay/reader validation.",
        "status": "Read the durable native attempt state without submitting work.",
        "wait": "Wait for native completion without spending model calls on periodic polling.",
        "logs": "Read real native execution and independent-verifier diagnostics.",
        "verify": "Require passing regressions and native artifacts bound to the current source bytes.",
    }
    result = {
        name: {
            "argv": [
                sys.executable,
                str(root / "harness/operation.py"),
                "--config",
                str(config),
                name,
            ],
            "description": description,
            "observation_only": name in {"status", "wait", "logs"},
        }
        for name, description in descriptions.items()
    }
    result["wait"]["wait_for"] = {
        "field": "/status",
        "pending_values": ["running", "submitted"],
        "success_values": ["completed"],
        "failure_values": ["failed", "uncertain", "not_submitted"],
        "poll_interval": 1.0,
    }
    if native_failure_handoff:
        result["wait"]["handoff_on_failure"] = True
    return result


def _endpoint_policy(protocol):
    fast, capable = protocol["worker_models"]
    return {
        "model": fast,
        "model_options": {"chat_template_kwargs": {"reasoning_effort": "low"}},
        "fallback_models": [
            {
                "model": capable,
                "model_options": {
                    "chat_template_kwargs": {"reasoning_effort": "medium"}
                },
            }
        ],
        "model_router": "token_factory",
        "routing_model": {"model": protocol["routing_model"]},
        "require_model_route": True,
        "model_criteria": {
            fast: "Prefer for precisely specified single-module repairs with executable checks.",
            capable: "Use for ambiguous or interacting lifecycle requirements that the cheaper worker cannot resolve.",
        },
    }


def _profile(root, directory, task, options, protocol):
    target, requirements = TASKS[task]
    workspace = directory / "workspaces" / task
    _candidate(root, workspace, target, protocol["baseline_ref"])
    instructions = requirements + "\n" + PROCESS
    (workspace / "TASK.md").write_text(instructions + "\n")
    path = directory / "configs" / f"{task}.json"
    config = _operation_config(
        root, workspace, directory / "operations" / task, [target], options
    )
    _write(path, config)
    return {
        **_endpoint_policy(protocol),
        "name": task,
        "description": requirements,
        "instructions": instructions,
        "workspace": str(workspace),
        "read_paths": [
            "npa/src/" + target,
            "TASK.md",
            *[Path(p).name for p in CHECK_FILES],
        ],
        "write_paths": ["npa/src/" + target],
        "operations": _operations(
            root, path, protocol.get("native_failure_handoff", False)
        ),
        "required_operations": ["verify"],
        "compact_context": True,
    }


def _prepare_arms(root, options, protocol):
    for number, order in enumerate(protocol["pairs"], 1):
        for arm in order:
            directory = root / "pairs" / f"pair-{number}" / arm
            profiles = [
                _profile(root, directory, task, options, protocol) for task in TASKS
            ]
            config = TeamConfig.model_validate(
                {
                    "state_directory": str(directory / "state"),
                    "profiles": profiles,
                    "default_profile": "physics",
                    "router": "explicit",
                }
            )
            _write(directory / "team.json", config.model_dump(mode="json"))


def _calibration(root, options):
    directory = root / "calibration"
    workspace = directory / "reference"
    for target in TARGETS:
        destination = workspace / "npa/src" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "source" / target, destination)
    config = _operation_config(
        root, workspace, directory / "operations", TARGETS, options, True
    )
    _write(directory / "reference.json", config)


def _freeze(root, protocol):
    inputs = {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    _write(
        root / "freeze.json",
        {
            "schema": "npa.specialists.repair-benchmark.freeze.v1",
            "protocol": protocol,
            "runtime_commit": commit,
            "inputs": inputs,
            "model_inference_started": False,
            "common_prompt_sha256": hashlib.sha256(
                (root / "prompt.txt").read_bytes()
            ).hexdigest(),
        },
    )


def _runtime_versions(options):
    commands = {
        "simulation": (
            options.native_python,
            "import mujoco, gymnasium_robotics, PIL, av",
        ),
        "reader": (
            options.reader_python,
            "from lerobot.datasets.lerobot_dataset import LeRobotDataset; import torch; assert torch.version.cuda is None",
        ),
    }
    for name, (interpreter, imports) in commands.items():
        if not interpreter.is_file():
            raise ValueError(f"missing {name} interpreter")
        subprocess.run(
            [str(interpreter.absolute()), "-c", imports],
            check=True,
            capture_output=True,
        )


def _options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-python", required=True, type=Path)
    parser.add_argument("--reader-python", required=True, type=Path)
    parser.add_argument(
        "--native-failure-handoff",
        action="store_true",
        help="Hand a failed native wait to the next configured model; preserve receipts and do not replay effects.",
    )
    return parser.parse_args()


def _main():
    options = _options()
    _runtime_versions(options)
    root = options.output.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    protocol = json.loads((HERE / "protocol.json").read_text())
    if options.native_failure_handoff:
        protocol["native_failure_handoff"] = True
    _copy_inputs(root)
    prompt = "Repair all three Workbench regression lanes and complete their required operations.\n"
    prompt += "\n".join(
        name + ": " + requirements for name, (_, requirements) in TASKS.items()
    )
    (root / "prompt.txt").write_text(prompt + "\n" + PROCESS + "\n")
    _prepare_arms(root, options, protocol)
    _calibration(root, options)
    _freeze(root, protocol)
    print(
        json.dumps(
            {
                "status": "prepared",
                "pairs": len(protocol["pairs"]),
                "lanes": list(TASKS),
            }
        )
    )


if __name__ == "__main__":
    _main()

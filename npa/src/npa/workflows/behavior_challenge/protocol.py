"""Validate a fixed BEHAVIOR 2026 evaluation selection and upstream checkout."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import BinaryIO

UPSTREAM_COMMIT = "b1979916ec1549b10a4e65e630bc6504a9af1b00"
WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"
EVAL_DIRECTORY = Path("OmniGibson/omnigibson/eval")
SPLITS = {"development": tuple(range(10, 20)), "report": tuple(range(10))}


def stream_digest(stream: BinaryIO) -> str:
    """Compute SHA-256 from bounded reads on every supported Python version.

    Args:
        stream: Binary stream positioned at the first byte to hash.
    Returns:
        Hexadecimal SHA-256 digest of the remaining bytes.
    Raises:
        OSError: The stream cannot be read.
    """
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading videos into memory.

    Args:
        path: File to read.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: The file cannot be read.
    """
    with path.open("rb") as stream:
        return stream_digest(stream)


def verify_upstream(root: Path) -> None:
    """Require the unmodified official v3.9.2 checkout.

    Args:
        root: BEHAVIOR-1K source checkout.
    Returns:
        None.
    Raises:
        ValueError: The revision or tracked source differs from the official tag.
        subprocess.CalledProcessError: Git cannot inspect the checkout.
    """
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("BEHAVIOR evaluation requires the official v3.9.2 commit")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if changed.strip():
        raise ValueError("BEHAVIOR tracked source must be unmodified")


def _validate_recipe(recipe: dict, available: list[str]) -> tuple[list[str], str]:
    required = {"schema", "tasks", "split", "policy_checkpoint_sha256"}
    if (
        not isinstance(recipe, dict)
        or not required.issubset(recipe)
        or set(recipe) - required - {"policy_ports"}
    ):
        raise ValueError(
            f"Recipe requires {sorted(required)}; only policy_ports is optional"
        )
    if recipe["schema"] != "npa.behavior.recipe.v1":
        raise ValueError("Unsupported BEHAVIOR recipe schema")
    split = recipe["split"]
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError(
            "Use development or report; training and hidden splits are excluded"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(recipe["policy_checkpoint_sha256"])):
        raise ValueError("Provide the fixed policy checkpoint SHA-256")
    tasks = available if recipe["tasks"] == "all" else recipe["tasks"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(t, str) for t in tasks)
    ):
        raise ValueError("tasks must be all or a nonempty list of official task names")
    if len(set(tasks)) != len(tasks) or not set(tasks).issubset(available):
        raise ValueError("tasks contains duplicate or non-challenge task names")
    return tasks, split


def _policy_ports(recipe: dict, tasks: list[str]) -> dict[str, int]:
    ports = recipe.get("policy_ports", {})
    if not isinstance(ports, dict):
        raise ValueError(
            "policy_ports must map selected tasks to distinct serving ports"
        )
    if len(tasks) > 1 or ports:
        if set(ports) != set(tasks):
            raise ValueError(
                "Multiple tasks require one explicit policy port per selected task"
            )
        if any(
            type(port) is not int or not 1 <= port <= 65535 for port in ports.values()
        ):
            raise ValueError("Policy ports must be integers between 1 and 65535")
        if len(set(ports.values())) != len(ports):
            raise ValueError(
                "Each task needs a distinct task-configured policy endpoint"
            )
    return ports


def make_plan(recipe: dict, root: Path) -> dict:
    """Freeze tasks, public indices, and actual evaluator instance IDs before execution.

    Args:
        recipe: Fixed policy identity, split, and task selection.
        root: Verified BEHAVIOR checkout containing the official task registry.
    Returns:
        Plan with all prescribed cases and a full-challenge score denominator.
    Raises:
        ValueError: The recipe or upstream task registry is invalid.
        OSError: The upstream task registry cannot be read.
    """
    registry = json.loads((root / "docs/challenge/task_data.json").read_text())
    available = [task["id"] for task in registry["tasks"]]
    if len(available) != 100 or len(set(available)) != 100:
        raise ValueError("Expected the official 100-task registry")
    tasks, split = _validate_recipe(recipe, available)
    ports = _policy_ports(recipe, tasks)
    cases = [
        {
            "task": task,
            "index": index,
            "instance_id": 301 + index,
            "rollout_id": 0,
            "policy_port": ports.get(task),
        }
        for task in tasks
        for index in SPLITS[split]
    ]
    return {
        "schema": "npa.behavior.plan.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "wrapper": WRAPPER,
        "recipe": recipe,
        "cases": cases,
        "challenge_denominator": 1000,
        "eligible_for_reporting": split == "report",
    }


def evaluator_argv(
    case: dict, *, root: Path, python: str, host: str, port: int, output: Path
) -> list[str]:
    """Construct the official evaluator command without protocol overrides.

    Args:
        case: One prescribed task and instance index.
        root: Verified upstream checkout.
        python: Interpreter with OmniGibson installed from that checkout.
        host: Policy WebSocket host.
        port: Policy WebSocket port.
        output: Worker-local output directory.
    Returns:
        Argument vector preserving upstream timeout, seed, robot, and metrics.
    Raises:
        ValueError: The policy port is invalid.
    """
    port = case.get("policy_port") or port
    if not 1 <= port <= 65535:
        raise ValueError("Policy port must be between 1 and 65535")
    options = {
        "--task-name": case["task"],
        "--mode": "public_test",
        "--instance-indices": str(case["index"]),
        "--num-rollouts": "1",
        "--env-wrapper": WRAPPER,
        "--robot-config": str(root / EVAL_DIRECTORY / "r1pro.yaml"),
        "--host": host,
        "--port": str(port),
        "--output-dir": str(output),
    }
    argv = [python, "-m", "omnigibson.eval.eval"]
    for option, value in options.items():
        argv.extend((option, value))
    return argv + ["--write-video", "--headless"]

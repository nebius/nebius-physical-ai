"""Bind retained planner queries to operator requests or pinned benchmark bytes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import yaml

from .benchmark_inventory import BENCHMARK_GROUPS, DATASET_FILES
from .schemas import BenchmarkManifest, DATASET_REVISION, PlanManifest


class QueryBindingError(ValueError):
    """Retained queries do not establish the requested planning task."""


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def planner_query(problem: dict, *, benchmark: bool = False) -> dict:
    """Retain the full input consumed by the fixed Franka planning contract.

    Args:
        problem: Validated operator problem or pinned raw benchmark problem.
        benchmark: Whether obstacles use the benchmark's scene schema.
    Returns:
        JSON-compatible robot, start, goal and pre-OBB scene evidence.
    Raises:
        KeyError, TypeError, ValueError: Required input fields are malformed.
    """
    return {
        "robot": "franka.yml",
        "start": [float(value) for value in problem["start"]],
        "goal_pose": {
            field: [float(value) for value in problem["goal_pose"][field]]
            for field in ("position_xyz", "quaternion_wxyz")
        },
        "scene": deepcopy(problem["obstacles"])
        if benchmark
        else {"cuboid": problem.get("cuboids", {})},
    }


def _dataset_problems(dataset: str, root: Path) -> dict:
    filename, expected_hash = DATASET_FILES[dataset]
    payload = (root / "robometrics/content/dataset" / filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise QueryBindingError("benchmark YAML bytes differ from pinned inventory")
    groups = yaml.safe_load(payload)
    expected_groups = BENCHMARK_GROUPS[dataset]
    if not isinstance(groups, dict) or set(groups) != set(expected_groups):
        raise QueryBindingError("benchmark YAML groups differ from pinned inventory")
    result = {}
    for group, (count, excluded) in expected_groups.items():
        problems = groups[group]
        if not isinstance(problems, list) or len(problems) != count:
            raise QueryBindingError("benchmark YAML population differs")
        for index, problem in enumerate(problems):
            invalid = problem["collision_buffer_ik"] < 0
            if invalid != (index in excluded):
                raise QueryBindingError("benchmark YAML exclusions differ")
            result[(dataset, f"{group}/{index}")] = (
                None if invalid else planner_query(problem, benchmark=True)
            )
    return result


def benchmark_queries() -> dict:
    """Read expected queries independently of runner output and vendor loaders.

    Args:
        None.
    Returns:
        Pinned (dataset, problem ID) queries, with None for excluded problems.
    Raises:
        QueryBindingError: Pinned local dataset bytes are unavailable or invalid.
    """
    root = Path(os.environ.get("NPA_CUROBO_DATASET_SOURCE", "/opt/robometrics"))
    try:
        if (root / "NPA_SOURCE_REVISION").read_text().strip() != DATASET_REVISION:
            raise QueryBindingError("benchmark dataset revision differs")
        result = {}
        for dataset in DATASET_FILES:
            result.update(_dataset_problems(dataset, root))
        return result
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise QueryBindingError(
            "pinned benchmark query evidence is unavailable"
        ) from exc


def _requested_queries(report, manifest):
    if report.get("kind") == "plan":
        inputs = PlanManifest.model_validate(manifest).model_dump(mode="json")
        expected = {
            ("kinematic", "operator", problem["id"]): planner_query(problem)
            for problem in inputs["problems"]
        }
    elif report.get("kind") == "benchmark":
        inputs = BenchmarkManifest.model_validate(manifest).model_dump(mode="json")
        queries = benchmark_queries()
        expected = {
            (mode, dataset, problem_id): query
            for mode in inputs["modes"]
            for (dataset, problem_id), query in queries.items()
        }
    else:
        raise QueryBindingError("unknown planning request kind")
    if _canonical(inputs) != _canonical(manifest):
        raise QueryBindingError("retained input manifest is not canonical")
    return expected


def validate_query_binding(report: dict, rows: list[dict]) -> None:
    """Reject substituted queries even when journal hashes and IDs agree.

    Args:
        report: Result containing the canonical input_manifest and input_sha256.
        rows: Retained planner journal records.
    Returns:
        None.
    Raises:
        QueryBindingError: Request identity, digest or any executed query differs.
    """
    try:
        manifest = report["input_manifest"]
        digest = hashlib.sha256(_canonical(manifest)).hexdigest()
        if report.get("input_sha256") != digest:
            raise QueryBindingError("input manifest digest differs")
        expected = _requested_queries(report, manifest)
        observed = [(row["mode"], row["dataset"], row["problem_id"]) for row in rows]
        if len(observed) != len(set(observed)) or set(observed) != set(expected):
            raise QueryBindingError("requested query population differs")
        for identity, row in zip(observed, rows):
            query = expected[identity]
            if (row["status"] == "invalid") != (query is None):
                raise QueryBindingError("requested query exclusion differs")
            if _canonical(row.get("query")) != _canonical(query):
                raise QueryBindingError("executed query differs from requested input")
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryBindingError(f"planner request binding failed: {exc}") from exc

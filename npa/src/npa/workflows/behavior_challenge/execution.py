"""Execute prescribed BEHAVIOR rollouts once and publish original evidence to S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

from .artifacts import (
    build_submission,
    inspect_rollout,
    snapshot_evaluator,
    write_summary,
)
from .protocol import evaluator_argv, file_digest, make_plan, verify_upstream
from .policy import managed_policy


def _s3_location(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("Provide an S3 bucket and nonempty task-scoped key or prefix")
    return parsed.netloc, parsed.path.lstrip("/")


def _upload_verified(storage: StorageClient, path: Path, uri: str) -> None:
    bucket, key = _s3_location(uri)
    storage.upload_file(str(path), uri)
    body = storage.s3.get_object(Bucket=bucket, Key=key)["Body"]
    digest = hashlib.sha256()
    try:
        for chunk in body.iter_chunks(chunk_size=1024 * 1024):
            digest.update(chunk)
    finally:
        body.close()
    if digest.hexdigest() != file_digest(path):
        raise ValueError("Uploaded artifact failed SHA-256 readback verification")


def _publish_policy_log(storage, path, uri, published):
    # The server can append while an upload is running; verify a stable snapshot.
    with tempfile.TemporaryDirectory(prefix="npa-behavior-log-") as directory:
        snapshot = Path(directory) / "policy.log"
        shutil.copyfile(path, snapshot)
        digest = file_digest(snapshot)
        if published.get("policy.log") != digest:
            _upload_verified(storage, snapshot, f"{uri.rstrip('/')}/policy.log")
            published["policy.log"] = digest


def _publish(storage: StorageClient, output: Path, uri: str, published: dict) -> None:
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative == "policy.log":
            _publish_policy_log(storage, path, uri, published)
            continue
        if relative in published and relative not in {
            "attempts.json",
            "summary.json",
        }:
            continue
        digest = file_digest(path)
        if published.get(relative) == digest:
            continue
        _upload_verified(storage, path, f"{uri.rstrip('/')}/{relative}")
        published[relative] = digest


def _runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    data_root = Path(args.data_root).resolve()
    metadata = data_root / "2026-challenge-task-instances/metadata/B100_task_misc.csv"
    if not metadata.is_file():
        raise ValueError(
            "Install authorized BEHAVIOR assets before evaluation; task metadata is missing"
        )
    environment = dict(os.environ)
    environment["OMNIGIBSON_DATA_PATH"] = str(data_root)
    source = args.upstream_root / "OmniGibson"
    environment["PYTHONPATH"] = (
        str(source) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    probe = (
        "import importlib.util; print(importlib.util.find_spec('omnigibson').origin)"
    )
    origin = subprocess.check_output(
        [args.evaluator_python, "-c", probe],
        env=environment,
        text=True,
    ).strip()
    if Path(origin).resolve() != source / "omnigibson/__init__.py":
        raise ValueError(
            "Evaluator interpreter does not resolve the pinned OmniGibson source"
        )
    return environment


def _write_instructions(output: Path, plan: dict, commands: list[list[str]]) -> None:
    text = [
        "# BEHAVIOR 2026 evaluation evidence",
        "",
        f"Upstream commit: {plan['upstream_commit']}",
        f"Policy checkpoint SHA-256 (operator-declared): {plan['recipe']['policy_checkpoint_sha256']}",
        f"Split: {plan['recipe']['split']}",
        "",
        "Run the policy server described in policy.md before these commands.",
        "The default R1Pro robot and RGBDFullResWrapper were used unchanged.",
        "Original evaluator sources, robot configuration, and MIT license are in evaluator/.",
        "Videos are in videos/ and must be linked separately in the submission portal.",
        "summary.json is an NPA artifact check, not an official leaderboard result.",
        "Policy identity, observation compliance, and 24 GB serving require independent review.",
        "",
        "## Exact evaluator commands",
        "",
        "```bash",
        *(shlex.join(command) for command in commands),
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(text))


def _commands(args: argparse.Namespace, plan: dict, output: Path) -> list[list[str]]:
    return [
        evaluator_argv(
            case,
            root=args.upstream_root,
            python=args.evaluator_python,
            host=args.host,
            port=args.port,
            output=output,
        )
        for case in plan["cases"]
    ]


def _run_case(
    command: list[str],
    args: argparse.Namespace,
    output: Path,
    case: dict,
    environment: dict,
) -> dict:
    log = output / f"{case['task']}_{case['instance_id']}.log"
    with log.open("wb") as stream:
        subprocess.run(
            command,
            cwd=args.upstream_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return inspect_rollout(output, case)


def _evaluate_cases(
    args: argparse.Namespace,
    plan: dict,
    output: Path,
    storage: StorageClient,
    environment: dict,
    published: dict,
) -> dict:
    records, attempts = [], []
    commands = _commands(args, plan, output)
    _write_instructions(output, plan, commands)
    try:
        for case, command in zip(plan["cases"], commands, strict=True):
            attempts.append(case)
            (output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
            _publish(storage, output, args.output_path, published)
            records.append(_run_case(command, args, output, case, environment))
            write_summary(output, plan, records)
            _publish(storage, output, args.output_path, published)
        verify_upstream(args.upstream_root)
        summary = write_summary(output, plan, records)
        if plan["eligible_for_reporting"]:
            build_submission(output, plan, records)
        return summary
    finally:
        write_summary(output, plan, records)
        _publish(storage, output, args.output_path, published)


def _execute_with_policy(args, plan, output, storage, environment):
    published = {}
    write_summary(output, plan, [])
    try:
        with managed_policy(args, plan, output):
            return _evaluate_cases(args, plan, output, storage, environment, published)
    finally:
        _publish(storage, output, args.output_path, published)


def _prepare_plan(
    args: argparse.Namespace, storage: StorageClient, output: Path
) -> dict:
    storage.download_file(args.input_path, str(output / "recipe.json"))
    recipe = json.loads((output / "recipe.json").read_bytes())
    plan = make_plan(recipe, args.upstream_root)
    storage.download_file(args.policy_readme_uri, str(output / "policy.md"))
    if not (output / "policy.md").read_text().strip():
        raise ValueError(
            "Provide policy serving and checkpoint reproduction instructions"
        )
    payload = (json.dumps(plan, indent=2) + "\n").encode()
    storage.put_bytes_conditional(
        payload,
        args.output_path.rstrip("/") + "/claim.json",
        if_none_match=True,
        content_type="application/json",
    )
    (output / "plan.json").write_bytes(payload)
    snapshot_evaluator(args.upstream_root, output)
    return plan


def evaluate(args: argparse.Namespace) -> dict:
    """Run a predeclared selection once using an operator-prepared licensed runtime.

    Args:
        args: Worker configuration parsed by the workflow module.
    Returns:
        Summary of validated original evaluator artifacts.
    Raises:
        ValueError: Recipe, source, runtime, or artifact validation fails.
        StorageError: S3 access fails or this output prefix was already claimed.
        subprocess.CalledProcessError: The official evaluator fails.
    """
    _s3_location(args.input_path)
    _s3_location(args.output_path)
    _s3_location(args.policy_readme_uri)
    args.upstream_root = args.upstream_root.resolve()
    verify_upstream(args.upstream_root)
    environment = _runtime_environment(args)
    storage = StorageClient.from_environment()
    output = Path(tempfile.mkdtemp(prefix="npa-behavior-"))
    completed = False
    try:
        plan = _prepare_plan(args, storage, output)
        result = _execute_with_policy(args, plan, output, storage, environment)
        completed = True
        return result
    finally:
        if completed:
            shutil.rmtree(output)
        else:
            print(f"BEHAVIOR local evidence retained at {output}", file=sys.stderr)

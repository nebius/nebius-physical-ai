"""Supervise the official radio-task policy and verify its loaded checkpoint bytes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from npa.workflows.byof.openpi import require_openpi_terms

from .protocol import file_digest, stream_digest

OPENPI_COMMIT = "0cc8e355f7bac0976db1cc3139b1ff0379feea60"
CHECKPOINT_PREFIX = "pi05_turn_on_the_radio/"
POLICY_FIELDS = ("policy_root", "policy_python", "policy_checkpoint", "policy_archive")


def _verify_source(root: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != OPENPI_COMMIT or dirty:
        raise ValueError(
            "Local policy requires the unchanged pinned BEHAVIOR OpenPI fork"
        )


def _verify_checkpoint(archive: Path, checkpoint: Path, expected: str) -> dict:
    if file_digest(archive) != expected:
        raise ValueError("Policy archive differs from the frozen recipe checkpoint")
    files = {}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            relative = member.filename.removeprefix(CHECKPOINT_PREFIX)
            target = (checkpoint / relative).resolve()
            if (
                relative == member.filename
                or not target.is_relative_to(checkpoint.resolve())
                or relative in files
            ):
                raise ValueError("Unexpected checkpoint archive layout")
            with source.open(member) as stream:
                digest = stream_digest(stream)
            if file_digest(target) != digest:
                raise ValueError(
                    "Loaded checkpoint bytes differ from the frozen archive"
                )
            files[relative] = digest
    actual = {
        str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()
    }
    if actual != set(files) or "assets/turning_on_radio/norm_stats.json" not in files:
        raise ValueError("Checkpoint files or radio normalization assets do not match")
    return files


def _policy_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.policy_python),
        "scripts/b1k/serve_b1k.py",
        "--robot",
        "b1k/R1Pro",
        "--task",
        "b1k/turning_on_radio",
        "--repo-id",
        "turning_on_radio",
        "--policy.config",
        "pi05_b1k",
        "--policy.dir",
        str(args.policy_checkpoint),
        "--control-mode",
        "receding_horizon",
        "--action-horizon",
        "16",
        "--port",
        str(args.port),
    ]


def _healthy(port: int) -> bool:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/healthz")
        return connection.getresponse().status == 200
    except (OSError, HTTPException):
        return False
    finally:
        connection.close()


def _wait_for_policy(process: subprocess.Popen, port: int) -> None:
    while process.poll() is None:
        if _healthy(port):
            return
        time.sleep(1)
    raise RuntimeError("Official policy exited before readiness; inspect policy.log")


def _stop_policy(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _prepare_policy(args: argparse.Namespace, plan: dict, output: Path) -> list[str]:
    require_openpi_terms()
    if args.host not in {"localhost", "127.0.0.1"}:
        raise ValueError("Managed policy requires a loopback evaluator host")
    if plan["recipe"]["tasks"] != ["turning_on_radio"]:
        raise ValueError(
            "The supplied official checkpoint supports turning_on_radio only"
        )
    if any(case.get("policy_port") not in {None, args.port} for case in plan["cases"]):
        raise ValueError("Managed policy port must match every planned case")
    _verify_source(args.policy_root)
    files = _verify_checkpoint(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
    )
    if _healthy(args.port):
        raise ValueError(
            "Policy port is already serving; refusing an unrelated endpoint"
        )
    command = _policy_command(args)
    evidence = {
        "schema": "npa.behavior.policy.v1",
        "source_commit": OPENPI_COMMIT,
        "checkpoint_archive_sha256": plan["recipe"]["policy_checkpoint_sha256"],
        "checkpoint_files": files,
        "command": command,
        "normalization_asset": "turning_on_radio",
        "memory_compliance": "unverified",
    }
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    return command


@contextmanager
def managed_policy(args: argparse.Namespace, plan: dict, output: Path):
    """Optionally serve the verified official radio checkpoint during evaluation.

    Args:
        args: Evaluator arguments with all four policy paths, or none.
        plan: Frozen single-task evaluation selection.
        output: Evidence directory; weights remain outside this directory.
    Returns:
        Context manager that stops its policy process on success or failure.
    Raises:
        ValueError: Consent, source, checkpoint, task, or endpoint validation fails.
        RuntimeError: The policy process exits before becoming ready.
        OSError: Files or the policy interpreter are unavailable.
    """
    selected = [bool(getattr(args, field, None)) for field in POLICY_FIELDS]
    if not any(selected):
        yield
        return
    if not all(selected):
        raise ValueError("Managed policy requires all four policy paths")
    command = _prepare_policy(args, plan, output)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    environment.pop("PYTHONPATH", None)
    with (output / "policy.log").open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=args.policy_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            print("Waiting for the official radio policy to become ready.", flush=True)
            _wait_for_policy(process, args.port)
            print("Official radio policy is ready.", flush=True)
            yield
        finally:
            _stop_policy(process)

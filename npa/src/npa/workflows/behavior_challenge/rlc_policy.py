"""Verify and supervise a pinned published RLC checkpoint on the 2026 evaluator."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .protocol import file_digest

SOURCE_COMMIT = "ca556f74a455cef7987a2be4537b5ac85cc56dd7"
OPENPI_COMMIT = "01177e0242a1c7e8fad2547caa0e987def614cda"
BEHAVIOR_COMMIT = "684a83050ddd398de231e6aa7fc605bc34458d4b"
MODEL_REVISION = "89545bc1b7aa7f2e687bc0032d091f132d715d4e"
NORMALIZATION = "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json"


def _verify_checkout(root: Path, expected: str) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if revision != expected or changes:
        raise ValueError("RLC source must match the clean pinned checkout")


def _task_checkpoint(root: Path, upstream: Path, task: str) -> tuple[int, str]:
    old = json.loads((root / "BEHAVIOR-1K/docs/challenge/task_data.json").read_text())
    new = json.loads((upstream / "docs/challenge/task_data.json").read_text())
    names = [item["id"] for item in old["tasks"]]
    current = [item["id"] for item in new["tasks"]]
    if len(names) != 50 or names != current[:50] or task not in names:
        raise ValueError("Published RLC checkpoints support the original 50 tasks only")
    task_id = names.index(task)
    mapping = json.loads((root / "task_checkpoint_mapping.json").read_text())
    matches = [
        name for name, row in mapping["checkpoints"].items() if task_id in row["tasks"]
    ]
    if len(matches) != 1:
        raise ValueError("RLC task must map to exactly one checkpoint")
    return task_id, matches[0]


def _verify_published_files(root: Path, checkpoint: str, files: dict) -> None:
    manifest = json.loads(Path(__file__).with_name("rlc-checkpoints.json").read_text())
    expected = manifest["checkpoints"][checkpoint]
    if manifest["revision"] != MODEL_REVISION or set(files) != set(expected):
        raise ValueError("Checkpoint file set differs from the pinned published model")
    for relative, identity in expected.items():
        path = root / relative
        if path.stat().st_size != identity["size"]:
            raise ValueError("Published checkpoint file size differs")
        if "sha256" in identity:
            matches = files[relative] == identity["sha256"]
        else:
            digest = hashlib.sha1(
                f"blob {identity['size']}\0".encode(), usedforsecurity=False
            )
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            matches = digest.hexdigest() == identity["git_blob_sha1"]
        if not matches:
            raise ValueError("Checkpoint bytes differ from the pinned published model")


def _record(output: Path, command: list[str], files: dict, checkpoint: str, plan: dict):
    adapters = {}
    for name in ("rlc_server.py", "rlc_observations.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
        adapters[name] = file_digest(output / name)
    shutil.copyfile(
        Path(__file__).with_name("POLICY_LICENSE"), output / "rlc-adapter.LICENSE"
    )
    evidence = {
        "schema": "npa.behavior.policy.v1",
        "kind": "rlc",
        "source_commit": SOURCE_COMMIT,
        "openpi_commit": OPENPI_COMMIT,
        "model_repository": "IliaLarchenko/behavior_submission",
        "published_model_revision": MODEL_REVISION,
        "checkpoint": checkpoint,
        "checkpoint_files": files,
        "checkpoint_archive_sha256": plan["recipe"]["policy_checkpoint_sha256"],
        "adapters": adapters,
        "command": command,
        "normalization_asset": NORMALIZATION,
        "memory_compliance": "unverified",
        "control": "published stage voting, rolling inpainting, compression and recovery",
        "transfer_limitation": "2025 training base velocity differs from 2026 robot-local velocity",
    }
    (output / "policy-provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


def _verify_task(args, plan):
    tasks = plan["recipe"]["tasks"]
    if (
        plan["recipe"]["split"] != "development"
        or not isinstance(tasks, list)
        or len(tasks) != 1
    ):
        raise ValueError("RLC transfer currently requires one development task")
    for relative, revision in (
        (".", SOURCE_COMMIT),
        ("openpi", OPENPI_COMMIT),
        ("BEHAVIOR-1K", BEHAVIOR_COMMIT),
    ):
        _verify_checkout(args.policy_root / relative, revision)
    return _task_checkpoint(args.policy_root, args.upstream_root, tasks[0])


def _verify_weights(args, plan, checkpoint):
    from .policy import _verify_checkpoint

    files = _verify_checkpoint(
        args.policy_archive,
        args.policy_checkpoint,
        plan["recipe"]["policy_checkpoint_sha256"],
        prefix=checkpoint + "/",
        normalization=NORMALIZATION,
    )
    _verify_published_files(args.policy_checkpoint, checkpoint, files)
    return files


def _command(args, task_id, output):
    return [
        str(args.policy_python),
        str(output / "rlc_server.py"),
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--task-id",
        str(task_id),
        "--port",
        str(args.port),
    ]


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Validate the RLC source, task, archive and launcher before policy execution.

    Args:
        args: Managed policy paths and loopback endpoint.
        plan: Frozen one-task development recipe.
        output: Evidence directory receiving launcher sources and provenance.
    Returns:
        Policy interpreter command using the published control settings.
    Raises:
        ValueError: Source, checkpoint, task, endpoint or development scope is invalid.
        OSError: A required source or checkpoint cannot be read.
    """
    from .policy import _healthy

    task_id, checkpoint = _verify_task(args, plan)
    files = _verify_weights(args, plan, checkpoint)
    if _healthy(args.port) or any(
        case.get("policy_port") not in {None, args.port} for case in plan["cases"]
    ):
        raise ValueError("Managed RLC policy requires its own matching loopback port")
    command = _command(args, task_id, output)
    _record(output, command, files, checkpoint, plan)
    return command

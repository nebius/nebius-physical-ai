"""Run the two-update RTX qualification and local reload under one Linux cgroup deadline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

LIMIT_SECONDS = 45 * 60
RECIPE = Path(__file__).resolve().parent


def _require_plan(settings):
    if (
        settings.get("launcher") != "torchrun"
        or settings.get("hardware", {}).get("label")
        != "RTX PRO 6000 Blackwell Server Edition"
        or tuple(
            settings.get(k)
            for k in (
                "nodes",
                "gpus",
                "gpus_per_node",
                "samples_per_rank",
                "gradient_accumulation",
                "global_batch",
                "steps",
            )
        )
        != (1, 8, 8, 64, 4, 2048, 2)
    ):
        raise ValueError("expected the fixed two-update single-host RTX qualification")


def _reload(run, python, root, settings, evidence, training):
    subprocess.run(
        [
            python,
            RECIPE / "evaluate.py",
            "--shared-root",
            root,
            "--run-dir",
            run,
            "--step",
            "2",
            "--seed",
            str(settings["seed"]),
            "--workers",
            "1",
            "--reload-only",
            "--output-dir",
            evidence / "reload",
        ],
        check=True,
    )
    reload = json.loads((evidence / "reload/reload.json").read_text())
    if reload["step"] != 2 or reload["loaded_sha256"] != training["checkpoint_sha256"]:
        raise ValueError("local policy reload disagrees with the update-2 checkpoint")
    return reload["loaded_sha256"]


def _worker(run, unit):
    memberships = Path("/proc/self/cgroup").read_text().splitlines()
    if not any(line.rsplit("/", 1)[-1] == unit for line in memberships):
        raise RuntimeError("qualification worker requires its assigned systemd scope")
    settings_bytes = (run / "run.json").read_bytes()
    settings = json.loads(settings_bytes)
    root = Path(settings["shared_root"])
    python = str(root / "framework/.venv/bin/python")
    _require_plan(settings)
    evidence = run / "qualification"
    started = time.monotonic()
    subprocess.run(
        [python, run / "recipe.py", "run-local", "--run-dir", run], check=True
    )
    after_train = time.monotonic()
    report_path = evidence / "native-training.json"
    with report_path.open("x") as report:
        subprocess.run(
            [python, RECIPE / "smoke_report.py", "--run-dir", run],
            stdout=report,
            check=True,
        )
    report_bytes = report_path.read_bytes()
    training = json.loads(report_bytes)
    run_hash = hashlib.sha256(settings_bytes).hexdigest()
    if training["run_sha256"] != run_hash:
        raise ValueError("training evidence belongs to a different plan")
    before_reload = time.monotonic()
    loaded = _reload(run, python, root, settings, evidence, training)
    result = {
        "run_sha256": run_hash,
        "checkpoint_sha256": loaded,
        "source_training_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "launch_through_train_exit_seconds": after_train - started,
        "smoke_verification_seconds": before_reload - after_train,
        "reload_wall_seconds": time.monotonic() - before_reload,
    }
    with (evidence / "verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)


def _scope_command(unit, run, limit):
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        unit,
        f"--property=RuntimeMaxSec={limit}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGKILL",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run),
        "--scope-unit",
        unit,
    ]


def _qualify(run, limit):
    if not 0 < limit <= LIMIT_SECONDS:
        raise ValueError("qualification deadline must be between 1 and 2700 seconds")
    run = run.resolve(strict=True)
    evidence = run / "qualification"
    evidence.mkdir(mode=0o700)
    # A user systemd scope also owns torchrun children that start new sessions.
    # The supervisor remains outside it so even a hard timeout leaves a receipt.
    unit = f"cosmos-qualification-{uuid.uuid4().hex}.scope"
    command = _scope_command(unit, run, limit)
    started = time.monotonic()
    result = {"status": "failed", "deadline_seconds": limit, "scope_unit": unit}
    try:
        with (evidence / "qualification.log").open("x") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result["scope_returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(
                f"qualification scope exited {child.returncode}; see {log.name}"
            )
        result.update(json.loads((evidence / "verified.json").read_text()))
        if time.monotonic() - started > limit:
            raise TimeoutError("qualification exceeded wall-clock deadline")
        result["status"] = "passed"
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        with (evidence / "qualification.json").open("x") as stream:
            json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--deadline-seconds", default=LIMIT_SECONDS, type=int)
    parser.add_argument("--scope-unit", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.scope_unit:
        _worker(args.run_dir, args.scope_unit)
    else:
        print(json.dumps(_qualify(args.run_dir, args.deadline_seconds)))

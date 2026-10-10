"""Run bounded batches through the standard durable workflow submit command."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import time
from typing import Callable

from npa.cli.invocation import internal_cli_argv
from npa.orchestration.npa_workflow.batch_plan import plan_batch
from npa.orchestration.npa_workflow.src_staging import (
    find_npa_package_root,
    source_fingerprint,
)


def _private_open(path: Path, mode: str = "w"):
    def opener(name, flags):
        descriptor = os.open(
            name, (flags & ~os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            os.close(descriptor)
            raise ValueError("batch ledger files must be owned regular files")
        if flags & (os.O_WRONLY | os.O_RDWR):
            os.fchmod(descriptor, 0o600)
        if flags & os.O_TRUNC:
            os.ftruncate(descriptor, 0)
        return descriptor

    return open(path, mode, opener=opener)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with _private_open(temporary) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _batch_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _validate_directory(directory)
    with _private_open(directory / "batch.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this batch already has an active local driver") from exc
        yield lock.fileno()


def _validate_directory(directory):
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("batch ledger must be an owned directory, not a symlink")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("batch ledger must have owner-only permissions (0700)")


def _initialize(plan: dict, directory: Path, resume: bool) -> dict:
    plan_path = directory / "plan.json"
    state_path = directory / "state.json"
    if plan_path.exists():
        saved = json.loads(_read_private(plan_path))
        if (
            not isinstance(saved, dict)
            or saved.get("fingerprint") != plan["fingerprint"]
        ):
            raise ValueError(
                "batch identity changed; use a new batch_id and output prefix"
            )
        if not resume:
            raise ValueError("batch already exists; inspect it and use --resume")
        if not state_path.exists():
            raise ValueError(
                "batch state is missing; prior launch absence cannot be proven"
            )
    else:
        if resume:
            raise ValueError("cannot resume a batch without its original local plan")
        if state_path.exists():
            raise ValueError(
                "batch plan is missing; retain and reconcile the original state"
            )
        _write_json(plan_path, plan)
    _stage_workflow_snapshot(plan, directory)
    if state_path.exists():
        return _read_state(state_path, plan)
    state = {
        "schema": "npa.workflow.batch-state.v1",
        "batch_id": plan["batch_id"],
        "fingerprint": plan["fingerprint"],
        "runs": {run["run_id"]: {"status": "pending"} for run in plan["runs"]},
    }
    _write_json(state_path, state)
    return state


def _read_private(path: Path) -> str:
    with _private_open(path, "r") as stream:
        return stream.read()


def _stage_workflow_snapshot(plan: dict, directory: Path) -> None:
    content = Path(plan["workflow"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != plan["workflow_sha256"]:
        raise ValueError("workflow changed during planning")
    with _private_open(directory / "workflow.yaml", "wb") as stream:
        stream.write(content)
    if plan["config_path"]:
        config = Path(plan["config_path"]).read_bytes()
        if hashlib.sha256(config).hexdigest() != plan["config_sha256"]:
            raise ValueError("SkyPilot config changed during planning")
        with _private_open(directory / "skypilot-config.yaml", "wb") as stream:
            stream.write(config)


def _read_state(path: Path, plan: dict) -> dict:
    state = json.loads(_read_private(path))
    expected = {run["run_id"] for run in plan["runs"]}
    if not isinstance(state, dict) or state.get("fingerprint") != plan["fingerprint"]:
        raise ValueError("batch state does not match the original plan")
    records = state.get("runs")
    if not isinstance(records, dict) or set(records) != expected:
        raise ValueError("batch state has missing or unexpected run identities")
    allowed = {"pending", "succeeded", "reconcile-required"}
    if any(
        not isinstance(row, dict) or row.get("status") not in allowed
        for row in records.values()
    ):
        raise ValueError("batch state contains an invalid status")
    for run_id, record in records.items():
        _validate_record(run_id, record)
    return state


def _validate_record(run_id, record):
    if record["status"] == "pending":
        if record != {"status": "pending"}:
            raise ValueError("pending batch record contains prior launch evidence")
        return
    attempt = record.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("batch record has an invalid attempt")
    basename = f"{run_id}-attempt-{attempt}"
    if (
        record.get("stdout") != f"{basename}.json"
        or record.get("stderr") != f"{basename}.log"
    ):
        raise ValueError("batch record logs do not match its run identity")


def _argv(plan: dict, run: dict, resume: bool) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "submit",
        plan["workflow"],
        "--project",
        plan["project"],
        "--runtime",
        "--no-deploy-if-absent",
        "--max-wait-seconds",
        "0",
        "--no-auto-load",
        "--output-format",
        "json",
        "--resume-run" if resume else "--run-id",
        run["run_id"],
        *run["input_args"],
    ]
    for key, value in run["vars"].items():
        args.extend(["--var", f"{key}={value}"])
    for key, flag in (("infra", "--infra"), ("config_path", "--config-path")):
        if plan[key]:
            args.extend([flag, plan[key]])
    for name in plan["secret_env"]:
        args.extend(["--secret-env", name])
    return internal_cli_argv(args)


def _launch(plan: dict, run: dict, directory: Path, state: dict, lock_fd: int):
    run_id = run["run_id"]
    previous = state["runs"][run_id]
    resume = previous["status"] != "pending"
    attempt = previous.get("attempt", 0) + 1
    basename = f"{run_id}-attempt-{attempt}"
    state["runs"][run_id] = {
        "status": "reconcile-required",
        "attempt": attempt,
        "stdout": f"{basename}.json",
        "stderr": f"{basename}.log",
    }
    _write_json(directory / "state.json", state)
    with _private_open(directory / f"{basename}.json") as stdout:
        with _private_open(directory / f"{basename}.log") as stderr:
            snapshot_plan = plan | {"workflow": str(directory / "workflow.yaml")}
            if plan["config_path"]:
                snapshot_plan["config_path"] = str(directory / "skypilot-config.yaml")
            return subprocess.Popen(
                _argv(snapshot_plan, run, resume),
                stdout=stdout,
                stderr=stderr,
                pass_fds=(lock_fd,),
            )


def _validate_source(plan):
    if source_fingerprint(find_npa_package_root()) != plan["source_sha256"]:
        raise ValueError(
            "NPA source changed during this batch; retain the original checkout"
        )


def _finish(process, run_id: str, directory: Path, state: dict) -> bool:
    record = state["runs"][run_id]
    try:
        result = json.loads(_read_private(directory / record["stdout"]))
    except (OSError, ValueError):
        result = {}
    succeeded = (
        process.returncode == 0
        and isinstance(result, dict)
        and result.get("status") == "succeeded"
        and result.get("run_id") == run_id
    )
    record.update(
        status="succeeded" if succeeded else "reconcile-required",
        returncode=process.returncode,
    )
    _write_json(directory / "state.json", state)
    return succeeded


def _drive(
    plan: dict,
    runs: list[dict],
    directory: Path,
    state: dict,
    concurrency: int,
    report: Callable,
    lock_fd: int,
):
    pending = iter(runs)
    active = {}
    admitting = True
    try:
        while True:
            # Keep an admission cohort intact until every member has
            # reconciled.  Refilling one completed slot while a sibling is
            # still unresolved can admit new work immediately before that
            # sibling reports a failure.
            if admitting and not active:
                _validate_source(plan)
                while admitting and len(active) < concurrency:
                    run = next(pending, None)
                    if run is None:
                        admitting = False
                        break
                    active[run["run_id"]] = _launch(
                        plan, run, directory, state, lock_fd
                    )
                    report(
                        f"Started {run['run_id']}; active workflow runs: {len(active)}"
                    )
            if not active:
                return
            time.sleep(0.2)
            for run_id, process in list(active.items()):
                if process.poll() is None:
                    continue
                if not _finish(process, run_id, directory, state):
                    admitting = False
                del active[run_id]
                report(f"{run_id}: {state['runs'][run_id]['status']}")
    finally:
        for process in active.values():
            if process.poll() is None:
                process.kill()
            process.wait()


def run_batch(
    manifest_path: Path,
    *,
    state_dir: Path,
    max_concurrent_runs: int,
    resume: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> dict:
    """Run a manifest with a caller-selected ceiling on active workflow clients.

    Args:
        manifest_path: Private batch YAML.
        state_dir: Durable local directory retained across driver restarts.
        max_concurrent_runs: Positive episode/workflow concurrency, not GPU count.
        resume: Reuse the original IDs through standard workflow resume.
        reporter: Optional progress sink.
    Returns:
        Local observation of each run; non-success requires reconciliation.
    Raises:
        ValueError: Invalid plan, changed identity, or concurrent local driver.
        OSError: Local persistence or child startup fails.
        KeyboardInterrupt: Interrupted clients leave remote jobs for reconciliation.
    """
    if type(max_concurrent_runs) is not int or max_concurrent_runs < 1:
        raise ValueError("max_concurrent_runs must be a positive integer")
    plan = plan_batch(manifest_path)
    directory = state_dir.resolve() / plan["batch_id"]
    with _batch_lock(directory) as lock_fd:
        state = _initialize(plan, directory, resume)
        _run_phases(
            plan,
            directory,
            state,
            max_concurrent_runs,
            reporter or (lambda _: None),
            lock_fd,
        )
        return state


def _run_phases(plan, directory, state, concurrency, report, lock_fd):
    unresolved = [
        run
        for run in plan["runs"]
        if state["runs"][run["run_id"]]["status"] == "reconcile-required"
    ]
    if len(unresolved) > concurrency:
        raise ValueError(
            "reconcile existing runs before lowering concurrency below their count"
        )
    completed = [
        run
        for run in plan["runs"]
        if state["runs"][run["run_id"]]["status"] == "succeeded"
    ]
    _drive(plan, unresolved, directory, state, concurrency, report, lock_fd)
    if any(
        record["status"] == "reconcile-required" for record in state["runs"].values()
    ):
        return
    _drive(plan, completed, directory, state, concurrency, report, lock_fd)
    if any(row["status"] == "reconcile-required" for row in state["runs"].values()):
        return
    pending = [
        run
        for run in plan["runs"]
        if state["runs"][run["run_id"]]["status"] == "pending"
    ]
    _drive(plan, pending, directory, state, concurrency, report, lock_fd)


def batch_status(state_dir: Path, batch_id: str) -> dict:
    """Read local batch observations without contacting or changing remote jobs.

    Args:
        state_dir: Directory used by run_batch.
        batch_id: Exact batch identifier.
    Returns:
        Last persisted state; reconcile-required does not mean remotely failed.
    Raises:
        ValueError: The batch identifier is unsafe.
        OSError: The state file is unavailable.
    """
    from pydantic import TypeAdapter
    from npa.orchestration.npa_workflow.batch_plan import Name

    TypeAdapter(Name).validate_python(batch_id)
    directory = state_dir.resolve() / batch_id
    _validate_directory(directory)
    plan = json.loads(_read_private(directory / "plan.json"))
    return _read_state(directory / "state.json", plan)

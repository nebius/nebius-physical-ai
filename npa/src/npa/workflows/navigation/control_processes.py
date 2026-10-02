"""Run reference controls sequentially in independent native child processes."""

import json
import os
import signal
import subprocess
import time

from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.control_protocol import (
    ARMS,
    accept_controls,
    create_request,
)


def check_native_log(path):
    """Reject the same native PhysX errors for every control and final process.

    Args:
        path: Complete child stdout/stderr log.
    Returns:
        None.
    Raises:
        RuntimeError: Native physics reported invalid interactions.
    """
    log = path.read_text()
    if "PhysX error:" in log or "simulation will miss interactions" in log:
        raise RuntimeError(
            "Isaac reported invalid physics; inspect private runtime.log"
        )


def run_native_process(argv, folder, arm, request_sha256):
    """Retain a sequential native child's complete log and exit provenance.

    Args:
        argv: Native interpreter invocation without a shell.
        folder: This child's fresh output directory.
        arm: Control name or main for the final workload.
        request_sha256: Same-attempt request digest.
    Returns:
        None after a successful native exit and clean physics log.
    Raises:
        RuntimeError: Native physics reports invalid interactions.
        subprocess.CalledProcessError: The native child exits unsuccessfully.
        OSError: Native process or its evidence cannot be created.
        KeyboardInterrupt: Propagated after terminating and waiting for the owned child.
    """
    log_path = folder / "runtime.log"
    started = time.monotonic_ns()
    with log_path.open("x") as log:
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            result = process.wait()
        except BaseException as interruption:
            _retain_interruption(
                process, log, folder, arm, request_sha256, started, interruption
            )
            raise
    _process_receipt(process, folder, arm, request_sha256, started)
    check_native_log(log_path)
    if result:
        raise subprocess.CalledProcessError(result, argv)


def _signal_owned_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass  # The child/session already exited; no other process is targeted.


def _terminate_owned_process(process):
    if process.returncode is not None:
        return  # Already reaped: never signal a potentially recycled process-group ID.
    signum = signal.SIGTERM
    while True:
        try:
            _signal_owned_group(process, signum)
            process.wait()
            return
        except KeyboardInterrupt:
            if process.returncode is not None:
                return
            # A repeated interruption requests immediate cleanup, never abandonment.
            signum = signal.SIGKILL


def _retain_interruption(process, log, folder, arm, digest, started, interruption):
    try:
        _terminate_owned_process(process)
    except BaseException as cleanup_error:
        interruption.add_note(
            f"Owned native child cleanup failed: {type(cleanup_error).__name__}"
        )
    try:
        log.flush()
        _process_receipt(
            process, folder, arm, digest, started, type(interruption).__name__
        )
    except BaseException as evidence_error:
        interruption.add_note(
            f"Native interruption receipt failed: {type(evidence_error).__name__}"
        )


def _process_receipt(process, folder, arm, digest, started, interruption=None):
    record = {
        "arm": arm,
        "request_sha256": digest,
        "wrapper_pid": process.pid,
        "owned_process_group": process.pid,
        "started_monotonic_ns": started,
        "finished_monotonic_ns": time.monotonic_ns(),
        "returncode": process.returncode,
        "interruption": interruption,
        "runtime_log_sha256": file_sha256(folder / "runtime.log"),
    }
    with (folder / "process.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_controls(interpreter, stage, recipe, source, output):
    """Run all four fresh controls and return bound final-child arguments.

    Args:
        interpreter: Installed Isaac launcher used by the original stage.
        stage: Original native mode.
        recipe: Sealed built-in reference recipe.
        source: Materialized input directory.
        output: Unique output directory for this attempt.
    Returns:
        Request and aggregate digest arguments for the fresh final child.
    Raises:
        ValueError: Any control, binding or comparison fails.
        RuntimeError: PhysX reports invalid physics.
        subprocess.CalledProcessError: A native control process fails.
    """
    request_sha = create_request(recipe, source, output, stage)
    for arm in ARMS:
        folder = output / "controls" / arm
        folder.mkdir()
        argv = [
            interpreter,
            "-m",
            "npa.workflows.navigation.runtime",
            stage,
            "--input-path",
            str(source),
            "--output-path",
            str(folder),
            "--visualizer",
            "none",
            "--control-arm",
            arm,
            "--control-request-sha256",
            request_sha,
        ]
        run_native_process(argv, folder, arm, request_sha)
    accepted = accept_controls(recipe, source, output, stage, request_sha)
    return [
        "--control-request-sha256",
        request_sha,
        "--control-acceptance-sha256",
        accepted,
    ]

"""Run a specialist independently or supervise one process per configured role."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .config import load_config
from .store import _private_file
from .team import SpecialistTeam


def run_worker(config_path: str, specialist: str):
    """Process queued work indefinitely, stopping gracefully at a durable boundary.

    Args: config_path: Operator-owned team JSON. specialist: Exact role to execute.
    Returns: None after SIGINT or SIGTERM.
    Raises: ValueError, OSError: Startup configuration or storage is invalid.
    """
    os.umask(0o077)
    team = SpecialistTeam(load_config(config_path))
    team.config.profile(specialist)
    stopped = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.set())
    while not stopped.is_set():
        try:
            task = team.work_once(specialist)
        except BlockingIOError:
            task = None
        if task is None or task["status"] in {"completed", "needs_attention"}:
            stopped.wait(1)
        elif task.get("next_observation_at", 0) > time.time():
            stopped.wait(min(1, task["next_observation_at"] - time.time()))


@contextmanager
def supervise(config_path: str):
    """Keep independent specialist processes alive for the lifetime of the service.

    Args: config_path: Shared operator team configuration.
    Returns: Context manager owning a supervisor thread and child processes.
    Raises: ValueError, OSError: Configuration or process creation fails.
    """
    config = load_config(config_path)
    stopped = threading.Event()
    processes = {}
    logs = {}
    thread = None
    try:
        directory = SpecialistTeam(config).store.directory
        _start_workers(config_path, config, directory, processes, logs)
        thread = threading.Thread(
            target=_watch,
            args=(config_path, config, directory, stopped, processes, logs),
            daemon=True,
        )
        thread.start()
        yield
    finally:
        stopped.set()
        if thread is not None:
            thread.join()
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
        for stream in logs.values():
            stream.close()


def _watch(config_path, config, directory, stopped, processes, logs):
    while not stopped.is_set():
        try:
            _start_workers(config_path, config, directory, processes, logs)
        except OSError:
            logging.getLogger(__name__).exception("Specialist restart failed")
        stopped.wait(1)


def _start_workers(config_path, config, directory, processes, logs):
    for profile in config.profiles:
        process = processes.get(profile.name)
        if process is not None and process.poll() is None:
            continue
        if profile.name not in logs:
            path = directory / (profile.name + ".log")
            _private_file(path)
            descriptor = os.open(path, os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW)
            logs[profile.name] = os.fdopen(descriptor, "ab")
        argv = [
            sys.executable,
            "-m",
            __name__,
            "--config",
            str(Path(config_path).resolve()),
            "--specialist",
            profile.name,
        ]
        processes[profile.name] = subprocess.Popen(
            argv, stdout=logs[profile.name], stderr=subprocess.STDOUT
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--specialist", required=True)
    options = parser.parse_args()
    run_worker(options.config, options.specialist)

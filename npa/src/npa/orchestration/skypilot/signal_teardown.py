"""Signal-safe teardown helpers for SkyPilot wrapper scripts."""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatchcase
import json
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from npa.orchestration.skypilot._bin import SkyBin
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    cluster_name_patterns_for_run,
    sky_environment,
)


@dataclass
class SignalTeardown:
    """Idempotently tear down clusters created by a SkyPilot wrapper."""

    run_id: str
    isolated_config_dir: Path | None = None
    config_path: Path | None = None
    sky_bin: SkyBin = None
    timeout: float = 900.0
    poll_interval: float = 10.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _launched: bool = field(default=False, init=False)
    _done: bool = field(default=False, init=False)

    def mark_launched(self, *, config_path: Path | None = None) -> None:
        """Record that the wrapper has started a SkyPilot submission."""

        with self._lock:
            self._launched = True
            if config_path is not None:
                self.config_path = config_path

    def teardown(self) -> CleanupResult:
        """Run ``sky down`` once and poll until matching clusters are absent."""

        with self._lock:
            if self._done or not self._launched:
                return CleanupResult()
            self._done = True

        patterns = cluster_name_patterns_for_run(self.run_id)
        cleanup = CleanupResult()
        down_errors: list[str] = []
        for pattern in patterns:
            cmd = self._sky_command(["down", "--yes", pattern])
            cleanup.commands.append(cmd)
            result = self._run(cmd, timeout=self.timeout)
            if result.returncode == 0:
                cleanup.resources_removed.append(pattern)
            else:
                down_errors.append(_format_command_error(cmd, result))

        missing, error = self._wait_until_absent(patterns)
        if error:
            cleanup.errors.extend(down_errors)
            cleanup.errors.append(error)
        elif missing:
            cleanup.errors.extend(down_errors)
            cleanup.errors.append(f"SkyPilot clusters still present after teardown timeout: {', '.join(missing)}")
        return cleanup

    def _wait_until_absent(self, patterns: list[str]) -> tuple[list[str], str]:
        deadline = time.monotonic() + self.timeout
        last_matches: list[str] = []
        last_error = ""
        while True:
            cmd = self._sky_command(["status", "--refresh", "--output", "json"])
            result = self._run(cmd, timeout=min(max(self.timeout, 1.0), 300.0))
            if result.returncode != 0:
                last_error = _format_command_error(cmd, result)
            else:
                last_error = ""
                last_matches = _matching_clusters(result.stdout, patterns)
                if not last_matches:
                    return [], ""
            if time.monotonic() >= deadline:
                return last_matches, last_error
            time.sleep(self.poll_interval)

    def _sky_command(self, args: list[str]) -> list[str]:
        executable = str(self.sky_bin or "sky")
        cmd = [executable, *args]
        if self.config_path is not None and "--config" not in cmd:
            cmd.insert(2, str(self.config_path))
            cmd.insert(2, "--config")
        return cmd

    def _run(self, cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            env=sky_environment(self.isolated_config_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )


def install_teardown_signal_handlers(teardown: Callable[[], CleanupResult]) -> dict[signal.Signals, signal.Handlers]:
    """Install SIGTERM/SIGINT handlers that run teardown and exit."""

    previous_handlers: dict[signal.Signals, signal.Handlers] = {}

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        teardown()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        sig = signal.Signals(signum)
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_signal)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[signal.Signals, signal.Handlers]) -> None:
    """Restore handlers returned by ``install_teardown_signal_handlers``."""

    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def _matching_clusters(output: str, patterns: list[str]) -> list[str]:
    try:
        payload = json.loads(output or "[]")
    except json.JSONDecodeError:
        return []
    clusters = payload if isinstance(payload, list) else payload.get("clusters", [])
    matches: list[str] = []
    for cluster in clusters or []:
        if not isinstance(cluster, dict):
            continue
        name = str(cluster.get("name") or cluster.get("cluster") or "")
        if name and any(fnmatchcase(name, pattern) for pattern in patterns):
            matches.append(name)
    return matches


def _format_command_error(cmd: list[str], result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    return f"{' '.join(cmd)}: {detail}"

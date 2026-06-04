from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from npa.orchestration.skypilot.cleanup import CleanupResult, cluster_name_patterns_for_run
from npa.orchestration.skypilot import signal_teardown as signal_teardown_module
from npa.orchestration.skypilot.signal_teardown import (
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)


def _fake_sky(tmp_path: Path) -> Path:
    sky = tmp_path / "sky"
    sky.write_text("#!/bin/sh\n", encoding="utf-8")
    sky.chmod(0o755)
    return sky


def test_signal_handlers_register_sigterm_and_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: dict[signal.Signals, signal.Handlers] = {}

    monkeypatch.setattr(signal_teardown_module.signal, "getsignal", lambda signum: signal.SIG_DFL)
    monkeypatch.setattr(signal_teardown_module.signal, "signal", lambda signum, handler: installed.setdefault(signum, handler))

    install_teardown_signal_handlers(lambda: CleanupResult())

    assert set(installed) == {signal.SIGTERM, signal.SIGINT}
    assert all(callable(handler) for handler in installed.values())


def test_signal_handler_runs_teardown_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: dict[signal.Signals, signal.Handlers] = {}
    calls = 0

    def fake_signal(signum, handler):
        installed[signum] = handler

    def teardown() -> CleanupResult:
        nonlocal calls
        calls += 1
        return CleanupResult(resources_removed=["cluster"])

    monkeypatch.setattr(signal_teardown_module.signal, "getsignal", lambda signum: signal.SIG_DFL)
    monkeypatch.setattr(signal_teardown_module.signal, "signal", fake_signal)
    install_teardown_signal_handlers(teardown)

    with pytest.raises(SystemExit) as exc:
        installed[signal.SIGTERM](signal.SIGTERM, None)

    assert exc.value.code == 128 + signal.SIGTERM
    assert calls == 1


def test_restore_signal_handlers_reinstalls_previous_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    restored: dict[signal.Signals, signal.Handlers] = {}
    previous = {signal.SIGTERM: signal.SIG_IGN, signal.SIGINT: signal.SIG_DFL}

    monkeypatch.setattr(signal_teardown_module.signal, "signal", lambda signum, handler: restored.setdefault(signum, handler))

    restore_signal_handlers(previous)

    assert restored == previous


def test_signal_teardown_runs_sky_down_polls_until_absent_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sky_bin = _fake_sky(tmp_path)
    config_path = tmp_path / "skypilot-config.yaml"
    run_id = "npa-signal-teardown-20260603T000000Z"
    patterns = cluster_name_patterns_for_run(run_id)
    matching_cluster = f"{patterns[0]}-worker"
    calls: list[list[str]] = []
    status_payloads = [
        json.dumps([{"name": matching_cluster}]),
        "[]",
    ]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["env"]["HOME"] == str(tmp_path / "home")
        if cmd[1] == "down":
            return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")
        if cmd[1] == "status":
            return subprocess.CompletedProcess(cmd, 0, stdout=status_payloads.pop(0), stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(signal_teardown_module.subprocess, "run", fake_run)

    teardown = SignalTeardown(
        run_id=run_id,
        isolated_config_dir=tmp_path,
        config_path=config_path,
        sky_bin=sky_bin,
        timeout=1,
        poll_interval=0,
    )
    teardown.mark_launched()

    result = teardown.teardown()
    second = teardown.teardown()

    assert result.errors == []
    assert second.commands == []
    assert second.resources_removed == []
    assert [cmd[1] for cmd in calls].count("down") == len(patterns)
    assert [cmd[1] for cmd in calls].count("status") == 2
    assert all("--config" in cmd for cmd in calls)
    forbidden_teardown_flag = "-" + "-down"
    assert all(forbidden_teardown_flag not in part for cmd in calls for part in cmd)

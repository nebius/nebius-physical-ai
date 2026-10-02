"""Reject empty GPU traces and keep diagnostic dependencies out of training math."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.flex_pi.training_profiler import CuptiProfile, write_gpu_trace
from npa.workbench.flex_pi import training_profiler_runtime as runtime


def _record(**updates):
    return {
        "name": "real-kernel",
        "cat": "kernel",
        "ph": "X",
        "pid": 0,
        "tid": 7,
        "start_ns": 1_790_000_000_000_000_001,
        "end_ns": 1_790_000_000_000_003_105,
        **updates,
    }


@pytest.mark.parametrize(
    "records", [[], [_record(cat="gpu_memory")], [_record(end_ns=0)]]
)
def test_gpu_profile_rejects_missing_kernels_or_invalid_intervals(tmp_path, records):
    destination = tmp_path / "profile.json"
    with pytest.raises(RuntimeError):
        write_gpu_trace(destination, records)
    assert not destination.exists()


def test_gpu_trace_keeps_integer_timestamp_precision_before_conversion(tmp_path):
    destination = tmp_path / "profile.json"
    receipt = write_gpu_trace(destination, [_record()])
    payload = json.loads(destination.read_text())
    event = payload["traceEvents"][0]
    assert event["ts"] == 0 and event["dur"] == 3.104
    assert receipt["gpu_kernel_events"] == 1
    assert receipt["instrumented_update"] == 5
    assert receipt["excluded_initial_updates"] == 6
    assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("dropped,callback_error", [(0, False), (1, False), (0, True)])
def test_profiler_captures_only_update_five_and_rejects_incomplete_records(
    tmp_path, dropped, callback_error
):
    profile = CuptiProfile.__new__(CuptiProfile)
    profile.destination = tmp_path / "trace.json"
    profile.updates = 0
    profile.active = False
    profile.records = [_record()]
    profile.callback_errors = ["bad-record"] if callback_error else []
    profile.receipt = None
    profile.kinds = [10, 1, 2]
    calls = []

    def query(context, stream, count):
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = dropped
        return 0

    profile.library = SimpleNamespace(cuptiActivityGetNumDroppedRecords=query)
    profile.synchronize = lambda: calls.append("sync")
    profile.api = SimpleNamespace(
        activity_register_callbacks=lambda *args: calls.append("register"),
        activity_enable=lambda kind: calls.append(("enable", kind)),
        activity_flush_all=lambda flag: calls.append(("flush", flag)),
        activity_disable=lambda kind: calls.append(("disable", kind)),
    )
    for _ in range(3):
        profile.step()
    assert not calls
    profile.step()
    assert profile.active and calls.count("register") == 1
    if dropped or callback_error:
        with pytest.raises(RuntimeError, match="missing or invalid"):
            profile.step()
        assert not profile.destination.exists()
    else:
        profile.step()
        assert profile.receipt["gpu_kernel_events"] == 1
        before = calls.copy()
        for _ in range(25):
            profile.step()
        assert calls == before
    assert not profile.active
    assert all(("disable", kind) in calls for kind in profile.kinds)


@pytest.mark.parametrize(
    "mode,gpu",
    [
        ("train", "B300"),
        ("qualify", "B300"),
        ("resume", "B300"),
        ("profile-resume", "B200"),
    ],
)
def test_profiler_runtime_does_not_change_nonprofile_or_nonb300_workers(
    tmp_path, mode, gpu
):
    environment = {"PYTHONPATH": "source", "NPA_FLEX_PI_PROFILE_BACKEND": "ambient"}
    updated, receipt = runtime.prepare_profiler_environment(
        {"rank_placement": [{"gpu": gpu}] * 4}, mode, tmp_path / "unused", environment
    )
    assert updated == {"PYTHONPATH": "source"}
    assert receipt is None
    assert environment["NPA_FLEX_PI_PROFILE_BACKEND"] == "ambient"
    assert not (tmp_path / "unused").exists()


def test_profiler_runtime_requires_homogeneous_verified_placement(tmp_path):
    with pytest.raises(RuntimeError, match="four verified B300"):
        runtime.prepare_profiler_environment(
            {"rank_placement": [{"gpu": "B300"}, {"gpu": "B200"}]},
            "profile",
            tmp_path / "unused",
            {},
        )
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_profiler_install_is_hash_pinned_and_never_replaces_compute_packages(
    tmp_path, monkeypatch, corrupt
):
    directory = tmp_path / "profiler"
    commands = []
    data = b"verified-library"
    monkeypatch.setattr(runtime, "CUPTI_SHA256", hashlib.sha256(data).hexdigest())

    def install(command, **kwargs):
        commands.append(command)
        library = directory / "nvidia/cu13/lib/libcupti.so.13"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"altered" if corrupt else data)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", install)
    args = (
        {"rank_placement": [{"gpu": "B300"}] * 4},
        "profile-resume",
        directory,
        {"PYTHONPATH": "source", "LD_LIBRARY_PATH": "vendor"},
    )
    if corrupt:
        with pytest.raises(RuntimeError, match="pinned hash"):
            runtime.prepare_profiler_environment(*args)
    else:
        env, receipt = runtime.prepare_profiler_environment(*args)
        assert env["PYTHONPATH"].endswith(":source")
        assert env["LD_LIBRARY_PATH"].endswith(":vendor")
        assert receipt["compute_stack_replaced"] is False
        assert receipt["cupti_sha256"] == hashlib.sha256(data).hexdigest()
    command = commands[0]
    assert all(
        flag in command
        for flag in [
            "--isolated",
            "--no-deps",
            "--no-index",
            "--require-hashes",
            "--target",
        ]
    )
    lock = Path(command[command.index("--requirement") + 1]).read_text()
    packages = [
        line.split(" @ ")[0]
        for line in lock.splitlines()
        if line and not line.startswith("#")
    ]
    assert set(packages) == {
        "cupti-python",
        "cuda-bindings",
        "cuda-pathfinder",
        "nvidia-cuda-cupti",
    }
    assert all(
        "--hash=sha256:" in line
        for line in lock.splitlines()
        if line and not line.startswith("#")
    )

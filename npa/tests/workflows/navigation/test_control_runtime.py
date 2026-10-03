"""Exercise native process boundaries and initialization parity without Kit."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import control_processes, measure, runtime, stages
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.contract import InitialCheckpoint
from npa.workflows.navigation.control_protocol import ARMS, REFERENCE


@pytest.mark.parametrize("stage", ["train", "evaluate", "evaluate-checkpoint"])
def test_control_initialization_matches_final_workload(
    recipe, tmp_path, monkeypatch, stage
):
    events = []
    settings, runner = _mock_initialization(recipe, tmp_path, monkeypatch, events)
    args = SimpleNamespace(stage=stage, input_path=tmp_path, output_path=tmp_path)
    control_init = runtime._initialize_control(args, recipe, runner, settings, "cuda:0")
    expected_events = list(events)
    events.clear()
    observed = []

    def before_work(adapter, env, wrapped, recipe, output, control, initialization):
        observed.append(initialization)
        raise RuntimeError("fixture: before learning/evaluation")

    monkeypatch.setattr(runtime, "_probes", before_work)
    env = SimpleNamespace(unwrapped=SimpleNamespace(device="cuda:0"))
    with pytest.raises(RuntimeError, match="before learning/evaluation"):
        runtime._run_environment_stage(
            args, recipe, env, None, runner, None, settings, {"fixture": True}
        )
    assert events == expected_events
    assert observed == [control_init]


def _mock_initialization(recipe, tmp_path, monkeypatch, events):
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"fixture-native-state")
    recipe.initial_checkpoint = InitialCheckpoint(
        file="baseline.pt", sha256=file_sha256(checkpoint)
    )
    initialized = {
        "mode": "resume_native_checkpoint",
        "checkpoint_sha256": file_sha256(checkpoint),
        "initial_iteration": 123,
    }
    monkeypatch.setattr(
        "npa.workflows.navigation.initialization.initialize_runner",
        lambda *args: events.append("restore-all-state") or initialized,
    )
    monkeypatch.setattr(
        runtime, "parameters", lambda _: events.append("copy-parameters")
    )
    monkeypatch.setattr(
        runtime,
        "policy_state_digest",
        lambda _: events.append("hash-policy") or "a" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "_load_evaluation",
        lambda *args: events.append("load-evaluation") or checkpoint,
    )
    settings = {"seed": recipe.train_cases[0].seed}
    (tmp_path / "agent.json").write_text(json.dumps(settings))
    runner = SimpleNamespace(
        current_learning_iteration=123,
        get_inference_policy=lambda device: events.append(("inference", device)),
    )
    return settings, runner


@pytest.mark.parametrize(
    "arm,digest,accepted",
    [
        (None, None, None),
        ("solo", None, None),
        (None, "a" * 64, None),
        ("solo", "a" * 64, "b" * 64),
    ],
)
def test_no_unchecked_reference_skip_option(recipe, arm, digest, accepted):
    recipe.adapter_module = REFERENCE
    args = SimpleNamespace(
        control_arm=arm,
        control_request_sha256=digest,
        control_acceptance_sha256=accepted,
    )
    with pytest.raises(ValueError, match="parent-bound fresh controls"):
        runtime._validate_control_request(args, recipe)


def test_custom_adapter_cannot_import_reference_qualification(recipe):
    args = SimpleNamespace(
        control_arm=None,
        control_request_sha256="a" * 64,
        control_acceptance_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="custom adapters"):
        runtime._validate_control_request(args, recipe)
    runtime._validate_control_request(SimpleNamespace(), recipe)


def test_sequential_children_retain_complete_logs_and_distinct_pids(tmp_path):
    pids = []
    previous_finish = 0
    for arm in ARMS:
        folder = tmp_path / arm
        folder.mkdir()
        argv = [sys.executable, "-c", "import os; print('native child', os.getpid())"]
        control_processes.run_native_process(argv, folder, arm, "a" * 64)
        receipt = json.loads((folder / "process.json").read_text())
        pids.append(receipt["wrapper_pid"])
        assert receipt["returncode"] == 0
        assert receipt["runtime_log_sha256"] == file_sha256(folder / "runtime.log")
        assert str(receipt["wrapper_pid"]) in (folder / "runtime.log").read_text()
        assert previous_finish <= receipt["started_monotonic_ns"]
        previous_finish = receipt["finished_monotonic_ns"]
    assert len(set(pids)) == 4


@pytest.mark.parametrize(
    "code,exception",
    [
        (
            "import sys; print('actual failure'); sys.exit(3)",
            subprocess.CalledProcessError,
        ),
        ("print('PhysX error: invalid contact buffers')", RuntimeError),
        ("print('simulation will miss interactions')", RuntimeError),
    ],
)
def test_native_errors_keep_log_and_exit_receipt(tmp_path, code, exception):
    with pytest.raises(exception):
        control_processes.run_native_process(
            [sys.executable, "-c", code], tmp_path, "solo", "a" * 64
        )
    assert (tmp_path / "runtime.log").stat().st_size > 0
    assert json.loads((tmp_path / "process.json").read_text())["arm"] == "solo"


@pytest.mark.parametrize("stage", ["train", "evaluate", "evaluate-checkpoint"])
def test_all_control_argv_keep_original_native_mode(
    stage, recipe, tmp_path, monkeypatch
):
    (tmp_path / "controls").mkdir()
    monkeypatch.setattr(control_processes, "create_request", lambda *args: "a" * 64)
    events = []
    monkeypatch.setattr(
        control_processes,
        "run_native_process",
        lambda argv, folder, arm, digest: events.append((argv, folder, arm, digest)),
    )

    def accept(*args):
        assert [row[2] for row in events] == list(ARMS)
        return "b" * 64

    monkeypatch.setattr(control_processes, "accept_controls", accept)
    pins = control_processes.run_controls(
        "/native python", stage, recipe, tmp_path / "input ; literal", tmp_path
    )
    assert pins == [
        "--control-request-sha256",
        "a" * 64,
        "--control-acceptance-sha256",
        "b" * 64,
    ]
    for argv, folder, arm, digest in events:
        assert argv[:4] == [
            "/native python",
            "-m",
            "npa.workflows.navigation.runtime",
            stage,
        ]
        assert argv[5] == str(tmp_path / "input ; literal")
        assert argv[7] == str(tmp_path / "controls" / arm)
        assert argv[8:10] == ["--visualizer", "none"]
        assert digest == "a" * 64 and folder.name == arm


def test_failed_arm_prevents_later_children_and_aggregate(
    recipe, tmp_path, monkeypatch
):
    (tmp_path / "controls").mkdir()
    monkeypatch.setattr(control_processes, "create_request", lambda *args: "a" * 64)
    events = []

    def run(argv, folder, arm, digest):
        events.append(arm)
        if arm == "repeat":
            raise RuntimeError("native control failed")

    monkeypatch.setattr(control_processes, "run_native_process", run)
    monkeypatch.setattr(
        control_processes, "accept_controls", lambda *args: pytest.fail("no acceptance")
    )
    with pytest.raises(RuntimeError, match="native control failed"):
        control_processes.run_controls("/native", "train", recipe, tmp_path, tmp_path)
    assert events == ["solo", "repeat"]


def test_standard_stage_runs_controls_before_distinct_final_child(
    recipe, tmp_path, monkeypatch
):
    native = tmp_path / "native"
    native.touch()
    monkeypatch.setenv("ISAAC_LAB_PYTHON", str(native))
    recipe.adapter_module = REFERENCE
    monkeypatch.setattr(stages, "read_recipe", lambda _: recipe)
    calls = []
    monkeypatch.setattr(
        control_processes,
        "run_controls",
        lambda *args: (
            calls.append("controls")
            or [
                "--control-request-sha256",
                "a" * 64,
                "--control-acceptance-sha256",
                "b" * 64,
            ]
        ),
    )

    def final(argv, output, arm, digest):
        assert calls == ["controls"]
        assert arm == "main" and digest == "a" * 64
        assert argv[-1] == "b" * 64
        calls.append("main")
        (output / "evaluation.json").write_text(
            json.dumps(
                {"schema": "npa.navigation.evaluation.v1", "policy_loaded": True}
            )
        )

    monkeypatch.setattr(control_processes, "run_native_process", final)
    assert stages._native("evaluate-checkpoint", tmp_path, tmp_path)["policy_loaded"]
    assert calls == ["controls", "main"]


def test_partial_control_trace_is_persisted_on_native_failure(
    recipe, tmp_path, monkeypatch
):
    def fail(*args, trace):
        trace.append(
            {
                "state": {"position_m": np.zeros((2, 3))},
                "observations": {"policy": np.zeros((2, 3))},
            }
        )
        raise RuntimeError("step failed")

    monkeypatch.setattr(measure, "_probe_trace", fail)
    with pytest.raises(RuntimeError, match="step failed"):
        measure._recorded_probe(
            None, None, None, recipe, [], tmp_path, "solo", retain_partial=True
        )
    with np.load(tmp_path / "probe-solo.npz", allow_pickle=False) as trace:
        assert trace["0/state/position_m"].shape == (2, 3)
    assert not (tmp_path / "control.json").exists()


@pytest.mark.parametrize(
    "interruption", [KeyboardInterrupt("operator interrupted"), SystemExit(17)]
)
def test_interruption_stops_owned_group_and_retains_original(
    tmp_path, monkeypatch, interruption
):
    import os

    child = _interrupted_child_script(tmp_path)
    processes = []
    start = _interrupted_launcher(tmp_path, subprocess.Popen, processes, interruption)
    monkeypatch.setattr(control_processes.subprocess, "Popen", start)
    with pytest.raises(type(interruption)) as raised:
        control_processes.run_native_process(
            [sys.executable, str(child)], tmp_path, "solo", "a" * 64
        )
    assert raised.value is interruption
    assert (
        processes[0].returncode == 0
    )  # Graceful exit cannot qualify an interrupted arm.
    _, wrapper, worker = (tmp_path / "runtime.log").read_text().split()
    for pid in (int(wrapper), int(worker)):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    record = json.loads((tmp_path / "process.json").read_text())
    assert record["interruption"] == type(interruption).__name__
    assert record["owned_process_group"] == record["wrapper_pid"] == int(wrapper)
    assert record["runtime_log_sha256"] == file_sha256(tmp_path / "runtime.log")


def _interrupted_child_script(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, subprocess, sys\n"
        "if len(sys.argv) > 1:\n"
        "    signal.pause()\n"
        "else:\n"
        "    worker = subprocess.Popen([sys.executable, __file__, 'worker'])\n"
        "    def stop(*args):\n"
        "        worker.wait()\n"
        "        sys.exit(0)\n"
        "    signal.signal(signal.SIGTERM, stop)\n"
        "    print('ready', os.getpid(), worker.pid, flush=True)\n"
        "    signal.pause()\n"
    )
    return child


def _interrupted_launcher(tmp_path, popen, processes, interruption):
    import os
    import time

    def start(argv, **kwargs):
        assert kwargs["start_new_session"] is True
        process = popen(argv, **kwargs)
        processes.append(process)
        wait = process.wait
        deadline = time.monotonic() + 10
        while "ready" not in (tmp_path / "runtime.log").read_text():
            if time.monotonic() > deadline:
                os.killpg(process.pid, 9)
                wait()
                pytest.fail("fixture child did not start")
            time.sleep(0.01)
        calls = 0

        def interrupted_wait(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise interruption
            return wait(*args, **kwargs)

        process.wait = interrupted_wait
        return process

    return start


def test_repeated_interrupt_kills_only_owned_group_and_reaps_child(monkeypatch):
    import signal

    signals = []
    waits = iter([KeyboardInterrupt(), 0])

    def wait():
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(
        control_processes.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    control_processes._terminate_owned_process(
        SimpleNamespace(pid=123, wait=wait, returncode=None)
    )
    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


@pytest.mark.parametrize("publication_error", [False, True])
def test_interrupted_stage_preserves_failure_and_original_exception(
    recipe, tmp_path, monkeypatch, publication_error
):
    interruption = KeyboardInterrupt("original")
    monkeypatch.setenv("NPA_TASK_IMAGE", recipe.image)
    monkeypatch.setattr(stages, "materialize", lambda source, target: target)
    monkeypatch.setattr(stages, "read_recipe", lambda source: recipe)
    retained = []

    def native(stage, source, output):
        folder = output / "controls/solo"
        folder.mkdir(parents=True)
        (folder / "runtime.log").write_text("partial native output")
        raise interruption

    def publish(output, destination):
        retained.append(json.loads((output / "failure.json").read_text()))
        assert (
            output / "controls/solo/runtime.log"
        ).read_text() == "partial native output"
        if publication_error:
            raise OSError("fixture transport unavailable")

    monkeypatch.setattr(stages, "_native", native)
    monkeypatch.setattr(stages, "publish", publish)
    with pytest.raises(KeyboardInterrupt) as raised:
        stages.run_stage("train", "input", str(tmp_path / "published"))
    assert raised.value is interruption
    assert retained[0]["status"] == "failed"
    if publication_error:
        assert "publication failed: OSError" in interruption.__notes__[0]


def test_already_reaped_child_never_signals_recycled_pid(monkeypatch):
    monkeypatch.setattr(
        control_processes.os, "killpg", lambda *args: pytest.fail("no live child")
    )
    control_processes._terminate_owned_process(SimpleNamespace(pid=123, returncode=0))

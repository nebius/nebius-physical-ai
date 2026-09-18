"""Exercise measured phase stalls and real owned-process cleanup without a simulator."""

import json
import os
import subprocess
import sys

import pytest

from npa.workbench.isaac_arena import phase_liveness as liveness


def _event(sequence, timestamp, event, action=1, phase="simulation_step"):
    return {"schema": "npa.isaac-arena.simulator-phase.v1", "sequence": sequence,
            "monotonic_ns": timestamp, "event": event, "action_step": action,
            "phase": phase, "rank": 0}


def _calibrated():
    progress = liveness.PhaseProgress()
    for index in range(8):
        progress.advance(_event(index * 2 + 1, index * 20 + 1, "begin"))
        progress.advance(_event(index * 2 + 2, index * 20 + 11, "end"))
    return progress


def test_stall_depends_on_measured_phase_advancement_not_total_runtime():
    progress = _calibrated()
    progress.advance(_event(17, 10**15, "begin", 5))
    assert progress.stalled(10**15 + 10 * 4096) is None
    failure = progress.stalled(10**15 + 10 * 4096 + 1)
    assert failure["phase"] == "simulation_step"
    assert failure["action_step"] == 5 and failure["baseline_samples"] == 8
    progress.advance(_event(18, 10**15 + 12, "end", 5))
    assert progress.stalled(10**30) is None


def test_cold_unmeasured_phase_does_not_borrow_another_phases_deadline():
    progress = _calibrated()
    progress.advance(_event(17, 200, "begin", phase="render_call"))
    assert progress.stalled(10**30) is None


def test_nested_advancement_and_rank_observers_do_not_reset_each_other():
    first, second = _calibrated(), _calibrated()
    first.advance(_event(17, 200, "begin", 5))
    second.advance(_event(17, 200, "begin", 5))
    second.advance(_event(18, 201, "end", 5))
    assert first.stalled(50000)
    assert second.stalled(50000) is None


@pytest.mark.parametrize("change", ["sequence", "clock", "nesting", "fields", "rank", "type"])
def test_invalid_progress_is_not_healthy_evidence(change):
    progress = _calibrated()
    row = _event(17, 200, "begin")
    if change == "sequence":
        row["sequence"] = 19
    elif change == "clock":
        row["monotonic_ns"] = 0
    elif change == "nesting":
        row["event"] = "end"
    elif change == "rank":
        row["rank"] = 1
    elif change == "type":
        row = []
    else:
        row["arbitrary_private_data"] = "must not be accepted"
    with pytest.raises(ValueError):
        progress.advance(row)


def test_partial_append_is_not_a_corrupt_record(tmp_path):
    path = tmp_path / "journal.jsonl"
    body = json.dumps(_event(1, 1, "begin"))
    path.write_text(body[:30])
    reader = liveness._JournalReader(path)
    reader.update()
    assert reader.progress.sequence == 0
    with path.open("a") as stream:
        stream.write(body[30:] + "\n")
    reader.update()
    assert reader.progress.sequence == 1


def test_real_stalled_child_is_stopped_and_diagnostic_retained(tmp_path):
    root, private = tmp_path / "artifacts", tmp_path / "private"
    run = root / "upstream/run"
    run.mkdir(parents=True)
    private.mkdir()
    source = """import json, pathlib, time
path=pathlib.Path(__import__('sys').argv[1])
base=time.monotonic_ns()
with path.open('w') as stream:
    for index in range(17):
        stream.write(json.dumps({'schema':'npa.isaac-arena.simulator-phase.v1',
          'sequence':index+1,'monotonic_ns':base+index,'rank':0,'action_step':5,
          'phase':'simulation_step','event':'begin' if index%2==0 else 'end'})+'\\n')
        stream.flush()
print('real child reached native stand-in',flush=True)
while True: time.sleep(1)
"""
    result = liveness.run_supervised(
        [sys.executable, "-c", source, str(run / "simulator-phases-rank0.jsonl")],
        artifact_root=root, private_dir=private, text=True,
    )
    assert result.returncode == 124
    assert "real child reached" in result.stdout
    evidence = json.loads((root / "simulator-liveness.json").read_text())
    assert evidence["status"] == "stalled" and evidence["action_step"] == 5
    assert not list(root.rglob("episode_results*"))


def test_success_preserves_exact_native_result_and_command(tmp_path):
    argv = [sys.executable, "-c", "print('native completed')"]
    result = liveness.run_supervised(argv, artifact_root=tmp_path, private_dir=tmp_path, text=True)
    assert result.args is argv and result.returncode == 0
    assert result.stdout == "native completed\n"
    assert not (tmp_path / "simulator-liveness.json").exists()


def test_state_only_evaluation_stages_private_log_without_replay_or_graphics(tmp_path):
    private = tmp_path / "private"
    assert not private.exists()
    result = liveness.run_supervised(
        [sys.executable, "-c", "print('state-only native result'); raise SystemExit(7)"],
        artifact_root=tmp_path, private_dir=private, text=True,
    )
    assert result.returncode == 7
    assert result.stdout == "state-only native result\n"
    assert (private / "simulator-process.log").read_text() == result.stdout
    assert private.stat().st_mode & 0o077 == 0


def test_termination_escalates_only_for_owned_process(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"],
                               start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline() == "ready\n"
    liveness._stop_owned_process(process)
    assert process.returncode == -9
    assert os.getpid() != process.pid


def test_termination_reaps_resistant_native_child_after_leader_exits(tmp_path):
    source = """import os, signal, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    while True: time.sleep(1)
while True: time.sleep(1)
"""
    process = subprocess.Popen([sys.executable, "-c", source], start_new_session=True,
                               stdout=subprocess.PIPE, text=True)
    child_pid = int(process.stdout.readline())
    liveness._stop_owned_process(process)
    assert process.returncode == -15
    # A killed orphan can remain a zombie until the host init reaps it.
    stat = __import__("pathlib").Path(f"/proc/{child_pid}/stat")
    for _ in range(100):
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        __import__("time").sleep(0.01)
    else:
        pytest.fail("native child remained active after the leader terminated")

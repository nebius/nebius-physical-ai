"""Fail stalled native phases from measured progress without limiting episode duration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .simulator_phases import _validate_event


# A phase must establish its own baseline; cold startup is not a stalled rollout.
_BASELINE_SAMPLES = 8
_SLOWDOWN_FACTOR = 4096


@dataclass
class PhaseProgress:
    """Track monotonic, nested native phase events for one simulator rank."""

    sequence: int = 0
    last_ns: int = 0
    stack: list[dict] = field(default_factory=list)
    samples: dict[str, tuple[int, int]] = field(default_factory=dict)
    rank: int | None = None

    def advance(self, row: dict) -> None:
        """Consume one validated event, rejecting discontinuous diagnostic evidence.

        Args:
            row: One complete scalar journal record.
        Returns:
            None.
        Raises:
            ValueError: Sequence, clock, schema or native nesting is invalid.
        """
        if not isinstance(row, dict) or row.get("schema") != "npa.isaac-arena.simulator-phase.v1":
            raise ValueError("invalid phase journal schema")
        rank = row.get("rank")
        if type(rank) is not int or rank < 0 or self.rank not in (None, rank):
            raise ValueError("phase rank changed or is invalid")
        self.rank = rank
        sequence, timestamp = row.get("sequence"), row.get("monotonic_ns")
        if type(sequence) is not int or sequence != self.sequence + 1:
            raise ValueError("phase sequence is discontinuous")
        if type(timestamp) is not int or timestamp < self.last_ns:
            raise ValueError("phase clock regressed")
        fields = {k: v for k, v in row.items() if k not in {
            "schema", "sequence", "monotonic_ns", "rank", "action_step", "phase", "event"}}
        _validate_event(row["phase"], row["event"], row["action_step"], fields)
        self.sequence, self.last_ns = sequence, timestamp
        if row["event"] == "begin":
            self.stack.append(row)
        elif row["event"] in {"end", "failed"}:
            self._complete(row)

    def _complete(self, row):
        if not self.stack:
            raise ValueError("phase ended without entry")
        start = self.stack.pop()
        if any(start.get(k) != row.get(k) for k in ("phase", "action_step", "render_call")):
            raise ValueError("phase nesting changed")
        if row["event"] == "end":
            count, maximum = self.samples.get(row["phase"], (0, 0))
            self.samples[row["phase"]] = (count + 1, max(maximum, row["monotonic_ns"] - start["monotonic_ns"]))

    def stalled(self, now_ns: int) -> dict | None:
        """Describe a calibrated phase only after measured absence of advancement.

        Args:
            now_ns: Current operating-system monotonic clock in nanoseconds.
        Returns:
            Sanitized stall evidence or None while progressing or uncalibrated.
        Raises:
            None.
        """
        if not self.stack:
            return None
        phase = self.stack[-1]
        count, maximum = self.samples.get(phase["phase"], (0, 0))
        elapsed = now_ns - self.last_ns
        if count < _BASELINE_SAMPLES or maximum <= 0 or elapsed <= maximum * _SLOWDOWN_FACTOR:
            return None
        return {"phase": phase["phase"], "action_step": phase["action_step"],
                "sequence": self.sequence, "baseline_samples": count,
                "baseline_max_ns": maximum, "no_progress_ns": elapsed,
                "slowdown_factor": _SLOWDOWN_FACTOR, "rank": phase["rank"]}


class _JournalReader:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.progress = PhaseProgress()

    def update(self):
        if self.path.stat().st_size < self.offset:
            raise ValueError("phase journal was truncated")
        with self.path.open() as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith("\n"):
                    break
                self.progress.advance(json.loads(line))
                self.offset = stream.tell()


def _observe(directory, readers):
    for path in directory.glob("upstream/*/simulator-phases-rank*.jsonl"):
        reader = readers.setdefault(path, _JournalReader(path))
        reader.update()
    for reader in readers.values():
        stalled = reader.progress.stalled(time.monotonic_ns())
        if stalled:
            return {"schema": "npa.isaac-arena.phase-liveness.v1", "status": "stalled", **stalled}
    return None


def _stop_owned_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.05)
    # The leader can exit while a native child still holds the GPU. Escalate
    # for the original process group even after its leader has been reaped.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
        return
    process.wait()


def _monitor(process, artifact_root):
    readers = {}
    while process.poll() is None:
        try:
            failure = _observe(artifact_root, readers)
        except (OSError, ValueError, KeyError, TypeError):
            failure = {"schema": "npa.isaac-arena.phase-liveness.v1", "status": "invalid_progress_evidence"}
        if failure:
            path = artifact_root / "simulator-liveness.json"
            with path.open("w") as stream:
                json.dump(failure, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            _stop_owned_process(process)
            return failure
        time.sleep(0.1)
    return None


def run_supervised(argv, *, artifact_root: Path, private_dir: Path, **kwargs):
    """Run the simulator with an independent observer and retain failed evidence.

    Args:
        argv: Unchanged native simulator command and arguments.
        artifact_root: Evaluation artifact directory with scalar phase journals.
        private_dir: Private log staging directory.
        kwargs: Existing subprocess.run settings; check and output capture are internal.
    Returns:
        CompletedProcess with nonzero status for stalled or invalid phase evidence.
    Raises:
        OSError: Process or evidence staging failed, after owned-process cleanup.
    """
    options = {k: v for k, v in kwargs.items() if k not in {"stdout", "stderr", "check"}}
    # State-only evaluations have neither replay staging nor graphics setup;
    # the supervisor owns creation of its private log directory in every mode.
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = private_dir / "simulator-process.log"
    with log_path.open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, **options)
        try:
            failure = _monitor(process, artifact_root)
        finally:
            _stop_owned_process(process)
    output = log_path.read_text(errors="replace")
    if failure:
        output += "\nSimulator phase liveness failed; retained scalar diagnostics.\n"
    return subprocess.CompletedProcess(argv, 124 if failure else process.returncode, output)

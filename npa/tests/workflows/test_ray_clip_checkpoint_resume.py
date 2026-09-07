# Responsibility: Prove checkpoint reuse and completion with serial actor queues.
"""Synthetic vectors exercise scheduling; separate live tests require CUDA."""

from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def application(monkeypatch):
    """Load the shipped application without leaking its generic module names."""
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("application", "validation", "worker")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    yield importlib.import_module("application")
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def vectors(shard):
    """Return identifiable unit vectors, explicitly without model inference."""
    import pyarrow as pa

    rows = [[float(i == row["record_id"]) for i in range(512)] for row in shard["rows"]]
    return pa.array(rows, type=pa.list_(pa.float32(), 512))


class Rendezvous:
    """Model a blocking first wave that requires distinct serial actors."""

    def __init__(self, participants):
        self.participants = participants
        self.gate = threading.Barrier(participants)
        self.finish = threading.Barrier(participants)
        self.arrivals = set()
        self.status = SimpleNamespace(remote=self.observation)

    def observation(self):
        """Expose only participants that actually reached the rendezvous."""
        return {"participants": len(self.arrivals), "overlap": len(self.arrivals) > 1, "events": []}


class SerialActor:
    """Run each actor's methods serially, as Ray does for synchronous actors."""

    def __init__(self, name, calls):
        self.name, self.calls = name, calls
        self.completed = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.infer = SimpleNamespace(remote=self.submit)
        self.status = SimpleNamespace(remote=lambda: {"instance_id": name, "inference_calls": self.completed})

    def submit(self, shard, barrier=None):
        """Record every scheduled call, including calls that cannot finish."""
        self.calls.append((shard["rows"][0]["record_id"], self.name))
        return self.executor.submit(self.run, shard, barrier)

    def run(self, shard, barrier):
        """Fail promptly on a deadlock instead of letting the test hang."""
        if barrier is not None:
            barrier.arrivals.add(self.name)
            barrier.gate.wait(timeout=3)
            barrier.finish.wait(timeout=3)
        self.completed += 1
        return vectors(shard), {"instance_id": self.name, "inference_seconds": 0.25}


@pytest.fixture
def session(application, tmp_path, monkeypatch):
    """Exercise the real session and checkpoint writer with observable queues."""
    calls, actors, barriers = [], [], []

    def get(value):
        """Resolve futures and Ray-style lists without hiding worker failures."""
        if isinstance(value, list):
            return [get(item) for item in value]
        return value.result(timeout=5) if isinstance(value, Future) else value

    def barrier(count):
        """Retain the rendezvous so its participant count is independently tested."""
        result = Rendezvous(count)
        barriers.append(result)
        return result

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        get=get, kill=lambda *a, **kw: None, is_initialized=lambda: False,
    ))

    def create(count, cached, total=7):
        """Create real committed Parquet fixtures before a new invocation."""
        args = SimpleNamespace(actors=count, model_revision="revision", output_path=str(tmp_path))
        result = application._InferenceSession(args, [[i] for i in range(total)])
        result.fingerprint = "execution"
        prepared = [application.worker.preprocess_shard([i]) for i in range(total)]
        for index in cached:
            application.commit_shard(
                tmp_path / "shards" / f"{index:06d}", prepared[index], vectors(prepared[index]),
                {"instance_id": "previous-job-actor", "inference_seconds": 5.0}, "revision", "execution",
            )
        actors.extend(SerialActor(f"current-actor-{i}", calls) for i in range(count))
        result.actors = actors
        result.prepare_shard = SimpleNamespace(remote=lambda ids: prepared[ids[0]])
        result.inference_barrier = SimpleNamespace(remote=barrier)
        result.started = result.model_ready = 0.0
        return result

    yield SimpleNamespace(create=create, calls=calls, barriers=barriers, path=tmp_path)
    for item in barriers:
        item.gate.abort()
        item.finish.abort()
    for actor in actors:
        actor.executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("actors,cached,total", [
    (2, set(range(7)), 7),
    (2, {0, 2, 4, 5, 6}, 7),  # sparse indices 1,3 used to select the same actor
    (4, {0, 1, 2, 3, 4, 5}, 7),
    (4, {0, 2, 4, 5, 6}, 7),
    (2, set(), 7),
    (1, {0, 2, 3, 5}, 7),
    (2, {0}, 1),
    (2, set(), 1),
])
def test_resume_schedules_only_missing_shards_and_completes(application, session, actors, cached, total):
    """Verify call counts, real Parquet completion, and sparse-wave liveness."""
    run = session.create(actors, cached, total)
    original = {p: p.read_bytes() for p in session.path.glob("shards/*/*")}
    run._commit_first_shard()
    run._infer_remaining_shards()
    observation = run._check_concurrency()
    expected = set(range(total)) - cached
    assert sorted(index for index, _ in session.calls) == sorted(expected)
    assert sum(actor["inference_calls"] for actor in run.final_actors) == len(expected)
    assert len(run.receipts) == total
    assert all(receipt["checkpoint_reused"] == (i in cached) for i, receipt in enumerate(run.receipts))
    assert all(path.read_bytes() == content for path, content in original.items())
    wave = min(actors, len(expected - {0}))
    assert [item.participants for item in session.barriers] == ([wave] if wave else [])
    assert observation["participants"] == wave
    assert observation["overlap"] == (wave > 1)
    report = application.aggregate(session.path, run.receipts, total)
    assert report["records"] == report["lance_rows"] == total
    timing = run._timing_report(observation)
    assert timing["inference_actor_seconds_sum"] == len(expected) * 0.25
    assert timing["retained_checkpoint_inference_actor_seconds_sum"] == len(cached) * 5.0
    assert timing["inferred_shards"] == sorted(expected)
    assert timing["reused_checkpoint_shards"] == sorted(cached)


@pytest.mark.parametrize("damage", ["bytes", "identity", "missing_data", "malformed_marker"])
def test_invalid_later_checkpoint_fails_before_any_later_inference(application, session, damage):
    """Never spend GPU work or overwrite corrupt committed bytes during resume."""
    run = session.create(2, {0, 4})
    path = session.path / "shards" / "000004"
    if damage == "bytes":
        (path / "embeddings.parquet").write_bytes(b"corrupt committed data")
    elif damage == "missing_data":
        (path / "embeddings.parquet").unlink()
    elif damage == "malformed_marker":
        (path / "commit.json").write_text("{")
    else:
        marker = json.loads((path / "commit.json").read_text())
        marker["identity"]["execution_fingerprint"] = "incompatible"
        (path / "commit.json").write_text(json.dumps(marker))
    before = {p: p.read_bytes() for p in path.iterdir()}
    run._commit_first_shard()
    with pytest.raises((ValueError, FileNotFoundError)):
        run._infer_remaining_shards()
    assert session.calls == []
    assert session.barriers == []
    assert all(p.read_bytes() == content for p, content in before.items())


def test_missing_commit_marker_recomputes_uncommitted_bytes(application, session):
    """A partial write without its commit marker does not count as a cache hit."""
    run = session.create(2, set(range(7)))
    path = session.path / "shards" / "000003"
    (path / "commit.json").unlink()
    (path / "embeddings.parquet").write_bytes(b"partial write")
    run._commit_first_shard()
    run._infer_remaining_shards()
    run._check_concurrency()
    assert [index for index, _ in session.calls] == [3]
    assert run.receipts[3]["checkpoint_reused"] is False
    assert application.aggregate(session.path, run.receipts, 7)["records"] == 7


def test_missing_overlap_still_fails_for_a_real_multi_actor_wave(session):
    """Sparse support must not waive the established multi-actor overlap gate."""
    run = session.create(2, {0})
    run._commit_first_shard()
    run._infer_remaining_shards()
    run.barrier.status.remote = lambda: {"participants": 2, "overlap": False, "events": []}
    with pytest.raises(ValueError, match="never overlapped"):
        run._check_concurrency()

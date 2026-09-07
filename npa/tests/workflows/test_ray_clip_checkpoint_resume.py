# Responsibility: Prove checkpoint reuse and completion with serial actor queues.
"""Synthetic vectors exercise scheduling; separate live tests require CUDA."""

from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def application(monkeypatch):
    """Load the shipped application without leaking its generic module names.

    Args:
        monkeypatch: Pytest fixture for isolated imports and environment changes.
    Returns:
        An iterator yielding the isolated application module.
    Raises:
        ImportError: The isolated reference module cannot be imported.
    """
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("application", "validation", "worker")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    yield importlib.import_module("application")
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def _vectors(shard):
    """Return identifiable unit vectors, explicitly without model inference."""
    import pyarrow as pa

    rows = [[float(i == row["record_id"]) for i in range(512)] for row in shard["rows"]]
    return pa.array(rows, type=pa.list_(pa.float32(), 512))


class _Rendezvous:
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


class _SerialActor:
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
        return _vectors(shard), {"instance_id": self.name, "inference_seconds": 0.25}


def _resolve_future(value):
    """Resolve futures and Ray-style lists without hiding worker failures."""
    if isinstance(value, list):
        return [_resolve_future(item) for item in value]
    return value.result(timeout=5) if isinstance(value, Future) else value


class _SessionFactory:
    """Keep serial actors and factual checkpoint fixtures in one test lifetime."""

    def __init__(self, application, directory):
        self.application = application
        self.path = directory
        self.calls, self.actors, self.barriers = [], [], []

    def _barrier(self, count):
        """Retain the rendezvous so its participant count is independently tested."""
        result = _Rendezvous(count)
        self.barriers.append(result)
        return result

    def _create(self, count, cached, total=7):
        """Create real committed Parquet fixtures before a new invocation."""
        arguments = SimpleNamespace(actors=count, model_revision="revision", output_path=str(self.path))
        result = self.application._InferenceSession(arguments, [[index] for index in range(total)])
        result.fingerprint = "execution"
        prepared = [self.application.worker.preprocess_shard([index]) for index in range(total)]
        for index in cached:
            self.application.commit_shard(
                self.path / "shards" / f"{index:06d}", prepared[index], _vectors(prepared[index]),
                {"instance_id": "previous-job-actor", "inference_seconds": 5.0}, "revision", "execution",
            )
        self.actors.extend(_SerialActor(f"current-actor-{index}", self.calls) for index in range(count))
        result.actors = self.actors
        result.prepare_shard = SimpleNamespace(remote=lambda ids: prepared[ids[0]])
        result.inference_barrier = SimpleNamespace(remote=self._barrier)
        result.started = result.model_ready = 0.0
        return result

    def _close(self):
        """Release every blocked rendezvous before joining serial executors."""
        for barrier in self.barriers:
            barrier.gate.abort()
            barrier.finish.abort()
        for actor in self.actors:
            actor.executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def session(application, tmp_path, monkeypatch):
    """Exercise the real session and checkpoint writer with observable queues.

    Args:
        application: Imported reference application under test.
        tmp_path: Pytest directory for committed checkpoint files.
        monkeypatch: Pytest fixture that isolates the synthetic Ray module.
    Returns:
        An iterator yielding the session factory and observable queues.
    Raises:
        None.
    """
    factory = _SessionFactory(application, tmp_path)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        get=_resolve_future, kill=lambda *args, **kwargs: None, is_initialized=lambda: False,
    ))
    yield SimpleNamespace(create=factory._create, calls=factory.calls,
                          barriers=factory.barriers, path=tmp_path)
    factory._close()


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
    """Verify call counts, real Parquet completion, and sparse-wave liveness.

    Args:
        application: Imported reference application under test.
        session: Session factory with observable serial actor queues.
        actors: Number of synthetic serial actors.
        cached: Indices with committed checkpoint fixtures.
        total: Total number of single-record shards.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
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
    """Never spend GPU work or overwrite corrupt committed bytes during resume.

    Args:
        application: Imported reference application under test.
        session: Session factory with observable serial actor queues.
        damage: Committed-checkpoint fault to inject.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
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
    """A partial write without its commit marker does not count as a cache hit.

    Args:
        application: Imported reference application under test.
        session: Session factory with observable serial actor queues.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
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
    """Sparse support must not waive the established multi-actor overlap gate.

    Args:
        session: Session factory with observable serial actor queues.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
    run = session.create(2, {0})
    run._commit_first_shard()
    run._infer_remaining_shards()
    run.barrier.status.remote = lambda: {"participants": 2, "overlap": False, "events": []}
    with pytest.raises(ValueError, match="never overlapped"):
        run._check_concurrency()


@pytest.fixture
def live_helpers():
    """Load the manual gate without enabling its remote runtime prerequisite.

    Args:
    Returns:
        The imported live-gate module.
    Raises:
        ImportError: The isolated reference module cannot be imported.
    """
    source = Path(__file__).parents[1] / "e2e/test_ray_clip_checkpoint_live.py"
    spec = importlib.util.spec_from_file_location("checkpoint_live_receipts", source)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    return live


def test_live_receipts_are_writable_private_and_immutable(tmp_path, live_helpers):
    """Exercise actual receipt I/O before allocating any live validation GPUs.

    Args:
        tmp_path: Pytest directory for private test files.
        live_helpers: Imported native live-gate helpers without cluster access.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
    receipt = tmp_path / "receipt.json"
    previous = os.umask(0)
    try:
        live_helpers._write_private(receipt, '{"actual_call_count": 3}')
    finally:
        os.umask(previous)
    assert receipt.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        live_helpers._write_private(receipt, "replacement")
    assert json.loads(receipt.read_text()) == {"actual_call_count": 3}


def _partial_archive():
    """Represent an incomplete remote result without invoking a model."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        item = tarfile.TarInfo("result/partial.txt")
        content = b"incomplete fixture; no inference"
        item.size, item.mode = len(content), 0o644
        stream.addfile(item, io.BytesIO(content))

    return archive.getvalue(), content


def test_live_startup_failure_preserves_native_evidence_and_partial_output(tmp_path, monkeypatch, live_helpers):
    """An early terminal failure must retain logs, status and its partial files.

    Args:
        tmp_path: Pytest directory for private test files.
        monkeypatch: Pytest fixture for isolated imports and environment changes.
        live_helpers: Imported native live-gate helpers without cluster access.
    Returns:
        None.
    Raises:
        AssertionError: The observed checkpoint or lifecycle contract differs.
    """
    status = SimpleNamespace(is_terminal=lambda: True)
    stopped, remote_calls = [], []
    client = SimpleNamespace(
        submit_job=lambda **kwargs: None,
        get_job_status=lambda identifier: status,
        get_job_logs=lambda identifier: "model startup failed",
        get_job_info=lambda identifier: SimpleNamespace(status=status),
        stop_job=lambda identifier: stopped.append(identifier),
    )
    monkeypatch.setitem(sys.modules, "ray.job_submission", SimpleNamespace(JobSubmissionClient=lambda address: client))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"evidence_dir": str(tmp_path / "evidence"), "remote_root": "/synthetic-output", "address": "unused"}))
    config.chmod(0o600)
    monkeypatch.setenv("NPA_RAY_CLIP_CHECKPOINT_LIVE_CONFIG", str(config))
    archive, content = _partial_archive()

    def remote(config, code):
        """Expose a synthetic partial tree through the same binary download path."""
        remote_calls.append(code)
        return b"true" if "is_dir()" in code else archive

    monkeypatch.setattr(live_helpers, "_remote", remote)
    with pytest.raises(AssertionError):
        live_helpers.test_native_clip_stop_resume_sparse_and_invalid_checkpoints(tmp_path)
    _verify_startup_evidence(tmp_path / "evidence", stopped, remote_calls, content)


def _verify_startup_evidence(evidence_root, stopped, remote_calls, content):
    """Check that early native failures retain private status, logs and output."""
    assert len(stopped) == 1 and len(remote_calls) == 2
    evidence, = evidence_root.iterdir()
    log, = evidence.glob("*-final.log")
    receipt, = evidence.glob("*-final.json")
    partial, = evidence.glob("*-cleanup-snapshot/result/partial.txt")
    assert log.read_text() == "model startup failed"
    assert "status" in json.loads(receipt.read_text())
    assert partial.read_bytes() == content
    assert json.loads((evidence / "job-cleanup.json").read_text())["errors"] == []
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in evidence.rglob("*") if path.is_file())

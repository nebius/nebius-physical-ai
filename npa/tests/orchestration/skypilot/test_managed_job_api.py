"""Inert pinned-shape request/result transport checks; no SkyPilot imports."""

import json
import os
from types import SimpleNamespace

import pytest

from npa.orchestration.skypilot import _managed_job_api as bridge


REQUEST = "00000000-0000-4000-8000-000000000001"
CONTEXT = "c" * 64


def _payload():
    return {
        "attempt": "synthetic-attempt",
        "context": CONTEXT,
        "yaml": "name: group\nexecution: parallel\n---\nname: first\n---\nname: second\n",
        "task_count": 2,
        "controller": "synthetic-controller",
        "name": "synthetic-job",
        "secrets": [("SYNTHETIC_SECRET", "synthetic-private-value")],
    }


def _fixture(result=([41], SimpleNamespace(cluster_name="synthetic-controller"))):
    calls = []
    dag = SimpleNamespace(tasks=[object(), object()])

    def launch(loaded, **kwargs):
        assert loaded is dag
        calls.append(("launch", kwargs))
        return REQUEST

    def get(request):
        assert request == REQUEST
        calls.append(("get", request))
        if isinstance(result, BaseException):
            raise result
        return result

    def loader(text, **kwargs):
        calls.append(("dag", text, kwargs))
        return dag

    sky = SimpleNamespace(
        __version__=bridge.SKY_VERSION,
        __commit__=bridge.SKY_SOURCE_COMMIT,
        jobs=SimpleNamespace(launch=launch),
        get=get,
    )
    return sky, loader, calls


def _run(
    tmp_path,
    *,
    result=([41], SimpleNamespace(cluster_name="synthetic-controller")),
    context=None,
):
    sky, loader, calls = _fixture(result)
    path = tmp_path / "observations.jsonl"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        bridge.run_bridge(
            _payload(),
            descriptor=descriptor,
            sky=sky,
            load_dag=loader,
            verify_context=context or (lambda: CONTEXT),
        )
    finally:
        os.close(descriptor)
    return path, calls


def test_native_success_preserves_full_dag_and_full_request_id(tmp_path):
    path, calls = _run(tmp_path)
    observed = bridge.decode_observation(
        path.read_bytes(), attempt="synthetic-attempt", context=CONTEXT, task_count=2
    )
    assert observed.request_id == REQUEST and observed.job_id == "41"
    assert observed.task_ids == (0, 1)
    assert [call[0] for call in calls] == ["dag", "launch", "get"]
    assert calls[0][1] == _payload()["yaml"]
    assert calls[0][2] == {"secrets_overrides": _payload()["secrets"]}
    assert b"synthetic-private-value" not in path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "result",
    [
        None,
        (),
        ([41, 42], None),
        ([True], None),
        (41, None),
        ([0], None),
        ([41], None),
        ([41], SimpleNamespace(cluster_name="foreign")),
        RuntimeError("allocation before error"),
        KeyboardInterrupt(),
    ],
)
def test_failure_preserves_request_without_result_or_retry(tmp_path, result):
    with pytest.raises(
        (bridge.NativeResultUnavailable, RuntimeError, KeyboardInterrupt)
    ):
        _run(tmp_path, result=result)
    rows = (tmp_path / "observations.jsonl").read_bytes().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["request_id"] == REQUEST


@pytest.mark.parametrize("changed_at", range(4))
def test_context_change_refuses_at_each_boundary(tmp_path, changed_at):
    count = 0

    def context():
        nonlocal count
        count += 1
        return "d" * 64 if count - 1 == changed_at else CONTEXT

    with pytest.raises(bridge.NativeResultUnavailable):
        _run(tmp_path, context=context)
    data = (tmp_path / "observations.jsonl").read_bytes()
    assert len(data.splitlines()) == (0 if changed_at < 2 else 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "extra",
        "foreign_attempt",
        "foreign_context",
        "prefix",
        "task_missing",
        "task_duplicate",
        "boolean_id",
        "duplicate_key",
    ],
)
def test_ipc_cannot_mint_or_adopt_an_identity(tmp_path, mutation):
    path, _ = _run(tmp_path)
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if mutation == "partial":
        data = data[: data.index(b"\n") + 1]
    elif mutation == "extra":
        data += data
    elif mutation == "duplicate_key":
        data = data.replace(
            b'"event": "result"', b'"event": "result", "event": "result"'
        )
    else:
        changes = {
            "foreign_attempt": ("attempt", "foreign"),
            "foreign_context": ("context", "d" * 64),
            "prefix": ("request_id", REQUEST[:8]),
            "task_missing": ("task_ids", [0]),
            "task_duplicate": ("task_ids", [0, 0]),
            "boolean_id": ("job_id", True),
        }
        key, value = changes[mutation]
        rows[1][key] = value
        data = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    with pytest.raises(bridge.NativeResultUnavailable):
        bridge.decode_observation(
            data, attempt="synthetic-attempt", context=CONTEXT, task_count=2
        )


def test_initial_id_loss_retains_empty_observation_and_never_gets(tmp_path):
    sky, loader, calls = _fixture()
    sky.jobs.launch = lambda *args, **kwargs: "prefix"
    descriptor = os.open(
        tmp_path / "observation", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        with pytest.raises(bridge.NativeResultUnavailable):
            bridge.run_bridge(
                _payload(),
                descriptor=descriptor,
                sky=sky,
                load_dag=loader,
                verify_context=lambda: CONTEXT,
            )
    finally:
        os.close(descriptor)
    assert [call[0] for call in calls] == ["dag"]
    assert not (tmp_path / "observation").read_bytes()

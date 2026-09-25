"""Inert pinned-shape request/result transport checks; no SkyPilot imports."""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from npa.orchestration.skypilot import _managed_job_api as bridge


REQUEST = "00000000-0000-4000-8000-000000000001"
CONTEXT = "c" * 64


@pytest.mark.parametrize(
    "failure",
    (
        "",
        "present",
        "api-drift",
        "wrong-handle",
        "job-id",
        "no-incarnation",
        "request-failure",
        "missing-provider",
        "invalid-provider",
    ),
)
def test_controller_provision_receipt_never_grants_managed_job_ownership(failure):
    name = "sky-jobs-controller-synthetic"
    provider = "sky-jobs-controller-shortened-synthetic"
    calls, rows = [], []
    task = object()
    dag = object()

    def launch(value, **kwargs):
        assert value is task
        assert kwargs == {
            "cluster_name": name,
            "retry_until_up": True,
            "fast": True,
            "_disable_controller_check": True,
            "_need_confirmation": False,
        }
        calls.append("controller-launch")
        return REQUEST

    def get(request):
        assert request == REQUEST
        calls.append("get")
        if failure == "request-failure":
            raise RuntimeError("synthetic request failure")
        return (
            1 if failure == "job-id" else None,
            SimpleNamespace(
                cluster_name="foreign" if failure == "wrong-handle" else name,
                cluster_name_on_cloud=(
                    None
                    if failure == "missing-provider"
                    else "bad/name"
                    if failure == "invalid-provider"
                    else provider
                ),
            ),
        )

    sky = SimpleNamespace(
        __version__=bridge.SKY_VERSION,
        __commit__=bridge.SKY_SOURCE_COMMIT,
        launch=launch,
        get=get,
    )
    payload = {
        "attempt": "synthetic-ensure",
        "context": CONTEXT,
        "controller": "",
        "yaml": "synthetic",
    }

    def incarnation(logical, cloud):
        assert logical == name and cloud == provider
        return "" if failure == "no-incarnation" else "e" * 64

    kwargs = dict(
        sky=sky,
        load_dag=lambda _text: dag,
        prepare_controller=lambda value: (name, task) if value is dag else None,
        verify_absent=lambda _name: failure != "present",
        verify_context=lambda: (
            "d" * 64 if failure == "api-drift" and calls else CONTEXT
        ),
        verify_incarnation=incarnation,
        observe=rows.append,
    )
    if failure:
        with pytest.raises((bridge.NativeResultUnavailable, RuntimeError)):
            bridge._ensure_controller_native(payload, **kwargs)
        assert not any(row["event"] == "controller_result" for row in rows)
        assert calls == (
            []
            if failure == "present"
            else ["controller-launch"]
            if failure == "api-drift"
            else ["controller-launch", "get"]
        )
        return
    bridge._ensure_controller_native(payload, **kwargs)
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    assert bridge.decode_controller_observation(
        raw, attempt=payload["attempt"], context=CONTEXT
    ) == (name, provider, "e" * 64)
    missing_provider = [dict(row) for row in rows]
    missing_provider[1].pop("controller_cloud_name")
    with pytest.raises(bridge.NativeResultUnavailable):
        bridge.decode_controller_observation(
            b"".join(json.dumps(row).encode() + b"\n" for row in missing_provider),
            attempt=payload["attempt"],
            context=CONTEXT,
        )
    with pytest.raises(bridge.NativeResultUnavailable):
        bridge.decode_observation(
            raw, attempt=payload["attempt"], context=CONTEXT, task_count=1
        )
    with pytest.raises(bridge.NativeResultUnavailable):
        bridge.decode_controller_observation(
            raw, attempt="other-attempt", context=CONTEXT
        )
    with pytest.raises(bridge.NativeResultUnavailable):
        bridge.decode_controller_observation(
            raw.splitlines(keepends=True)[0],
            attempt=payload["attempt"],
            context=CONTEXT,
        )


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


@pytest.mark.parametrize(
    "mutation",
    ("", "missing", "duplicate", "wrong", "mutable", "wrong-task", "wrong-placeholder"),
)
def test_robotwin_native_image_is_bound_before_any_launch(mutation):
    image = "registry.example/team/npa-robotwin@sha256:" + "a" * 64
    secret_name = "NPA_INTERNAL_BYOF_ROBOTWIN_IMAGE"

    class Resource:
        def __init__(self, image_id):
            self.image_id = image_id

        def copy(self, *, image_id):
            return Resource(image_id)

    task = SimpleNamespace(
        name="byof-solution-smoke-robotwin-rtxpro",
        resources={Resource({None: f"docker:${{{secret_name}}}"})},
    )
    task.set_resources = lambda resource: setattr(task, "resources", {resource})
    dag = SimpleNamespace(tasks=[task])
    sky, _loader, calls = _fixture()
    sky.jobs.launch = lambda loaded, **kwargs: (
        calls.append(("launch", next(iter(loaded.tasks[0].resources)).image_id))
        or REQUEST
    )
    payload = {
        **_payload(),
        "task_count": 1,
        "secrets": [(secret_name, image)],
        "robotwin_image_sha256": hashlib.sha256(image.encode()).hexdigest(),
    }
    if mutation == "missing":
        payload["secrets"] = []
    elif mutation == "duplicate":
        payload["secrets"] *= 2
    elif mutation == "wrong":
        payload["secrets"] = [(secret_name, image[:-1] + "b")]
    elif mutation == "mutable":
        payload["secrets"] = [
            (secret_name, "registry.example/team/npa-robotwin:latest")
        ]
    elif mutation == "wrong-task":
        task.name = "another-task"
    elif mutation == "wrong-placeholder":
        task.resources = {Resource({None: "docker:${ANOTHER_IMAGE}"})}
    observations = []
    if mutation:
        with pytest.raises(bridge.NativeResultUnavailable):
            bridge._launch_native(
                payload,
                sky=sky,
                load_dag=lambda *args, **kwargs: dag,
                observe=observations.append,
                verify_context=lambda: CONTEXT,
            )
        assert calls == [] and observations == []
    else:
        bridge._launch_native(
            payload,
            sky=sky,
            load_dag=lambda *args, **kwargs: dag,
            observe=observations.append,
            verify_context=lambda: CONTEXT,
        )
        assert calls[0] == ("launch", {None: "docker:" + image})
        assert image not in json.dumps(observations)
        assert payload["yaml"] == _payload()["yaml"]

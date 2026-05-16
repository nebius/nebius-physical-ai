from __future__ import annotations

import subprocess

import pytest

from npa.clients.serverless import (
    AuthError,
    EndpointInfo,
    EndpointNotFoundError,
    EndpointSpec,
    EndpointStatus,
    JobInfo,
    NotEnoughResourcesError,
    QuotaError,
    ServerlessClient,
    ServerlessClientError,
    _NER_PATTERNS,
    _classify_error,
)


def _result(args: list[str], code: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args, code, stdout=stdout, stderr=stderr)


def _endpoint_json(
    *,
    endpoint_id: str = "endpoint-1",
    name: str = "cosmos",
    parent_id: str = "project-1",
    state: str = "RUNNING",
    url: str = "https://cosmos.example",
) -> str:
    return (
        "{"
        f'"metadata": {{"id": "{endpoint_id}", "name": "{name}", "parent_id": "{parent_id}"}},'
        f'"status": {{"state": "{state}", "url": "{url}"}}'
        "}"
    )


def _job_json(state: str = "SUCCEEDED") -> str:
    return f'{{"metadata": {{"id": "job-1", "name": "cosmos-train", "parent_id": "project-1"}}, "status": {{"state": "{state}", "output_uris": ["s3://bucket/jobs/cosmos-train/"], "message": "tail"}}}}'


def _create_job(client: ServerlessClient, **kwargs):
    defaults = {"project_id": "project-1", "name": "cosmos-train", "image": "registry/cosmos:cuda12", "command": "bash -lc train", "gpu_type": "gpu-h200-sxm", "gpu_count": 1, "output_path": "s3://bucket/jobs/cosmos-train/"}
    return client.create_job(**(defaults | kwargs))


def test_create_endpoint_builds_expected_args() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, _endpoint_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
        container_ports=[8080],
        env={"MODEL": "cosmos"},
        volumes=["s3://bucket:/data:rw"],
        disk_size="500Gi",
        shm_size="32Gi",
        working_dir="/workspace",
        preemptible=True,
    )

    info = client.create_endpoint(spec)

    assert info == EndpointInfo(
        id="endpoint-1",
        name="cosmos",
        project_id="project-1",
        status=EndpointStatus.RUNNING,
        url="https://cosmos.example",
        raw=info.raw,
    )
    assert calls == [
        [
            "nebius",
            "ai",
            "endpoint",
            "create",
            "--parent-id",
            "project-1",
            "--name",
            "cosmos",
            "--image",
            "registry/cosmos:cuda12",
            "--auth",
            "none",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--public",
            "--container-port",
            "8080",
            "--env",
            "MODEL=cosmos",
            "--volume",
            "s3://bucket:/data:rw",
            "--disk-size",
            "500Gi",
            "--shm-size",
            "32Gi",
            "--working-dir",
            "/workspace",
            "--preemptible",
            "--format",
            "json",
        ]
    ]


def test_create_endpoint_extra_env_none_adds_no_env_flags() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, _endpoint_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    client.create_endpoint(spec, extra_env=None)

    assert "--env" not in calls[0]


def test_create_endpoint_extra_env_adds_repeatable_env_flags() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, _endpoint_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    client.create_endpoint(spec, extra_env={"FOO": "bar"})

    assert ["--env", "FOO=bar"] == calls[0][-4:-2]


def test_create_endpoint_extra_env_masks_sensitive_debug_logs(caplog) -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, _endpoint_json())

    caplog.set_level("DEBUG", logger="npa.clients.serverless")
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    client.create_endpoint(spec, extra_env={"HF_TOKEN": "hf_secret_value"})

    assert "HF_TOKEN=hf_secret_value" in calls[0]
    assert "hf_secret_value" not in caplog.text
    assert "HF_TOKEN=<redacted>" in caplog.text


def test_create_endpoint_resolves_endpoint_when_cli_returns_operation() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "create":
            return _result(
                args,
                0,
                '[1/1] waiting for operation "op-1" over resource "endpoint-1" to complete\n'
                '{"id":"op-1","resource_id":"endpoint-1","status":{}}',
            )
        if args[3] == "list":
            return _result(args, 0, '{"items": [' + _endpoint_json() + "]}")
        raise AssertionError(args)

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    info = client.create_endpoint(spec)

    assert info.id == "endpoint-1"
    assert info.name == "cosmos"
    assert [call[3] for call in calls] == ["create", "list"]


def test_create_endpoint_resolves_by_name_when_cli_returns_progress_only() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "create":
            return _result(args, 0, "[1/1] waiting for operation to complete\n")
        if args[3] == "list":
            return _result(args, 0, '{"items": [' + _endpoint_json() + "]}")
        raise AssertionError(args)

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    info = client.create_endpoint(spec)

    assert info.id == "endpoint-1"
    assert info.name == "cosmos"
    assert [call[3] for call in calls] == ["create", "list"]


def test_create_endpoint_requires_image() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    with pytest.raises(ValueError, match="Container image"):
        client.create_endpoint(spec)


@pytest.mark.parametrize("key", ["HF_TOKEN", "NGC_API_KEY", "PASSWORD", "AWS_SECRET_ACCESS_KEY"])
def test_create_endpoint_refuses_secret_like_env_vars(key: str) -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
        env={key: "secret-value"},
    )

    with pytest.raises(ValueError, match=key):
        client.create_endpoint(spec)


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("permission denied for project", AuthError),
        ("403 forbidden", AuthError),
        ("endpoint not found", EndpointNotFoundError),
        ("rpc error: code = NotFound desc = not found request = 3649f403-e540", EndpointNotFoundError),
        ("404 resource does not exist", EndpointNotFoundError),
        ("quota exceeded: max endpoints reached", QuotaError),
        ("some other failure", ServerlessClientError),
    ],
)
def test_classify_error(stderr: str, expected: type[ServerlessClientError]) -> None:
    assert _classify_error(1, stderr) is expected


@pytest.mark.parametrize("pattern", _NER_PATTERNS)
def test_create_endpoint_ner_patterns_raise_not_enough_resources(pattern: str) -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 1, stderr=f"serverless allocation failed: {pattern}")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    with pytest.raises(NotEnoughResourcesError):
        client.create_endpoint(spec)


def test_create_endpoint_auth_error_not_classified_as_ner() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 1, stderr="permission denied for project project-1")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    with pytest.raises(AuthError):
        client.create_endpoint(spec)


def test_error_hierarchy() -> None:
    assert issubclass(QuotaError, NotEnoughResourcesError)
    assert not issubclass(AuthError, NotEnoughResourcesError)
    assert not issubclass(EndpointNotFoundError, NotEnoughResourcesError)


def test_not_enough_resources_carries_project_and_platform() -> None:
    def fake_runner(args, **kwargs):
        return _result(
            args,
            1,
            stderr='no platform found with name = "gpu-h200-sxm"',
        )

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="8gpu-128vcpu-1600gb",
    )

    with pytest.raises(NotEnoughResourcesError) as exc_info:
        client.create_endpoint(spec)

    exc = exc_info.value
    assert exc.project_id == "project-1"
    assert exc.platform == "gpu-h200-sxm"
    assert exc.preset == "8gpu-128vcpu-1600gb"
    assert exc.gpu_count == 8
    assert exc.raw_stderr == 'no platform found with name = "gpu-h200-sxm"'


def test_quota_error_has_quota_classification() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 1, stderr="quota exceeded for project project-1")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(QuotaError) as exc_info:
        client.list_endpoints("project-1")

    assert exc_info.value.error_class == "quota"
    assert "quota increase" in exc_info.value.suggested_alternatives[0]


def test_auth_error_has_hint() -> None:
    err = AuthError("permission denied")

    assert err.hint
    assert "nebius profile create" in err.hint


def test_endpoint_not_found_carries_endpoint_metadata() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, "{}")
        return _result(args, 0, "{}")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(EndpointNotFoundError) as exc_info:
        client.get_endpoint("project-1", "cosmos")

    assert exc_info.value.project_id == "project-1"
    assert exc_info.value.endpoint_name == "cosmos"
    assert exc_info.value.endpoint_id == "cosmos"


def test_suggested_alternatives_populated_for_capacity_error() -> None:
    err = NotEnoughResourcesError("capacity", error_class="capacity")

    assert err.suggested_alternatives == []

    def fake_runner(args, **kwargs):
        return _result(args, 1, stderr="insufficient capacity")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(NotEnoughResourcesError) as exc_info:
        client.list_endpoints("project-1")

    assert "Retry in a few minutes" in exc_info.value.suggested_alternatives


def test_str_returns_just_message_for_backward_compat() -> None:
    err = NotEnoughResourcesError("plain message", project_id="project-1")

    assert str(err) == "plain message"
    assert err.args == ("plain message",)


def test_classify_queue_state_running_returns_running() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)

    assert client.classify_queue_state(JobInfo("job-1", "train", "project-1", status="running")) == "running"


def test_classify_queue_state_recently_queued_returns_scheduled() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    job = JobInfo("job-1", "train", "project-1", status="queued", queued_for_seconds=30)

    assert client.classify_queue_state(job) == "scheduled"


def test_classify_queue_state_long_queued_returns_waiting_for_capacity() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    job = JobInfo("job-1", "train", "project-1", status="queued", queued_for_seconds=600)

    assert client.classify_queue_state(job) == "waiting_for_capacity"


def test_classify_queue_state_with_explicit_scheduling_state() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    job = JobInfo(
        "job-1",
        "train",
        "project-1",
        status="queued",
        scheduling_state="insufficient capacity",
    )

    assert client.classify_queue_state(job) == "waiting_for_capacity"


def test_classify_queue_state_respects_threshold_override() -> None:
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)
    job = JobInfo("job-1", "train", "project-1", status="queued", queued_for_seconds=11)

    assert client.classify_queue_state(job, threshold_seconds=10) == "waiting_for_capacity"


def test_job_parser_derives_queue_metadata() -> None:
    raw = (
        '{"metadata": {"id": "job-1", "name": "train", "parent_id": "project-1", '
        '"createdAt": "2000-01-01T00:00:00Z"}, '
        '"spec": {"platform": "gpu-h200-sxm", "preset": "8gpu-128vcpu-1600gb"}, '
        '"status": {"state": "QUEUED", "schedulingState": "accepted"}}'
    )
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda *a, **k: None)

    info = client._parse_job_info(raw, project_id="project-1")

    assert info.status == "queued"
    assert info.scheduling_state == "accepted"
    assert info.platform == "gpu-h200-sxm"
    assert info.gpu_count == 8
    assert info.queued_for_seconds > 180


def test_list_endpoints_parses_empty_object_as_empty_list() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 0, "{}")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    assert client.list_endpoints("project-1") == []


def test_list_endpoints_parses_items() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 0, '{"items": [' + _endpoint_json(name="a") + "]}")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    endpoints = client.list_endpoints("project-1")

    assert len(endpoints) == 1
    assert endpoints[0].name == "a"
    assert endpoints[0].status is EndpointStatus.RUNNING


def test_get_endpoint_finds_by_name_from_list() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, '{"items": [' + _endpoint_json(name="cosmos") + "]}")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    info = client.get_endpoint("project-1", "cosmos")

    assert info.id == "endpoint-1"
    assert len(calls) == 1
    assert calls[0][1:4] == ["ai", "endpoint", "list"]


def test_get_endpoint_falls_back_to_get_by_id() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, "{}")
        return _result(args, 0, _endpoint_json(endpoint_id="endpoint-2"))

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    info = client.get_endpoint("project-1", "endpoint-2")

    assert info.id == "endpoint-2"
    assert calls[1][1:] == ["ai", "endpoint", "get", "--id", "endpoint-2", "--format", "json"]


def test_get_endpoint_not_found_raises() -> None:
    def fake_runner(args, **kwargs):
        if args[3] == "list":
            return _result(args, 0, "{}")
        return _result(args, 1, stderr="endpoint not found")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(EndpointNotFoundError):
        client.get_endpoint("project-1", "missing")


def test_delete_endpoint_deletes_resolved_id() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, '{"items": [' + _endpoint_json() + "]}")
        return _result(args, 0, "")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    client.delete_endpoint("project-1", "cosmos")

    assert calls[-1][1:] == ["ai", "endpoint", "delete", "--id", "endpoint-1"]


def test_delete_endpoint_is_idempotent_for_missing_endpoint() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, "{}")
        return _result(args, 1, stderr="not found")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    client.delete_endpoint("project-1", "missing")

    assert len(calls) == 2


@pytest.mark.parametrize("method,command", [("stop_endpoint", "stop"), ("start_endpoint", "start")])
def test_start_stop_endpoint(method: str, command: str) -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, '{"items": [' + _endpoint_json() + "]}")
        return _result(args, 0, _endpoint_json(state="STOPPED" if command == "stop" else "RUNNING"))

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    info = getattr(client, method)("project-1", "cosmos")

    assert calls[-1][1:] == ["ai", "endpoint", command, "--id", "endpoint-1", "--format", "json"]
    assert info.id == "endpoint-1"


def test_get_endpoint_logs_uses_resolved_id() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "list":
            return _result(args, 0, '{"items": [' + _endpoint_json() + "]}")
        return _result(args, 0, "line 1\nline 2\n")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    logs = client.get_endpoint_logs("project-1", "cosmos", tail=20, since="10m")

    assert logs == "line 1\nline 2\n"
    assert calls[-1][1:] == ["ai", "endpoint", "logs", "endpoint-1", "--tail", "20", "--since", "10m"]


def test_wait_for_running_polls_until_running() -> None:
    states = iter(["CREATING", "RUNNING"])

    def fake_runner(args, **kwargs):
        state = next(states)
        return _result(args, 0, '{"items": [' + _endpoint_json(state=state) + "]}")

    client = ServerlessClient(
        nebius_bin="nebius",
        subprocess_runner=fake_runner,
        sleep=lambda seconds: None,
    )

    info = client.wait_for_running("project-1", "cosmos", timeout=10, poll_interval=0)

    assert info.status is EndpointStatus.RUNNING


def test_wait_for_running_raises_on_failed_status() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 0, '{"items": [' + _endpoint_json(state="FAILED") + "]}")

    client = ServerlessClient(
        nebius_bin="nebius",
        subprocess_runner=fake_runner,
        sleep=lambda seconds: None,
    )

    with pytest.raises(ServerlessClientError, match="terminal status failed"):
        client.wait_for_running("project-1", "cosmos", timeout=10, poll_interval=0)


def test_wait_for_running_times_out() -> None:
    def fake_runner(args, **kwargs):
        return _result(args, 0, '{"items": [' + _endpoint_json(state="CREATING") + "]}")

    client = ServerlessClient(
        nebius_bin="nebius",
        subprocess_runner=fake_runner,
        sleep=lambda seconds: None,
    )

    with pytest.raises(TimeoutError, match="did not reach running"):
        client.wait_for_running("project-1", "cosmos", timeout=0, poll_interval=0)


def test_subprocess_env_is_not_used_for_nonsecret_args() -> None:
    observed_kwargs = {}

    def fake_runner(args, **kwargs):
        observed_kwargs.update(kwargs)
        return _result(args, 0, _endpoint_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    spec = EndpointSpec(
        name="cosmos",
        project_id="project-1",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
    )

    client.create_endpoint(spec)

    assert observed_kwargs["env"] is None


def test_create_job_builds_args_and_masks_extra_env(caplog) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_runner(args, **kwargs):
        calls.append((args, kwargs))
        return _result(args, 0, _job_json())

    caplog.set_level("DEBUG", logger="npa.clients.serverless")
    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    info = _create_job(client, env={"MODE": "smoke"}, extra_env={"HF_TOKEN": "hf_secret_value"})

    assert info.status == "succeeded"
    assert info.output_uris == ("s3://bucket/jobs/cosmos-train/",)
    assert calls[0][0][1:4] == ["ai", "job", "create"]
    assert calls[0][1]["timeout"] == 300
    assert "--subnet-id" not in calls[0][0]
    _create_job(client, subnet_id="vpcsubnet-1")
    assert calls[1][0][calls[1][0].index("--subnet-id") + 1] == "vpcsubnet-1"
    assert "MODE=smoke" in calls[0][0]
    assert "HF_TOKEN=hf_secret_value" in calls[0][0]
    assert "hf_secret_value" not in caplog.text
    assert "HF_TOKEN=<redacted>" in caplog.text


def test_create_job_parser_fallback_resolves_by_name_and_ner_raises() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "create":
            return _result(args, 0, "[1/1] waiting\n")
        if args[3] == "get":
            return _result(args, 1, stderr="not found")
        return _result(args, 0, _job_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    assert _create_job(client).id == "job-1"
    assert [call[3] for call in calls] == ["create", "get", "get-by-name"]

    def fake_runner(args, **kwargs):
        return _result(args, 1, stderr="scheduling failed: no GPU available")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(NotEnoughResourcesError):
        _create_job(client, gpu_count=8)


def test_create_job_timeout_fallback_resolves_by_name() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        if args[3] == "create":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if args[3] == "get":
            return _result(args, 1, stderr="not found")
        return _result(args, 0, _job_json())

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    assert _create_job(client).id == "job-1"
    assert [call[3] for call in calls] == ["create", "get", "get-by-name"]


def test_create_job_timeout_fallback_raises_when_lookup_fails() -> None:
    def fake_runner(args, **kwargs):
        if args[3] == "create":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return _result(args, 1, stderr="not found")

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)

    with pytest.raises(ServerlessClientError, match="lookup-by-name recovery failed"):
        _create_job(client)


def test_job_state_cancel_idempotency_and_poll() -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        return _result(args, 0, _job_json(state="SUCCEEDED"))

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner)
    assert client.cancel_job("job-1", "project-1").status == "succeeded"
    assert [call[3] for call in calls] == ["get"]

    states = iter([ServerlessClientError("temporary"), _job_json(state="RUNNING"), _job_json()])

    def fake_runner(args, **kwargs):
        value = next(states)
        if isinstance(value, ServerlessClientError):
            return _result(args, 1, stderr="temporary")
        return _result(args, 0, value)

    client = ServerlessClient(nebius_bin="nebius", subprocess_runner=fake_runner, sleep=lambda seconds: None)
    assert client.poll_job("job-1", "project-1", interval_s=0, ceiling_s=10).status == "succeeded"

    running = ServerlessClient(nebius_bin="nebius", subprocess_runner=lambda args, **kwargs: _result(args, 0, _job_json(state="RUNNING")), sleep=lambda seconds: None)
    with pytest.raises(TimeoutError, match="did not finish"):
        running.poll_job("job-1", "project-1", interval_s=0, ceiling_s=0)

    interrupt_calls: list[list[str]] = []

    def interrupted_runner(args, **kwargs):
        interrupt_calls.append(args)
        return _result(args, 0, _job_json(state="RUNNING" if args[3] == "get" else "CANCELLED"))

    interrupted = ServerlessClient(nebius_bin="nebius", subprocess_runner=interrupted_runner, sleep=lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        interrupted.poll_job("job-1", "project-1", interval_s=1, ceiling_s=10)
    assert any(call[3] == "cancel" for call in interrupt_calls)

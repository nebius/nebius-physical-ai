from __future__ import annotations

import subprocess

import pytest

from npa.clients.serverless import (
    AuthError,
    EndpointInfo,
    EndpointNotFoundError,
    EndpointSpec,
    EndpointStatus,
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

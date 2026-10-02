"""Resolved storage scopes preserve selected credentials without process-global drift."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from threading import Barrier

import pytest

from npa.orchestration.npa_workflow.submit_credentials import STORAGE_ENDPOINT_ENV_NAMES
from npa.orchestration.skypilot.cleanup import sky_environment
from npa.orchestration.skypilot.storage_context import call_with_workflow_storage
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config


@pytest.fixture
def storage_state():
    return WorkflowS3Config(
        bucket="example-bucket",
        prefix="workflow/run",
        endpoint_url="https://storage.example.invalid",
        aws_access_key_id="selected-access",
        aws_secret_access_key="selected-secret",
        project="selected-project",
    )


def test_selected_pair_replaces_entire_ambient_storage_identity_only(
    monkeypatch, storage_state
):
    for name in (
        *STORAGE_ENDPOINT_ENV_NAMES,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "NPA_SKYPILOT_PROJECT",
    ):
        monkeypatch.setenv(name, "other-principal")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/frozen-source")
    monkeypatch.setenv("NPA_S3_PREFIX", "unchanged-prefix")
    before = dict(os.environ)
    observed = call_with_workflow_storage(storage_state, sky_environment)
    assert observed["AWS_ACCESS_KEY_ID"] == storage_state.aws_access_key_id
    assert observed["AWS_SECRET_ACCESS_KEY"] == storage_state.aws_secret_access_key
    assert all(
        observed[name] == storage_state.endpoint_url
        for name in STORAGE_ENDPOINT_ENV_NAMES
    )
    assert observed["AWS_SESSION_TOKEN"] == observed["AWS_SECURITY_TOKEN"] == ""
    assert observed["NPA_SKYPILOT_PROJECT"] == storage_state.project
    changed = {
        *STORAGE_ENDPOINT_ENV_NAMES,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "NPA_SKYPILOT_PROJECT",
    }
    assert {k: v for k, v in observed.items() if k not in changed} == {
        k: v for k, v in before.items() if k not in changed
    }
    assert dict(os.environ) == before and sky_environment() == before


def test_nested_failure_restores_outer_scope_and_caller(storage_state):
    before = dict(os.environ)
    other = replace(
        storage_state,
        aws_access_key_id="other-access",
        aws_secret_access_key="other-secret",
    )

    def fail():
        assert sky_environment()["AWS_ACCESS_KEY_ID"] == "other-access"
        raise RuntimeError("fixture operation failed")

    def outer():
        with pytest.raises(RuntimeError, match="fixture operation failed"):
            call_with_workflow_storage(other, fail)
        assert sky_environment()["AWS_ACCESS_KEY_ID"] == "selected-access"

    call_with_workflow_storage(storage_state, outer)
    assert dict(os.environ) == before and sky_environment() == before


def test_parallel_project_scopes_do_not_exchange_credentials(storage_state):
    barrier = Barrier(2)
    states = [
        storage_state,
        replace(
            storage_state,
            aws_access_key_id="other-access",
            aws_secret_access_key="other-secret",
            project="other-project",
        ),
    ]

    def observe():
        barrier.wait()
        environment = sky_environment()
        return environment["AWS_ACCESS_KEY_ID"], environment["NPA_SKYPILOT_PROJECT"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(call_with_workflow_storage, state, observe) for state in states
        ]
        assert [future.result() for future in futures] == [
            (state.aws_access_key_id, state.project) for state in states
        ]


@pytest.mark.parametrize(
    "missing", ["aws_access_key_id", "aws_secret_access_key", "endpoint_url"]
)
def test_partial_selected_identity_cannot_start_a_controller(storage_state, missing):
    with pytest.raises(ValueError, match="Resolved workflow storage requires"):
        call_with_workflow_storage(
            replace(storage_state, **{missing: ""}),
            lambda: pytest.fail("must not execute"),
        )


def test_explicit_owner_environment_takes_precedence(storage_state):
    owned = {"AWS_ACCESS_KEY_ID": "owned-transaction", "NPA_SRC_S3_URI": "unchanged"}
    assert (
        call_with_workflow_storage(storage_state, sky_environment, environment=owned)
        == owned
    )


@pytest.mark.parametrize("project", [None, ""])
def test_uri_resolved_storage_without_project_clears_ambient_alias(
    storage_state, monkeypatch, project
):
    monkeypatch.setenv("NPA_SKYPILOT_PROJECT", "unrelated-project")
    observed = call_with_workflow_storage(
        replace(storage_state, project=project), sky_environment
    )
    assert observed["NPA_SKYPILOT_PROJECT"] == ""
    assert all(isinstance(value, str) for value in observed.values())
    assert os.environ["NPA_SKYPILOT_PROJECT"] == "unrelated-project"

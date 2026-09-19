# npa: publication-enforcement=libero
from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow.submit_credentials import resolve_submit_credentials


def _configured(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda project=None: SimpleNamespace(
            checkpoint_bucket=f"{project}-bucket",
            endpoint_url=f"https://storage.{project}.nebius.cloud",
            aws_access_key_id="project-ak",
            aws_secret_access_key="project-sk",
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **kwargs: SimpleNamespace(
            tokens={
                "HF_TOKEN": "configured-hf",
                "NEBIUS_TOKEN_FACTORY_KEY": "configured-tf",
                "NGC_API_KEY": "configured-ngc",
            },
            hf_token="configured-hf",
            s3_access_key_id="shared-ak",
            s3_secret_access_key="shared-sk",
            s3_endpoint="https://storage.shared.nebius.cloud",
            s3_bucket="shared-bucket",
        ),
    )


def test_selected_project_endpoint_and_credentials_are_resolved(monkeypatch) -> None:
    _configured(monkeypatch)
    context = resolve_submit_credentials(
        project="test-rtx",
        requested=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        environ={},
    )

    assert context.endpoint_url == "https://storage.test-rtx.nebius.cloud"
    assert context.bucket == "test-rtx-bucket"
    assert context.secret_values == {
        "AWS_ACCESS_KEY_ID": "project-ak",
        "AWS_SECRET_ACCESS_KEY": "project-sk",
        "HF_TOKEN": "configured-hf",
    }
    assert context.access_key_id == "project-ak"
    assert context.secret_access_key == "project-sk"
    assert "project-sk" not in repr(context)
    assert "project-ak" not in repr(context)


def test_normal_resolution_keeps_session_token_with_selected_pair(monkeypatch) -> None:
    _configured(monkeypatch)
    context = resolve_submit_credentials(
        project="test-rtx",
        requested=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ),
        environ={
            "AWS_ACCESS_KEY_ID": "environment-ak",
            "AWS_SECRET_ACCESS_KEY": "environment-sk",
            "AWS_SESSION_TOKEN": "environment-session",
            "AWS_ENDPOINT_URL": "https://storage.example",
        },
        workflow_env={"AWS_SESSION_TOKEN": "different-workflow-session"},
    )

    assert context.access_key_id == "environment-ak"
    assert context.secret_access_key == "environment-sk"
    assert context.session_token == "environment-session"
    assert context.secret_values["AWS_SESSION_TOKEN"] == "environment-session"
    assert "different-workflow-session" not in context.secret_values.values()


def test_explicit_environment_and_endpoint_take_precedence(monkeypatch) -> None:
    _configured(monkeypatch)
    context = resolve_submit_credentials(
        project="test-rtx",
        explicit_endpoint="storage.explicit.nebius.cloud",
        requested=("HF_TOKEN", "NGC_API_KEY"),
        environ={"HF_TOKEN": "explicit-hf", "NGC_API_KEY": "explicit-ngc"},
    )

    assert context.endpoint_url == "https://storage.explicit.nebius.cloud"
    assert context.secret_values == {
        "HF_TOKEN": "explicit-hf",
        "NGC_API_KEY": "explicit-ngc",
    }


def test_missing_requested_secret_is_reported_without_values(monkeypatch) -> None:
    _configured(monkeypatch)
    context = resolve_submit_credentials(
        project="test-rtx", requested=("UNSUPPORTED_MISSING",), environ={}
    )

    assert context.secret_values == {}
    assert context.missing == ("UNSUPPORTED_MISSING",)


def test_supported_configured_tokens_and_custom_declared_token_are_forwarded(
    monkeypatch,
) -> None:
    _configured(monkeypatch)
    original = resolve_submit_credentials.__globals__["load_credentials"]

    def configured_with_custom(**kwargs):
        value = original(**kwargs)
        value.tokens["CUSTOM_WORKFLOW_TOKEN"] = "configured-custom"
        return value

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        configured_with_custom,
    )
    context = resolve_submit_credentials(
        project="test-rtx",
        requested=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "NGC_API_KEY",
            "CUSTOM_WORKFLOW_TOKEN",
        ),
        environ={},
    )

    assert context.secret_values == {
        "NEBIUS_TOKEN_FACTORY_KEY": "configured-tf",
        "NGC_API_KEY": "configured-ngc",
        "CUSTOM_WORKFLOW_TOKEN": "configured-custom",
    }


@pytest.mark.parametrize(
    "missing", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
)
def test_control_plane_authorized_triplet_rejects_every_partial_environment(
    monkeypatch, missing
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda *_args, **_kwargs: pytest.fail("saved project storage is forbidden"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **_kwargs: pytest.fail("configured credentials are forbidden"),
    )
    environ = {
        "AWS_ACCESS_KEY_ID": "control-plane-access",
        "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }
    environ.pop(missing)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    for name in {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    } - set(environ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="complete.*triplet"):
        resolve_submit_credentials(
            environ=environ, require_process_environment_triplet=True
        )


def test_control_plane_authorized_triplet_never_reads_or_mixes_saved_credentials(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda *_args, **_kwargs: pytest.fail("saved project storage is forbidden"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **_kwargs: pytest.fail("configured credentials are forbidden"),
    )
    environ = {
        "AWS_ACCESS_KEY_ID": "caller-supplied-access",
        "AWS_SECRET_ACCESS_KEY": "caller-supplied-secret",
        "AWS_SESSION_TOKEN": "caller-supplied-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }
    for name, value in {
        "AWS_ACCESS_KEY_ID": "control-plane-access",
        "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }.items():
        monkeypatch.setenv(name, value)
    context = resolve_submit_credentials(
        environ=environ,
        workflow_env={
            "AWS_ACCESS_KEY_ID": "hostile-workflow-access",
            "AWS_SECRET_ACCESS_KEY": "hostile-workflow-secret",
            "AWS_SESSION_TOKEN": "hostile-workflow-session",
        },
        require_process_environment_triplet=True,
    )

    assert context.access_key_id == "control-plane-access"
    assert context.secret_access_key == "control-plane-secret"
    assert context.session_token == "control-plane-session"
    assert context.endpoint_url == "https://storage.example"
    assert "control-plane-session" not in repr(context)
    assert "hostile-workflow" not in repr(context)


def test_control_plane_triplet_normalizes_and_deduplicates_requested_names(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda *_args, **_kwargs: pytest.fail("saved project storage was accessed"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **_kwargs: pytest.fail("configured credentials were accessed"),
    )
    environ = {
        "AWS_ACCESS_KEY_ID": "control-plane-access",
        "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    context = resolve_submit_credentials(
        requested=(
            " AWS_SESSION_TOKEN ",
            "AWS_SESSION_TOKEN",
            "",
            " AWS_ACCESS_KEY_ID ",
        ),
        environ=environ,
        require_process_environment_triplet=True,
    )

    assert context.secret_values == {
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ACCESS_KEY_ID": "control-plane-access",
    }
    assert context.missing == ()


def test_control_plane_triplet_resolves_process_environment_endpoint_requests(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda *_args, **_kwargs: pytest.fail("saved project storage was accessed"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **_kwargs: pytest.fail("configured credentials were accessed"),
    )
    environ = {
        "AWS_ACCESS_KEY_ID": "control-plane-access",
        "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    context = resolve_submit_credentials(
        requested=(" AWS_ENDPOINT_URL ", "AWS_ENDPOINT_URL_S3"),
        environ={
            "AWS_ACCESS_KEY_ID": "control-plane-access",
            "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
            "AWS_SESSION_TOKEN": "control-plane-session",
            "AWS_ENDPOINT_URL": "https://storage.example",
        },
        require_process_environment_triplet=True,
    )

    assert context.secret_values == {
        "AWS_ENDPOINT_URL": "https://storage.example",
        "AWS_ENDPOINT_URL_S3": "https://storage.example",
    }
    assert context.missing == ()


@pytest.mark.parametrize(
    "requested",
    [
        ("NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",),
        ("NPA_LIBERO_AUTHENTICATED_CALLER_B64",),
        ("NPA_LIBERO_CUSTOMER_TERMS_ACK",),
    ],
)
def test_control_plane_triplet_rejects_customer_evidence_requests(
    requested, monkeypatch
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        lambda *_args, **_kwargs: pytest.fail("saved project storage was accessed"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.load_credentials",
        lambda **_kwargs: pytest.fail("configured credentials were accessed"),
    )
    for name, value in {
        "AWS_ACCESS_KEY_ID": "control-plane-access",
        "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
        "AWS_SESSION_TOKEN": "control-plane-session",
        "AWS_ENDPOINT_URL": "https://storage.example",
    }.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(
        ValueError, match="only the process-environment storage credential triplet"
    ):
        resolve_submit_credentials(
            environ={
                "AWS_ACCESS_KEY_ID": "control-plane-access",
                "AWS_SECRET_ACCESS_KEY": "control-plane-secret",
                "AWS_SESSION_TOKEN": "control-plane-session",
                "AWS_ENDPOINT_URL": "https://storage.example",
                requested[0]: "customer-evidence",
            },
            requested=requested,
            require_process_environment_triplet=True,
        )

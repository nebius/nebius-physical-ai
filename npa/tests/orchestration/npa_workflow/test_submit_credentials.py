from __future__ import annotations

from types import SimpleNamespace

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

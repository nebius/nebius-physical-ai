from __future__ import annotations

from email.message import Message
import json
from urllib.error import HTTPError

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.rerun import RerunHostError, _verify_browser_cors
from npa.clients.config import EnvironmentConfig, StorageConfig
from npa.clients.nebius import (
    BucketCorsPlan,
    RERUN_BROWSER_CORS_RULE_ID,
    RERUN_BROWSER_ORIGIN,
    apply_bucket_rerun_cors,
    plan_bucket_rerun_cors,
    rerun_browser_cors_rule,
)


runner = CliRunner()


def _bucket(*rules: dict, version: str = "7") -> dict:
    return {
        "metadata": {
            "id": "storagebucket-synthetic",
            "name": "bucket-synthetic",
            "resource_version": version,
        },
        "spec": {"cors": {"rules": list(rules)}},
    }


def test_plan_adds_least_privilege_rule_and_preserves_unrelated(mocker) -> None:
    unrelated = {
        "id": "existing-app",
        "allowed_origins": ["https://viewer.example"],
        "allowed_methods": ["GET", "HEAD"],
        "allowed_headers": ["Content-Type"],
    }
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name", return_value=_bucket(unrelated)
    )

    plan = plan_bucket_rerun_cors("project-synthetic", "bucket-synthetic")

    assert plan.changed is True
    assert plan.preserved_rule_count == 1
    assert list(plan.desired_rules) == [unrelated, rerun_browser_cors_rule()]
    assert rerun_browser_cors_rule() == {
        "id": RERUN_BROWSER_CORS_RULE_ID,
        "allowed_origins": [RERUN_BROWSER_ORIGIN],
        "allowed_methods": ["GET"],
        "allowed_headers": ["Range"],
        "expose_headers": ["Accept-Ranges", "Content-Length", "Content-Range"],
        "max_age_seconds": 3600,
    }


def test_plan_replaces_only_stale_npa_rule(mocker) -> None:
    unrelated = {
        "id": "existing-app",
        "allowed_origins": ["https://viewer.example"],
        "allowed_methods": ["GET"],
        "allowed_headers": ["Range"],
    }
    stale = {
        "id": RERUN_BROWSER_CORS_RULE_ID,
        "allowed_origins": ["*"],
        "allowed_methods": ["PUT"],
        "allowed_headers": ["*"],
    }
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value=_bucket(unrelated, stale),
    )

    plan = plan_bucket_rerun_cors("project-synthetic", "bucket-synthetic")

    assert list(plan.desired_rules) == [unrelated, rerun_browser_cors_rule()]


def test_plan_accepts_an_existing_compatible_rule_without_mutation(mocker) -> None:
    compatible = {
        "id": "operator-managed",
        "allowed_origins": [RERUN_BROWSER_ORIGIN],
        "allowed_methods": ["GET", "HEAD"],
        "allowed_headers": ["Range", "Content-Type"],
        "expose_headers": ["Accept-Ranges", "Content-Length", "Content-Range"],
    }
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name", return_value=_bucket(compatible)
    )

    plan = plan_bucket_rerun_cors("project-synthetic", "bucket-synthetic")

    assert plan.changed is False
    assert plan.desired_rules == plan.current_rules


def test_apply_uses_optimistic_version_and_read_back_verification(mocker) -> None:
    unrelated = {
        "id": "existing-app",
        "allowed_origins": ["https://viewer.example"],
        "allowed_methods": ["GET"],
        "allowed_headers": ["Range"],
    }
    updated = _bucket(unrelated, rerun_browser_cors_rule(), version="8")
    get_bucket = mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        side_effect=[_bucket(unrelated), updated],
    )
    run = mocker.patch("npa.clients.nebius._run_json", return_value={})

    result = apply_bucket_rerun_cors("project-synthetic", "bucket-synthetic")

    assert result.changed is True
    assert get_bucket.call_count == 2
    argv = run.call_args.args[0]
    assert argv[:6] == [
        "storage",
        "bucket",
        "update",
        "--id",
        "storagebucket-synthetic",
        "--cors-rules",
    ]
    assert argv[-2:] == ["--resource-version", "7"]
    assert "existing-app" in argv[6]
    assert RERUN_BROWSER_CORS_RULE_ID in argv[6]


class _CorsResponse:
    status = 200

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_browser_probe_sends_and_validates_range_preflight(mocker) -> None:
    response = _CorsResponse(
        {
            "Access-Control-Allow-Origin": RERUN_BROWSER_ORIGIN,
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Range",
            "Access-Control-Expose-Headers": "Accept-Ranges, Content-Length, Content-Range",
        }
    )
    urlopen = mocker.patch("npa.cli.rerun.urlopen", return_value=response)

    _verify_browser_cors(
        "https://storage.example/object?signature=secret",
        bucket="bucket-synthetic",
        project="project-synthetic",
    )

    request = urlopen.call_args.args[0]
    assert request.get_method() == "OPTIONS"
    assert request.get_header("Origin") == RERUN_BROWSER_ORIGIN
    assert request.get_header("Access-control-request-method") == "GET"
    assert request.get_header("Access-control-request-headers") == "Range"


def test_browser_probe_403_gives_admin_remedy_without_signed_url(mocker) -> None:
    signed_url = "https://storage.example/object?signature=secret"
    mocker.patch(
        "npa.cli.rerun.urlopen",
        side_effect=HTTPError(signed_url, 403, "Forbidden", Message(), None),
    )

    with pytest.raises(RerunHostError) as raised:
        _verify_browser_cors(
            signed_url,
            bucket="bucket-synthetic",
            project="project-synthetic",
        )

    message = str(raised.value)
    assert "HTTP 403" in message
    assert "npa storage bucket cors --apply" in message
    assert "scoped S3 object key" in message
    assert "rerun <recording.rrd>" in message
    assert signed_url not in message
    assert "signature=secret" not in message


def test_storage_cors_cli_is_plan_only_by_default(mocker) -> None:
    plan = BucketCorsPlan(
        bucket_id="storagebucket-synthetic",
        resource_version="7",
        current_rules=(),
        desired_rules=(rerun_browser_cors_rule(),),
        changed=True,
    )
    configure = mocker.patch(
        "npa.rerun.configure_browser_cors", return_value=plan
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "cors",
            "--project",
            "project-synthetic",
            "--name",
            "bucket-synthetic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "needs an additive bucket-admin update" in result.output
    configure.assert_called_once_with(
        target_bucket="bucket-synthetic",
        target_project="project-synthetic",
        target_project_id="",
        apply=False,
    )


def test_storage_cors_cli_apply_has_stable_json(mocker) -> None:
    plan = BucketCorsPlan(
        bucket_id="storagebucket-synthetic",
        resource_version="8",
        current_rules=(),
        desired_rules=(rerun_browser_cors_rule(),),
        changed=True,
    )
    mocker.patch("npa.rerun.configure_browser_cors", return_value=plan)

    result = runner.invoke(
        app,
        ["storage", "bucket", "cors", "--apply", "--output", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "allowed_origin": RERUN_BROWSER_ORIGIN,
        "apply": True,
        "cors_rule_id": RERUN_BROWSER_CORS_RULE_ID,
        "preserved_rule_count": 0,
        "status": "updated",
    }
    assert result.stdout == (
        json.dumps(json.loads(result.stdout), indent=2, sort_keys=True) + "\n"
    )
    assert "command diagnostics were separated from JSON stdout" not in result.stderr


def test_sdk_routes_configured_project_to_control_plane(mocker) -> None:
    from npa import rerun

    mocker.patch(
        "npa.rerun.resolve_environment",
        return_value=EnvironmentConfig(
            project_id="project-synthetic",
            tenant_id="tenant-synthetic",
            region="us-central1",
        ),
    )
    mocker.patch(
        "npa.rerun.resolve_project_storage",
        return_value=StorageConfig(
            checkpoint_bucket="s3://bucket-synthetic/prefix",
            endpoint_url="https://storage.example",
        ),
    )
    operation = mocker.patch(
        "npa.rerun.apply_bucket_rerun_cors",
        return_value=BucketCorsPlan("id", "7", (), (), False),
    )

    rerun.configure_browser_cors(target_project="project-synthetic", apply=True)

    operation.assert_called_once_with("project-synthetic", "bucket-synthetic")


@pytest.mark.parametrize("target_project", [None, "typoed-alias"])
def test_sdk_missing_project_configuration_is_actionable(
    mocker, target_project: str | None
) -> None:
    from npa import rerun

    mocker.patch("npa.rerun.resolve_environment", return_value=None)
    storage = mocker.patch("npa.rerun.resolve_project_storage")
    operation = mocker.patch("npa.rerun.plan_bucket_rerun_cors")

    with pytest.raises(ValueError, match="Target project is not configured") as raised:
        rerun.configure_browser_cors(target_project=target_project)

    message = str(raised.value)
    assert "target_project_id" in message
    assert "npa configure" in message
    if target_project:
        assert repr(target_project) in message
    storage.assert_not_called()
    operation.assert_not_called()


def test_sdk_explicit_project_id_and_bucket_do_not_resolve_scoped_storage(mocker) -> None:
    from npa import rerun

    storage = mocker.patch(
        "npa.rerun.resolve_project_storage",
        side_effect=AssertionError("explicit target must bypass storage resolution"),
    )
    environment = mocker.patch(
        "npa.rerun.resolve_environment",
        side_effect=AssertionError("explicit target must bypass environment resolution"),
    )
    operation = mocker.patch(
        "npa.rerun.plan_bucket_rerun_cors",
        return_value=BucketCorsPlan("id", "7", (), (), False),
    )

    rerun.configure_browser_cors(
        target_project_id="project-synthetic",
        target_bucket="bucket-synthetic",
    )

    storage.assert_not_called()
    environment.assert_not_called()
    operation.assert_called_once_with("project-synthetic", "bucket-synthetic")

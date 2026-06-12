from __future__ import annotations

from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import (
    ServerlessConfig,
    SSHConfig,
    StorageConfig,
    WorkbenchConfig,
)

runner = CliRunner()

OUTPUT_URI = "s3://bucket/results/out.mp4"
REMOTE_PATH = "/opt/cosmos-data/outputs/out.mp4"


def _cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://cosmos:8080",
        ssh=SSHConfig(host="cosmos", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        hf_token="",
        app_status="",
        runtime="vm",
        serverless=ServerlessConfig(),
    )


def _client_error(code: str, message: str | None = None) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message or code}}, "PutObject"
    )


def _upload_failed(code: str) -> S3UploadFailedError:
    # boto3's managed upload wraps the S3 ClientError as S3UploadFailedError.
    client_error = _client_error(code)
    try:
        raise client_error
    except ClientError as exc:
        err = S3UploadFailedError(
            f"Failed to upload local to bucket/key: {exc}"
        )
        err.__context__ = exc
        return err


def _patch_infer(mocker, store, *, output_path: str | None = REMOTE_PATH, ssh=None):
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    completed = {"job_id": "job-1", "status": "completed"}
    if output_path:
        completed["output_path"] = output_path
    http.job_status.return_value = completed
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh or mocker.MagicMock())
    mocker.patch("npa.cli.cosmos._storage_client_for_config", return_value=store)
    return http


def _assert_clean_upload_failure(result, *, expect_vm_path: bool) -> None:
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"uncaught exception escaped the command: {result.exception!r}"
    )
    assert result.exit_code == 1
    assert "Generation succeeded but uploading" in result.output
    assert "not a generation failure" in result.output
    assert "Traceback" not in result.output
    if expect_vm_path:
        assert REMOTE_PATH in result.output
        assert "still on the Cosmos VM" in result.output
    else:
        assert "still on the Cosmos VM" not in result.output


DIRECT_UPLOAD_ERRORS = [
    pytest.param(_upload_failed("AccessDenied"), id="access_denied"),
    pytest.param(NoCredentialsError(), id="no_credentials"),
]


@pytest.mark.parametrize("injected", DIRECT_UPLOAD_ERRORS)
def test_infer_remote_upload_error_is_clean(injected, mocker) -> None:
    """Generation completes, the remote-output upload fails -> clean error."""
    store = mocker.MagicMock()
    store.upload_file.side_effect = injected
    _patch_infer(mocker, store)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "infer",
            "--prompt", "a red cube on a table",
            "--output-path", OUTPUT_URI,
        ],
    )

    _assert_clean_upload_failure(result, expect_vm_path=True)


def test_infer_inline_upload_error_is_clean(mocker) -> None:
    """No VM-side artifact (inline result) -> clean error, no false VM path."""
    store = mocker.MagicMock()
    store.upload_file.side_effect = _upload_failed("AccessDenied")
    _patch_infer(mocker, store, output_path=None)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "infer",
            "--prompt", "a red cube on a table",
            "--output-path", "s3://bucket/results/out.json",
        ],
    )

    _assert_clean_upload_failure(result, expect_vm_path=False)


def test_infer_allow_host_creds_does_not_fall_back_for_managed_upload(mocker) -> None:
    """--allow-host-creds does not fall back for a managed-upload denial (still clean)."""
    store = mocker.MagicMock()
    store.upload_file.side_effect = _upload_failed("AccessDenied")
    ssh = mocker.MagicMock()
    _patch_infer(mocker, store, ssh=ssh)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "infer",
            "--prompt", "a red cube on a table",
            "--output-path", OUTPUT_URI,
            "--allow-host-creds",
        ],
    )

    _assert_clean_upload_failure(result, expect_vm_path=True)
    ssh.run_or_raise.assert_not_called()

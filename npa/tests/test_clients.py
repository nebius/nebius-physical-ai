from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from npa.clients.config import SSHConfig
from npa.clients.http import HTTPClient, ServerError
from npa.clients import nebius
from npa.clients.nebius import NebiusError
from npa.clients.ssh import SSHClient, SSHError, SSHTimeoutError, format_remote_failure
from npa.clients.storage import StorageClient, StorageError, _parse_bucket_uri


def test_http_client_builds_request_and_returns_json(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    response.json.return_value = {"status": "ok"}
    request = mocker.patch("httpx.request", return_value=response)

    client = HTTPClient("http://server/", timeout=10, retries=1)

    assert client.health() == {"status": "ok"}
    request.assert_called_once_with(
        "GET", "http://server/health", json=None, timeout=10
    )


def test_http_client_fetches_job_status(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    response.json.return_value = {"job_id": "job/1", "status": "completed"}
    request = mocker.patch("httpx.request", return_value=response)

    assert HTTPClient("http://server").job_status("job/1", timeout=3.0) == {
        "job_id": "job/1",
        "status": "completed",
    }
    request.assert_called_once_with(
        "GET", "http://server/jobs/job%2F1", json=None, timeout=3.0
    )


def test_http_client_maps_client_and_server_errors(mocker) -> None:
    response = mocker.MagicMock(status_code=404, text="missing")
    mocker.patch("httpx.request", return_value=response)

    with pytest.raises(ServerError, match="Client error 404"):
        HTTPClient("http://server", retries=1).status()

    response.status_code = 500
    response.text = "boom"
    with pytest.raises(ServerError, match="Server error 500"):
        HTTPClient("http://server", retries=1).status()


def test_http_client_retries_connect_errors(mocker) -> None:
    ok = mocker.MagicMock(status_code=200)
    ok.json.return_value = {"ok": True}
    request = mocker.patch(
        "httpx.request",
        side_effect=[httpx.ConnectError("no route"), ok],
    )
    sleep = mocker.patch("time.sleep")

    assert HTTPClient("http://server", retries=2).health() == {"ok": True}
    assert request.call_count == 2
    sleep.assert_called_once_with(1)


def test_http_client_wait_healthy_false_on_timeout(mocker) -> None:
    client = HTTPClient("http://server")
    mocker.patch.object(client, "health", side_effect=ServerError("down"))
    values = iter([0.0, 0.2, 0.4])
    mocker.patch("time.monotonic", side_effect=lambda: next(values))
    mocker.patch("time.sleep")

    assert client.wait_healthy(timeout=0.3, interval=0.1) is False


def test_storage_parse_bucket_uri() -> None:
    assert _parse_bucket_uri("s3://bucket/prefix/path") == ("bucket", "prefix/path")

    with pytest.raises(StorageError, match="Expected s3://"):
        _parse_bucket_uri("https://bucket/prefix")


def test_storage_client_requires_endpoint() -> None:
    with pytest.raises(StorageError, match="endpoint URL"):
        StorageClient(endpoint_url="", aws_access_key_id="", aws_secret_access_key="")


def test_storage_client_lists_checkpoints(mock_s3) -> None:
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "checkpoints/job-a/"},
                {"Prefix": "checkpoints/job-b/"},
            ]
        }
    ]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )

    result = client.list_checkpoints("s3://bucket/checkpoints")

    assert result == [
        {"name": "job-a", "uri": "s3://bucket/checkpoints/job-a/"},
        {"name": "job-b", "uri": "s3://bucket/checkpoints/job-b/"},
    ]
    mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
    paginator.paginate.assert_called_once_with(
        Bucket="bucket", Prefix="checkpoints/", Delimiter="/"
    )


def test_storage_client_uploads_and_downloads_directories(
    tmp_path: Path, mock_s3
) -> None:
    local = tmp_path / "local"
    (local / "nested").mkdir(parents=True)
    (local / "nested" / "file.txt").write_text("data")
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "prefix/nested/file.txt"}]},
    ]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )

    uploaded = client.upload_directory(
        str(local), "s3://bucket/base", remote_prefix="run"
    )
    download_dir = tmp_path / "download"
    downloaded = client.download_directory("s3://bucket/prefix", str(download_dir))

    assert uploaded == "s3://bucket/base/run/"
    mock_s3.upload_file.assert_called_once_with(
        str(local / "nested" / "file.txt"),
        "bucket",
        "base/run/nested/file.txt",
    )
    assert downloaded == str(download_dir)
    mock_s3.download_file.assert_called_once_with(
        "bucket",
        "prefix/nested/file.txt",
        str(download_dir / "nested" / "file.txt"),
    )


@pytest.mark.parametrize(
    "key",
    [
        "prefix/../../escape.txt",
        "prefix/nested/../escape.txt",
        "prefix/./escape.txt",
        "prefix//tmp/escape.txt",
        "prefix/nested\\escape.txt",
        "other/escape.txt",
    ],
)
def test_storage_client_rejects_unsafe_directory_object_keys(
    tmp_path: Path, mock_s3, key: str
) -> None:
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [{"Contents": [{"Key": key}]}]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )

    with pytest.raises(StorageError, match="outside|unsafe"):
        client.download_directory("s3://bucket/prefix", str(tmp_path / "download"))

    mock_s3.download_file.assert_not_called()
    assert not (tmp_path / "escape.txt").exists()


def test_storage_client_downloads_object_via_head_object_when_list_is_empty(
    tmp_path: Path, mock_s3
) -> None:
    local = tmp_path / "result.json"
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [{"Contents": []}]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )

    downloaded = client.download_path("s3://bucket/prefix/result.json", str(local))

    assert downloaded == str(local)
    mock_s3.head_object.assert_called_once_with(
        Bucket="bucket", Key="prefix/result.json"
    )
    mock_s3.download_file.assert_called_once_with(
        "bucket", "prefix/result.json", str(local)
    )


def test_storage_client_downloads_exact_object_without_list_or_head(
    tmp_path: Path, mock_s3
) -> None:
    body = mock_s3.get_object.return_value["Body"]
    body.iter_chunks.return_value = [b"checkpoint", b"-bytes"]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )
    local = tmp_path / "model.pt"

    downloaded = client.download_file("s3://bucket/checkpoints/model.pt", str(local))

    assert downloaded == str(local)
    mock_s3.get_object.assert_called_once_with(
        Bucket="bucket", Key="checkpoints/model.pt"
    )
    assert local.read_bytes() == b"checkpoint-bytes"
    body.close.assert_called_once_with()
    mock_s3.head_object.assert_not_called()
    mock_s3.get_paginator.assert_not_called()


def test_storage_client_uploads_and_downloads_files(tmp_path: Path, mock_s3) -> None:
    local = tmp_path / "result.json"
    local.write_text("{}")
    paginator = mock_s3.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "inputs/image.jpg"}]},
    ]
    client = StorageClient(
        endpoint_url="https://storage",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )

    uploaded = client.upload_file(str(local), "s3://bucket/results/result.json")
    downloaded = client.download_path(
        "s3://bucket/inputs/image.jpg", str(tmp_path / "image.jpg")
    )

    assert uploaded == "s3://bucket/results/result.json"
    mock_s3.upload_file.assert_called_once_with(
        str(local),
        "bucket",
        "results/result.json",
    )
    assert downloaded == str(tmp_path / "image.jpg")
    mock_s3.download_file.assert_called_once_with(
        "bucket",
        "inputs/image.jpg",
        str(tmp_path / "image.jpg"),
    )


def test_ssh_connect_uses_paramiko_config(mocker) -> None:
    paramiko_client = mocker.MagicMock()
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    SSHClient(SSHConfig(host="host", user="ubuntu", key_path="~/key"))._connect()

    paramiko_client.set_missing_host_key_policy.assert_called_once()
    paramiko_client.connect.assert_called_once_with(
        hostname="host",
        username="ubuntu",
        key_filename=str(Path("~/key").expanduser()),
        timeout=15,
        look_for_keys=False,
    )


def test_ssh_connect_maps_errors(mocker) -> None:
    paramiko_client = mocker.MagicMock()
    paramiko_client.connect.side_effect = RuntimeError("refused")
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    with pytest.raises(SSHError, match="SSH connection.*failed"):
        SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))._connect()


def test_ssh_run_reads_stdout_stderr_and_closes(mocker) -> None:
    channel = mocker.MagicMock()
    channel.recv.side_effect = [b"hello\n", b""]
    channel.recv_stderr.side_effect = [b"warn\n", b""]
    channel.recv_exit_status.return_value = 0
    transport = mocker.MagicMock()
    transport.open_session.return_value = channel
    paramiko_client = mocker.MagicMock()
    paramiko_client.get_transport.return_value = transport
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    result = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key")).run(
        "echo hello"
    )

    assert result == (0, "hello\n", "warn\n")
    channel.exec_command.assert_called_once_with("echo hello")
    paramiko_client.close.assert_called_once()


def test_ssh_run_timeout_watchdog_bounds_connection_and_command(
    monkeypatch, mocker
) -> None:
    callbacks = []

    class ImmediateTimer:
        daemon = False

        def __init__(self, interval, callback) -> None:
            assert interval == 2.0
            callbacks.append(callback)

        def start(self) -> None:
            callbacks[0]()

        def cancel(self) -> None:
            return None

    paramiko_client = mocker.MagicMock()
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)
    monkeypatch.setattr("npa.clients.ssh.threading.Timer", ImmediateTimer)
    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))

    with pytest.raises(SSHTimeoutError, match="timed out after 2s"):
        client.run("command-containing-secret", timeout=2.0)

    paramiko_client.connect.assert_called_once_with(
        hostname="host",
        username="ubuntu",
        key_filename="key",
        timeout=2.0,
        look_for_keys=False,
        banner_timeout=2.0,
        auth_timeout=2.0,
        channel_timeout=2.0,
    )
    assert paramiko_client.close.call_count >= 1


def test_ssh_run_or_raise_maps_nonzero(mocker) -> None:
    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))
    mocker.patch.object(client, "run", return_value=(7, "", "bad"))

    with pytest.raises(SSHError, match="Command failed"):
        client.run_or_raise("false")


def test_format_remote_failure_compact_by_default(monkeypatch) -> None:
    monkeypatch.delenv("NPA_DEBUG", raising=False)
    stderr = "\n".join(f"line {i}" for i in range(100)) + "\nERROR: 403 gated download"

    msg = format_remote_failure(7, stderr, label="Cosmos install")

    # Compact: label + exit + stderr tail only; older stderr lines are trimmed.
    assert "Command failed (exit 7): Cosmos install" in msg
    assert "ERROR: 403 gated download" in msg
    assert "line 0" not in msg
    assert "NPA_DEBUG=1" in msg
    assert len(msg.splitlines()) < 30


def test_format_remote_failure_debug_lifts_stderr_truncation(monkeypatch) -> None:
    monkeypatch.setenv("NPA_DEBUG", "1")
    stderr = "\n".join(f"line {i}" for i in range(100))

    msg = format_remote_failure(7, stderr, label="Cosmos install")

    # NPA_DEBUG lifts only the stderr truncation; the hint disappears.
    assert "line 0" in msg
    assert "NPA_DEBUG=1" not in msg


def test_format_remote_failure_without_label(monkeypatch) -> None:
    monkeypatch.delenv("NPA_DEBUG", raising=False)
    msg = format_remote_failure(1, "", label=None)
    assert msg.splitlines()[0] == "Command failed (exit 1)"
    assert "stderr: <empty>" in msg


def test_run_or_raise_passes_label_never_command(mocker, monkeypatch) -> None:
    monkeypatch.delenv("NPA_DEBUG", raising=False)
    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))
    mocker.patch.object(client, "run", return_value=(3, "", "boom"))

    with pytest.raises(SSHError) as exc_info:
        client.run_or_raise("bash -lc 'huge script'", label="GR00T install")

    message = str(exc_info.value)
    assert "GR00T install" in message
    assert "huge script" not in message
    assert "boom" in message


def test_ssh_run_or_raise_omits_command_to_avoid_leaking_secrets(mocker) -> None:
    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))
    secret_command = "bash -lc 'export AWS_SECRET_ACCESS_KEY=topsecret && do_install'"
    mocker.patch.object(client, "run", return_value=(1, "", "boom"))

    with pytest.raises(SSHError) as excinfo:
        client.run_or_raise(secret_command)

    message = str(excinfo.value)
    assert "topsecret" not in message
    assert "AWS_SECRET_ACCESS_KEY" not in message
    assert "Command failed (exit 1)" in message
    assert "boom" in message


def test_ssh_download_file_uses_sftp(tmp_path: Path, mocker) -> None:
    sftp = mocker.MagicMock()
    paramiko_client = mocker.MagicMock()
    paramiko_client.open_sftp.return_value = sftp
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    local = tmp_path / "nested" / "out.mp4"
    result = SSHClient(
        SSHConfig(host="host", user="ubuntu", key_path="key")
    ).download_file("/remote/out.mp4", str(local))

    assert result == str(local)
    sftp.get.assert_called_once_with("/remote/out.mp4", str(local))
    sftp.close.assert_called_once()
    paramiko_client.close.assert_called_once()


def test_ssh_private_text_is_owner_only_before_secret_write(mocker) -> None:
    events: list[object] = []

    class RemoteFile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("file_closed")

        def write(self, content: str) -> None:
            events.append(("write", content))

        def flush(self) -> None:
            events.append("flush")

    sftp = mocker.MagicMock()
    sftp.open.side_effect = lambda path, mode: (
        events.append(("open", path, mode)) or RemoteFile()
    )
    sftp.chmod.side_effect = lambda path, mode: events.append(("chmod", path, mode))
    paramiko_client = mocker.MagicMock()
    paramiko_client.open_sftp.return_value = sftp
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))
    assert (
        client.upload_private_text("SECRET-SENTINEL", "/tmp/private") == "/tmp/private"
    )

    assert events[:3] == [
        ("open", "/tmp/private", "wx"),
        ("chmod", "/tmp/private", 0o600),
        ("write", "SECRET-SENTINEL"),
    ]
    sftp.close.assert_called_once()
    paramiko_client.close.assert_called_once()


def test_nebius_run_invokes_cli_and_maps_errors(mocker) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/nebius")
    mocker.patch("npa.clients.nebius._warn_if_nebius_version_mismatch")
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["nebius"], returncode=0, stdout="ok\n", stderr=""
        ),
    )

    assert nebius._run(["iam", "get-access-token"]) == "ok"
    run.assert_called_once()
    call = run.call_args
    assert call.args[0] == ["/usr/bin/nebius", "iam", "get-access-token"]
    assert call.kwargs["capture_output"] is True
    assert call.kwargs["text"] is True
    # A sanitized env is always passed so a stale NEBIUS_IAM_TOKEN cannot shadow
    # the active profile.
    assert isinstance(call.kwargs["env"], dict)

    run.return_value = subprocess.CompletedProcess(
        args=["nebius"], returncode=1, stdout="", stderr="nope\n"
    )
    with pytest.raises(NebiusError, match="failed"):
        nebius._run(["bad"])


def test_nebius_requires_binary(mocker) -> None:
    mocker.patch("shutil.which", return_value=None)

    with pytest.raises(NebiusError, match="not found"):
        nebius._require_nebius()


def test_nebius_warns_once_on_tested_cli_version_mismatch(mocker) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/nebius")
    mocker.patch("npa.clients.nebius.supported_tool_version", return_value="0.12.254")
    mocker.patch("npa.clients.nebius._NEBIUS_VERSION_CHECKED", False)
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["nebius", "version"], returncode=0, stdout="0.12.227\n", stderr=""
        ),
    )

    with pytest.warns(RuntimeWarning, match="0.12.227 is tested-compatible"):
        assert nebius._require_nebius() == "/usr/bin/nebius"

    assert nebius._require_nebius() == "/usr/bin/nebius"
    run.assert_called_once_with(
        ["/usr/bin/nebius", "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_nebius_blocks_untested_cli_version_with_exact_remedy(mocker) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/nebius")
    mocker.patch("npa.clients.nebius._NEBIUS_VERSION_CHECKED", False)
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["nebius", "version"], returncode=0, stdout="0.13.0\n", stderr=""
        ),
    )

    with pytest.raises(NebiusError, match=r"Unsupported Nebius CLI 0\.13\.0") as caught:
        nebius._require_nebius()

    assert "NEBIUS_CLI_VERSION=0.12.254 bash" in str(caught.value)
    assert nebius._NEBIUS_VERSION_CHECKED is False


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess(["nebius", "version"], 1, "", "failed"),
        subprocess.CompletedProcess(["nebius", "version"], 0, "unknown", ""),
    ],
)
def test_nebius_blocks_unverifiable_cli_version_with_remedy(mocker, result) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/nebius")
    mocker.patch("npa.clients.nebius._NEBIUS_VERSION_CHECKED", False)
    mocker.patch("subprocess.run", return_value=result)

    with pytest.raises(NebiusError, match="NEBIUS_CLI_VERSION=0.12.254"):
        nebius._require_nebius()


def test_nebius_run_json_and_token(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=['{"ok": true}', "token"])

    assert nebius._run_json(["cmd"]) == {"ok": True}
    assert nebius.get_iam_token() == "token"


def test_nebius_run_json_empty_payload(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", return_value="")

    assert nebius._run_json(["empty"]) == {}


def test_nebius_iam_token_errors_when_all_sources_missing(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", return_value="")
    mocker.patch("npa.clients.nebius._env_iam_token", return_value="")
    mocker.patch("npa.clients.nebius._candidate_iam_token_files", return_value=[])
    mocker.patch("npa.clients.nebius._metadata_iam_token", return_value="")

    with pytest.raises(NebiusError, match="Unable to resolve IAM token"):
        nebius.get_iam_token()


def test_workbench_credential_resolution_is_agent_independent(mocker, tmp_path) -> None:
    """load_credentials + get_iam_token must work with no agent and no attached SA.

    A workbench/CI machine resolves IAM via its nebius CLI profile only. This
    guards that the agent-VM refactor (attached-SA/metadata token source) never
    became the *only* path.
    """
    from npa.clients.credentials import load_credentials

    # No env token, no token files, no metadata: only the CLI profile answers.
    mocker.patch("npa.clients.nebius._run", return_value="cli-profile-token")
    mocker.patch("npa.clients.nebius._env_iam_token", return_value="")
    mocker.patch("npa.clients.nebius._candidate_iam_token_files", return_value=[])
    mocker.patch("npa.clients.nebius._metadata_iam_token", return_value="")

    creds = load_credentials(path=tmp_path / "credentials.yaml", environ={})
    assert creds is not None  # resolves fine without any agent present
    assert nebius.get_iam_token() == "cli-profile-token"


def test_nebius_iam_token_from_env(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", return_value="")
    mocker.patch("npa.clients.nebius._env_iam_token", return_value="env-token")

    assert nebius.get_iam_token() == "env-token"


def test_nebius_iam_token_from_file_when_cli_unconfigured(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=NebiusError("no profile"))
    mocker.patch("npa.clients.nebius._env_iam_token", return_value="")
    mocker.patch(
        "npa.clients.nebius._candidate_iam_token_files", return_value=["/tmp/token"]
    )
    mocker.patch("npa.clients.nebius._read_iam_token_file", return_value="file-token")

    assert nebius.get_iam_token() == "file-token"


def test_nebius_iam_token_from_metadata_fallback(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=NebiusError("no profile"))
    mocker.patch("npa.clients.nebius._env_iam_token", return_value="")
    mocker.patch("npa.clients.nebius._candidate_iam_token_files", return_value=[])
    mocker.patch("npa.clients.nebius._metadata_iam_token", return_value="meta-token")

    assert nebius.get_iam_token() == "meta-token"


def test_nebius_service_account_reuses_existing(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"metadata": {"id": "sa-id"}},
    )

    created: list[str] = []
    assert (
        nebius.ensure_service_account("project", name="svc", on_created=created.append)
        == "sa-id"
    )
    assert created == []
    run_json.assert_called_once()


def test_nebius_service_account_reuses_id_from_permission_denied(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=nebius.NebiusError(
            "nebius iam service-account get-by-name failed (exit 15):\n"
            "Permission denied PermissionDenied: service iam, "
            "resource ID: serviceaccount-u00example"
        ),
    )
    mocker.patch("npa.clients.nebius._saved_service_account_id", return_value="")

    assert nebius.ensure_service_account("project") == "serviceaccount-u00example"


def test_nebius_service_account_creates_when_not_found_despite_saved(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=[
            nebius.NebiusError(
                "nebius iam service-account get-by-name failed (exit 5):\n"
                "NotFound: 'npa-agent' not found in project"
            ),
            {"metadata": {"id": "serviceaccount-new"}},
        ],
    )
    mocker.patch(
        "npa.clients.nebius._saved_service_account_id",
        return_value="serviceaccount-saved",
    )

    assert (
        nebius.ensure_service_account("project", name="npa-agent")
        == "serviceaccount-new"
    )
    assert run_json.call_count == 2


def test_nebius_service_account_reports_ownership_only_when_created(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=[
            nebius.NebiusError("NotFound: 'lerobot-training' not found"),
            {"metadata": {"id": "serviceaccount-created"}},
        ],
    )
    created: list[str] = []

    account_id = nebius.ensure_service_account("project", on_created=created.append)

    assert account_id == "serviceaccount-created"
    assert created == ["serviceaccount-created"]
    assert run_json.call_count == 2


def test_nebius_bootstrap_returns_verifiable_storage_account_ownership(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")

    def create_account(
        project_id, name="lerobot-training", *, description="", on_created=None
    ):
        if on_created:
            on_created("serviceaccount-created")
        return "serviceaccount-created"

    mocker.patch(
        "npa.clients.nebius.ensure_service_account", side_effect=create_account
    )
    mocker.patch("npa.clients.nebius.ensure_bucket")
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"id": "bucket-id"}},
    )
    mocker.patch(
        "npa.clients.nebius._existing_editors_binding",
        side_effect=nebius.NebiusError(
            "PermissionDenied: tenant-wide editors inventory"
        ),
    )
    mocker.patch(
        "npa.clients.nebius.ensure_storage_capability_binding",
        return_value=nebius.StorageIamBindingEvidence(
            nebius.IamBindingState.CREATED,
            nebius.STORAGE_RUNTIME_ROLE,
            "bucket-id",
            "group-id",
            nebius.STORAGE_BINDING_GROUP_NAME,
        ),
    )
    mocker.patch("npa.clients.nebius.ensure_access_key", return_value=("key", "secret"))

    result = nebius.bootstrap_environment("project", "tenant", "eu-north1")

    assert result["service_account_id"] == "serviceaccount-created"
    assert result["service_account_name"] == "lerobot-training"
    assert result["service_account_project_id"] == "project"
    assert result["service_account_managed_by"] == "npa"


def test_nebius_saved_storage_credentials_prefers_configured_bucket(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius._saved_service_account_id", return_value="sa-id")
    mocker.patch(
        "npa.clients.credentials.load_credentials",
        return_value=SimpleNamespace(
            s3_access_key_id="key",
            s3_secret_access_key="secret",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_bucket="s3://lerobot-test0123/checkpoints/",
        ),
    )

    result = nebius._saved_storage_credentials(
        project_id="project",
        tenant_id="tenant",
        region="us-central1",
        bucket_name="npa-bucket-default",
        service_account_id="sa-id",
    )

    assert result is not None
    assert result["s3_bucket"] == "lerobot-test0123"


def test_nebius_bootstrap_stops_before_bucket_or_key_when_required_iam_fails(
    mocker,
) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa-id")
    mocker.patch(
        "npa.clients.nebius.ensure_storage_capability_binding",
        side_effect=nebius.NebiusError(
            "rpc error: PermissionDenied desc = No permission"
        ),
    )
    bucket = mocker.patch("npa.clients.nebius.ensure_bucket", return_value="bucket")
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"id": "bucket-id"}},
    )
    mocker.patch(
        "npa.clients.nebius._existing_editors_binding",
        side_effect=nebius.NebiusError(
            "PermissionDenied: tenant-wide editors inventory"
        ),
    )
    key = mocker.patch(
        "npa.clients.nebius.ensure_access_key", return_value=("key", "secret")
    )

    messages: list[str] = []
    with pytest.raises(nebius.NebiusError, match="project-scoped admin permission"):
        nebius.bootstrap_environment(
            "project", "tenant", "eu-north1", on_status=messages.append
        )

    bucket.assert_called_once()
    key.assert_not_called()


def test_nebius_bootstrap_reuses_saved_storage_on_access_key_permission_denied(
    mocker,
) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa-id")
    mocker.patch("npa.clients.nebius.ensure_bucket", return_value="bucket")
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"id": "bucket-id"}},
    )
    mocker.patch(
        "npa.clients.nebius._existing_editors_binding",
        return_value=nebius.StorageIamBindingEvidence(
            nebius.IamBindingState.EXISTING,
            "editor",
            "tenant",
            "editors-id",
            "editors",
            compatibility_fallback=True,
        ),
    )
    mocker.patch(
        "npa.clients.nebius.ensure_access_key",
        side_effect=nebius.NebiusError("Permission denied PermissionDenied"),
    )
    mocker.patch(
        "npa.clients.nebius._saved_storage_credentials",
        return_value={
            "iam_token": "iam",
            "service_account_id": "sa-id",
            "nebius_api_key": "key",
            "nebius_secret_key": "secret",
            "s3_bucket": "bucket",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "nebius_project_id": "project",
            "nebius_region": "eu-north1",
        },
    )

    result = nebius.bootstrap_environment("project", "tenant", "eu-north1")

    assert result["nebius_api_key"] == "key"
    assert result["service_account_id"] == "sa-id"


def test_nebius_find_active_access_key_prefers_requested_name(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._list_access_key_metadata",
        return_value=[
            {
                "metadata": {"id": "other-id", "name": "other"},
                "spec": {"account": {"service_account_id": "sa"}},
                "status": {"state": "ACTIVE"},
            },
            {
                "metadata": {"id": "target-id", "name": "lerobot-access-key"},
                "spec": {
                    "account": {"service_account": {"id": "sa"}},
                    "expires_at": "1970-01-01T00:00:00Z",
                },
                "status": {"state": "ACTIVE"},
            },
        ],
    )

    result = nebius._find_active_access_key(
        "project",
        "sa",
        key_name="lerobot-access-key",
    )

    assert result is not None
    assert result["metadata"]["id"] == "target-id"


def test_access_key_inventory_exposes_only_cli_allowlisted_jsonpath_fields(
    mocker,
) -> None:
    mocker.patch("npa.clients.nebius._iam_profile_args", return_value=([], ""))
    run = mocker.patch(
        "npa.clients.nebius._run",
        side_effect=[
            "accesskey-a",
            "accesskey-a",
            "lerobot-access-key",
            "serviceaccount-a",
            NebiusError("service_account_id is not found"),
            "ACTIVE",
            "1970-01-01T00:00:00Z",
        ],
    )

    items = nebius._list_access_key_metadata("project-a")

    assert items == [
        {
            "metadata": {"id": "accesskey-a", "name": "lerobot-access-key"},
            "spec": {
                "account": {
                    "service_account": {"id": "serviceaccount-a"},
                    "service_account_id": "",
                },
                "expires_at": "1970-01-01T00:00:00Z",
            },
            "status": {"state": "ACTIVE"},
        }
    ]
    args = run.call_args_list[0].args[0]
    command_index = args.index("iam")
    assert args[command_index : command_index + 4] == [
        "iam",
        "v2",
        "access-key",
        "list",
    ]
    # Exercise CI's profile-free command shape deterministically.  Profile
    # prefixing is orthogonal to the secret-safe projection guarded here.
    assert command_index == 0
    assert "--all" in args
    output_format = args[args.index("--format") + 1]
    assert output_format.startswith("jsonpath=")
    assert all(
        field not in output_format.lower()
        for field in ("secret", "credential", "password", "private_key", "token")
    )
    for call in run.call_args_list[1:]:
        projection = call.args[0][call.args[0].index("--format") + 1]
        assert projection.startswith("jsonpath={.")
        assert all(
            field not in projection.lower()
            for field in ("secret", "credential", "password", "private_key", "token")
        )


def test_access_key_inventory_tolerates_mixed_unrelated_entries(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=[
            "accesskey-unrelated\naccesskey-agent",
            "accesskey-unrelated",
            "unrelated",
            NebiusError("id is not found"),
            NebiusError("service_account_id is not found"),
            "ACTIVE",
            "",
            "accesskey-agent",
            "npa-agent-access-key",
            "serviceaccount-agent",
            NebiusError("service_account_id is not found"),
            "ACTIVE",
            "",
        ],
    )

    items = nebius._list_access_key_metadata("project-a")

    assert len(items) == 2
    assert items[0]["spec"]["account"]["service_account"]["id"] == ""
    assert items[1]["spec"]["account"]["service_account"]["id"] == (
        "serviceaccount-agent"
    )


def test_access_key_inventory_rejects_conflicting_service_account_shapes(
    mocker,
) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=[
            "accesskey-a",
            "accesskey-a",
            "key-a",
            "serviceaccount-one",
            "serviceaccount-two",
        ],
    )

    with pytest.raises(NebiusError, match="conflicting service-account identities"):
        nebius._list_access_key_metadata("project-a")


def test_access_key_inventory_rejects_secret_bearing_scalar_response(mocker) -> None:
    secret_value = "provider-sensitive-value"
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=[
            "accesskey-a",
            "accesskey-a",
            json.dumps({"secret": secret_value}),
        ],
    )

    with pytest.raises(NebiusError, match="potentially secret-bearing") as caught:
        nebius._list_access_key_metadata("project-a")

    assert secret_value not in str(caught.value)


@pytest.mark.parametrize(
    "provider_message",
    [
        "jsonpath execution failed: items is not found",
        "jsonpath: cannot range over items because it is null",
        "jsonpath: cannot iterate items because it is nil",
    ],
    ids=["omitted-items", "null-items-range", "nil-items-iterate"],
)
def test_access_key_inventory_treats_missing_or_null_items_as_empty(
    mocker, provider_message
) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=NebiusError(
            "nebius iam v2 access-key failed (exit 1):\n" + provider_message
        ),
    )

    assert nebius._list_access_key_metadata("project-a") == []


def test_access_key_inventory_treats_empty_formatter_output_as_empty(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        return_value="\n<no value>\n",
    )

    assert nebius._list_access_key_metadata("project-a") == []


def test_access_key_inventory_preserves_provider_failures(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=NebiusError("nebius iam v2 access-key failed: PermissionDenied"),
    )

    with pytest.raises(NebiusError, match="PermissionDenied"):
        nebius._list_access_key_metadata("project-a")


def test_access_key_inventory_does_not_swallow_provider_failure_with_items_text(
    mocker,
) -> None:
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=NebiusError(
            "nebius iam v2 access-key failed: PermissionDenied; items is not found"
        ),
    )

    with pytest.raises(NebiusError, match="PermissionDenied"):
        nebius._list_access_key_metadata("project-a")


def test_nebius_error_redaction_keeps_access_key_canaries_out_of_all_outputs(
    mocker, capsys, caplog
) -> None:
    canaries = (
        "NPA_CANARY_SECRET_1_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_2_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_3_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_4_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_5_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_6_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_7_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_8_DO_NOT_DISCLOSE",
    )
    provider_error = json.dumps(
        {
            "status": {
                "secret": canaries[0],
                "secretAccessKey": canaries[1],
            },
            "nested": {
                "credentials": {
                    "aws_secret_access_key": canaries[2],
                    "private-key": canaries[3],
                    "secret_value": canaries[4],
                    "clientSecret": canaries[5],
                    "privateKeyData": canaries[6],
                    "refresh_token": canaries[7],
                },
            },
        }
    )
    mocker.patch("npa.clients.nebius._require_nebius", return_value="nebius")
    mocker.patch(
        "npa.clients.nebius.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=provider_error
        ),
    )

    with pytest.raises(NebiusError) as caught:
        nebius._run(
            [
                "iam",
                "v2",
                "access-key",
                "list",
                "--parent-id",
                "project-a",
                "--format",
                nebius._ACCESS_KEY_LIST_JSONPATH,
            ]
        )

    captured = capsys.readouterr()
    serialized = json.dumps(
        {
            "exception": str(caught.value),
            "args": caught.value.args,
            "stdout": captured.out,
            "stderr": captured.err,
            "logs": caplog.text,
        }
    )
    for canary in canaries:
        assert canary not in serialized
    # The parent ``credentials`` field is itself sensitive, so the complete
    # nested mapping is replaced by one marker rather than one per child.
    assert serialized.count("<redacted>") >= 3


def test_malformed_access_key_allowlist_output_is_discarded_without_echo(
    mocker,
) -> None:
    canary = "NPA_CANARY_SECRET_MALFORMED_DO_NOT_DISCLOSE"
    mocker.patch("npa.clients.nebius._run", return_value=f"accesskey-a\t{canary}")

    with pytest.raises(NebiusError) as caught:
        nebius._list_access_key_metadata("project-a")

    assert canary not in str(caught.value)


def test_nebius_plain_text_diagnostics_redact_nested_secret_field_variants() -> None:
    canaries = (
        "NPA_CANARY_SECRET_TEXT_1_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_TEXT_2_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_TEXT_3_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_TEXT_4_DO_NOT_DISCLOSE",
        "NPA_CANARY_SECRET_TEXT_5_DO_NOT_DISCLOSE",
    )
    diagnostic = (
        f"provider error: clientSecret={canaries[0]} "
        f"privateKeyData: {canaries[1]} refresh_token={canaries[2]}\n"
        f"Authorization: Bearer {canaries[3]}\n"
        "private_key: -----BEGIN PRIVATE KEY-----\n"
        f"{canaries[4]}\n"
        "-----END PRIVATE KEY-----"
    )

    redacted = nebius.redact_nebius_output(diagnostic)

    assert redacted.count("<redacted>") == len(canaries)
    for canary in canaries:
        assert canary not in redacted


def test_nebius_ensure_access_key_does_not_delete_existing_key_without_secret(
    mocker,
) -> None:
    existing = {
        "metadata": {"id": "existing-id", "name": "lerobot-access-key"},
        "spec": {"account": {"service_account_id": "sa"}},
        "status": {"state": "ACTIVE"},
    }
    mocker.patch("npa.clients.nebius._find_active_access_key", return_value=existing)
    run = mocker.patch("npa.clients.nebius._run")
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=[
            {"status": {"aws_access_key_id": "old-access"}},
            NebiusError("secret unavailable"),
            {"metadata": {"id": "new-id"}},
            {"status": {"aws_access_key_id": "new-access"}},
            {"secret": "new-secret"},
        ],
    )

    assert nebius.ensure_access_key("project", "sa") == ("new-access", "new-secret")
    run.assert_not_called()
    create_args = run_json.call_args_list[2].args[0]
    assert create_args[:4] == ["iam", "v2", "access-key", "create"]
    assert create_args[create_args.index("--name") + 1].startswith(
        "lerobot-access-key-"
    )


def test_nebius_bucket_name_and_bootstrap_order(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa")
    bucket = mocker.patch("npa.clients.nebius.ensure_bucket", return_value="bucket")
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"id": "bucket-id"}},
    )
    mocker.patch("npa.clients.nebius._existing_editors_binding", return_value=None)
    narrow = mocker.patch(
        "npa.clients.nebius.ensure_storage_capability_binding",
        return_value=nebius.StorageIamBindingEvidence(
            nebius.IamBindingState.CREATED,
            nebius.STORAGE_RUNTIME_ROLE,
            "bucket-id",
            "group-id",
            nebius.STORAGE_BINDING_GROUP_NAME,
        ),
    )
    mocker.patch("npa.clients.nebius.ensure_access_key", return_value=("key", "secret"))
    statuses: list[str] = []

    result = nebius.bootstrap_environment(
        "project",
        "tenant",
        "eu-north1",
        on_status=statuses.append,
    )

    assert nebius.bucket_name_for("tenant", "project").startswith("npa-bucket-")
    narrow.assert_called_once()
    bucket.assert_called_once()
    assert result["iam_token"] == "iam"
    assert result["s3_endpoint"] == "https://storage.eu-north1.nebius.cloud"
    assert statuses[0] == "Getting IAM access token..."


def test_nebius_bootstrap_uses_explicit_bucket_name(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa")
    bucket = mocker.patch("npa.clients.nebius.ensure_bucket", return_value="chosen")
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"id": "bucket-id"}},
    )
    mocker.patch(
        "npa.clients.nebius._existing_editors_binding",
        return_value=nebius.StorageIamBindingEvidence(
            nebius.IamBindingState.EXISTING,
            "editor",
            "tenant",
            "editors-id",
            "editors",
            compatibility_fallback=True,
        ),
    )
    mocker.patch("npa.clients.nebius.ensure_access_key", return_value=("key", "secret"))

    result = nebius.bootstrap_environment(
        "project",
        "tenant",
        "eu-north1",
        bucket_name="chosen",
        bucket_max_size_bytes=123,
    )

    assert result["s3_bucket"] == "chosen"
    bucket.assert_called_once_with(
        "project",
        "chosen",
        max_size_bytes=123,
        default_storage_class="standard",
        on_created=mocker.ANY,
        allow_existing=True,
    )


def test_nebius_bootstrap_agent_environment_uses_npa_agent_sa(mocker) -> None:
    bootstrap = mocker.patch(
        "npa.clients.nebius.bootstrap_environment",
        return_value={"service_account_id": "sa-agent"},
    )
    mocker.patch("npa.clients.nebius.get_service_account_id_by_name", return_value=None)

    result = nebius.bootstrap_agent_environment("project", "tenant", "eu-north1")

    assert result["service_account_id"] == "sa-agent"
    kwargs = bootstrap.call_args.kwargs
    assert kwargs["service_account_name"] == nebius.AGENT_SERVICE_ACCOUNT_NAME
    assert kwargs["access_key_name"] == nebius.AGENT_ACCESS_KEY_NAME


def test_nebius_agent_bootstrap_reuses_verified_storage_without_access_key_iam(
    mocker,
) -> None:
    verified_storage = {
        "s3_bucket": "configured-bucket",
        "s3_prefix": "artifacts",
        "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        "nebius_api_key": "configured-access",
        "nebius_secret_key": "configured-secret",
    }
    mocker.patch(
        "npa.clients.nebius.get_service_account_id_by_name",
        return_value="serviceaccount-agent",
    )
    service_account = mocker.patch(
        "npa.clients.nebius.ensure_service_account",
        return_value="serviceaccount-agent",
    )
    mocker.patch("npa.clients.nebius.ensure_editors_membership")
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam-token")
    full_bootstrap = mocker.patch("npa.clients.nebius.bootstrap_environment")
    list_keys = mocker.patch("npa.clients.nebius._list_access_key_metadata")
    create_key = mocker.patch("npa.clients.nebius.ensure_access_key")

    result = nebius.bootstrap_agent_environment(
        "project",
        "tenant",
        "eu-north1",
        reuse_storage_credentials=verified_storage,
    )

    assert result["service_account_id"] == "serviceaccount-agent"
    assert result["nebius_api_key"] == "configured-access"
    assert service_account.call_args.kwargs["allow_saved_fallback"] is False
    full_bootstrap.assert_not_called()
    list_keys.assert_not_called()
    create_key.assert_not_called()


def test_agent_bootstrap_records_successful_iam_creation_atomically(
    tmp_path, monkeypatch, mocker
) -> None:
    import yaml

    from npa.clients import credentials

    path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    mocker.patch("npa.clients.nebius.get_service_account_id_by_name", return_value=None)

    def bootstrap(*_args, on_resource_created, **_kwargs):
        on_resource_created(
            "service_account", {"id": "serviceaccount-agent", "name": "npa-agent"}
        )
        on_resource_created(
            "access_key", {"id": "accesskey-agent", "name": "npa-agent-key"}
        )
        return {"service_account_id": "serviceaccount-agent"}

    mocker.patch("npa.clients.nebius.bootstrap_environment", side_effect=bootstrap)

    nebius.bootstrap_agent_environment("project", "tenant", "eu-north1")

    record = yaml.safe_load(path.read_text())["agent_iam"]["projects"]["project"]
    assert record["status"] == "complete"
    assert record["resources"]["service_account"]["id"] == "serviceaccount-agent"
    assert "accesskey-agent" in record["resources"]["access_keys"]


def test_agent_bootstrap_removes_a_rolled_back_key_on_a_reused_account(
    tmp_path, monkeypatch, mocker
) -> None:
    import yaml

    from npa.clients import credentials

    path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    mocker.patch(
        "npa.clients.nebius.get_service_account_id_by_name",
        return_value="serviceaccount-preexisting",
    )

    def bootstrap(*_args, on_resource_created, **_kwargs):
        on_resource_created(
            "access_key", {"id": "accesskey-new", "name": "npa-agent-key"}
        )
        raise NebiusError("provider failed after key creation")

    mocker.patch("npa.clients.nebius.bootstrap_environment", side_effect=bootstrap)
    delete_key = mocker.patch("npa.clients.nebius.delete_access_key")
    delete_account = mocker.patch("npa.clients.nebius.delete_service_account")

    with pytest.raises(NebiusError, match="provider failed"):
        nebius.bootstrap_agent_environment("project", "tenant", "eu-north1")

    delete_key.assert_called_once_with("accesskey-new")
    delete_account.assert_not_called()
    assert "agent_iam" not in (yaml.safe_load(path.read_text()) or {})


def test_resolve_service_account_id_uses_saved_config(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._saved_service_account_id",
        return_value="serviceaccount-saved",
    )
    assert nebius.resolve_service_account_id("project") == "serviceaccount-saved"


def test_resolve_service_account_id_parses_permission_denied_lookup(mocker) -> None:
    mocker.patch("npa.clients.nebius._saved_service_account_id", return_value="")
    mocker.patch(
        "npa.clients.nebius.get_service_account_id_by_name",
        side_effect=lambda _project, name: (
            "serviceaccount-u00s24wzj2wk8z9tqq"
            if name == nebius.DEFAULT_SERVICE_ACCOUNT_NAME
            else None
        ),
    )

    assert nebius.resolve_service_account_id("project-u00zhx4tpr00xh99b28n52") == (
        "serviceaccount-u00s24wzj2wk8z9tqq"
    )


def test_get_service_account_id_by_name_parses_id_from_permission_denied(
    mocker,
) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=nebius.NebiusError(
            "Permission denied PermissionDenied: service iam, "
            "resource ID: serviceaccount-u00s24wzj2wk8z9tqq"
        ),
    )

    assert nebius.get_service_account_id_by_name("project", "lerobot-training") == (
        "serviceaccount-u00s24wzj2wk8z9tqq"
    )


def test_strict_service_account_name_lookup_rejects_ambiguous_empty_success(
    mocker,
) -> None:
    mocker.patch("npa.clients.nebius._run_json", return_value={})

    with pytest.raises(
        NebiusError, match="presence or absence could not be established"
    ):
        nebius.get_service_account_id_by_name(
            "project", "lerobot-training", strict=True
        )


def test_exact_service_account_verification_rejects_ambiguous_empty_success(
    mocker,
) -> None:
    mocker.patch("npa.clients.nebius._run_json", return_value={})

    with pytest.raises(NebiusError, match="no service-account ID"):
        nebius.service_account_exists("serviceaccount-storage")


def test_exact_service_account_identity_verifies_profile_project_and_tenant(
    mocker,
) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=[
            {
                "metadata": {
                    "id": "serviceaccount-storage",
                    "name": "lerobot-training",
                    "parent_id": "project-a",
                }
            },
            {"metadata": {"id": "project-a", "parent_id": "tenant-a"}},
        ],
    )

    identity = nebius.get_service_account_identity(
        "serviceaccount-storage",
        project_id="project-a",
        tenant_id="tenant-a",
        expected_name="lerobot-training",
        profile="operator-profile",
    )

    assert identity == nebius.ServiceAccountIdentity(
        "serviceaccount-storage",
        "lerobot-training",
        "project-a",
        "tenant-a",
        "operator-profile",
    )
    assert run_json.call_args_list[0].args[0][:2] == ["--profile", "operator-profile"]
    assert run_json.call_args_list[1].args[0] == [
        "--profile",
        "operator-profile",
        "iam",
        "project",
        "get",
        "--id",
        "project-a",
    ]


def test_exact_service_account_identity_never_treats_missing_scope_as_absence(
    mocker,
) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "metadata": {
                "id": "serviceaccount-storage",
                "name": "lerobot-training",
            }
        },
    )

    with pytest.raises(NebiusError, match="incomplete service-account identity"):
        nebius.get_service_account_identity(
            "serviceaccount-storage",
            project_id="project-a",
            tenant_id="tenant-a",
            expected_name="lerobot-training",
        )


def test_nebius_bootstrap_agent_environment_falls_back_on_permission_denied(
    mocker,
) -> None:
    mocker.patch(
        "npa.clients.nebius.bootstrap_environment",
        side_effect=nebius.NebiusError("Permission denied PermissionDenied"),
    )
    mocker.patch(
        "npa.clients.nebius.get_service_account_id_by_name", return_value="sa-existing"
    )
    mocker.patch(
        "npa.clients.nebius._saved_storage_credentials",
        return_value={
            "service_account_id": "sa-existing",
            "nebius_api_key": "key",
            "nebius_secret_key": "secret",
            "s3_bucket": "bucket",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    result = nebius.bootstrap_agent_environment("project", "tenant", "eu-north1")

    assert result["nebius_api_key"] == "key"
    assert result["service_account_id"] == "sa-existing"


def test_nebius_bucket_exists(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        side_effect=[
            {"metadata": {"name": "npa-bucket-abc", "parent_id": "project"}},
            nebius.NebiusError("NotFound: bucket does not exist"),
        ],
    )

    assert nebius.bucket_exists("project", "npa-bucket-abc") is True
    assert nebius.bucket_exists("project", "other") is False
    assert all(
        call.args[0][:3] == ["storage", "bucket", "get-by-name"]
        for call in run_json.call_args_list
    )


def test_cli_env_strips_stale_iam_token(monkeypatch) -> None:
    # A stale ambient NEBIUS_IAM_TOKEN / NEBIUS_IAM_TOKEN_FILE shadows the active
    # profile and causes AccessDenied on storage calls (or a false profile
    # readiness); the CLI env must drop both by default.
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/stale-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)
    env = nebius.nebius_cli_env()
    assert "NEBIUS_IAM_TOKEN" not in env
    assert "NEBIUS_IAM_TOKEN_FILE" not in env


def test_cli_env_keeps_token_when_reuse_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "injected-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/injected-token")
    monkeypatch.setenv("NPA_REUSE_IAM_TOKEN", "1")
    env = nebius.nebius_cli_env()
    assert env["NEBIUS_IAM_TOKEN"] == "injected-token"
    assert env["NEBIUS_IAM_TOKEN_FILE"] == "/tmp/injected-token"


def test_cli_env_sanitizes_provided_base(monkeypatch) -> None:
    # Callers can sanitize an already-customized environment (e.g. one that adds
    # KUBECONFIG); the stale token is still dropped from the provided base.
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)
    base = {"NEBIUS_IAM_TOKEN": "stale", "KEEP": "yes"}
    out = nebius.nebius_cli_env(base)
    assert "NEBIUS_IAM_TOKEN" not in out
    assert out["KEEP"] == "yes"


def test_run_invokes_cli_without_stale_token(monkeypatch, mocker) -> None:
    # End-to-end: `_run` must pass a sanitized env to the nebius subprocess.
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)
    mocker.patch("npa.clients.nebius._require_nebius", return_value="/usr/bin/nebius")
    run = mocker.patch(
        "npa.clients.nebius.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    assert nebius._run(["storage", "bucket", "list"]) == "ok"
    passed_env = run.call_args.kwargs["env"]
    assert "NEBIUS_IAM_TOKEN" not in passed_env


def test_is_permission_denied_matches_access_denied() -> None:
    # Object storage reports authorization failures as AccessDenied; the
    # predicate must catch it (and the gRPC PermissionDenied code) so configure
    # can show IAM guidance instead of a raw rpc dump.
    assert nebius.is_permission_denied(
        "nebius storage bucket list failed (exit 15): code = PermissionDenied "
        "desc = AccessDenied: Access denied"
    )
    assert nebius.is_permission_denied("AccessDenied: Access denied")
    assert nebius.is_permission_denied("Permission denied")
    assert nebius.is_permission_denied(
        "S3 write probe was forbidden; the configured access key lacks permission."
    )
    assert not nebius.is_permission_denied("NotFound: bucket missing")


def test_nebius_bucket_exact_lookup_does_not_enumerate_project(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "metadata": {"name": "npa-bucket-abc", "parent_id": "project"}
        },
    )

    assert nebius.bucket_exists("project", "npa-bucket-abc") is True
    args = run_json.call_args.args[0]
    assert args[:3] == ["storage", "bucket", "get-by-name"]
    assert args[args.index("--name") + 1] == "npa-bucket-abc"
    assert "--all" not in args


def test_nebius_ensure_bucket_reuses_existing_without_create(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=True)
    run = mocker.patch("npa.clients.nebius._run")

    assert (
        nebius.ensure_bucket("project", "npa-bucket-abc", max_size_bytes=123)
        == "npa-bucket-abc"
    )
    run.assert_not_called()


def test_nebius_ensure_bucket_applies_max_size_on_create(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    run = mocker.patch("npa.clients.nebius._run")

    nebius.ensure_bucket("project", "npa-bucket-abc", max_size_bytes=50 * 1024**3)

    args = run.call_args.args[0]
    assert "--max-size-bytes" in args
    assert args[args.index("--max-size-bytes") + 1] == str(50 * 1024**3)
    assert "--default-storage-class" in args
    assert args[args.index("--default-storage-class") + 1] == "standard"


def test_nebius_ensure_bucket_reuses_on_already_exists_conflict(mocker) -> None:
    # Existence check missed the bucket (race / stale page), so create races an
    # existing bucket. When it turns out to be in this project, reuse it instead
    # of failing with a name conflict.
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=NebiusError("nebius storage bucket create failed: AlreadyExists"),
    )
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={"metadata": {"name": "npa-bucket-abc"}},
    )

    assert nebius.ensure_bucket("project", "npa-bucket-abc") == "npa-bucket-abc"


def test_nebius_ensure_bucket_refuses_generated_name_create_race(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=nebius.NebiusError("AlreadyExists: bucket exists"),
    )
    mocker.patch(
        "npa.clients.nebius.get_bucket_by_name",
        return_value={
            "metadata": {"name": "npa-bucket-abc", "parent_id": "project"}
        },
    )

    with pytest.raises(nebius.NebiusError, match="refusing to adopt"):
        nebius.ensure_bucket(
            "project",
            "npa-bucket-abc",
            allow_existing=False,
        )


def test_nebius_ensure_bucket_reports_clear_conflict_when_name_taken_elsewhere(
    mocker,
) -> None:
    # Name is globally taken but not in this project -> a real, actionable
    # conflict rather than a raw create error.
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    mocker.patch(
        "npa.clients.nebius._run",
        side_effect=NebiusError("AlreadyExists: bucket name is taken"),
    )
    mocker.patch("npa.clients.nebius.get_bucket_by_name", return_value=None)

    with pytest.raises(NebiusError, match="already taken"):
        nebius.ensure_bucket("project", "npa-bucket-abc")


def test_nebius_ensure_bucket_applies_enhanced_storage_class(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    run = mocker.patch("npa.clients.nebius._run")

    nebius.ensure_bucket(
        "project",
        "npa-bucket-abc",
        default_storage_class="enhanced_throughput",
    )

    args = run.call_args.args[0]
    assert args[args.index("--default-storage-class") + 1] == "enhanced_throughput"


def test_nebius_normalize_bucket_storage_class() -> None:
    assert nebius.normalize_bucket_storage_class("") == "standard"
    assert nebius.normalize_bucket_storage_class("enhanced") == "enhanced_throughput"
    assert (
        nebius.normalize_bucket_storage_class("ENHANCED_THROUGHPUT")
        == "enhanced_throughput"
    )


def test_nebius_ensure_bucket_unlimited_omits_max_size(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    run = mocker.patch("npa.clients.nebius._run")

    nebius.ensure_bucket("project", "npa-bucket-abc")

    assert "--max-size-bytes" not in run.call_args.args[0]


def test_nebius_current_project_and_tenant_from_profile(mocker) -> None:
    run = mocker.patch(
        "npa.clients.nebius._run",
        side_effect=["project-xyz", "tenant-xyz"],
    )

    assert nebius.current_project_id() == "project-xyz"
    assert nebius.current_tenant_id() == "tenant-xyz"
    assert run.call_args_list[0].args[0] == ["config", "get", "parent-id"]
    assert run.call_args_list[1].args[0] == ["config", "get", "tenant-id"]


def test_nebius_current_project_id_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=NebiusError("no profile"))

    assert nebius.current_project_id() == ""


def test_nebius_list_tenants_parses_items(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "items": [
                {
                    "metadata": {"id": "tenant-a", "name": "acme"},
                    "status": {"region": "eu-north1"},
                }
            ]
        },
    )

    assert nebius.list_tenants() == [
        {"id": "tenant-a", "name": "acme", "region": "eu-north1"}
    ]


def test_nebius_list_tenants_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("no profile"))

    assert nebius.list_tenants() == []


def test_nebius_list_projects_in_tenant_skips_non_active(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "items": [
                {
                    "metadata": {"id": "project-1", "name": "prod"},
                    "status": {"region": "eu-north1", "container_state": "ACTIVE"},
                },
                {
                    "metadata": {"id": "project-2", "name": "gone"},
                    "status": {"region": "us-central1", "container_state": "DELETING"},
                },
            ]
        },
    )

    result = nebius.list_projects_in_tenant("tenant-a")

    assert result == [
        {
            "id": "project-1",
            "name": "prod",
            "tenant_id": "tenant-a",
            "region": "eu-north1",
        }
    ]
    assert run_json.call_args.args[0] == [
        "iam",
        "project",
        "list",
        "--parent-id",
        "tenant-a",
        "--all",
    ]


def test_nebius_list_projects_in_tenant_empty_without_tenant(mocker) -> None:
    run_json = mocker.patch("npa.clients.nebius._run_json")

    assert nebius.list_projects_in_tenant("") == []
    run_json.assert_not_called()


def test_nebius_list_accessible_projects_spans_tenants(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius.list_tenants",
        return_value=[
            {"id": "tenant-a", "name": "a", "region": "eu-north1"},
            {"id": "tenant-b", "name": "b", "region": "us-central1"},
        ],
    )
    mocker.patch(
        "npa.clients.nebius.list_projects_in_tenant",
        side_effect=[
            [
                {
                    "id": "project-1",
                    "name": "p1",
                    "tenant_id": "tenant-a",
                    "region": "eu-north1",
                }
            ],
            [
                {
                    "id": "project-2",
                    "name": "p2",
                    "tenant_id": "tenant-b",
                    "region": "us-central1",
                }
            ],
        ],
    )

    result = nebius.list_accessible_projects()

    assert [p["id"] for p in result] == ["project-1", "project-2"]


def test_nebius_get_project_region_reads_status(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "status": {"region": "uk-south1"},
            "spec": {"region": "eu-north1"},
        },
    )

    assert nebius.get_project_region("project-abc") == "uk-south1"


def test_nebius_get_project_region_falls_back_to_spec(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"status": {}, "spec": {"region": "eu-north1"}},
    )

    assert nebius.get_project_region("project-abc") == "eu-north1"


def test_nebius_get_project_region_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.get_project_region("project-abc") == ""


def test_nebius_get_project_region_empty_without_project() -> None:
    assert nebius.get_project_region("") == ""


def test_nebius_get_project_tenant_id_reads_parent(mocker) -> None:
    """Recovers the tenant when the CLI profile only knows the project."""
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"metadata": {"id": "project-abc", "parent_id": "tenant-xyz"}},
    )

    assert nebius.get_project_tenant_id("project-abc") == "tenant-xyz"


def test_nebius_get_project_tenant_id_accepts_camel_case(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"metadata": {"parentId": "tenant-xyz"}},
    )

    assert nebius.get_project_tenant_id("project-abc") == "tenant-xyz"


def test_nebius_get_project_tenant_id_best_effort(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.get_project_tenant_id("project-abc") == ""
    assert nebius.get_project_tenant_id("") == ""


def test_nebius_get_project_name(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"metadata": {"id": "project-abc", "name": "tle-workbench"}},
    )

    assert nebius.get_project_name("project-abc") == "tle-workbench"


def test_nebius_get_project_name_best_effort(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.get_project_name("project-abc") == ""


def test_nebius_set_profile_project_writes_both_ids(mocker) -> None:
    run = mocker.patch("npa.clients.nebius._run", return_value="")

    assert nebius.set_profile_project("project-abc", "tenant-xyz") is True
    assert [call.args[0] for call in run.call_args_list] == [
        ["config", "set", "parent-id", "project-abc"],
        ["config", "set", "tenant-id", "tenant-xyz"],
    ]


def test_nebius_set_profile_project_skips_empty_tenant(mocker) -> None:
    run = mocker.patch("npa.clients.nebius._run", return_value="")

    assert nebius.set_profile_project("project-abc") is True
    assert [call.args[0] for call in run.call_args_list] == [
        ["config", "set", "parent-id", "project-abc"]
    ]


def test_nebius_set_profile_project_is_best_effort(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=NebiusError("no cli"))

    assert nebius.set_profile_project("project-abc", "tenant-xyz") is False
    assert nebius.set_profile_project("") is False


def _public_ip_quota_items() -> dict:
    return {
        "items": [
            {
                "metadata": {"name": "vpc.ipv4-address.public.count"},
                "spec": {"limit": "10", "region": "us-central1"},
                "status": {"usage": "10"},
            },
            {
                "metadata": {"name": "vpc.ipv4-address.public.count"},
                "spec": {"limit": "3", "region": "uk-south1"},
                "status": {"usage": "0"},
            },
            {
                "metadata": {"name": "compute.vcpu.count"},
                "spec": {"limit": "100", "region": "us-central1"},
                "status": {"usage": "4"},
            },
        ]
    }


def test_nebius_public_ipv4_quota_matches_region(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", return_value=_public_ip_quota_items())

    assert nebius.get_public_ipv4_quota("tenant-x", "us-central1") == (10, 10)
    assert nebius.get_public_ipv4_quota("tenant-x", "uk-south1") == (0, 3)


def test_nebius_public_ipv4_quota_unknown_region_is_none(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", return_value=_public_ip_quota_items())

    assert nebius.get_public_ipv4_quota("tenant-x", "eu-west1") == (None, None)


def test_nebius_public_ipv4_quota_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.get_public_ipv4_quota("tenant-x", "us-central1") == (None, None)


def test_nebius_public_ipv4_quota_requires_tenant_and_region() -> None:
    assert nebius.get_public_ipv4_quota("", "us-central1") == (None, None)
    assert nebius.get_public_ipv4_quota("tenant-x", "") == (None, None)


def _compute_instance_quota_items() -> dict:
    return {
        "items": [
            {
                "metadata": {"name": "compute.instance.count"},
                "spec": {"limit": "5", "region": "us-central1"},
                "status": {"usage": "2"},
            },
            # The reported failure: limit 0 and Nebius omits `status.usage`.
            {
                "metadata": {"name": "compute.instance.count"},
                "spec": {"limit": "0", "region": "eu-north1"},
                "status": {},
            },
            # A tenant-wide (region-less) allowance used as a fallback.
            {
                "metadata": {"name": "compute.instance.count"},
                "spec": {"limit": "7"},
                "status": {"usage": "1"},
            },
        ]
    }


def test_nebius_compute_instance_quota_matches_region(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json", return_value=_compute_instance_quota_items()
    )

    assert nebius.get_compute_instance_quota("tenant-x", "us-central1") == (2, 5)


def test_nebius_compute_instance_quota_limit_zero_reads_missing_usage_as_zero(
    mocker,
) -> None:
    """A real `limit 0` with no `status.usage` must gate (usage 0 >= limit 0)."""
    mocker.patch(
        "npa.clients.nebius._run_json", return_value=_compute_instance_quota_items()
    )

    usage, limit = nebius.get_compute_instance_quota("tenant-x", "eu-north1")
    assert (usage, limit) == (0, 0)


def test_nebius_compute_instance_quota_falls_back_to_region_less(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json", return_value=_compute_instance_quota_items()
    )

    # No per-region match for uk-south1 -> the tenant-wide allowance is used.
    assert nebius.get_compute_instance_quota("tenant-x", "uk-south1") == (1, 7)


def test_nebius_compute_instance_quota_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.get_compute_instance_quota("tenant-x", "us-central1") == (None, None)


def test_lazy_storage_client_does_not_connect_until_it_is_used(monkeypatch) -> None:
    """Holding one must be free; the whole point is local runs need no credentials."""

    import copy

    from npa.clients import storage

    def explode(**kwargs: object) -> object:
        raise AssertionError("built a StorageClient without a remote URI being touched")

    monkeypatch.setattr(
        storage.StorageClient, "from_environment", staticmethod(explode)
    )
    client = storage.LazyStorageClient()

    # Several stdlib paths probe for dunders; forwarding those would connect.
    copy.deepcopy(client)
    assert not hasattr(client, "__deepcopy__")
    repr(client)


def test_lazy_storage_client_builds_once_on_first_real_call(monkeypatch) -> None:
    from npa.clients import storage

    built: list[int] = []

    class FakeClient:
        def download_path(self, uri: str, dest: str) -> str:
            return dest

    def build(**kwargs: object) -> FakeClient:
        built.append(1)
        return FakeClient()

    monkeypatch.setattr(storage.StorageClient, "from_environment", staticmethod(build))
    client = storage.LazyStorageClient()
    assert client.download_path("s3://b/k", "/tmp/x") == "/tmp/x"
    assert client.download_path("s3://b/k", "/tmp/y") == "/tmp/y"
    assert len(built) == 1

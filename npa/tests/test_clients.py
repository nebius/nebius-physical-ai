from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from npa.clients.config import SSHConfig
from npa.clients.http import HTTPClient, ServerError
from npa.clients import nebius
from npa.clients.nebius import NebiusError
from npa.clients.ssh import SSHClient, SSHError
from npa.clients.storage import StorageClient, StorageError, _parse_bucket_uri


def test_http_client_builds_request_and_returns_json(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    response.json.return_value = {"status": "ok"}
    request = mocker.patch("httpx.request", return_value=response)

    client = HTTPClient("http://server/", timeout=10, retries=1)

    assert client.health() == {"status": "ok"}
    request.assert_called_once_with("GET", "http://server/health", json=None, timeout=10)


def test_http_client_fetches_job_status(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    response.json.return_value = {"job_id": "job/1", "status": "completed"}
    request = mocker.patch("httpx.request", return_value=response)

    assert HTTPClient("http://server").job_status("job/1", timeout=3.0) == {
        "job_id": "job/1",
        "status": "completed",
    }
    request.assert_called_once_with("GET", "http://server/jobs/job%2F1", json=None, timeout=3.0)


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
        {"CommonPrefixes": [{"Prefix": "checkpoints/job-a/"}, {"Prefix": "checkpoints/job-b/"}]}
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

    uploaded = client.upload_directory(str(local), "s3://bucket/base", remote_prefix="run")
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

    downloaded = client.download_path(
        "s3://bucket/prefix/result.json", str(local)
    )

    assert downloaded == str(local)
    mock_s3.head_object.assert_called_once_with(
        Bucket="bucket", Key="prefix/result.json"
    )
    mock_s3.download_file.assert_called_once_with(
        "bucket", "prefix/result.json", str(local)
    )


def test_storage_client_uploads_and_downloads_files(
    tmp_path: Path, mock_s3
) -> None:
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


def test_ssh_run_or_raise_maps_nonzero(mocker) -> None:
    client = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key"))
    mocker.patch.object(client, "run", return_value=(7, "", "bad"))

    with pytest.raises(SSHError, match="Command failed"):
        client.run_or_raise("false")


def test_ssh_download_file_uses_sftp(tmp_path: Path, mocker) -> None:
    sftp = mocker.MagicMock()
    paramiko_client = mocker.MagicMock()
    paramiko_client.open_sftp.return_value = sftp
    mocker.patch("paramiko.SSHClient", return_value=paramiko_client)

    local = tmp_path / "nested" / "out.mp4"
    result = SSHClient(SSHConfig(host="host", user="ubuntu", key_path="key")).download_file(
        "/remote/out.mp4", str(local)
    )

    assert result == str(local)
    sftp.get.assert_called_once_with("/remote/out.mp4", str(local))
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
    run.assert_called_once_with(
        ["/usr/bin/nebius", "iam", "get-access-token"],
        capture_output=True,
        text=True,
    )

    run.return_value = subprocess.CompletedProcess(
        args=["nebius"], returncode=1, stdout="", stderr="nope\n"
    )
    with pytest.raises(NebiusError, match="failed"):
        nebius._run(["bad"])


def test_nebius_requires_binary(mocker) -> None:
    mocker.patch("shutil.which", return_value=None)

    with pytest.raises(NebiusError, match="not found"):
        nebius._require_nebius()


def test_nebius_warns_once_on_cli_version_mismatch(mocker) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/nebius")
    mocker.patch("npa.clients.nebius.supported_tool_version", return_value="0.12.192")
    mocker.patch("npa.clients.nebius._NEBIUS_VERSION_CHECKED", False)
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["nebius", "version"], returncode=0, stdout="0.13.0\n", stderr=""
        ),
    )

    with pytest.warns(RuntimeWarning, match="expected 0.12.192; found 0.13.0"):
        assert nebius._require_nebius() == "/usr/bin/nebius"

    assert nebius._require_nebius() == "/usr/bin/nebius"
    run.assert_called_once_with(
        ["/usr/bin/nebius", "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_nebius_run_json_and_token(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", side_effect=['{"ok": true}', "", "token"])

    assert nebius._run_json(["cmd"]) == {"ok": True}
    assert nebius._run_json(["empty"]) == {}
    assert nebius.get_iam_token() == "token"


def test_nebius_empty_token_errors(mocker) -> None:
    mocker.patch("npa.clients.nebius._run", return_value="")

    with pytest.raises(NebiusError, match="empty token"):
        nebius.get_iam_token()


def test_nebius_service_account_reuses_existing(mocker) -> None:
    run_json = mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"metadata": {"id": "sa-id"}},
    )

    assert nebius.ensure_service_account("project", name="svc") == "sa-id"
    run_json.assert_called_once()


def test_nebius_find_active_access_key_prefers_requested_name(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "items": [
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
        },
    )

    result = nebius._find_active_access_key(
        "project",
        "sa",
        key_name="lerobot-access-key",
    )

    assert result is not None
    assert result["metadata"]["id"] == "target-id"


def test_nebius_ensure_access_key_does_not_delete_existing_key_without_secret(mocker) -> None:
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
    assert create_args[create_args.index("--name") + 1].startswith("lerobot-access-key-")


def test_nebius_bucket_name_and_bootstrap_order(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa")
    editors = mocker.patch("npa.clients.nebius.ensure_editors_membership")
    bucket = mocker.patch("npa.clients.nebius.ensure_bucket", return_value="bucket")
    mocker.patch("npa.clients.nebius.ensure_access_key", return_value=("key", "secret"))
    statuses: list[str] = []

    result = nebius.bootstrap_environment(
        "project",
        "tenant",
        "eu-north1",
        on_status=statuses.append,
    )

    assert nebius.bucket_name_for("tenant", "project").startswith("lerobot-")
    editors.assert_called_once_with("tenant", "sa")
    bucket.assert_called_once()
    assert result["iam_token"] == "iam"
    assert result["s3_endpoint"] == "https://storage.eu-north1.nebius.cloud"
    assert statuses[0] == "Getting IAM access token..."


def test_nebius_bootstrap_uses_explicit_bucket_name(mocker) -> None:
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa")
    mocker.patch("npa.clients.nebius.ensure_editors_membership")
    bucket = mocker.patch("npa.clients.nebius.ensure_bucket", return_value="chosen")
    mocker.patch("npa.clients.nebius.ensure_access_key", return_value=("key", "secret"))

    result = nebius.bootstrap_environment(
        "project",
        "tenant",
        "eu-north1",
        bucket_name="chosen",
        bucket_max_size_bytes=123,
    )

    assert result["s3_bucket"] == "chosen"
    bucket.assert_called_once_with("project", "chosen", max_size_bytes=123)


def test_nebius_bucket_exists(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={"items": [{"metadata": {"name": "lerobot-abc"}}]},
    )

    assert nebius.bucket_exists("project", "lerobot-abc") is True
    assert nebius.bucket_exists("project", "other") is False


def test_nebius_ensure_bucket_reuses_existing_without_create(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=True)
    run = mocker.patch("npa.clients.nebius._run")

    assert nebius.ensure_bucket("project", "lerobot-abc", max_size_bytes=123) == "lerobot-abc"
    run.assert_not_called()


def test_nebius_ensure_bucket_applies_max_size_on_create(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    run = mocker.patch("npa.clients.nebius._run")

    nebius.ensure_bucket("project", "lerobot-abc", max_size_bytes=50 * 1024**3)

    args = run.call_args.args[0]
    assert "--max-size-bytes" in args
    assert args[args.index("--max-size-bytes") + 1] == str(50 * 1024**3)


def test_nebius_ensure_bucket_unlimited_omits_max_size(mocker) -> None:
    mocker.patch("npa.clients.nebius.bucket_exists", return_value=False)
    run = mocker.patch("npa.clients.nebius._run")

    nebius.ensure_bucket("project", "lerobot-abc")

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


def test_nebius_discover_container_registry_builds_url(mocker) -> None:
    mocker.patch(
        "npa.clients.nebius._run_json",
        return_value={
            "items": [
                {
                    "metadata": {"id": "registry-e00abc"},
                    "status": {"registry_fqdn": "cr.eu-north1.nebius.cloud"},
                }
            ]
        },
    )

    assert (
        nebius.discover_container_registry("project")
        == "cr.eu-north1.nebius.cloud/e00abc"
    )


def test_nebius_discover_container_registry_empty_without_project() -> None:
    assert nebius.discover_container_registry("") == ""


def test_nebius_discover_container_registry_best_effort_on_error(mocker) -> None:
    mocker.patch("npa.clients.nebius._run_json", side_effect=NebiusError("denied"))

    assert nebius.discover_container_registry("project") == ""

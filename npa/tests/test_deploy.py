from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from npa.clients.config import SSHConfig
from npa.clients.ssh import SSHClient
from npa.deploy import configurator, provisioner
from npa.deploy.provisioner import ProvisionerError


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_build_var_args_preserves_key_values() -> None:
    assert provisioner._build_var_args({"a": "1", "b": "two"}) == [
        "-var",
        "a=1",
        "-var",
        "b=two",
    ]


def test_prepare_working_dir_copies_tf_files_and_writes_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "main.tf").write_text("resource {}\n")
    base = tmp_path / "workbenches"
    monkeypatch.setattr(provisioner, "_BUNDLED_TF_DIR", bundled)
    monkeypatch.setattr(provisioner, "_WORKBENCH_BASE", base)

    work_dir = provisioner.prepare_working_dir(
        "project",
        "name",
        bucket="state-bucket",
        region="eu-north1",
        endpoint="https://storage",
    )

    assert work_dir == base / "project" / "name"
    assert (work_dir / "main.tf").read_text() == "resource {}\n"
    backend = (work_dir / "backend.tf").read_text()
    assert 'bucket = "state-bucket"' in backend
    assert 'key    = "npa/terraform-state/project/name/terraform.tfstate"' in backend
    assert 's3 = "https://storage"' in backend


def test_cloud_init_branches_bootstrap_by_workbench_type() -> None:
    template = (PACKAGE_ROOT / "src/npa/deploy/terraform/cloud_init.yaml.tpl").read_text()
    assert "\t" not in template
    branches = template.split('%{ if workbench_type == "fiftyone" ~}')
    container_marker = '%{ if workbench_type == "lerobot-container" ~}'
    fiftyone_write_files = branches[1].split("%{ else ~}", 1)[0]
    fiftyone_runcmd = branches[2].split("%{ else ~}", 1)[0]
    fiftyone_branches = fiftyone_write_files + fiftyone_runcmd
    groot_base_marker = 'echo "=== GR00T base VM setup started - $(date) ==="'
    groot_branch = template.split(groot_base_marker, 1)[1].split("%{ else ~}", 1)[0]
    container_and_lerobot = template.split(container_marker, 1)[1]
    container_branch, lerobot_branch = container_and_lerobot.split("%{ else ~}", 1)

    assert "/etc/npa-fiftyone/env" in fiftyone_branches
    assert "/opt/fiftyone/venv" in fiftyone_branches
    assert "npa-fiftyone-app.service" in fiftyone_branches
    assert "fiftyone==${fiftyone_version}" in fiftyone_branches
    assert "/opt/lerobot" not in fiftyone_branches
    assert "lerobot[pusht" not in fiftyone_branches
    assert "setup_22.x" not in fiftyone_branches

    assert "GR00T base VM setup" in groot_branch
    assert "/opt/groot /opt/isaac-lab" in groot_branch
    assert "Installing LeRobot ${lerobot_version}" not in groot_branch
    assert "lerobot[pusht" not in groot_branch
    assert "GR00T container VM setup" in template

    assert "/opt/lerobot/.env" in lerobot_branch
    assert "Installing LeRobot ${lerobot_version}" in lerobot_branch
    assert 'LEROBOT_PIP_SPEC="lerobot[pusht,libero]==${lerobot_version}"' in lerobot_branch
    assert 'LEROBOT_PIP_SPEC="lerobot[training,evaluation,pusht,libero]==${lerobot_version}"' in lerobot_branch
    assert 'if [ "${lerobot_version}" = "0.6.0" ]' in lerobot_branch
    assert '"$LEROBOT_VENV/bin/pip" install "$LEROBOT_PIP_SPEC"' in lerobot_branch

    assert "LeRobot container VM setup" in container_branch
    assert "$DEPLOY_ROOT/checkpoints" in container_branch
    assert "Installing LeRobot ${lerobot_version}" not in container_branch
    assert "lerobot[pusht,libero]" not in container_branch


def test_cloud_init_mounts_cosmos_data_disk() -> None:
    template = (PACKAGE_ROOT / "src/npa/deploy/terraform/cloud_init.yaml.tpl").read_text()

    assert '%{ if workbench_type == "cosmos" ~}' in template
    assert "/dev/disk/by-id/virtio-npa-cosmos-data" in template
    assert "/opt/cosmos-data" in template
    assert "mkfs.ext4 -F" in template
    assert "$COSMOS_DATA_MOUNT/models" in template
    assert "$COSMOS_DATA_MOUNT/hf_cache" in template


def test_cloud_init_mounts_groot_data_disk() -> None:
    template = (PACKAGE_ROOT / "src/npa/deploy/terraform/cloud_init.yaml.tpl").read_text()

    assert "/dev/disk/by-id/virtio-npa-groot-data" in template
    assert "/opt/groot-data" in template
    assert "mkfs.ext4 -F \"$GROOT_DATA_DEVICE\"" in template
    assert "$GROOT_DATA_MOUNT/models" in template
    assert "$GROOT_DATA_MOUNT/hf_cache" in template
    assert "$GROOT_DATA_MOUNT/checkpoints" in template
    assert "$GROOT_DATA_MOUNT/eval_data_cache" in template
    assert 'workbench_type == "groot" || workbench_type == "groot-container"' in template
    assert '%{ if workbench_type == "groot" ~}' in template
    assert "ln -s \"$GROOT_DATA_MOUNT\" /opt/groot" in template


def test_terraform_template_receives_workbench_type_and_versions() -> None:
    main_tf = (PACKAGE_ROOT / "src/npa/deploy/terraform/main.tf").read_text()
    variables_tf = (PACKAGE_ROOT / "src/npa/deploy/terraform/variables.tf").read_text()

    assert "workbench_type   = var.workbench_type" in main_tf
    assert "lerobot_version  = var.lerobot_version" in main_tf
    assert "fiftyone_version = var.fiftyone_version" in main_tf
    assert "secondary_disks = concat(" in main_tf
    assert 'device_id   = "npa-cosmos-data"' in main_tf
    assert 'device_id   = "npa-groot-data"' in main_tf
    assert 'contains(["groot", "groot-container"], var.workbench_type)' in main_tf
    assert "var.cosmos_data_disk_size_gb" in main_tf
    assert "var.data_disk_size_gb" in main_tf
    assert 'variable "workbench_type"' in variables_tf
    assert 'default     = "lerobot"' in variables_tf
    assert 'variable "fiftyone_version"' in variables_tf
    assert 'variable "cosmos_data_disk_size_gb"' in variables_tf
    assert 'variable "data_disk_size_gb"' in variables_tf
    assert 'variable "boot_disk_size_gb"' in variables_tf
    assert "default     = 100" in variables_tf


def test_boot_disk_defaults_are_runtime_aware() -> None:
    assert provisioner.boot_disk_tf_vars("container") == {"boot_disk_size_gb": "250"}
    assert provisioner.boot_disk_tf_vars("vm") == {}


def test_boot_disk_disk_size_override_applies_to_any_runtime() -> None:
    assert provisioner.boot_disk_tf_vars("container", 384) == {"boot_disk_size_gb": "384"}
    assert provisioner.boot_disk_tf_vars("vm", 384) == {"boot_disk_size_gb": "384"}


def test_boot_disk_disk_size_override_must_be_positive() -> None:
    with pytest.raises(ValueError, match="--disk-size must be positive"):
        provisioner.boot_disk_tf_vars("container", 0)


def test_default_image_family_for_rtx6000_and_b300() -> None:
    assert (
        provisioner.default_image_family_for_platform("gpu-rtx6000")
        == "ubuntu24.04-cuda13.0"
    )
    assert (
        provisioner.default_image_family_for_platform("gpu-b300-sxm")
        == "ubuntu24.04-cuda13.0"
    )
    assert provisioner.default_image_family_for_platform("gpu-h200-sxm") is None


def test_apply_default_image_family_respects_explicit_override() -> None:
    vars_auto: dict[str, str] = {}
    provisioner.apply_default_image_family(vars_auto, "gpu-rtx6000")
    assert vars_auto["image_family"] == "ubuntu24.04-cuda13.0"

    vars_override = {"image_family": "ubuntu24.04-cuda12"}
    provisioner.apply_default_image_family(vars_override, "gpu-rtx6000")
    assert vars_override["image_family"] == "ubuntu24.04-cuda12"


def test_working_dir_path_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provisioner, "_WORKBENCH_BASE", tmp_path)
    work_dir = provisioner.working_dir_path("project", "name")
    work_dir.mkdir(parents=True)
    (work_dir / "file").write_text("x")

    provisioner.cleanup_working_dir("project", "name")

    assert not work_dir.exists()


def test_require_terraform_errors_when_missing(mocker) -> None:
    mocker.patch("shutil.which", return_value=None)

    with pytest.raises(ProvisionerError, match="terraform binary not found"):
        provisioner._require_terraform()


def test_run_invokes_terraform_and_maps_capture_errors(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("shutil.which", return_value="/usr/bin/terraform")
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"], returncode=0, stdout="ok", stderr=""
        ),
    )

    result = provisioner._run(["plan"], cwd=tmp_path, capture=True)

    assert result.stdout == "ok"
    run.assert_called_once()

    run.return_value = subprocess.CompletedProcess(
        args=["terraform"], returncode=1, stdout="", stderr="bad"
    )
    with pytest.raises(ProvisionerError, match="terraform plan failed"):
        provisioner._run(["plan"], cwd=tmp_path, capture=True)


def test_run_strips_stale_iam_token_from_terraform_env(
    tmp_path: Path, monkeypatch, mocker
) -> None:
    # A stale ambient NEBIUS_IAM_TOKEN must not reach the Terraform subprocess,
    # otherwise the Nebius provider prefers it over the fresh -var iam_token.
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    monkeypatch.setenv("NPA_NEBIUS_IAM_TOKEN", "stale-token-2")
    monkeypatch.setenv("PATH_MARKER", "keep-me")
    mocker.patch("shutil.which", return_value="/usr/bin/terraform")
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"], returncode=0, stdout="ok", stderr=""
        ),
    )

    provisioner._run(["plan"], cwd=tmp_path, capture=True)

    passed_env = run.call_args.kwargs["env"]
    assert "NEBIUS_IAM_TOKEN" not in passed_env
    assert "NPA_NEBIUS_IAM_TOKEN" not in passed_env
    assert passed_env.get("PATH_MARKER") == "keep-me"


def test_terraform_command_wrappers_delegate_to_run(tmp_path: Path, mocker) -> None:
    run = mocker.patch(
        "npa.deploy.provisioner._run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"], returncode=0, stdout="planned", stderr=""
        ),
    )

    provisioner.init(tmp_path, backend_config={"access_key": "key"})
    assert run.call_args.args[0] == [
        "init",
        "-input=false",
        "-reconfigure",
        "-backend-config",
        "access_key=key",
    ]

    assert provisioner.plan(tmp_path, {"gpu": "h100"}) == "planned"
    assert "-var" in run.call_args.args[0]

    mocker.patch("npa.deploy.provisioner.outputs", return_value={"vm_ip": "1.2.3.4"})
    assert provisioner.apply(tmp_path, {"gpu": "h100"}) == {"vm_ip": "1.2.3.4"}

    provisioner.destroy(tmp_path, {"gpu": "h100"})
    assert run.call_args.args[0][0] == "destroy"


def test_run_sets_terraform_plugin_cache_dir(tmp_path: Path, monkeypatch, mocker) -> None:
    monkeypatch.delenv("TF_PLUGIN_CACHE_DIR", raising=False)
    monkeypatch.setattr(provisioner, "_TF_PLUGIN_CACHE_DIR", tmp_path / "tf-cache")
    mocker.patch("shutil.which", return_value="/usr/bin/terraform")
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"], returncode=0, stdout="ok", stderr=""
        ),
    )

    provisioner._run(["init"], cwd=tmp_path, capture=True)

    env = run.call_args.kwargs["env"]
    assert env["TF_PLUGIN_CACHE_DIR"] == str(tmp_path / "tf-cache")
    assert (tmp_path / "tf-cache").is_dir()


def test_run_respects_preexisting_plugin_cache_env(tmp_path: Path, monkeypatch, mocker) -> None:
    monkeypatch.setenv("TF_PLUGIN_CACHE_DIR", "/custom/cache")
    mocker.patch("shutil.which", return_value="/usr/bin/terraform")
    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"], returncode=0, stdout="ok", stderr=""
        ),
    )

    provisioner._run(["init"], cwd=tmp_path, capture=True)

    assert run.call_args.kwargs["env"]["TF_PLUGIN_CACHE_DIR"] == "/custom/cache"


def test_init_retries_transient_registry_failure(tmp_path: Path, mocker) -> None:
    calls = {"n": 0}

    def fake_run(args, *, cwd, capture=False, stream=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProvisionerError(
                "terraform init failed (exit 1):\ncould not connect to registry.terraform.io: "
                "net/http: request canceled while waiting for connection (Client.Timeout exceeded "
                "while awaiting headers)"
            )
        return subprocess.CompletedProcess(args=["terraform"], returncode=0, stdout="", stderr="")

    mocker.patch("npa.deploy.provisioner._run", side_effect=fake_run)
    slept: list[float] = []

    provisioner.init(tmp_path, retries=3, backoff_seconds=0.01, sleep=slept.append)

    assert calls["n"] == 2  # failed once, succeeded on retry
    assert slept == [0.01]


def test_init_does_not_retry_config_error(tmp_path: Path, mocker) -> None:
    def fake_run(args, *, cwd, capture=False, stream=False):
        raise ProvisionerError("terraform init failed (exit 1):\nInvalid provider configuration")

    mocker.patch("npa.deploy.provisioner._run", side_effect=fake_run)
    slept: list[float] = []

    with pytest.raises(ProvisionerError, match="Invalid provider configuration"):
        provisioner.init(tmp_path, retries=3, backoff_seconds=0.01, sleep=slept.append)

    assert slept == []  # non-transient => no retry


def test_init_gives_up_after_retries(tmp_path: Path, mocker) -> None:
    def fake_run(args, *, cwd, capture=False, stream=False):
        raise ProvisionerError("could not connect to registry.terraform.io: i/o timeout")

    mocker.patch("npa.deploy.provisioner._run", side_effect=fake_run)
    slept: list[float] = []

    with pytest.raises(ProvisionerError, match="registry.terraform.io"):
        provisioner.init(tmp_path, retries=3, backoff_seconds=0.01, sleep=slept.append)

    assert len(slept) == 2  # retried after attempts 1 and 2, then raised on 3


def test_apply_and_destroy_raise_on_nonzero(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "npa.deploy.provisioner._run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"],
            returncode=1,
            stdout="",
            stderr="PermissionDenied: service compute",
        ),
    )

    with pytest.raises(ProvisionerError, match="PermissionDenied: service compute"):
        provisioner.apply(tmp_path)
    with pytest.raises(ProvisionerError, match="terraform destroy failed"):
        provisioner.destroy(tmp_path)


def test_outputs_parses_terraform_json(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "npa.deploy.provisioner._run",
        return_value=subprocess.CompletedProcess(
            args=["terraform"],
            returncode=0,
            stdout=json.dumps({"vm_ip": {"value": "1.2.3.4", "type": "string"}}),
            stderr="",
        ),
    )

    assert provisioner.outputs(tmp_path) == {"vm_ip": "1.2.3.4"}


def test_install_lerobot_checks_remote_import(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "0.5.0\n", "")

    assert configurator.install_lerobot(ssh) is True

    ssh.run.return_value = (1, "", "")
    assert configurator.install_lerobot(ssh) is False


def test_health_check_polls_until_success(mocker) -> None:
    response = mocker.MagicMock(status_code=500)
    ok = mocker.MagicMock(status_code=200)
    get = mocker.patch("httpx.get", side_effect=[response, ok])
    sleep = mocker.patch("time.sleep")

    assert configurator.health_check("http://vm:8080", retries=2, backoff=0.1) is True
    assert get.call_count == 2
    sleep.assert_called_once_with(0.1)


def test_health_check_returns_false_after_retries(mocker) -> None:
    mocker.patch("httpx.get", side_effect=httpx.ConnectError("down"))
    mocker.patch("time.sleep")

    assert configurator.health_check("http://vm:8080", retries=1, backoff=0) is False


def test_health_check_auto_falls_back_to_ssh(mocker) -> None:
    ssh = mocker.MagicMock()
    public = mocker.patch("npa.deploy.configurator.health_check", return_value=False)
    ssh_health = mocker.patch("npa.deploy.configurator.health_check_ssh", return_value=True)

    healthy, note = configurator.health_check_auto(
        "http://vm:8080",
        mode=configurator.HealthCheckMode.auto,
        ssh=ssh,
        port=8080,
        host="vm",
        retries=1,
        backoff=0,
    )

    assert healthy is True
    assert note == "Public port 8080 unreachable; service healthy via SSH on vm."
    public.assert_called_once_with("http://vm:8080", retries=1, backoff=0)
    ssh_health.assert_called_once_with(ssh, 8080, retries=1, backoff=0)


def test_health_check_auto_reports_failed_when_public_and_ssh_fail(mocker) -> None:
    ssh = mocker.MagicMock()
    mocker.patch("npa.deploy.configurator.health_check", return_value=False)
    mocker.patch("npa.deploy.configurator.health_check_ssh", return_value=False)

    healthy, note = configurator.health_check_auto(
        "http://vm:8080",
        mode="auto",
        ssh=ssh,
        port=8080,
        host="vm",
        retries=1,
        backoff=0,
    )

    assert healthy is False
    assert note == ""


def test_health_check_auto_limits_public_retry_budget(mocker) -> None:
    ssh = mocker.MagicMock()
    public = mocker.patch("npa.deploy.configurator.health_check", return_value=False)
    ssh_health = mocker.patch("npa.deploy.configurator.health_check_ssh", return_value=True)

    healthy, _ = configurator.health_check_auto(
        "http://vm:5151",
        mode="auto",
        ssh=ssh,
        port=5151,
        host="vm",
        retries=120,
        backoff=2,
    )

    assert healthy is True
    public.assert_called_once_with("http://vm:5151", retries=3, backoff=2)
    ssh_health.assert_called_once_with(ssh, 5151, retries=120, backoff=2)


def test_write_manifest_writes_json_command(mocker) -> None:
    ssh = mocker.MagicMock()

    configurator.write_manifest(ssh, tool="lerobot", version="1", deployed_by="test")

    command = ssh.run_or_raise.call_args.args[0]
    assert "/etc/npa/manifest.json" in command
    assert '"tool": "lerobot"' in command
    assert '"version": "1"' in command


def test_write_remote_env_file_renders_shell_safe_values(mocker) -> None:
    ssh = mocker.MagicMock()
    uploads: list[str] = []
    mocker.patch(
        "npa.deploy.configurator._sftp_upload",
        side_effect=lambda _ssh, local, _remote: uploads.append(Path(local).read_text()),
    )

    configurator.write_remote_env_file(
        ssh,
        "/etc/npa/env",
        {"S3_SECRET_KEY": "abc$def!`cmd`'\"\\tail"},
    )

    assert uploads == ["S3_SECRET_KEY='abc$def!`cmd`'\\''\"\\tail'\n"]


def test_audit_remote_env_catches_missing_credential(mocker) -> None:
    ssh = mocker.MagicMock()
    mocker.patch(
        "npa.deploy.configurator.read_remote_env_keys",
        return_value={"HF_TOKEN": "hf-token"},
    )

    missing = configurator.audit_remote_env(
        ssh,
        "/etc/npa/env",
        {"HF_TOKEN": "hf-token", "AWS_SECRET_ACCESS_KEY": "secret"},
    )

    assert missing == ["AWS_SECRET_ACCESS_KEY"]


def test_read_remote_env_keys_parses_without_shell_expansion(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "HF_TOKEN='hf_abc'\\''123'\nAWS_SECRET_ACCESS_KEY=abc$def!`cmd`'\"\\tail\n",
        "",
    )

    values = configurator.read_remote_env_keys(
        ssh,
        "/etc/npa/env",
        ["HF_TOKEN", "AWS_SECRET_ACCESS_KEY"],
    )

    assert values == {
        "HF_TOKEN": "hf_abc'123",
        "AWS_SECRET_ACCESS_KEY": "abc$def!`cmd`'\"\\tail",
    }
    assert ssh.run_or_raise.call_args.args[0] == "sudo cat /etc/npa/env"


def test_deploy_workbench_container_adds_groups_and_devices(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "44\n", ""),
        (0, "109\n", ""),
    ]
    mocker.patch("npa.deploy.configurator.install_container_runtime")

    configurator.deploy_workbench_container(
        ssh,
        image_ref="registry.example/npa-genesis:1",
        container_name="npa-genesis",
        group_add=["0", "video", "render"],
        devices=["/dev/dri"],
    )

    assert ssh.run.call_args_list[0].args[0] == "getent group video | cut -d: -f3"
    assert ssh.run.call_args_list[1].args[0] == "getent group render | cut -d: -f3"
    run_cmd = ssh.run_or_raise.call_args_list[-1].args[0]
    assert "--group-add 0" in run_cmd
    assert "--group-add 44" in run_cmd
    assert "--group-add 109" in run_cmd
    assert "--device /dev/dri" in run_cmd


def test_deploy_server_runs_expected_remote_steps(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh._config = SSHConfig(host="vm", user="ubuntu", key_path="key")
    run = mocker.patch("subprocess.run")
    uploads: list[tuple[str, str]] = []
    mocker.patch(
        "npa.deploy.configurator._sftp_upload",
        side_effect=lambda _ssh, local, remote: uploads.append((local, remote)),
    )

    configurator.deploy_server(
        ssh,
        {
            "server_port": 8080,
            "checkpoint_bucket": "s3://bucket/checkpoints/",
            "storage_endpoint": "https://storage",
        },
    )

    run.assert_called_once()
    remote_paths = [remote for _local, remote in uploads]
    assert "/tmp/npa-deploy/npa.tgz" in remote_paths
    assert "/tmp/npa-server.yaml" in remote_paths
    assert "/tmp/npa-server.env" in remote_paths
    assert any("systemctl restart npa-lerobot-server" in call.args[0] for call in ssh.run_or_raise.call_args_list)


def test_sftp_upload_uses_paramiko(mocker) -> None:
    ssh = SSHClient(SSHConfig(host="vm", user="ubuntu", key_path="~/key"))
    sftp = mocker.MagicMock()
    client = mocker.MagicMock()
    client.open_sftp.return_value = sftp
    mocker.patch("paramiko.SSHClient", return_value=client)

    configurator._sftp_upload(ssh, "/local/file", "/remote/file")

    client.connect.assert_called_once()
    sftp.put.assert_called_once_with("/local/file", "/remote/file")
    client.close.assert_called_once()

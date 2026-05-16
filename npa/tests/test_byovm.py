from __future__ import annotations

import pytest

from npa.clients.credentials import CredentialsConfig
from npa.clients.config import StorageConfig
from npa.deploy.byovm import (
    GPUInfo,
    apply_project_storage_vars,
    apply_storage_env_vars,
    detect_gpu_info,
    gpu_config_fields,
    gpu_env_fields,
    resolve_byovm_target,
    runtime_uses_container,
    select_visible_devices,
)
from npa.clients.ssh import SSHError


class FakeSSH:
    def __init__(self, output: str) -> None:
        self.output = output

    def run_or_raise(self, command: str):
        assert "nvidia-smi" in command
        return 0, self.output, ""


def test_resolve_byovm_target_prefers_cli_then_env_then_credentials() -> None:
    creds = CredentialsConfig(ssh_host="creds-host", ssh_user="creds-user", ssh_key_path="/creds/key")

    from_env = resolve_byovm_target(
        credentials=creds,
        environ={
            "NPA_BYOVM_HOST": "env-host",
            "NPA_BYOVM_SSH_USER": "env-user",
            "NPA_BYOVM_SSH_KEY": "/env/key",
        },
    )
    assert from_env.host == "env-host"
    assert from_env.user == "env-user"
    assert from_env.key_path == "/env/key"

    from_cli = resolve_byovm_target(
        host="cli-host",
        ssh_user="cli-user",
        ssh_key="/cli/key",
        credentials=creds,
        environ={
            "NPA_BYOVM_HOST": "env-host",
            "NPA_BYOVM_SSH_KEY": "/env/key",
        },
    )
    assert from_cli.host == "cli-host"
    assert from_cli.user == "cli-user"
    assert from_cli.key_path == "/cli/key"


def test_resolve_byovm_target_requires_host_and_key() -> None:
    with pytest.raises(ValueError, match="BYOVM target is incomplete"):
        resolve_byovm_target(credentials=CredentialsConfig(), environ={})


def test_detect_gpu_info_parses_nvidia_smi_names() -> None:
    info = detect_gpu_info(FakeSSH("NVIDIA H200\nNVIDIA H200\n"))  # type: ignore[arg-type]

    assert info.count == 2
    assert info.names == ["NVIDIA H200", "NVIDIA H200"]
    assert info.primary_name == "NVIDIA H200"


def test_detect_gpu_info_requires_at_least_one_gpu() -> None:
    with pytest.raises(SSHError, match="returned no GPUs"):
        detect_gpu_info(FakeSSH(""))  # type: ignore[arg-type]


def test_select_visible_devices_all_or_limited() -> None:
    assert select_visible_devices(4) == (4, "0,1,2,3")
    assert select_visible_devices(4, 2) == (2, "0,1")
    assert select_visible_devices(4, 0) == (4, "0,1,2,3")


def test_select_visible_devices_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="No GPUs"):
        select_visible_devices(0)
    with pytest.raises(ValueError, match="exceeds detected"):
        select_visible_devices(2, 4)
    with pytest.raises(ValueError, match="must be positive"):
        select_visible_devices(2, -1)


def test_gpu_config_and_env_fields() -> None:
    info = GPUInfo(count=4, names=["NVIDIA L40S"] * 4)

    assert gpu_config_fields(info, effective_count=2, visible_devices="0,1") == {
        "gpu_platform": "NVIDIA L40S",
        "gpu_preset": "2gpu-byovm",
        "gpu_count": 2,
        "detected_gpu_count": 4,
        "detected_gpu_names": ["NVIDIA L40S"] * 4,
        "cuda_visible_devices": "0,1",
        "managed_lifecycle": False,
    }
    assert gpu_env_fields(info, effective_count=2, visible_devices="0,1") == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NPA_GPU_COUNT": "2",
        "NPA_DETECTED_GPU_COUNT": "4",
        "NPA_GPU_TYPE": "NVIDIA_L40S",
    }


def test_byovm_uses_container_runtime_path() -> None:
    assert runtime_uses_container("byovm")


def test_apply_storage_env_vars_respects_explicit_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setenv("NPA_CHECKPOINT_BUCKET", "s3://bucket/checkpoints/")

    merged = {"nebius_api_key": "saved-access"}
    apply_storage_env_vars(merged, explicit_vars={"s3_endpoint": "explicit"})

    assert merged["nebius_api_key"] == "env-access"
    assert merged["nebius_secret_key"] == "env-secret"
    assert merged["s3_bucket"] == "s3://bucket/checkpoints/"
    assert "s3_endpoint" not in merged


def test_apply_project_storage_vars_warns_when_missing(mocker) -> None:
    mocker.patch(
        "npa.clients.config.resolve_project_storage",
        return_value=StorageConfig(
            checkpoint_bucket="",
            endpoint_url="",
        ),
    )
    warnings: list[str] = []

    found = apply_project_storage_vars(
        {},
        project="proj",
        explicit_vars={},
        warn=warnings.append,
    )

    assert found is False
    assert warnings == [
        "Warning: Project proj has no object-storage settings. "
        "S3 operations on this workbench will fail unless configured manually."
    ]

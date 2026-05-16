from __future__ import annotations

import pytest

from npa.serverless_common import (
    build_serverless_job_env,
    build_serverless_output_upload_cmd,
    resolve_gpu_platform,
    split_serverless_env,
    validate_output_path,
)


def test_build_serverless_job_env_basic() -> None:
    env = build_serverless_job_env(output_path="s3://bucket/prefix")

    assert env["NPA_OUTPUT_PATH"] == "s3://bucket/prefix"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["HF_HOME"] == "/tmp/hf_home"


def test_build_serverless_job_env_with_hf_token() -> None:
    env = build_serverless_job_env(output_path="s3://bucket/prefix", hf_token="hf_secret")

    assert env["HF_TOKEN"] == "hf_secret"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_secret"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf_secret"


def test_build_serverless_job_env_with_s3_creds() -> None:
    env = build_serverless_job_env(
        output_path="s3://bucket/prefix",
        s3_credentials={
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "endpoint_url": "https://storage.example",
        },
    )

    assert env["AWS_ACCESS_KEY_ID"] == "key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert env["NEBIUS_S3_ENDPOINT"] == "https://storage.example"


def test_split_serverless_env_separates_secrets() -> None:
    safe, secret = split_serverless_env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_TOKEN": "hf_secret",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "PASSWORD_FILE": "pw",
        }
    )

    assert safe == {"PYTHONUNBUFFERED": "1"}
    assert secret["HF_TOKEN"] == "hf_secret"
    assert secret["AWS_ACCESS_KEY_ID"] == "key"
    assert secret["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert secret["PASSWORD_FILE"] == "pw"


@pytest.mark.parametrize(
    ("gpu_type", "platform", "preset"),
    [
        ("h200", "gpu-h200-sxm", "1gpu-16vcpu-200gb"),
        ("h100", "gpu-h100-sxm", "1gpu-16vcpu-200gb"),
        ("b300", "gpu-b300-sxm", "1gpu-24vcpu-346gb"),
        ("l40s", "gpu-l40s-a", "1gpu-40vcpu-160gb"),
        ("gpu-rtx-pro-6000", "gpu-rtx6000", "1gpu-24vcpu-218gb"),
    ],
)
def test_resolve_gpu_platform_known_types(gpu_type: str, platform: str, preset: str) -> None:
    assert resolve_gpu_platform(gpu_type) == (platform, preset, 1)


def test_resolve_gpu_platform_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown GPU type"):
        resolve_gpu_platform("unknown-gpu")


def test_validate_output_path_accepts_s3() -> None:
    validate_output_path("s3://bucket/prefix")


@pytest.mark.parametrize("uri", ["", "file:///tmp/out", "s3://bucket"])
def test_validate_output_path_rejects_bad_scheme(uri: str) -> None:
    with pytest.raises(ValueError):
        validate_output_path(uri)


def test_build_output_upload_cmd_contains_boto3() -> None:
    cmd = build_serverless_output_upload_cmd("/tmp/out", "s3://bucket/prefix/")

    assert "boto3" in cmd
    assert "NPA_OUTPUT_PATH" in cmd
    assert "/tmp/out" in cmd

from __future__ import annotations

import pytest

from npa.cli.path_contract import (
    FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR,
    PathContractError,
    validate_read_path,
    validate_write_path,
)


def test_validator_accepts_s3_uri() -> None:
    assert validate_read_path("s3://bucket/key", tool="Tool") == "s3://bucket/key"
    assert validate_write_path("s3://bucket/key", tool="Tool") == "s3://bucket/key"


@pytest.mark.parametrize(
    "path",
    [
        "/abs/path",
        "rel/path",
        "file:///abs/path",
        "http://example.com/dataset",
    ],
)
def test_write_validator_rejects_local_and_plain_http_paths(path: str) -> None:
    with pytest.raises(PathContractError) as excinfo:
        validate_write_path(path, tool="Tool")

    message = str(excinfo.value)
    assert "Tool --output-path expects an S3 URI" in message
    assert "S3 handoff contract" in message


@pytest.mark.parametrize(
    "path",
    [
        "user/repo",
        "user/repo:revision",
        "https://huggingface.co/datasets/user/repo",
    ],
)
def test_read_validator_accepts_hugging_face_dataset_refs(path: str) -> None:
    assert validate_read_path(path, tool="Tool") == path


def test_read_validator_rejects_relative_path() -> None:
    with pytest.raises(PathContractError) as excinfo:
        validate_read_path("rel/path", tool="Tool")

    message = str(excinfo.value)
    assert "Tool --input-path expects an S3 URI or a Hugging Face Hub dataset" in message
    assert "S3 handoff contract" in message


def test_fiftyone_vm_local_error_text_is_exact() -> None:
    with pytest.raises(PathContractError) as excinfo:
        validate_read_path(
            "/opt/cosmos-data/outputs/cosmos.mp4",
            tool="FiftyOne load-dataset",
            vm_local_message=FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR,
        )

    assert str(excinfo.value) == FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR

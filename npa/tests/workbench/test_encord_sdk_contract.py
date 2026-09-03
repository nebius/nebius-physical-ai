from __future__ import annotations

import inspect

import pytest
from packaging.version import Version

encord = pytest.importorskip(
    "encord",
    reason="requires the 'encord' optional extra (pip install -e 'npa[encord]')",
)


def test_installed_encord_0_1_202_surface() -> None:
    from encord.dataset import Dataset
    from encord.orm.dataset import StorageLocation
    from encord.orm.storage import (
        DataUploadImage,
        DataUploadItems,
        DataUploadVideo,
        UploadLongPollingState,
    )
    from encord.project import Project
    from encord.storage import StorageFolder, StorageItem
    from encord.user_client import EncordUserClient

    version = Version(encord.__version__)
    assert Version("0.1.202") <= version < Version("0.2")
    _has_parameters(
        EncordUserClient.create_with_ssh_private_key,
        "ssh_private_key",
        "ssh_private_key_path",
        "domain",
    )
    _has_parameters(
        StorageFolder.add_private_data_to_folder_start,
        "integration_id",
        "private_files",
        "ignore_errors",
    )
    _has_parameters(
        StorageFolder.add_private_data_to_folder_get_result,
        "upload_job_id",
        "timeout_seconds",
    )
    _has_parameters(
        StorageFolder.list_items,
        "include_client_metadata",
        "page_size",
    )
    _has_parameters(StorageFolder.upload_image, "file_path", "client_metadata")
    _has_parameters(StorageFolder.upload_video, "file_path", "client_metadata")
    _has_parameters(Dataset.link_items, "item_uuids", "duplicates_behavior")
    _has_parameters(Project.list_label_rows_v2, "include_client_metadata")
    _has_parameters(Project.create_bundle, "bundle_size")

    required_poll_fields = {
        "status",
        "items_with_names",
        "errors",
        "units_pending_count",
        "units_done_count",
        "units_error_count",
        "units_cancelled_count",
        "unit_errors",
    }
    assert required_poll_fields <= set(UploadLongPollingState.model_fields)
    for attribute in (
        "uuid",
        "name",
        "client_metadata",
        "url",
        "file_size",
        "mime_type",
        "get_signed_url",
    ):
        assert hasattr(StorageItem, attribute), attribute
    assert hasattr(StorageLocation, "CORD_STORAGE")

    image = DataUploadImage(
        objectUrl="https://storage.test.example/source-bucket/incoming/clip.png",
        clientMetadata={
            "npa": {"source_uri": "s3://source-bucket/incoming/clip.png"}
        },
    )
    video = DataUploadVideo(
        objectUrl="https://storage.test.example/source-bucket/incoming/clip.mp4",
        clientMetadata={
            "npa": {"source_uri": "s3://source-bucket/incoming/clip.mp4"}
        },
    )
    payload = DataUploadItems(
        images=[image], videos=[video], skipDuplicateUrls=True
    ).to_dict()
    assert payload["skipDuplicateUrls"] is True
    assert payload["upsertMetadata"] is False
    assert payload["images"][0]["clientMetadata"]["npa"]["source_uri"].endswith(
        "clip.png"
    )
    assert payload["videos"][0]["objectUrl"].endswith("clip.mp4")


def _has_parameters(callable_, *names: str) -> None:
    parameters = inspect.signature(callable_).parameters
    assert set(names) <= set(parameters), f"{callable_.__qualname__}: {parameters}"

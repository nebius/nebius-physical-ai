"""Encord SaaS client seam: title-or-id resolution, auth transports, endpoint."""

from __future__ import annotations

import pytest

from encord_fakes import ENDPOINT, FakeDataset, FakeFolder, FakeUserClient, fake_uuid
from npa.workbench.encord.client import (
    CORD_STORAGE_LOCATION,
    default_user_client,
    resolve_dataset,
    resolve_folder,
    resolve_integration,
    resolve_public_endpoint,
)
from npa.workbench.encord.schemas import EncordAuthError, EncordToolError


def test_resolve_integration_by_title_and_id() -> None:
    client = FakeUserClient()
    ref = resolve_integration(client, "nebius-s3")
    assert (ref.id, ref.title, ref.created) == (fake_uuid(3), "nebius-s3", False)
    by_id = resolve_integration(client, fake_uuid(3))
    assert (by_id.id, by_id.title) == (fake_uuid(3), "nebius-s3")
    with pytest.raises(EncordToolError, match="No Encord cloud integration titled"):
        resolve_integration(client, "missing")
    with pytest.raises(EncordToolError, match="No Encord cloud integration with id"):
        resolve_integration(client, fake_uuid(4))
    with pytest.raises(EncordToolError, match="must not be empty"):
        resolve_integration(client, "  ")


def test_resolve_folder_creates_on_missing_title_only() -> None:
    client = FakeUserClient()
    ref = resolve_folder(client, "fresh")
    assert ref.created is True and client.created_folders == ["fresh"]
    assert ref.title == "fresh" and ref.id == str(ref.obj.uuid)
    again = resolve_folder(client, "fresh")
    assert again.created is False and again.obj is ref.obj
    with pytest.raises(KeyError):
        resolve_folder(client, fake_uuid(99))


def test_resolvers_never_guess_between_same_titled_objects() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="dup"), FakeFolder(name="dup")])
    with pytest.raises(EncordToolError, match="Multiple Encord storage folders"):
        resolve_folder(client, "dup")
    assert client.created_folders == []
    client.datasets[fake_uuid(1)] = FakeDataset(fake_uuid(1), "dup-ds")
    client.datasets[fake_uuid(2)] = FakeDataset(fake_uuid(2), "dup-ds")
    with pytest.raises(EncordToolError, match="pass the dataset hash"):
        resolve_dataset(client, "dup-ds")


def test_resolve_dataset_title_create_and_pull_no_create() -> None:
    client = FakeUserClient()
    ref = resolve_dataset(client, "new-ds")
    assert ref.created is True and ref.title == "new-ds"
    assert ref.id in client.datasets
    assert resolve_dataset(client, "new-ds").created is False
    with pytest.raises(EncordToolError, match="No Encord dataset titled"):
        resolve_dataset(client, "absent", create=False)


def test_cord_storage_location_matches_the_sdk_enum() -> None:
    """The injected-client seam never imports the SDK; pin the value it stands for."""

    dataset_orm = pytest.importorskip("encord.orm.dataset")
    assert int(dataset_orm.StorageLocation.CORD_STORAGE) == CORD_STORAGE_LOCATION


def test_default_user_client_requires_secret_and_decodes_b64() -> None:
    with pytest.raises(EncordAuthError, match="No Encord credential"):
        default_user_client({})
    with pytest.raises(EncordAuthError, match="not valid base64"):
        default_user_client({"ENCORD_SSH_KEY_B64": "!!!not-base64!!!"})


def test_default_user_client_ignores_a_raw_pem_transport() -> None:
    """Exactly two transports exist; the truncation-prone raw PEM is not one."""

    raw_pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----"
    with pytest.raises(EncordAuthError, match="No Encord credential"):
        default_user_client({"ENCORD_SSH_KEY": raw_pem})


def test_resolve_public_endpoint_prefers_env() -> None:
    assert resolve_public_endpoint({"AWS_ENDPOINT_URL": ENDPOINT + "/"}) == ENDPOINT
    with pytest.raises(EncordToolError, match="No S3 endpoint"):
        resolve_public_endpoint({})


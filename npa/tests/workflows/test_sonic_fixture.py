"""Unit coverage for the SONIC export fixture builder (no infrastructure)."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.sonic_fixture import (
    DEFAULT_ACT_DIM,
    DEFAULT_OBS_DIM,
    FIXTURE_SCHEMA,
    SonicFixtureError,
    build_and_publish,
    split_s3_uri,
    upload,
)


def test_split_s3_uri_round_trips() -> None:
    assert split_s3_uri("s3://bucket/a/b/checkpoint.pt") == ("bucket", "a/b/checkpoint.pt")


@pytest.mark.parametrize("uri", ["", "bucket/key", "https://example.invalid/x", "s3://bucket"])
def test_split_s3_uri_rejects_bad_input(uri: str) -> None:
    with pytest.raises(SonicFixtureError):
        split_s3_uri(uri)


class _FakeS3:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        self.uploads.append((local, bucket, key))


def test_upload_uses_the_injected_client(tmp_path: Path) -> None:
    local = tmp_path / "checkpoint.pt"
    local.write_bytes(b"not-a-real-checkpoint")
    client = _FakeS3()

    uri = upload(local, "s3://bucket/prefix/checkpoint.pt", client=client)

    assert uri == "s3://bucket/prefix/checkpoint.pt"
    assert client.uploads == [(str(local), "bucket", "prefix/checkpoint.pt")]


def test_build_and_publish_writes_a_loadable_policy_checkpoint(tmp_path: Path) -> None:
    """The fixture must be a checkpoint the shipped exporter can actually load."""

    torch = pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")
    client = _FakeS3()

    result = build_and_publish(
        checkpoint_uri="s3://bucket/prefix/checkpoint.pt",
        workdir=tmp_path,
        obs_dim=6,
        act_dim=3,
        hidden=8,
        client=client,
    )

    assert result["schema"] == FIXTURE_SCHEMA
    assert result["obs_dim"] == 6 and result["act_dim"] == 3
    assert result["bytes"] > 0
    assert result["checkpoint_uri"] == "s3://bucket/prefix/checkpoint.pt"
    assert client.uploads

    payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    policy = payload["policy"]
    assert isinstance(policy, torch.nn.Module)
    # The same shape contract `npa workbench sonic export` will trace.
    action = policy(torch.zeros(1, 6))
    assert tuple(action.shape) == (1, 3)
    # The exporter needs these to resolve dims without an --obs-spec, and they must
    # survive the save/load round trip (live regression: SkyPilot job 188 failed with
    # "observation dimension is required" because the first fixture had neither).
    assert policy.obs_dim == 6
    assert policy.action_dim == 3


def test_build_and_publish_is_deterministic(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")

    first = build_and_publish(workdir=tmp_path / "a", obs_dim=4, act_dim=2, hidden=4)
    second = build_and_publish(workdir=tmp_path / "b", obs_dim=4, act_dim=2, hidden=4)

    def _weights(path: str) -> list[float]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return [
            round(float(value), 6)
            for tensor in payload["policy"].state_dict().values()
            for value in tensor.flatten().tolist()
        ]

    assert _weights(first["checkpoint_path"]) == _weights(second["checkpoint_path"])


def test_fixture_policy_exposes_the_dims_the_exporter_looks_for() -> None:
    """Pin the attribute names `_resolve_dim` / `_resolve_action_dim` search for."""

    pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")
    from npa.workflows.sonic_fixture import build_policy_module

    policy = build_policy_module(obs_dim=7, act_dim=4, hidden=5)

    # `obs_dim` and `action_dim` are the names the exporter tries first in each list.
    assert getattr(policy, "obs_dim", None) == 7
    assert getattr(policy, "action_dim", None) == 4


def test_default_dims_match_the_locomotion_specs() -> None:
    """Pin the defaults so a staged fixture keeps matching the SONIC twins."""

    assert (DEFAULT_OBS_DIM, DEFAULT_ACT_DIM) == (48, 12)


def test_build_rejects_degenerate_shapes(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")

    with pytest.raises(SonicFixtureError):
        build_and_publish(workdir=tmp_path, obs_dim=0, act_dim=3, hidden=4)

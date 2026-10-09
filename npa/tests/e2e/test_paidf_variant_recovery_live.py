"""Read-only live verification of newly produced immutable PAIDF recovery receipts."""

from datetime import datetime
import json
import os
from pathlib import Path
import stat
import tempfile
from urllib.parse import urlsplit

from botocore.exceptions import ClientError
import pytest

from npa.workflows import paidf_variant_recovery as recovery
from npa.workflows.paidf_cosmos3 import validate_committed_augment_manifest
from npa.workflows.paidf_cosmos3_media import video_sha256

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


class _ReadOnlyRunStorage:
    def __init__(self, client, run_uri, fresh_after):
        self.client = client
        self.run_uri = run_uri
        self.fresh_after = fresh_after

    def read_bytes_with_etag(self, uri):
        assert uri.startswith(self.run_uri), "Read escaped the exact selected run"
        parsed = urlsplit(uri)
        try:
            response = self.client.get_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None
            raise
        assert response["LastModified"] >= self.fresh_after, (
            "Object predates the selected fresh run"
        )
        with response["Body"] as stream:
            payload = stream.read()
        return payload, response["ETag"]


def _config(path):
    source = Path(path)
    assert source.is_file() and not source.is_symlink()
    assert source.stat().st_uid == os.getuid()
    assert stat.S_IMODE(source.stat().st_mode) & 0o077 == 0, (
        "Live configuration must be owner-only"
    )
    config = json.loads(source.read_bytes())
    assert set(config) == {
        "project",
        "run_uri",
        "run_id",
        "fresh_after",
        "input_sha256",
        "runtime_image",
        "variant_count",
        "require_recovery",
    }
    assert type(config["require_recovery"]) is bool
    parsed = urlsplit(config["run_uri"])
    assert parsed.scheme == "s3" and parsed.netloc and parsed.path.endswith("/")
    assert not parsed.query and not parsed.fragment
    return config


def _read(storage, uri):
    current = storage.read_bytes_with_etag(uri)
    assert current is not None, "Required exact run artifact is absent"
    return json.loads(current[0])


def test_completed_native_variants_have_verified_immutable_recovery_receipts():
    path = os.environ.get("NPA_PAIDF_VARIANT_RECOVERY_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not os.environ.get(
        "NPA_PAIDF_VARIANT_RECOVERY_LIVE_CONFIG"
    ):
        pytest.skip(
            "Requires owner-private configuration for a completed candidate-source run"
        )
    from npa.clients.project_credentials import s3_client_for_project

    config = _config(path)
    fresh_after = datetime.fromisoformat(config["fresh_after"])
    assert fresh_after.tzinfo is not None
    client = s3_client_for_project(config["project"], allow_host_creds=True)
    storage = _ReadOnlyRunStorage(client, config["run_uri"], fresh_after)
    augment = config["run_uri"] + "cosmos_augmented/"
    manifest = _read(storage, augment + "manifest.json")
    variants = validate_committed_augment_manifest(manifest, augment)
    assert manifest["run_id"] == config["run_id"]
    assert manifest["verified_variant_recovery"] is True
    assert len(variants) == config["variant_count"]
    if config["require_recovery"]:
        assert manifest["recovered_variant_count"] > 0, (
            "No actual same-run reuse was recorded"
        )
    _audit_variants(storage, augment, manifest, config)


def _audit_variants(storage, augment, manifest, config):
    attempt = manifest["attempt"]
    batch_uri = augment + f"_generation-recovery/attempt-{attempt:02d}/batch.json"
    batch = _read(storage, batch_uri)
    identity = batch["identity"]
    assert (
        len(identity["requests"])
        == len(manifest["variants"])
        == config["variant_count"]
    ), "Live audit must cover every requested and committed variant"
    assert identity["run_id"] == config["run_id"]
    assert identity["runtime_image"] == recovery._image_binding(config["runtime_image"])
    assert identity["npa_sources"] == recovery._source_binding(), (
        "Actual source differs from this reviewed checkout"
    )
    assert identity["input_sha256"] == config["input_sha256"]
    coordinator = recovery.VariantRecovery.for_audit(storage, augment, attempt, batch)
    assert coordinator.batch_sha256 == manifest["recovery_batch_sha256"]
    source_bytes = storage.read_bytes_with_etag(identity["input_video_uri"])
    assert source_bytes is not None
    with tempfile.TemporaryDirectory(prefix="npa-paidf-recovery-live-audit-") as tmp:
        source = Path(tmp) / "source.mp4"
        source.write_bytes(source_bytes[0])
        assert video_sha256(source) == config["input_sha256"]
        for index, request in enumerate(identity["requests"]):
            verified = coordinator.recover(index, request, source)
            assert verified == manifest["variants"][index]

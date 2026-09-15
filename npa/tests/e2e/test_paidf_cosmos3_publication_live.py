"""Verify unblended publication of retained real Cosmos outputs through S3.

Set NPA_INTEGRATION_E2E=1, NPA_PAIDF_RAW_EVIDENCE_DIR (downloaded variant
directories with raw_cosmos_video.mp4 and metadata.json), NPA_PAIDF_REPAIR_URI
(a fresh S3 augment prefix), and NPA_PAIDF_REPAIR_DIR (private local readback).
This exercises publication without submitting GPU generation or promoting data.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient
from npa.workflows import paidf_cosmos3 as c3

pytestmark = pytest.mark.e2e


def _publication_inputs():
    names = ("NPA_PAIDF_RAW_EVIDENCE_DIR", "NPA_PAIDF_REPAIR_URI", "NPA_PAIDF_REPAIR_DIR")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or any(
        not os.environ.get(name) for name in names
    ):
        pytest.skip("requires retained real model output and a fresh private destination")
    source, destination, readback = (os.environ[name] for name in names)
    credentials = load_credentials()
    storage = StorageClient.from_environment(
        endpoint_url=credentials.s3_endpoint,
        aws_access_key_id=credentials.s3_access_key_id,
        aws_secret_access_key=credentials.s3_secret_access_key,
    )
    return Path(source), destination, Path(readback), storage


def _publish_retained_clip(folder, destination, storage):
    metadata = json.loads((folder / "metadata.json").read_text())
    raw = folder / "raw_cosmos_video.mp4"
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert metadata["engine"] == c3.ENGINE
    assert digest == metadata["motion_preservation"]["raw_cosmos_video_sha256"]
    assert not c3._list_keys(destination + "/" + folder.name, storage=storage)
    metadata["input_provenance_uri"] = metadata["lineage"]["input_provenance_uri"]
    variant = c3._publish_variant(
        result={"output_path": str(raw)}, output_uri=destination,
        clip=folder.name, variables=metadata["variables"], metadata=metadata,
        storage=storage,
    )
    return variant, digest


@pytest.mark.timeout(0)
def test_retained_cosmos_outputs_publish_without_blending():
    source, destination, readback, storage = _publication_inputs()
    folders = sorted(path.parent for path in source.glob("*/raw_cosmos_video.mp4"))
    assert folders, "requires real retained Cosmos generation artifacts"
    receipts = []
    for folder in folders:
        variant, digest = _publish_retained_clip(folder, destination, storage)
        local = readback / folder.name
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        video = local / "augmented_video.mp4"
        storage.download_path(variant["augmented_video_uri"], str(video))
        assert hashlib.sha256(video.read_bytes()).hexdigest() == digest
        metadata_uri = variant["augmented_video_uri"].rsplit("/", 1)[0] + "/metadata.json"
        storage.download_path(metadata_uri, str(local / "metadata.json"))
        metadata = json.loads((local / "metadata.json").read_text())
        assert metadata["motion_preservation"] is None
        assert metadata["published_video_sha256"] == digest
        assert variant["frame_count"] > 0
        receipts.append({**variant, "published_video_sha256": digest})
    (readback / "publication-evidence.json").write_text(json.dumps({
        "variants": receipts, "gpu_generation_submitted": False,
        "publication_verified": True, "training_approved": False,
    }, indent=2) + "\n")

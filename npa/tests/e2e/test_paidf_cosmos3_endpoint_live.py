"""Read-only audit of twelve actual hosted PAIDF variants; never submits jobs.

Set NPA_INTEGRATION_E2E=1, NPA_E2E_PAIDF_ENDPOINT_RUN_URI and the normal NPA S3
environment after running the main workflow with use_hosted_endpoint=true and
variant_count=12. This proves delivered generation, not quality acceptance or
completion of every downstream stage. Keep private run locations outside Git.
"""

import json
import os
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames
from npa.workflows.paidf_cosmos3 import (
    _read_json,
    _sha256,
    validate_committed_augment_manifest,
)
from npa.workflows.paidf_cosmos3_media import verify_pair
from npa.workflows.video_padding_preservation import resolve_source_content_region

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _assert_published_pixels(source, raw, published, region):
    width, height = region["canvas"]
    left, top, right, bottom = region["bounds"]
    mask = np.ones((height, width), dtype=bool)
    mask[top:bottom, left:right] = False
    frames = 0
    with ExitStack() as stack:
        streams = [
            stack.enter_context(closing(_iter_rgb_frames(path, height, width)))
            for path in (source, raw, published)
        ]
        for reference, generated, actual in zip(*streams, strict=True):
            assert np.array_equal(actual[mask], reference[mask])
            assert np.array_equal(
                actual[top:bottom, left:right], generated[top:bottom, left:right]
            )
            frames += 1
    assert frames == region["padding_detection"]["decoded_frames"]


def _audit_padding(storage, source, video, metadata, region, tmp_path):
    assert metadata["source_content_region"]["bounds"] == region["bounds"]
    width, height = region["canvas"]
    if region["bounds"] == [0, 0, width, height]:
        return
    receipt = metadata["padding_preservation"]
    assert receipt["status"] == "verified"
    assert receipt["source_sha256"] == _sha256(source)
    assert receipt["published_sha256"] == _sha256(video)
    raw = Path(storage.download_path(receipt["raw_model_video_uri"], str(tmp_path)))
    assert _sha256(raw) == receipt["raw_model_sha256"]
    _assert_published_pixels(source, raw, video, region)


def test_completed_twelve_variant_endpoint_delivery(tmp_path):
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("explicit live integration opt-in required")
    root = os.environ.get("NPA_E2E_PAIDF_ENDPOINT_RUN_URI", "").rstrip("/")
    if not root.startswith("s3://"):
        pytest.skip("provide the private completed endpoint run URI")
    storage = StorageClient.from_environment()
    augment = root + "/cosmos_augmented/"
    manifest = _read_json(augment + "manifest.json", storage=storage)
    variants = validate_committed_augment_manifest(manifest, augment)
    assert manifest["engine"] == "nebius-stream"
    assert manifest["variant_count"] == 12
    assert manifest["variant_parallelism"] == 12
    assert len({v["endpoint_job_id"] for v in variants}) == 12
    source = Path(
        storage.download_path(root + "/input/source.mp4", str(tmp_path / "source.mp4"))
    )
    source_hash = _sha256(source)
    source_region = resolve_source_content_region(source)
    for variant in variants:
        video = Path(
            storage.download_path(
                variant["augmented_video_uri"],
                str(tmp_path / (variant["clip"] + ".mp4")),
            )
        )
        alignment = verify_pair(source, video, 24)
        assert alignment == variant["temporal_alignment"]
        receipt = _read_json(variant["endpoint_execution_uri"], storage=storage)
        assert receipt["schema"] == "npa.cosmos3.stream-execution.v1"
        assert receipt["status"] == "verified" and receipt["guardrails"] is True
        assert receipt["job_id"] == variant["endpoint_job_id"]
        assert receipt["source_sha256"] == source_hash
        remote = receipt["manifest"]
        assert remote["model"] == "nvidia/" + manifest["model"]
        assert remote["effective"]["guardrails"] is True
        assert remote["effective"]["structural_control"] == "edge"
        metadata = _read_json(
            variant["augmented_video_uri"].replace(
                "augmented_video.mp4", "metadata.json"
            ),
            storage=storage,
        )
        assert metadata["engine"] == "nebius-stream"
        assert metadata["published_video_sha256"] == _sha256(video)
        assert "native_execution_uri" not in metadata
        _audit_padding(
            storage,
            source,
            video,
            metadata,
            source_region,
            tmp_path / (variant["clip"] + "-raw.mp4"),
        )
        state = _read_json(receipt["state_uri"], storage=storage)
        assert state["job"]["status"] == "succeeded"
        assert state["job_id"] == receipt["job_id"]
        assert state["request"]["generation"]["seed"] == variant["seed"]
        assert state["request"]["generation"]["num_inference_steps"] == variant["steps"]
        assert "api_key" not in json.dumps(receipt).lower()

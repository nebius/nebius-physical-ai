"""Detect and restore borders in retained real model outputs using live storage."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames
from npa.workflows.paidf_cosmos3_media import verify_pair
from npa.workflows.video_padding_detection import detect_source_padding
from npa.workflows.video_padding_preservation import preserve_source_padding
from .test_paidf_appearance_recipe_live import _download, _read
from .test_paidf_padding_evaluation_live import _cases

pytestmark = [pytest.mark.e2e]


def _assert_pixels(source, raw, published, region):
    width, height = region["canvas"]
    left, top, right, bottom = region["bounds"]
    mask = np.ones((height, width), bool)
    mask[top:bottom, left:right] = False
    decoded = [
        _iter_rgb_frames(path, height, width) for path in (source, raw, published)
    ]
    frames = 0
    for original, generated, actual in zip(*decoded, strict=True):
        assert np.array_equal(actual[mask], original[mask])
        assert np.array_equal(
            actual[top:bottom, left:right], generated[top:bottom, left:right]
        )
        frames += 1
    assert frames == region["padding_detection"]["decoded_frames"]


def _restore_variant(client, root, item, source, region, destination, expected_frames):
    clip = item["clip"]
    assert clip.startswith("variant-") and clip.removeprefix("variant-").isdigit()
    directory = destination / clip
    directory.mkdir()
    relative = "cosmos_augmented/" + clip + "/"
    metadata = _read(client, root, relative + "metadata.json")
    raw = _download(
        client,
        root,
        relative + "augmented_video.mp4",
        directory / "raw_model_video.mp4",
    )
    published = directory / "augmented_video.mp4"
    receipt = preserve_source_padding(source, raw, published, region)
    assert receipt["raw_model_sha256"] == metadata["published_video_sha256"]
    alignment = verify_pair(source, published)
    assert alignment["decoded_frames"] == expected_frames
    assert alignment["generated_sha256"] == receipt["published_sha256"]
    _assert_pixels(source, raw, published, region)
    (directory / "padding-preservation.json").write_text(json.dumps(receipt, indent=2))
    return {"clip": clip, **receipt}


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "case", _cases(), ids=lambda case: "retained-run" if case else "disabled"
)
def test_detect_and_preserve_padding_in_retained_outputs(case):
    """Read actual outputs and prove lossless scene and border preservation.

    Args:
        case: Private run location, label and expected decoded frame count.
    Returns:
        None.
    Raises:
        AssertionError: Detection, lineage, timing or exact pixels differ.
    """
    from npa.clients.project_credentials import s3_client_for_project

    client = s3_client_for_project(os.environ["NPA_E2E_PROJECT"])
    label = case["label"]
    assert label.isascii() and label.replace("-", "").isalnum()
    destination = Path(os.environ["NPA_PAIDF_PADDING_PRESERVATION_DIR"]) / label
    destination.mkdir(mode=0o700, parents=True)
    source = _download(
        client, case["run_uri"], "input/source.mp4", destination / "source.mp4"
    )
    region = detect_source_padding(source)  # Deliberately no preparation metadata.
    assert region["padding_detection"]["status"] == "detected"
    manifest = _read(client, case["run_uri"], "cosmos_augmented/manifest.json")
    assert manifest["status"] == "executed" and manifest["variants"]
    variants = [
        _restore_variant(
            client,
            case["run_uri"],
            item,
            source,
            region,
            destination,
            case["expected_frames"],
        )
        for item in manifest["variants"]
    ]
    (destination / "summary.json").write_text(
        json.dumps({"source_content_region": region, "variants": variants}, indent=2)
    )

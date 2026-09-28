"""Retain public scan attribution only after verifying the frozen capture lineage."""

from npa.workflows.field_failure.artifacts import _read
from npa.workflows.navigation.publication import _validate_record


def sample_credit(output_root, plan):
    """Resolve the sealed sample and credit the verified public office capture.

    Args:
        output_root: Run-scoped reference-demo S3 root.
        plan: Frozen reference plan with the measured capture manifest hash.
    Returns:
        Public attribution and capture identity, or None for a different dataset.
    Raises:
        ValueError: Publication, capture bytes or frozen geometry lineage differ.
        KeyError: Required frozen capture identity is absent.
        OSError: Published evidence cannot be retrieved.
    """
    source = output_root + "/sample"
    completion, _ = _read(source + "/completion.json")
    files = _validate_record(completion)
    if completion != _read(source + "/claim.json")[0]:
        raise ValueError("sample completion differs from its immutable claim")
    expected = plan["cohorts"]["geometry"]["capture_manifest_sha256"]
    if "capture.json" not in files or files["capture.json"] != expected:
        raise ValueError("sample capture differs from frozen geometry lineage")
    capture, _ = _read(
        source + "/" + completion["attempt"] + "/capture.json", expected=expected
    )
    credit = _public_credit(capture)
    return {**credit, "capture_manifest_sha256": expected} if credit else None


def _public_credit(capture):
    from npa.workbench.nurec.navigation_sample import (
        ARCHIVE_SHA256,
        ARCHIVE_URL,
        ATTRIBUTION,
    )

    source = capture.get("source", {})
    identity = {
        "dataset": "TUM RGB-D benchmark, fr3/long_office_household",
        "url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC-BY-4.0",
    }
    if any(source.get(name) != value for name, value in identity.items()):
        return None
    return {
        "attribution": ATTRIBUTION,
        "source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
        "license": "Creative Commons Attribution 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": ARCHIVE_SHA256,
        "modifications": "RGB-D capture fused into a measured TSDF surface and rendered in Isaac navigation rollouts. Scored frames are resized and JPEG encoded for display.",
    }

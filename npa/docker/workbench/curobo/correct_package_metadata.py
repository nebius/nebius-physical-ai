"""Correct one invalid classifier in the exact reviewed cuRobo source release.

The pinned build backend validates metadata normally. This is not a runtime
source patch: unexpected source, metadata or replacement bytes fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
SOURCE_ARCHIVE_SHA256 = "ea6d8310b1ab109ceaac046b15b4de31ecb7c3b52c2a4c762c58316c81cbbf2f"
UPSTREAM_METADATA_SHA256 = "4c93ee00a80dbc46e45fb6a1dd9486b57c8547e59bd948c30978a8ac7ed03a44"
CORRECTED_METADATA_SHA256 = "ca1967835fbf45a89617a70d5cbf596cd4368623b84a8013dfca2b6bedae32b9"
ORIGINAL_CLASSIFIER = b'"Topic :: Scientific/Engineering :: Robotics"'
CORRECTED_CLASSIFIER = b'"Topic :: Scientific/Engineering"'
CHANGE_NOTICE = (
    b"# NPA packaging change: the unregistered Robotics Trove classifier is replaced\n"
    b"# with its registered Scientific/Engineering parent; runtime source is unchanged.\n"
)


def correct_package_metadata(source_root: Path) -> dict:
    """Apply the single reviewed change and return its reproducible receipt."""
    metadata_path = source_root / "pyproject.toml"
    revision_path = source_root / "NPA_SOURCE_REVISION"
    for path in (metadata_path, revision_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("cuRobo metadata inputs must be regular, unlinked files")
    if revision_path.read_bytes() != (SOURCE_REVISION + "\n").encode():
        raise ValueError("cuRobo metadata correction requires the reviewed source revision")
    original = metadata_path.read_bytes()
    if hashlib.sha256(original).hexdigest() != UPSTREAM_METADATA_SHA256:
        raise ValueError("cuRobo upstream metadata hash does not match the reviewed release")
    if original.count(ORIGINAL_CLASSIFIER) != 1:
        raise ValueError("cuRobo metadata must contain exactly one reviewed classifier")
    corrected = CHANGE_NOTICE + original.replace(ORIGINAL_CLASSIFIER, CORRECTED_CLASSIFIER)
    if hashlib.sha256(corrected).hexdigest() != CORRECTED_METADATA_SHA256:
        raise ValueError("cuRobo corrected metadata hash does not match the reviewed change")
    # Every input and output byte is checked before changing the sole owned file.
    # The prominent comment preserves upstream's copyright and license notices.
    metadata_path.write_bytes(corrected)
    return {
        "schema_version": "npa.curobo.metadata-correction.v1",
        "source_revision": SOURCE_REVISION,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "changed_file": "pyproject.toml",
        "before_sha256": UPSTREAM_METADATA_SHA256,
        "after_sha256": CORRECTED_METADATA_SHA256,
        "field": "project.classifiers[5]",
        "original_classifier": ORIGINAL_CLASSIFIER.decode().strip('"'),
        "corrected_classifier": CORRECTED_CLASSIFIER.decode().strip('"'),
        "scope": "One classifier and a packaging-change comment; runtime source is unchanged.",
        "classifier_registry": "https://pypi.org/classifiers/",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(correct_package_metadata(args.source_root), indent=2))

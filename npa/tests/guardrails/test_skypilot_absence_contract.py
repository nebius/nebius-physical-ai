"""A Sky upgrade must re-review the complete absence naming source contract."""

import hashlib
import json
import re

from npa.orchestration.skypilot import _bin
from npa.orchestration.skypilot.absence_naming_contract import (
    NAMING_SOURCE_SHA256,
    NAMING_SOURCE_VERSION,
)


def test_absence_naming_version_matches_required_sky_pin():
    assert NAMING_SOURCE_VERSION == _bin.REQUIRED_SKYPILOT_VERSION


def test_reviewed_naming_contract_covers_nine_valid_source_hashes():
    assert len(NAMING_SOURCE_SHA256) == 9
    assert all(
        re.fullmatch(r"[a-f0-9]{64}", value) for value in NAMING_SOURCE_SHA256.values()
    )
    # Updating this digest requires the documented nine-source upgrade review.
    encoded = json.dumps(
        NAMING_SOURCE_SHA256, sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "24a0e3bb9254641bfa3dc71c70cf53e4e14944cdf4eedce4acccf2bccd438617"
    )

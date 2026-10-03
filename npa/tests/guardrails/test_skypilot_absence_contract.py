"""A Sky upgrade must re-review the complete absence naming source contract."""

import hashlib
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path

import pytest

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


def test_installed_skypilot_wheel_matches_reviewed_nine_source_hashes():
    spec = importlib.util.find_spec("sky")
    if spec is None:
        pytest.skip(
            "SkyPilot is installed separately; run this gate in its pinned environment"
        )
    assert importlib.metadata.version("skypilot") == _bin.REQUIRED_SKYPILOT_VERSION
    assert spec.origin is not None, "Installed Sky package source is unavailable"
    root = Path(spec.origin).parent.parent
    for name, expected in NAMING_SOURCE_SHA256.items():
        source = root / name
        assert source.is_file(), f"Missing reviewed native source: {name}"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name

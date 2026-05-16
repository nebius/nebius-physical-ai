from __future__ import annotations

import os

import pytest


@pytest.mark.e2e
def test_e2e_bucket_fixture_creates_and_cleans(
    e2e_test_bucket: str,
    s3_helper,
) -> None:
    """The e2e_test_bucket fixture creates a real bucket and later cleans it up."""
    assert e2e_test_bucket.startswith("npa-e2e-test-")
    assert s3_helper.list_objects(e2e_test_bucket) == []


@pytest.mark.e2e
def test_e2e_skip_when_env_not_set() -> None:
    """If this test runs, the gating environment variable is enabled."""
    assert os.getenv("NPA_INTEGRATION_E2E")

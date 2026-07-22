from __future__ import annotations

import pytest

from npa.orchestration.skypilot.capacity import is_capacity_error


@pytest.mark.parametrize(
    "text",
    [
        "sky.exceptions.ResourcesUnavailableError: No resources available",
        "Nebius: insufficient capacity for H100 in eu-north1",
        "InsufficientInstanceCapacity: try again later",
        "Error: out of capacity",
        "0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
        "Quota exceeded for accelerator H200",
        "No launchable resource found; retry later",
    ],
)
def test_capacity_errors_are_retryable(text: str) -> None:
    assert is_capacity_error(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Traceback: KeyError 'POLICY_IMAGE'",
        "ImagePullBackOff: manifest unknown",
        "AssertionError: expected artifact in s3://bucket/prefix",
        "Training step 0/500 complete",  # progress, not capacity
        "Permission denied (publickey)",
    ],
)
def test_non_capacity_errors_are_not_retryable(text: str | None) -> None:
    assert not is_capacity_error(text)

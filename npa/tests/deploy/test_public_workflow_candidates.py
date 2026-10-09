"""Keep workflow validation defaults governed without promoting stale releases."""

from copy import deepcopy

import pytest

from npa.deploy.images import (
    _validate_workflow_image_candidate,
    public_release_manifest,
)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("development_sha", "short-sha", "full 40-character"),
        ("published_digest", "mutable-tag", "digest is invalid"),
        ("validation_tool_refs", [], "scope is invalid"),
        ("validation_tool_refs", ["workbench.cosmos3.generate"], "scope is invalid"),
        (
            "validation_tool_refs",
            ["workbench.cosmos_curate.curate"] * 2,
            "scope is invalid",
        ),
        ("scope_reason", "", "needs a reason"),
        ("build_run_url", "https://example.invalid/build", "build evidence is invalid"),
    ],
)
def test_candidate_identity_and_scope_fail_closed(field, value, message) -> None:
    entry = deepcopy(
        public_release_manifest()["workflow_validation_candidates"]["cosmos-curate"]
    )
    entry[field] = value
    with pytest.raises((RuntimeError, ValueError), match=message):
        _validate_workflow_image_candidate("cosmos-curate", entry)


def test_candidate_cannot_claim_an_unrelated_tool() -> None:
    entry = public_release_manifest()["workflow_validation_candidates"]["cosmos-curate"]
    with pytest.raises(RuntimeError, match="Invalid public workflow candidate"):
        _validate_workflow_image_candidate("rerun-viewer", entry)


def test_candidates_remain_outside_accepted_release_inventory() -> None:
    manifest = public_release_manifest()
    candidates = set(manifest["workflow_validation_candidates"])
    assert candidates == {"cosmos3", "cosmos-evaluator", "cosmos-curate"}
    assert candidates <= set(manifest["publication_pending"])
    assert not candidates & set(manifest["releases"])

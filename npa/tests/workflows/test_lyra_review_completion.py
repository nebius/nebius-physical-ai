"""Prevent incomplete Lyra evidence from being published as a finished review."""

from copy import deepcopy

import pytest

from npa.workflows.lyra_demo import _require_complete


@pytest.fixture
def completed_review():
    return {
        "evidence": {
            "reconstruction": {"views": 128},
            "geometry": {"triangles": 100},
            "actions": {"attempted": 1, "accepted": 0},
        },
        "source_video": "data:video/mp4;base64,source",
        "reconstruction_video": "data:video/mp4;base64,rendered",
        "mesh": {"triangles": "triangles"},
        "trials": [{"accepted": False}],
    }


@pytest.mark.parametrize("stage", ["reconstruction", "geometry", "actions"])
def test_finished_review_rejects_missing_stage(completed_review, stage):
    completed_review["evidence"][stage] = None
    with pytest.raises(ValueError, match=stage):
        _require_complete(completed_review)


@pytest.mark.parametrize(
    "field", ["source_video", "reconstruction_video", "mesh", "trials"]
)
def test_finished_review_requires_displayable_artifacts(completed_review, field):
    completed_review[field] = None
    with pytest.raises(ValueError, match="Complete review requires"):
        _require_complete(completed_review)


def test_finished_review_distinguishes_failure_from_unexecuted(completed_review):
    _require_complete(completed_review)
    empty = deepcopy(completed_review)
    empty["evidence"]["actions"]["attempted"] = 0
    with pytest.raises(ValueError, match="native action attempt"):
        _require_complete(empty)

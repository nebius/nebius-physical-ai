from __future__ import annotations

import json
import math

import pytest

from npa.workflows.sim2real.camera_views import (
    CAMERA_VIEW_SPECS,
    camera_view_names,
    camera_views_json,
)


def test_default_camera_views_cover_primary_side_and_overhead() -> None:
    assert camera_view_names() == ("primary", "side", "overhead")
    payload = json.loads(camera_views_json())
    assert [view["name"] for view in payload] == ["primary", "side", "overhead"]


def test_camera_view_aliases_are_ordered_deduplicated_and_keep_primary() -> None:
    assert camera_view_names("left,top,left") == ("primary", "side", "overhead")
    assert camera_view_names("front,side") == ("primary", "side")


def test_camera_view_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown Sim2Real camera view"):
        camera_view_names("primary,diagonal")


def test_camera_view_quaternions_are_normalized_and_poses_are_distinct() -> None:
    positions = {spec.position for spec in CAMERA_VIEW_SPECS.values()}
    assert len(positions) == 3
    for spec in CAMERA_VIEW_SPECS.values():
        assert math.isclose(
            sum(value * value for value in spec.rotation), 1.0, abs_tol=1e-6
        )

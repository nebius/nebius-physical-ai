"""Publication refusal categories must remain effective when tools are combined."""

from __future__ import annotations

import pytest

from npa.deploy import images


@pytest.mark.parametrize(
    "inventory",
    (
        "RESTRICTED_PUBLICATION_TOOLS",
        "RESTRICTED_DERIVED_IMAGES",
        "PENDING_REDISTRIBUTION_TOOLS",
        "NEUTRAL_UNBUILT_CANDIDATE_TOOLS",
    ),
)
def test_each_refusal_category_blocks_an_existing_public_tool(monkeypatch, inventory):
    tool = "foxglove-embed"
    assert tool in images.publicly_publishable_tools()
    original = {
        name: getattr(images, name)
        for name in (
            "RESTRICTED_PUBLICATION_TOOLS",
            "RESTRICTED_DERIVED_IMAGES",
            "PENDING_REDISTRIBUTION_TOOLS",
            "NEUTRAL_UNBUILT_CANDIDATE_TOOLS",
        )
    }
    monkeypatch.setattr(images, inventory, original[inventory] | {tool})

    assert not images.is_publicly_redistributable(tool)
    assert tool not in images.publicly_publishable_tools()
    assert tool in images.restricted_image_names()
    assert tool in images.omniverse_restricted_image_names()
    for name, values in original.items():
        if name != inventory:
            assert getattr(images, name) == values

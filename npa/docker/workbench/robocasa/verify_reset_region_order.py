"""Verify native counter-region order across XML identity allocation histories."""

from __future__ import annotations

import json
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
from robocasa.models.fixtures.counter import Counter


def _counter(reverse_allocation: bool) -> Counter:
    """Keep physical candidate order fixed while varying element identity order."""
    elements = [ElementTree.Element("geom") for _ in range(2)]
    allocation_order = list(set(elements))
    left, right = allocation_order[::-1] if reverse_allocation else allocation_order
    for element, position in ((left, "-1 0 0"), (right, "1 0 0")):
        element.set("pos", position)
        element.set("size", "0.4 0.4 0.1")
    counter = Counter.__new__(Counter)
    counter._obj = ElementTree.Element(
        "body", pos="0 0 0", euler="0 0 0", quat="1 0 0 0"
    )
    counter.size = np.array([3.0, 1.0, 1.0])
    counter._name = "test_counter"
    counter._get_counter_geoms = lambda: {
        "top_left": [left],
        "top_right": [right],
        "top_duplicate": [left],
    }
    return counter


def verify_region_order() -> dict[str, int]:
    """Exercise actual native region enumeration without a simulator or renderer.

    Args:
        None.
    Returns:
        Number of native ordering and membership controls passed.
    Raises:
        AssertionError: Native regions change order or duplicate membership.
    """
    reference = SimpleNamespace(pos=np.zeros(3), rot=0.0, quat=np.array([1, 0, 0, 0]))
    env = SimpleNamespace(get_fixture=lambda fixture: fixture)
    expected = {
        "any": [-1.0, 1.0],
        "left_right": [-1.0, 1.0],
        "left": [-1.0],
        "right": [1.0],
    }
    count = 0
    for reverse in (False, True):
        counter = _counter(reverse)
        for location, offsets in expected.items():
            regions = counter.get_reset_regions(env, ref=reference, loc=location)
            observed = [float(region["offset"][0]) for region in regions.values()]
            assert observed == offsets, (
                "native_region_order_mismatch",
                reverse,
                location,
                observed,
                offsets,
            )
            assert list(regions) == [f"geom_{index}" for index in range(len(offsets))]
            count += 1
    return {"native_order_and_membership_controls": count}


if __name__ == "__main__":
    print(json.dumps(verify_region_order(), sort_keys=True))

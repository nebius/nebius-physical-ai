"""GPU stages must not become acceptable by deleting device provenance."""

from __future__ import annotations

import pytest
from test_sim2real_stage14_thirteenth_review_controls import (
    ROOT,
    SOURCE_SHA,
    _component,
    _evidence,
    _gold,
    _rehash,
)

from npa.workflows.sim2real import component_authority


@pytest.mark.parametrize("stage", [3, 7, 9, 10])
@pytest.mark.parametrize(
    "removed", [("gpu_products",), ("gpu_rows",), ("gpu_products", "gpu_rows")]
)
def test_stage14_rejects_missing_required_device_provenance(
    stage: int, removed: tuple[str, ...]
) -> None:
    components = [_component(index) for index in range(1, 14)]
    for key in removed:
        components[stage - 1]["artifacts"].pop(key)
    _rehash(components[stage - 1])

    with pytest.raises(ValueError, match=f"Stage {stage} execution provenance"):
        component_authority.validate_component_records(
            components,
            root=ROOT,
            evidence=_evidence(),
            gold=_gold(),
            expected_source_sha=SOURCE_SHA,
        )


def test_cpu_stage_does_not_require_device_provenance() -> None:
    components = [_component(index) for index in range(1, 14)]
    for key in ("gpu_products", "gpu_rows"):
        components[0]["artifacts"].pop(key)
    _rehash(components[0])

    component_authority.validate_component_records(
        components,
        root=ROOT,
        evidence=_evidence(),
        gold=_gold(),
        expected_source_sha=SOURCE_SHA,
    )

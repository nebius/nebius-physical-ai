"""Audit twelve fresh starter variants, with defaults or the checked-in recipe.

The first test covers the canonical workflow with only ``--var variant_count=12``.
Set NPA_INTEGRATION_E2E=1, NPA_E2E_PROJECT, NPA_E2E_PAIDF_FANOUT_RUN_URI and
NPA_E2E_PAIDF_FANOUT_FRESH_AFTER (the UTC timestamp before submission).
The second covers docs/workbench/examples/paidf-cosmos3-starter-12.yaml, using
NPA_E2E_PAIDF_STARTER12_RUN_URI and NPA_E2E_PAIDF_STARTER12_FRESH_AFTER.
These tests read completed outputs and never submit or resume GPU work.
"""

from pathlib import Path

import pytest
import yaml

from .test_paidf_cosmos3_starter_live import audit_starter_pipeline


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


def test_completed_fresh_twelve_variant_starter_pipeline() -> None:
    audit_starter_pipeline(
        "NPA_E2E_PAIDF_FANOUT_RUN_URI",
        "NPA_E2E_PAIDF_FANOUT_FRESH_AFTER",
        variant_count=12,
    )


RECIPE = (
    Path(__file__).resolve().parents[3]
    / "docs/workbench/examples/paidf-cosmos3-starter-12.yaml"
)


def test_completed_fresh_twelve_variant_starter_recipe_pipeline() -> None:
    audit_starter_pipeline(
        "NPA_E2E_PAIDF_STARTER12_RUN_URI",
        "NPA_E2E_PAIDF_STARTER12_FRESH_AFTER",
        variant_count=12,
        recipe=yaml.safe_load(RECIPE.read_text())["config"],
    )

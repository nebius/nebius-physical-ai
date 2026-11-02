"""Audit twelve fresh starter variants with the main workflow's current defaults.

Run the canonical workflow with only ``--var variant_count=12`` plus private
storage and GPU placement selections. Set NPA_INTEGRATION_E2E=1,
NPA_E2E_PROJECT, NPA_E2E_PAIDF_FANOUT_RUN_URI and
NPA_E2E_PAIDF_FANOUT_FRESH_AFTER (the UTC timestamp before submission).
This test reads completed outputs and never submits or resumes GPU work.
"""

import pytest

from .test_paidf_cosmos3_starter_live import audit_starter_pipeline


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]


def test_completed_fresh_twelve_variant_starter_pipeline() -> None:
    audit_starter_pipeline(
        "NPA_E2E_PAIDF_FANOUT_RUN_URI",
        "NPA_E2E_PAIDF_FANOUT_FRESH_AFTER",
        variant_count=12,
    )

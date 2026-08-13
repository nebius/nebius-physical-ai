"""LTX-2.5 workbench tool: operator licensing gate and output provenance.

The container definition lives in ``npa/docker/workbench/ltx2/``; it bakes no
LTX-2.5 code and no LTX-2.5 weights. This package holds the parts that must be
unit-testable without a GPU: validating the operator's declaration under the
LTX-2.x Community License Agreement, stamping that declaration onto generated
artifacts, and the fail-closed gate that keeps commercially-declared output out
of downstream trainers.
"""

from __future__ import annotations

from npa.workbench.ltx2.licensing import (
    GateDecision,
    LicenseDeclaration,
    LtxLicenseError,
    ProvenanceRecord,
    check_training_consumer,
    declaration_from_env,
)

__all__ = [
    "GateDecision",
    "LicenseDeclaration",
    "LtxLicenseError",
    "ProvenanceRecord",
    "check_training_consumer",
    "declaration_from_env",
]

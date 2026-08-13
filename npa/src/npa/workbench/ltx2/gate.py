"""Stamp and enforce LTX-2.5 output provenance across a workflow.

Two operations, one restriction. :func:`stamp_run` writes the provenance
manifest next to the video a run generated, recording which licence text the
operator accepted and what Attachment A(18) therefore permits. :func:`gate_run`
reads that manifest back in a later workflow state and decides whether a named
consumer — a policy trainer, a fine-tuning job — may use those artifacts.

The split matters because the two run in different containers on different
nodes, and the second one must not be able to reach the permissive answer by
accident. Anything the gate cannot read, cannot parse, or does not recognise
denies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from npa.workbench.ltx2.artifacts import (
    GATE_REPORT_FILENAME,
    MANIFEST_FILENAME,
    load_manifest,
    resolve_uri,
    write_json,
)
from npa.workbench.ltx2.licensing import (
    GateDecision,
    LicenseDeclaration,
    ProvenanceRecord,
    check_training_consumer,
    declaration_from_env,
)

GATE_SCHEMA = "npa.ltx2.gate.v1"


@dataclass(frozen=True)
class StampResult:
    """Where the provenance manifest landed and what it says."""

    manifest_uri: str
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"manifest_uri": self.manifest_uri, "manifest": self.manifest}


@dataclass(frozen=True)
class GateResult:
    """A gate decision plus the report artifact recording it."""

    decision: GateDecision
    consumer: str
    manifest_uri: str
    report_uri: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "consumer": self.consumer,
            "manifest_uri": self.manifest_uri,
        }
        payload.update(self.decision.as_dict())
        if self.report_uri:
            payload["report_uri"] = self.report_uri
        return payload


def stamp_run(
    *,
    run_id: str,
    outputs: list[str],
    manifest_uri: str,
    env: Mapping[str, str],
    model_files: list[str] | None = None,
    storage: Any | None = None,
) -> StampResult:
    """Validate the operator declaration and stamp it onto a run's outputs.

    Raises :class:`~npa.workbench.ltx2.licensing.LtxLicenseError` when the
    declaration is absent or invalid, so an unlicensed run cannot produce a
    manifest that would later read as permission.
    """

    declaration: LicenseDeclaration = declaration_from_env(env)
    record = ProvenanceRecord(
        declaration=declaration,
        run_id=run_id,
        outputs=tuple(outputs),
        model_files=tuple(model_files or ()),
    )
    payload = record.as_dict()
    written = write_json(
        payload, manifest_uri, filename=MANIFEST_FILENAME, storage=storage
    )
    return StampResult(manifest_uri=written, manifest=payload)


def gate_run(
    *,
    manifest_uri: str,
    consumer: str,
    report_uri: str = "",
    storage: Any | None = None,
) -> GateResult:
    """Decide whether *consumer* may train on the artifacts *manifest_uri* covers."""

    manifest = load_manifest(manifest_uri, storage=storage)
    decision = check_training_consumer(manifest, consumer=consumer)
    resolved_manifest_uri = resolve_uri(manifest_uri, filename=MANIFEST_FILENAME)
    result = GateResult(
        decision=decision,
        consumer=consumer,
        manifest_uri=resolved_manifest_uri,
    )
    if not report_uri:
        return result

    written = write_json(
        result.as_dict(), report_uri, filename=GATE_REPORT_FILENAME, storage=storage
    )
    return GateResult(
        decision=decision,
        consumer=consumer,
        manifest_uri=resolved_manifest_uri,
        report_uri=written,
    )


__all__ = ["GATE_SCHEMA", "GateResult", "StampResult", "gate_run", "stamp_run"]

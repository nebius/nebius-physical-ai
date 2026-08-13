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
    PROVENANCE_SCHEMA,
    GateDecision,
    LicenseDeclaration,
    LtxLicenseError,
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
    artifacts: tuple[str, ...] = ()
    report_uri: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "consumer": self.consumer,
            "manifest_uri": self.manifest_uri,
            "artifacts": list(self.artifacts),
        }
        payload.update(self.decision.as_dict())
        if self.report_uri:
            payload["report_uri"] = self.report_uri
        return payload


def _reconcile_with_the_run(
    declaration: LicenseDeclaration,
    *,
    declaration_uri: str,
    storage: Any | None,
) -> None:
    """Require this state's declaration to match the one the generation ran under.

    ``stamp`` executes in a different container, on a different node, with its own
    environment — so deriving the declaration from ``os.environ`` alone leaves a
    seam exactly where the licence chain of custody must not have one. If the
    operator's environment drifted between states, or secret forwarding differs,
    the manifest the gate later trusts would record this state's opinion rather
    than the run's.

    The GPU container wrote what it actually ran under. Any disagreement, and any
    inability to read or recognise that record, refuses.
    """

    from npa.workbench.ltx2.artifacts import DECLARATION_FILENAME, load_manifest

    recorded = load_manifest(
        declaration_uri, storage=storage, filename=DECLARATION_FILENAME
    )
    if not isinstance(recorded, Mapping):
        raise LtxLicenseError(
            f"Cannot read the generation's own declaration at {declaration_uri}. "
            "Refusing to stamp a manifest from this state's environment alone: "
            "that would record what we believe now, not what the run did."
        )
    if recorded.get("schema") != PROVENANCE_SCHEMA:
        raise LtxLicenseError(
            f"The generation's declaration has schema {recorded.get('schema')!r}, "
            f"expected {PROVENANCE_SCHEMA}. Refusing rather than guessing."
        )

    stated = recorded.get("operator_declaration")
    stated = stated if isinstance(stated, Mapping) else {}
    restrictions = recorded.get("restrictions")
    restrictions = restrictions if isinstance(restrictions, Mapping) else {}
    mismatches = [
        f"{field}: the run declared {theirs!r}, this state has {ours!r}"
        for field, theirs, ours in (
            ("entity_class", stated.get("entity_class"), declaration.entity_class),
            ("use_class", stated.get("use_class"), declaration.use_class),
            (
                "derived_model_training",
                restrictions.get("derived_model_training"),
                declaration.derived_model_training,
            ),
        )
        if theirs != ours
    ]
    if mismatches:
        raise LtxLicenseError(
            "The declaration this state would stamp disagrees with the one the "
            "generation ran under:\n  " + "\n  ".join(mismatches) + "\n"
            "Refusing to overwrite the run's own record. Re-run generation under "
            "the declaration you mean, or fix this state's environment."
        )


def stamp_run(
    *,
    run_id: str,
    outputs: list[str],
    manifest_uri: str,
    env: Mapping[str, str],
    model_files: list[str] | None = None,
    storage: Any | None = None,
    declaration_uri: str = "",
) -> StampResult:
    """Validate the operator declaration and stamp it onto a run's outputs.

    Raises :class:`~npa.workbench.ltx2.licensing.LtxLicenseError` when the
    declaration is absent or invalid, so an unlicensed run cannot produce a
    manifest that would later read as permission — and, when *declaration_uri* is
    given, when it disagrees with what the generation actually ran under.
    """

    declaration: LicenseDeclaration = declaration_from_env(env)
    if declaration_uri:
        _reconcile_with_the_run(
            declaration, declaration_uri=declaration_uri, storage=storage
        )
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
    artifacts: list[str] | None = None,
) -> GateResult:
    """Decide whether *consumer* may train on the artifacts *manifest_uri* covers.

    Naming *artifacts* binds the decision to specific bytes: the manifest has to
    claim them, so one run's permissive manifest cannot clear another run's video.
    """

    manifest = load_manifest(manifest_uri, storage=storage)
    decision = check_training_consumer(
        manifest, consumer=consumer, artifacts=tuple(artifacts or ())
    )
    resolved_manifest_uri = resolve_uri(manifest_uri, filename=MANIFEST_FILENAME)
    result = GateResult(
        decision=decision,
        consumer=consumer,
        manifest_uri=resolved_manifest_uri,
        artifacts=tuple(artifacts or ()),
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
        artifacts=tuple(artifacts or ()),
        report_uri=written,
    )


__all__ = ["GATE_SCHEMA", "GateResult", "StampResult", "gate_run", "stamp_run"]

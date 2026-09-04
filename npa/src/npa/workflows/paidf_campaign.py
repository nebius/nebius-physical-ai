"""Provider-neutral contracts for reusing one PAIDF campaign across partners.

The base campaign is immutable. Partner runs consume its exact artifact
references and publish to a separate prefix. This module defines the handoff
envelopes, the provider-neutral candidate/decision/reconciliation contracts,
and accepted-replacement selection; provider execution remains in
provider-specific code or workflow stages.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


CAMPAIGN_SCHEMA = "npa.paidf.campaign.v1"
FROZEN_SOURCE_MANIFEST_SCHEMA = "npa.paidf.frozen-source-manifest.v1"
PARTNER_INPUT_SCHEMA = "npa.paidf.partner-input.v1"
EXECUTION_RECEIPT_SCHEMA = "npa.paidf.execution-receipt.v1"
PARTNER_RESULT_SCHEMA = "npa.paidf.partner-result.v1"
CANDIDATE_MANIFEST_SCHEMA = "npa.paidf.candidates.v1"
DECISION_MANIFEST_SCHEMA = "npa.paidf.decisions.v1"
RECONCILIATION_SCHEMA = "npa.paidf.reconciliation.v1"
ACCEPTED_REPLACEMENTS_SCHEMA = "npa.paidf.accepted-variants.v1"

BASE_STAGES = ("source", "generation", "evaluation")
PARTNER_ROLES = frozenset({"curation", "observability", "simulation"})
PROVIDER_ROLES = frozenset({"curation", "evaluation"})
TERMINAL_STATUSES = frozenset({"completed", "failed"})
DECISION_STATES = ("accept", "reject", "review")
DECISION_FIELDS = frozenset(
    {"candidate_id", "decision", "evidence", "reason", "decided_at"}
)
PROVIDER_FIELDS = frozenset({"name", "role"})
VARIANT_IDENTITY_FIELDS = frozenset({"variant_id", "output_sha256", "source_sha256"})
RECONCILIATION_TOTAL_FIELDS = frozenset(
    {"candidates", "keep", "drop", "unresolved_review", "identity_gaps"}
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_REF = re.compile(r"^(?:env|secret)://[A-Za-z0-9._/-]+$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class PaidfContractError(ValueError):
    """Raised when a PAIDF campaign or partner envelope is unsafe or invalid."""


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an immutable-base campaign manifest.

    The campaign carries exactly the three base stages; every stage artifact's
    digest must equal its stage fingerprint, so a campaign cannot claim one
    identity while pinning another.
    """

    campaign = _object(payload, "campaign")
    _exact_schema(campaign, CAMPAIGN_SCHEMA, "campaign")
    _identifier(campaign.get("campaign_id"), "campaign.campaign_id")

    artifacts = _object(campaign.get("artifacts"), "campaign.artifacts")
    fingerprints = _object(
        campaign.get("stage_fingerprints"), "campaign.stage_fingerprints"
    )
    if set(artifacts) != set(BASE_STAGES):
        raise PaidfContractError(
            "campaign artifacts must contain exactly the base stages: "
            + ", ".join(BASE_STAGES)
        )
    if set(fingerprints) != set(BASE_STAGES):
        raise PaidfContractError(
            "campaign stage fingerprints must contain exactly the base stages: "
            + ", ".join(BASE_STAGES)
        )
    for stage in BASE_STAGES:
        ref = _artifact_ref(artifacts[stage], f"campaign.artifacts.{stage}")
        fingerprint = _sha256(
            fingerprints[stage], f"campaign.stage_fingerprints.{stage}"
        )
        if ref["sha256"] != fingerprint:
            raise PaidfContractError(
                f"campaign artifact digest for {stage} does not match its "
                "stage fingerprint"
            )

    return dict(campaign)


def build_partner_input(
    campaign: Mapping[str, Any],
    *,
    campaign_manifest_uri: str,
    run_id: str,
    provider: str,
    role: str,
    provider_config: Mapping[str, Any],
    output_prefix: str,
    executed_stages: Iterable[str],
    credential_refs: Iterable[str] = (),
    reused_stages: Iterable[str] = BASE_STAGES,
) -> dict[str, Any]:
    """Build a provider-neutral input envelope for a derivative partner run."""

    base = validate_campaign(campaign)
    reuse = _stage_list(list(reused_stages), "reused_stages")
    execute = _stage_list(list(executed_stages), "executed_stages")
    unknown_reuse = set(reuse) - set(BASE_STAGES)
    if unknown_reuse:
        raise PaidfContractError(
            "partner input contains unknown base stages: "
            + ", ".join(sorted(unknown_reuse))
        )
    artifacts = _object(base["artifacts"], "campaign.artifacts")
    payload: dict[str, Any] = {
        "schema": PARTNER_INPUT_SCHEMA,
        "run_id": run_id,
        "parent_campaign": {
            "campaign_id": base["campaign_id"],
            "manifest_uri": campaign_manifest_uri,
            "sha256": canonical_digest(base),
        },
        "provider": {"name": provider, "role": role},
        "provider_config": dict(provider_config),
        "credential_refs": list(credential_refs),
        "source_mode": "read-only",
        "reuse": {
            "stages": reuse,
            "artifacts": {stage: artifacts[stage] for stage in reuse},
        },
        "execute": {"stages": execute},
        "output_prefix": output_prefix,
    }
    return validate_partner_input(payload, campaign=base)


def build_antioch_input(
    campaign: Mapping[str, Any],
    *,
    campaign_manifest_uri: str,
    run_id: str,
    provider_config: Mapping[str, Any],
    output_prefix: str,
    credential_refs: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the thin Antioch overlay without rerunning base PAIDF stages."""

    return build_partner_input(
        campaign,
        campaign_manifest_uri=campaign_manifest_uri,
        run_id=run_id,
        provider="antioch",
        role="simulation",
        provider_config=provider_config,
        output_prefix=output_prefix,
        executed_stages=(
            "partner-simulation",
            "result-normalization",
            "comparison",
        ),
        credential_refs=credential_refs,
        reused_stages=BASE_STAGES,
    )


def validate_partner_input(
    payload: Mapping[str, Any], *, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a derivative run against the exact immutable base campaign."""

    base = validate_campaign(campaign)
    partner_input = _object(payload, "partner input")
    _exact_schema(partner_input, PARTNER_INPUT_SCHEMA, "partner input")
    _identifier(partner_input.get("run_id"), "partner_input.run_id")

    parent = _object(
        partner_input.get("parent_campaign"), "partner_input.parent_campaign"
    )
    if parent.get("campaign_id") != base["campaign_id"]:
        raise PaidfContractError("partner input references a different campaign ID")
    _s3_uri(
        parent.get("manifest_uri"),
        "partner_input.parent_campaign.manifest_uri",
        object_required=True,
    )
    if parent.get("sha256") != canonical_digest(base):
        raise PaidfContractError("partner input campaign digest does not match")

    provider = _object(partner_input.get("provider"), "partner_input.provider")
    _identifier(provider.get("name"), "partner_input.provider.name")
    if provider.get("role") not in PARTNER_ROLES:
        raise PaidfContractError(
            "partner_input.provider.role must be curation, observability, or simulation"
        )
    _artifact_ref(
        partner_input.get("provider_config"), "partner_input.provider_config"
    )
    credential_refs = _list(
        partner_input.get("credential_refs"), "partner_input.credential_refs"
    )
    for index, ref in enumerate(credential_refs):
        if not isinstance(ref, str) or not _CREDENTIAL_REF.fullmatch(ref):
            raise PaidfContractError(
                "partner_input.credential_refs"
                f"[{index}] must be an env:// or secret:// reference"
            )
    if partner_input.get("source_mode") != "read-only":
        raise PaidfContractError("partner input must consume the base as read-only")

    reuse = _object(partner_input.get("reuse"), "partner_input.reuse")
    reused_stages = _stage_list(reuse.get("stages"), "partner_input.reuse.stages")
    if not reused_stages:
        raise PaidfContractError("partner input must reuse at least one base stage")
    unknown_reuse = set(reused_stages) - set(BASE_STAGES)
    if unknown_reuse:
        raise PaidfContractError(
            "partner input contains unknown base stages: "
            + ", ".join(sorted(unknown_reuse))
        )
    reused_artifacts = _object(
        reuse.get("artifacts"), "partner_input.reuse.artifacts"
    )
    if set(reused_artifacts) != set(reused_stages):
        raise PaidfContractError(
            "partner input reuse artifacts must exactly match reused stages"
        )
    base_artifacts = _object(base["artifacts"], "campaign.artifacts")
    for stage in reused_stages:
        _artifact_ref(
            reused_artifacts.get(stage),
            f"partner_input.reuse.artifacts.{stage}",
        )
        if reused_artifacts.get(stage) != base_artifacts[stage]:
            raise PaidfContractError(
                f"partner input changed the base artifact reference for {stage}"
            )

    execute = _object(partner_input.get("execute"), "partner_input.execute")
    executed_stages = _stage_list(
        execute.get("stages"), "partner_input.execute.stages"
    )
    if not executed_stages:
        raise PaidfContractError("partner input must execute at least one overlay stage")
    forbidden = set(executed_stages) & set(BASE_STAGES)
    if forbidden:
        raise PaidfContractError(
            "partner runs cannot execute immutable base stages: "
            + ", ".join(sorted(forbidden))
        )
    overlap = set(executed_stages) & set(reused_stages)
    if overlap:
        raise PaidfContractError(
            "partner stages cannot be both reused and executed: "
            + ", ".join(sorted(overlap))
        )

    output_prefix = _s3_uri(
        partner_input.get("output_prefix"),
        "partner_input.output_prefix",
        object_required=False,
    )
    if not output_prefix.endswith("/"):
        raise PaidfContractError("partner_input.output_prefix must end with '/'")
    campaign_prefix = _s3_parent(parent["manifest_uri"])
    if output_prefix.startswith(campaign_prefix) or campaign_prefix.startswith(
        output_prefix
    ):
        raise PaidfContractError(
            "partner output prefix must be separate from the base campaign"
        )

    return dict(partner_input)


def build_execution_receipt(
    partner_input: Mapping[str, Any],
    *,
    status: str,
    stage_results: Mapping[str, str],
) -> dict[str, Any]:
    """Build a receipt that records which declared overlay stages ran."""

    receipt: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "run_id": partner_input.get("run_id"),
        "input_sha256": canonical_digest(partner_input),
        "provider": partner_input.get("provider"),
        "status": status,
        "stage_results": dict(stage_results),
        "source_mutated": False,
    }
    return validate_execution_receipt(receipt, partner_input=partner_input)


def validate_execution_receipt(
    payload: Mapping[str, Any], *, partner_input: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate execution evidence against the exact partner input envelope."""

    receipt = _object(payload, "execution receipt")
    _exact_schema(receipt, EXECUTION_RECEIPT_SCHEMA, "execution receipt")
    if receipt.get("run_id") != partner_input.get("run_id"):
        raise PaidfContractError("execution receipt run ID does not match input")
    if receipt.get("input_sha256") != canonical_digest(partner_input):
        raise PaidfContractError("execution receipt input digest does not match")
    if receipt.get("provider") != partner_input.get("provider"):
        raise PaidfContractError("execution receipt provider does not match input")
    status = receipt.get("status")
    if status not in TERMINAL_STATUSES:
        raise PaidfContractError("execution receipt status must be completed or failed")
    if receipt.get("source_mutated") is not False:
        raise PaidfContractError("execution receipt must prove source_mutated is false")

    stage_results = _object(
        receipt.get("stage_results"), "execution_receipt.stage_results"
    )
    declared = _object(partner_input.get("execute"), "partner_input.execute").get(
        "stages"
    )
    if set(stage_results) != set(_stage_list(declared, "partner_input.execute.stages")):
        raise PaidfContractError(
            "execution receipt must account for every declared overlay stage"
        )
    allowed_stage_statuses = {"completed", "failed", "skipped"}
    if any(value not in allowed_stage_statuses for value in stage_results.values()):
        raise PaidfContractError("execution receipt contains an invalid stage status")
    if status == "completed" and any(
        value != "completed" for value in stage_results.values()
    ):
        raise PaidfContractError(
            "completed execution receipt requires every stage to be completed"
        )
    return dict(receipt)


def build_partner_result(
    partner_input: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the terminal partner result without changing the base campaign."""

    validate_execution_receipt(execution_receipt, partner_input=partner_input)
    result: dict[str, Any] = {
        "schema": PARTNER_RESULT_SCHEMA,
        "run_id": partner_input.get("run_id"),
        "input_sha256": canonical_digest(partner_input),
        "execution_receipt_sha256": canonical_digest(execution_receipt),
        "parent_campaign": partner_input.get("parent_campaign"),
        "provider": partner_input.get("provider"),
        "status": execution_receipt.get("status"),
        "artifacts": dict(artifacts),
        "source_mutated": False,
    }
    return validate_partner_result(
        result,
        partner_input=partner_input,
        execution_receipt=execution_receipt,
    )


def validate_partner_result(
    payload: Mapping[str, Any],
    *,
    partner_input: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a partner result and its isolated artifact write surface."""

    validate_execution_receipt(execution_receipt, partner_input=partner_input)
    result = _object(payload, "partner result")
    _exact_schema(result, PARTNER_RESULT_SCHEMA, "partner result")
    if result.get("run_id") != partner_input.get("run_id"):
        raise PaidfContractError("partner result run ID does not match input")
    if result.get("input_sha256") != canonical_digest(partner_input):
        raise PaidfContractError("partner result input digest does not match")
    if result.get("execution_receipt_sha256") != canonical_digest(execution_receipt):
        raise PaidfContractError("partner result receipt digest does not match")
    for field in ("parent_campaign", "provider"):
        if result.get(field) != partner_input.get(field):
            raise PaidfContractError(f"partner result {field} does not match input")
    if result.get("status") != execution_receipt.get("status"):
        raise PaidfContractError("partner result status does not match receipt")
    if result.get("source_mutated") is not False:
        raise PaidfContractError("partner result must prove source_mutated is false")

    output_prefix = str(partner_input.get("output_prefix"))
    artifacts = _object(result.get("artifacts"), "partner_result.artifacts")
    if result.get("status") == "completed" and not artifacts:
        raise PaidfContractError("completed partner result requires artifacts")
    for name, artifact in artifacts.items():
        _identifier(name, f"partner_result.artifacts.{name}.name")
        ref = _artifact_ref(artifact, f"partner_result.artifacts.{name}")
        if not ref["uri"].startswith(output_prefix):
            raise PaidfContractError(
                f"partner result artifact {name} is outside the output prefix"
            )
    return dict(result)


def validate_candidate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a provider-neutral candidate manifest (one row per variant).

    Every candidate carries a stable ``candidate_id`` and the exact source
    identity it was generated from. Candidate IDs must be unique; source
    episode, camera, variant, and evaluation references must be present.
    """

    manifest = _object(payload, "candidate manifest")
    _exact_schema(manifest, CANDIDATE_MANIFEST_SCHEMA, "candidate manifest")
    candidates = _list(manifest.get("candidates"), "candidate manifest.candidates")
    seen: set[str] = set()
    for index, raw in enumerate(candidates):
        candidate = _object(raw, f"candidate manifest.candidates[{index}]")
        candidate_id = _identifier(
            candidate.get("candidate_id"),
            f"candidate manifest.candidates[{index}].candidate_id",
        )
        if candidate_id in seen:
            raise PaidfContractError(
                f"duplicate candidate_id in candidate manifest: {candidate_id}"
            )
        seen.add(candidate_id)
        _identifier(
            candidate.get("source_episode_id"),
            f"candidate {candidate_id}.source_episode_id",
        )
        _source_episode_index(
            candidate.get("source_episode_index"),
            f"candidate {candidate_id}.source_episode_index",
        )
        _identifier(
            candidate.get("camera_key"), f"candidate {candidate_id}.camera_key"
        )
        variant = _object(candidate.get("variant"), f"candidate {candidate_id}.variant")
        _identifier(variant.get("variant_id"), f"candidate {candidate_id}.variant.variant_id")
        _sha256(
            variant.get("output_sha256"),
            f"candidate {candidate_id}.variant.output_sha256",
        )
        _sha256(
            variant.get("source_sha256"),
            f"candidate {candidate_id}.variant.source_sha256",
        )
        _object(candidate.get("evaluation"), f"candidate {candidate_id}.evaluation")
    return dict(manifest)


def build_candidate(
    *,
    candidate_id: str,
    source_episode_id: str,
    source_episode_index: int,
    camera_key: str,
    variant_id: str,
    output_sha256: str,
    source_sha256: str,
    evaluation: Mapping[str, Any],
    variant_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one candidate record for a candidate manifest."""

    candidate: dict[str, Any] = {
        "candidate_id": _identifier(candidate_id, "candidate_id"),
        "source_episode_id": _identifier(source_episode_id, "source_episode_id"),
        "source_episode_index": _source_episode_index(
            source_episode_index, "source_episode_index"
        ),
        "camera_key": _identifier(camera_key, "camera_key"),
        "variant": {
            "variant_id": _identifier(variant_id, "variant_id"),
            "output_sha256": _sha256(output_sha256, "output_sha256"),
            "source_sha256": _sha256(source_sha256, "source_sha256"),
        },
        "evaluation": _object(evaluation, "evaluation"),
    }
    if variant_extras:
        extras = dict(variant_extras)
        overlap = set(extras) & set(VARIANT_IDENTITY_FIELDS)
        if overlap:
            raise PaidfContractError(
                "variant_extras cannot replace candidate identity fields: "
                + ", ".join(sorted(overlap))
            )
        candidate["variant"].update(extras)
    return candidate


def validate_decision_manifest(
    payload: Mapping[str, Any], *, candidates: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a provider decision manifest against the exact candidate set.

    The provider is exactly ``name`` plus a ``curation`` or ``evaluation``
    role, and every decision carries exactly ``candidate_id``, ``decision``,
    ``evidence``, ``reason``, and a caller-supplied ``decided_at`` in
    canonical UTC-second form. Fail closed unless every candidate has exactly
    one decision with an ``accept``, ``reject``, or ``review`` state, and no
    decision names an unknown candidate.
    """

    validate_candidate_manifest(candidates)
    manifest = _object(payload, "decision manifest")
    _exact_schema(manifest, DECISION_MANIFEST_SCHEMA, "decision manifest")
    _decision_provider(manifest.get("provider"), "decision manifest.provider")
    expected = {
        entry["candidate_id"]
        for entry in _list(candidates["candidates"], "candidates")
    }
    decisions = _list(manifest.get("decisions"), "decision manifest.decisions")
    seen: set[str] = set()
    for index, raw in enumerate(decisions):
        decision = _object(raw, f"decision manifest.decisions[{index}]")
        candidate_id = _identifier(
            decision.get("candidate_id"), f"decisions[{index}].candidate_id"
        )
        if candidate_id not in expected:
            raise PaidfContractError(
                f"decision names an unknown candidate: {candidate_id}"
            )
        if candidate_id in seen:
            raise PaidfContractError(
                f"candidate has more than one decision: {candidate_id}"
            )
        seen.add(candidate_id)
        if set(decision) != set(DECISION_FIELDS):
            raise PaidfContractError(
                f"decision for {candidate_id} must contain exactly "
                "candidate_id, decision, evidence, reason, and decided_at"
            )
        state = decision.get("decision")
        if state not in DECISION_STATES:
            raise PaidfContractError(
                f"decision for {candidate_id} must be one of "
                f"{', '.join(DECISION_STATES)}; got {state!r}"
            )
        _object(decision.get("evidence"), f"decisions[{index}].evidence")
        if not isinstance(decision.get("reason"), str) or not decision["reason"]:
            raise PaidfContractError(
                f"decision for {candidate_id} requires a nonempty reason"
            )
        _utc_timestamp(decision.get("decided_at"), f"decisions[{index}].decided_at")
    missing = expected - seen
    if missing:
        raise PaidfContractError(
            "decisions are missing for candidates: "
            + ", ".join(sorted(missing))
        )
    return dict(manifest)


def reconcile_decisions(
    candidates: Mapping[str, Any], decisions: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconcile decisions into the authoritative keep/drop manifest.

    Only ``accept`` decisions enter the training set. ``review`` is a valid
    but unresolved state: it is reported explicitly and excluded from the
    training set so a later human process can resolve it without blocking the
    headless run.
    """

    validate_decision_manifest(decisions, candidates=candidates)
    by_id = {
        decision["candidate_id"]: decision
        for decision in decisions["decisions"]
    }
    keep: list[dict[str, Any]] = []
    drop: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for candidate in candidates["candidates"]:
        candidate_id = candidate["candidate_id"]
        decision = by_id[candidate_id]
        record = {
            "candidate_id": candidate_id,
            "decision": decision["decision"],
            "reason": decision["reason"],
        }
        if decision["decision"] == "accept":
            keep.append(record)
        else:
            drop.append(record)
            if decision["decision"] == "review":
                unresolved.append(candidate_id)
    return {
        "schema": RECONCILIATION_SCHEMA,
        "totals": {
            "candidates": len(candidates["candidates"]),
            "keep": len(keep),
            "drop": len(drop),
            "unresolved_review": len(unresolved),
            "identity_gaps": 0,
        },
        "keep": keep,
        "drop": drop,
        "unresolved_review_candidate_ids": unresolved,
        "provider": decisions.get("provider"),
    }


def reconcile_provider_export(
    candidates: Mapping[str, Any],
    export_items: Iterable[Mapping[str, Any]],
    *,
    external_id_field: str = "external_id",
) -> dict[str, Any]:
    """Reconcile a provider export to the candidate manifest by exact ID.

    ``export_items`` are provider-side records; each must carry the Workbench
    ``candidate_id`` in ``external_id_field`` and a non-null terminal decision
    in ``accept``, ``reject``, or ``review``. This helper performs identity
    and decision-state normalization only; provider, evidence, reason, and
    timestamp are validated in the decision manifest. Fail closed on missing,
    duplicate, orphan, or invalid-state identities.
    """

    candidate_manifest = validate_candidate_manifest(candidates)
    expected = {
        entry["candidate_id"] for entry in candidate_manifest["candidates"]
    }
    seen: dict[str, int] = {}
    joined: list[dict[str, Any]] = []
    for index, item in enumerate(export_items):
        record = dict(item)
        external = record.get(external_id_field)
        if not isinstance(external, str) or not external:
            raise PaidfContractError(
                f"export item [{index}] is missing {external_id_field}"
            )
        if external in seen:
            raise PaidfContractError(
                f"export contains a duplicate {external_id_field}: {external}"
            )
        seen[external] = index
        if external not in expected:
            raise PaidfContractError(
                f"export item {external} is an orphan with no candidate"
            )
        if record.get("decision") not in DECISION_STATES:
            raise PaidfContractError(
                f"export item {external} must carry a non-null accept, reject, "
                f"or review decision; got {record.get('decision')!r}"
            )
        joined.append(record)
    missing = sorted(expected - set(seen))
    if missing:
        raise PaidfContractError(
            "export is missing candidates: " + ", ".join(missing)
        )
    return {
        "schema": RECONCILIATION_SCHEMA,
        "external_id_field": external_id_field,
        "totals": {
            "candidates": len(expected),
            "export_items": len(joined),
            "identity_gaps": 0,
        },
        "joined": joined,
    }


def load_frozen_source_manifest(
    manifest_ref: str,
    *,
    expected_sha256: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Load the frozen Phase 0 source manifest and fail closed on any drift.

    The manifest bytes must hash exactly to ``expected_sha256`` (the digest
    pinned in the workflow config), the manifest must record the locked
    source-contract digest, and its episodes must keep the train/held-out
    trajectory-content split disjoint. This is the run-time half of the
    source lock; the planning half lives in
    ``npa.orchestration.npa_workflow.spec._validate_frozen_source_lock``.
    """

    raw = _read_artifact_bytes(manifest_ref)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PaidfContractError(
            "frozen source manifest bytes do not match the locked digest "
            f"({expected_sha256})"
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PaidfContractError(
            f"frozen source manifest is not valid JSON: {exc}"
        ) from exc
    _exact_schema(manifest, FROZEN_SOURCE_MANIFEST_SCHEMA, "frozen source manifest")
    contract = _object(
        manifest.get("source_contract"), "frozen source manifest.source_contract"
    )
    if contract.get("sha256") != expected_contract_sha256:
        raise PaidfContractError(
            "frozen source manifest does not record the locked source-contract "
            f"digest ({expected_contract_sha256})"
        )
    episodes = _list(manifest.get("episodes"), "frozen source manifest.episodes")
    if not episodes:
        raise PaidfContractError("frozen source manifest records no episodes")
    for index, episode in enumerate(episodes):
        record = _object(episode, f"frozen source manifest.episodes[{index}]")
        _sha256(
            record.get("trajectory_sha256"),
            f"frozen source manifest episode {index}.trajectory_sha256",
        )
        if record.get("split") not in ("train", "heldout"):
            raise PaidfContractError(
                f"frozen source manifest episode {index} has an unknown split"
            )
    train = {
        episode["trajectory_sha256"]
        for episode in episodes
        if episode["split"] == "train"
    }
    heldout = {
        episode["trajectory_sha256"]
        for episode in episodes
        if episode["split"] == "heldout"
    }
    if train & heldout:
        raise PaidfContractError(
            "frozen source manifest has trajectory content overlap across splits"
        )
    return dict(manifest)


def build_base_campaign(
    campaign_id: str,
    *,
    stage_refs: Mapping[str, str],
    stage_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Build and validate the immutable base-campaign manifest.

    ``stage_refs`` and ``stage_sha256s`` are keyed by base stage
    (``source``, ``generation``, ``evaluation``). The artifact URIs are
    pinned exactly as the producing stages published them.
    """

    if set(stage_refs) != set(BASE_STAGES):
        raise PaidfContractError(
            "base campaign requires exactly the stages: " + ", ".join(BASE_STAGES)
        )
    if set(stage_sha256s) != set(BASE_STAGES):
        raise PaidfContractError(
            "base campaign digest keys must contain exactly the stages: "
            + ", ".join(BASE_STAGES)
        )
    artifacts = {
        stage: {
            "uri": stage_refs[stage],
            "sha256": stage_sha256s[stage],
            "schema": f"npa.paidf.{stage}-stage.v1",
        }
        for stage in BASE_STAGES
    }
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": campaign_id,
        "artifacts": artifacts,
        "stage_fingerprints": dict(stage_sha256s),
    }
    return validate_campaign(campaign)


def freeze_base_campaign_stage(
    campaign_id: str,
    *,
    source_manifest_ref: str,
    generation_manifest_ref: str,
    evaluation_report_ref: str,
    campaign_output_ref: str,
) -> dict[str, Any]:
    """Hash the three immutable base-stage artifacts and publish the campaign.

    Each artifact is read (local path or ``s3://`` URI), hashed byte-for-byte,
    and pinned into the campaign manifest, which is validated and written to
    ``campaign_output_ref``. Derivative partner runs consume this manifest;
    regenerating any base stage creates a new campaign instead.
    """

    stage_refs = {
        "source": source_manifest_ref,
        "generation": generation_manifest_ref,
        "evaluation": evaluation_report_ref,
    }
    stage_sha256s = {
        stage: hashlib.sha256(_read_artifact_bytes(ref)).hexdigest()
        for stage, ref in stage_refs.items()
    }
    campaign = build_base_campaign(
        campaign_id, stage_refs=stage_refs, stage_sha256s=stage_sha256s
    )
    payload = json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    _write_artifact_bytes(campaign_output_ref, payload.encode("utf-8"))
    return campaign


def build_candidates_stage(
    augment_manifest_ref: str,
    provenance_ref: str,
    evaluator_report_ref: str,
    disposition_ref: str,
    output_ref: str,
) -> dict[str, Any]:
    """Build the provider-neutral candidate manifest from real PAIDF evidence.

    One candidate per committed Cosmos 3 variant in the canonical augment
    manifest. Source episode and camera identity come from the staged input
    provenance the manifest's lineage names; output identity is the real
    published video's byte digest. Fails closed on any missing evidence.
    """

    from npa.workflows.paidf_cosmos3 import validate_committed_augment_manifest

    augment = validate_committed_augment_manifest(
        json.loads(_read_artifact_bytes(augment_manifest_ref))
    )
    provenance = json.loads(_read_artifact_bytes(provenance_ref))
    if provenance.get("schema") != "npa.paidf.cosmos3.input.v1":
        raise PaidfContractError(
            "input provenance schema must be npa.paidf.cosmos3.input.v1"
        )
    if provenance.get("episode") is None or not provenance.get("camera"):
        raise PaidfContractError(
            "candidate build requires a source dataset selection "
            "(provenance episode and camera)"
        )
    episode_index = _source_episode_index(
        provenance.get("episode"), "input provenance.episode"
    )
    camera_key = _identifier(provenance.get("camera"), "input provenance.camera")
    source_sha256 = _sha256(provenance.get("sha256"), "input provenance.sha256")
    evaluator = json.loads(_read_artifact_bytes(evaluator_report_ref))
    if evaluator.get("schema") != "npa.cosmos_evaluator.report.v1":
        raise PaidfContractError(
            "evaluator report schema must be npa.cosmos_evaluator.report.v1"
        )
    clips_by_id = _evaluator_clips_by_id(evaluator)
    disposition = json.loads(_read_artifact_bytes(disposition_ref))
    if disposition.get("schema") != "npa.data_factory.quality_disposition.v1":
        raise PaidfContractError(
            "quality disposition schema must be npa.data_factory.quality_disposition.v1"
        )
    candidates = []
    variant_ids = {str(variant["clip"]) for variant in augment}
    unexpected_evaluator_ids = sorted(set(clips_by_id) - variant_ids)
    if unexpected_evaluator_ids:
        raise PaidfContractError(
            "evaluator report names variants outside the committed augment set: "
            + ", ".join(unexpected_evaluator_ids)
        )
    for variant in augment:
        clip = str(variant["clip"])
        video_uri = str(variant["augmented_video_uri"])
        evaluation = clips_by_id.get(clip)
        if evaluation is None:
            raise PaidfContractError(
                f"evaluator report has no evidence for variant {clip}"
            )
        candidate = build_candidate(
            candidate_id=clip,
            source_episode_id=f"episode_{episode_index:06d}",
            source_episode_index=episode_index,
            camera_key=camera_key,
            variant_id=clip,
            output_sha256=hashlib.sha256(_read_artifact_bytes(video_uri)).hexdigest(),
            source_sha256=source_sha256,
            evaluation={
                "score": evaluation.get("score"),
                "passed": evaluation.get("passed"),
                "status": evaluation.get("status"),
                "skipped": evaluation.get("skipped", []),
                "quality_status": disposition.get("quality_status"),
                "threshold": disposition.get("threshold"),
            },
            variant_extras={
                "video_uri": video_uri,
                "video_bytes": variant.get("video_bytes"),
                "frame_count": variant.get("frame_count"),
                "seed": variant.get("seed"),
                "guidance": variant.get("guidance"),
                "steps": variant.get("steps"),
            },
        )
        candidates.append(candidate)
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "run_evidence": {
            "augment_manifest_ref": augment_manifest_ref,
            "quality_status": disposition.get("quality_status"),
        },
        "candidates": candidates,
    }
    validate_candidate_manifest(manifest)
    _write_artifact_bytes(
        output_ref,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def build_decisions_stage(
    candidates_manifest_ref: str,
    evaluator_report_ref: str,
    disposition_ref: str,
    output_ref: str,
    *,
    decided_at: str,
) -> dict[str, Any]:
    """Route each candidate to accept, reject, or review from real evidence.

    A rejected run disposition routes every candidate to reject. Otherwise a
    completed clip that passed the hard gates routes to accept, a completed
    clip that failed routes to reject, and anything uncertain (a missing or
    non-completed clip evaluation) routes to review, which stays excluded
    from training. The provider is the automated evaluator with an
    ``evaluation`` role, and every decision carries the caller-supplied
    ``decided_at`` timestamp; this canonical builder never reads the wall
    clock.
    """

    candidates = validate_candidate_manifest(
        json.loads(_read_artifact_bytes(candidates_manifest_ref))
    )
    evaluator = json.loads(_read_artifact_bytes(evaluator_report_ref))
    if evaluator.get("schema") != "npa.cosmos_evaluator.report.v1":
        raise PaidfContractError(
            "evaluator report schema must be npa.cosmos_evaluator.report.v1"
        )
    disposition = json.loads(_read_artifact_bytes(disposition_ref))
    if disposition.get("schema") != "npa.data_factory.quality_disposition.v1":
        raise PaidfContractError(
            "quality disposition schema must be npa.data_factory.quality_disposition.v1"
        )
    decided_at = _utc_timestamp(decided_at, "decided_at")
    clips_by_id = _evaluator_clips_by_id(evaluator)
    candidate_ids = {
        candidate["candidate_id"] for candidate in candidates["candidates"]
    }
    unexpected_evaluator_ids = sorted(set(clips_by_id) - candidate_ids)
    if unexpected_evaluator_ids:
        raise PaidfContractError(
            "evaluator report names candidates outside the candidate manifest: "
            + ", ".join(unexpected_evaluator_ids)
        )
    run_rejected = disposition.get("quality_status") == "rejected"
    decisions = []
    for candidate in candidates["candidates"]:
        candidate_id = candidate["candidate_id"]
        if run_rejected:
            state, reason = "reject", "run quality disposition rejected the wave"
        else:
            clip = clips_by_id.get(candidate_id)
            if clip is None:
                state = "review"
                reason = "no completed evaluator evidence for this variant"
            elif str(clip.get("status")) != "completed":
                state = "review"
                reason = f"evaluator clip status is {clip.get('status')!r}"
            elif clip.get("passed") is True:
                state = "accept"
                reason = "hard gates passed"
            else:
                state = "reject"
                reason = "one or more hard gates failed"
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": state,
                "evidence": {
                    "evaluator_score": (clips_by_id.get(candidate_id) or {}).get("score"),
                    "quality_status": disposition.get("quality_status"),
                },
                "reason": reason,
                "decided_at": decided_at,
            }
        )
    manifest = {
        "schema": DECISION_MANIFEST_SCHEMA,
        "provider": {"name": "cosmos-evaluator", "role": "evaluation"},
        "decisions": decisions,
    }
    validate_decision_manifest(manifest, candidates=candidates)
    _write_artifact_bytes(
        output_ref,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def accepted_replacements_stage(
    candidates_manifest_ref: str,
    reconciliation_ref: str,
    output_ref: str,
) -> dict[str, Any]:
    """Derive the accepted replacement manifest by exact-ID reconciliation.

    Joins the authoritative keep/drop reconciliation to the candidate
    manifest and fails closed unless the reconciliation covers every
    candidate exactly once with consistent states: ``keep`` holds only
    ``accept``, ``drop`` holds only ``reject`` or ``review``, and no
    unresolved ``review`` item reaches the replacements. Only accepted
    replacements are emitted, carrying every field the downstream dataset
    builder requires (episode index, camera key, video URI, and lineage).
    """

    candidates = validate_candidate_manifest(
        json.loads(_read_artifact_bytes(candidates_manifest_ref))
    )
    reconciliation = json.loads(_read_artifact_bytes(reconciliation_ref))
    if reconciliation.get("schema") != RECONCILIATION_SCHEMA:
        raise PaidfContractError(
            f"reconciliation schema must be {RECONCILIATION_SCHEMA}"
        )
    keep_states = _reconciliation_entries(
        reconciliation.get("keep"), "reconciliation.keep"
    )
    drop_states = _reconciliation_entries(
        reconciliation.get("drop"), "reconciliation.drop"
    )
    candidate_ids = {
        candidate["candidate_id"] for candidate in candidates["candidates"]
    }
    covered = set(keep_states) | set(drop_states)
    unknown = sorted(covered - candidate_ids)
    if unknown:
        raise PaidfContractError(
            "reconciliation names unknown candidates: " + ", ".join(unknown)
        )
    missing = sorted(candidate_ids - covered)
    if missing:
        raise PaidfContractError(
            "reconciliation is missing candidates: " + ", ".join(missing)
        )
    overlapping = sorted(set(keep_states) & set(drop_states))
    if overlapping:
        raise PaidfContractError(
            "candidates appear in both keep and drop: " + ", ".join(overlapping)
        )
    inconsistent_keep = sorted(
        candidate_id
        for candidate_id, state in keep_states.items()
        if state != "accept"
    )
    if inconsistent_keep:
        raise PaidfContractError(
            "reconciliation keeps candidates with a non-accept decision: "
            + ", ".join(inconsistent_keep)
        )
    inconsistent_drop = sorted(
        candidate_id
        for candidate_id, state in drop_states.items()
        if state not in ("reject", "review")
    )
    if inconsistent_drop:
        raise PaidfContractError(
            "reconciliation drops candidates with a non-terminal decision: "
            + ", ".join(inconsistent_drop)
        )
    unresolved = _list(
        reconciliation.get("unresolved_review_candidate_ids"),
        "reconciliation.unresolved_review_candidate_ids",
    )
    unresolved_ids = [
        _identifier(candidate_id, f"reconciliation.unresolved_review_candidate_ids[{index}]")
        for index, candidate_id in enumerate(unresolved)
    ]
    if len(unresolved_ids) != len(set(unresolved_ids)):
        raise PaidfContractError(
            "reconciliation.unresolved_review_candidate_ids contains duplicates"
        )
    review_drop_ids = {
        candidate_id for candidate_id, state in drop_states.items() if state == "review"
    }
    if set(unresolved_ids) != review_drop_ids:
        missing_review = sorted(review_drop_ids - set(unresolved_ids))
        extra_review = sorted(set(unresolved_ids) - review_drop_ids)
        details = []
        if missing_review:
            details.append("missing " + ", ".join(missing_review))
        if extra_review:
            details.append("unexpected " + ", ".join(extra_review))
        raise PaidfContractError(
            "reconciliation unresolved review IDs must exactly match review drops: "
            + "; ".join(details)
        )
    _decision_provider(reconciliation.get("provider"), "reconciliation.provider")
    totals = _object(reconciliation.get("totals"), "reconciliation.totals")
    if set(totals) != set(RECONCILIATION_TOTAL_FIELDS):
        raise PaidfContractError(
            "reconciliation.totals must contain exactly candidates, keep, drop, "
            "unresolved_review, and identity_gaps"
        )
    expected_totals = {
        "candidates": len(candidate_ids),
        "keep": len(keep_states),
        "drop": len(drop_states),
        "unresolved_review": len(review_drop_ids),
        "identity_gaps": 0,
    }
    for name, expected_total in expected_totals.items():
        actual_total = totals.get(name)
        if (
            not isinstance(actual_total, int)
            or isinstance(actual_total, bool)
            or actual_total != expected_total
        ):
            raise PaidfContractError(
                f"reconciliation.totals.{name} must equal {expected_total}"
            )
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates["candidates"]}
    replacements = []
    for candidate_id in sorted(keep_states):
        candidate = by_id[candidate_id]
        variant = candidate["variant"]
        video_uri = variant.get("video_uri")
        if not video_uri:
            raise PaidfContractError(
                f"accepted candidate {candidate_id} records no video_uri"
            )
        replacements.append(
            {
                "episode_index": candidate["source_episode_index"],
                "camera_key": candidate["camera_key"],
                "video_uri": video_uri,
                "lineage": {
                    "candidate_id": candidate_id,
                    "variant_id": variant["variant_id"],
                    "output_sha256": variant["output_sha256"],
                    "source_episode_id": candidate["source_episode_id"],
                    "source_sha256": variant["source_sha256"],
                },
            }
        )
    manifest = {
        "schema": ACCEPTED_REPLACEMENTS_SCHEMA,
        "totals": {
            "candidates": len(candidates["candidates"]),
            "accepted": len(replacements),
            "dropped_or_review": len(drop_states),
        },
        "replacements": replacements,
    }
    _write_artifact_bytes(
        output_ref,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def _reconciliation_entries(
    entries: Any, label: str
) -> dict[str, Any]:
    """Read reconciliation rows as ``candidate_id -> decision`` mappings.

    Fails closed on non-object rows, unsafe candidate IDs, or duplicate
    candidate IDs within one list.
    """

    rows = _list(entries, label)
    states: dict[str, Any] = {}
    for index, row in enumerate(rows):
        record = _object(row, f"{label}[{index}]")
        candidate_id = _identifier(
            record.get("candidate_id"), f"{label}[{index}].candidate_id"
        )
        if candidate_id in states:
            raise PaidfContractError(
                f"{label} contains a duplicate candidate_id: {candidate_id}"
            )
        states[candidate_id] = record.get("decision")
    return states


def _decision_provider(value: Any, label: str) -> dict[str, Any]:
    provider = _object(value, label)
    if set(provider) != set(PROVIDER_FIELDS):
        raise PaidfContractError(f"{label} must contain exactly name and role")
    _identifier(provider.get("name"), f"{label}.name")
    if provider.get("role") not in PROVIDER_ROLES:
        raise PaidfContractError(f"{label}.role must be curation or evaluation")
    return provider


def _evaluator_clips_by_id(evaluator: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_clips = evaluator.get("clips")
    clips = _list([] if raw_clips is None else raw_clips, "evaluator report.clips")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw_clip in enumerate(clips):
        clip = _object(raw_clip, f"evaluator report.clips[{index}]")
        clip_id = _identifier(
            clip.get("clip_id"), f"evaluator report.clips[{index}].clip_id"
        )
        if clip_id in by_id:
            raise PaidfContractError(
                f"evaluator report contains duplicate clip_id: {clip_id}"
            )
        by_id[clip_id] = clip
    return by_id


def _read_artifact_bytes(ref: str) -> bytes:
    """Read artifact bytes from a local path or an ``s3://`` URI."""

    if ref.startswith("s3://"):
        from npa.clients.storage import LazyStorageClient

        return bytes(LazyStorageClient().read_bytes_with_etag(ref)[0])
    return Path(ref).read_bytes()


def _write_artifact_bytes(ref: str, payload: bytes) -> None:
    """Write artifact bytes to a local path or an ``s3://`` URI."""

    if ref.startswith("s3://"):
        from npa.clients.storage import LazyStorageClient

        LazyStorageClient().put_bytes_conditional(payload, ref, if_none_match=True)
        return
    path = Path(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PaidfContractError(f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PaidfContractError(f"{label} must be a list")
    return value


def _exact_schema(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("schema") != expected:
        raise PaidfContractError(f"{label} schema must be {expected}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise PaidfContractError(f"{label} must be a safe nonempty identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PaidfContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _source_episode_index(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PaidfContractError(
            f"{label} must be a non-negative integer episode index"
        )
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise PaidfContractError(
            f"{label} must be canonical UTC seconds in YYYY-MM-DDTHH:MM:SSZ form"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PaidfContractError(
            f"{label} is not a real calendar time: {value!r}"
        ) from exc
    return value


def _s3_uri(value: Any, label: str, *, object_required: bool) -> str:
    if not isinstance(value, str):
        raise PaidfContractError(f"{label} must be an S3 URI")
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise PaidfContractError(f"{label} must include an S3 bucket and key")
    if parsed.query or parsed.fragment:
        raise PaidfContractError(f"{label} must not include a query or fragment")
    if object_required and value.endswith("/"):
        raise PaidfContractError(f"{label} must name an exact object")
    return value


def _s3_parent(uri: str) -> str:
    return uri.rsplit("/", 1)[0] + "/"


def _artifact_ref(value: Any, label: str) -> dict[str, Any]:
    ref = _object(value, label)
    _s3_uri(ref.get("uri"), f"{label}.uri", object_required=True)
    _sha256(ref.get("sha256"), f"{label}.sha256")
    schema = ref.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        raise PaidfContractError(f"{label}.schema must be nonempty")
    return ref


def _stage_list(value: Any, label: str) -> list[str]:
    stages = _list(value, label)
    for index, stage in enumerate(stages):
        _identifier(stage, f"{label}[{index}]")
    if len(stages) != len(set(stages)):
        raise PaidfContractError(f"{label} contains duplicate stages")
    return stages

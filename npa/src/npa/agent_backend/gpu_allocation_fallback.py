"""Typed, consent-gated GPU allocation fallback state machine."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA = "npa.agent.gpu-allocation-fallback/v1"
DEFAULT_THRESHOLD = 3
ON_DEMAND = "on-demand"
PREEMPTIBLE = "preemptible"

PLACEMENT_CODES = {
    "quota_exhausted": "quota_exhausted",
    "capacity_exhausted": "capacity_exhausted",
    "unschedulable_gpu": "unschedulable_gpu",
    "insufficient_gpu": "unschedulable_gpu",
    "no_compatible_product": "no_compatible_product",
    "affinity_mismatch": "no_compatible_product",
}
NON_PLACEMENT_CODES = {
    "auth",
    "rbac",
    "network",
    "image_pull",
    "checkpoint",
    "application",
    "runtime",
    "cancelled",
    "timeout",
}
_NON_PLACEMENT_RE = re.compile(
    r"\b(unauthenticated|permission denied|forbidden|rbac|imagepull|errimagepull|"
    r"pull access denied|connection refused|dns|checkpoint|traceback|cuda error|"
    r"application error|cancelled|timed? out)\b",
    re.IGNORECASE,
)
_PLACEMENT_PATTERNS = (
    (
        "quota_exhausted",
        re.compile(
            r"\b(quota (?:exceeded|exhausted|shortfall)|resource exhausted)\b", re.I
        ),
    ),
    (
        "capacity_exhausted",
        re.compile(
            r"\b(insufficient capacity|capacity (?:unavailable|exhausted)|out of capacity)\b",
            re.I,
        ),
    ),
    (
        "unschedulable_gpu",
        re.compile(r"\b(unschedulable|insufficient (?:nvidia\.com/)?gpu)\b", re.I),
    ),
    (
        "no_compatible_product",
        re.compile(
            r"\b(no matching (?:compatible )?(?:gpu )?product|affinity mismatch|didn't match.*affinity)\b",
            re.I,
        ),
    ),
)
INVARIANT_KEYS = (
    "gpu_family",
    "gpu_product",
    "gpu_count",
    "image",
    "image_digest",
    "sm",
    "rt_cores_required",
    "backend",
    "model",
    "workload_tier",
    "execution_mode",
    "boot_disk_count",
    "boot_disk_size_bytes",
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def logical_allocation_ref(value: str) -> str:
    """Return a redacted stable reference for a caller's logical allocation key."""

    return _digest(str(value))


def classify_failure(code: str = "", message: str = "") -> dict[str, Any]:
    """Classify only concrete placement evidence as qualifying."""

    normalized = str(code or "").strip().lower().replace("-", "_")
    if normalized in NON_PLACEMENT_CODES or _NON_PLACEMENT_RE.search(
        str(message or "")
    ):
        return {"category": normalized or "non_placement", "qualifying": False}
    if normalized in PLACEMENT_CODES:
        return {"category": PLACEMENT_CODES[normalized], "qualifying": True}
    for category, pattern in _PLACEMENT_PATTERNS:
        if pattern.search(str(message or "")):
            return {"category": category, "qualifying": True}
    return {"category": "other", "qualifying": False}


def normalized_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compatibility/execution/disk invariants without identifiers."""

    return {key: request.get(key) for key in INVARIANT_KEYS}


def _invariants_complete(request: Mapping[str, Any]) -> bool:
    normalized = normalized_request(request)
    if any(normalized[key] is None or normalized[key] == "" for key in INVARIANT_KEYS):
        return False
    try:
        return (
            int(normalized["gpu_count"]) > 0
            and int(normalized["boot_disk_count"]) > 0
            and int(normalized["boot_disk_size_bytes"]) > 0
        )
    except (TypeError, ValueError):
        return False


def candidate_is_compatible(
    request: Mapping[str, Any], candidate: Mapping[str, Any] | None
) -> bool:
    """Allow a candidate to differ only in capacity pool."""

    if not isinstance(candidate, Mapping):
        return False
    expected = normalized_request(request)
    actual = normalized_request(candidate)
    return (
        _invariants_complete(request)
        and _invariants_complete(candidate)
        and expected == actual
        and str(candidate.get("pool") or "") == PREEMPTIBLE
    )


def new_state(logical_allocation: str, request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": SCHEMA,
        "logical_allocation_ref": logical_allocation_ref(logical_allocation),
        "request_digest": _digest(normalized_request(request)),
        "invariants": normalized_request(request),
        "selected_pool": ON_DEMAND,
        "qualifying_attempts": 0,
        "status": "tracking",
        "pending_action_digest": "",
        "prompted_evidence_digest": "",
        "declined_evidence_digest": "",
        "consent": "not-requested",
        "provenance": [],
    }


def _public_evidence(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    source = str((evidence or {}).get("source") or "unknown")
    if source not in {
        "provider-preflight",
        "scheduler",
        "terraform",
        "skypilot",
        "unknown",
    }:
        source = "unknown"
    return {
        "source": source,
        "on_demand_impossible": bool((evidence or {}).get("on_demand_impossible")),
        "preemptible_available": bool((evidence or {}).get("preemptible_available")),
        "evidence_version": (
            _digest(str((evidence or {}).get("fingerprint")))
            if (evidence or {}).get("fingerprint")
            else ""
        ),
    }


def _provenance(
    state: dict[str, Any], *, classification: str, evidence_digest: str, outcome: str
) -> None:
    rows = state.setdefault("provenance", [])
    rows.append(
        {
            "attempt": int(state.get("qualifying_attempts") or 0),
            "classification": classification,
            "evidence_digest": evidence_digest,
            "selected_pool": state.get("selected_pool", ON_DEMAND),
            "consent_outcome": outcome,
        }
    )
    del rows[:-32]


def record_attempt(
    state: Mapping[str, Any] | None,
    *,
    logical_allocation: str,
    request: Mapping[str, Any],
    failure_code: str = "",
    failure_message: str = "",
    evidence: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    success: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance one logical allocation and return a zero-token prompt decision."""

    current = dict(state or {})
    request_digest = _digest(normalized_request(request))
    logical_ref = logical_allocation_ref(logical_allocation)
    if (
        current.get("apiVersion") != SCHEMA
        or current.get("logical_allocation_ref") != logical_ref
        or current.get("request_digest") != request_digest
    ):
        current = new_state(logical_allocation, request)
    public_evidence = _public_evidence(evidence)
    evidence_digest = _digest(public_evidence)

    if (
        current.get("selected_pool") == PREEMPTIBLE
        and current.get("consent") == "accepted"
    ):
        _provenance(
            current,
            classification="already_selected",
            evidence_digest=evidence_digest,
            outcome="ignored",
        )
        return current, {"prompt": False, "reason": "preemptible_already_selected"}

    if str(request.get("pool") or ON_DEMAND) != ON_DEMAND:
        _provenance(
            current,
            classification="not_on_demand",
            evidence_digest=evidence_digest,
            outcome="ignored",
        )
        return current, {"prompt": False, "reason": "not_on_demand"}

    if success:
        current.update(
            {
                "qualifying_attempts": 0,
                "status": "succeeded",
                "pending_action_digest": "",
                "prompted_evidence_digest": "",
                "declined_evidence_digest": "",
                "consent": "not-requested",
            }
        )
        _provenance(
            current,
            classification="success",
            evidence_digest=evidence_digest,
            outcome="success",
        )
        return current, {"prompt": False, "reason": "allocation_succeeded"}

    classification = classify_failure(failure_code, failure_message)
    if classification["qualifying"]:
        current["qualifying_attempts"] = (
            int(current.get("qualifying_attempts") or 0) + 1
        )
    _provenance(
        current,
        classification=str(classification["category"]),
        evidence_digest=evidence_digest,
        outcome="recorded",
    )
    compatible = candidate_is_compatible(request, candidate)
    immediate = bool(
        public_evidence["source"] == "provider-preflight"
        and public_evidence["on_demand_impossible"]
        and public_evidence["preemptible_available"]
        and compatible
    )
    threshold_met = bool(
        classification["qualifying"]
        and int(current["qualifying_attempts"]) >= max(1, int(threshold))
        and compatible
    )
    already_prompted = current.get("prompted_evidence_digest") == evidence_digest
    declined_same = current.get("declined_evidence_digest") == evidence_digest
    should_prompt = (
        (immediate or threshold_met) and not already_prompted and not declined_same
    )
    if not should_prompt:
        return current, {
            "prompt": False,
            "reason": "suppressed"
            if already_prompted or declined_same
            else "threshold_not_met",
            "classification": classification,
        }

    proposed = {
        "action": "switch_gpu_capacity_pool",
        "logical_allocation_ref": logical_ref,
        "from": ON_DEMAND,
        "to": PREEMPTIBLE,
        "invariants": normalized_request(request),
        "evidence_digest": evidence_digest,
    }
    action_digest = _digest(proposed)
    current.update(
        {
            "status": "awaiting-consent",
            "pending_action_digest": action_digest,
            "prompted_evidence_digest": evidence_digest,
            "consent": "pending",
        }
    )
    return current, {
        "prompt": True,
        "reason": "deterministic_preflight"
        if immediate
        else "failed_attempt_threshold",
        "classification": classification,
        "message": "On-demand GPU placement cannot proceed. Switch this identical allocation to preemptible capacity?",
        "proposed_action": {**proposed, "digest": action_digest},
    }


def record_consent(
    state: Mapping[str, Any],
    *,
    accepted: bool,
    confirmed_action_digest: str = "",
) -> dict[str, Any]:
    """Record consent; acceptance requires the exact pending action digest."""

    current = dict(state)
    pending = str(current.get("pending_action_digest") or "")
    if not pending:
        raise ValueError("no GPU allocation fallback is awaiting consent")
    if accepted and confirmed_action_digest != pending:
        raise ValueError(
            "confirmation is not bound to the pending GPU allocation action"
        )
    evidence_digest = str(current.get("prompted_evidence_digest") or "")
    if accepted:
        current.update(
            {
                "selected_pool": PREEMPTIBLE,
                "status": "consented",
                "consent": "accepted",
                "pending_action_digest": "",
            }
        )
        outcome = "accepted"
    else:
        current.update(
            {
                "selected_pool": ON_DEMAND,
                "status": "declined",
                "consent": "declined",
                "declined_evidence_digest": evidence_digest,
                "pending_action_digest": "",
            }
        )
        outcome = "declined"
    _provenance(
        current,
        classification="consent",
        evidence_digest=evidence_digest,
        outcome=outcome,
    )
    return current


def public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the already-redacted durable record exposed by the API."""

    return {
        key: state.get(key)
        for key in (
            "apiVersion",
            "logical_allocation_ref",
            "request_digest",
            "invariants",
            "selected_pool",
            "qualifying_attempts",
            "status",
            "consent",
            "provenance",
        )
    }

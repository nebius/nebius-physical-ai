"""Human approval planning for exact Hugging Face and NVIDIA NGC artifacts.

This module deliberately does not perform legal acceptance.  It discovers the
catalog requirements, probes upstream access with the operator's credentials,
and returns official pages plus a safe command that can resume the interrupted
operation.  Browser opening belongs to an explicitly consenting CLI/UI client.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from npa.workbench.model_access import (
    HF,
    NGC,
    GatedAsset,
    assets_for,
    usable_hf_payload_probe,
)

SCHEMA_VERSION = "npa.workbench.access-approval.v1"
DEFAULT_STATE_PATH = Path.home() / ".npa" / "access-approvals.json"


class AccessStatus(str, Enum):
    READY = "Ready"
    PENDING = "Pending"
    DENIED = "Denied"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class AccessEvidence:
    requirement: GatedAsset
    status: AccessStatus
    reason: str
    checked_at: str
    credential_fingerprint: str
    cached: bool = False

    def as_dict(self) -> dict[str, object]:
        item = self.requirement
        return {
            "provider": item.provider,
            "artifact": item.repo,
            "artifact_type": item.repo_type,
            "revision": item.revision,
            "gated": item.gated,
            "capabilities": list(item.capabilities),
            "official_url": item.official_url,
            "terms_revision": item.terms_revision,
            "probe_path": item.probe_path,
            "status": self.status.value,
            "reason": self.reason,
            "checked_at": self.checked_at,
            "cached": self.cached,
        }


def exact_requirements(
    capabilities: Iterable[str] | None = None, *, gated_only: bool = True
) -> tuple[GatedAsset, ...]:
    """Return a stable, deduplicated exact requirement closure."""

    by_identity: dict[tuple[str, str, str, str], GatedAsset] = {}
    for item in assets_for(capabilities):
        if item.provider not in {HF, NGC}:
            continue
        if gated_only and not item.gated:
            continue
        # Token Factory is an independent optional hosted product and is never
        # part of HF/NGC approval preparation, even for a checkpoint someone
        # could separately choose to self-host.
        if set(item.capabilities) <= {"token_factory"}:
            continue
        key = (item.provider, item.repo, item.repo_type, item.revision)
        existing = by_identity.get(key)
        if existing is None:
            by_identity[key] = item
            continue
        by_identity[key] = GatedAsset(
            repo=item.repo,
            provider=item.provider,
            capabilities=tuple(
                sorted(set(existing.capabilities).union(item.capabilities))
            ),
            gated=existing.gated or item.gated,
            note=existing.note or item.note,
            repo_type=item.repo_type,
            revision=item.revision,
            official_url=item.official_url,
            terms_revision=item.terms_revision,
            probe_path=item.probe_path,
        )
    return tuple(
        by_identity[key]
        for key in sorted(by_identity, key=lambda value: (value[0], value[1], value[3]))
    )


def requirements_for_tool_refs(tool_refs: Iterable[str]) -> tuple[GatedAsset, ...]:
    """Resolve access requirements from the selected real toolRef metadata."""

    from npa.orchestration.npa_workflow.catalog import validate_tool_ref

    capabilities: list[str] = []
    for tool_ref in dict.fromkeys(str(item) for item in tool_refs if str(item)):
        capabilities.extend(validate_tool_ref(tool_ref).access_capabilities)
    if not capabilities:
        return ()
    return exact_requirements(capabilities)


def catalog_digest(requirements: Iterable[GatedAsset]) -> str:
    payload = [
        {
            "provider": item.provider,
            "artifact": item.repo,
            "artifact_type": item.repo_type,
            "revision": item.revision,
            "terms_revision": item.terms_revision,
            "probe_path": item.probe_path,
        }
        for item in requirements
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def credential_fingerprint(provider: str, secret: str) -> str:
    if not secret:
        return "missing"
    return hashlib.sha256(f"{provider}\0{secret}".encode("utf-8")).hexdigest()


def _cache_key(item: GatedAsset, fingerprint: str) -> str:
    payload = "\0".join(
        (
            item.provider,
            item.repo,
            item.repo_type,
            item.revision,
            item.terms_revision,
            item.probe_path,
            fingerprint,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"schema_version": SCHEMA_VERSION, "evidence": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return {"schema_version": SCHEMA_VERSION, "evidence": {}}
    if not isinstance(payload.get("evidence"), dict):
        payload["evidence"] = {}
    return payload


def save_state(payload: Mapping[str, Any], path: Path = DEFAULT_STATE_PATH) -> None:
    """Atomically persist non-secret evidence in an owner-only file."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _hf_evidence(
    item: GatedAsset,
    token: str,
    validator: Callable[..., Any] | None,
) -> tuple[AccessStatus, str]:
    if not token and item.gated:
        return AccessStatus.PENDING, "missing_credentials"
    if validator is None:
        return AccessStatus.UNAVAILABLE, "probe_unavailable"
    if not usable_hf_payload_probe(item):
        return AccessStatus.UNAVAILABLE, "exact_payload_probe_missing"
    try:
        result = validator(
            token,
            item.repo,
            item.repo_type,
            item.revision,
            item.probe_path,
        )
    except Exception:  # noqa: BLE001 - provider diagnostics may echo secrets
        return AccessStatus.UNAVAILABLE, "provider_unavailable"
    if bool(getattr(result, "ok", False)):
        return AccessStatus.READY, "exact_artifact_access_verified"
    status_code = getattr(result, "status_code", None)
    if status_code in {401, 403} and item.gated:
        return AccessStatus.PENDING, "manual_approval_required_or_pending"
    if status_code in {401, 403}:
        return AccessStatus.DENIED, "credential_or_entitlement_denied"
    return AccessStatus.UNAVAILABLE, "provider_unavailable"


def _ngc_evidence(
    item: GatedAsset,
    key: str,
    validator: Callable[..., str] | None,
) -> tuple[AccessStatus, str]:
    if not key:
        return AccessStatus.PENDING, "missing_credentials"
    if validator is None:
        return AccessStatus.UNAVAILABLE, "probe_unavailable"
    try:
        try:
            outcome = str(validator(key, image=item.repo) or "unreachable")
        except TypeError:
            outcome = str(validator(key) or "unreachable")
    except Exception:  # noqa: BLE001 - provider diagnostics may echo secrets
        return AccessStatus.UNAVAILABLE, "provider_unavailable"
    if outcome == "reachable":
        return AccessStatus.READY, "exact_artifact_access_verified"
    if outcome in {"entitlement-required", "tags-401", "tags-403"}:
        return AccessStatus.DENIED, "artifact_entitlement_denied"
    if outcome in {"auth-no-token", "auth-401", "auth-403"}:
        return AccessStatus.DENIED, "credential_denied"
    return AccessStatus.UNAVAILABLE, "provider_unavailable"


def probe_requirements(
    requirements: Iterable[GatedAsset],
    *,
    hf_token: str,
    ngc_key: str,
    hf_validator: Callable[..., Any] | None,
    ngc_validator: Callable[..., str] | None,
    state_path: Path = DEFAULT_STATE_PATH,
    force: bool = False,
    now: Callable[[], datetime] | None = None,
) -> list[AccessEvidence]:
    """Probe exact artifact access and persist only sanitized evidence.

    A cached Ready result is reused only while credential fingerprint, exact
    artifact revision, and governing terms revision are unchanged.  Non-ready
    evidence is always re-probed so a browser approval can resume immediately.
    """

    clock = now or (lambda: datetime.now(timezone.utc))
    items = tuple(requirements)
    state = load_state(state_path)
    stored = dict(state.get("evidence") or {})
    evidence: list[AccessEvidence] = []
    for item in items:
        secret = hf_token if item.provider == HF else ngc_key
        fingerprint = credential_fingerprint(item.provider, secret)
        key = _cache_key(item, fingerprint)
        cached = stored.get(key)
        if (
            not force
            and isinstance(cached, dict)
            and cached.get("status") == AccessStatus.READY.value
        ):
            evidence.append(
                AccessEvidence(
                    requirement=item,
                    status=AccessStatus.READY,
                    reason="unchanged_verified_access",
                    checked_at=str(cached.get("checked_at") or ""),
                    credential_fingerprint=fingerprint,
                    cached=True,
                )
            )
            continue
        if item.provider == HF:
            status, reason = _hf_evidence(item, hf_token, hf_validator)
        else:
            status, reason = _ngc_evidence(item, ngc_key, ngc_validator)
        checked_at = clock().isoformat()
        item_evidence = AccessEvidence(
            requirement=item,
            status=status,
            reason=reason,
            checked_at=checked_at,
            credential_fingerprint=fingerprint,
        )
        evidence.append(item_evidence)
        stored[key] = {
            "provider": item.provider,
            "artifact": item.repo,
            "artifact_type": item.repo_type,
            "revision": item.revision,
            "terms_revision": item.terms_revision,
            "probe_path": item.probe_path,
            "credential_fingerprint": fingerprint,
            "status": status.value,
            "reason": reason,
            "checked_at": checked_at,
        }
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "catalog_digest": catalog_digest(items),
            "evidence": stored,
        }
    )
    save_state(state, state_path)
    return evidence


def approval_plan(
    evidence: Iterable[AccessEvidence], *, resume_command: str
) -> dict[str, object]:
    rows = list(evidence)
    missing = [row for row in rows if row.status != AccessStatus.READY]
    providers: dict[str, list[dict[str, object]]] = {HF: [], NGC: []}
    for row in missing:
        providers[row.requirement.provider].append(row.as_dict())
    providers = {key: value for key, value in providers.items() if value}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if not missing else "blocked",
        "counts": {
            "hf": len(providers.get(HF, [])),
            "ngc": len(providers.get(NGC, [])),
        },
        "providers": providers,
        "official_urls": list(
            dict.fromkeys(
                row.requirement.official_url
                for row in missing
                if row.requirement.official_url
            )
        ),
        "resume_command": resume_command,
        "legal_assent_performed": False,
    }


def blocked(plan: Mapping[str, object]) -> bool:
    return str(plan.get("status") or "") == "blocked"


def safe_resume_command(argv: Iterable[str]) -> str:
    """Return an exact copyable CLI retry without secret option values."""

    values = list(argv)
    if not values:
        return "npa workbench health access --prepare"
    if values[0].endswith("python") or values[0].endswith("python3"):
        try:
            module_index = values.index("-m")
        except ValueError:
            module_index = -1
        if module_index >= 0 and values[module_index + 1 : module_index + 2] == ["npa"]:
            values = ["npa", *values[module_index + 2 :]]
    elif Path(values[0]).name == "npa":
        values[0] = "npa"
    secret_flags = {
        "--hf-token",
        "--ngc-key",
        "--registry-password",
        "--auth-password",
        "--token",
    }
    cleaned: list[str] = []
    skip_next = False
    for value in values:
        if skip_next:
            skip_next = False
            continue
        if value in secret_flags:
            skip_next = True
            continue
        if any(value.startswith(flag + "=") for flag in secret_flags):
            continue
        cleaned.append(value)
    return shlex.join(cleaned)


__all__ = [
    "AccessEvidence",
    "AccessStatus",
    "DEFAULT_STATE_PATH",
    "SCHEMA_VERSION",
    "approval_plan",
    "blocked",
    "catalog_digest",
    "credential_fingerprint",
    "exact_requirements",
    "load_state",
    "probe_requirements",
    "requirements_for_tool_refs",
    "save_state",
    "safe_resume_command",
]

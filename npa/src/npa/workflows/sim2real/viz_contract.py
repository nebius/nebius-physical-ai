"""Run-level policy and artifact provenance supplied to Stage 14."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from npa.workflows.sim2real.capture import runtime_parameter_metadata


_MISSING = object()
_SHA256 = re.compile(r"[0-9a-f]{64}")
IdentityEvidence = tuple[tuple[str, object], ...]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _checkpoint_uri(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        return None
    parsed = urlparse(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or any(char in parsed.netloc for char in "@:%")
        or not parsed.path.lstrip("/")
        or not parsed.path.endswith(".pt")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    return value


def _sha256(value: object) -> str | None:
    if not isinstance(value, str) or value != value.strip():
        return None
    digest = value.lower()
    return digest if _SHA256.fullmatch(digest) else None


def _positive_size(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _strict_bool(value: object) -> bool | None:
    return value if value is True or value is False else None


def _reconcile_identity_field(
    *,
    label: str,
    evidence: IdentityEvidence,
    normalize: Any,
) -> tuple[object | None, list[str]]:
    normalized: list[tuple[str, object]] = []
    errors: list[str] = []
    for source, raw_value in evidence:
        value = normalize(raw_value)
        if value is None:
            errors.append(f"{source} {label} is malformed")
        else:
            normalized.append((source, value))
    if len({value for _source, value in normalized}) > 1:
        errors.append(f"{label} sources disagree")
    return (normalized[0][1] if normalized else None), errors


def _field_evidence(
    payload: dict[str, Any],
    key: str,
    source: str,
) -> IdentityEvidence:
    return ((source, payload[key]),) if key in payload else ()


def _identity_sources(
    report: dict[str, Any],
    provenance: dict[str, Any],
    *,
    provenance_key: str,
    report_keys: tuple[str, ...],
    fallback: object,
    fallback_label: str,
    additional: IdentityEvidence,
) -> IdentityEvidence:
    sources = _field_evidence(
        provenance,
        provenance_key,
        f"policy_inference_provenance.{provenance_key}",
    )
    for key in report_keys:
        sources += _field_evidence(report, key, key)
    if fallback is not _MISSING:
        sources += ((fallback_label, fallback),)
    return sources + additional


def _checkpoint_identity(
    report: dict[str, Any],
    provenance: dict[str, Any],
    *,
    uri_fallback: object,
    digest_fallback: object,
    size_fallback: object,
    uri_evidence: IdentityEvidence,
    digest_evidence: IdentityEvidence,
    size_evidence: IdentityEvidence,
    generator_evidence: IdentityEvidence,
) -> dict[str, Any]:
    errors = [
        f"policy_inference_provenance.{key} is missing"
        for key in ("checkpoint_uri", "checkpoint_sha256", "checkpoint_size_bytes")
        if key not in provenance
    ]
    uri, field_errors = _reconcile_identity_field(
        label="checkpoint URI",
        evidence=_identity_sources(
            report,
            provenance,
            provenance_key="checkpoint_uri",
            report_keys=("policy_checkpoint", "policy_checkpoint_uri"),
            fallback=uri_fallback,
            fallback_label="selected checkpoint URI",
            additional=uri_evidence,
        ),
        normalize=_checkpoint_uri,
    )
    errors.extend(field_errors)
    digest, field_errors = _reconcile_identity_field(
        label="checkpoint SHA-256",
        evidence=_identity_sources(
            report,
            provenance,
            provenance_key="checkpoint_sha256",
            report_keys=("policy_checkpoint_sha256",),
            fallback=digest_fallback,
            fallback_label="selected checkpoint SHA-256",
            additional=digest_evidence,
        ),
        normalize=_sha256,
    )
    errors.extend(field_errors)
    size, field_errors = _reconcile_identity_field(
        label="checkpoint size",
        evidence=_identity_sources(
            report,
            provenance,
            provenance_key="checkpoint_size_bytes",
            report_keys=("policy_checkpoint_size_bytes",),
            fallback=size_fallback,
            fallback_label="selected checkpoint size",
            additional=size_evidence,
        ),
        normalize=_positive_size,
    )
    errors.extend(field_errors)

    generator_sources = _field_evidence(
        provenance,
        "generator_policy_sha256",
        "policy_inference_provenance.generator_policy_sha256",
    )
    generator_sources += _field_evidence(
        report, "generator_policy_sha256", "generator_policy_sha256"
    )
    generator_sources += _field_evidence(
        report, "policy_generator_sha256", "policy_generator_sha256"
    )
    generator_sources += generator_evidence
    generator: object | None = None
    if generator_sources:
        generator, field_errors = _reconcile_identity_field(
            label="generator checkpoint SHA-256",
            evidence=generator_sources,
            normalize=_sha256,
        )
        errors.extend(field_errors)
        if generator is not None and digest is not None and generator != digest:
            errors.append(
                "generator checkpoint SHA-256 does not match inference checkpoint"
            )
    return {
        "heldout_policy_checkpoint": str(uri or ""),
        "heldout_policy_checkpoint_sha256": str(digest or ""),
        "heldout_policy_generator_sha256": str(generator or ""),
        "heldout_policy_checkpoint_size_bytes": int(size or 0),
        "heldout_policy_identity_verified": not errors,
        "heldout_policy_identity_errors": errors,
    }


def _learned_policy_declaration(
    provenance: dict[str, Any],
    *,
    loaded_for_inference: bool,
    identity_verified: bool,
) -> dict[str, Any]:
    stock_or_scripted = _strict_bool(provenance.get("stock_or_scripted_policy"))
    actor_is_learned = _strict_bool(provenance.get("actor_is_learned"))
    scripted_controller = _strict_bool(provenance.get("scripted_post_actor_controller"))
    composition = _text(provenance.get("policy_composition"))
    controller = provenance.get("post_actor_controller")
    controller_declared = "post_actor_controller" in provenance
    learned_actor_only = bool(
        loaded_for_inference
        and identity_verified
        and stock_or_scripted is False
        and actor_is_learned is True
        and scripted_controller is False
        and composition == "learned_actor_only"
        and controller_declared
        and controller is None
    )
    return {
        "heldout_policy_stock_or_scripted_policy": stock_or_scripted,
        "heldout_policy_actor_is_learned": actor_is_learned,
        "heldout_policy_scripted_post_actor_controller": scripted_controller,
        "heldout_policy_composition": composition,
        "heldout_policy_post_actor_controller": controller,
        "heldout_policy_post_actor_controller_declared": controller_declared,
        "heldout_policy_learned_actor_only": learned_actor_only,
    }


def _project_heldout_policy_metadata(
    heldout_report: dict[str, Any] | None,
    *,
    checkpoint_fallback: object = _MISSING,
    checkpoint_sha256_fallback: object = _MISSING,
    checkpoint_size_fallback: object = _MISSING,
    checkpoint_uri_evidence: IdentityEvidence = (),
    checkpoint_sha256_evidence: IdentityEvidence = (),
    checkpoint_size_evidence: IdentityEvidence = (),
    generator_sha256_evidence: IdentityEvidence = (),
) -> dict[str, Any]:
    report = heldout_report if isinstance(heldout_report, dict) else {}
    raw_provenance = report.get("policy_inference_provenance")
    provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
    errors: list[str] = []
    if not isinstance(raw_provenance, dict):
        errors.append("policy_inference_provenance is missing or is not an object")

    identity = _checkpoint_identity(
        report,
        provenance,
        uri_fallback=checkpoint_fallback,
        digest_fallback=checkpoint_sha256_fallback,
        size_fallback=checkpoint_size_fallback,
        uri_evidence=checkpoint_uri_evidence,
        digest_evidence=checkpoint_sha256_evidence,
        size_evidence=checkpoint_size_evidence,
        generator_evidence=generator_sha256_evidence,
    )
    errors.extend(identity["heldout_policy_identity_errors"])
    identity["heldout_policy_identity_errors"] = errors
    loaded_for_inference = provenance.get("loaded_for_inference") is True
    return {
        **identity,
        "heldout_policy_loaded_for_inference": loaded_for_inference,
        **_learned_policy_declaration(
            provenance,
            loaded_for_inference=loaded_for_inference,
            identity_verified=identity["heldout_policy_identity_verified"],
        ),
    }


def heldout_policy_metadata(
    heldout_report: dict[str, Any] | None,
    *,
    checkpoint_fallback: object = _MISSING,
    checkpoint_sha256_fallback: object = _MISSING,
    checkpoint_size_fallback: object = _MISSING,
    checkpoint_uri_evidence: IdentityEvidence = (),
    checkpoint_sha256_evidence: IdentityEvidence = (),
    checkpoint_size_evidence: IdentityEvidence = (),
    generator_sha256_evidence: IdentityEvidence = (),
) -> dict[str, Any]:
    """Project held-out evidence only after sealing exact policy identity.

    Args:
        heldout_report: Stage 10 held-out evaluation record.
        checkpoint_fallback: Validation-selected checkpoint URI.
        checkpoint_sha256_fallback: Validation-selected checkpoint digest.
        checkpoint_size_fallback: Validation-selected checkpoint size.
        checkpoint_uri_evidence: Additional labeled checkpoint URI sources.
        checkpoint_sha256_evidence: Additional labeled checkpoint digest sources.
        checkpoint_size_evidence: Additional labeled checkpoint size sources.
        generator_sha256_evidence: Additional labeled generator digest sources.

    Returns:
        Strict run metadata plus explicit identity errors and policy semantics.

    Raises:
        None.
    """

    return _project_heldout_policy_metadata(
        heldout_report,
        checkpoint_fallback=checkpoint_fallback,
        checkpoint_sha256_fallback=checkpoint_sha256_fallback,
        checkpoint_size_fallback=checkpoint_size_fallback,
        checkpoint_uri_evidence=checkpoint_uri_evidence,
        checkpoint_sha256_evidence=checkpoint_sha256_evidence,
        checkpoint_size_evidence=checkpoint_size_evidence,
        generator_sha256_evidence=generator_sha256_evidence,
    )


def _candidate_policy_metadata(
    candidate: dict[str, Any],
    *,
    checkpoint: str,
) -> dict[str, Any]:
    return {
        "policy_checkpoint": checkpoint,
        "policy_checkpoint_identity": (
            candidate.get("policy_checkpoint_identity") or candidate.get("identity", "")
        ),
        "policy_checkpoint_sha256": (
            candidate.get("policy_checkpoint_sha256") or candidate.get("sha256", "")
        ),
        "policy_checkpoint_size_bytes": candidate.get(
            "policy_checkpoint_size_bytes",
            candidate.get("size_bytes", ""),
        ),
        "policy_download_command": (
            candidate.get("policy_download_command")
            or candidate.get("authenticated_download_command", "")
        ),
        "policy_ui_action": (
            candidate.get("policy_ui_action") or candidate.get("ui_action", "")
        ),
        "policy_deployable": candidate.get("deployable_policy") is True,
    }


def _artifact_metadata(
    config: Any,
    artifact_root: str,
    *,
    progress: bool,
) -> dict[str, Any]:
    report_name = "sim2real-progress.rrd" if progress else "sim2real.rrd"
    return {
        "run_id": config.run_id,
        "artifact_root": artifact_root + "/",
        "rrd_s3_uri": f"{artifact_root}/reports/{report_name}",
        "candidate_s3_uri": f"{artifact_root}/checkpoints/candidate/candidate.json",
    }


def _viewer_metadata(config: Any) -> dict[str, Any]:
    viewer_command = (
        "npa workbench sim2real rerun serve "
        f"--run-id {config.run_id} --s3-bucket {config.s3_bucket} "
        f"--s3-prefix {config.s3_prefix}"
    )
    return {
        "runtime_parameters": runtime_parameter_metadata(),
        "orchestrator_job_name": config.run_id,
        "orchestrator_node_product": config.k8s_gpu_product,
        "viewer_command": viewer_command,
    }


def visualization_run_metadata(
    *,
    config: Any,
    artifact_root: str,
    policy_checkpoint: str = "",
    candidate: dict[str, Any] | None = None,
    heldout_report: dict[str, Any] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Build a stable Rerun/MCAP checkpoint provenance contract.
    Args:
        config: Sim2Real run configuration.
        artifact_root: Run-scoped artifact URI.
        policy_checkpoint: Validation-selected checkpoint URI.
        candidate: Optional candidate access metadata.
        heldout_report: Optional Stage 10 held-out report.
        progress: Whether this is an in-progress recording.
    Returns:
        Metadata consumed by the visualization emitters.
    Raises:
        None.
    """

    candidate = candidate if isinstance(candidate, dict) else {}
    heldout = heldout_report if isinstance(heldout_report, dict) else {}
    candidate_uri_evidence = _field_evidence(
        candidate,
        "policy_checkpoint_uri",
        "candidate.policy_checkpoint_uri",
    ) + _field_evidence(
        candidate,
        "checkpoint_uri",
        "candidate.checkpoint_uri",
    )
    candidate_digest_evidence = _field_evidence(
        candidate,
        "policy_checkpoint_sha256",
        "candidate.policy_checkpoint_sha256",
    ) + _field_evidence(candidate, "sha256", "candidate.sha256")
    candidate_size_evidence = _field_evidence(
        candidate,
        "policy_checkpoint_size_bytes",
        "candidate.policy_checkpoint_size_bytes",
    ) + _field_evidence(candidate, "size_bytes", "candidate.size_bytes")
    candidate_generator_evidence = _field_evidence(
        candidate,
        "generator_policy_sha256",
        "candidate.generator_policy_sha256",
    ) + _field_evidence(
        candidate,
        "policy_generator_sha256",
        "candidate.policy_generator_sha256",
    )
    heldout_metadata = heldout_policy_metadata(
        heldout,
        checkpoint_fallback=(policy_checkpoint if policy_checkpoint else _MISSING),
        checkpoint_uri_evidence=candidate_uri_evidence,
        checkpoint_sha256_evidence=candidate_digest_evidence,
        checkpoint_size_evidence=candidate_size_evidence,
        generator_sha256_evidence=candidate_generator_evidence,
    )
    return {
        **_artifact_metadata(config, artifact_root, progress=progress),
        **_candidate_policy_metadata(candidate, checkpoint=policy_checkpoint),
        **heldout_metadata,
        **_viewer_metadata(config),
    }

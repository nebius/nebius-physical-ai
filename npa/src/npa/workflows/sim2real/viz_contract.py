"""Run-level policy and artifact provenance supplied to Stage 14."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.capture import runtime_parameter_metadata


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sha256(value: object) -> str:
    digest = _text(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return ""
    return digest


def _positive_size(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return 0
    return value


def _strict_bool(value: object) -> bool | None:
    return value if value is True or value is False else None


def _provided(value: object, *, zero_is_absent: bool = False) -> bool:
    if value is None or value == "":
        return False
    return not (zero_is_absent and type(value) is int and value == 0)


def _reconcile_identity_field(
    *,
    producer_values: tuple[object, ...],
    corroborating_values: tuple[object, ...],
    normalize: Any,
    missing_error: str,
    malformed_error: str,
    disagreement_error: str,
    zero_is_absent: bool = False,
) -> tuple[Any, bool, list[str]]:
    raw_values = [
        value
        for value in (*producer_values, *corroborating_values)
        if _provided(value, zero_is_absent=zero_is_absent)
    ]
    normalized_values = [normalize(value) for value in raw_values if normalize(value)]
    producer_value = next(
        (normalize(value) for value in producer_values if normalize(value)),
        "",
    )
    errors = []
    if not producer_value:
        errors.append(missing_error)
    if len(normalized_values) != len(raw_values):
        errors.append(malformed_error)
    if len(set(normalized_values)) > 1:
        errors.append(disagreement_error)
    verified = bool(
        producer_value
        and len(normalized_values) == len(raw_values)
        and len(set(normalized_values)) == 1
    )
    fallback = normalized_values[0] if normalized_values else ""
    return producer_value or fallback, verified, errors


def _checkpoint_uri_evidence(
    report: dict[str, Any],
    provenance: dict[str, Any],
    fallback: str,
) -> tuple[str, bool, list[str]]:
    return _reconcile_identity_field(
        producer_values=(provenance.get("checkpoint_uri"),),
        corroborating_values=(
            fallback,
            report.get("policy_checkpoint"),
            report.get("policy_checkpoint_uri"),
        ),
        normalize=_text,
        missing_error="inference checkpoint URI is missing or malformed",
        malformed_error="checkpoint URI evidence is malformed",
        disagreement_error="checkpoint URI evidence does not agree",
    )


def _checkpoint_digest_evidence(
    report: dict[str, Any],
    provenance: dict[str, Any],
    fallback: str,
) -> tuple[str, str, bool, list[str]]:
    generator = provenance.get("generator_policy_sha256") or report.get(
        "generator_policy_sha256"
    )
    digest, verified, errors = _reconcile_identity_field(
        producer_values=(provenance.get("checkpoint_sha256"),),
        corroborating_values=(
            generator,
            fallback,
            report.get("policy_checkpoint_sha256"),
        ),
        normalize=_sha256,
        missing_error="inference checkpoint SHA-256 is missing or malformed",
        malformed_error="checkpoint SHA-256 evidence is malformed",
        disagreement_error="checkpoint SHA-256 evidence does not agree",
    )
    return digest, _sha256(generator), verified, errors


def _checkpoint_size_evidence(
    report: dict[str, Any],
    provenance: dict[str, Any],
    fallback: object,
) -> tuple[int, bool, list[str]]:
    return _reconcile_identity_field(
        producer_values=(provenance.get("checkpoint_size_bytes"),),
        corroborating_values=(fallback, report.get("policy_checkpoint_size_bytes")),
        normalize=_positive_size,
        missing_error="inference checkpoint size is missing or malformed",
        malformed_error="checkpoint size evidence is malformed",
        disagreement_error="checkpoint size evidence does not agree",
        zero_is_absent=True,
    )


def _checkpoint_identity(
    report: dict[str, Any],
    provenance: dict[str, Any],
    *,
    uri_fallback: str,
    digest_fallback: str,
    size_fallback: object,
) -> dict[str, Any]:
    uri, uri_verified, uri_errors = _checkpoint_uri_evidence(
        report, provenance, uri_fallback
    )
    digest, generator, digest_verified, digest_errors = _checkpoint_digest_evidence(
        report, provenance, digest_fallback
    )
    size, size_verified, size_errors = _checkpoint_size_evidence(
        report, provenance, size_fallback
    )
    return {
        "heldout_policy_checkpoint": uri,
        "heldout_policy_checkpoint_sha256": digest,
        "heldout_policy_generator_sha256": generator,
        "heldout_policy_checkpoint_size_bytes": size,
        "heldout_policy_identity_verified": (
            uri_verified and digest_verified and size_verified
        ),
        "heldout_policy_identity_errors": (uri_errors + digest_errors + size_errors),
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
    heldout_report: dict[str, Any],
    *,
    checkpoint_fallback: str = "",
    checkpoint_sha256_fallback: str = "",
    checkpoint_size_fallback: object = 0,
) -> dict[str, Any]:
    raw_provenance = heldout_report.get("policy_inference_provenance")
    provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
    errors: list[str] = []
    if raw_provenance is not None and not isinstance(raw_provenance, dict):
        errors.append("policy_inference_provenance is not an object")

    identity = _checkpoint_identity(
        heldout_report,
        provenance,
        uri_fallback=checkpoint_fallback,
        digest_fallback=checkpoint_sha256_fallback,
        size_fallback=checkpoint_size_fallback,
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
    heldout_report: dict[str, Any],
    *,
    checkpoint_fallback: str = "",
    checkpoint_sha256_fallback: str = "",
    checkpoint_size_fallback: object = 0,
) -> dict[str, Any]:
    """Project held-out evidence only after sealing exact policy identity.

    Args:
        heldout_report: Stage 10 held-out evaluation record.
        checkpoint_fallback: Validation-selected checkpoint URI.
        checkpoint_sha256_fallback: Validation-selected checkpoint digest.
        checkpoint_size_fallback: Validation-selected checkpoint size.

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
    )


def _candidate_policy_metadata(
    candidate: dict[str, Any],
    *,
    checkpoint: str,
) -> dict[str, Any]:
    return {
        "policy_checkpoint": checkpoint,
        "policy_checkpoint_identity": candidate.get("policy_checkpoint_identity", ""),
        "policy_checkpoint_sha256": candidate.get("policy_checkpoint_sha256", ""),
        "policy_checkpoint_size_bytes": candidate.get(
            "policy_checkpoint_size_bytes", ""
        ),
        "policy_download_command": candidate.get("policy_download_command", ""),
        "policy_ui_action": candidate.get("policy_ui_action", ""),
        "policy_deployable": candidate.get("deployable_policy", False),
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

    candidate = candidate or {}
    heldout = heldout_report or {}
    heldout_metadata = heldout_policy_metadata(
        heldout,
        checkpoint_fallback=policy_checkpoint,
        checkpoint_sha256_fallback=str(heldout.get("policy_checkpoint_sha256") or ""),
        checkpoint_size_fallback=heldout.get("policy_checkpoint_size_bytes", 0),
    )
    return {
        **_artifact_metadata(config, artifact_root, progress=progress),
        **_candidate_policy_metadata(candidate, checkpoint=policy_checkpoint),
        **heldout_metadata,
        **_viewer_metadata(config),
    }

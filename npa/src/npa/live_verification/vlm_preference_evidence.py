"""Bind retained preference reports to frozen independent inputs and neutral orders."""

from dataclasses import asdict, fields
import json
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace

from npa.literal_values import require_boolean, require_number
from npa.workbench import vlm_eval


def frozen_preference_expectations(
    options: vlm_eval.VlmPreferenceComparisonRequest,
    *,
    source_sha256: tuple = (None, None),
) -> dict:
    """Freeze request bindings before executing an audit.

    Args:
        options: Operator-selected request with private output and input paths.
        source_sha256: Optional frozen source-file digests for both independent inputs.
    Returns:
        Report field bindings derived from actual normalized independent images.
    Raises:
        ValueError: The request or input images cannot establish a valid contract.
        OSError: An input or rubric cannot be read.
    """
    context = _frozen_context(options, source_sha256)
    requests, orders = vlm_eval._preference_requests(context)
    expected = _report_bindings(context, requests)
    for name, request, order in zip(
        ("first_order", "reversed_order"), requests, orders, strict=True
    ):
        expected.update(_order_bindings(context, name, request, order))
    return expected


def _frozen_context(options, source_sha256):
    _validate_options(options)
    rubric = (
        _read_regular_file(Path(options.rubric_path)).decode("utf-8").strip()
        if options.rubric_path
        else vlm_eval._load_rubric(rubric=options.rubric, rubric_path="")
    )
    vlm_eval._validate_blinded_text(options.task, rubric)
    return vlm_eval._preference_context(
        options,
        vlm_eval.preference_comparison_result_uri_for(options.output_path),
        rubric,
        *(
            _freeze_image(path, digest)
            for path, digest in zip(
                (options.baseline_path, options.candidate_path),
                source_sha256,
                strict=True,
            )
        ),
    )


def _validate_options(options) -> None:
    for name, value in asdict(options).items():
        if name == "timeout_s":
            require_number(value, field=name, minimum=0)
        elif not isinstance(value, str):
            raise ValueError("invalid_preference_request_field")
    vlm_eval._validate_preference_request(options)


def _freeze_image(input_path: str, expected_sha256) -> vlm_eval.SelectedFrame:
    with vlm_eval._materialized_input(input_path) as local:
        candidates = (
            vlm_eval._preference_image_candidates(local) if local.is_dir() else [local]
        )
        if len(candidates) != 1:
            raise ValueError("invalid_preference_image_count")
        content = _read_regular_file(candidates[0])
    if (
        expected_sha256 is not None
        and hashlib.sha256(content).hexdigest() != expected_sha256
    ):
        raise ValueError("preference_control_input_changed")
    # Run the production normalizer on descriptor-verified bytes, never a live FIFO.
    with tempfile.TemporaryDirectory(prefix="npa-preference-binding-") as directory:
        snapshot = Path(directory) / "image.png"
        snapshot.write_bytes(content)
        return vlm_eval._load_single_preference_image(snapshot)


def _read_regular_file(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("invalid_preference_input_file")
        return stream.read()


def _report_bindings(context, requests) -> dict:
    expected = {
        name: getattr(context, name)
        for name in (
            "model",
            "task",
            "rubric",
            "baseline_path",
            "candidate_path",
            "output_path",
            "result_uri",
        )
    }
    expected.update(
        schema_version=vlm_eval.PREFERENCE_COMPARISON_SCHEMA_VERSION,
        normalized_baseline_sha256=vlm_eval._frame_evidence(context.baseline).sha256,
        normalized_candidate_sha256=vlm_eval._frame_evidence(context.candidate).sha256,
        unordered_pair_sha256=vlm_eval._assert_counterbalanced_requests(
            requests, context.baseline, context.candidate
        ),
        requests_counterbalanced=True,
        deployment_status="audit_only",
        operational_rate_estimated=False,
    )
    return expected


def _order_bindings(context, name, request, order) -> dict:
    order_id, first_arm, second_arm, frames = order
    expected = {
        f"{name}.order_id": order_id,
        f"{name}.A_arm": first_arm,
        f"{name}.B_arm": second_arm,
        f"{name}.transport_request_sha256": vlm_eval._sha256_json(request),
    }
    evidence = asdict(vlm_eval._preference_request_evidence(context, request, frames))
    evidence.pop("requested_at")
    expected.update({f"{name}.request.{key}": value for key, value in evidence.items()})
    return json.loads(json.dumps(expected))


def validate_preference_report(report: dict, verdicts: tuple) -> None:
    """Check complete report shape, request hashes and mapped decision consistency.

    Args:
        report: Retained report whose frozen request fields already matched.
        verdicts: Two independently reparsed strict provider verdicts in order.
    Returns:
        None.
    Raises:
        ValueError: A schema, digest, typed metadata or mapped decision is inconsistent.
    """
    _require_fields(report, vlm_eval.VlmPreferenceComparisonReport)
    mapped = []
    for name, verdict in zip(("first_order", "reversed_order"), verdicts, strict=True):
        order = report[name]
        _validate_order_hashes(order)
        mapped.append(
            {"A": order["A_arm"], "B": order["B_arm"]}.get(
                verdict.preference, verdict.preference
            )
        )
    outcomes = tuple(
        SimpleNamespace(verdict=verdict, error=None) for verdict in verdicts
    )
    _validate_decision(report, outcomes, mapped)
    if report["limitations"] != list(vlm_eval._preference_limitations()):
        raise ValueError("invalid_retained_preference_report")
    _require_timestamp(report["generated_at"])


def _validate_decision(report, outcomes, mapped) -> None:
    status = vlm_eval._preference_status(outcomes, tuple(mapped))
    eligible = status in vlm_eval._CONSISTENT_PREFERENCE_STATUSES
    if (report["mapped_preferences"], report["status"]) != (mapped, status):
        raise ValueError("invalid_retained_preference_decision")
    if (
        require_boolean(report["agreement_eligible"], field="agreement_eligible")
        != eligible
    ):
        raise ValueError("invalid_retained_preference_decision")
    if (
        require_boolean(report["escalation_required"], field="escalation_required")
        == eligible
    ):
        raise ValueError("invalid_retained_preference_decision")


def _validate_order_hashes(order: dict) -> None:
    _require_fields(order, vlm_eval.VlmPreferenceOutcome)
    request = order["request"]
    _require_fields(request, vlm_eval.VlmRequestEvidence)
    if order["transport_request_sha256"] != vlm_eval._sha256_json(
        order["transport_request"]
    ):
        raise ValueError("invalid_retained_preference_request")
    if request["request_manifest_sha256"] != vlm_eval._sha256_json(
        request["request_manifest"]
    ):
        raise ValueError("invalid_retained_preference_request")
    _require_timestamp(request["requested_at"])


def _require_fields(value: dict, schema) -> None:
    if not isinstance(value, dict) or set(value) != {
        field.name for field in fields(schema)
    }:
        raise ValueError("invalid_retained_preference_schema")


def _require_timestamp(value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_retained_preference_timestamp")

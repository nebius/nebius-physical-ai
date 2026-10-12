"""Separate retained input authority from the writer of regenerated recordings."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from npa.workflows.sim2real.workflow_io import (
    build_component_record,
    component_record_history_uri,
    validate_component_record,
)

REGENERATION_SCHEMA = "npa.sim2real.regeneration.v1"
PRODUCER_FIELDS = (
    "source_sha",
    "image",
    "image_digest",
    "workflow_job",
    "execution_mode",
)


def authority_sha256(value: Any) -> str:
    """Hash a canonical JSON authority document.

    Args:
        value: JSON-serializable authority material.
    Returns:
        Lowercase SHA-256 of its canonical encoding.
    Raises:
        TypeError: If the material is not JSON serializable.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_writer(writer: Any) -> dict[str, Any]:
    """Require the current task's complete CPU writer identity.

    Args:
        writer: Source, immutable image/digest, job and execution mode.
    Returns:
        Validated provenance without inherited GPU claims.
    Raises:
        ValueError: If writer identity is missing or unqualified.
    """
    if not isinstance(writer, dict) or set(writer) != set(PRODUCER_FIELDS):
        raise ValueError("regeneration writer identity is incomplete")
    # This CPU replay does not inherit the original GPU/device claim.
    return build_component_record(
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence="Regeneration writer identity",
        artifacts={},
        execution_provenance=writer,
    )["artifacts"]


def regeneration_generation(
    authority: dict[str, Any], rrd_sha: str, mcap_sha: str
) -> str:
    """Bind a generation to immutable input, current writer and output bytes.

    Args:
        authority: Separate input and writer authority.
        rrd_sha: Encoded recording digest.
        mcap_sha: Encoded MCAP digest, or empty when absent.
    Returns:
        Content-addressed generation identifier.
    Raises:
        KeyError: If the authority is incomplete.
    """
    return authority_sha256(
        {
            "input": authority["input"],
            "writer": authority["writer"],
            "rrd_sha256": rrd_sha,
            "mcap_sha256": mcap_sha,
        }
    )


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _input_scope(report: dict, retained: dict) -> None:
    report_uri = str(report.get("report_uri") or "")
    root, marker, _ = report_uri.partition("/reports/")
    if not marker or not root.startswith("s3://"):
        raise ValueError("regeneration lacks a run-scoped report identity")
    uri = retained.get("report_uri")
    if not isinstance(uri, str) or not uri.startswith(f"{root}/reports/generations/"):
        raise ValueError("regeneration input report is not immutable and run-scoped")
    suffix = uri.removeprefix(f"{root}/reports/generations/").split("/")
    if (
        len(suffix) != 2
        or suffix[-1] != "sim2real-report.json"
        or (
            len(suffix[0]) not in {32, 64}
            or any(c not in "0123456789abcdef" for c in suffix[0])
        )
    ):
        raise ValueError("regeneration input report generation is invalid")
    original = retained["stage14_record"]
    if retained.get("stage14_history_uri") != component_record_history_uri(
        root, 14, original["content_sha256"]
    ):
        raise ValueError("regeneration input history is not immutable and run-scoped")


def _validate_input_identity(report, retained, expected_input_source):
    if not isinstance(retained, dict) or set(retained) != {
        "source_sha",
        "report_uri",
        "report_sha256",
        "report_size_bytes",
        "report_authority_sha256",
        "stage14_history_uri",
        "stage14_record",
        "upstream_component_sha256",
    }:
        raise ValueError("regeneration input authority is incomplete")
    if (
        report.get("source_sha") != expected_input_source
        or retained["source_sha"] != expected_input_source
    ):
        raise ValueError("regeneration relabeled the retained input source")
    if not _digest(retained["report_sha256"]) or not _digest(
        retained["report_authority_sha256"]
    ):
        raise ValueError("regeneration input report hash is invalid")
    if (
        type(retained["report_size_bytes"]) is not int
        or retained["report_size_bytes"] <= 0
    ):
        raise ValueError("regeneration input report size is invalid")
    _input_scope(report, retained)


def _validate_retained_components(report, retained):
    original = retained["stage14_record"]
    validate_component_record(
        original,
        expected_stage=14,
        expected_name="stage_14_rerun_viz",
        expected_tier="WORKS",
        required_artifacts=("rrd", "report", "report_authority_sha256"),
    )
    if (
        original["artifacts"]["report_authority_sha256"]
        != retained["report_authority_sha256"]
    ):
        raise ValueError("regeneration changed the retained Stage14 authority")
    components = report.get("component_records")
    if not isinstance(components, list) or not components:
        raise ValueError("regeneration lacks retained ComponentRecords")
    upstream = [component.get("content_sha256") for component in components[:-1]]
    if upstream != retained["upstream_component_sha256"] or not all(
        _digest(value) for value in upstream
    ):
        raise ValueError("regeneration changed an upstream ComponentRecord")


def _validate_writer_binding(artifacts, retained, writer):
    if any(artifacts.get(key) != writer[key] for key in PRODUCER_FIELDS):
        raise ValueError("regeneration record does not name its current writer")
    if any(key in artifacts for key in ("gpu_products", "gpu_rows")):
        raise ValueError("regeneration inherited a GPU execution claim")
    if artifacts.get("regeneration_input_sha256") != authority_sha256(retained):
        raise ValueError("regeneration input binding is stale")
    if artifacts.get("regeneration_writer_sha256") != authority_sha256(writer):
        raise ValueError("regeneration writer binding is stale")


def _validate_output_identity(report, artifacts, name, identity):
    if name == "mcap" and identity is None:
        if report.get("mcap_uri") or artifacts.get("mcap"):
            raise ValueError("regeneration MCAP absence disagrees with the report")
        return
    if not isinstance(identity, dict) or set(identity) != {
        "sha256",
        "size_bytes",
        "uri",
    }:
        raise ValueError(f"regeneration {name} identity is incomplete")
    if (
        not _digest(identity["sha256"])
        or type(identity["size_bytes"]) is not int
        or identity["size_bytes"] <= 0
    ):
        raise ValueError(f"regeneration {name} content identity is invalid")
    if identity["uri"] != report.get(f"{name}_uri") or identity["uri"] != artifacts.get(
        name
    ):
        raise ValueError(f"regeneration {name} URI binding is stale")
    if identity["sha256"] != artifacts.get(f"{name}_sha256") or identity[
        "size_bytes"
    ] != artifacts.get(f"{name}_size_bytes"):
        raise ValueError(f"regeneration {name} content binding is stale")


def _validate_output_generation(report, artifacts, authority):
    output = authority["output"]
    if not isinstance(output, dict) or set(output) != {"rrd", "mcap", "generation"}:
        raise ValueError("regeneration output identity is incomplete")
    for name in ("rrd", "mcap"):
        _validate_output_identity(report, artifacts, name, output[name])
    generation = regeneration_generation(
        authority, output["rrd"]["sha256"], (output["mcap"] or {}).get("sha256", "")
    )
    if output["generation"] != generation or report.get("publication_id") != generation:
        raise ValueError(
            "regeneration generation does not bind input, writer and output"
        )


def validate_regeneration_authority(
    report: dict, record: dict, *, expected_input_source: str
) -> str:
    """Validate retained inputs and return the independently qualified writer source.

    Args:
        report: Sealed regenerated report.
        record: Current Stage14 ComponentRecord.
        expected_input_source: Historical run source, not the current writer.
    Returns:
        Exact source of the current recording writer.
    Raises:
        ValueError: If input, writer or output authority is invalid.
    """
    authority = report.get("regeneration")
    if not isinstance(authority, dict) or set(authority) != {
        "schema",
        "input",
        "writer",
        "output",
    }:
        raise ValueError("regeneration authority is incomplete")
    if authority["schema"] != REGENERATION_SCHEMA:
        raise ValueError("regeneration authority schema is invalid")
    retained = authority["input"]
    writer = validate_writer(authority["writer"])
    _validate_input_identity(report, retained, expected_input_source)
    _validate_retained_components(report, retained)
    _validate_writer_binding(record["artifacts"], retained, writer)
    _validate_output_generation(report, record["artifacts"], authority)
    return writer["source_sha"]

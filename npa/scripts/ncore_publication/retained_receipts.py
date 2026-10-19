"""Validate explicitly versioned historical receipts without rewriting their bytes."""

import hashlib
from pathlib import PurePosixPath
import re

from image_byte_scan import core as W, prepare as P

from . import process

HASH = re.compile(r"[0-9a-f]{64}")
S3_PRODUCER = "npa/src/npa/workbench/nurec/s3_probe.py"
S3_CONTROLS = (
    "conditional_create",
    "conditional_nonoverwrite_rejected",
    "exact_readback",
    "enumerated_exactly_once",
    "absent_after_delete",
)


def bound_file(root, record):
    """Read a hash-bound private original beneath the selected evidence root.

    Args:
        root: Private analysis directory.
        record: Exact relative path, byte count and SHA256.
    Returns:
        The authenticated path.
    Raises:
        ValueError, OSError: Identity, ownership or containment differs.
    """
    W.require(isinstance(record, dict), "retained_record_required")
    relative = PurePosixPath(str(record.get("path", "")))
    W.require(
        relative.parts and not relative.is_absolute() and ".." not in relative.parts,
        "retained_relative_path",
    )
    path = root / relative
    W.require(
        all(
            not item.is_symlink()
            for item in (path, *path.parents)
            if item.is_relative_to(root)
        ),
        "retained_symlink",
    )
    binding = P.binding(path)
    W.require(
        type(record.get("bytes")) is int
        and record["bytes"] > 0
        and HASH.fullmatch(str(record.get("sha256", "")))
        and path.stat().st_size == record["bytes"]
        and binding["sha256"] == record["sha256"]
        and process.file_sha(path) == record["sha256"],
        "retained_file_binding",
    )
    return path


def bound_json(root, record):
    """Decode a protected original after checking its immutable file binding.

    Args:
        root: Private analysis directory.
        record: Exact path, bytes and SHA256.
    Returns:
        The original JSON object.
    Raises:
        ValueError, OSError: The original is invalid or substituted.
    """
    value = W.bound_json(P.binding(bound_file(root, record)))
    W.require(isinstance(value, dict), "retained_object_required")
    return value


def source_file(record, expected_path):
    """Authenticate an original producer module against its immutable Git object.

    Args:
        record: Commit, path, blob and SHA256 of the original module.
        expected_path: Contractually selected module, not caller-selected code.
    Returns:
        Original module bytes, without importing or executing old source.
    Raises:
        ValueError, OSError: Commit or complete file identity differs.
    """
    import subprocess

    W.require(
        isinstance(record, dict)
        and record.get("path") == expected_path
        and re.fullmatch(r"[0-9a-f]{40}", str(record.get("commit", ""))),
        "retained_producer_source",
    )
    spec = record["commit"] + ":" + expected_path
    raw = subprocess.check_output(["git", "show", spec], cwd=process.ROOT)
    blob = subprocess.check_output(["git", "rev-parse", spec], cwd=process.ROOT)
    W.require(
        blob.decode().strip() == record.get("blob")
        and hashlib.sha256(raw).hexdigest() == record.get("sha256"),
        "retained_producer_source_binding",
    )
    return raw


def validate_s3_probe(probe, provenance, *, receipt_sha256):
    """Consume the genuine v1 ok contract with original scope and producer binding.

    Args:
        probe: Unmodified original S3 receipt.
        provenance: Separately bound original producer, scope and execution record.
        receipt_sha256: SHA256 computed from the original receipt bytes.
    Returns:
        None. This historical capability is not fresh connectivity or cleanup.
    Raises:
        ValueError: Any literal control, identity, size or hash is invalid.
    """
    W.require(
        provenance.get("format") == "npa_ncore_s3_probe_provenance_v1"
        and provenance.get("receipt_sha256") == receipt_sha256,
        "retained_s3_provenance",
    )
    producer = source_file(provenance.get("producer"), S3_PRODUCER)
    W.require(
        producer == (process.ROOT / S3_PRODUCER).read_bytes(),
        "retained_s3_producer_contract",
    )
    W.require(
        probe.get("format") == "npa_ncore_s3_handoff_probe_v1"
        and probe.get("status") == "ok"
        and all(probe.get(key) is True for key in S3_CONTROLS)
        and probe.get("delete_status") == "confirmed"
        and type(probe.get("payload_bytes")) is int
        and probe["payload_bytes"] > 0,
        "retained_s3_controls",
    )
    _s3_hashes(probe, provenance)


def _s3_hashes(probe, provenance):
    fields = (
        "scope_sha256",
        "payload_sha256",
        "etag_sha256",
        "before_inventory_sha256",
        "during_inventory_sha256",
        "after_inventory_sha256",
    )
    W.require(
        all(HASH.fullmatch(str(probe.get(key, ""))) for key in fields),
        "retained_s3_hashes",
    )
    W.require(
        HASH.fullmatch(str(provenance.get("scope_sha256", "")))
        and probe["scope_sha256"] == provenance["scope_sha256"],
        "retained_s3_original_scope",
    )


def s3_probe(manifest, evidence_root, probe):
    """Bind the versioned adapter to the selected original scope/execution files.

    Args:
        manifest: Proposed aggregate with exact qualification-control hashes.
        evidence_root: Protected workload evidence directory.
        probe: Original already hash-verified probe.
    Returns:
        None.
    Raises:
        ValueError, OSError: Original provenance or any control is missing.
    """
    controls = manifest["qualification_controls"]
    path = evidence_root / "s3-probe-provenance.json"
    W.require(
        process.file_sha(path) == controls.get("s3_probe_provenance_sha256"),
        "retained_s3_provenance_binding",
    )
    provenance = W.bound_json(P.binding(path))
    validate_s3_probe(
        probe, provenance, receipt_sha256=controls["s3_probe_receipt_sha256"]
    )
    selection = bound_json(evidence_root, provenance.get("scope_selection"))
    bucket, prefix = selection.get("bucket"), selection.get("prefix")
    W.require(
        isinstance(bucket, str)
        and bucket
        and "/" not in bucket
        and isinstance(prefix, str)
        and prefix
        and not prefix.startswith("/")
        and not prefix.endswith("/")
        and hashlib.sha256(f"s3://{bucket}/{prefix}/".encode()).hexdigest()
        == probe["scope_sha256"]
        and selection.get("execution_sha") == provenance["producer"]["commit"],
        "retained_s3_scope_selection",
    )
    bound_file(evidence_root, provenance.get("execution"))

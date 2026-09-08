"""Summarize prior successful gates for the five publication audit candidates.

This helper executes no scanner and grants no publication permission. Call it
only after the workflow's mandatory security/SBOM gates (and, for post, exact
digest and anonymous-pull gates) succeed. Raw reports and policies stay private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

TOOLS = frozenset({"curobo", "openpi", "robocasa", "cosmos3-nano-video", "cosmos3-super-benchmark"})
GRAPH_COUNTS = ("layer_count", "entries_read", "regular_files_read", "content_bytes_read", "retained_runtime_count", "required_payload_count", "verified_torch_adapter_count")
SCAN_COUNTS = ("records", "scanned_bytes", "verified_zero_bytes", "regular_files", "regular_bytes", "findings")


class ReceiptError(ValueError):
    """A required gate report is incomplete, inconsistent, or malformed."""


def _require(condition):
    if not condition:
        raise ReceiptError("required validation evidence is incomplete or inconsistent")


def _hex(value, *, length=64, prefix=""):
    _require(isinstance(value, str) and re.fullmatch(re.escape(prefix) + "[0-9a-f]{" + str(length) + "}", value) is not None)
    return value


def _counts(value, names):
    result = {}
    for name in names:
        count = value.get(name)
        _require(type(count) is int and count >= 0)
        result[name] = count
    return result


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result)
        result[name] = value
    return result


def _read_json(path):
    _require(not path.is_symlink() and path.is_file())
    body = path.read_bytes()
    value = json.loads(body, object_pairs_hook=_pairs, parse_constant=lambda _: _require(False))
    _require(isinstance(value, dict))
    return value, hashlib.sha256(body).hexdigest()


def _file_hash(path):
    _require(not path.is_symlink() and path.is_file())
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _payload(path):
    raw, digest = _read_json(path)
    _require(raw.get("format") == "npa_restricted_payload_scan_v2")
    _require(raw.get("source") == "tarball" and raw.get("history_only") is False)
    _require(raw.get("scan_complete") is True and raw.get("verdict") == "clean")
    _require(raw.get("payload_hits") == [] and raw.get("history_hits") == [])
    counts = _counts(raw, ("entries_scanned",))
    _require(counts["entries_scanned"] > 0)
    for key in ("allowlisted_paths_present", "weight_shaped_paths"):
        _require(isinstance(raw.get(key), list))
        counts[key + "_count"] = len(raw[key])
    return {"report_sha256": digest, "scan_complete": True, "history_only": False, "verdict": "clean", **counts}


def _graph(path, tool, image_id):
    raw, digest = _read_json(path)
    schema = "npa.openpi.image-verification.v1" if tool == "openpi" else "npa.curobo.image-verification.v1"
    _require(raw.get("schema_version") == schema and raw.get("valid") is True)
    _require(raw.get("findings") == [] and raw.get("expected_image_id") == image_id)
    if tool == "robocasa":
        _require(raw.get("tool") == "robocasa")
    config_id = _hex(raw.get("image_config_digest"), prefix="sha256:")
    manifest_id = raw.get("image_manifest_digest")
    if manifest_id is not None:
        _hex(manifest_id, prefix="sha256:")
    # Docker save may reconstruct an OCI manifest with uncompressed layers.
    # The workflow binds the pulled config/diffIDs; transport manifests differ.
    _require(image_id in {config_id, manifest_id})
    counts = _counts(raw, GRAPH_COUNTS)
    _require(counts["layer_count"] > 0 and counts["entries_read"] > 0 and counts["regular_files_read"] > 0)
    _require(counts["content_bytes_read"] > 0 and counts["retained_runtime_count"] == 8 and counts["required_payload_count"] >= 9)
    _require(counts["entries_read"] >= counts["regular_files_read"] >= counts["required_payload_count"])
    layers = raw.get("verified_layer_diff_ids")
    _require(isinstance(layers, list) and len(layers) == counts["layer_count"])
    for diff_id in layers:
        _hex(diff_id, prefix="sha256:")
    safe = {"report_sha256": digest, "valid": True, "archive_sha256": _hex(raw.get("docker_save_sha256")),
            "image_config_digest": config_id, "contract_sha256": _hex(raw.get("contract_sha256")), **counts}
    if manifest_id is not None:
        safe["saved_manifest_digest"] = manifest_id
    return raw, safe


def _cosmos(path):
    raw, digest = _read_json(path)
    _require(raw.get("format") == "npa_cosmos3_serving_payload_scan_v1")
    _require(raw.get("scan_complete") is True and raw.get("verdict") == "clean")
    _require(raw.get("payload_hits") == [] and raw.get("history_hits") == [])
    counts = _counts(raw, ("entries_scanned",))
    _require(counts["entries_scanned"] > 0)
    return {"report_sha256": digest, "scan_complete": True, "verdict": "clean", **counts}


def _native_links(phase_root, graph, graph_hash, raw, raw_hash, policy, source_sha):
    authorization, _ = _read_json(phase_root / "authorization/authorization.json")
    _require(authorization.get("schema_version") == "npa.image-byte-scan-authorization.v1")
    _require(authorization.get("accepted_verification") is True)
    _require(authorization.get("expected_image_id") == graph["expected_image_id"])
    _require(authorization.get("verification_report", {}).get("sha256") == graph_hash)
    _require(authorization.get("archive", {}).get("sha256") == graph["docker_save_sha256"])
    _require(raw.get("authorization_sha256") == _canonical_hash(authorization))
    _require(policy.get("report_sha256") == raw_hash and policy.get("image_source_sha") == source_sha)
    _require(policy.get("records_sha256") == _file_hash(phase_root / "scan/records.jsonl"))
    for key in ("archive_sha256", "authorization_sha256", "image_config_digest", "image_manifest_digest"):
        _require(policy.get(key) == raw.get(key))
    _require(raw.get("archive_sha256") == graph["docker_save_sha256"])
    _require(raw.get("image_config_digest") == graph["image_config_digest"])
    _require(raw.get("image_manifest_digest") == graph.get("image_manifest_digest"))
    _require(raw.get("expected_image_id") == graph["expected_image_id"])


def _native(phase_root, graph, graph_hash, source_sha):
    raw, raw_hash = _read_json(phase_root / "scan/report.json")
    policy, policy_hash = _read_json(phase_root / "scan/public-policy-acceptance.json")
    _require(raw.get("schema_version") == "npa.image-byte-scan.v1")
    _require(raw.get("complete") is True and raw.get("helper_joined") is True and "failure_code" not in raw)
    _require(type(raw.get("valid")) is bool)
    _require(policy.get("schema_version") == "npa.image-byte-policy-acceptance.v1")
    _require(policy.get("accepted") is True and policy.get("mode") == "reviewed-exact-native-content")
    counts = _counts(raw, SCAN_COUNTS)
    accepted = _counts(policy, ("raw_findings", "accepted_native_occurrences", "unresolved_occurrences"))
    _require(accepted["raw_findings"] == accepted["accepted_native_occurrences"] == counts["findings"] and accepted["unresolved_occurrences"] == 0)
    _require(policy.get("raw_scan_valid") is raw["valid"] and raw["valid"] == (counts["findings"] == 0))
    _require(counts["regular_files"] == graph["regular_files_read"] and counts["regular_bytes"] == graph["content_bytes_read"])
    _require(counts["records"] > 0 and counts["scanned_bytes"] >= counts["regular_bytes"])
    summary = raw.get("helper_summary", {})
    _require(summary.get("type") == "summary")
    helper = _counts(summary, ("files", "bytes", "findings"))
    # Accepted public-native policy cannot include confidentiality findings;
    # those add to the ledger total but never to the native helper's count.
    _require(helper["files"] == counts["records"] and helper["bytes"] == counts["scanned_bytes"] and helper["findings"] == counts["findings"])
    _require(len(raw.get("layers", [])) == graph["layer_count"])
    _require([row.get("diff_id") for row in raw["layers"]] == graph["verified_layer_diff_ids"])
    _native_links(phase_root, graph, graph_hash, raw, raw_hash, policy, source_sha)
    return {"report_sha256": raw_hash, "acceptance_report_sha256": policy_hash, "complete": True,
            "helper_joined": True, "raw_scan_valid": raw["valid"], "accepted": True,
            "catalog_sha256": _hex(policy.get("catalog_sha256")), **counts, **accepted}


def _sbom(runner_temp, tool):
    raw, digest = _read_json(runner_temp / f"{tool}-sbom.spdx.json")
    _require(raw.get("spdxVersion") == "SPDX-2.3")
    _require(isinstance(raw.get("packages"), list) and len(raw["packages"]) > 0)
    return {"sha256": digest, "format": "SPDX-2.3", "package_count": len(raw["packages"])}


def _tool_evidence(tool, phase, prefix, runner_temp, gate_root, image_id, source_sha):
    if tool.startswith("cosmos3-"):
        gate = _cosmos(runner_temp / f"{prefix}-cosmos3-serving-payload.json")
        archive_hash = _hex((runner_temp / f"{prefix}-archive.sha256").read_text().strip())
        return {"cosmos_payload": gate}, archive_hash
    if tool == "curobo":
        _require(gate_root is not None)
        graph_path = gate_root / phase / "graph.json"
    else:
        graph_path = runner_temp / f"{prefix}-tool-payload.json"
    raw, graph = _graph(graph_path, tool, image_id)
    gates = {"layer_graph": graph}
    if tool == "curobo":
        gates["complete_byte_scan"] = _native(gate_root / phase, raw, graph["report_sha256"], source_sha)
    binding = runner_temp / f"{prefix}-archive.sha256"
    if binding.exists():
        _require(_hex(binding.read_text().strip()) == graph["archive_sha256"])
    return gates, graph["archive_sha256"]


def create_receipt(*, tool: str, source_sha: str, image_id: str, phase: str,
                   runner_temp: Path, digest: str | None = None,
                   curobo_byte_gate_root: Path | None = None) -> dict:
    """Read fixed gate outputs and return an allowlisted public evidence summary.

    Args:
        tool: One of the five publication audit candidates.
        source_sha: Full committed source revision supplied by the trusted workflow.
        image_id: Exact independently inspected config or manifest digest.
        phase: Pre-publication or post-publication gate phase.
        runner_temp: Private directory containing the workflow's fixed report names.
        digest: Exact public registry digest, required only for post-publication.
        curobo_byte_gate_root: Private cuRobo scanner root with the selected phase.

    Returns:
        Hashes, validated counts and fixed status enums; no raw findings or paths.

    Raises:
        ValueError: A required report or binding is incomplete or inconsistent.
        OSError: Required evidence cannot be read.
    """
    _require(tool in TOOLS and phase in {"pre", "post"})
    _hex(source_sha, length=40)
    _hex(image_id, prefix="sha256:")
    _require((phase == "post") == (digest is not None))
    if digest is not None:
        _hex(digest, prefix="sha256:")
    prefix = tool + ("-pushed" if phase == "post" else "")
    gates = {"restricted_payload": _payload(runner_temp / f"{prefix}-payload.json")}
    selected, archive_hash = _tool_evidence(tool, phase, prefix, runner_temp, curobo_byte_gate_root, image_id, source_sha)
    gates.update(selected)
    result = {"schema_version": "npa.public-image-validation-receipt.v1", "tool": tool,
              "source_sha": source_sha, "image_id": image_id, "phase": phase,
              "archive_sha256": archive_hash, "gates": gates, "sbom": _sbom(runner_temp, tool),
              "scope": "prior-gate-summary", "executes_scanners": False,
              "requires_successful_workflow_prerequisites": True}
    if digest is not None:
        result["registry_digest"] = digest
    return result


def _write_receipt(path, receipt):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    """Write a sanitized receipt from the trusted workflow's prior gate outputs.

    Args:
        None.
    Returns:
        Zero when written, one when required evidence is refused.
    Raises:
        SystemExit: Argument parsing fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=sorted(TOOLS), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--runner-temp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest")
    parser.add_argument("--curobo-byte-gate-root", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        receipt = create_receipt(**args)
        _write_receipt(output, receipt)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # An always-run artifact upload must not mistake a prior receipt for
        # the failed invocation's result. Unlink never follows an output symlink.
        output.unlink(missing_ok=True)
        print("Public validation receipt refused: required gate evidence is incomplete or inconsistent.")
        return 1
    print("Sanitized prior-gate receipt written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

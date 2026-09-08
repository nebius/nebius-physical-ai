"""Validate receipt binding and privacy with synthetic reports, never scanners."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/public_image_validation_receipt.py"
SPEC = importlib.util.spec_from_file_location("public_image_receipts", SCRIPT)
RECEIPTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPTS)
SOURCE = "a" * 40
IMAGE = "sha256:" + "b" * 64
ARCHIVE = "c" * 64
REMOTE = "sha256:" + "d" * 64
SENTINEL = "SYNTHETIC-PRIVATE-CUSTOMER-POLICY-CONTENT-MUST-NOT-EXPORT"


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph(tool):
    return {
        "schema_version": "npa.openpi.image-verification.v1" if tool == "openpi" else "npa.curobo.image-verification.v1",
        "tool": tool, "valid": True, "findings": [], "expected_image_id": IMAGE,
        "image_config_digest": IMAGE, "image_manifest_digest": None,
        "docker_save_sha256": ARCHIVE, "contract_sha256": "e" * 64,
        "verified_layer_diff_ids": ["sha256:" + "f" * 64],
        "layer_count": 1, "entries_read": 130, "regular_files_read": 125,
        "content_bytes_read": 8192, "retained_runtime_count": 8,
        "required_payload_count": 124, "verified_torch_adapter_count": 53,
        "untrusted_extra": SENTINEL,
    }


def _native(root, graph_hash, graph):
    authorization = {"schema_version": "npa.image-byte-scan-authorization.v1",
        "accepted_verification": True, "expected_image_id": IMAGE,
        "archive": {"path": SENTINEL, "sha256": ARCHIVE},
        "verification_report": {"path": SENTINEL, "sha256": graph_hash}}
    _put(root / "authorization/authorization.json", authorization)
    raw = {"schema_version": "npa.image-byte-scan.v1", "valid": False, "complete": True,
        "helper_joined": True, "authorization_sha256": RECEIPTS._canonical_hash(authorization),
        "archive_sha256": ARCHIVE, "image_config_digest": IMAGE, "image_manifest_digest": None,
        "expected_image_id": IMAGE, "records": 200, "scanned_bytes": 16384,
        "verified_zero_bytes": 4096, "regular_files": 125, "regular_bytes": 8192,
        "findings": 2, "layers": [{"diff_id": graph["verified_layer_diff_ids"][0]}],
        "helper_summary": {"type": "summary", "files": 200, "bytes": 16384, "findings": 2},
        "confidentiality_policy": {"private": SENTINEL}}
    report_hash = _put(root / "scan/report.json", raw)
    ledger = root / "scan/records.jsonl"
    ledger.write_text(json.dumps({"private_synthetic_record": SENTINEL}) + "\n")
    policy = {"schema_version": "npa.image-byte-policy-acceptance.v1", "accepted": True,
        "mode": "reviewed-exact-native-content", "raw_scan_valid": False, "raw_findings": 2,
        "accepted_native_occurrences": 2, "unresolved_occurrences": 0, "catalog_sha256": "1" * 64,
        "image_source_sha": SOURCE, "report_sha256": report_hash,
        "records_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        **{key: raw[key] for key in ("archive_sha256", "authorization_sha256", "image_config_digest", "image_manifest_digest")},
        "untrusted_private_finding": SENTINEL}
    _put(root / "scan/public-policy-acceptance.json", policy)


@pytest.fixture
def candidate(tmp_path):
    def make(tool="openpi", phase="pre"):
        prefix = tool + ("-pushed" if phase == "post" else "")
        payload = {"format": "npa_restricted_payload_scan_v2", "source": "tarball",
            "image": SENTINEL, "history_only": False, "scan_complete": True,
            "entries_scanned": 140, "verdict": "clean", "payload_hits": [], "history_hits": [],
            "allowlisted_paths_present": [SENTINEL], "weight_shaped_paths": [SENTINEL]}
        _put(tmp_path / f"{prefix}-payload.json", payload)
        _put(tmp_path / f"{tool}-sbom.spdx.json", {"spdxVersion": "SPDX-2.3", "packages": [{"name": SENTINEL}]})
        (tmp_path / f"{prefix}-archive.sha256").write_text(ARCHIVE + "\n")
        gate_root = tmp_path / "private-byte-gate"
        if tool.startswith("cosmos3-"):
            _put(tmp_path / f"{prefix}-cosmos3-serving-payload.json", {
                "format": "npa_cosmos3_serving_payload_scan_v1", "scan_complete": True,
                "entries_scanned": 130, "verdict": "clean", "payload_hits": [], "history_hits": [],
                "private": SENTINEL})
        else:
            graph = _graph(tool)
            graph_path = gate_root / phase / "graph.json" if tool == "curobo" else tmp_path / f"{prefix}-tool-payload.json"
            graph_hash = _put(graph_path, graph)
            if tool == "curobo":
                _native(gate_root / phase, graph_hash, graph)
        return {"tool": tool, "source_sha": SOURCE, "image_id": IMAGE, "phase": phase,
                "runner_temp": tmp_path, "digest": REMOTE if phase == "post" else None,
                "curobo_byte_gate_root": gate_root if tool == "curobo" else None}
    return make


def _change(path, key, value):
    raw = json.loads(path.read_text())
    raw[key] = value
    _put(path, raw)


@pytest.mark.parametrize("tool", sorted(RECEIPTS.TOOLS))
@pytest.mark.parametrize("phase", ["pre", "post"])
def test_five_candidates_export_only_sanitized_prior_gate_receipts(candidate, tool, phase):
    args = candidate(tool, phase)
    receipt = RECEIPTS.create_receipt(**args)
    encoded = json.dumps(receipt)
    assert SENTINEL not in encoded
    assert str(args["runner_temp"]) not in encoded
    assert "findings" not in receipt["gates"]["restricted_payload"]
    assert receipt["archive_sha256"] == ARCHIVE
    assert receipt["executes_scanners"] is False
    assert receipt["scope"] == "prior-gate-summary"
    assert receipt["sbom"]["package_count"] == 1
    assert ("registry_digest" in receipt) == (phase == "post")
    if tool == "curobo":
        native = receipt["gates"]["complete_byte_scan"]
        assert native["raw_scan_valid"] is False and native["accepted"] is True
        assert native["accepted_native_occurrences"] == native["findings"] == 2


@pytest.mark.parametrize("key,value", [
    ("scan_complete", False), ("scan_complete", 1), ("history_only", True),
    ("source", "registry"), ("entries_scanned", 0), ("entries_scanned", True),
    ("entries_scanned", -1), ("verdict", "restricted-payload-detected"),
    ("payload_hits", [SENTINEL]), ("history_hits", [SENTINEL]),
    ("weight_shaped_paths", SENTINEL),
])
def test_partial_history_or_denied_payload_cannot_get_receipt(candidate, key, value):
    args = candidate()
    _change(args["runner_temp"] / "openpi-payload.json", key, value)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


@pytest.mark.parametrize("key,value", [
    ("valid", False), ("valid", 1), ("expected_image_id", REMOTE),
    ("docker_save_sha256", SENTINEL), ("findings", [SENTINEL]),
    ("layer_count", 0), ("verified_layer_diff_ids", []), ("content_bytes_read", -1),
    ("retained_runtime_count", 7), ("regular_files_read", 1),
    ("image_manifest_digest", SENTINEL), ("contract_sha256", SENTINEL),
])
def test_incomplete_or_mismatched_graph_cannot_get_receipt(candidate, key, value):
    args = candidate()
    _change(args["runner_temp"] / "openpi-tool-payload.json", key, value)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_saved_and_registry_serializations_bind_same_config_without_equal_manifests(candidate):
    args = candidate("openpi", "post")
    layer = b"synthetic uncompressed layer bytes"
    compressed = gzip.compress(layer, mtime=0)
    config = {"rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(layer).hexdigest()]}}
    config_bytes = json.dumps(config).encode()
    config_id = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    config_descriptor = {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": config_id, "size": len(config_bytes)}
    manifests = []
    for content, media in [(layer, "tar"), (compressed, "tar+gzip")]:
        manifest = {"schemaVersion": 2, "config": config_descriptor,
            "layers": [{"mediaType": "application/vnd.oci.image.layer.v1." + media,
                        "digest": "sha256:" + hashlib.sha256(content).hexdigest(), "size": len(content)}]}
        manifests.append("sha256:" + hashlib.sha256(json.dumps(manifest).encode()).hexdigest())
    assert gzip.decompress(compressed) == layer and manifests[0] != manifests[1]
    path = args["runner_temp"] / "openpi-pushed-tool-payload.json"
    graph = json.loads(path.read_text())
    graph.update(expected_image_id=config_id, image_config_digest=config_id, image_manifest_digest=manifests[0],
                 verified_layer_diff_ids=config["rootfs"]["diff_ids"])
    _put(path, graph)
    args.update(image_id=config_id, digest=manifests[1])
    receipt = RECEIPTS.create_receipt(**args)
    assert receipt["gates"]["layer_graph"]["saved_manifest_digest"] == manifests[0]
    assert receipt["gates"]["layer_graph"]["image_config_digest"] == config_id
    assert receipt["registry_digest"] == manifests[1]
    _change(path, "image_config_digest", IMAGE)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_confidentiality_findings_cannot_be_laundered_as_native_acceptance(candidate):
    args = candidate("curobo")
    root = args["curobo_byte_gate_root"] / "pre/scan"
    raw = json.loads((root / "report.json").read_text())
    # Native helper findings are only one source of the ledger's total.
    # Adding even one confidentiality finding prevents public-native acceptance.
    raw["findings"] += 1
    report_hash = _put(root / "report.json", raw)
    policy = json.loads((root / "public-policy-acceptance.json").read_text())
    policy.update(report_sha256=report_hash, raw_findings=3, accepted_native_occurrences=3)
    _put(root / "public-policy-acceptance.json", policy)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


@pytest.mark.parametrize("file,key,value", [
    ("report.json", "complete", False), ("report.json", "helper_joined", False),
    ("report.json", "expected_image_id", REMOTE), ("report.json", "regular_bytes", 5),
    ("report.json", "helper_summary", {"type": "summary", "files": 1, "bytes": 16384, "findings": 2}),
    ("public-policy-acceptance.json", "accepted", False),
    ("public-policy-acceptance.json", "accepted", 1),
    ("public-policy-acceptance.json", "image_source_sha", "2" * 40),
    ("public-policy-acceptance.json", "report_sha256", "2" * 64),
    ("public-policy-acceptance.json", "records_sha256", "2" * 64),
    ("public-policy-acceptance.json", "accepted_native_occurrences", 1),
    ("public-policy-acceptance.json", "unresolved_occurrences", 1),
    ("public-policy-acceptance.json", "raw_scan_valid", True),
    ("public-policy-acceptance.json", "authorization_sha256", "2" * 64),
])
def test_native_scan_and_policy_acceptance_must_stay_linked(candidate, file, key, value):
    args = candidate("curobo")
    _change(args["curobo_byte_gate_root"] / "pre/scan" / file, key, value)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_changed_native_ledger_and_authorization_refuse(candidate):
    args = candidate("curobo")
    (args["curobo_byte_gate_root"] / "pre/scan/records.jsonl").write_text("changed\n")
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)
    args = candidate("curobo")
    _change(args["curobo_byte_gate_root"] / "pre/authorization/authorization.json", "accepted_verification", False)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


@pytest.mark.parametrize("tool", ["openpi", "cosmos3-nano-video"])
def test_archive_binding_mismatch_or_garbage_refuses(candidate, tool):
    args = candidate(tool)
    (args["runner_temp"] / f"{tool}-archive.sha256").write_text(SENTINEL)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_missing_cosmos_archive_binding_refuses(candidate):
    args = candidate("cosmos3-super-benchmark")
    (args["runner_temp"] / "cosmos3-super-benchmark-archive.sha256").unlink()
    with pytest.raises(OSError):
        RECEIPTS.create_receipt(**args)


@pytest.mark.parametrize("body", [b'{"format": "one", "format": "two"}', b'{"value": NaN}', b'[]'])
def test_malformed_json_refuses_without_exporting_content(candidate, body):
    args = candidate()
    (args["runner_temp"] / "openpi-payload.json").write_bytes(body)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_report_symlink_refuses(candidate):
    args = candidate()
    report = args["runner_temp"] / "openpi-payload.json"
    target = args["runner_temp"] / "other.json"
    report.rename(target)
    report.symlink_to(target)
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


@pytest.mark.parametrize("key,value", [("tool", "unrelated"), ("source_sha", "short"), ("image_id", SENTINEL), ("phase", "other"), ("digest", REMOTE)])
def test_input_identity_and_phase_are_strict(candidate, key, value):
    args = candidate()
    args[key] = value
    with pytest.raises(ValueError):
        RECEIPTS.create_receipt(**args)


def test_cli_writes_private_receipt_and_removes_stale_success_on_refusal(candidate):
    args = candidate()
    output = args["runner_temp"] / "receipt.json"
    command = [sys.executable, str(SCRIPT), "--tool", "openpi", "--source-sha", SOURCE,
               "--image-id", IMAGE, "--phase", "pre", "--runner-temp", str(args["runner_temp"]), "--output", str(output)]
    success = subprocess.run(command, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr
    assert output.stat().st_mode & 0o777 == 0o600
    assert SENTINEL not in output.read_text()
    _change(args["runner_temp"] / "openpi-payload.json", "payload_hits", [SENTINEL])
    failure = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failure.returncode == 1
    assert not output.exists()
    assert SENTINEL not in failure.stdout + failure.stderr

"""The hosted failure summary exposes counts while preserving gate failure."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_image_byte_confidentiality import compile_policy


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "npa/docker/workbench/robotwin/byte_gate.sh"
POLICY = ROOT / "npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json"
KINDS = (
    "layer_regular_content",
    "outer_regular_content",
    "raw_tar_header",
    "raw_tar_extension",
    "logical_tar_path",
    "logical_tar_link",
    "verified_zero_content",
    "unexplained_tar_trailer",
    "nonzero_tar_padding",
    "other",
)


@pytest.mark.parametrize(
    "case",
    [
        "reviewed",
        "native",
        "outer",
        "literal",
        "structural",
        "customer_regex",
        "infra_regex",
        "customer_line_only",
        "customer_record_only",
        "both_regex",
        "outer_regex",
        "unknown_shape",
        "unknown_view",
        "unknown_kind",
        "count_mismatch",
        "private_count",
        "incomplete",
        "private_error",
        "missing",
        "malformed",
    ],
)
def test_failure_summary_is_sanitized_and_keeps_original_exit(tmp_path, case):
    phase = tmp_path / "phase"
    scan = phase / "scan"
    scan.mkdir(parents=True)
    entry = json.loads(POLICY.read_text())["entries"][0]
    marker = "synthetic-private-match-path-and-regex"
    row = {
        "type": "record",
        "kind": entry["record_kind"],
        "sha256": entry["record_sha256"],
        "bytes": entry["record_bytes"],
        "findings": list(entry["native_findings"]),
        "private_path": marker,
    }
    report = {"complete": True, "helper_joined": True, "private_value": marker}
    if case in {"native", "outer"}:
        row["sha256"] = "0" * 64
    if case == "outer":
        row["kind"] = "outer_regular_content"
    if case == "incomplete":
        report.update(complete=False, failure_code="helper_unexpected_eof")
    if case == "private_error":
        report["failure_code"] = marker
    rows = [row]
    if case in {"literal", "structural"}:
        rows.append(
            {
                "type": "finding",
                "rule_id": "private_literal" if case == "literal" else "pkcs12-file",
                "matched_value": marker,
                "pattern": marker,
            }
        )
    if case in {
        "customer_regex",
        "infra_regex",
        "customer_line_only",
        "customer_record_only",
        "both_regex",
        "outer_regex",
        "unknown_view",
    }:
        # Use the actual scanner's six-field regex output. Ledger.send appends
        # this to record.findings alongside three-field native helper findings.
        customer, raw = marker, marker.encode()
        if case == "customer_line_only":
            customer, raw = "^" + marker + "$", (marker + "\nother").encode()
        if case == "customer_record_only":
            customer, raw = "first[\\s\\S]*last", b"first\nlast"
        policy = compile_policy(
            customer if case != "infra_regex" else "absent-customer",
            marker if case in {"infra_regex", "both_regex"} else None,
        )
        findings = [asdict(f) for f in policy.scan_record(raw).findings]
        assert len(findings) == (2 if case == "both_regex" else 1)
        if case == "unknown_view":
            findings[0]["views"] = [marker]
        if case == "outer_regex":
            rows.append({**row, "kind": "outer_regular_content", "findings": findings})
        else:
            row["findings"].extend(findings)
    if case == "unknown_shape":
        row["findings"].append({"private_value": marker})
    if case == "unknown_kind":
        row["kind"] = marker
    report["findings"] = sum(
        len(r.get("findings", [])) if r["type"] == "record" else 1 for r in rows
    )
    report["helper_summary"] = {"findings": len(entry["native_findings"])}
    if case == "count_mismatch":
        report["findings"] += 1
    if case == "private_count":
        report["helper_summary"]["findings"] = marker
    if case != "missing":
        (scan / "report.json").write_text(json.dumps(report))
    (scan / "records.jsonl").write_text(
        marker if case == "malformed" else "\n".join(json.dumps(r) for r in rows)
    )
    # Execute the exact failure branch, including its final exit, without a scan.
    script = GATE.read_text().split("if (( scan_status != 0 )); then\n", 1)[1]
    script = script.rsplit("\nfi", 1)[0].replace(
        "npa/.venv/bin/python", shlex.quote(sys.executable)
    )
    result = subprocess.run(
        ["bash", "-euc", script],
        cwd=ROOT,
        env={**os.environ, "phase": str(phase), "scan_status": "23"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 23
    assert result.stderr == ""
    assert marker not in result.stdout and str(phase) not in result.stdout
    summary = json.loads(result.stdout.removeprefix("RoboTwin byte gate diagnostic: "))
    if case in {"missing", "malformed"}:
        assert summary == {"diagnostic": "unavailable"}
        return
    customer = case in {
        "customer_regex",
        "customer_line_only",
        "customer_record_only",
        "both_regex",
        "outer_regex",
    }
    infra = case in {"infra_regex", "both_regex"}
    views = {
        rule: {"line_only": 0, "record_only": 0, "line_and_record": 0}
        for rule in ("customer-denylist", "infra-denylist")
    }
    regex_kinds = dict.fromkeys(KINDS, 0)
    if customer or infra:
        view = {
            "customer_line_only": "line_only",
            "customer_record_only": "record_only",
        }.get(case, "line_and_record")
        if customer:
            views["customer-denylist"][view] = 1
        if infra:
            views["infra-denylist"][view] = 1
        regex_kinds[
            "outer_regular_content" if case == "outer_regex" else entry["record_kind"]
        ] = int(customer) + int(infra)
    unreviewed = case in {"native", "outer", "unknown_kind"}
    native_kinds = dict.fromkeys(KINDS, 0)
    if unreviewed:
        native_kinds[row["kind"] if row["kind"] in KINDS else "other"] = 1
    assert summary == {
        "complete": case != "incomplete",
        "helper_joined": True,
        "native_findings": len(entry["native_findings"]),
        "confidentiality_findings": int(case == "literal") + int(customer) + int(infra),
        "customer_regex_findings": int(customer),
        "infra_regex_findings": int(infra),
        "literal_findings": int(case == "literal"),
        "structural_findings": int(case == "structural"),
        "unclassified_findings": int(case in {"unknown_shape", "unknown_view"}),
        "regex_views": views,
        "regex_findings_by_kind": regex_kinds,
        "unreviewed_native_records": int(unreviewed),
        "unreviewed_outer_records": int(case == "outer"),
        "unreviewed_native_record_kinds": native_kinds,
        "reported_total_findings": report["findings"],
        "helper_native_findings": None
        if case == "private_count"
        else len(entry["native_findings"]),
        "finding_counts_conserved": case not in {"count_mismatch", "private_count"},
        "scan_failure": "helper_unexpected_eof"
        if case == "incomplete"
        else "unspecified",
    }

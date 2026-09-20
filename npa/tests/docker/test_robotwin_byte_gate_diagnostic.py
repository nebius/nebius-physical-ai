"""The hosted failure summary exposes counts while preserving gate failure."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "npa/docker/workbench/robotwin/byte_gate.sh"
POLICY = ROOT / "npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json"


@pytest.mark.parametrize(
    "case", ["reviewed", "native", "outer", "confidentiality", "incomplete", "private_error", "missing", "malformed"]
)
def test_failure_summary_is_sanitized_and_keeps_original_exit(tmp_path, case):
    phase = tmp_path / "phase"
    scan = phase / "scan"
    scan.mkdir(parents=True)
    entry = json.loads(POLICY.read_text())["entries"][0]
    marker = "synthetic-private-match-path-and-regex"
    row = {
        "type": "record", "kind": entry["record_kind"],
        "sha256": entry["record_sha256"], "bytes": entry["record_bytes"],
        "findings": entry["native_findings"], "private_path": marker,
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
    if case == "confidentiality":
        rows.append({"type": "finding", "matched_value": marker, "pattern": marker})
    if case != "missing":
        (scan / "report.json").write_text(json.dumps(report))
    (scan / "records.jsonl").write_text(
        marker if case == "malformed" else "\n".join(json.dumps(r) for r in rows)
    )
    # Execute the exact failure branch, including its final exit, without a scan.
    script = GATE.read_text().split("if (( scan_status != 0 )); then\n", 1)[1]
    script = script.rsplit("\nfi", 1)[0].replace("npa/.venv/bin/python", shlex.quote(sys.executable))
    result = subprocess.run(
        ["bash", "-euc", script], cwd=ROOT,
        env={**os.environ, "phase": str(phase), "scan_status": "23"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 23
    assert result.stderr == ""
    assert marker not in result.stdout and str(phase) not in result.stdout
    summary = json.loads(result.stdout.removeprefix("RoboTwin byte gate diagnostic: "))
    if case in {"missing", "malformed"}:
        assert summary == {"diagnostic": "unavailable"}
        return
    assert summary == {
        "complete": case != "incomplete", "helper_joined": True,
        "native_findings": len(entry["native_findings"]),
        "confidentiality_findings": int(case == "confidentiality"),
        "unreviewed_native_records": int(case in {"native", "outer"}),
        "unreviewed_outer_records": int(case == "outer"),
        "scan_failure": "helper_unexpected_eof" if case == "incomplete" else "unspecified",
    }

#!/usr/bin/env bash
# Inspect the same saved RoboTwin image before push and after its exact pull.
set -euo pipefail
umask 077
archive="${1:?saved image archive required}"
image="${2:?inspected Docker image required}"
phase_name="${3:?pre or post required}"
[[ "$phase_name" == pre || "$phase_name" == post ]]
root="${ROBOTWIN_BYTE_GATE_ROOT:?prepared byte tools required}"
phase="$root/$phase_name"
mkdir -m 0700 "$phase"
mv "$archive" "$phase/image.tar"
chmod 0600 "$phase/image.tar"
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
npa/.venv/bin/python npa/scripts/scan_image_robotwin_payload.py \
  --docker-save "$phase/image.tar" --output "$phase/payload.json"
npa/.venv/bin/python npa/scripts/image_byte_scan/robotwin_verification.py \
  --analysis-root "$root" --trusted-root "$PWD" --archive "$phase/image.tar" \
  --expected-image-id "$image_id" --output-dir "$phase/graph"
npa/.venv/bin/python npa/scripts/image_byte_scan/prepare.py authorize \
  --analysis-root "$root" --trusted-root "$PWD" \
  --tools-receipt "$root/tools/dependency-receipt.json" \
  --native-receipt "$root/native/dependencies.json" \
  --archive "$phase/image.tar" --verification-report "$phase/graph/verification.json" \
  --expected-image-id "$image_id" --output-dir "$phase/authorization" --policy-mode ci-regex
scan_status=0
npa/.venv/bin/python npa/scripts/scan_image_bytes.py \
  --analysis-root "$root" --trusted-root "$PWD" \
  --authorization "$phase/authorization/authorization.json" --output-dir "$phase/scan" \
  --public-native-policy "$PWD/npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json" \
  --public-native-policy-sha256 "${ROBOTWIN_PUBLIC_NATIVE_POLICY_SHA256:?reviewed policy required}" \
  || scan_status=$?
if (( scan_status != 0 )); then
  # Counts and fixed categories only: the ledger, matches and policy stay private.
  npa/.venv/bin/python - "$phase/scan" \
    "$PWD/npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json" <<'PY' || true
import json
from pathlib import Path
import sys

try:
    scan = Path(sys.argv[1])
    report = json.loads((scan / "report.json").read_text())
    rows = [json.loads(line) for line in (scan / "records.jsonl").read_text().splitlines()]
    policy = json.loads(Path(sys.argv[2]).read_text())

    def identity(kind, digest, size, findings):
        return kind, digest, size, tuple(sorted(json.dumps(f, sort_keys=True) for f in findings))

    approved = {
        identity(e["record_kind"], e["record_sha256"], e["record_bytes"], e["native_findings"])
        for e in policy["entries"]
    }
    native_fields = {"rule_id", "start_line", "end_line"}
    regex_fields = native_fields | {"start_byte", "end_byte", "views"}
    kinds = (
        "layer_regular_content", "outer_regular_content", "raw_tar_header",
        "raw_tar_extension", "logical_tar_path", "logical_tar_link",
        "verified_zero_content", "unexplained_tar_trailer", "nonzero_tar_padding", "other",
    )
    regex_views = {
        rule: {"line_only": 0, "record_only": 0, "line_and_record": 0}
        for rule in ("customer-denylist", "infra-denylist")
    }
    view_names = {("line",): "line_only", ("record",): "record_only",
                  ("line", "record"): "line_and_record"}
    regex_kinds = dict.fromkeys(kinds, 0)
    native = []
    unknown = 0
    for row in rows:
        if row.get("type") != "record":
            continue
        findings = []
        for finding in row["findings"]:
            if isinstance(finding, dict) and set(finding) == native_fields:
                findings.append(finding)
            elif (isinstance(finding, dict) and set(finding) == regex_fields
                  and isinstance(finding["rule_id"], str)
                  and finding["rule_id"] in regex_views
                  and isinstance(finding["views"], list)
                  and all(isinstance(v, str) for v in finding["views"])
                  and tuple(sorted(finding["views"])) in view_names):
                view = view_names[tuple(sorted(finding["views"]))]
                regex_views[finding["rule_id"]][view] += 1
                regex_kinds[row["kind"] if row["kind"] in kinds else "other"] += 1
            else:
                unknown += 1
        if findings:
            native.append({**row, "findings": findings})
    unreviewed = [r for r in native if identity(r["kind"], r["sha256"], r["bytes"], r["findings"]) not in approved]
    unreviewed_kinds = dict.fromkeys(kinds, 0)
    for row in unreviewed:
        unreviewed_kinds[row["kind"] if row["kind"] in kinds else "other"] += 1
    literals = sum(r.get("type") == "finding" and r.get("rule_id") == "private_literal" for r in rows)
    structural = sum(r.get("type") == "finding" and r.get("rule_id") != "private_literal" for r in rows)
    customer = sum(regex_views["customer-denylist"].values())
    infra = sum(regex_views["infra-denylist"].values())

    def count(value):
        return value if type(value) is int and value >= 0 else None

    summary = {
        "complete": report.get("complete") is True,
        "helper_joined": report.get("helper_joined") is True,
        "native_findings": sum(len(r["findings"]) for r in native),
        "confidentiality_findings": literals + customer + infra,
        "customer_regex_findings": customer,
        "infra_regex_findings": infra,
        "literal_findings": literals,
        "structural_findings": structural,
        "unclassified_findings": unknown,
        "regex_views": regex_views,
        "regex_findings_by_kind": regex_kinds,
        "unreviewed_native_records": len(unreviewed),
        "unreviewed_outer_records": sum(r["kind"] == "outer_regular_content" for r in unreviewed),
        "unreviewed_native_record_kinds": unreviewed_kinds,
        "reported_total_findings": count(report.get("findings")),
        "helper_native_findings": count(report.get("helper_summary", {}).get("findings")),
    }
    summary["finding_counts_conserved"] = (
        summary["native_findings"] == summary["helper_native_findings"]
        and summary["native_findings"] + literals + customer + infra + structural + unknown
        == summary["reported_total_findings"]
    )
    # Never print an arbitrary error or a value copied from a private report.
    codes = {code: code for code in (
        "helper_unexpected_eof", "helper_exit_status", "scan_cancelled",
        "uninterpretable_input_or_scanner_failure", "invalid_scan_configuration",
    )}
    summary["scan_failure"] = codes.get(report.get("failure_code"), "unspecified")
except Exception:
    summary = {"diagnostic": "unavailable"}
print("RoboTwin byte gate diagnostic: " + json.dumps(summary, sort_keys=True))
PY
  if [[ -n "${ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL:-}" ]]; then
    # Even unexpected library diagnostics must never expose the signed URL.
    if npa/.venv/bin/python npa/docker/workbench/robotwin/private_failure_upload.py "$phase" >/dev/null 2>&1; then
      echo 'RoboTwin private failure evidence upload completed'
    else
      echo 'RoboTwin private failure evidence upload failed'
    fi
  fi
  exit "$scan_status"
fi

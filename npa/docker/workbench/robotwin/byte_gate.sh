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
    native = [r for r in rows if r.get("type") == "record" and r.get("findings")]
    unreviewed = [r for r in native if identity(r["kind"], r["sha256"], r["bytes"], r["findings"]) not in approved]
    summary = {
        "complete": report.get("complete") is True,
        "helper_joined": report.get("helper_joined") is True,
        "native_findings": sum(len(r["findings"]) for r in native),
        "confidentiality_findings": sum(r.get("type") == "finding" for r in rows),
        "unreviewed_native_records": len(unreviewed),
        "unreviewed_outer_records": sum(r["kind"] == "outer_regular_content" for r in unreviewed),
    }
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
  exit "$scan_status"
fi

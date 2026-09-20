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
npa/.venv/bin/python npa/scripts/scan_image_bytes.py \
  --analysis-root "$root" --trusted-root "$PWD" \
  --authorization "$phase/authorization/authorization.json" --output-dir "$phase/scan" \
  --public-native-policy "$PWD/npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json" \
  --public-native-policy-sha256 "${ROBOTWIN_PUBLIC_NATIVE_POLICY_SHA256:?reviewed policy required}"

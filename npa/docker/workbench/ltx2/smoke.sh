#!/usr/bin/env bash
# Golden eval for npa-ltx2, run against the BUILT image.
#
# The claim this image makes is "contains no LTX-2.5 and refuses to fetch it
# without the operator's own entitlement". That claim is about the artifact, so
# it is checked against the artifact: the refusal is exercised in every
# direction, against private directories asserted empty afterwards.
set -euo pipefail

ltx-runtime health
ltx-runtime version
ltx-runtime status
ltx-runtime terms

# Each refusal runs against directories only this smoke can see, the same way
# `ltx-runtime assert-refusal` does. Watching the real caches cannot express "the
# refusal downloaded nothing" once they are durable and shared
# (docs/workbench/model-weight-cache.md): the second run legitimately finds what
# the first fetched, and another LTX stage can be writing to them throughout. A
# write that lands in the shared tree is not attributable to this smoke; a write
# that lands here is.
PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-ltx-smoke.XXXXXX")"
trap 'rm -rf "$PROBE_DIR"' EXIT
PROBE_RUNTIME_CACHE="$PROBE_DIR/runtime"
PROBE_MODEL_CACHE="$PROBE_DIR/weights"
mkdir -p "$PROBE_RUNTIME_CACHE" "$PROBE_MODEL_CACHE"

caches_are_untouched() {
  test -z "$(find "$PROBE_RUNTIME_CACHE" -mindepth 1 -print -quit 2>/dev/null)"
  test -z "$(find "$PROBE_MODEL_CACHE" -mindepth 1 -print -quit 2>/dev/null)"
}

# `-u HF_TOKEN` rather than `HF_TOKEN=`: an empty assignment still reads as a
# token assignment, both to a human skimming for leaked credentials and to the
# image's own secret scanner, which flagged that line. Unsetting says what is
# meant and leaves nothing that looks like a secret in a shipped file.
refuses_with_78() {
  local desc="$1" mode="$2"; shift 2
  set +e
  env "NPA_LTX_RUNTIME_CACHE=$PROBE_RUNTIME_CACHE" \
      "NPA_LTX_MODEL_CACHE=$PROBE_MODEL_CACHE" \
      "$@" ltx-runtime "$mode" >/tmp/ltx-out 2>/tmp/ltx-err
  local rc=$?
  set -e
  test "$rc" = 78 || { echo "expected 78 for ${desc}, got ${rc}" >&2; exit 1; }
  caches_are_untouched || { echo "cache was written for ${desc}" >&2; exit 1; }
}

# The source is licensed material too (Section 1.9), so it is gated on the same
# Hugging Face entitlement as the weights and refuses without it.
refuses_with_78 "source fetch without an entitlement" ensure -u HF_TOKEN
grep -q "GATED repository" /tmp/ltx-err

refuses_with_78 "weight fetch without an entitlement" fetch-weights -u HF_TOKEN
grep -q "GATED repository" /tmp/ltx-err
grep -q "Nothing has been downloaded" /tmp/ltx-err

# NVIDIA's CUDA terms are a separate vendor's decision and must refuse
# independently. `assert-refusal` reaches that gate without this script ever
# holding a token, real or fake, and asserts which gate refused in each case.
ltx-runtime assert-refusal | grep -Fq NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK
caches_are_untouched

# The image must carry no LTX payload of its own.
test -z "$(find / -xdev \( -name 'ltx_core' -o -name 'ltx_pipelines' -o -name 'ltx-2.5-*' \) -print -quit 2>/dev/null)"
test -z "$(find / -xdev -name '*.safetensors' -print -quit 2>/dev/null)"

echo "npa-ltx2 smoke: OK (refusals enforced, no LTX payload present)"

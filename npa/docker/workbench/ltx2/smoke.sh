#!/usr/bin/env bash
# Golden eval for npa-ltx2, run against the BUILT image.
#
# The claim this image makes is "contains no LTX-2.5 and refuses to fetch it
# without an operator declaration". That claim is about the artifact, so it is
# checked against the artifact: the refusal is exercised in every direction and
# the caches are asserted still empty afterwards.
set -euo pipefail

ltx-runtime health
ltx-runtime version
ltx-runtime status

RUNTIME_CACHE="${NPA_LTX_RUNTIME_CACHE:-/workspace/.cache/npa/ltx2/runtime}"
MODEL_CACHE="${NPA_LTX_MODEL_CACHE:-/workspace/model-cache/ltx-2.5}"

caches_are_empty() {
  test -z "$(find "$RUNTIME_CACHE" -mindepth 1 -print -quit 2>/dev/null)"
  test -z "$(find "$MODEL_CACHE" -mindepth 1 -print -quit 2>/dev/null)"
}

refuses_with_78() {
  local desc="$1"; shift
  set +e
  env "$@" ltx-runtime ensure >/tmp/ltx-out 2>/tmp/ltx-err
  local rc=$?
  set -e
  test "$rc" = 78 || { echo "expected 78 for ${desc}, got ${rc}" >&2; exit 1; }
  grep -q "Nothing has been downloaded" /tmp/ltx-err
  caches_are_empty || { echo "cache was written for ${desc}" >&2; exit 1; }
}

# No declaration at all.
refuses_with_78 "no declaration" \
  NPA_LTX_ACCEPT_COMMUNITY_LICENSE= NPA_LTX_ENTITY_CLASS= NPA_LTX_USE_CLASS=

# Licence accepted, but the entity/use questions unanswered: acceptance alone
# must not be reused as an answer to the revenue-threshold question.
refuses_with_78 "acceptance without entity/use declaration" \
  NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES NPA_LTX_ENTITY_CLASS= NPA_LTX_USE_CLASS=

# A Commercial Entity declaring commercial use without a paid agreement is the
# combination Section 2.1 prohibits outright.
refuses_with_78 "commercial entity, commercial use, no agreement" \
  NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES \
  NPA_LTX_ENTITY_CLASS=commercial \
  NPA_LTX_USE_CLASS=commercial \
  NPA_LTX_COMMERCIAL_AGREEMENT_REF=

# Fully declared for Lightricks, but NVIDIA's terms are a separate vendor's
# decision and must be refused independently.
refuses_with_78 "no NVIDIA runtime acceptance" \
  NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES \
  NPA_LTX_ENTITY_CLASS=community \
  NPA_LTX_USE_CLASS=non-commercial \
  NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS=

# Weight fetch is gated by the declaration too, and separately by the operator's
# own Hugging Face entitlement.
set +e
env NPA_LTX_ACCEPT_COMMUNITY_LICENSE= ltx-runtime fetch-weights >/dev/null 2>/tmp/ltx-err
rc=$?
set -e
test "$rc" = 78
caches_are_empty

set +e
# `-u HF_TOKEN` rather than `HF_TOKEN=`: an empty assignment still reads as a
# token assignment, both to a human skimming for leaked credentials and to the
# image's own secret scanner, which flagged this line. Unsetting says what is
# meant and leaves nothing that looks like a secret in a shipped file.
env -u HF_TOKEN \
    NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES \
    NPA_LTX_ENTITY_CLASS=community \
    NPA_LTX_USE_CLASS=non-commercial \
    ltx-runtime fetch-weights >/dev/null 2>/tmp/ltx-err
rc=$?
set -e
test "$rc" = 78
grep -q "GATED repository" /tmp/ltx-err
caches_are_empty

# The image must carry no LTX payload of its own.
test -z "$(find / -xdev \( -name 'ltx_core' -o -name 'ltx_pipelines' -o -name 'ltx-2.5-*' \) -print -quit 2>/dev/null)"
test -z "$(find / -xdev -name '*.safetensors' -print -quit 2>/dev/null)"

echo "npa-ltx2 smoke: OK (refusals enforced, no LTX payload present)"

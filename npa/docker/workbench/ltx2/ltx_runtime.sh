#!/usr/bin/env bash
# Runtime delivery for LTX-2.5. The image ships none of it.
#
# Two separate vendors have to deliver to the operator directly, and neither
# entitlement is ever baked:
#
#   Lightricks  ltx-core / ltx-pipelines source AND the gated LTX-2.5 weights,
#               under the LTX-2.x Community License Agreement. Both are licensed
#               material (Section 1.9 covers the accompanying source code), and
#               both arrive under the operator's own HF_TOKEN.
#   NVIDIA      the CUDA PyTorch stack that upstream's own pins pull from
#               download.pytorch.org/whl/cu132.
#
# There is nothing here for an operator to declare. The LTX-2.x agreement forms
# by conduct — "By downloading, using, accessing or distributing any portion or
# element of LTX-2.x, you agree ... to be bound by this Agreement" — so a local
# variable saying YES never formed it. huggingface.co/Lightricks/LTX-2.5 is a
# gated repository, and Lightricks grants access only after a human accepts the
# terms there, which makes a working token evidence of acceptance rather than a
# self-certification. Compliance with the agreement is the operator's own
# responsibility; this script only refuses to fetch what they are not entitled
# to receive.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70
readonly NVIDIA_ACCEPT_ENV=NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS
# Every variable that could pre-answer an entitlement question, in one place so
# the refusal proof cannot drift out of sync with the set it has to scrub.
ENTITLEMENT_ENVS=(
  "$NVIDIA_ACCEPT_ENV"
  HF_TOKEN
)

# Licence facts, kept in sync with npa/src/npa/workbench/ltx2/licensing.py by
# npa/tests/workbench/test_ltx2_licensing.py. They are duplicated here rather
# than imported because the image bakes none of our Python beyond the video
# check — but a wrong URL is a factual error about someone's legal position, so
# the duplication is tested rather than trusted.
readonly LICENSE_NAME="LTX-2.x Community License Agreement"
readonly LICENSE_DATE="2026-08-11"
# Pinned to the fetched ref, not `main`: the licence is versioned by date and
# has been reissued once, so a mutable URL would stop naming the text a given
# run was accepted against.
readonly LICENSE_URL="https://github.com/Lightricks/LTX-2/blob/fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca/LICENSE.md"
readonly ACCEPTABLE_USE_POLICY_URL="https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf"
readonly COMMERCIAL_LICENSE_CONTACT="ltxv-licensing@lightricks.com"

CACHE_ROOT="${NPA_LTX_RUNTIME_CACHE:-/workspace/.cache/npa/ltx2/runtime}"
MODEL_CACHE="${NPA_LTX_MODEL_CACHE:-/workspace/model-cache/ltx-2.5}"
SOURCE_REPO="${NPA_LTX_SOURCE_REPO:-https://github.com/Lightricks/LTX-2.git}"
SOURCE_REF="${NPA_LTX_SOURCE_REF:-fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca}"
WEIGHTS_REPO="${NPA_LTX_WEIGHTS_REPO:-Lightricks/LTX-2.5}"
# The source ref is pinned and verified against the fetched HEAD; the weights had
# neither, so a manifest could name the repository but not the bytes. We cannot
# hardcode a commit here — the repo is gated, so its revisions are only visible
# under the operator's own token — and inventing a plausible-looking sha would be
# worse than none. Instead the mutable ref is resolved to an immutable commit
# BEFORE the download, and that commit is what gets fetched and recorded.
WEIGHTS_REF="${NPA_LTX_WEIGHTS_REF:-main}"
readonly WEIGHTS_REVISION_FILE=.npa_weights_revision
UV_EXTRA="${NPA_LTX_UV_EXTRA:-natten}"

log() { printf 'ltx-runtime: %s\n' "$*" >&2; }
die() { local code="$1"; shift; log "$*"; exit "$code"; }

require_nvidia_acceptance() {
  case "$(printf '%s' "${!NVIDIA_ACCEPT_ENV:-}" | tr '[:lower:]' '[:upper:]')" in
    YES) return 0 ;;
  esac
  cat >&2 <<EOF
ltx-runtime: refusing to download the NVIDIA CUDA PyTorch runtime.

This image contains no CUDA wheels. Installing LTX-2.5's dependencies resolves
torch and torchaudio from NVIDIA/PyTorch's CUDA 13.2 wheel index, as pinned by
upstream's own packages/ltx-core/pyproject.toml. Nebius cannot accept NVIDIA's
terms for the operator, so acceptance is never baked in.

Review the current terms before proceeding:
  https://docs.nvidia.com/cuda/eula/index.html
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/

Set ${NVIDIA_ACCEPT_ENV}=YES to record explicit acceptance.
Any other value refuses with exit ${EX_CONFIG}. Nothing has been downloaded.
EOF
  exit "$EX_CONFIG"
}

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] && return 0
  cat >&2 <<EOF
ltx-runtime: refusing to fetch LTX-2.5 without HF_TOKEN.

https://huggingface.co/${WEIGHTS_REPO} is a GATED repository. Access is granted
to your own Hugging Face account after you accept Lightricks' terms on that page
(a fine-grained token needs the "read gated repos" scope). We never redistribute
LTX-2.5 and never hold an entitlement on your behalf.

The token gates the SOURCE as well as the weights. Section 1.9 of the
${LICENSE_NAME} folds the accompanying source code into
the licensed material, so ${SOURCE_REPO} is not a free
download either — and your entitlement on the gated repository is the only
evidence of acceptance anyone here can check.

Accept the terms with Lightricks, then export HF_TOKEN.
Nothing has been downloaded.
EOF
  exit "$EX_CONFIG"
}

source_tree() { printf '%s/src/%s' "$CACHE_ROOT" "$SOURCE_REF"; }
venv_python() { printf '%s/.venv/bin/python' "$(source_tree)"; }

ready_source() {
  local tree; tree="$(source_tree)"
  [[ -f "$tree/.complete" && -x "$tree/.venv/bin/python" ]]
}

fetch_source() {
  # Both gates first: this function is the only place that reaches the network
  # for software, and it must never do so before the operator is entitled to it.
  # The source is licensed material too — Section 1.9 names the GitHub repo — so
  # it is fetched under the same Hugging Face entitlement as the weights, which
  # is the only acceptance evidence anyone here can check.
  require_hf_token
  require_nvidia_acceptance

  local tree tmp; tree="$(source_tree)"
  ready_source && return 0

  mkdir -p "$CACHE_ROOT/src"
  exec 9>"$CACHE_ROOT/.install.lock"
  flock 9
  ready_source && return 0

  # A killed installer must not leave a half-synced tree that later looks ready.
  find "$CACHE_ROOT/src" -maxdepth 1 -type d -name ".*.tmp.*" -exec rm -rf -- {} +
  tmp="$CACHE_ROOT/src/.${SOURCE_REF}.tmp.$$"
  trap 'rm -rf -- "$tmp"' EXIT

  git init -q "$tmp"
  git -C "$tmp" remote add origin "$SOURCE_REPO"
  git -C "$tmp" fetch -q --depth 1 origin "$SOURCE_REF"
  git -C "$tmp" checkout -q --detach FETCH_HEAD
  # The ref is what the spec and the capability artifact claim ran, so verify
  # what arrived rather than trusting the fetch.
  [[ "$(git -C "$tmp" rev-parse HEAD)" == "$SOURCE_REF" ]] \
    || die "$EX_SOFTWARE" "fetched source ref does not match ${SOURCE_REF}"
  [[ -s "$tmp/LICENSE.md" ]] \
    || die "$EX_SOFTWARE" "upstream LICENSE.md missing; refusing to install"
  rm -rf "$tmp/.git"

  # Upstream's own resolution: uv reads packages/ltx-core/pyproject.toml, which
  # is where the cu132 torch index and the transformers<5.15 bound live. We do
  # not re-pin them here; a second, divergent pin set is how a working upstream
  # install turns into an unsupported one.
  ( cd "$tmp" && uv sync --extra "$UV_EXTRA" )
  "$tmp/.venv/bin/python" -c 'import ltx_pipelines, ltx_core'  # fail before publishing the tree

  # The exact resolved closure is evidence, and it is only knowable after a real
  # resolution — so it is captured here instead of being asserted up front.
  "$tmp/.venv/bin/python" -m uv pip freeze > "$tmp/npa_resolved_inventory.txt" 2>/dev/null \
    || uv pip freeze --python "$tmp/.venv/bin/python" > "$tmp/npa_resolved_inventory.txt"
  : > "$tmp/.complete"
  rm -rf "$tree"
  mv "$tmp" "$tree"
  # Release the install lock explicitly. The fd is not CLOEXEC, so in `exec`
  # mode it would otherwise stay open for the entire generation job and block
  # any sibling that arrived mid-install on a shared cache volume.
  flock -u 9
  exec 9>&-
  trap - EXIT
  log "source ${SOURCE_REF} installed at ${tree}"
}

resolve_weights_revision() {
  # Resolve the ref to the commit it points at right now, under the operator's
  # own token. Downloading a branch name and then recording the branch name
  # would mean the manifest identifies something that changes underneath it.
  python3 -c 'import sys
from huggingface_hub import HfApi
print(HfApi().repo_info(sys.argv[1], revision=sys.argv[2]).sha)' \
    "$WEIGHTS_REPO" "$WEIGHTS_REF"
}

fetch_weights() {
  require_hf_token

  mkdir -p "$MODEL_CACHE"
  local files=(
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    "vae/ltx-2.5-video-vae-bf16.safetensors"
    "vae/ltx-2.5-audio-vae-bf16.safetensors"
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
  )
  if [[ -n "${NPA_LTX_WEIGHT_FILES:-}" ]]; then
    IFS=',' read -r -a files <<< "$NPA_LTX_WEIGHT_FILES"
  fi
  local revision
  revision="$(resolve_weights_revision)"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] \
    || die "$EX_UNAVAILABLE" \
      "could not resolve ${WEIGHTS_REPO}@${WEIGHTS_REF} to a commit"

  log "fetching ${#files[@]} weight files from ${WEIGHTS_REPO}@${revision}"
  hf download "$WEIGHTS_REPO" "${files[@]}" \
    --revision "$revision" --local-dir "$MODEL_CACHE"
  # Record what was actually delivered, next to the bytes, so a later reader
  # can name the weights rather than just the repository they came from.
  printf '%s\n' "$revision" > "$MODEL_CACHE/$WEIGHTS_REVISION_FILE"
  # Written last, so an interrupted pull does not read back as a complete one.
  : > "$MODEL_CACHE/.complete"
}

unentitled() {
  # Run a command with every entitlement variable scrubbed, so a refusal holds
  # regardless of what leaked into the builder or the operator's shell.
  local args=() name
  for name in "${ENTITLEMENT_ENVS[@]}"; do args+=(-u "$name"); done
  env "${args[@]}" "$@"
}

assert_gate() {
  # Assert that a gate refused *and* that it was the gate we meant. Checking only
  # the exit code would make this proof vacuous: both gates refuse with 78, so a
  # token check that had been broken open to accept everything would still
  # "pass" on the strength of NVIDIA's refusal downstream of it. The refusal each
  # gate prints names its own variable, so that is what identifies which one
  # fired.
  local what="$1" expect_env="$2" rc=0 out
  shift 2
  out="$("$@" 2>&1)" || rc=$?
  [[ "$rc" == "$EX_CONFIG" ]] \
    || die "$EX_SOFTWARE" "expected refusal ${EX_CONFIG} from ${what}, got ${rc}"
  [[ "$out" == *"$expect_env"* ]] \
    || die "$EX_SOFTWARE" "${what} refused, but not on ${expect_env}: ${out:0:200}"
}

nvidia_gate_without_acceptance() {
  unset "$NVIDIA_ACCEPT_ENV"
  require_nvidia_acceptance
}

assert_refusal() {
  # Proves, inside the build and again against the pushed image, that the
  # download path refuses without an entitlement — without the build ever running
  # the download path in a way that could succeed. Scrubbing the entitlement
  # variables also keeps the build history free of `ltx-runtime ensure`, which
  # the payload scanner treats as a baked fetch precisely because it normally is
  # one.

  # The wired-up paths, through the real entry point. Both must stop on the
  # Hugging Face entitlement specifically: the source is licensed material too,
  # so it is the gate that must run first on either path.
  assert_gate "'ensure'" HF_TOKEN unentitled "$0" ensure
  assert_gate "'fetch-weights'" HF_TOKEN unentitled "$0" fetch-weights

  # NVIDIA's terms are a different vendor's decision, and reaching that gate
  # through `ensure` would need a token — which this image promises never to
  # contain, even a fake one. assert_gate runs the function inside a command
  # substitution, so the unset is confined to its own subshell.
  assert_gate "the NVIDIA runtime gate" "$NVIDIA_ACCEPT_ENV" \
    nvidia_gate_without_acceptance

  [[ -z "$(find "$CACHE_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]] \
    || die "$EX_SOFTWARE" "refusal wrote to ${CACHE_ROOT}"
  [[ -z "$(find "$MODEL_CACHE" -mindepth 1 -print -quit 2>/dev/null)" ]] \
    || die "$EX_SOFTWARE" "refusal wrote to ${MODEL_CACHE}"
  printf 'NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_ENTITLEMENT_OK\n'
}

terms() {
  # Facts only, and safe to run with nothing set: there is no declaration to
  # make here, and acceptance happens with Lightricks on the gated repository.
  cat <<EOF
npa-ltx2 ships no LTX-2.5 code and no LTX-2.5 weights. Running it asks
Lightricks' own channels to deliver them to you:
  source:  ${SOURCE_REPO} @ ${SOURCE_REF}
  weights: https://huggingface.co/${WEIGHTS_REPO} (gated)

LTX-2.5 is not OSI open source. It is licensed under the
${LICENSE_NAME} (${LICENSE_DATE}):
  ${LICENSE_URL}
  ${ACCEPTABLE_USE_POLICY_URL}

The Agreement binds by use: "By downloading, using, accessing or distributing
any portion or element of LTX-2.x, you agree that you have read and accepted to
be bound by this Agreement." Accept it with Lightricks, on the gated repository
page, with your own Hugging Face account, then export HF_TOKEN. That token is
all this container requires of you, and it gates the source as well as the
weights.

Two obligations are yours alone, and nothing here checks them for you:
  Section 2.1      an Entity whose annual revenue is at or above \$10,000,000,
                   counting all affiliates under common Control, needs a paid
                   Commercial Use Agreement for any use outside the Section 2.2
                   non-commercial carve-out. Contact ${COMMERCIAL_LICENSE_CONTACT}.
  Attachment A(18) for commercial use, the Outputs may not be used to train,
                   improve, or fine-tune any other machine learning model. A
                   robot policy is another machine learning model.
EOF
}

status() {
  local ready="absent" weights="absent"
  ready_source && ready="ready"
  [[ -f "$MODEL_CACHE/.complete" ]] && weights="present"
  [[ "$weights" == "absent" \
     && -n "$(find "$MODEL_CACHE" -name '*.safetensors' -print -quit 2>/dev/null)" ]] \
    && weights="partial"
  local revision="unknown"
  [[ -s "$MODEL_CACHE/$WEIGHTS_REVISION_FILE" ]] \
    && revision="$(<"$MODEL_CACHE/$WEIGHTS_REVISION_FILE")"
  printf '{"source":"%s","source_ref":"%s","weights":"%s","weights_revision":"%s","cache":"%s","model_cache":"%s"}\n' \
    "$ready" "$SOURCE_REF" "$weights" "$revision" "$CACHE_ROOT" "$MODEL_CACHE"
}

ensure_ssh_host_keys() {
  command -v ssh-keygen >/dev/null 2>&1 || return 0
  [[ -s /etc/ssh/ssh_host_ed25519_key ]] && return 0
  sudo -n ssh-keygen -A >/dev/null 2>&1 || true
}

mode="${1:-ensure}"
case "$mode" in
  ensure|warm)
    fetch_source
    ;;
  fetch-weights)
    fetch_weights
    ;;
  assert-refusal)
    assert_refusal
    ;;
  status)
    status
    ;;
  terms)
    terms
    ;;
  health)
    # Must not touch the network, accept anything, or require an entitlement:
    # this runs as the container HEALTHCHECK.
    [[ -r /opt/npa/ltx2/video_check.py && -x /opt/npa/ltx2/validate_video.py ]] \
      || exit "$EX_SOFTWARE"
    [[ -r /usr/share/doc/npa-ltx2/REDISTRIBUTION.md ]] || exit "$EX_SOFTWARE"
    command -v uv >/dev/null 2>&1 || exit "$EX_SOFTWARE"
  # `hf` is how the gated weights arrive; a HEALTHCHECK that ignores it defers
  # the failure to fetch-weights, long after the pod looked healthy.
  command -v hf >/dev/null 2>&1 || exit "$EX_SOFTWARE"
    printf '{"status":"ok","source_ref":"%s","payload":"runtime-fetch"}\n' "$SOURCE_REF"
    ;;
  version)
    printf 'ltx-2.5 source=%s weights=%s provisioning=runtime-fetch\n' \
      "$SOURCE_REF" "$WEIGHTS_REPO"
    ;;
  exec)
    shift
    [[ $# -gt 0 ]] || die "$EX_CONFIG" "exec requires a command"
    fetch_source
    ensure_ssh_host_keys
    export PATH="$(source_tree)/.venv/bin:$PATH"
    cd "$(source_tree)"
    exec "$@"
    ;;
  *)
    die "$EX_CONFIG" \
      "unknown mode '$mode' (use ensure, fetch-weights, assert-refusal, status, terms, health, version, or exec)"
    ;;
esac

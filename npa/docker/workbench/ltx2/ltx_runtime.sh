#!/usr/bin/env bash
# Runtime delivery for LTX-2.5. The image ships none of it.
#
# Two separate vendors have to deliver to the operator directly, so there are two
# independent acceptance gates and neither is ever baked:
#
#   Lightricks  ltx-core / ltx-pipelines source AND the gated LTX-2.5 weights,
#               under the LTX-2.x Community License Agreement.
#   NVIDIA      the CUDA PyTorch stack that upstream's own pins pull from
#               download.pytorch.org/whl/cu132.
#
# The Lightricks gate is enforced by the copied, unit-tested licensing module
# (ltx_gate.py) rather than by shell string comparisons here, so the container
# and the repo cannot disagree about what a valid declaration is.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70
readonly NVIDIA_ACCEPT_ENV=NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS
readonly LTX_ACCEPT_ENV=NPA_LTX_ACCEPT_COMMUNITY_LICENSE
# Every variable that could pre-answer a licensing question, in one place so the
# refusal proof cannot drift out of sync with the set it has to scrub.
DECLARATION_ENVS=(
  "$LTX_ACCEPT_ENV"
  NPA_LTX_ENTITY_CLASS
  NPA_LTX_USE_CLASS
  NPA_LTX_COMMERCIAL_AGREEMENT_REF
  "$NVIDIA_ACCEPT_ENV"
  HF_TOKEN
)

CACHE_ROOT="${NPA_LTX_RUNTIME_CACHE:-/workspace/.cache/npa/ltx2/runtime}"
MODEL_CACHE="${NPA_LTX_MODEL_CACHE:-/workspace/model-cache/ltx-2.5}"
GATE="${NPA_LTX_GATE:-/opt/npa/ltx2/ltx_gate.py}"
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

require_ltx_acceptance() {
  # Delegates to the tested module. It refuses with 78 and prints the terms,
  # the two Lightricks URLs, and what each declaration value means.
  python3 "$GATE" check || exit "$EX_CONFIG"
}

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

source_tree() { printf '%s/src/%s' "$CACHE_ROOT" "$SOURCE_REF"; }
venv_python() { printf '%s/.venv/bin/python' "$(source_tree)"; }

ready_source() {
  local tree; tree="$(source_tree)"
  [[ -f "$tree/.complete" && -x "$tree/.venv/bin/python" ]]
}

fetch_source() {
  # Both gates first: this function is the only place that reaches the network
  # for software, and it must never do so before the operator has declared.
  require_ltx_acceptance
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
  # The ref is the provenance record stamped onto every artifact, so verify it
  # rather than trusting the fetch.
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
  trap - EXIT
  log "source ${SOURCE_REF} installed at ${tree}"
}

require_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] && return 0
  cat >&2 <<EOF
ltx-runtime: refusing to fetch LTX-2.5 weights without HF_TOKEN.

https://huggingface.co/${WEIGHTS_REPO} is a GATED repository. Access is granted
to your own Hugging Face account after you accept Lightricks' terms on that page
(a fine-grained token needs the "read gated repos" scope). We never redistribute
the weights and never hold an entitlement on your behalf.

Accept the terms, then export HF_TOKEN. Nothing has been downloaded.
EOF
  exit "$EX_CONFIG"
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
  require_ltx_acceptance
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
  # Record what was actually delivered, next to the bytes, so the provenance
  # manifest names the weights rather than just the repository.
  printf '%s\n' "$revision" > "$MODEL_CACHE/$WEIGHTS_REVISION_FILE"
}

undeclared() {
  # Run a command with every acceptance variable scrubbed, so a refusal holds
  # regardless of what leaked into the builder or the operator's shell.
  local args=() name
  for name in "${DECLARATION_ENVS[@]}"; do args+=(-u "$name"); done
  env "${args[@]}" "$@"
}

assert_gate() {
  # Assert that a gate refused *and* that it was the gate we meant. Checking only
  # the exit code would make this proof vacuous: three independent gates all
  # refuse with 78, so a licence gate that had been broken open to accept
  # everything would still "pass" on the strength of NVIDIA's or Hugging Face's
  # refusal downstream of it. The refusal each gate prints names its own
  # variable, so that is what identifies which one fired.
  local what="$1" expect_env="$2" rc=0 out
  shift 2
  out="$("$@" 2>&1)" || rc=$?
  [[ "$rc" == "$EX_CONFIG" ]] \
    || die "$EX_SOFTWARE" "expected refusal ${EX_CONFIG} from ${what}, got ${rc}"
  [[ "$out" == *"$expect_env"* ]] \
    || die "$EX_SOFTWARE" "${what} refused, but not on ${expect_env}: ${out:0:200}"
}

nvidia_gate_undeclared() { unset "$NVIDIA_ACCEPT_ENV"; require_nvidia_acceptance; }
hf_gate_undeclared() { unset HF_TOKEN; require_hf_token; }

assert_refusal() {
  # Proves, inside the build and again against the pushed image, that the
  # download path refuses without a declaration — without the build ever running
  # the download path in a way that could succeed. Scrubbing the acceptance
  # variables also keeps the build history free of `ltx-runtime ensure`, which
  # the payload scanner treats as a baked fetch precisely because it normally is
  # one.

  # The wired-up paths, through the real entry point. Both must stop on the
  # Lightricks gate specifically, because it is the one that must run first.
  assert_gate "'ensure'" "$LTX_ACCEPT_ENV" undeclared "$0" ensure
  assert_gate "'fetch-weights'" "$LTX_ACCEPT_ENV" undeclared "$0" fetch-weights

  # The two downstream gates, exercised directly. Reaching them through `ensure`
  # would require a valid declaration, and writing an acceptance value into this
  # script — even only to test with — is exactly what the image promises never to
  # contain. assert_gate runs them inside a command substitution, so each unset
  # below is confined to its own subshell.
  assert_gate "the NVIDIA runtime gate" "$NVIDIA_ACCEPT_ENV" nvidia_gate_undeclared
  assert_gate "the gated-weights token check" HF_TOKEN hf_gate_undeclared

  [[ -z "$(find "$CACHE_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]] \
    || die "$EX_SOFTWARE" "refusal wrote to ${CACHE_ROOT}"
  [[ -z "$(find "$MODEL_CACHE" -mindepth 1 -print -quit 2>/dev/null)" ]] \
    || die "$EX_SOFTWARE" "refusal wrote to ${MODEL_CACHE}"
  printf 'NPA_LTX_BOOTSTRAP_REFUSES_WITHOUT_DECLARATION_OK\n'
}

status() {
  local ready="absent" weights="absent"
  ready_source && ready="ready"
  [[ -n "$(find "$MODEL_CACHE" -name '*.safetensors' -print -quit 2>/dev/null)" ]] \
    && weights="present"
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
    exec python3 "$GATE" terms
    ;;
  provenance)
    exec python3 "$GATE" provenance
    ;;
  health)
    # Must not touch the network, accept anything, or require a declaration:
    # this runs as the container HEALTHCHECK.
    [[ -r "$GATE" && -r /opt/npa/ltx2/licensing.py ]] || exit "$EX_SOFTWARE"
    [[ -r /usr/share/doc/npa-ltx2/REDISTRIBUTION.md ]] || exit "$EX_SOFTWARE"
    command -v uv >/dev/null 2>&1 || exit "$EX_SOFTWARE"
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
      "unknown mode '$mode' (use ensure, fetch-weights, assert-refusal, status, terms, provenance, health, version, or exec)"
    ;;
esac

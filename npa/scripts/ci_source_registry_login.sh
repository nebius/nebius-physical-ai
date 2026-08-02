#!/usr/bin/env bash
# Log `crane` into the Nebius source registry for a CI job, from whichever credential the
# environment provides, and refuse to continue on one that is provably expired.
#
# Two workflows need this (publish-public-images and public-mirror-health) and getting it
# wrong is how the mirror broke twice, so it lives here rather than inline in each.
#
# CREDENTIALS, preferred in the order tried. Nebius CR authenticates `docker login` as user
# "iam" with either, and they differ only in how long they survive -- which is the whole game
# for a workflow dispatched by hand months apart:
#
#   NEBIUS_SA_CREDENTIALS_JSON  an authorized-key credentials JSON for a service account with
#                               viewer on the registry. A fresh access token is minted here,
#                               in the job, so nothing expires between dispatches.
#   NEBIUS_CR_TOKEN             a pre-issued token. Must be a long-lived static key
#                               (`nebius iam static-key issue --service=CONTAINER_REGISTRY`,
#                               6 months by default), NOT `nebius iam get-access-token`
#                               output: an access token lives 12 hours, so a stored one is
#                               dead by the next dispatch. That was the first failure.
#
# Other environment:
#   REQUIRE_CREDENTIAL  "true" (default) to fail when neither secret is set; "false" to
#                       report `available=false` and exit 0, so a rehearsal without secrets
#                       still runs and says what it could not check.
#   PYTHON              interpreter with `npa` importable (default `python`; locally use
#                       npa/.venv/bin/python).
#   GITHUB_OUTPUT       when set, `available=true|false` is appended for later `if:` guards.
#   RUNNER_TEMP         scratch directory (default: a fresh mktemp -d).
#
# The token is never echoed: it is written to a 0600 file, piped from there, and shredded on
# exit including on failure.
set -euo pipefail
umask 077

PYTHON="${PYTHON:-python}"
REQUIRE_CREDENTIAL="${REQUIRE_CREDENTIAL:-true}"
scratch="${RUNNER_TEMP:-$(mktemp -d)}"
token_file="${scratch}/npa-source-registry-token"
sa_file="${scratch}/nebius-sa-credentials.json"
trap 'rm -f "${token_file}" "${sa_file}"' EXIT

# Both hosts: the primary registry and its mirror. Only the primary is read by the current
# plan, but a --source-registry override pointing at the mirror must not need a second login.
REGISTRY_HOSTS="${REGISTRY_HOSTS:-cr.eu-north1.nebius.cloud cr.us-central1.nebius.cloud}"

emit_available() {
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "available=$1" >> "${GITHUB_OUTPUT}"
  return 0
}

if [ -n "${NEBIUS_SA_CREDENTIALS_JSON:-}" ]; then
  echo "Minting a fresh access token from NEBIUS_SA_CREDENTIALS_JSON."
  if ! command -v nebius >/dev/null 2>&1; then
    curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
    export PATH="${HOME}/.nebius/bin:${PATH}"
  fi
  printf '%s' "${NEBIUS_SA_CREDENTIALS_JSON}" > "${sa_file}"
  nebius profile create \
    --endpoint api.nebius.cloud \
    --service-account-file "${sa_file}" \
    --profile npa-ci-source-registry
  nebius --profile npa-ci-source-registry iam get-access-token > "${token_file}"
elif [ -n "${NEBIUS_CR_TOKEN:-}" ]; then
  printf '%s' "${NEBIUS_CR_TOKEN}" > "${token_file}"
elif [ "${REQUIRE_CREDENTIAL}" != "true" ]; then
  echo "::notice::Neither NEBIUS_SA_CREDENTIALS_JSON nor NEBIUS_CR_TOKEN is set, so the source registry cannot be checked."
  emit_available false
  exit 0
else
  echo "A source registry credential (NEBIUS_SA_CREDENTIALS_JSON or NEBIUS_CR_TOKEN) is required" >&2
  exit 1
fi

# Offline verdict before any registry round trip. An expired access token is the failure this
# has actually had, and it is readable from the token's own `exp`, so name it in a second
# rather than as a wall of indistinguishable UNAUTHORIZED lines two minutes later.
"${PYTHON}" -m npa.deploy.publish_public --describe-credential < "${token_file}"

# `crane auth login` writes a config file and exits 0 for ANY string without contacting the
# registry, so this proves nothing on its own -- the caller must still read a real manifest.
for host in ${REGISTRY_HOSTS}; do
  crane auth login "${host}" -u iam --password-stdin < "${token_file}"
done
emit_available true

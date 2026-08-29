#!/usr/bin/env bash
#
# Prepare / verify access to gated HF and NGC artifacts the workbench needs.
#
# Given a Hugging Face token and an NVIDIA NGC API key, this reports whether the
# token already has access to each gated model the workbench uses and prints the
# exact "Agree and access repository" URL for anything still gated.
#
# NOTE: Hugging Face gated licenses must be accepted interactively on each model
# page — there is no API to accept them on your behalf — so this script automates
# the check + guidance and (optionally) persists your keys; it does not click the
# license button for you.
#
# Despite its legacy filename, this script never accepts terms. Usage:
#   scripts/accept-model-access.sh                       # use ~/.npa/credentials.yaml
#   HF_TOKEN=hf_xxx NGC_API_KEY=nvapi-xxx scripts/accept-model-access.sh
#   scripts/accept-model-access.sh --capability groot,cosmos
#
# Any extra arguments are forwarded to `npa workbench health access --prepare`.
set -euo pipefail

NPA_BIN="${NPA_BIN:-npa}"

# Credentials remain in supported environment/config APIs and never enter argv.
exec "${NPA_BIN}" workbench health access --prepare "$@"

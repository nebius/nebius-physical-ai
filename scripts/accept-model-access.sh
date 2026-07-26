#!/usr/bin/env bash
#
# Accept / verify access to every gated model the workbench needs.
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
# Usage:
#   scripts/accept-model-access.sh                       # use ~/.npa/credentials.yaml
#   HF_TOKEN=hf_xxx NGC_API_KEY=nvapi-xxx scripts/accept-model-access.sh
#   scripts/accept-model-access.sh --hf-token hf_xxx --ngc-key nvapi-xxx --set-credentials
#   scripts/accept-model-access.sh --capability groot,cosmos
#
# Any extra arguments are forwarded to `npa workbench health access`
# (e.g. --json, --offline, --warn-only, --capability, --set-credentials).
set -euo pipefail

NPA_BIN="${NPA_BIN:-npa}"

args=()
# Promote HF_TOKEN / NGC_API_KEY from the environment to explicit flags so the
# same invocation works with or without ~/.npa/credentials.yaml.
if [ -n "${HF_TOKEN:-}" ]; then
  args+=(--hf-token "${HF_TOKEN}")
fi
if [ -n "${NGC_API_KEY:-}" ]; then
  args+=(--ngc-key "${NGC_API_KEY}")
fi

# Expand args safely even when empty under `set -u` (bash 3.2 on stock macOS
# errors on a bare "${args[@]}" when the array is empty).
exec "${NPA_BIN}" workbench health access ${args[@]+"${args[@]}"} "$@"

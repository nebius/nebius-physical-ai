#!/usr/bin/env bash
# Run a genuine source-staging capability, not an import or metadata placeholder.
# The candidate image remains private and unqualified until the six-stage native
# workflow completes; this command is only its raw-input component check.
set -euo pipefail

: "${NPA_SMOKE_OUTPUT_DIR:?NPA_SMOKE_OUTPUT_DIR is required}"
exec lingbot-va-runtime exec python -m npa.solutions.lingbot_va stage-raw \
  --output-uri "file://${NPA_SMOKE_OUTPUT_DIR}/lingbot-va-raw"

#!/usr/bin/env bash
# Static bootstrap gate only. It does not fetch Tencent material or infer.
set -euo pipefail

hy-world-runtime health
hy-world-runtime bootstrap-integrity

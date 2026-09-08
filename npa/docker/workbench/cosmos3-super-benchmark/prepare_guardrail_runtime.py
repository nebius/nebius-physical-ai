"""Fetch separately gated guardrail assets only for an enabled guardrail runtime."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Apply the selected guardrail policy before any guardrail asset download.

    Args:
        None.

    Returns:
        Zero after skipping disabled guardrails or completing their materializer.

    Raises:
        ValueError: The guardrail selection is neither on nor off.
        subprocess.CalledProcessError: The enabled guardrail materializer fails.
    """
    mode = os.environ.get("NPA_COSMOS3_SERVE_GUARDRAILS", "off")
    if mode == "off":
        print("[npa-cosmos3] guardrails disabled; no guardrail assets requested")
        return 0
    if mode != "on":
        raise ValueError("NPA_COSMOS3_SERVE_GUARDRAILS must be on or off")
    subprocess.run(
        [sys.executable, "/opt/npa-cosmos3-serving/prepare_guardrail_payload.py"],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

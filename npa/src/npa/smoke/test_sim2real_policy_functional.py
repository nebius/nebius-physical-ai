"""Sim2Real policy-image functional golden eval (variant + envgen + Genesis CUDA)."""

from __future__ import annotations

import os
import sys

from npa.smoke import test_sim2real_envgen_functional as envgen_smoke


def main() -> int:
    variant = os.environ.get("NPA_SIM2REAL_POLICY_VARIANT", "reference")
    print(f"policy_variant={variant}")
    if variant not in {"reference", "explore"}:
        print(f"[FAIL] unexpected NPA_SIM2REAL_POLICY_VARIANT={variant!r}")
        return 1
    return envgen_smoke.main()


if __name__ == "__main__":
    sys.exit(main())
